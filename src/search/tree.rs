//! The search tree: an arena of nodes with PUCT selection, value backup, and proofs.
//!
//! Nodes live in one `Vec`; a node's children are a contiguous run of indices allocated
//! when it is expanded, so selection scans a slice and never chases pointers. Values are
//! stored from the side to move *at that node*: a parent reads a child's value negated,
//! and backup negates at every ply. The draw probability is carried alongside (it is the
//! same from both sides), so a node's win/draw/loss can be reported, not only its value.
//!
//! Proofs are exact game-theoretic results (mate, stalemate, repetition, tablebase) that
//! propagate toward the root: a node with a lost child is won, a node whose children are
//! all proven takes the best of them. A proven node keeps its exact value whatever is
//! backed up through it, and carries the distance in plies so the shortest mate is
//! preferred and `score mate N` can be reported.

use crate::search::time::RootStats;
use shakmaty::Move;

use crate::encoding::LegalActions;

/// Index of the root in every tree.
pub const ROOT: u32 = 0;

/// Buffers for [`Tree::allocate`], reused across calls.
#[derive(Debug, Default, Clone)]
pub struct AllocScratch {
    /// Allotment per child, in child order, after `allocate`.
    pub shares: Vec<u32>,
    terms: Vec<Term>,
}

/// One child's PUCT terms with the parent's visit count held fixed.
#[derive(Debug, Clone, Copy)]
struct Term {
    q: f32,
    /// `cpuct · prior · √N(parent)`.
    u: f32,
    /// Visits including virtual ones, before this allocation.
    n: u64,
}

impl Term {
    /// Score after `extra` more visits.
    #[inline]
    fn score(&self, extra: u32) -> f32 {
        self.q + self.u / (1 + self.n + u64::from(extra)) as f32
    }

    /// How many further visits beyond `have` still score at least `t`, capped at `cap`
    /// (unbounded when `t` is at or below the value term, since the score never drops
    /// below `q`).
    fn count_at_least(&self, t: f32, have: u32, cap: u32) -> u32 {
        if t <= self.q {
            return cap.saturating_add(1);
        }
        // q + u/(1+n+j) >= t  ⇔  j <= u/(t−q) − 1 − n
        let last = self.u / (t - self.q) - 1.0 - self.n as f32;
        if last < have as f32 {
            return 0;
        }
        let mut k = ((last - have as f32).min(cap as f32) as u32).saturating_add(1);
        // Agree with the score function at the boundary despite rounding.
        while k > 0 && self.score(have + k - 1) < t {
            k -= 1;
        }
        while k <= cap && self.score(have + k) >= t {
            k += 1;
        }
        k.min(cap)
    }
}

/// An exact result from the side to move at the node that holds it. Distances are plies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Proof {
    Win(u16),
    Draw,
    Loss(u16),
}

impl Proof {
    /// `+1`, `0`, `-1`.
    #[inline]
    pub fn value(self) -> f32 {
        match self {
            Proof::Win(_) => 1.0,
            Proof::Draw => 0.0,
            Proof::Loss(_) => -1.0,
        }
    }

    /// Draw probability: 1 for a draw, 0 otherwise.
    #[inline]
    pub fn draw(self) -> f32 {
        match self {
            Proof::Draw => 1.0,
            Proof::Win(_) | Proof::Loss(_) => 0.0,
        }
    }

    /// The same result seen from the parent, one ply earlier.
    #[inline]
    pub fn from_parent(self) -> Proof {
        match self {
            Proof::Win(n) => Proof::Loss(n + 1),
            Proof::Draw => Proof::Draw,
            Proof::Loss(n) => Proof::Win(n + 1),
        }
    }

    /// Whether this result is preferable to `other` for the side that holds it: any win
    /// beats a draw beats any loss; shorter wins and longer losses are better.
    pub fn better_than(self, other: Proof) -> bool {
        match (self, other) {
            (Proof::Win(a), Proof::Win(b)) => a < b,
            (Proof::Win(_), _) => true,
            (Proof::Draw, Proof::Loss(_)) => true,
            (Proof::Draw, _) => false,
            (Proof::Loss(a), Proof::Loss(b)) => a > b,
            (Proof::Loss(_), _) => false,
        }
    }

    /// `score mate N` in moves from the holder's point of view, negative when being mated.
    pub fn mate_in_moves(self) -> Option<i32> {
        match self {
            Proof::Win(plies) => Some((i32::from(plies) + 1) / 2),
            Proof::Draw => None,
            Proof::Loss(plies) => Some(-(i32::from(plies) / 2)),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// The move that led here; `None` at the root.
    pub mv: Option<Move>,
    pub prior: f32,
    /// Completed visits. `u64`: a `u32` wraps after 4.29e9 visits, which a long `go
    /// infinite` on a GPU reaches at the root, and a wrapped count would upset every mean
    /// and PUCT term above the node. Widening costs 8 bytes per node (56 to 64 with the
    /// padding) against saturating a `u32`, which would have to freeze the node's sums and
    /// stop the search at the root; the memory is the smaller price.
    pub visits: u64,
    /// Leaves below this node that are waiting for evaluation; they count as visits during
    /// selection so one batch spreads over different leaves. Bounded by the leaves in
    /// flight, so `u32` is plenty.
    pub virtual_visits: u32,
    /// Sum of backed-up values from this node's side to move. `f64`: an `f32` sum stops
    /// moving once it passes ~2^24, which a GPU reaches at the root in minutes.
    pub value_sum: f64,
    /// Sum of backed-up draw probabilities; the same from either side.
    pub draw_sum: f64,
    /// This node is itself in the batch being evaluated.
    pub pending: bool,
    /// Sum of the priors of children that have been visited, for first-play urgency.
    explored: f32,
    /// `(first child index, count)` once expanded.
    children: Option<(u32, u32)>,
    pub proof: Option<Proof>,
}

impl Node {
    fn new(mv: Option<Move>, prior: f32) -> Self {
        Self {
            mv,
            prior,
            visits: 0,
            virtual_visits: 0,
            value_sum: 0.0,
            draw_sum: 0.0,
            pending: false,
            explored: 0.0,
            children: None,
            proof: None,
        }
    }

    /// Prior mass of the children that have been visited (for first-play urgency).
    pub fn explored_mass(&self) -> f32 {
        self.explored
    }

    /// Mean backed-up value from this node's side to move; 0 before any visit.
    #[inline]
    pub fn mean(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            (self.value_sum / self.visits as f64) as f32
        }
    }

    /// Mean backed-up draw probability; 0 before any visit.
    #[inline]
    pub fn draw_mean(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            (self.draw_sum / self.visits as f64) as f32
        }
    }

    pub fn is_expanded(&self) -> bool {
        self.children.is_some()
    }
}

#[derive(Debug, Clone)]
pub struct Tree {
    nodes: Vec<Node>,
}

impl Default for Tree {
    fn default() -> Self {
        Self::new()
    }
}

impl Tree {
    pub fn new() -> Self {
        Self {
            nodes: vec![Node::new(None, 1.0)],
        }
    }

    pub fn with_capacity(nodes: usize) -> Self {
        let mut all = Vec::with_capacity(nodes.max(1));
        all.push(Node::new(None, 1.0));
        Self { nodes: all }
    }

    /// Nodes in the arena; never zero, the root is always there.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn node(&self, index: u32) -> &Node {
        &self.nodes[index as usize]
    }

    #[inline]
    pub fn node_mut(&mut self, index: u32) -> &mut Node {
        &mut self.nodes[index as usize]
    }

    /// Indices of a node's children (empty when unexpanded or childless).
    pub fn children(&self, index: u32) -> std::ops::Range<u32> {
        match self.nodes[index as usize].children {
            Some((first, count)) => first..first + count,
            None => 0..0,
        }
    }

    fn child_slice(&self, index: u32) -> &[Node] {
        let range = self.children(index);
        &self.nodes[range.start as usize..range.end as usize]
    }

    /// Give `index` one child per legal move, with `priors` in the same order.
    pub fn expand(&mut self, index: u32, legal: &LegalActions, priors: &[f32]) {
        debug_assert_eq!(legal.len(), priors.len());
        debug_assert!(!self.nodes[index as usize].is_expanded());
        let first = self.nodes.len() as u32;
        self.nodes.extend(
            legal
                .moves
                .iter()
                .zip(priors)
                .map(|(&(mv, _), &prior)| Node::new(Some(mv), prior.max(0.0))),
        );
        self.nodes[index as usize].children = Some((first, legal.len() as u32));
    }

    /// Split `budget` visits among the children of `parent` the way sequential PUCT
    /// selection would, with the parent's visit count held fixed: the leading child takes
    /// visits until the runner-up would overtake it, then the runner-up leads, and so on.
    /// `scratch.shares` receives one allotment per child, in child order.
    ///
    /// The score is `q + cpuct · prior · √N(parent) / (1 + N(child))`, visits including
    /// virtual ones and the allotment so far. An unvisited child is assumed slightly worse
    /// than its parent's current estimate, the more so the more prior mass is already
    /// explored (Lc0's first-play urgency). Ties go to the earlier child.
    ///
    /// Each child's scores form a decreasing sequence in its visit count, so the greedy
    /// allocation is the `budget` largest values across all sequences. A few greedy rounds
    /// settle the usual case where one child dominates; otherwise the cutoff score is
    /// found by bisection and the ties at the cutoff are resolved greedily.
    pub fn allocate(
        &self,
        parent: u32,
        budget: u32,
        cpuct: f32,
        fpu_reduction: f32,
        scratch: &mut AllocScratch,
    ) {
        const GREEDY_ROUNDS: u32 = 6;
        const BISECTIONS: u32 = 30;

        let node = &self.nodes[parent as usize];
        let children = self.child_slice(parent);
        debug_assert!(!children.is_empty(), "allocate on a childless node");
        let scale = cpuct * ((node.visits + u64::from(node.virtual_visits)).max(1) as f32).sqrt();
        let unvisited_q = node.mean() - fpu_reduction * node.explored.sqrt();

        let AllocScratch { shares, terms } = scratch;
        shares.clear();
        shares.resize(children.len(), 0);
        terms.clear();
        terms.extend(children.iter().map(|child| Term {
            q: match child.proof {
                Some(proof) => -proof.value(),
                None if child.visits > 0 => -child.mean(),
                None => unvisited_q,
            },
            u: scale * child.prior,
            n: child.visits + u64::from(child.virtual_visits),
        }));

        let mut remaining = budget;
        let mut rounds = 0;
        while remaining > 0 {
            // ---- greedy round: the leader takes visits until the runner-up would lead
            let mut best = 0usize;
            let mut best_score = f32::NEG_INFINITY;
            let mut second = f32::NEG_INFINITY;
            for (offset, term) in terms.iter().enumerate() {
                let score = term.score(shares[offset]);
                if score > best_score {
                    second = best_score;
                    best_score = score;
                    best = offset;
                } else if score > second {
                    second = score;
                }
            }
            let leader = &terms[best];
            let take = if leader.q >= second {
                remaining // its value alone beats every rival's score: visits never flip it
            } else {
                let room = (leader.u / (second - leader.q)
                    - 1.0
                    - (leader.n + u64::from(shares[best])) as f32)
                    .max(0.0);
                (room.min(remaining as f32 - 1.0) as u32 + 1).min(remaining)
            };
            shares[best] += take;
            remaining -= take;
            rounds += 1;
            // Stepping costs one scan per visit; bisection costs ~BISECTIONS scans
            // whatever the budget, so it only pays when many visits are still to place.
            if remaining == 0 || rounds < GREEDY_ROUNDS || remaining < BISECTIONS * 2 {
                continue;
            }

            // ---- many near-ties: bisect the cutoff score instead of stepping
            // `count(t)` = values still unallocated with score >= t; decreasing in t.
            let count = |t: f32| -> u32 {
                let mut total = 0u32;
                for (offset, term) in terms.iter().enumerate() {
                    total = total.saturating_add(term.count_at_least(t, shares[offset], remaining));
                    if total > remaining {
                        break;
                    }
                }
                total
            };
            // At `lo` every term gets at least `remaining` visits; at `hi` none gets any.
            let mut lo = terms.iter().map(|t| t.q).fold(f32::INFINITY, f32::min) - 1.0;
            let mut hi = best_score + 1.0;
            for _ in 0..BISECTIONS {
                let mid = 0.5 * (lo + hi);
                if count(mid) >= remaining {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            // Everything scoring at least `hi` is among the top `remaining` values.
            for (offset, term) in terms.iter().enumerate() {
                let k = term.count_at_least(hi, shares[offset], remaining);
                shares[offset] += k;
                remaining -= k;
            }
            // The rest sit in the tie band between `lo` and `hi`: a handful, resolved greedily.
        }
    }

    /// Add `count` virtual visits to every node on `path`.
    pub fn reserve_n(&mut self, path: &[u32], count: u32) {
        for &index in path {
            self.nodes[index as usize].virtual_visits += count;
        }
    }

    /// Release `count` virtual visits from every node on `path`.
    pub fn unreserve_n(&mut self, path: &[u32], count: u32) {
        for &index in path {
            let node = &mut self.nodes[index as usize];
            debug_assert!(node.virtual_visits >= count, "virtual visit underflow");
            node.virtual_visits -= count;
        }
    }

    /// Add a visit along `path` with `leaf_value` from the leaf's side to move, negating
    /// each ply, and `leaf_draw`, the leaf's draw probability. A proven node records its
    /// exact result instead.
    pub fn backup(&mut self, path: &[u32], leaf_value: f32, leaf_draw: f32) {
        self.backup_n(path, leaf_value, leaf_draw, 1);
    }

    /// [`Self::backup`] applied `count` times at once (a proven or terminal leaf that
    /// sequential search would have visited `count` times).
    pub fn backup_n(&mut self, path: &[u32], leaf_value: f32, leaf_draw: f32, count: u32) {
        let mut value = leaf_value;
        let weight = count as f32;
        for (depth, &index) in path.iter().enumerate().rev() {
            let node = &mut self.nodes[index as usize];
            if node.visits == 0 && depth > 0 {
                let prior = node.prior;
                self.nodes[path[depth - 1] as usize].explored += prior;
            }
            let node = &mut self.nodes[index as usize];
            node.visits += u64::from(count);
            let (v, d) = match node.proof {
                Some(proof) => (proof.value(), proof.draw()),
                None => (value, leaf_draw),
            };
            node.value_sum += f64::from(weight * v);
            node.draw_sum += f64::from(weight * d);
            value = -value;
        }
    }

    /// Mark the leaf of `path` as proven and propagate toward the root: a node with a
    /// lost child is won (by the shortest route); a node whose children are all proven
    /// takes the best of them; otherwise propagation stops. Each node is reachable by one
    /// path, so path-dependent results (repetition) are sound to prove on it.
    pub fn solve(&mut self, path: &[u32], proof: Proof) {
        let (&leaf, ancestors) = path.split_last().expect("a path has a leaf");
        self.set_proof(leaf, proof);
        for &index in ancestors.iter().rev() {
            if self.nodes[index as usize].proof.is_some() {
                break;
            }
            let children = self.child_slice(index);
            if children.is_empty() {
                break;
            }
            let mut best: Option<Proof> = None;
            let mut all_proven = true;
            for child in children {
                match child.proof {
                    Some(child_proof) => {
                        let seen = child_proof.from_parent();
                        if best.is_none_or(|current| seen.better_than(current)) {
                            best = Some(seen);
                        }
                    }
                    None => all_proven = false,
                }
            }
            let proven = match best {
                Some(win @ Proof::Win(_)) => win,
                Some(other) if all_proven => other,
                _ => break,
            };
            self.set_proof(index, proven);
        }
    }

    fn set_proof(&mut self, index: u32, proof: Proof) {
        let node = &mut self.nodes[index as usize];
        node.proof = Some(proof);
        node.value_sum = f64::from(proof.value()) * node.visits as f64;
        node.draw_sum = f64::from(proof.draw()) * node.visits as f64;
    }

    /// True when every root move but one is a proven loss: there is nothing left to decide.
    pub fn forced_by_proof(&self) -> bool {
        let children = self.child_slice(ROOT);
        if children.len() < 2 {
            return false;
        }
        let unrefuted = children
            .iter()
            .filter(|child| !matches!(child.proof, Some(Proof::Win(_))))
            .count();
        unrefuted == 1
    }

    /// The root move to play, or `None` when the root has no children: the first of
    /// [`Self::ranked_root_children`].
    pub fn best_child(&self) -> Option<u32> {
        self.children(ROOT)
            .min_by(|&a, &b| self.root_order(self.node(a), self.node(b)))
    }

    /// The root's children from the move to play downward, the order `MultiPV` reports
    /// them in. Proven wins come first, the shortest first; then the unrefuted moves, the
    /// most visited first and the highest prior among equals; proven losses last. Moves
    /// tied on all of that keep their move-list order.
    pub fn ranked_root_children(&self) -> Vec<u32> {
        let mut ranked: Vec<u32> = self.children(ROOT).collect();
        ranked.sort_by(|&a, &b| self.root_order(self.node(a), self.node(b)));
        ranked
    }

    /// Ordering of two root children for [`Self::ranked_root_children`].
    fn root_order(&self, a: &Node, b: &Node) -> std::cmp::Ordering {
        // 0: a proven win for us (a loss for the child's mover); 1: undecided; 2: a proven
        // loss.
        fn class(node: &Node) -> u8 {
            match node.proof {
                Some(Proof::Loss(_)) => 0,
                Some(Proof::Win(_)) => 2,
                _ => 1,
            }
        }
        class(a)
            .cmp(&class(b))
            .then_with(|| match (a.proof, b.proof) {
                (Some(Proof::Loss(x)), Some(Proof::Loss(y))) => x.cmp(&y),
                _ => std::cmp::Ordering::Equal,
            })
            .then_with(|| b.visits.cmp(&a.visits))
            .then_with(|| b.prior.total_cmp(&a.prior))
    }

    /// The child of `index` reached by `mv`.
    pub fn child_by_move(&self, index: u32, mv: Move) -> Option<u32> {
        let range = self.children(index);
        self.child_slice(index)
            .iter()
            .position(|child| child.mv == Some(mv))
            .map(|offset| range.start + offset as u32)
    }

    /// The tree below the node `moves` lead to from the root, as a tree of its own, or
    /// `None` when that node was never expanded (nothing to reuse). Visits, values, priors
    /// and proofs are kept; the discarded part of the tree is what the moves left behind.
    /// Only for a finished search: nothing may be pending.
    pub fn reroot(&self, moves: &[Move]) -> Option<Tree> {
        let mut index = ROOT;
        for &mv in moves {
            index = self.child_by_move(index, mv)?;
        }
        if !self.nodes[index as usize].is_expanded() {
            return None;
        }
        let mut nodes: Vec<Node> = Vec::with_capacity(self.nodes.len() / 2);
        let mut root = self.nodes[index as usize].clone();
        root.mv = None;
        root.children = None;
        nodes.push(root);
        // (old index, new index) of nodes whose children still have to be copied.
        let mut queue: Vec<(u32, u32)> = vec![(index, ROOT)];
        while let Some((old, new)) = queue.pop() {
            let range = self.children(old);
            if range.is_empty() {
                continue;
            }
            let first = nodes.len() as u32;
            for (offset, child) in self.child_slice(old).iter().enumerate() {
                debug_assert!(!child.pending && child.virtual_visits == 0);
                let mut copy = child.clone();
                copy.children = None;
                nodes.push(copy);
                if child.is_expanded() {
                    queue.push((range.start + offset as u32, first + offset as u32));
                }
            }
            nodes[new as usize].children = Some((first, range.len() as u32));
        }
        Some(Tree { nodes })
    }

    /// What the time manager reads at a batch boundary: the two largest root visit counts
    /// and whether the most-visited move is also the best-valued one (exact results
    /// counting as their value). Ties on value go to the more-visited move.
    pub fn root_stats(&self) -> RootStats {
        let children = self.child_slice(ROOT);
        let mut best = (0u64, f32::NEG_INFINITY, 0usize);
        let mut second_visits = 0u64;
        let mut best_q = (f32::NEG_INFINITY, 0u64, 0usize);
        for (offset, child) in children.iter().enumerate() {
            let q = match child.proof {
                Some(proof) => -proof.value(),
                None if child.visits > 0 => -child.mean(),
                None => continue,
            };
            if child.visits > best.0 || (child.visits == best.0 && q > best.1) {
                second_visits = best.0;
                best = (child.visits, q, offset);
            } else if child.visits > second_visits {
                second_visits = child.visits;
            }
            if q > best_q.0 || (q == best_q.0 && child.visits > best_q.1) {
                best_q = (q, child.visits, offset);
            }
        }
        RootStats {
            best_visits: best.0,
            second_visits,
            decided: best.0 > 0 && best.2 == best_q.2,
        }
    }

    /// Principal variation: the chosen root move, then the most visited child at each ply.
    pub fn pv(&self) -> Vec<Move> {
        self.best_child()
            .map_or_else(Vec::new, |best| self.pv_from(best))
    }

    /// The line starting with the root child `index`: its move, then the most visited
    /// child at each ply.
    pub fn pv_from(&self, mut index: u32) -> Vec<Move> {
        let mut line = Vec::new();
        loop {
            line.push(self.nodes[index as usize].mv.expect("children carry moves"));
            let range = self.children(index);
            let next = self
                .child_slice(index)
                .iter()
                .enumerate()
                .filter(|(_, child)| child.visits > 0)
                .max_by(|(_, a), (_, b)| a.visits.cmp(&b.visits).then(a.prior.total_cmp(&b.prior)))
                .map(|(offset, _)| range.start + offset as u32);
            match next {
                Some(child) => index = child,
                None => return line,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Game;
    use shakmaty::CastlingMode;

    fn game(fen: Option<&str>) -> Game {
        Game::from_uci(fen, &[], CastlingMode::Standard).unwrap()
    }

    fn uniform(legal: &LegalActions) -> Vec<f32> {
        vec![1.0 / legal.len() as f32; legal.len()]
    }

    #[test]
    fn proof_ordering_and_distances() {
        assert!(Proof::Win(3).better_than(Proof::Win(5)));
        assert!(Proof::Win(9).better_than(Proof::Draw));
        assert!(Proof::Draw.better_than(Proof::Loss(1)));
        assert!(Proof::Loss(4).better_than(Proof::Loss(2)));
        assert!(!Proof::Loss(4).better_than(Proof::Draw));
        assert_eq!(Proof::Loss(0).from_parent(), Proof::Win(1));
        assert_eq!(Proof::Win(1).from_parent(), Proof::Loss(2));
        assert_eq!(Proof::Win(1).mate_in_moves(), Some(1));
        assert_eq!(Proof::Win(3).mate_in_moves(), Some(2));
        assert_eq!(Proof::Loss(2).mate_in_moves(), Some(-1));
        assert_eq!(Proof::Draw.mate_in_moves(), None);
    }

    /// Child chosen by `allocate` with a budget of one: sequential PUCT selection.
    fn select(tree: &Tree, parent: u32) -> u32 {
        tree.children(parent).start
            + allocate(tree, parent, 1)
                .iter()
                .position(|&a| a == 1)
                .unwrap() as u32
    }

    fn allocate(tree: &Tree, parent: u32, budget: u32) -> Vec<u32> {
        let mut scratch = AllocScratch::default();
        tree.allocate(parent, budget, 1.5, 0.25, &mut scratch);
        scratch.shares
    }

    #[test]
    fn expand_allocate_backup_signs() {
        let root = game(None);
        let legal = LegalActions::of(&root);
        let mut tree = Tree::new();
        tree.expand(ROOT, &legal, &uniform(&legal));
        assert_eq!(tree.children(ROOT).len(), 20);
        assert_eq!(tree.len(), 21);

        // Fresh root, equal priors: the first child leads and a single visit flips the
        // choice, so a budget spreads one visit per child.
        let first = select(&tree, ROOT);
        assert_eq!(first, 1);
        assert_eq!(allocate(&tree, ROOT, 20), vec![1; 20]);
        let out = allocate(&tree, ROOT, 45);
        assert_eq!(out.iter().sum::<u32>(), 45);
        assert!(out.iter().all(|&a| a == 2 || a == 3), "{out:?}");

        // A leaf value of +1 for the child (its side wins) is -1 for the root.
        tree.backup(&[ROOT, first], 1.0, 0.0);
        assert_eq!(tree.node(first).visits, 1);
        assert_eq!(tree.node(first).mean(), 1.0);
        assert_eq!(tree.node(ROOT).visits, 1);
        assert_eq!(tree.node(ROOT).mean(), -1.0);
        // Parent-relative FPU: unvisited children sit just below the parent's mean (now -1),
        // so the visited child is still preferred while its visit count is tiny...
        assert_eq!(select(&tree, ROOT), first);
        // ...and loses out once its exploration term has shrunk.
        tree.backup_n(&[ROOT, first], 1.0, 0.0, 2);
        assert_eq!(tree.node(first).visits, 3);
        assert_eq!(tree.node(ROOT).visits, 3);
        assert_eq!(tree.node(ROOT).mean(), -1.0);
        assert_ne!(select(&tree, ROOT), first);

        // Virtual visits count like visits: reserving a child steers selection away.
        let next = select(&tree, ROOT);
        tree.reserve_n(&[ROOT, next], 5);
        assert_eq!(tree.node(next).virtual_visits, 5);
        assert_ne!(select(&tree, ROOT), next);
        tree.unreserve_n(&[ROOT, next], 5);
        assert_eq!(select(&tree, ROOT), next);
        assert_eq!(tree.node(ROOT).virtual_visits, 0);
    }

    #[test]
    fn allocate_matches_sequential_selection() {
        // Sharp policy: the leader absorbs many visits before the runner-up overtakes.
        let root = game(None);
        let legal = LegalActions::of(&root);
        let mut tree = Tree::new();
        let mut priors = vec![0.01; legal.len()];
        priors[3] = 1.0 - 0.01 * (legal.len() - 1) as f32;
        tree.expand(ROOT, &legal, &priors);
        let leader = 1 + 3;

        // Simulate sequential selection with the parent's N held fixed, the way
        // `allocate` models it: reserve on the child only, one visit at a time.
        let mut sequential = vec![0u32; legal.len()];
        for _ in 0..200 {
            let pick = select(&tree, ROOT);
            sequential[(pick - 1) as usize] += 1;
            tree.reserve_n(&[pick], 1);
        }
        for (offset, &count) in sequential.iter().enumerate() {
            tree.unreserve_n(&[1 + offset as u32], count);
        }
        let out = allocate(&tree, ROOT, 200);
        assert_eq!(out, sequential);
        assert!(out[3] > 20, "{out:?}");
        assert_eq!(select(&tree, ROOT), leader);
    }

    #[test]
    fn allocate_matches_sequential_selection_with_mixed_values() {
        // Visited children with different values, some virtual visits, a proven child:
        // the bisection path must agree with stepping one visit at a time.
        let root = game(None);
        let legal = LegalActions::of(&root);
        let mut tree = Tree::new();
        let priors: Vec<f32> = (0..legal.len()).map(|i| 0.02 + 0.006 * i as f32).collect();
        tree.expand(ROOT, &legal, &priors);
        let children: Vec<u32> = tree.children(ROOT).collect();
        for (i, &child) in children.iter().enumerate() {
            let value = ((i * 7) % 11) as f32 / 10.0 - 0.5;
            tree.backup_n(&[ROOT, child], value, 0.0, 1 + (i as u32 % 4));
        }
        tree.reserve_n(&[children[2]], 3);
        tree.node_mut(children[9]).proof = Some(Proof::Draw);

        for &budget in &[1u32, 7, 64, 500, 2000] {
            let mut sequential = vec![0u32; legal.len()];
            for _ in 0..budget {
                let pick = select(&tree, ROOT);
                sequential[(pick - 1) as usize] += 1;
                tree.reserve_n(&[pick], 1);
            }
            for (offset, &count) in sequential.iter().enumerate() {
                tree.unreserve_n(&[1 + offset as u32], count);
            }
            let out = allocate(&tree, ROOT, budget);
            assert_eq!(out.iter().sum::<u32>(), budget);
            assert_eq!(out, sequential, "budget {budget}");
        }
    }

    #[test]
    fn solve_propagates_mate_with_distance() {
        // White to move, Qh7# is one of several legal moves.
        let root = game(Some("6k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1"));
        let legal = LegalActions::of(&root);
        let mut tree = Tree::new();
        tree.expand(ROOT, &legal, &uniform(&legal));
        let mate_index = tree
            .children(ROOT)
            .find(|&index| root.uci(tree.node(index).mv.unwrap()) == "d1d8")
            .unwrap();
        // The mated leaf is a loss for its side to move at distance 0.
        tree.backup(&[ROOT, mate_index], -1.0, 0.0);
        tree.solve(&[ROOT, mate_index], Proof::Loss(0));
        assert_eq!(tree.node(mate_index).proof, Some(Proof::Loss(0)));
        assert_eq!(tree.node(ROOT).proof, Some(Proof::Win(1)));
        assert_eq!(tree.node(ROOT).mean(), 1.0);
        assert_eq!(tree.best_child(), Some(mate_index));
        assert_eq!(root.uci(tree.pv()[0]), "d1d8");
        // Backing anything else up through a proven node keeps its exact value.
        tree.backup(&[ROOT, mate_index], 0.3, 0.0);
        assert_eq!(tree.node(ROOT).mean(), 1.0);
    }

    #[test]
    fn never_plays_a_proven_loss_and_all_proven_takes_the_best() {
        let root = game(None);
        let legal = LegalActions::of(&root);
        let mut tree = Tree::new();
        tree.expand(ROOT, &legal, &uniform(&legal));
        let children: Vec<u32> = tree.children(ROOT).collect();
        // Most visited move is a proven loss for us (a win for the child's mover).
        tree.backup(&[ROOT, children[0]], 0.0, 0.0);
        tree.backup(&[ROOT, children[0]], 0.0, 0.0);
        tree.node_mut(children[0]).proof = Some(Proof::Win(3));
        tree.backup(&[ROOT, children[1]], 0.0, 0.0);
        assert_eq!(tree.best_child(), Some(children[1]));
        assert!(!tree.forced_by_proof());
        // The ranking agrees: the undecided move first, the lost one last of all.
        let ranked = tree.ranked_root_children();
        assert_eq!(ranked.len(), children.len());
        assert_eq!(ranked[0], children[1]);
        assert_eq!(*ranked.last().unwrap(), children[0]);
        // Unvisited moves keep their move order behind the visited one.
        assert_eq!(ranked[1], children[2]);

        // Prove every child: root becomes the best flipped result (a draw here).
        for &child in &children {
            tree.node_mut(child).proof = Some(Proof::Win(3));
        }
        tree.node_mut(children[5]).proof = Some(Proof::Draw);
        assert!(tree.forced_by_proof());
        tree.solve(&[ROOT, children[5]], Proof::Draw);
        assert_eq!(tree.node(ROOT).proof, Some(Proof::Draw));
        assert_eq!(tree.best_child(), Some(children[5]));
        assert_eq!(tree.node(ROOT).draw_mean(), 1.0);
    }

    #[test]
    fn draws_accumulate_and_proofs_fix_them() {
        let root = game(None);
        let legal = LegalActions::of(&root);
        let mut tree = Tree::new();
        tree.expand(ROOT, &legal, &uniform(&legal));
        let children: Vec<u32> = tree.children(ROOT).collect();
        // Draw probabilities are not negated on the way up; values are.
        tree.backup(&[ROOT, children[0]], 0.2, 0.5);
        tree.backup(&[ROOT, children[1]], -0.4, 0.1);
        assert!((tree.node(children[0]).draw_mean() - 0.5).abs() < 1e-6);
        assert!((tree.node(ROOT).draw_mean() - 0.3).abs() < 1e-6);
        assert!((tree.node(ROOT).mean() - 0.1).abs() < 1e-6);
        // A proven draw counts as a certain draw however it is reached, and rewrites what
        // was accumulated before the proof.
        tree.backup(&[ROOT, children[2]], 0.9, 0.0);
        tree.set_proof(children[2], Proof::Draw);
        assert_eq!(tree.node(children[2]).draw_mean(), 1.0);
        assert_eq!(tree.node(children[2]).mean(), 0.0);
        tree.backup(&[ROOT, children[2]], 0.9, 0.0);
        assert_eq!(tree.node(children[2]).draw_mean(), 1.0);
        // A win is never a draw.
        tree.backup(&[ROOT, children[3]], -1.0, 0.0);
        tree.solve(&[ROOT, children[3]], Proof::Loss(0));
        assert_eq!(tree.node(ROOT).proof, Some(Proof::Win(1)));
        assert_eq!(tree.node(ROOT).draw_mean(), 0.0);
        assert_eq!(tree.ranked_root_children()[0], children[3]);
    }
}
