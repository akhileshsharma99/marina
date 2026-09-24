//! Batched PUCT search. Runs on its own thread; reports through [`SearchEvent`]s.
//!
//! A batch of up to `Batch` leaves is gathered in a few walks of the tree: at each node the
//! remaining budget is split among the children the way sequential PUCT selection would
//! split it ([`Tree::allocate`]), and each child is entered once with its share, so the
//! top of the tree is scored once per walk rather than once per leaf. Leaves waiting for
//! the network hold virtual visits so later walks and batches go elsewhere. Leaves that
//! are terminal (mate, stalemate, insufficient material, a repeated position, the
//! fifty-move rule) or already proven are backed up immediately without using a network
//! slot, and their exact results propagate as proofs.
//!
//! The network runs on its own thread with up to [`IN_FLIGHT`] batches queued, so the
//! next gather overlaps the current evaluation. The search stops on the node or time
//! limit, on `stop`, or when the root is proven or has one unrefuted move left. A
//! pondering (`go ponder`) or infinite search never stops on its own: with nothing left
//! to search it holds until `stop`, or until `ponderhit` turns the pondering search into
//! a normal one with the clock running from that moment.

pub mod stop;
pub mod time;
pub mod tree;

use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use shakmaty::Move;

use crate::encoding::{EncodedBoard, LegalActions, encode_board};
use crate::engine::SearchEvent;
use crate::net::{Evaluation, Network, NetworkError};
use crate::options::{Options, ScoreType};
use crate::position::{DrawRules, Game, GameEnd};
use crate::search::time::{Budget, Clock, Pacer, TimeState};
use crate::tablebase::{TB_PLIES, Tablebase};
use crate::uci::command::GoLimits;
use crate::uci::output::{Info, Score};
pub use stop::StopSignal;
pub use tree::{AllocScratch, Proof, ROOT, Tree};

/// UCI null move, sent when there is nothing to play.
pub const NULL_MOVE: &str = "0000";

/// How often to print an `info` line during a long search (the first comes with the first
/// batch).
const INFO_INTERVAL: Duration = Duration::from_millis(250);
/// How often a held search (pondering or infinite, with nothing left to search) looks for
/// `stop` or `ponderhit`.
const HOLD_POLL: Duration = Duration::from_millis(1);
/// How often a search waiting on the network looks for `stop` and the hard deadline, so
/// neither waits for a slow batch to come back.
const WAIT_POLL: Duration = Duration::from_millis(2);

/// What a running search tells its caller.
pub enum Report<'a> {
    /// A rendered `info` line.
    Info(String),
    /// The search is over and its `bestmove` can go out now, before the batches still at
    /// the network are collected and the tree is stored, which may take a network call.
    /// GUIs expect `bestmove` within tens of milliseconds of `stop`.
    Done(&'a Summary),
}
/// Nodes reserved for a fresh tree; a game move at a fast clock fits without growing.
const INITIAL_TREE_CAPACITY: usize = 4096;

/// Why the search ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    NodeLimit,
    /// The mean simulation depth reached `go depth`.
    DepthLimit,
    TimeLimit,
    /// The GUI sent `stop`.
    Stopped,
    ProvenWin,
    ProvenDraw,
    ProvenLoss,
    /// Every root move but one is a proven loss.
    ForcedByProof,
    /// No leaf could be gathered (the tree below the root is fully proven or terminal).
    Exhausted,
    OnlyMove,
    GameOver,
    /// The root is in the tablebases: the move came from DTZ, no search was run.
    Tablebase,
    /// The network stopped answering (`Summary::failure` says why); the best move so far.
    NetworkFailed,
    /// The runner-up could not catch the leader in the time left (`search::time`).
    SmartPruning,
}

/// What a finished search knows about itself, for `info`, tests and the bench.
#[derive(Debug, Clone, PartialEq)]
pub struct Summary {
    pub best: Option<Move>,
    pub ponder: Option<Move>,
    /// Completed simulations (a leaf backed up, from the network or an exact result).
    pub nodes: u64,
    pub network_evals: u64,
    pub batches: u64,
    /// Tree walks made to fill the batches (one walk places many leaves; several may be
    /// needed when shares are lost to collisions).
    pub walks: u64,
    /// Visits the batches could not place: shares beyond the first on a leaf, and shares
    /// that reached a leaf already in flight. Never re-descended; a count of lost work.
    pub collisions: u64,
    /// Leaves (or the root) settled by the tablebases.
    pub tbhits: u64,
    /// Root visits the search started with (tree reuse); `nodes` counts only new ones.
    pub reused: u64,
    pub elapsed: Duration,
    pub max_depth: u32,
    pub avg_depth: f32,
    /// Root value from the engine's side, in `[-1, 1]`.
    pub value: f32,
    /// Root draw probability; with `value` this is the root's win/draw/loss.
    pub draw: f32,
    pub proof: Option<Proof>,
    pub stop_reason: StopReason,
    /// Set with [`StopReason::NetworkFailed`].
    pub failure: Option<NetworkError>,
    /// What a move played on the clock leaves for the game's time bookkeeping.
    pub settlement: Option<Settlement>,
}

/// Inputs to [`TimeState::settle`] from one move: the budget it was played under, the
/// time it used on the clock (from `go`, or from `ponderhit` when it pondered first) and
/// the simulations made in that time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Settlement {
    pub budget: Budget,
    pub used: Duration,
    pub new: u64,
}

impl Summary {
    pub fn nps(&self) -> f64 {
        let secs = self.elapsed.as_secs_f64();
        if secs > 0.0 {
            self.nodes as f64 / secs
        } else {
            0.0
        }
    }
}

/// Search settings, taken from the options at `go`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Params {
    pub cpuct: f32,
    pub fpu_reduction: f32,
    pub batch: u32,
    /// Batches evaluated concurrently with tree work (`InFlight`): 1 is sequential, 2
    /// hides the network's latency behind the next gather at the price of a second batch
    /// chosen without feedback from the first.
    pub in_flight: u32,
    /// Batch ramp divisor (`BatchRamp`): a batch asks for at most `tree size / ramp`
    /// leaves, at least [`RAMP_MIN`], capped at `batch`; 0 turns the ramp off.
    pub batch_ramp: u32,
    /// Collision budget (`CollisionBudget`) as a percentage of the batch: stop filling once
    /// the visits that could not be placed exceed it; 0 means no limit.
    pub collision_budget_percent: u32,
    pub move_overhead: Duration,
    /// Root moves to report a line for (`MultiPV`); the search itself is unchanged.
    pub multipv: u32,
    /// Report `wdl` on every `info` line (`UCI_ShowWDL`).
    pub show_wdl: bool,
    /// How `score cp` is derived from the value (`ScoreType`).
    pub score_type: ScoreType,
}

/// Default number of batches in flight.
pub const IN_FLIGHT: u32 = 2;

/// Smallest batch the ramp will ask for.
pub const RAMP_MIN: u64 = 8;

impl From<&Options> for Params {
    fn from(options: &Options) -> Self {
        Self {
            cpuct: options.cpuct(),
            fpu_reduction: options.fpu_reduction(),
            batch: options.batch().max(1),
            in_flight: options.in_flight.max(1),
            batch_ramp: options.batch_ramp,
            collision_budget_percent: options.collision_budget_percent,
            move_overhead: Duration::from_millis(u64::from(options.move_overhead_ms)),
            multipv: options.multipv,
            show_wdl: options.show_wdl,
            score_type: options.score_type,
        }
    }
}

/// What ends this search besides `stop`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Limits {
    pub nodes: Option<u64>,
    /// Stop once the mean simulation depth (the `info depth` figure) reaches this.
    pub depth: Option<u32>,
    /// The clock, when the search is time-managed.
    pub time: Option<TimeLimit>,
    /// Never finish on our own (`go infinite`, or `go` with no limit at all).
    pub until_stopped: bool,
    /// `go ponder`: none of the above applies and the search never finishes on its own
    /// until `ponderhit`, from when they all do with the clock running from that moment;
    /// `stop` ends it with a `bestmove` for the pondered position.
    pub ponder: bool,
}

/// A time-managed search's clock. The budget is drawn from it when the clock is armed,
/// at `go` or at `ponderhit`, so it accounts for the tree at that moment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeLimit {
    /// `go movetime`: exactly this long, no early stop.
    Movetime(Duration),
    /// A share of the clock, with the game's bookkeeping (bank, nps, reuse) that sizes it.
    Clock { clock: Clock, state: TimeState },
}

impl Limits {
    /// Resolve `go` arguments against the clock of the side to move; `state` is the
    /// game's time bookkeeping so far.
    pub fn from_go(limits: &GoLimits, game: &Game, state: TimeState) -> Self {
        let until_stopped = limits.infinite || limits.is_unbounded();
        let time = if until_stopped {
            None
        } else if let Some(movetime) = limits.movetime {
            Some(TimeLimit::Movetime(Duration::from_millis(movetime)))
        } else {
            let (remaining, increment) = match game.turn() {
                shakmaty::Color::White => (limits.wtime, limits.winc),
                shakmaty::Color::Black => (limits.btime, limits.binc),
            };
            remaining.map(|remaining| TimeLimit::Clock {
                clock: Clock {
                    remaining: Duration::from_millis(remaining),
                    increment: Duration::from_millis(increment.unwrap_or(0)),
                    movestogo: limits.movestogo,
                },
                state,
            })
        };
        Self {
            nodes: limits.nodes,
            depth: limits.depth,
            time,
            until_stopped,
            ponder: limits.ponder,
        }
    }
}

/// State the engine shares with every search: what it searches with, and what one search
/// leaves for the next.
#[derive(Clone)]
pub struct Shared {
    pub network: Arc<dyn Network>,
    /// Which loaded network `network` is: the engine bumps it on every load. A saved tree
    /// carries the generation that built it and is only continued by a search of the same
    /// one, so a tree left by a search that was still running on the old network when the
    /// GUI switched is never re-rooted under the new one.
    pub generation: u64,
    pub tablebase: Option<Arc<Tablebase>>,
    pub time_state: Arc<Mutex<TimeState>>,
    /// The last search's tree, for the next `go` to continue from.
    pub tree: Arc<Mutex<Option<SavedTree>>>,
}

/// What one search leaves for the next: its tree, the game it was rooted at, and the
/// network generation whose priors and values it holds.
pub struct SavedTree {
    pub generation: u64,
    pub game: Game,
    pub tree: Tree,
}

/// Entry point for the search thread: run, sending `info` lines and exactly one
/// `bestmove` as [`SearchEvent`]s, then leave the tree for the next move to continue from.
pub fn run(
    game: Game,
    limits: GoLimits,
    options: Options,
    shared: Shared,
    stop: Arc<StopSignal>,
    events: Sender<SearchEvent>,
) {
    // A search that panics (a bug, in a build that unwinds) must still answer, or the UCI
    // loop waits for a `bestmove` forever and every later `go` queues behind it.
    struct Answer<'a> {
        events: &'a Sender<SearchEvent>,
        sent: bool,
    }
    impl Drop for Answer<'_> {
        fn drop(&mut self) {
            if !self.sent && std::thread::panicking() {
                let _ = self.events.send(SearchEvent::Failed(
                    "the search thread panicked; see stderr".to_string(),
                ));
                let _ = self.events.send(SearchEvent::BestMove {
                    best: NULL_MOVE.to_string(),
                    ponder: None,
                });
            }
        }
    }
    let mut answer = Answer {
        events: &events,
        sent: false,
    };
    // The clock runs from here: re-rooting a large tree is part of this move's time.
    let started = Instant::now();
    let params = Params::from(&options);
    let previous = shared
        .tree
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .take();
    let tree = match previous {
        Some(saved) if saved.generation != shared.generation => {
            // Left by a search of another network (one that was still running when the
            // GUI switched, so it stored its tree after the engine forgot the old one).
            // Its priors and values are not this network's.
            tracing::debug!(
                tree = saved.generation,
                network = shared.generation,
                "tree discarded: it came from another network"
            );
            None
        }
        Some(saved) if limits.searchmoves.is_none() => game
            .continuation_from(&saved.game)
            .and_then(|moves| saved.tree.reroot(moves)),
        _ => None,
    };
    match &tree {
        Some(tree) => tracing::debug!(visits = tree.node(ROOT).visits, "tree reused"),
        None => tracing::debug!("tree fresh"),
    }
    let state = *shared
        .time_state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let resolved = Limits::from_go(&limits, &game, state);
    let (summary, tree) = search_from(
        started,
        &game,
        &resolved,
        limits.searchmoves.as_deref(),
        &params,
        shared.network.as_ref(),
        shared.tablebase.as_deref(),
        tree,
        &stop,
        &mut |report| match report {
            Report::Info(line) => {
                let _ = events.send(SearchEvent::Info(line));
            }
            Report::Done(summary) => {
                let best = summary
                    .best
                    .map_or_else(|| NULL_MOVE.to_string(), |mv| game.uci(mv));
                let ponder = summary.ponder.map(|mv| game.uci(mv));
                if let Some(error) = &summary.failure {
                    let _ = events.send(SearchEvent::Failed(error.to_string()));
                }
                let _ = events.send(SearchEvent::BestMove { best, ponder });
            }
        },
    );
    answer.sent = true;
    if let Some(settlement) = summary.settlement {
        shared
            .time_state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .settle(
                settlement.budget,
                settlement.used,
                settlement.new,
                summary.reused,
                summary.reused + summary.nodes,
            );
    }
    // A tree whose root was restricted (`searchmoves`, a single legal move) is not a
    // search of the position and must not seed the next one.
    let reusable = limits.searchmoves.is_none()
        && summary.stop_reason != StopReason::OnlyMove
        && summary.failure.is_none();
    *shared
        .tree
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = reusable.then(|| SavedTree {
        generation: shared.generation,
        game: game.clone(),
        tree,
    });
}

/// [`search_with_tree`] from an empty tree, discarding the tree: the single-shot form the
/// tests use.
#[cfg(test)]
#[allow(clippy::too_many_arguments)] // the search's inputs; bundling them would only rename this
pub fn search(
    game: &Game,
    limits: &Limits,
    searchmoves: Option<&[String]>,
    params: &Params,
    network: &dyn Network,
    tablebase: Option<&Tablebase>,
    stop: &StopSignal,
    report: &mut dyn FnMut(Report<'_>),
) -> Summary {
    search_with_tree(
        game,
        limits,
        searchmoves,
        params,
        network,
        tablebase,
        None,
        stop,
        report,
    )
    .0
}

/// Search `game` until a limit, a proof, or `stop`, reporting through `report`; continues
/// from `tree` (a re-rooted earlier tree) when given, and returns the tree for the next
/// move to continue from.
#[allow(clippy::too_many_arguments)] // the search's inputs; bundling them would only rename this
pub fn search_with_tree(
    game: &Game,
    limits: &Limits,
    searchmoves: Option<&[String]>,
    params: &Params,
    network: &dyn Network,
    tablebase: Option<&Tablebase>,
    tree: Option<Tree>,
    stop: &StopSignal,
    report: &mut dyn FnMut(Report<'_>),
) -> (Summary, Tree) {
    search_from(
        Instant::now(),
        game,
        limits,
        searchmoves,
        params,
        network,
        tablebase,
        tree,
        stop,
        report,
    )
}

/// [`search_with_tree`] with the clock already running since `started`.
#[allow(clippy::too_many_arguments)] // the search's inputs; bundling them would only rename this
fn search_from(
    started: Instant,
    game: &Game,
    limits: &Limits,
    searchmoves: Option<&[String]>,
    params: &Params,
    network: &dyn Network,
    tablebase: Option<&Tablebase>,
    tree: Option<Tree>,
    stop: &StopSignal,
    report: &mut dyn FnMut(Report<'_>),
) -> (Summary, Tree) {
    let mut searcher = Searcher::new(game, params, network, tablebase, tree);
    let summary = searcher.run(limits, searchmoves, stop, started, report);
    (summary, searcher.tree)
}

struct Searcher<'a> {
    root: &'a Game,
    params: &'a Params,
    network: &'a dyn Network,
    tablebase: Option<&'a Tablebase>,
    tree: Tree,
    /// Clock plan of a time-managed search; `None` while pondering.
    pacer: Option<Pacer>,
    /// `go ponder` and no `ponderhit` yet: no limit applies and nothing finishes the
    /// search but `stop`.
    pondering: bool,
    /// `nodes` when the clock was armed; the settlement counts from there.
    nodes_at_clock: u64,
    nodes: u64,
    tbhits: u64,
    reused: u64,
    network_evals: u64,
    batches: u64,
    walks: u64,
    collisions: u64,
    max_depth: u32,
    depth_sum: u64,
    depth_count: u64,
    /// The root's own evaluation, reported until its children have been visited.
    root_value: f32,
    root_draw: f32,
    /// Why the network stopped answering, when it did.
    failure: Option<NetworkError>,
    last_info: Instant,
    /// `info depth` so far: the running maximum of the rounded mean depth.
    reported_depth: u32,
    /// Per-depth allocation buffers for `visit`, recycled.
    scratch: Vec<AllocScratch>,
}

/// A leaf waiting for the network. Its path lives in the batch's flat path buffer.
struct Pending {
    board: EncodedBoard,
    legal: LegalActions,
    path_start: usize,
    path_len: usize,
}

/// One batch on its way to the network and back. Buffers are recycled between batches.
#[derive(Default)]
struct Batch {
    leaves: Vec<Pending>,
    /// Paths of `leaves`, back to back.
    paths: Vec<u32>,
    /// `(node, virtual visits)` to release once the batch has been applied.
    reserved: Vec<(u32, u32)>,
    evaluations: Vec<Evaluation>,
    /// The network could not evaluate this batch.
    failed: Option<NetworkError>,
    /// Handed back unevaluated: `stop` came first.
    abandoned: bool,
}

impl Batch {
    fn clear(&mut self) {
        self.leaves.clear();
        self.paths.clear();
        self.reserved.clear();
        self.evaluations.clear();
        self.failed = None;
        self.abandoned = false;
    }
}

/// What one gather produced.
struct Gathered {
    /// Leaves handed to the network.
    leaves: usize,
    /// Simulations completed on the spot (proven or terminal leaves).
    immediate: u64,
}

impl<'a> Searcher<'a> {
    fn new(
        root: &'a Game,
        params: &'a Params,
        network: &'a dyn Network,
        tablebase: Option<&'a Tablebase>,
        tree: Option<Tree>,
    ) -> Self {
        Self {
            root,
            params,
            network,
            tablebase,
            tree: tree.unwrap_or_else(|| Tree::with_capacity(INITIAL_TREE_CAPACITY)),
            pacer: None,
            pondering: false,
            nodes_at_clock: 0,
            nodes: 0,
            tbhits: 0,
            reused: 0,
            network_evals: 0,
            batches: 0,
            walks: 0,
            collisions: 0,
            max_depth: 0,
            depth_sum: 0,
            depth_count: 0,
            root_value: 0.0,
            root_draw: 0.0,
            failure: None,
            last_info: Instant::now(),
            reported_depth: 0,
            scratch: Vec::new(),
        }
    }

    fn run(
        &mut self,
        limits: &Limits,
        searchmoves: Option<&[String]>,
        stop: &StopSignal,
        started: Instant,
        report: &mut dyn FnMut(Report<'_>),
    ) -> Summary {
        self.pondering = limits.ponder;
        // The first report comes with the first batch, then one every interval.
        self.last_info = started.checked_sub(INFO_INTERVAL).unwrap_or(started);
        self.reused = self.tree.node(ROOT).visits;
        if !self.pondering {
            self.arm_clock(limits, started);
        }
        let mut legal = LegalActions::of(self.root);
        if let Some(only) = searchmoves {
            let restricted: Vec<(Move, usize)> = legal
                .moves
                .iter()
                .copied()
                .filter(|&(mv, _)| only.iter().any(|uci| *uci == self.root.uci(mv)))
                .collect();
            if !restricted.is_empty() {
                legal.moves = restricted;
            }
        }
        // Only "no legal moves" ends the game here: a threefold repetition or fifty moves
        // are draws the GUI adjudicates (or not); a move is owed either way.
        if legal.is_empty() {
            let summary = self.finish(started, StopReason::GameOver, limits, stop);
            return self.conclude(summary, report);
        }
        if legal.len() == 1 && !limits.until_stopped {
            if !self.tree.node(ROOT).is_expanded() {
                self.tree.expand(ROOT, &legal, &[1.0]);
            }
            let summary = self.finish(started, StopReason::OnlyMove, limits, stop);
            return self.conclude(summary, report);
        }
        // In the tablebases the root move comes from DTZ, not from search: the value head
        // cannot tell a won ending from a drawn one, and WDL alone would shuffle.
        if let Some((mv, proof)) = self
            .tablebase
            .and_then(|tb| tb.root(self.root.position()))
            .filter(|(mv, _)| searchmoves.is_none() || legal.moves.iter().any(|&(m, _)| m == *mv))
        {
            self.tbhits += 1;
            let mut summary = self.finish(started, StopReason::Tablebase, limits, stop);
            summary.best = Some(mv);
            summary.ponder = None;
            summary.proof = Some(proof);
            summary.value = proof.value();
            summary.draw = proof.draw();
            return self.conclude(summary, report);
        }

        // The root is evaluated on its own so the first batch has children to spread over;
        // a reused tree brings its root already expanded and visited.
        if self.tree.node(ROOT).is_expanded() {
            self.root_value = self.tree.node(ROOT).mean();
            self.root_draw = self.tree.node(ROOT).draw_mean();
        } else {
            let board = encode_board(self.root);
            let evaluation = match self.network.evaluate(&[(&board, &legal)]) {
                Ok(mut evaluations) => evaluations.remove(0),
                Err(error) => {
                    self.failure = Some(error);
                    let summary = self.summary(started, StopReason::NetworkFailed);
                    return self.conclude(summary, report);
                }
            };
            self.network_evals += 1;
            self.root_value = evaluation.value();
            self.root_draw = evaluation.wdl[1];
            self.tree.expand(ROOT, &legal, &evaluation.priors);
        }

        // The network runs on its own thread; up to `in_flight` batches wait there while
        // this thread gathers the next. Batches come back in order and are applied in order.
        let in_flight = self.params.in_flight.max(1) as usize;
        let network = self.network;
        let (to_backend, backend_rx) = mpsc::sync_channel::<Batch>(in_flight);
        let (to_search, results_rx) = mpsc::channel::<Batch>();
        std::thread::scope(|scope| {
            scope.spawn(move || {
                for mut batch in backend_rx {
                    // After `stop` nothing queued is worth a network call: hand it back
                    // unevaluated so the search can release it and answer.
                    if stop.is_set() {
                        batch.abandoned = true;
                    } else {
                        let leaves: Vec<(&EncodedBoard, &LegalActions)> = batch
                            .leaves
                            .iter()
                            .map(|leaf| (&leaf.board, &leaf.legal))
                            .collect();
                        match network.evaluate(&leaves) {
                            Ok(evaluations) => batch.evaluations = evaluations,
                            Err(error) => batch.failed = Some(error),
                        }
                    }
                    if to_search.send(batch).is_err() {
                        break;
                    }
                }
            });

            let mut line = self.root.clone();
            let mut path: Vec<u32> = Vec::with_capacity(64);
            let mut pool: Vec<Batch> = (0..=in_flight).map(|_| Batch::default()).collect();
            let mut queued = 0usize; // leaves in flight
            let mut in_flight_count = 0usize;

            let reason = loop {
                if self.pondering {
                    if stop.is_set() {
                        break StopReason::Stopped;
                    }
                    if stop.is_ponderhit() {
                        self.ponderhit(limits);
                    }
                }
                if !self.pondering
                    && let Some(reason) = self.reached(limits, stop)
                {
                    break reason;
                }
                // No node limit while pondering: it applies from `ponderhit`.
                let node_limit = if self.pondering { None } else { limits.nodes };

                // ---- fill the pipeline
                let mut immediate = 0u64;
                while in_flight_count < in_flight {
                    let mut want = self.batch_size();
                    if let Some(limit) = node_limit {
                        want = want.min(limit.saturating_sub(self.nodes + queued as u64));
                    }
                    if want == 0 || self.proof_stop().is_some() {
                        break;
                    }
                    let mut batch = pool.pop().unwrap_or_default();
                    batch.clear();
                    let gathered =
                        self.gather(&mut batch, want, node_limit, queued, &mut line, &mut path);
                    immediate += gathered.immediate;
                    if gathered.leaves == 0 {
                        // Nothing for the network, but the walk may have reserved paths to
                        // leaves already in flight; those reservations end here.
                        self.release(&batch);
                        pool.push(batch);
                        break;
                    }
                    queued += gathered.leaves;
                    in_flight_count += 1;
                    to_backend
                        .send(batch)
                        .expect("the backend thread outlives the search loop");
                }

                if in_flight_count == 0 {
                    if immediate == 0 {
                        // Nothing left to search: the root is proven or the tree below it
                        // is exhausted.
                        if self.holds(limits) {
                            match self.hold(limits, stop) {
                                Some(reason) => break reason,
                                None => continue, // ponderhit: the limits now decide
                            }
                        }
                        break self.proof_stop().unwrap_or(StopReason::Exhausted);
                    }
                    continue;
                }
                // ---- apply the oldest batch, unless `stop` or the hard deadline come first
                let batch = loop {
                    match results_rx.recv_timeout(WAIT_POLL) {
                        Ok(batch) => break Some(batch),
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            if stop.is_set() {
                                break None;
                            }
                            if !self.pondering
                                && self
                                    .pacer
                                    .as_ref()
                                    .is_some_and(|pacer| Instant::now() >= pacer.hard_deadline())
                            {
                                break None;
                            }
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => {
                            unreachable!("the backend thread outlives the search loop")
                        }
                    }
                };
                let Some(batch) = batch else {
                    break if stop.is_set() {
                        StopReason::Stopped
                    } else {
                        StopReason::TimeLimit
                    };
                };
                in_flight_count -= 1;
                queued -= batch.leaves.len();
                if let Some(error) = batch.failed.clone() {
                    self.discard(&batch);
                    self.failure = Some(error);
                    break StopReason::NetworkFailed;
                }
                if batch.abandoned {
                    // Only after `stop`, which the top of the loop is about to see.
                    self.discard(&batch);
                    pool.push(batch);
                    continue;
                }
                self.apply(&batch);
                pool.push(batch);
                if let Some(pacer) = self.pacer.as_mut() {
                    let stats = self.tree.root_stats();
                    if let Some(reason) = pacer.after_batch(stats, self.nodes, Instant::now()) {
                        break reason;
                    }
                }

                if self.last_info.elapsed() >= INFO_INTERVAL {
                    // Progress report; the stop reason is not part of an `info` line.
                    let summary = self.summary(started, StopReason::Stopped);
                    self.emit_info(&summary, report);
                }
            };

            // The move is decided now; the batches still at the network are collected
            // afterwards so the tree stays consistent for the next move (and what has
            // been evaluated is not thrown away), without keeping the `bestmove` waiting.
            let summary = self.summary(started, reason);
            let summary = self.conclude(summary, report);
            drop(to_backend);
            for batch in results_rx.iter().take(in_flight_count) {
                if batch.abandoned || batch.failed.is_some() || self.failure.is_some() {
                    self.discard(&batch);
                } else {
                    self.apply(&batch);
                }
            }
            summary
        })
    }

    /// Deliver a finished search: the final report (unless the GUI stopped the search,
    /// when it has moved on and the last periodic report stands), then `Done`, which sends
    /// the `bestmove`.
    fn conclude(&mut self, summary: Summary, report: &mut dyn FnMut(Report<'_>)) -> Summary {
        if summary.stop_reason != StopReason::Stopped {
            self.emit_info(&summary, report);
        }
        report(Report::Done(&summary));
        summary
    }

    /// A limit the search has reached, checked between batches. A proof ends an ordinary
    /// search here; an infinite one holds on it instead (the pipeline finds nothing to
    /// gather and [`Self::hold`]s).
    fn reached(&self, limits: &Limits, stop: &StopSignal) -> Option<StopReason> {
        if !limits.until_stopped
            && let Some(reason) = self.proof_stop()
        {
            return Some(reason);
        }
        if stop.is_set() {
            return Some(StopReason::Stopped);
        }
        if limits.nodes.is_some_and(|limit| self.nodes >= limit) {
            return Some(StopReason::NodeLimit);
        }
        if limits
            .depth
            .is_some_and(|limit| self.depth_count > 0 && self.mean_depth() >= limit as f32)
        {
            return Some(StopReason::DepthLimit);
        }
        if self
            .pacer
            .as_ref()
            .is_some_and(|pacer| Instant::now() >= pacer.hard_deadline())
        {
            return Some(StopReason::TimeLimit);
        }
        None
    }

    /// Start the clock at `now`: draw the move's budget from the time limit against the
    /// tree as it stands (its root visits are what a reused or pondered tree brings).
    fn arm_clock(&mut self, limits: &Limits, now: Instant) {
        self.nodes_at_clock = self.nodes;
        self.pacer = limits.time.map(|time| match time {
            TimeLimit::Movetime(movetime) => Pacer::new(
                Budget::exact(movetime, self.params.move_overhead),
                now,
                false,
            ),
            TimeLimit::Clock { clock, state } => {
                let inherited = self.tree.node(ROOT).visits;
                let budget = Budget::adaptive(
                    clock,
                    self.root.game_ply(),
                    self.params.move_overhead,
                    state.bank,
                    state.reuse_for(inherited),
                );
                Pacer::new(budget, now, true)
            }
        });
    }

    /// The pondered move was played: search on under the limits, clock from now.
    fn ponderhit(&mut self, limits: &Limits) {
        self.pondering = false;
        self.arm_clock(limits, Instant::now());
    }

    /// Whether the search may not finish on its own.
    fn holds(&self, limits: &Limits) -> bool {
        self.pondering || limits.until_stopped
    }

    /// Nothing left to search but not allowed to finish: wait for `stop`, or for
    /// `ponderhit` while pondering (`None`: the search goes on under its limits).
    fn hold(&mut self, limits: &Limits, stop: &StopSignal) -> Option<StopReason> {
        loop {
            if stop.is_set() {
                return Some(StopReason::Stopped);
            }
            if self.pondering && stop.is_ponderhit() {
                self.ponderhit(limits);
                return None;
            }
            std::thread::sleep(HOLD_POLL);
        }
    }

    /// The summary of a search that is over before it began (no moves, one move, a
    /// tablebase root), once it may be delivered: at once for an ordinary search, after
    /// `stop` or `ponderhit` for one that holds.
    fn finish(
        &mut self,
        started: Instant,
        reason: StopReason,
        limits: &Limits,
        stop: &StopSignal,
    ) -> Summary {
        if self.holds(limits)
            && let Some(stopped) = self.hold(limits, stop)
        {
            return self.summary(started, stopped);
        }
        self.summary(started, reason)
    }

    /// Leaves to ask for in the next batch: the configured `Batch`, ramped up with the size
    /// of the tree so a small tree is not swamped. A batch chooses all its leaves with no
    /// feedback between them; the smaller it is relative to the tree, the closer the
    /// search stays to sequential PUCT. Visits in flight count as tree size so the ramp
    /// does not stall while the pipeline is full.
    fn batch_size(&self) -> u64 {
        let batch = u64::from(self.params.batch);
        if self.params.batch_ramp == 0 {
            return batch;
        }
        let root = self.tree.node(ROOT);
        let size = root.visits + u64::from(root.virtual_visits);
        (size / u64::from(self.params.batch_ramp))
            .max(RAMP_MIN)
            .min(batch)
    }

    /// Gather one batch of up to `want` visits into `batch` with a single walk of the tree:
    /// at each node the budget is split among the children the way sequential selection
    /// would split it (see [`Tree::allocate`]), and each child is entered once with its
    /// share. A share that reaches an unexpanded leaf is one network evaluation; the rest
    /// of that share, and any share that reaches a leaf already in flight, are visits the
    /// batch could not place (counted as collisions, but never re-descended). Proven and
    /// terminal leaves are backed up on the spot, weighted by their share.
    fn gather(
        &mut self,
        batch: &mut Batch,
        want: u64,
        node_limit: Option<u64>,
        queued: usize,
        line: &mut Game,
        path: &mut Vec<u32>,
    ) -> Gathered {
        let mut gathered = Gathered {
            leaves: 0,
            immediate: 0,
        };
        // The batch is budgeted in leaves handed to the network. One walk places at most
        // its budget in leaves and usually fewer (shares lost to collisions), so walk
        // again with what is left; the reservations of earlier walks steer later ones
        // elsewhere. Stop when the batch is full, when a walk places nothing new, or when
        // the visits lost to collisions exceed the budget: past that point the batch is
        // being filled with leaves the search did not ask for, and a smaller, truer batch
        // is worth more than a full one.
        let lost_before = self.collisions;
        let lost_limit = if self.params.collision_budget_percent == 0 {
            u64::MAX
        } else {
            want * u64::from(self.params.collision_budget_percent) / 100
        };
        loop {
            let placed = gathered.leaves as u64;
            let mut budget = want.saturating_sub(placed);
            if let Some(limit) = node_limit {
                budget = budget.min(limit.saturating_sub(self.nodes + queued as u64 + placed));
            }
            let budget = budget.min(u64::from(u32::MAX)) as u32;
            if budget == 0 || self.proof_stop().is_some() {
                break;
            }
            if placed > 0 && self.collisions - lost_before > lost_limit {
                break;
            }
            line.clone_from(self.root);
            path.clear();
            path.push(ROOT);
            self.walks += 1;
            self.visit(ROOT, budget, line, path, batch, &mut gathered);
            if gathered.leaves as u64 == placed {
                break;
            }
        }
        gathered
    }

    /// Place `budget` visits at `node` (the last entry of `path`, with `line` at its
    /// position), recursing into children by their allotments.
    fn visit(
        &mut self,
        node: u32,
        budget: u32,
        line: &mut Game,
        path: &mut Vec<u32>,
        batch: &mut Batch,
        gathered: &mut Gathered,
    ) {
        let depth = (path.len() - 1) as u32;
        let current = self.tree.node(node);

        if current.pending {
            // Already in flight: nothing below it can be reached until it is evaluated.
            self.tree.reserve_n(path, budget);
            batch.reserved.extend(path.iter().map(|&n| (n, budget)));
            self.collisions += u64::from(budget);
            return;
        }
        if let Some(proof) = current.proof {
            self.tree
                .backup_n(path, proof.value(), proof.draw(), budget);
            self.nodes += u64::from(budget);
            gathered.immediate += u64::from(budget);
            self.record_depth(depth);
            return;
        }
        if !current.is_expanded() {
            self.record_depth(depth);
            // One move generation serves the terminal check and the expansion.
            let legal_moves = line.legal_moves();
            if let Some(proof) = terminal_proof(line, &legal_moves) {
                self.tree
                    .backup_n(path, proof.value(), proof.draw(), budget);
                self.tree.solve(path, proof);
                self.nodes += u64::from(budget);
                gathered.immediate += u64::from(budget);
                return;
            }
            if let Some(proof) = self.tablebase.and_then(|tb| tb.probe(line.position())) {
                self.tbhits += 1;
                self.tree
                    .backup_n(path, proof.value(), proof.draw(), budget);
                self.tree.solve(path, proof);
                self.nodes += u64::from(budget);
                gathered.immediate += u64::from(budget);
                return;
            }
            self.tree.node_mut(node).pending = true;
            self.tree.reserve_n(path, budget);
            batch.reserved.extend(path.iter().map(|&n| (n, budget)));
            self.collisions += u64::from(budget - 1);
            batch.leaves.push(Pending {
                board: encode_board(line),
                legal: LegalActions::from_moves(line.position(), &legal_moves),
                path_start: batch.paths.len(),
                path_len: path.len(),
            });
            batch.paths.extend_from_slice(path);
            gathered.leaves += 1;
            return;
        }

        let children = self.tree.children(node);
        if children.is_empty() {
            return; // expanded with no moves cannot happen: terminals are never expanded
        }
        let mut allotments = self.scratch.pop().unwrap_or_default();
        self.tree.allocate(
            node,
            budget,
            self.params.cpuct,
            self.params.fpu_reduction,
            &mut allotments,
        );
        for (offset, &share) in allotments.shares.iter().enumerate() {
            if share == 0 {
                continue;
            }
            let child = children.start + offset as u32;
            let mv = self.tree.node(child).mv.expect("children carry moves");
            let before = line.position().clone();
            line.play(mv);
            path.push(child);
            self.visit(child, share, line, path, batch, gathered);
            path.pop();
            line.undo(before);
        }
        self.scratch.push(allotments);
    }

    fn record_depth(&mut self, depth: u32) {
        self.max_depth = self.max_depth.max(depth);
        self.depth_sum += u64::from(depth);
        self.depth_count += 1;
    }

    fn mean_depth(&self) -> f32 {
        if self.depth_count > 0 {
            self.depth_sum as f32 / self.depth_count as f32
        } else {
            0.0
        }
    }

    /// Expand and back up every evaluated leaf of `batch`, then release its reservations.
    fn apply(&mut self, batch: &Batch) {
        assert_eq!(
            batch.evaluations.len(),
            batch.leaves.len(),
            "the network answered a different number of positions than asked"
        );
        self.network_evals += batch.leaves.len() as u64;
        self.batches += 1;
        for (leaf, evaluation) in batch.leaves.iter().zip(&batch.evaluations) {
            let leaf_path = &batch.paths[leaf.path_start..leaf.path_start + leaf.path_len];
            let index = *leaf_path.last().expect("a path has a leaf");
            self.tree.node_mut(index).pending = false;
            self.tree.expand(index, &leaf.legal, &evaluation.priors);
            // A non-finite value would poison every mean above it; count it as a draw.
            let (value, draw) = (evaluation.value(), evaluation.wdl[1]);
            let (value, draw) = if value.is_finite() && draw.is_finite() {
                (value, draw.clamp(0.0, 1.0))
            } else {
                (0.0, 1.0)
            };
            self.tree.backup(leaf_path, value, draw);
            self.nodes += 1;
        }
        self.release(batch);
    }

    /// Give up on a batch the network could not evaluate: its leaves are no longer
    /// pending and their reservations are released, so the tree is consistent for the
    /// summary.
    fn discard(&mut self, batch: &Batch) {
        for leaf in &batch.leaves {
            let leaf_path = &batch.paths[leaf.path_start..leaf.path_start + leaf.path_len];
            let index = *leaf_path.last().expect("a path has a leaf");
            self.tree.node_mut(index).pending = false;
        }
        self.release(batch);
    }

    /// Release the virtual visits a batch's walks reserved.
    fn release(&mut self, batch: &Batch) {
        for &(node, count) in &batch.reserved {
            self.tree.unreserve_n(&[node], count);
        }
    }

    fn proof_stop(&self) -> Option<StopReason> {
        match self.tree.node(ROOT).proof {
            Some(Proof::Win(_)) => Some(StopReason::ProvenWin),
            Some(Proof::Draw) => Some(StopReason::ProvenDraw),
            Some(Proof::Loss(_)) => Some(StopReason::ProvenLoss),
            None if self.tree.forced_by_proof() => Some(StopReason::ForcedByProof),
            None => None,
        }
    }

    fn summary(&self, started: Instant, stop_reason: StopReason) -> Summary {
        let pv = self.tree.pv();
        let root = self.tree.node(ROOT);
        let (value, draw) = match root.proof {
            Some(proof) => (proof.value(), proof.draw()),
            None if root.visits > 0 => (root.mean(), root.draw_mean()),
            None => (self.root_value, self.root_draw),
        };
        let now = Instant::now();
        Summary {
            best: pv.first().copied(),
            ponder: pv.get(1).copied(),
            nodes: self.nodes,
            network_evals: self.network_evals,
            batches: self.batches,
            walks: self.walks,
            collisions: self.collisions,
            tbhits: self.tbhits,
            reused: self.reused,
            elapsed: now.duration_since(started),
            max_depth: self.max_depth,
            avg_depth: self.mean_depth(),
            value,
            draw,
            proof: root.proof,
            stop_reason,
            failure: self.failure.clone(),
            settlement: self.pacer.as_ref().and_then(|pacer| {
                Some(Settlement {
                    budget: pacer.adaptive_budget()?,
                    used: pacer.elapsed(now),
                    new: self.nodes - self.nodes_at_clock,
                })
            }),
        }
    }

    /// One `info` line per reported root move: the move to play first, then the next
    /// `MultiPV - 1` moves in [`Tree::ranked_root_children`] order, each with its own
    /// score, `wdl` and line; a move not yet visited carries the search's first-play
    /// estimate for it. Depth, time and node counts are the search's. GUIs expect the full
    /// set of lines at one depth before they show anything, and a depth that never falls
    /// within a search; the mean simulation depth can dip as the tree broadens, so the
    /// reported depth is its running maximum.
    fn emit_info(&mut self, summary: &Summary, report: &mut dyn FnMut(Report<'_>)) {
        self.last_info = Instant::now();
        self.reported_depth = self
            .reported_depth
            .max((summary.avg_depth.round() as u32).max(1));
        let lines = self.params.multipv.max(1) as usize;
        let ranked = self.tree.ranked_root_children();
        let mut emit = |rank: usize, assessment: Assessment, pv: &[String]| {
            let pv: Vec<&str> = pv.iter().map(String::as_str).collect();
            let line = Info {
                depth: Some(self.reported_depth),
                seldepth: Some(summary.max_depth.max(1)),
                multipv: (lines > 1).then_some(rank as u32 + 1),
                score: Some(assessment.score),
                wdl: self.params.show_wdl.then_some(assessment.wdl),
                time: Some(summary.elapsed.as_millis() as u64),
                nodes: Some(summary.nodes),
                nps: Some(summary.nps() as u64),
                tbhits: Some(summary.tbhits),
                pv: &pv,
                ..Info::default()
            };
            if let Some(rendered) = line.render() {
                report(Report::Info(rendered));
            }
        };
        let score_type = self.params.score_type;
        let root = Assessment::of(summary.proof, summary.value, summary.draw, score_type);
        if ranked.is_empty() {
            // Nothing expanded at the root (a tablebase move, no moves, a network failure):
            // the root's own assessment and the move, if there is one.
            let pv: Vec<String> = summary
                .best
                .map(|mv| self.root.uci(mv))
                .into_iter()
                .collect();
            emit(0, root, &pv);
            return;
        }
        // What selection assumes of a move it has not tried: the root's value less the
        // first-play reduction, and the root's draw probability.
        let root_node = self.tree.node(ROOT);
        let unvisited = Assessment::of(
            None,
            root_node.mean() - self.params.fpu_reduction * root_node.explored_mass().sqrt(),
            summary.draw,
            score_type,
        );
        for (rank, &child) in ranked.iter().take(lines).enumerate() {
            let node = self.tree.node(child);
            let assessment = if node.visits > 0 || node.proof.is_some() {
                Assessment::of(
                    node.proof.map(Proof::from_parent),
                    -node.mean(),
                    node.draw_mean(),
                    score_type,
                )
            } else if rank == 0 {
                root // the move to play before any visit came back: the root's own view
            } else {
                unvisited
            };
            let pv: Vec<String> = self
                .tree
                .pv_from(child)
                .iter()
                .map(|&mv| self.root.uci(mv))
                .collect();
            emit(rank, assessment, &pv);
        }
    }
}

/// A position's `score` and `wdl` from the engine's side, for an `info` line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Assessment {
    score: Score,
    wdl: (u32, u32, u32),
}

impl Assessment {
    /// From a proof when there is one, else from the value `q` and draw probability `d`.
    fn of(proof: Option<Proof>, q: f32, d: f32, score_type: ScoreType) -> Self {
        let score = match proof {
            // Tablebase proofs know the result but not the distance.
            Some(Proof::Win(plies)) if plies >= TB_PLIES => Score::Cp(TB_CENTIPAWNS),
            Some(Proof::Loss(plies)) if plies >= TB_PLIES => Score::Cp(-TB_CENTIPAWNS),
            Some(proof) => match proof.mate_in_moves() {
                Some(moves) => Score::Mate(moves),
                None => Score::Cp(0),
            },
            None => Score::Cp(centipawns(q, score_type)),
        };
        let wdl = match proof {
            Some(proof) => wdl_permille(proof.value(), proof.draw()),
            None => wdl_permille(q, d),
        };
        Self { score, wdl }
    }
}

/// Win, draw and loss in permille from the value `q = P(win) - P(loss)` and the draw
/// probability `d`; the three sum to 1000.
pub fn wdl_permille(q: f32, d: f32) -> (u32, u32, u32) {
    let d = d.clamp(0.0, 1.0);
    let w = ((1.0 + q - d) / 2.0).clamp(0.0, 1.0);
    let l = ((1.0 - q - d) / 2.0).clamp(0.0, 1.0);
    let total = w + d + l;
    let w = (1000.0 * w / total).round() as u32;
    let l = (1000.0 * l / total).round() as u32;
    (w, 1000u32.saturating_sub(w + l), l)
}

/// Exact result of a position inside the tree, from its side to move, if the game is over
/// there under [`DrawRules::SEARCH`]. `legal` are the position's legal moves.
fn terminal_proof(game: &Game, legal: &shakmaty::MoveList) -> Option<Proof> {
    match game.game_end_given(DrawRules::SEARCH, legal)? {
        GameEnd::Checkmate { .. } => Some(Proof::Loss(0)),
        GameEnd::Stalemate
        | GameEnd::InsufficientMaterial
        | GameEnd::Repetition
        | GameEnd::HalfmoveClock => Some(Proof::Draw),
    }
}

/// Score reported for a tablebase win (Stockfish reports its own large constant too).
pub const TB_CENTIPAWNS: i32 = 9990;

/// Largest score reported for a position the search has not proven: the value head's
/// "certainly won" saturates here, below [`TB_CENTIPAWNS`] and any `mate`, so a GUI can
/// tell the three apart. Lc0's tangent mapping runs to 9999 at q = 0.999, which prints
/// like a forced mate and swings by thousands between reports as a confident line settles.
pub const MAX_CENTIPAWNS: i32 = 3000;

/// Expected value `q` in `[-1, 1]` to a centipawn score under `score_type`, saturating at
/// ±[`MAX_CENTIPAWNS`]. `Centipawn` is the logistic relation between centipawns and
/// expected score `s = (q + 1) / 2`, `cp = 400 log10(s / (1 - s))`, the scale rating
/// tools and most engines' users share (+100 ≈ 64%, +200 ≈ 76%, +500 ≈ 95%); `Lc0` is
/// `90 tan(1.5637 q)`, which reads smaller in the middle (q = 0.5 is +89 against +191).
/// The logistic tops out at +1320 (q clamped to 0.999) on its own; the cap bites only Lc0's.
pub fn centipawns(value: f32, score_type: ScoreType) -> i32 {
    let q = value.clamp(-0.999, 0.999);
    let cp = match score_type {
        ScoreType::Centipawn => {
            let s = (q + 1.0) / 2.0;
            400.0 * (s / (1.0 - s)).log10()
        }
        ScoreType::Lc0 => 90.0 * (1.5637 * q).tan(),
    };
    cp.round()
        .clamp(-(MAX_CENTIPAWNS as f32), MAX_CENTIPAWNS as f32) as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::{HashNetwork, LatencyNetwork, UniformNetwork};
    use shakmaty::CastlingMode;

    fn game(fen: Option<&str>, moves: &str) -> Game {
        let moves: Vec<String> = moves.split_whitespace().map(str::to_string).collect();
        Game::from_uci(fen, &moves, CastlingMode::Standard).unwrap()
    }

    fn params(batch: u32) -> Params {
        Params {
            cpuct: 1.5,
            fpu_reduction: 0.25,
            batch,
            in_flight: IN_FLIGHT,
            batch_ramp: 4,
            collision_budget_percent: 100,
            move_overhead: Duration::ZERO,
            multipv: 1,
            show_wdl: false,
            score_type: ScoreType::Centipawn,
        }
    }

    fn nodes(n: u64) -> Limits {
        Limits {
            nodes: Some(n),
            depth: None,
            time: None,
            until_stopped: false,
            ponder: false,
        }
    }

    const INFINITE: Limits = Limits {
        nodes: None,
        depth: None,
        time: None,
        until_stopped: true,
        ponder: false,
    };

    fn run(game: &Game, limits: &Limits, batch: u32, network: &dyn Network) -> Summary {
        let stop = StopSignal::new();
        search(
            game,
            limits,
            None,
            &params(batch),
            network,
            None,
            &stop,
            &mut |_| {},
        )
    }

    #[test]
    fn reused_tree_continues_the_search() {
        let start = game(None, "");
        let net = HashNetwork::default();
        let stop = StopSignal::new();
        let (first, tree) = search_with_tree(
            &start,
            &nodes(400),
            None,
            &params(8),
            &net,
            None,
            None,
            &stop,
            &mut |_| {},
        );
        let best = first.best.unwrap();
        // Play the chosen move and the opponent's most-visited reply.
        let reply_index = tree.child_by_move(ROOT, best).unwrap();
        let reply = tree
            .children(reply_index)
            .max_by_key(|&c| tree.node(c).visits)
            .map(|c| tree.node(c).mv.unwrap())
            .unwrap();
        let continued = game(None, &format!("{} {}", start.uci(best), start.uci(reply)));
        let moves = continued.continuation_from(&start).unwrap().to_vec();
        assert_eq!(moves, vec![best, reply]);
        let reused = tree.reroot(&moves).unwrap();
        let carried = reused.node(ROOT).visits;
        assert!(carried > 0 && carried < 400, "{carried}");
        // Every node of the re-rooted tree is a real copy: values and priors intact.
        let old_child = tree.node(tree.child_by_move(reply_index, reply).unwrap());
        assert_eq!(reused.node(ROOT).visits, old_child.visits);
        assert_eq!(reused.node(ROOT).value_sum, old_child.value_sum);
        assert_eq!(reused.children(ROOT).len(), continued.legal_moves().len());

        let (second, _) = search_with_tree(
            &continued,
            &nodes(100),
            None,
            &params(8),
            &net,
            None,
            Some(reused),
            &stop,
            &mut |_| {},
        );
        assert_eq!(second.reused, carried);
        assert_eq!(second.nodes, 100, "node limit counts new simulations only");
        assert_eq!(
            second.network_evals, 100,
            "the reused root is not re-evaluated"
        );
        // A fresh search of the same position matches a fresh tree's accounting.
        let fresh = run(&continued, &nodes(100), 8, &net);
        assert_eq!(fresh.reused, 0);
        assert_eq!(fresh.network_evals, 101);
    }

    /// The tree a search leaves is tagged with its network generation, and only a search of
    /// the same generation continues from it. This is what protects the first search after
    /// a `setoption name Network` from a tree the old network's search stored after the
    /// engine had already forgotten the old tree.
    #[test]
    fn a_tree_from_another_network_generation_is_not_reused() {
        let start = game(None, "");
        let network: Arc<dyn Network> = Arc::new(HashNetwork::default());
        let saved = Arc::new(Mutex::new(None));
        let shared = |generation: u64| Shared {
            network: Arc::clone(&network),
            generation,
            tablebase: None,
            time_state: Arc::new(Mutex::new(TimeState::default())),
            tree: Arc::clone(&saved),
        };
        let go = |nodes: u64| GoLimits {
            nodes: Some(nodes),
            ..GoLimits::default()
        };
        let (events, _received) = mpsc::channel();
        let root_visits = || {
            let guard = saved.lock().unwrap();
            let saved = guard.as_ref().expect("a finished search leaves its tree");
            (saved.generation, saved.tree.node(ROOT).visits)
        };
        let options = Options::default();
        let stop = Arc::new(StopSignal::new());

        super::run(
            start.clone(),
            go(200),
            options.clone(),
            shared(1),
            Arc::clone(&stop),
            events.clone(),
        );
        assert_eq!(root_visits(), (1, 200));
        // The same generation continues from the tree: 200 inherited plus 100 new.
        super::run(
            start.clone(),
            go(100),
            options.clone(),
            shared(1),
            Arc::clone(&stop),
            events.clone(),
        );
        assert_eq!(root_visits(), (1, 300));
        // Another generation starts afresh, and the tree it leaves carries its own tag.
        super::run(start, go(100), options, shared(2), stop, events);
        assert_eq!(root_visits(), (2, 100));
    }

    /// Visits and values of every expanded node equal one (its own evaluation) plus its
    /// children's, with the children's values negated; `explored` equals the visited
    /// children's prior mass. Holds for a finished search and must survive re-rooting.
    fn check_tree(tree: &Tree) {
        for index in 0..tree.len() as u32 {
            let node = tree.node(index);
            assert!(
                !node.pending && node.virtual_visits == 0,
                "node {index} still in flight"
            );
            if !node.is_expanded() {
                continue;
            }
            let mut visits = 0u64;
            let mut value = 0.0f32;
            let mut explored = 0.0f32;
            for child in tree.children(index).map(|c| tree.node(c)) {
                visits += child.visits;
                value -= child.value_sum as f32;
                if child.visits > 0 {
                    explored += child.prior;
                }
            }
            if node.proof.is_none() {
                assert!(
                    node.visits == visits + 1 || (index == ROOT && node.visits == visits),
                    "node {index}: visits {} vs children {visits}",
                    node.visits
                );
            }
            assert!(
                (tree.node(index).explored_mass() - explored).abs() < 1e-3,
                "node {index}: explored {} vs {explored}",
                tree.node(index).explored_mass()
            );
            if node.proof.is_none() {
                // value_sum = own evaluation − Σ children's sums (their side to move).
                let own = node.value_sum as f32 - value;
                assert!(
                    own.abs() <= 1.0 + 1e-3,
                    "node {index}: own value {own} (visits {})",
                    node.visits
                );
            }
        }
    }

    #[test]
    fn rerooted_tree_keeps_its_invariants() {
        let start = game(None, "");
        let (_, tree) = search_with_tree(
            &start,
            &nodes(600),
            None,
            &params(8),
            &HashNetwork::default(),
            None,
            None,
            &StopSignal::new(),
            &mut |_| {},
        );
        check_tree(&tree);
        let best = tree.pv();
        let reused = tree.reroot(&best[..2]).unwrap();
        check_tree(&reused);
        assert_eq!(
            reused.node(ROOT).visits,
            tree.node(
                tree.child_by_move(tree.child_by_move(ROOT, best[0]).unwrap(), best[1])
                    .unwrap()
            )
            .visits
        );
    }

    /// A walk whose every share lands on a leaf already in flight hands the network
    /// nothing; its reservations must still be released, or they outlive the search.
    #[test]
    fn collision_only_gathers_leave_no_reservations() {
        // Three legal moves (Kg8, Kg7, Kh7) and a batch of 8 with two in flight: the second
        // batch finds every leaf pending and places nothing.
        let g = game(Some("7k/8/8/8/8/8/8/R6K b - - 0 1"), "");
        for limit in [8u64, 30, 200] {
            let (summary, tree) = search_with_tree(
                &g,
                &nodes(limit),
                None,
                &params(8),
                &UniformNetwork,
                None,
                None,
                &StopSignal::new(),
                &mut |_| {},
            );
            assert!(summary.collisions > 0, "{summary:?}");
            check_tree(&tree);
        }
    }

    #[test]
    fn reroot_declines_unexpanded_and_unknown_paths() {
        let start = game(None, "");
        let (_, tree) = search_with_tree(
            &start,
            &nodes(50),
            None,
            &params(8),
            &HashNetwork::default(),
            None,
            None,
            &StopSignal::new(),
            &mut |_| {},
        );
        // 50 nodes over 20 root moves: some root child is still unexpanded, and re-rooting
        // on it has nothing to reuse.
        let leaf = tree
            .children(ROOT)
            .find(|&c| !tree.node(c).is_expanded())
            .map(|c| tree.node(c).mv.unwrap())
            .expect("an unexpanded root child");
        assert!(tree.reroot(&[leaf]).is_none());
        // A move that is not in the tree at all.
        let unknown = start
            .legal_moves()
            .iter()
            .copied()
            .find(|&mv| tree.child_by_move(ROOT, mv).is_none());
        if let Some(mv) = unknown {
            assert!(tree.reroot(&[mv]).is_none());
        }
        assert!(
            tree.reroot(&[]).is_some(),
            "the same position reuses the whole tree"
        );
    }

    #[test]
    fn node_limit_is_exact_and_deterministic() {
        let g = game(None, "");
        let a = run(&g, &nodes(500), 16, &HashNetwork::default());
        let b = run(&g, &nodes(500), 16, &HashNetwork::default());
        assert_eq!(a.nodes, 500);
        assert_eq!(a.stop_reason, StopReason::NodeLimit);
        assert_eq!(a.best, b.best);
        assert_eq!(a.value, b.value);
        assert!(a.max_depth >= 3, "{a:?}");
        assert!(a.avg_depth > 1.0);
    }

    #[test]
    fn batch_ramps_with_the_tree() {
        // With Batch 256 and 800 nodes an unramped search needs 3 to 4 network calls; the
        // ramp starts at RAMP_MIN and allows `tree size / batch_ramp` leaves per batch, so
        // many more.
        let g = game(None, "");
        let summary = run(&g, &nodes(800), 256, &HashNetwork::default());
        assert_eq!(summary.nodes, 800);
        assert!(summary.batches >= 10, "{summary:?}");
        // Early batches are small, later ones larger: mean well under the cap.
        let mean = summary.network_evals as f64 / summary.batches as f64;
        assert!(mean < 128.0, "mean batch {mean} {summary:?}");
        // Every leaf sent was evaluated once; leaves never exceed simulations.
        assert!(summary.network_evals <= summary.nodes + 1);
    }

    #[test]
    fn finds_mate_in_one_by_proof() {
        // Qd8# is available; the network knows nothing about it.
        let g = game(Some("6k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1"), "");
        let summary = run(&g, &nodes(2000), 8, &UniformNetwork);
        assert_eq!(g.uci(summary.best.unwrap()), "d1d8");
        assert_eq!(summary.proof, Some(Proof::Win(1)));
        assert_eq!(summary.stop_reason, StopReason::ProvenWin);
        assert!(summary.nodes < 2000, "proof should stop early: {summary:?}");
    }

    #[test]
    fn finds_mate_in_two_with_distance() {
        // Doubled rooks on the e-file: 1.Re8+ Rxe8 2.Rxe8#; Black's reply is forced.
        let g = game(Some("r5k1/5ppp/8/8/8/8/4RPPP/4R1K1 w - - 0 1"), "");
        let summary = run(&g, &nodes(20_000), 32, &HashNetwork::default());
        assert_eq!(summary.proof, Some(Proof::Win(3)), "{summary:?}");
        assert_eq!(summary.stop_reason, StopReason::ProvenWin);
        assert_eq!(g.uci(summary.best.unwrap()), "e2e8");
        assert_eq!(summary.proof.unwrap().mate_in_moves(), Some(2));
    }

    #[test]
    fn avoids_a_proven_loss() {
        // Black to move with White's queen on h5: some replies allow mate in one. The move
        // chosen must not be one of them.
        let g = game(Some("6k1/5p1p/6p1/7Q/8/8/5PPP/6K1 b - - 0 1"), "");
        let summary = run(&g, &nodes(3000), 16, &HashNetwork::default());
        let best = summary.best.unwrap();
        let mut after = g.clone();
        after.play(best);
        let mates_in_one = after.legal_moves().iter().any(|&reply| {
            let mut next = after.clone();
            next.play(reply);
            next.game_end()
                .is_some_and(|end| matches!(end, GameEnd::Checkmate { .. }))
        });
        assert!(!mates_in_one, "{} walks into mate", g.uci(best));
    }

    #[test]
    fn terminal_positions_and_only_moves() {
        let mated = game(None, "f2f3 e7e5 g2g4 d8h4");
        let summary = run(&mated, &nodes(10), 8, &UniformNetwork);
        assert_eq!(summary.best, None);
        assert_eq!(summary.stop_reason, StopReason::GameOver);

        // Checked king in the corner: a7 stays on the rook's file, b7 is covered by the
        // pawn, so Kb8 is the one legal move.
        let only = game(Some("k7/8/2P5/8/8/8/8/R6K b - - 0 1"), "");
        assert_eq!(only.legal_moves().len(), 1);
        let summary = run(&only, &nodes(100), 8, &UniformNetwork);
        assert_eq!(summary.stop_reason, StopReason::OnlyMove);
        assert_eq!(summary.nodes, 0);
        assert_eq!(only.uci(summary.best.unwrap()), "a8b8");
    }

    #[test]
    fn repetition_inside_the_tree_is_a_draw() {
        // The knights have shuffled out and back once already; Black's f6g8 now repeats the
        // start position, which the search treats as a draw and proves at that leaf.
        let g = game(None, "g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1");
        let (summary, tree) = search_with_tree(
            &g,
            &nodes(300),
            None,
            &params(8),
            &UniformNetwork,
            None,
            None,
            &StopSignal::new(),
            &mut |_| {},
        );
        assert!(summary.best.is_some());
        let back = g
            .legal_moves()
            .iter()
            .copied()
            .find(|&mv| g.uci(mv) == "f6g8")
            .unwrap();
        let child = tree
            .child_by_move(ROOT, back)
            .expect("root children are expanded");
        assert_eq!(tree.node(child).proof, Some(Proof::Draw));
    }

    #[test]
    fn stop_signal_ends_an_unbounded_search() {
        let g = game(None, "");
        let stop = Arc::new(StopSignal::new());
        let stopper = Arc::clone(&stop);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            stopper.stop();
        });
        let summary = search(
            &g,
            &INFINITE,
            None,
            &params(64),
            &UniformNetwork,
            None,
            &stop,
            &mut |_| {},
        );
        assert_eq!(summary.stop_reason, StopReason::Stopped);
        assert!(summary.nodes > 0, "{summary:?}");
    }

    #[test]
    fn deadline_is_honoured() {
        let g = game(None, "");
        let limits = Limits {
            time: Some(TimeLimit::Movetime(Duration::from_millis(30))),
            ..nodes(u64::MAX)
        };
        let stop = StopSignal::new();
        let summary = search(
            &g,
            &limits,
            None,
            &params(64),
            &UniformNetwork,
            None,
            &stop,
            &mut |_| {},
        );
        assert_eq!(summary.stop_reason, StopReason::TimeLimit);
        assert!(summary.elapsed < Duration::from_secs(2), "{summary:?}");
    }

    #[test]
    fn searchmoves_restricts_the_root() {
        let g = game(None, "");
        let stop = StopSignal::new();
        let only = vec!["a2a3".to_string()];
        let summary = search(
            &g,
            &nodes(50),
            Some(&only),
            &params(8),
            &HashNetwork::default(),
            None,
            &stop,
            &mut |_| {},
        );
        assert_eq!(summary.stop_reason, StopReason::OnlyMove);
        assert_eq!(g.uci(summary.best.unwrap()), "a2a3");
    }

    #[test]
    fn info_lines_carry_score_and_pv() {
        let g = game(None, "");
        let stop = StopSignal::new();
        let mut lines = Vec::new();
        search(
            &g,
            &nodes(200),
            None,
            &params(16),
            &HashNetwork::default(),
            None,
            &stop,
            &mut |report| {
                if let Report::Info(line) = report {
                    lines.push(line)
                }
            },
        );
        let last = lines.last().unwrap();
        assert!(last.starts_with("info depth "), "{last}");
        assert!(
            last.contains(" score cp ") && last.contains(" nodes 200 ") && last.contains(" pv ")
        );
        for st in [ScoreType::Centipawn, ScoreType::Lc0] {
            assert_eq!(centipawns(0.0, st), 0);
            assert!(centipawns(0.5, st) > 50 && centipawns(-0.5, st) < -50);
            assert_eq!(centipawns(1.0, st), centipawns(0.999, st));
            assert_eq!(centipawns(-1.0, st), -centipawns(1.0, st));
            assert!(centipawns(1.0, st) > 1000 && centipawns(1.0, st) <= MAX_CENTIPAWNS);
            // Monotone.
            let mut last = i32::MIN;
            for i in -99..=99 {
                let cp = centipawns(i as f32 / 100.0, st);
                assert!(cp >= last, "{st:?} at q={}", i as f32 / 100.0);
                last = cp;
            }
        }
        // The logistic scale: +100 is 64% expected score, i.e. q = 0.28.
        assert_eq!(centipawns(0.28, ScoreType::Centipawn), 100);
        assert_eq!(centipawns(0.5, ScoreType::Centipawn), 191);
        assert_eq!(centipawns(1.0, ScoreType::Centipawn), 1320); // its own ceiling
        assert_eq!(centipawns(0.5, ScoreType::Lc0), 89);
        assert_eq!(centipawns(1.0, ScoreType::Lc0), MAX_CENTIPAWNS);
        const { assert!(MAX_CENTIPAWNS < TB_CENTIPAWNS) };
    }

    #[test]
    fn clock_limits_resolve_against_the_side_to_move() {
        let g = game(None, "e2e4");
        let go = GoLimits {
            wtime: Some(60_000),
            btime: Some(10_000),
            binc: Some(1000),
            ..GoLimits::default()
        };
        let state = TimeState {
            bank: Duration::from_millis(100),
            ..TimeState::default()
        };
        let limits = Limits::from_go(&go, &g, state);
        assert!(!limits.until_stopped && !limits.ponder);
        let Some(TimeLimit::Clock { clock, state: kept }) = limits.time else {
            panic!("{limits:?}");
        };
        assert_eq!(clock.remaining, Duration::from_millis(10_000));
        assert_eq!(clock.increment, Duration::from_millis(1000));
        assert_eq!(kept, state);
        let infinite = Limits::from_go(
            &GoLimits {
                infinite: true,
                ..GoLimits::default()
            },
            &g,
            state,
        );
        assert!(infinite.until_stopped && infinite.time.is_none());
        let ponder = Limits::from_go(&GoLimits { ponder: true, ..go }, &g, state);
        assert!(ponder.ponder && ponder.time.is_some());
    }

    #[test]
    fn a_clock_move_settles_the_bank_from_its_budget() {
        // 10 s + 0.1 s at ply 0 budgets ~270 ms soft; a search of a handful of nodes uses
        // almost none of it, and the settlement says what was budgeted and used.
        let g = game(None, "");
        let limits = Limits {
            nodes: Some(20),
            time: Some(TimeLimit::Clock {
                clock: Clock {
                    remaining: Duration::from_secs(10),
                    increment: Duration::from_millis(100),
                    movestogo: None,
                },
                state: TimeState::default(),
            }),
            ..nodes(20)
        };
        let summary = run(&g, &limits, 8, &UniformNetwork);
        assert_eq!(summary.stop_reason, StopReason::NodeLimit);
        let settlement = summary.settlement.expect("a clock move settles");
        assert!(
            settlement.budget.soft > Duration::from_millis(200),
            "{settlement:?}"
        );
        assert_eq!(settlement.new, 20);
        assert!(settlement.used <= summary.elapsed);
        // `go movetime` is exact and settles nothing.
        let exact = Limits {
            time: Some(TimeLimit::Movetime(Duration::from_millis(20))),
            ..nodes(u64::MAX)
        };
        assert_eq!(run(&g, &exact, 8, &UniformNetwork).settlement, None);
    }

    #[test]
    fn wdl_and_multipv_lines() {
        let g = game(None, "");
        let stop = StopSignal::new();
        let mut lines = Vec::new();
        let p = Params {
            multipv: 3,
            show_wdl: true,
            ..params(16)
        };
        let summary = search(
            &g,
            &nodes(300),
            None,
            &p,
            &HashNetwork::default(),
            None,
            &stop,
            &mut |report| {
                if let Report::Info(line) = report {
                    lines.push(line)
                }
            },
        );
        // The final report is the last three lines: multipv 1, 2, 3 with distinct moves,
        // each carrying a wdl that sums to 1000 and a pv starting with its move.
        let last: Vec<&String> = lines.iter().rev().take(3).collect::<Vec<_>>();
        let last: Vec<&String> = last.into_iter().rev().collect();
        let mut moves = Vec::new();
        for (i, line) in last.iter().enumerate() {
            let tag = format!(" multipv {} ", i + 1);
            assert!(line.contains(&tag), "{line}");
            let words: Vec<&str> = line.split_whitespace().collect();
            let at = |key: &str| {
                words
                    .iter()
                    .position(|&w| w == key)
                    .unwrap_or_else(|| panic!("{line}"))
            };
            let w = at("wdl");
            let wdl: u32 = words[w + 1..w + 4]
                .iter()
                .map(|v| v.parse::<u32>().unwrap())
                .sum();
            assert_eq!(wdl, 1000, "{line}");
            moves.push(words[at("pv") + 1].to_string());
        }
        assert_eq!(moves.len(), 3);
        assert!(moves[0] != moves[1] && moves[1] != moves[2] && moves[0] != moves[2]);
        assert_eq!(moves[0], g.uci(summary.best.unwrap()));
        // Root wdl is consistent with the root value.
        let (w, d, l) = wdl_permille(summary.value, summary.draw);
        assert_eq!(w + d + l, 1000);
        assert!(((w as f32 - l as f32) / 1000.0 - summary.value).abs() < 0.01);

        // A proven mate reports a certain win.
        let mate = game(Some("6k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1"), "");
        let mut lines = Vec::new();
        search(
            &mate,
            &nodes(2000),
            None,
            &Params {
                show_wdl: true,
                ..params(8)
            },
            &UniformNetwork,
            None,
            &stop,
            &mut |report| {
                if let Report::Info(line) = report {
                    lines.push(line)
                }
            },
        );
        let last = lines.last().unwrap();
        assert!(
            last.contains(" score mate 1 wdl 1000 0 0 ") && !last.contains("multipv"),
            "{last}"
        );
        assert_eq!(wdl_permille(1.0, 0.0), (1000, 0, 0));
        assert_eq!(wdl_permille(0.0, 1.0), (0, 1000, 0));
        assert_eq!(wdl_permille(-1.0, 0.0), (0, 0, 1000));
        assert_eq!(wdl_permille(0.0, 0.0), (500, 0, 500));
    }

    #[test]
    fn every_multipv_line_is_reported_even_before_a_move_is_visited() {
        // Two simulations, three lines asked for: the third move has not been visited and
        // is reported at the first-play estimate, so a GUI waiting for all three sees them.
        let g = game(None, "");
        let mut lines = Vec::new();
        search(
            &g,
            &nodes(2),
            None,
            &Params {
                multipv: 3,
                ..params(8)
            },
            &UniformNetwork,
            None,
            &StopSignal::new(),
            &mut |report| {
                if let Report::Info(line) = report {
                    lines.push(line)
                }
            },
        );
        let tags: Vec<bool> = (1..=3)
            .map(|i| lines.iter().any(|l| l.contains(&format!(" multipv {i} "))))
            .collect();
        assert_eq!(tags, vec![true, true, true], "{lines:?}");
        // Depth never falls within a search.
        let depths: Vec<u32> = lines
            .iter()
            .map(|l| l.split_whitespace().nth(2).unwrap().parse().unwrap())
            .collect();
        assert!(depths.windows(2).all(|w| w[0] <= w[1]), "{depths:?}");
    }

    #[test]
    fn stop_is_answered_without_waiting_for_the_network() {
        // Batches take 300 ms at the network. `stop` must be answered (`Done`) within a
        // poll or two, not after the batches in flight come back; those are collected
        // afterwards and the second one, still queued, is not evaluated at all.
        let g = game(None, "");
        let net = LatencyNetwork {
            inner: UniformNetwork,
            per_batch: Duration::from_millis(300),
            per_leaf: Duration::ZERO,
        };
        let stop = Arc::new(StopSignal::new());
        let stopper = Arc::clone(&stop);
        let stopped_at = Arc::new(Mutex::new(None));
        let mark = Arc::clone(&stopped_at);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(400)); // root eval + first batch under way
            stopper.stop();
            *mark.lock().unwrap() = Some(Instant::now());
        });
        let mut done_at = None;
        let started = Instant::now();
        let summary = search(
            &g,
            &INFINITE,
            None,
            &params(8),
            &net,
            None,
            &stop,
            &mut |report| {
                if let Report::Done(_) = report {
                    done_at = Some(Instant::now());
                }
            },
        );
        let returned = started.elapsed();
        let stopped_at = stopped_at.lock().unwrap().expect("stopped");
        let latency = done_at.expect("Done reported").duration_since(stopped_at);
        assert!(
            latency < Duration::from_millis(60),
            "bestmove took {latency:?}"
        );
        assert_eq!(summary.stop_reason, StopReason::Stopped);
        // Returning waits for the batch being evaluated (≤ 300 ms) but not for the queued
        // one too.
        assert!(
            returned < Duration::from_millis(400 + 300 + 100),
            "{returned:?}"
        );
    }

    #[test]
    fn infinite_search_holds_on_a_proven_root_until_stopped() {
        // Mate in one: an ordinary search stops on the proof; an infinite one must not
        // send its move before `stop`.
        let g = game(Some("6k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1"), "");
        let stop = Arc::new(StopSignal::new());
        let stopper = Arc::clone(&stop);
        let started = Instant::now();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            stopper.stop();
        });
        let summary = search(
            &g,
            &INFINITE,
            None,
            &params(8),
            &UniformNetwork,
            None,
            &stop,
            &mut |_| {},
        );
        assert!(
            started.elapsed() >= Duration::from_millis(150),
            "{summary:?}"
        );
        assert_eq!(summary.stop_reason, StopReason::Stopped);
        assert_eq!(summary.proof, Some(Proof::Win(1)));
        assert_eq!(g.uci(summary.best.unwrap()), "d1d8");
    }

    #[test]
    fn ponder_holds_then_ponderhit_searches_under_the_limits() {
        let g = game(None, "");
        // Stopped while pondering: a bestmove for the pondered position, at once.
        let stop = Arc::new(StopSignal::new());
        let stopper = Arc::clone(&stop);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            stopper.stop();
        });
        let limits = Limits {
            ponder: true,
            ..nodes(10)
        };
        let summary = search(
            &g,
            &limits,
            None,
            &params(8),
            &UniformNetwork,
            None,
            &stop,
            &mut |_| {},
        );
        assert_eq!(summary.stop_reason, StopReason::Stopped);
        assert!(
            summary.nodes > 10,
            "the node limit does not apply while pondering: {summary:?}"
        );
        assert!(summary.best.is_some());
        assert_eq!(summary.settlement, None);

        // Ponderhit: the clock runs from then, and the settlement counts from then.
        let stop = Arc::new(StopSignal::new());
        let hitter = Arc::clone(&stop);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(60));
            hitter.ponderhit();
        });
        let limits = Limits {
            ponder: true,
            nodes: None,
            time: Some(TimeLimit::Clock {
                clock: Clock {
                    remaining: Duration::from_secs(1),
                    increment: Duration::ZERO,
                    movestogo: None,
                },
                state: TimeState::default(),
            }),
            ..nodes(0)
        };
        let started = Instant::now();
        let summary = search(
            &g,
            &limits,
            None,
            &params(8),
            &UniformNetwork,
            None,
            &stop,
            &mut |_| {},
        );
        assert!(
            matches!(
                summary.stop_reason,
                StopReason::TimeLimit | StopReason::SmartPruning
            ),
            "{summary:?}"
        );
        let settlement = summary.settlement.expect("a ponderhit move settles");
        assert!(settlement.used < started.elapsed());
        assert!(
            settlement.new < summary.nodes,
            "{settlement:?} of {}",
            summary.nodes
        );
        assert!(summary.elapsed >= Duration::from_millis(60));

        // A ponder on a mate-in-one holds on the proof and delivers it at ponderhit.
        let mate = game(Some("6k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1"), "");
        let stop = Arc::new(StopSignal::new());
        let hitter = Arc::clone(&stop);
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            hitter.ponderhit();
        });
        let started = Instant::now();
        let summary = search(
            &mate,
            &Limits {
                ponder: true,
                ..nodes(2000)
            },
            None,
            &params(8),
            &UniformNetwork,
            None,
            &stop,
            &mut |_| {},
        );
        assert!(started.elapsed() >= Duration::from_millis(100));
        assert_eq!(summary.stop_reason, StopReason::ProvenWin);
        assert_eq!(mate.uci(summary.best.unwrap()), "d1d8");
    }
}
