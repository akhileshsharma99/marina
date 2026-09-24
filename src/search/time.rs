//! Clock allocation and early stopping.
//!
//! A move gets a **soft** budget, its share of the time left in the game, plus whatever
//! earlier moves saved, and a **hard** cap. The search stops at the soft budget once the
//! most-visited root move is also the best-valued one, at the hard cap regardless, and
//! earlier than either as soon as the runner-up can no longer catch the leader in the
//! visits the remaining time can buy (smart pruning). Time saved goes into a per-game bank
//! that the next move spends.

use std::time::{Duration, Instant};

use crate::search::StopReason;

/// Median remaining game length in moves at a given ply, from a log-logistic model of
/// game lengths (Clark & El-Taha's median residual life): `midpoint` is the median game
/// length, `steepness` how sharply the distribution falls around it.
pub fn moves_left(ply: u32, midpoint: f32, steepness: f32) -> f32 {
    let move_number = ply as f32 / 2.0;
    let ratio = move_number / midpoint;
    (midpoint * (1.0 + 2.0 * ratio.powf(steepness)).powf(1.0 / steepness) - move_number).max(1.0)
}

/// Move at which half the game is expected to be over; Lc0's fit to real games.
pub const MIDPOINT: f32 = 51.5;
/// How sharply the expectation drops past the midpoint; Lc0's fit to real games.
pub const STEEPNESS: f32 = 7.0;

/// The side to move's clock as `go` reported it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Clock {
    pub remaining: Duration,
    pub increment: Duration,
    pub movestogo: Option<u32>,
}

/// The time plan for one move.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    /// Spend this much unless the root is undecided.
    pub soft: Duration,
    /// Never spend more.
    pub hard: Duration,
    /// `soft` before the bank was added, for settling the bank afterwards.
    pub base: Duration,
}

/// What a reused tree brings to a move, for budgeting in nodes rather than time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reuse {
    /// Root visits the search starts with.
    pub reused_nodes: u64,
    /// Nodes per second this game has been getting.
    pub nps: f64,
    /// Fraction of a search's nodes that survive into the next move, on average.
    pub factor: f32,
}

/// Never budget fewer new nodes than this share of a plain move's worth, however much of
/// the tree survived: the position still has to be checked at the new root.
pub const MIN_NEW_SHARE: f32 = 0.2;

impl Budget {
    /// Share of the game's time for this move. `bank` is time saved by earlier moves.
    /// With `reuse`, the share is budgeted in nodes: every move aims at the same total
    /// tree, so a move that inherits most of its tree spends less and a surprise spends
    /// more (the idea behind Lc0's `smooth` time manager).
    pub fn adaptive(
        clock: Clock,
        ply: u32,
        overhead: Duration,
        bank: Duration,
        reuse: Option<Reuse>,
    ) -> Self {
        let moves = match clock.movestogo {
            Some(n) if n > 0 => n as f32,
            _ => moves_left(ply, MIDPOINT, STEEPNESS),
        };
        let pool = clock.remaining.as_secs_f32() + clock.increment.as_secs_f32() * (moves - 1.0)
            - overhead.as_secs_f32() * (moves + 2.0);
        let share = Duration::from_secs_f32((pool / moves).max(0.001));
        let base = match reuse {
            Some(reuse) if reuse.nps > 0.0 => {
                let avg_new = share.as_secs_f64() * reuse.nps;
                let target_total = avg_new / f64::from(1.0 - reuse.factor.clamp(0.0, MAX_REUSE));
                let target_new = (target_total - reuse.reused_nodes as f64)
                    .max(avg_new * f64::from(MIN_NEW_SHARE));
                Duration::from_secs_f64((target_new / reuse.nps).max(0.001))
            }
            _ => share,
        };
        let hard = (clock.remaining / 2)
            .saturating_sub(overhead)
            .max(Duration::from_millis(1))
            .min(share * 2 + bank);
        let soft = (base + bank).min(hard);
        Self { soft, hard, base }
    }

    /// `go movetime`: exactly that, less the overhead.
    pub fn exact(movetime: Duration, overhead: Duration) -> Self {
        let budget = movetime
            .saturating_sub(overhead)
            .max(Duration::from_millis(1));
        Self {
            soft: budget,
            hard: budget,
            base: budget,
        }
    }
}

/// What the root looks like at a batch boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RootStats {
    pub best_visits: u64,
    pub second_visits: u64,
    /// The most-visited move is also the highest-valued one.
    pub decided: bool,
}

/// Smart pruning does not stop before this many batches have been applied: the nps
/// estimate needs them, and the first batches carry the network's warm-up.
pub const PRUNE_MIN_BATCHES: u64 = 3;
/// Lc0's safety factor on the playouts the remaining time is assumed to buy.
pub const PRUNE_FACTOR: f64 = 1.33;

/// Applies a [`Budget`] to a running search.
#[derive(Debug)]
pub struct Pacer {
    budget: Budget,
    started: Instant,
    /// When the first batch came back; nps is measured from here.
    first_batch: Option<Instant>,
    batches: u64,
    /// Whether smart pruning and the soft stop apply (not for `go movetime`).
    adaptive: bool,
}

impl Pacer {
    pub fn new(budget: Budget, started: Instant, adaptive: bool) -> Self {
        Self {
            budget,
            started,
            first_batch: None,
            batches: 0,
            adaptive,
        }
    }

    /// The hard deadline, for the loop's cheap per-iteration check.
    pub fn hard_deadline(&self) -> Instant {
        self.started + self.budget.hard
    }

    /// The budget this move is played under, when it is a share of the clock to settle
    /// afterwards (`None` for `go movetime`).
    pub fn adaptive_budget(&self) -> Option<Budget> {
        self.adaptive.then_some(self.budget)
    }

    /// Time since the clock was armed.
    pub fn elapsed(&self, now: Instant) -> Duration {
        now.duration_since(self.started)
    }

    /// Nodes per second since the first batch, once measurable.
    pub fn nps(&self, nodes: u64, now: Instant) -> Option<f64> {
        let since = now.duration_since(self.first_batch?).as_secs_f64();
        (since > 0.0).then(|| nodes as f64 / since)
    }

    /// Called after every batch is applied; `Some` when the search should stop.
    pub fn after_batch(&mut self, root: RootStats, nodes: u64, now: Instant) -> Option<StopReason> {
        self.batches += 1;
        if self.first_batch.is_none() {
            self.first_batch = Some(now);
        }
        let elapsed = now.duration_since(self.started);
        if elapsed >= self.budget.hard {
            return Some(StopReason::TimeLimit);
        }
        if !self.adaptive {
            return None;
        }
        if elapsed >= self.budget.soft && root.decided {
            return Some(StopReason::TimeLimit);
        }
        // Smart pruning: the runner-up cannot catch the leader before the deadline.
        if self.batches >= PRUNE_MIN_BATCHES {
            let deadline = if root.decided {
                self.budget.soft
            } else {
                self.budget.hard
            };
            if let Some(nps) = self.nps(nodes, now) {
                let remaining = deadline.saturating_sub(elapsed).as_secs_f64();
                let playouts = remaining * nps / PRUNE_FACTOR;
                let lead = root.best_visits.saturating_sub(root.second_visits) as f64;
                if lead > playouts {
                    return Some(StopReason::SmartPruning);
                }
            }
        }
        None
    }
}

/// Where the per-game estimate of the reused share of the tree starts (Lc0's default).
pub const INITIAL_REUSE: f32 = 0.5;
/// The most the reuse estimate may claim, so a lucky streak cannot starve a move.
pub const MAX_REUSE: f32 = 0.7;
/// Weight of the latest move in the per-game estimates (half-life of a few moves).
pub const ESTIMATE_RATE: f32 = 0.25;
/// Moves shorter than this say nothing reliable about nps.
pub const MIN_NPS_SAMPLE: Duration = Duration::from_millis(50);

/// Time bookkeeping that lives across the moves of one game.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeState {
    /// Time earlier moves did not use; the next move spends it.
    pub bank: Duration,
    /// Nodes per second, smoothed over the game's moves; `None` until measured.
    pub nps: Option<f64>,
    /// Fraction of a search's nodes that survived into the next move, smoothed.
    pub reuse: f32,
    /// Total root visits (reused + new) when the last search ended.
    pub last_total: u64,
}

impl Default for TimeState {
    fn default() -> Self {
        Self {
            bank: Duration::ZERO,
            nps: None,
            reuse: INITIAL_REUSE,
            last_total: 0,
        }
    }
}

impl TimeState {
    /// What the next budget should assume about a tree carrying `reused_nodes`, once nps
    /// has been measured.
    pub fn reuse_for(&self, reused_nodes: u64) -> Option<Reuse> {
        Some(Reuse {
            reused_nodes,
            nps: self.nps?,
            factor: self.reuse,
        })
    }

    /// Settle a finished move: what was budgeted but not used goes into the bank, what
    /// was used beyond the budget comes out of it; nps is updated from the `new`
    /// simulations made in the `used` time on the clock, the reuse fraction from the
    /// `reused` root visits the move inherited from the previous search, and `total` root
    /// visits (inherited, pondered and new) are what the next move's inheritance is
    /// measured against.
    pub fn settle(&mut self, budget: Budget, used: Duration, new: u64, reused: u64, total: u64) {
        // Saved time is banked, but only up to a couple of moves' worth: the clock already
        // carries the rest forward, and a run of instant moves must not fund a huge think.
        let available = self.bank + budget.base;
        self.bank = available.saturating_sub(used).min(budget.base * 2);
        if used >= MIN_NPS_SAMPLE && new > 0 {
            let observed = new as f64 / used.as_secs_f64();
            self.nps = Some(match self.nps {
                Some(nps) => nps + f64::from(ESTIMATE_RATE) * (observed - nps),
                None => observed,
            });
        }
        if self.last_total > 0 {
            let observed = (reused as f32 / self.last_total as f32).min(1.0);
            self.reuse += ESTIMATE_RATE * (observed - self.reuse);
        }
        self.last_total = total;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moves_left_follows_the_game() {
        let start = moves_left(0, MIDPOINT, STEEPNESS);
        assert!((start - MIDPOINT).abs() < 0.01, "{start}");
        let early = moves_left(40, MIDPOINT, STEEPNESS); // move 20: ~31.5
        assert!(early > 28.0 && early < 35.0, "{early}");
        let mid = moves_left(80, MIDPOINT, STEEPNESS); // move 40: ~13.7
        assert!(mid > 10.0 && mid < 18.0, "{mid}");
        // The distribution's heavy tail: a game that has lasted 100 moves is expected to
        // go on for a while yet, more than one at move 60.
        let late = moves_left(200, MIDPOINT, STEEPNESS);
        assert!(
            late > moves_left(120, MIDPOINT, STEEPNESS) && late < mid,
            "{late}"
        );
        assert!(moves_left(2000, MIDPOINT, STEEPNESS) >= 1.0);
    }

    fn hyperbullet(remaining_ms: u64) -> Clock {
        Clock {
            remaining: Duration::from_millis(remaining_ms),
            increment: Duration::from_millis(100),
            movestogo: None,
        }
    }

    #[test]
    fn adaptive_budget_is_a_share_of_the_pool() {
        let overhead = Duration::from_millis(20);
        let b = Budget::adaptive(hyperbullet(10_000), 0, overhead, Duration::ZERO, None);
        // pool ≈ 10 + 0.1·50.5 − 0.02·53.5 ≈ 13.98 s over 51.5 moves ≈ 271 ms.
        assert!(
            b.base.as_millis() > 250 && b.base.as_millis() < 300,
            "{b:?}"
        );
        assert_eq!(b.soft, b.base);
        assert_eq!(b.hard, b.base * 2);
        // Almost out of time: the hard cap is half the clock less the overhead.
        let b = Budget::adaptive(hyperbullet(100), 100, overhead, Duration::ZERO, None);
        assert_eq!(b.hard, Duration::from_millis(30));
        assert!(b.soft <= b.hard);
    }

    #[test]
    fn bank_is_spent_and_capped() {
        let overhead = Duration::from_millis(20);
        let plain = Budget::adaptive(hyperbullet(10_000), 20, overhead, Duration::ZERO, None);
        let rich = Budget::adaptive(
            hyperbullet(10_000),
            20,
            overhead,
            Duration::from_millis(200),
            None,
        );
        assert_eq!(rich.base, plain.base);
        assert_eq!(rich.soft, plain.soft + Duration::from_millis(200));
        let flooded = Budget::adaptive(
            hyperbullet(1_000),
            20,
            overhead,
            Duration::from_secs(5),
            None,
        );
        assert!(flooded.soft <= flooded.hard);
        assert_eq!(flooded.hard, Duration::from_millis(480));
    }

    #[test]
    fn settle_banks_savings_and_repays_overruns() {
        let budget = Budget {
            soft: Duration::from_millis(300),
            hard: Duration::from_millis(600),
            base: Duration::from_millis(300),
        };
        let mut state = TimeState::default();
        state.settle(budget, Duration::from_millis(100), 1000, 0, 1000);
        assert_eq!(state.bank, Duration::from_millis(200));
        state.settle(budget, Duration::from_millis(450), 1000, 0, 1000);
        assert_eq!(state.bank, Duration::from_millis(50));
        state.settle(budget, Duration::from_millis(900), 1000, 0, 1000);
        assert_eq!(state.bank, Duration::ZERO);
    }

    #[test]
    fn estimates_track_nps_and_reuse() {
        let budget = Budget {
            soft: Duration::from_millis(300),
            hard: Duration::from_millis(600),
            base: Duration::from_millis(300),
        };
        let mut state = TimeState::default();
        assert!(state.reuse_for(100).is_none(), "no nps yet");
        state.settle(budget, Duration::from_millis(200), 2000, 0, 2000); // 10K nps
        assert_eq!(state.nps, Some(10_000.0));
        assert_eq!(state.last_total, 2000);
        // Next move reused half the tree: the estimate moves a quarter of the way there.
        state.settle(budget, Duration::from_millis(200), 2000, 1000, 3000);
        assert!((state.reuse - (0.5 + 0.25 * (0.5 - 0.5))).abs() < 1e-6);
        state.settle(budget, Duration::from_millis(200), 2000, 0, 2000); // nothing reused
        assert!(
            (state.reuse - (0.5 - 0.125)).abs() < 1e-6,
            "{}",
            state.reuse
        );
        // An instant move says nothing about nps.
        state.settle(budget, Duration::from_millis(1), 5, 0, 5);
        assert_eq!(state.nps, Some(10_000.0));
    }

    #[test]
    fn reuse_budgets_in_nodes() {
        let overhead = Duration::from_millis(20);
        let plain = Budget::adaptive(hyperbullet(10_000), 20, overhead, Duration::ZERO, None);
        let nps = 10_000.0; // plain share ≈ 285 ms ≈ 2850 new nodes
        // Half the tree survives on average and this move inherited exactly that much:
        // target total = 2·avg_new, minus the reused avg_new → the plain budget.
        let avg_new = plain.base.as_secs_f64() * nps;
        let typical = Reuse {
            reused_nodes: avg_new as u64,
            nps,
            factor: 0.5,
        };
        let same = Budget::adaptive(
            hyperbullet(10_000),
            20,
            overhead,
            Duration::ZERO,
            Some(typical),
        );
        assert!(
            (same.base.as_secs_f64() - plain.base.as_secs_f64()).abs() < 0.002,
            "{same:?}"
        );
        // A surprise (nothing reused) gets twice the share; an inherited full tree gets the floor.
        let surprise = Budget::adaptive(
            hyperbullet(10_000),
            20,
            overhead,
            Duration::ZERO,
            Some(Reuse {
                reused_nodes: 0,
                ..typical
            }),
        );
        assert!((surprise.base.as_secs_f64() - 2.0 * plain.base.as_secs_f64()).abs() < 0.002);
        assert_eq!(surprise.soft, surprise.hard.min(surprise.base));
        let inherited = Budget::adaptive(
            hyperbullet(10_000),
            20,
            overhead,
            Duration::ZERO,
            Some(Reuse {
                reused_nodes: 10 * avg_new as u64,
                ..typical
            }),
        );
        let floor = plain.base.as_secs_f64() * f64::from(MIN_NEW_SHARE);
        assert!(
            (inherited.base.as_secs_f64() - floor).abs() < 0.002,
            "{inherited:?}"
        );
        // The hard cap follows the time share, not the node budget.
        assert_eq!(surprise.hard, plain.hard);
    }

    #[test]
    fn pacer_stops_soft_when_decided_and_prunes_a_clear_lead() {
        let budget = Budget {
            soft: Duration::from_millis(300),
            hard: Duration::from_millis(600),
            base: Duration::from_millis(300),
        };
        let t0 = Instant::now();
        let mut pacer = Pacer::new(budget, t0, true);
        let undecided = RootStats {
            best_visits: 50,
            second_visits: 40,
            decided: false,
        };
        let decided = RootStats {
            decided: true,
            ..undecided
        };
        // Before soft: keep going.
        assert_eq!(
            pacer.after_batch(decided, 100, t0 + Duration::from_millis(10)),
            None
        );
        assert_eq!(
            pacer.after_batch(decided, 200, t0 + Duration::from_millis(20)),
            None
        );
        // At soft, undecided: continue; decided: stop.
        assert_eq!(
            pacer.after_batch(undecided, 3000, t0 + Duration::from_millis(310)),
            None
        );
        assert_eq!(
            pacer.after_batch(decided, 3100, t0 + Duration::from_millis(320)),
            Some(StopReason::TimeLimit)
        );
        // Hard stops regardless.
        assert_eq!(
            pacer.after_batch(undecided, 6000, t0 + Duration::from_millis(600)),
            Some(StopReason::TimeLimit)
        );
        // Smart pruning: 10 nps measured, 200 ms to the soft deadline buys ~1.5 playouts,
        // the leader is 1000 ahead.
        let mut pacer = Pacer::new(budget, t0, true);
        let lead = RootStats {
            best_visits: 1200,
            second_visits: 200,
            decided: true,
        };
        pacer.after_batch(lead, 1, t0 + Duration::from_millis(10));
        pacer.after_batch(lead, 1, t0 + Duration::from_millis(50));
        assert_eq!(
            pacer.after_batch(lead, 1, t0 + Duration::from_millis(100)),
            Some(StopReason::SmartPruning)
        );
        // A non-adaptive pacer (`go movetime`) never stops before the hard cap.
        let mut pacer = Pacer::new(budget, t0, false);
        for i in 1..10 {
            assert_eq!(
                pacer.after_batch(lead, 1, t0 + Duration::from_millis(30 * i)),
                None
            );
        }
    }
}
