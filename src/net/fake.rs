//! Fake networks for building and benchmarking the search without a real one.

use super::{Evaluation, Leaf, Network, NetworkError};
use crate::encoding::ACTION_SIZE;

/// Equal priors over the legal moves, value zero. The simplest possible evaluator; trees
/// under it are flat, which makes it good for measuring raw tree throughput and useless
/// for measuring search quality.
#[derive(Debug, Default, Clone, Copy)]
pub struct UniformNetwork;

impl Network for UniformNetwork {
    fn evaluate(&self, batch: &[Leaf<'_>]) -> Result<Vec<Evaluation>, NetworkError> {
        Ok(batch
            .iter()
            .map(|(_, legal)| {
                let n = legal.len().max(1);
                Evaluation {
                    priors: vec![1.0 / n as f32; legal.len()],
                    wdl: [1.0 / 3.0; 3],
                }
            })
            .collect())
    }

    fn describe(&self) -> String {
        "uniform fake network (equal priors, value 0)".to_string()
    }
}

/// Deterministic pseudo-random priors and values derived from a hash of the position and
/// each move. Priors are a softmax over hashed logits with a temperature that makes a few
/// moves dominate, like a real policy; the value is in roughly `[-0.6, 0.6]`. The same
/// position always gets the same evaluation, so searches are reproducible run to run.
#[derive(Debug, Clone, Copy)]
pub struct HashNetwork {
    /// Mixes into every hash so different seeds give different "networks".
    pub seed: u64,
    /// Softmax temperature over the hashed logits; lower = sharper policy.
    pub temperature: f32,
}

impl Default for HashNetwork {
    fn default() -> Self {
        Self {
            seed: 0x9E37_79B9_7F4A_7C15,
            temperature: 0.5,
        }
    }
}

#[inline]
fn mix(mut x: u64) -> u64 {
    // splitmix64 finaliser.
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

#[inline]
fn unit(hash: u64) -> f32 {
    // Top 24 bits -> [0, 1).
    (hash >> 40) as f32 / (1u64 << 24) as f32
}

impl HashNetwork {
    fn position_hash(&self, board: &crate::encoding::EncodedBoard) -> u64 {
        let mut h = self.seed;
        for (index, &token) in board.pieces.iter().enumerate() {
            if token != 0 {
                h = mix(h ^ ((index as u64) << 8 | u64::from(token)));
            }
        }
        h = mix(h ^ (u64::from(board.castling) << 4 | u64::from(board.en_passant)));
        h
    }
}

impl Network for HashNetwork {
    fn evaluate(&self, batch: &[Leaf<'_>]) -> Result<Vec<Evaluation>, NetworkError> {
        Ok(batch
            .iter()
            .map(|(board, legal)| {
                let base = self.position_hash(board);
                let mut logits: Vec<f32> = legal
                    .moves
                    .iter()
                    .map(|&(_, action)| {
                        debug_assert!(action < ACTION_SIZE);
                        unit(mix(base ^ (action as u64 + 1))) / self.temperature
                    })
                    .collect();
                let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for logit in &mut logits {
                    *logit = (*logit - max).exp();
                    sum += *logit;
                }
                for logit in &mut logits {
                    *logit /= sum.max(f32::MIN_POSITIVE);
                }
                let value = (unit(mix(base ^ 0xA5A5)) - 0.5) * 1.2;
                let draw = 0.3 + 0.4 * unit(mix(base ^ 0x5A5A)) * (1.0 - value.abs());
                let win = ((1.0 - draw) + value) / 2.0;
                let loss = 1.0 - draw - win;
                Evaluation {
                    priors: logits,
                    wdl: [win.max(0.0), draw.max(0.0), loss.max(0.0)],
                }
            })
            .collect())
    }

    fn describe(&self) -> String {
        format!(
            "hash fake network (seed {:#x}, temperature {})",
            self.seed, self.temperature
        )
    }
}

/// Wraps a network so every call takes at least `per_batch + per_leaf × n`, standing in
/// for a GPU in tree benchmarks (a real batch has a fixed launch cost plus a per-row cost).
/// Waits by spinning, since the intervals are milliseconds and a sleep would overshoot.
pub struct LatencyNetwork<N> {
    pub inner: N,
    pub per_batch: std::time::Duration,
    pub per_leaf: std::time::Duration,
}

impl<N: Network> Network for LatencyNetwork<N> {
    fn evaluate(&self, batch: &[Leaf<'_>]) -> Result<Vec<Evaluation>, NetworkError> {
        let started = std::time::Instant::now();
        let out = self.inner.evaluate(batch);
        let due = self.per_batch + self.per_leaf * batch.len() as u32;
        while started.elapsed() < due {
            std::hint::spin_loop();
        }
        out
    }

    fn describe(&self) -> String {
        format!(
            "{} + simulated latency {:?}/batch {:?}/leaf",
            self.inner.describe(),
            self.per_batch,
            self.per_leaf
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::{LegalActions, encode_board};
    use crate::position::Game;
    use shakmaty::CastlingMode;

    fn leaf(fen: Option<&str>) -> (crate::encoding::EncodedBoard, LegalActions) {
        let game = Game::from_uci(fen, &[], CastlingMode::Standard).unwrap();
        (encode_board(&game), LegalActions::of(&game))
    }

    #[test]
    fn uniform_covers_legal_moves_and_sums_to_one() {
        let (board, legal) = leaf(None);
        let out = UniformNetwork.evaluate(&[(&board, &legal)]).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].priors.len(), 20);
        assert!((out[0].priors.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert_eq!(out[0].value(), 0.0);
    }

    #[test]
    fn hash_network_is_deterministic_sharp_and_bounded() {
        let (board, legal) = leaf(None);
        let net = HashNetwork::default();
        let a = net.evaluate(&[(&board, &legal)]).unwrap();
        let b = net.evaluate(&[(&board, &legal)]).unwrap();
        assert_eq!(a, b);
        let priors = &a[0].priors;
        assert_eq!(priors.len(), 20);
        assert!((priors.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        let max = priors.iter().cloned().fold(0.0, f32::max);
        assert!(max > 0.1, "policy should favour some moves: max {max}");
        let wdl = a[0].wdl;
        assert!((wdl.iter().sum::<f32>() - 1.0).abs() < 1e-5, "{wdl:?}");
        assert!(wdl.iter().all(|&p| (0.0..=1.0).contains(&p)));
        assert!(a[0].value().abs() <= 0.6 + 1e-6);
        // A different position gets a different evaluation; a different seed too.
        let (other, other_legal) = leaf(Some(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        ));
        assert_ne!(
            net.evaluate(&[(&other, &other_legal)]).unwrap()[0].wdl,
            a[0].wdl
        );
        let seeded = HashNetwork {
            seed: 7,
            ..HashNetwork::default()
        };
        assert_ne!(
            seeded.evaluate(&[(&board, &legal)]).unwrap()[0].priors,
            a[0].priors
        );
    }

    #[test]
    fn batches_preserve_order() {
        let (a, la) = leaf(None);
        let (b, lb) = leaf(Some("8/8/8/3k4/8/8/4Q3/4K3 w - - 0 1"));
        let out = HashNetwork::default()
            .evaluate(&[(&a, &la), (&b, &lb), (&a, &la)])
            .unwrap();
        assert_eq!(out[0], out[2]);
        assert_eq!(out[1].priors.len(), lb.len());
    }
}
