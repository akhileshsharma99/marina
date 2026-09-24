//! `marina verify`: run a network on the golden positions (`vectors.npz`, shipped with the
//! weights) and compare with the golden outputs. Every backend must pass this before it is
//! used.
//!
//! Compared per position: policy logits over the legal actions, the priors (softmax over
//! legal actions), and the WDL probabilities. The tolerances default to what fp32 on a
//! different summation order can be held to; fp16 backends pass looser ones.

use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use shakmaty::CastlingMode;
use thiserror::Error;

use crate::encoding::{ACTION_SIZE, EncodedBoard, LegalActions, encode_board};
use crate::net::{self, CpuNetwork, Network};
use crate::options::{Backend, Precision};
use crate::position::Game;

#[derive(Debug, Error)]
pub enum VerifyError {
    #[error("cannot read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("vectors.npz: {0}")]
    Npz(String),
    #[error(transparent)]
    Load(#[from] net::LoadError),
    #[error(transparent)]
    Network(#[from] net::NetworkError),
    #[error("position {index} {fen:?}: {reason}")]
    Position {
        index: usize,
        fen: String,
        reason: String,
    },
}

/// Raw policy logits `[n, ACTION_SIZE]` for a batch, when a backend can expose them. A
/// device error is reported like one from `Network::evaluate`.
pub type RawLogits<'a> = dyn Fn(&[&EncodedBoard]) -> Result<Vec<f32>, net::NetworkError> + 'a;

/// Largest inflated `.npy` member accepted from `vectors.npz`: the declared size sizes the
/// read buffer before a byte is inflated, so a corrupt or hostile header must not be able
/// to ask for arbitrary memory.
const MAX_NPY_BYTES: u64 = 1 << 30;

/// Largest absolute differences allowed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tolerances {
    pub logits: f32,
    pub priors: f32,
    pub wdl: f32,
}

impl Tolerances {
    /// fp32 backends: differences come from summation order only.
    pub const FP32: Tolerances = Tolerances {
        logits: 2e-3,
        priors: 2e-4,
        wdl: 2e-4,
    };
    /// fp16 activations with fp32 accumulation, and the CUDA backend's tanh GELU on
    /// networks trained with the erf form (`net::cuda`; measured worth +19 Elo for the
    /// speed it buys). Precision alone is a few 1e-3; the activation adds up to 8.0e-3 /
    /// 1.1e-2 (priors / wdl) on `small` and 5.1e-3 / 6.9e-3 on `nano`, more on wider
    /// networks trained with the erf form.
    pub const FP16: Tolerances = Tolerances {
        logits: 1e-1,
        priors: 2e-2,
        wdl: 2e-2,
    };
    /// e4m3 linear layers (`net::cuda`, `Precision=fp8` or `auto` on Ada and newer): the
    /// three-bit mantissa moves the priors by 6.4e-2 / 6.7e-2 on `small`, 2.2e-1 / 1.1e-1 on
    /// `nano`, and the best move on 1 to 3% of the golden positions, inherent to the format
    /// (a torch simulation gives the same), and measured worth +48 ± 14 Elo on `small` for
    /// the 40% more nodes. These bounds catch a broken path (which produces mismatches on
    /// most positions), not drift.
    pub const FP8: Tolerances = Tolerances {
        logits: 1e-1,
        priors: 3e-1,
        wdl: 2e-1,
    };
}

/// Worst differences seen, and where.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Report {
    pub positions: usize,
    pub max_logit_diff: f32,
    pub max_prior_diff: f32,
    pub max_wdl_diff: f32,
    pub worst_position: Option<(usize, String)>,
    /// Positions where the most likely legal move differs from the reference's.
    pub argmax_mismatches: usize,
}

impl Report {
    pub fn passes(&self, tol: &Tolerances) -> bool {
        self.max_logit_diff <= tol.logits
            && self.max_prior_diff <= tol.priors
            && self.max_wdl_diff <= tol.wdl
    }
}

/// The golden set: FENs and reference outputs.
pub struct Golden {
    pub fens: Vec<String>,
    /// `[n, ACTION_SIZE]`, `-inf` on illegal actions.
    pub policy_logits: Vec<f32>,
    /// `[n, ACTION_SIZE]`, zero on illegal actions.
    pub priors: Vec<f32>,
    /// `[n, 3]`.
    pub wdl: Vec<f32>,
}

impl Golden {
    pub fn load(path: &Path) -> Result<Self, VerifyError> {
        let bytes = std::fs::read(path).map_err(|source| VerifyError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let mut archive = zip::ZipArchive::new(Cursor::new(bytes))
            .map_err(|e| VerifyError::Npz(e.to_string()))?;
        let fens: Vec<String> = read(&mut archive, "fen")?;
        let n = fens.len();
        let policy_logits: Vec<f32> = read(&mut archive, "policy_logits")?;
        let priors: Vec<f32> = read(&mut archive, "priors")?;
        let wdl: Vec<f32> = read(&mut archive, "wdl")?;
        if policy_logits.len() != n * ACTION_SIZE
            || priors.len() != n * ACTION_SIZE
            || wdl.len() != n * 3
        {
            return Err(VerifyError::Npz(
                "array shapes do not match the FEN count".into(),
            ));
        }
        Ok(Self {
            fens,
            policy_logits,
            priors,
            wdl,
        })
    }

    pub fn len(&self) -> usize {
        self.fens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.fens.is_empty()
    }
}

/// Inflate one `.npy` member fully, then parse it: element-at-a-time reads through the
/// decompressor are two orders of magnitude slower.
fn read<T: npyz::Deserialize>(
    archive: &mut zip::ZipArchive<Cursor<Vec<u8>>>,
    name: &str,
) -> Result<Vec<T>, VerifyError> {
    use std::io::Read;
    let mut member = archive
        .by_name(&format!("{name}.npy"))
        .map_err(|e| VerifyError::Npz(format!("{name}: {e}")))?;
    if member.size() > MAX_NPY_BYTES {
        return Err(VerifyError::Npz(format!(
            "{name}: member declares {} bytes; at most {MAX_NPY_BYTES} are read",
            member.size()
        )));
    }
    let mut raw = Vec::with_capacity(member.size() as usize);
    member
        .read_to_end(&mut raw)
        .map_err(|e| VerifyError::Npz(format!("{name}: {e}")))?;
    let npy = npyz::NpyFile::new(&raw[..]).map_err(|e| VerifyError::Npz(format!("{name}: {e}")))?;
    npy.into_vec::<T>()
        .map_err(|e| VerifyError::Npz(format!("{name}: {e}")))
}

/// Compare `network` against `golden` on its first `limit` positions (all when `None`).
/// `raw_logits`, when the backend exposes them, sharpens the logit comparison; otherwise
/// only priors and WDL are compared.
pub fn compare(
    network: &dyn Network,
    raw_logits: Option<&RawLogits<'_>>,
    golden: &Golden,
    limit: Option<usize>,
    batch: usize,
) -> Result<Report, VerifyError> {
    let count = limit.map_or(golden.len(), |l| l.min(golden.len()));
    let mut report = Report {
        positions: count,
        ..Report::default()
    };
    let mut worst = 0.0f32;
    for start in (0..count).step_by(batch.max(1)) {
        let end = (start + batch.max(1)).min(count);
        let mut games = Vec::with_capacity(end - start);
        for index in start..end {
            let fen = &golden.fens[index];
            let game = Game::from_uci(Some(fen), &[], CastlingMode::Standard).map_err(|e| {
                VerifyError::Position {
                    index,
                    fen: fen.clone(),
                    reason: e.to_string(),
                }
            })?;
            games.push(game);
        }
        let boards: Vec<EncodedBoard> = games.iter().map(encode_board).collect();
        let legals: Vec<LegalActions> = games.iter().map(LegalActions::of).collect();
        let leaves: Vec<(&EncodedBoard, &LegalActions)> = boards.iter().zip(&legals).collect();
        let evaluations = network.evaluate(&leaves)?;
        let logits = raw_logits
            .map(|f| f(&boards.iter().collect::<Vec<_>>()))
            .transpose()?;

        for (offset, (evaluation, legal)) in evaluations.iter().zip(&legals).enumerate() {
            let index = start + offset;
            let gold_priors = &golden.priors[index * ACTION_SIZE..(index + 1) * ACTION_SIZE];
            let gold_logits = &golden.policy_logits[index * ACTION_SIZE..(index + 1) * ACTION_SIZE];
            let mut here = 0.0f32;
            let mut our_best = (f32::NEG_INFINITY, 0usize);
            let mut gold_best = (f32::NEG_INFINITY, 0usize);
            for (i, &(_, action)) in legal.moves.iter().enumerate() {
                let diff = (evaluation.priors[i] - gold_priors[action]).abs();
                report.max_prior_diff = report.max_prior_diff.max(diff);
                here = here.max(diff);
                if evaluation.priors[i] > our_best.0 {
                    our_best = (evaluation.priors[i], action);
                }
                if gold_priors[action] > gold_best.0 {
                    gold_best = (gold_priors[action], action);
                }
                if let Some(ours) = &logits {
                    let diff = (ours[offset * ACTION_SIZE + action] - gold_logits[action]).abs();
                    report.max_logit_diff = report.max_logit_diff.max(diff);
                }
            }
            if our_best.1 != gold_best.1 {
                report.argmax_mismatches += 1;
            }
            for k in 0..3 {
                let diff = (evaluation.wdl[k] - golden.wdl[index * 3 + k]).abs();
                report.max_wdl_diff = report.max_wdl_diff.max(diff);
                here = here.max(diff);
            }
            if here > worst {
                worst = here;
                report.worst_position = Some((index, golden.fens[index].clone()));
            }
        }
    }
    Ok(report)
}

fn threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// Load the network `weights` names (an embedded name or a directory, as `WeightsFile`)
/// on `backend` and verify it against `vectors`.
pub fn run(
    weights: &str,
    vectors: &Path,
    backend: Backend,
    precision: Precision,
    limit: Option<usize>,
    batch: usize,
    repeat: usize,
) -> Result<(Report, Tolerances, String, std::time::Duration), VerifyError> {
    let golden = Golden::load(vectors)?;
    let (arch, parsed) = net::parse(weights)?;
    let (network, raw, tol): (
        Box<dyn Network>,
        Option<Box<RawLogits<'static>>>,
        Tolerances,
    ) = match backend {
        Backend::Cpu => {
            let cpu = Arc::new(CpuNetwork::new(arch, parsed, threads()));
            let for_logits = Arc::clone(&cpu);
            let raw = move |boards: &[&EncodedBoard]| Ok(for_logits.logits(boards).0);
            (Box::new(cpu), Some(Box::new(raw)), Tolerances::FP32)
        }
        #[cfg(feature = "cuda")]
        Backend::Cuda | Backend::Auto => {
            let network =
                net::cuda::CudaNetwork::new(arch, &parsed, 0, batch, net::cuda_fp8(precision))
                    .map_err(net::LoadError::Cuda)?;
            let tol = if network.is_fp8() {
                Tolerances::FP8
            } else {
                Tolerances::FP16
            };
            (Box::new(network), None, tol)
        }
        #[cfg(all(not(feature = "cuda"), not(target_os = "macos")))]
        Backend::Auto => {
            let _ = precision;
            let cpu = Arc::new(CpuNetwork::new(arch, parsed, threads()));
            let for_logits = Arc::clone(&cpu);
            let raw = move |boards: &[&EncodedBoard]| Ok(for_logits.logits(boards).0);
            (Box::new(cpu), Some(Box::new(raw)), Tolerances::FP32)
        }
        #[cfg(not(feature = "cuda"))]
        Backend::Cuda => return Err(net::LoadError::Unavailable(Backend::Cuda).into()),
        #[cfg(target_os = "macos")]
        Backend::Metal => {
            let fp16 = net::metal_fp16(precision);
            let metal = Arc::new(
                net::metal::MetalNetwork::new(arch, &parsed, batch, fp16)
                    .map_err(net::LoadError::Metal)?,
            );
            let for_logits = Arc::clone(&metal);
            let raw = move |boards: &[&EncodedBoard]| {
                for_logits
                    .logits(boards)
                    .map(|(policy, _)| policy)
                    .map_err(|e| net::NetworkError(e.to_string()))
            };
            let tol = if fp16 {
                Tolerances::FP16
            } else {
                Tolerances::FP32
            };
            (Box::new(metal), Some(Box::new(raw)), tol)
        }
        #[cfg(all(not(feature = "cuda"), target_os = "macos"))]
        Backend::Auto => {
            let fp16 = net::metal_fp16(precision);
            let metal = Arc::new(
                net::metal::MetalNetwork::new(arch, &parsed, batch, fp16)
                    .map_err(net::LoadError::Metal)?,
            );
            let for_logits = Arc::clone(&metal);
            let raw = move |boards: &[&EncodedBoard]| {
                for_logits
                    .logits(boards)
                    .map(|(policy, _)| policy)
                    .map_err(|e| net::NetworkError(e.to_string()))
            };
            let tol = if fp16 {
                Tolerances::FP16
            } else {
                Tolerances::FP32
            };
            (Box::new(metal), Some(Box::new(raw)), tol)
        }
        #[cfg(not(target_os = "macos"))]
        Backend::Metal => return Err(net::LoadError::Unavailable(Backend::Metal).into()),
    };
    let describe = network.describe();
    // One untimed pass warms up JIT compilation and library heuristics; then `repeat`
    // timed passes, so the rate reflects steady state. The report is from the last pass.
    let mut report = compare(network.as_ref(), raw.as_deref(), &golden, limit, batch)?;
    let started = std::time::Instant::now();
    for _ in 0..repeat.max(1) {
        report = compare(network.as_ref(), raw.as_deref(), &golden, limit, batch)?;
    }
    let elapsed = started.elapsed() / repeat.max(1) as u32;
    Ok((report, tol, describe, elapsed))
}
