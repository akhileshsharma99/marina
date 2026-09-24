//! The network the search evaluates leaves with.
//!
//! Every backend (CPU, CUDA, Metal) implements [`Network`]: encoded positions in, priors over the
//! legal moves and a win/draw/loss estimate out, in batches. [`resolve`] turns a
//! network spec (`<embedded>`, a name from `nets.toml`, or a directory; what `Options::weights_spec` yields) into a loaded
//! backend. Three fakes live here so the search can be built and measured without any real
//! network: [`UniformNetwork`] gives equal priors and a zero value; [`HashNetwork`] gives
//! deterministic pseudo-random priors and values, so trees have the shape a real network
//! produces (a few favoured moves per node) while every run is reproducible; and
//! [`LatencyNetwork`] wraps another to add a GPU-like delay per batch.

pub mod cpu;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod embedded;
pub mod fake;
pub mod gemm;
#[cfg(target_os = "macos")]
pub mod metal;
pub mod model;
pub mod weights;

use std::path::Path;

pub use cpu::CpuNetwork;
pub use fake::{HashNetwork, LatencyNetwork, UniformNetwork};
pub use model::Architecture;
use thiserror::Error;
pub use weights::Weights;

use crate::encoding::{EncodedBoard, LegalActions};
use crate::options::{Backend, Precision};

#[derive(Debug, Error)]
pub enum LoadError {
    #[error(transparent)]
    Model(#[from] model::ModelError),
    #[error(transparent)]
    Weights(#[from] weights::WeightsError),
    #[error(transparent)]
    Embedded(#[from] embedded::EmbeddedError),
    #[cfg(feature = "cuda")]
    #[error(transparent)]
    Cuda(#[from] cuda::CudaError),
    #[cfg(target_os = "macos")]
    #[error(transparent)]
    Metal(#[from] metal::MetalError),
    #[error("backend {0} is not available in this build")]
    Unavailable(Backend),
    #[error("cannot read {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("{path} is not a regular file")]
    NotAFile { path: String },
    #[error("{path} is {size} bytes; at most {max} are read")]
    TooLarge { path: String, size: u64, max: u64 },
}

/// Largest `model.json` read from a directory; a card is a few kilobytes.
const MAX_MODEL_JSON_BYTES: u64 = 16 << 20;
/// Largest `model.safetensors` read from a directory.
const MAX_SAFETENSORS_BYTES: u64 = 4 << 30;

/// Read a user-supplied file whole, refusing anything that is not a regular file (a FIFO
/// or a device such as `/dev/zero` would block or never end) and anything larger than
/// `max` (the file is held in memory before it is parsed). Symlinks are followed, as
/// `fs::metadata` does; the check is on what they point at.
fn read_bounded(path: &Path, max: u64) -> Result<Vec<u8>, LoadError> {
    let io = |source| LoadError::Io {
        path: path.display().to_string(),
        source,
    };
    let meta = std::fs::metadata(path).map_err(io)?;
    if !meta.is_file() {
        return Err(LoadError::NotAFile {
            path: path.display().to_string(),
        });
    }
    if meta.len() > max {
        return Err(LoadError::TooLarge {
            path: path.display().to_string(),
            size: meta.len(),
            max,
        });
    }
    std::fs::read(path).map_err(io)
}

/// How the backends are sized: from the `Threads` and `Batch` options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Device {
    /// Threads the CPU backend's forward pass runs on.
    pub threads: usize,
    /// Largest batch the backend must accept; the CUDA backend allocates for it.
    pub max_batch: usize,
    /// Arithmetic of the linear layers, resolved per backend (see [`Precision`]).
    pub precision: Precision,
}

impl Default for Device {
    fn default() -> Self {
        Self {
            threads: 1,
            max_batch: 128,
            precision: Precision::Auto,
        }
    }
}

/// Resolve what `WeightsFile` names and load it on `backend`: `<embedded>` is the default
/// network compiled into the binary, a name from `nets.toml` is that network (an error
/// saying where to get it when it is not built in), anything else is a directory holding
/// `model.json` + `model.safetensors`.
pub fn resolve(
    spec: &str,
    backend: Backend,
    device: Device,
) -> Result<Box<dyn Network>, LoadError> {
    let (arch, weights) = parse(spec)?;
    build(arch, weights, backend, device)
}

/// Read and parse what `WeightsFile` names (see [`resolve`]) without choosing a backend.
pub fn parse(spec: &str) -> Result<(Architecture, Weights), LoadError> {
    let spec = spec.trim();
    let embedded = if spec == crate::options::EMBEDDED_WEIGHTS {
        Some(embedded::default()?)
    } else if embedded::is_known(spec) {
        Some(embedded::find(spec)?)
    } else {
        None
    };
    match embedded {
        Some(net) => {
            let arch = Architecture::from_json(net.model_json)?;
            let weights = Weights::from_bytes(net.model_safetensors, &arch)?;
            Ok((arch, weights))
        }
        None => {
            let dir = Path::new(spec);
            let card_path = dir.join("model.json");
            let card = read_bounded(&card_path, MAX_MODEL_JSON_BYTES)?;
            // The same failure `fs::read_to_string` reports for a file that is not UTF-8.
            let card = std::str::from_utf8(&card).map_err(|e| LoadError::Io {
                path: card_path.display().to_string(),
                source: std::io::Error::new(std::io::ErrorKind::InvalidData, e),
            })?;
            let arch = Architecture::from_json(card)?;
            let bytes = read_bounded(&dir.join("model.safetensors"), MAX_SAFETENSORS_BYTES)?;
            let weights = Weights::from_bytes(&bytes, &arch)?;
            Ok((arch, weights))
        }
    }
}

/// Instantiate `backend` for parsed weights. `Auto` picks the fastest backend this build
/// has for this machine: CUDA when the build has it and a device answers, Metal on macOS,
/// otherwise the CPU.
pub fn build(
    arch: Architecture,
    weights: Weights,
    backend: Backend,
    device: Device,
) -> Result<Box<dyn Network>, LoadError> {
    match backend {
        Backend::Cpu => Ok(Box::new(CpuNetwork::new(arch, weights, device.threads))),
        #[cfg(feature = "cuda")]
        Backend::Cuda => Ok(Box::new(cuda::CudaNetwork::new(
            arch,
            &weights,
            0,
            device.max_batch,
            cuda_fp8(device.precision),
        )?)),
        #[cfg(feature = "cuda")]
        Backend::Auto => {
            // The driver library is probed first: cudarc aborts, rather than errors, when
            // it is missing, which would defeat the fallback.
            if !cuda::available() {
                tracing::warn!("no CUDA driver library; using the CPU backend");
                return Ok(Box::new(CpuNetwork::new(arch, weights, device.threads)));
            }
            match cuda::CudaNetwork::new(
                arch.clone(),
                &weights,
                0,
                device.max_batch,
                cuda_fp8(device.precision),
            ) {
                Ok(network) => Ok(Box::new(network)),
                Err(error) => {
                    tracing::warn!(%error, "no usable CUDA device; using the CPU backend");
                    Ok(Box::new(CpuNetwork::new(arch, weights, device.threads)))
                }
            }
        }
        #[cfg(all(not(feature = "cuda"), target_os = "macos"))]
        Backend::Auto => {
            if !metal::available() {
                tracing::warn!("no Metal device; using the CPU backend");
                return Ok(Box::new(CpuNetwork::new(arch, weights, device.threads)));
            }
            match metal::MetalNetwork::new(
                arch.clone(),
                &weights,
                device.max_batch,
                metal_fp16(device.precision),
            ) {
                Ok(network) => Ok(Box::new(network)),
                Err(error) => {
                    tracing::warn!(%error, "Metal backend failed to load; using the CPU backend");
                    Ok(Box::new(CpuNetwork::new(arch, weights, device.threads)))
                }
            }
        }
        #[cfg(all(not(feature = "cuda"), not(target_os = "macos")))]
        Backend::Auto => Ok(Box::new(CpuNetwork::new(arch, weights, device.threads))),
        #[cfg(not(feature = "cuda"))]
        Backend::Cuda => Err(LoadError::Unavailable(Backend::Cuda)),
        #[cfg(target_os = "macos")]
        Backend::Metal => Ok(Box::new(metal::MetalNetwork::new(
            arch,
            &weights,
            device.max_batch,
            metal_fp16(device.precision),
        )?)),
        #[cfg(not(target_os = "macos"))]
        Backend::Metal => Err(LoadError::Unavailable(Backend::Metal)),
    }
}

/// The CUDA backend's choice for `precision`: fp8 where asked or automatic (it falls back
/// to fp16 itself when the card or the network cannot), fp16 otherwise; fp32 is not a mode
/// it has.
pub fn cuda_fp8(precision: Precision) -> bool {
    match precision {
        Precision::Auto | Precision::Fp8 => true,
        Precision::Fp16 => false,
        Precision::Fp32 => {
            tracing::warn!(
                "Precision=fp32: the CUDA backend runs its linear layers in fp16 at most"
            );
            false
        }
    }
}

/// The Metal backend's choice for `precision`: fp16 where asked or automatic, fp32 where
/// asked; fp8 is not a format Apple GPUs have.
pub fn metal_fp16(precision: Precision) -> bool {
    match precision {
        Precision::Auto | Precision::Fp16 => true,
        Precision::Fp32 => false,
        Precision::Fp8 => {
            tracing::warn!("Precision=fp8: Apple GPUs have no fp8; the Metal backend runs fp16");
            true
        }
    }
}

/// What the network says about one position, from the side to move.
#[derive(Debug, Clone, PartialEq)]
pub struct Evaluation {
    /// One prior per legal move, in [`LegalActions::moves`] order, summing to 1.
    pub priors: Vec<f32>,
    /// Win, draw, loss probabilities.
    pub wdl: [f32; 3],
}

impl Evaluation {
    /// Expected outcome `P(win) - P(loss)` in `[-1, 1]`, the value the search backs up.
    #[inline]
    pub fn value(&self) -> f32 {
        self.wdl[0] - self.wdl[2]
    }
}

/// A batch item: the encoded position and its legal moves.
pub type Leaf<'a> = (&'a EncodedBoard, &'a LegalActions);

/// A backend that can no longer evaluate (a device error). The search stops on it and the
/// engine reports it; nothing is ever substituted for a network's answer.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{0}")]
pub struct NetworkError(pub String);

pub trait Network: Send + Sync {
    /// Evaluate a batch; the result has one [`Evaluation`] per input, in order. Priors must
    /// cover exactly the legal moves (nothing else is ever seen by the tree).
    fn evaluate(&self, batch: &[Leaf<'_>]) -> Result<Vec<Evaluation>, NetworkError>;

    /// Human-readable description for `info string` at load time.
    fn describe(&self) -> String;
}

impl<N: Network + ?Sized> Network for Box<N> {
    fn evaluate(&self, batch: &[Leaf<'_>]) -> Result<Vec<Evaluation>, NetworkError> {
        (**self).evaluate(batch)
    }

    fn describe(&self) -> String {
        (**self).describe()
    }
}

impl<N: Network + ?Sized> Network for std::sync::Arc<N> {
    fn evaluate(&self, batch: &[Leaf<'_>]) -> Result<Vec<Evaluation>, NetworkError> {
        (**self).evaluate(batch)
    }

    fn describe(&self) -> String {
        (**self).describe()
    }
}
