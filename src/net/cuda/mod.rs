//! CUDA backend: fp16 weights, activations and residual stream with fp32 accumulation,
//! cuBLASLt for the linear layers (the residual adds and the GELU fused into their
//! epilogues) and the kernels in `kernels.cu` for everything else (attention included).
//!
//! The GELU is always the tanh form here, whatever the card says the network was trained
//! with: on an erf-trained network the mismatch moves the priors by ~7e-3 and the fused
//! epilogue is +13% throughput, and an SPRT at 10+0.1 on `small` measured the trade at
//! +19 ± 9 Elo for the fusion (2026-09-19). The CPU backend applies the card's form exactly.
//!
//! On Ada and newer, when the card carries the calibration (`encoder.fp8.activation_amax`)
//! and `Precision` is `fp8` or `auto` (its default), the four linear layers run on e4m3 tensor
//! cores: weights quantised at load with one scale per matrix, activations quantised by
//! the kernels that produce them (layer norm, attention, GELU) with static per-tensor
//! scales from the calibration, fp32 accumulation, fp16 outputs. Attention and the heads
//! stay fp16. The e4m3 noise moves the priors by up to ~6e-2 on `small` and the best move
//! on ~1.6% of the golden positions; an SPRT at 10+0.1 measured the trade at +48 ± 14 Elo
//! for the 40% more nodes (2026-09-19). Where FP8 cannot run (older device, uncalibrated
//! card) the layers run in fp16 and the load says so.
//!
//! Batches are padded to one of [`BUCKETS`]. For each bucket the whole forward pass,
//! host-to-device copies from pinned buffers, token embedding, the layers (layer norm →
//! QKV GEMM → fused attention → out-proj GEMM → layer norm → FFN GEMMs), final norm,
//! the policy kernel over legal actions only,
//! the value kernel, and the copies back, is captured once into a CUDA graph and replayed
//! per batch, so a batch costs one launch. cuBLASLt plans (descriptors and algorithm) are
//! built once per shape at load.

mod gemm;

use std::sync::{Arc, Mutex};

use cudarc::cublaslt::result as lt;
use cudarc::cublaslt::sys as ltsys;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaGraph, CudaModule, CudaSlice, CudaStream, DevicePtr,
    DriverError, LaunchConfig, PinnedHostSlice, PushKernelArg, sys,
};
use cudarc::nvrtc::Ptx;
use half::f16;
use thiserror::Error;

use self::gemm::Plan;
use super::model::{Architecture, WDL_OUTPUTS};
use super::weights::{Linear, Norm, Weights};
use super::{Evaluation, Leaf, Network, NetworkError};
use crate::encoding::EncodedBoard;

/// PTX for `kernels.cu`, compiled by build.rs.
const KERNELS_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

/// Bytes per position in the board upload: 64 piece tokens, castling, en passant.
const BOARD_BYTES: usize = 66;
/// Legal-action slots per position (218 is the most legal moves in any position).
const MAX_LEGAL: usize = 224;
/// Batch sizes with a captured graph; a batch pads up to the next one.
const BUCKETS: [usize; 8] = [8, 16, 32, 64, 128, 256, 512, 1024];
/// Largest batch the buffers are sized for.
/// Largest batch a bucket serves; `Options` caps `Batch` to the same number.
pub const MAX_BATCH: usize = 1024;
/// cuBLASLt workspace.
const WORKSPACE_BYTES: usize = 32 << 20;

#[derive(Debug, Error)]
pub enum CudaError {
    #[error("CUDA driver: {0}")]
    Driver(#[from] DriverError),
    #[error("cuBLASLt: {0}")]
    Blas(#[from] lt::CublasError),
    #[error("batch of {0} positions exceeds what this network was sized for")]
    BatchTooLarge(usize),
    #[error("position with {0} legal moves exceeds MAX_LEGAL {MAX_LEGAL}")]
    TooManyMoves(usize),
    #[error("d_model {0} is not a multiple of 32; the CUDA kernels reduce across full warps")]
    Width(usize),
    #[error("stream capture produced no graph")]
    CaptureFailed,
    #[error("network needs {0} tokens per position; kernels support at most 96")]
    TooManyTokens(usize),
    #[error("head dimension {0} is not supported by the attention kernel (32 or 64)")]
    HeadDim(usize),
}

/// Per-tensor headroom over the calibration maxima: a position outside the golden set can
/// exceed them; beyond this the conversion saturates at e4m3's 448.
const FP8_MARGIN: f32 = 1.25;
/// Largest finite e4m3 value.
const E4M3_MAX: f32 = 448.0;
/// Activation scales per layer in `Buffers::scales`: norm1 out, attention out, norm2 out,
/// GELU out.
const SCALES_PER_LAYER: usize = 4;

struct DevNorm {
    gamma: CudaSlice<f16>,
    beta: CudaSlice<f16>,
}

struct DevLinear {
    out_features: usize,
    in_features: usize,
    weight: CudaSlice<f16>,
    bias: CudaSlice<f16>,
    /// The weight as e4m3 with its dequantisation scale (one fp32), FP8 builds only.
    weight8: Option<(CudaSlice<u8>, CudaSlice<f32>)>,
}

struct DevLayer {
    norm1: DevNorm,
    in_proj: DevLinear,
    out_proj: DevLinear,
    norm2: DevNorm,
    linear1: DevLinear,
    linear2: DevLinear,
}

struct DevWeights {
    piece_embed: CudaSlice<f16>,
    square_embed: CudaSlice<f16>,
    state_embed: CudaSlice<f16>,
    castling_w: CudaSlice<f16>,
    castling_b: CudaSlice<f16>,
    en_passant_embed: CudaSlice<f16>,
    layers: Vec<DevLayer>,
    final_norm: DevNorm,
    policy_norm: DevNorm,
    policy_w: CudaSlice<f16>,
    policy_b: CudaSlice<f16>,
    value_norm: DevNorm,
    value_w1: CudaSlice<f16>,
    value_b1: CudaSlice<f16>,
    value_w2: CudaSlice<f16>,
    value_b2: CudaSlice<f16>,
}

struct Kernels {
    embed_tokens: CudaFunction,
    layernorm: CudaFunction,
    square_layernorm: CudaFunction,
    attention: CudaFunction,
    gelu_inplace: CudaFunction,
    policy_legal: CudaFunction,
    value_head: CudaFunction,
    /// e4m3 producers for the FP8 GEMMs.
    layernorm_e4m3: CudaFunction,
    attention_e4m3: CudaFunction,
    gelu_e4m3: CudaFunction,
    quantize_e4m3: CudaFunction,
}

/// Page-locked host memory addressed through a raw pointer. cudarc's `PinnedHostSlice`
/// synchronises an event on every access, which is not allowed while a graph is being
/// captured; the allocation is kept for its lifetime and the pointer used directly. The
/// stream is synchronised before every host read and after every host write by `forward`.
struct Pinned<T> {
    _alloc: PinnedHostSlice<T>,
    ptr: *mut T,
    len: usize,
}

impl<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits> Pinned<T> {
    fn new(ctx: &Arc<CudaContext>, len: usize) -> Result<Self, DriverError> {
        // SAFETY: the memory is written before it is read.
        let mut alloc: PinnedHostSlice<T> = unsafe { ctx.alloc_pinned(len)? };
        let ptr = alloc.as_mut_ptr()?;
        Ok(Self {
            _alloc: alloc,
            ptr,
            len,
        })
    }

    /// # Safety
    /// The stream must not be reading or writing this memory.
    unsafe fn slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// # Safety
    /// As [`Self::slice`].
    #[allow(clippy::mut_from_ref)]
    unsafe fn slice_mut(&self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

/// Device activation buffers and pinned host staging, sized for the instance's `max_batch`.
struct Buffers {
    boards_host: Pinned<u8>,
    legal_host: Pinned<i32>,
    priors_host: Pinned<f32>,
    wdl_host: Pinned<f32>,
    boards: CudaSlice<u8>,
    legal: CudaSlice<i32>,
    /// The residual stream, fp16: the GEMM epilogues add into it (β = 1).
    x: CudaSlice<f16>,
    h: CudaSlice<f16>,
    qkv: CudaSlice<f16>,
    attn: CudaSlice<f16>,
    ff: CudaSlice<f16>,
    /// e4m3 copies of `h`, `attn` and `ff` as the FP8 GEMMs read them (one byte each;
    /// allocated only for FP8).
    h8: CudaSlice<u8>,
    attn8: CudaSlice<u8>,
    ff8: CudaSlice<u8>,
    /// [`SCALES_PER_LAYER`] fp32 activation scales per layer.
    scales: CudaSlice<f32>,
    squares: CudaSlice<f16>,
    priors: CudaSlice<f32>,
    wdl: CudaSlice<f32>,
}

struct LayerPlans {
    in_proj: Plan,
    out_proj: Plan,
    linear1: Plan,
    linear2: Plan,
}

/// Everything shape-dependent for one batch size.
struct Bucket {
    positions: usize,
    layers: Vec<LayerPlans>,
    graph: Option<CudaGraph>,
}

pub struct CudaNetwork {
    arch: Architecture,
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    lt: ltsys::cublasLtHandle_t,
    workspace: CudaSlice<u8>,
    _module: Arc<CudaModule>,
    kernels: Kernels,
    weights: DevWeights,
    state: Mutex<State>,
    device_name: String,
    /// Largest batch this instance serves; buffers and buckets are sized for it.
    max_batch: usize,
    /// The linear layers run on e4m3.
    fp8: bool,
}

struct State {
    buffers: Buffers,
    buckets: Vec<Bucket>,
}

/// Dynamic shared memory the attention kernel needs per block: Q, K, V as fp16 with rows
/// padded to a multiple of 16 and 8 columns of padding, plus a 16-row fp32 score strip.
fn attention_shared_bytes(arch: &Architecture) -> usize {
    let (t, hd) = (arch.tokens, arch.head_dim());
    let rows = t.div_ceil(16) * 16;
    3 * rows * (hd + 8) * std::mem::size_of::<f16>() + 16 * (rows + 8) * std::mem::size_of::<f32>()
}

/// Whether the CUDA driver and cuBLASLt libraries can be loaded on this machine. cudarc
/// aborts the process when they are missing, so this is checked before touching it.
pub fn available() -> bool {
    fn loads(names: &[&str]) -> bool {
        // SAFETY: only loading the library to see that it exists; nothing is called.
        names
            .iter()
            .any(|name| unsafe { libloading::Library::new(name) }.is_ok())
    }
    if cfg!(target_os = "windows") {
        loads(&["nvcuda.dll"]) && loads(&["cublasLt64_12.dll", "cublasLt64_13.dll"])
    } else {
        loads(&["libcuda.so.1", "libcuda.so"]) && loads(&["libcublasLt.so.12", "libcublasLt.so"])
    }
}

impl CudaNetwork {
    /// Put `w` on CUDA device `ordinal` and build the per-bucket plans and graphs for
    /// batches up to `max_batch` (rounded up to a bucket, at most [`MAX_BATCH`]). `fp8`
    /// asks for e4m3 linear layers; without an Ada-or-newer device and a calibrated card
    /// the layers run in fp16 (see [`Self::is_fp8`]).
    pub fn new(
        arch: Architecture,
        w: &Weights,
        ordinal: usize,
        max_batch: usize,
        fp8: bool,
    ) -> Result<Self, CudaError> {
        let max_batch = BUCKETS
            .iter()
            .copied()
            .find(|&bucket| bucket >= max_batch)
            .unwrap_or(MAX_BATCH);
        if !arch.d_model.is_multiple_of(32) {
            // The reduction kernels shuffle across full warps.
            return Err(CudaError::Width(arch.d_model));
        }
        if arch.tokens > 96 {
            return Err(CudaError::TooManyTokens(arch.tokens));
        }
        let ctx = CudaContext::new(ordinal)?;
        // One stream, ordered by construction; the per-slice event bookkeeping would only
        // add work and cannot be captured into a graph.
        // SAFETY: every operation below goes through `self.stream` in program order.
        unsafe { ctx.disable_event_tracking() };
        let device_name = ctx.name()?;
        let fp8 = fp8
            && {
                let major = ctx.attribute(
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                )?;
                let minor = ctx.attribute(
                    sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                )?;
                if (major, minor) < (8, 9) {
                    tracing::warn!(
                        device = %device_name,
                        "FP8 needs compute capability 8.9 (Ada); this is {major}.{minor}: linear layers in fp16"
                    );
                    false
                } else if arch.fp8_amax.is_none() {
                    tracing::warn!(
                        "FP8 needs the calibration in model.json (encoder.fp8.activation_amax); this card has none: linear layers in fp16"
                    );
                    false
                } else {
                    true
                }
            };
        // A created stream: the legacy default stream cannot be captured into a graph.
        let stream = ctx.new_stream()?;
        let lt = lt::create_handle()?;
        let workspace: CudaSlice<u8> = stream.alloc_zeros(WORKSPACE_BYTES)?;
        let module = ctx.load_module(Ptx::from_src(KERNELS_PTX))?;
        let kernels = Kernels {
            embed_tokens: module.load_function("embed_tokens")?,
            layernorm: module.load_function("layernorm")?,
            square_layernorm: module.load_function("square_layernorm")?,
            attention: {
                let function = module.load_function(match arch.head_dim() {
                    32 => "attention_hd32",
                    64 => "attention_hd64",
                    other => return Err(CudaError::HeadDim(other)),
                })?;
                // K, V and the scores for one (position, head) can exceed the 48 KB a
                // kernel gets by default.
                function.set_attribute(
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    attention_shared_bytes(&arch) as i32,
                )?;
                function
            },
            gelu_inplace: module.load_function("gelu_inplace")?,
            policy_legal: module.load_function("policy_legal")?,
            value_head: module.load_function("value_head")?,
            layernorm_e4m3: module.load_function("layernorm_e4m3")?,
            attention_e4m3: {
                let function = module.load_function(match arch.head_dim() {
                    32 => "attention_hd32_e4m3",
                    _ => "attention_hd64_e4m3",
                })?;
                function.set_attribute(
                    sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    attention_shared_bytes(&arch) as i32,
                )?;
                function
            },
            gelu_e4m3: module.load_function("gelu_e4m3")?,
            quantize_e4m3: module.load_function("quantize_e4m3")?,
        };

        let up = |v: &[f32]| -> Result<CudaSlice<f16>, DriverError> {
            let halves: Vec<f16> = v.iter().map(|&x| f16::from_f32(x)).collect();
            stream.clone_htod(&halves)
        };
        let norm = |n: &Norm| -> Result<DevNorm, DriverError> {
            Ok(DevNorm {
                gamma: up(&n.weight)?,
                beta: up(&n.bias)?,
            })
        };
        let linear = |l: &Linear| -> Result<DevLinear, CudaError> {
            let weight = up(&l.weight)?;
            let weight8 = if fp8 {
                // One dequantisation scale per matrix, from its largest magnitude.
                let amax = l.weight.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
                let scale = stream.clone_htod(&[(amax / E4M3_MAX).max(f32::MIN_POSITIVE)])?;
                let mut codes: CudaSlice<u8> = stream.alloc_zeros(l.weight.len())?;
                let n = l.weight.len() as i32;
                // SAFETY: arguments match kernels.cu; the buffers cover `n` elements.
                unsafe {
                    stream
                        .launch_builder(&kernels.quantize_e4m3)
                        .arg(&weight)
                        .arg(&mut codes)
                        .arg(&n)
                        .arg(&scale)
                        .launch(LaunchConfig::for_num_elems(n as u32))?;
                }
                Some((codes, scale))
            } else {
                None
            };
            Ok(DevLinear {
                out_features: l.out_features,
                in_features: l.in_features,
                weight,
                bias: up(&l.bias)?,
                weight8,
            })
        };
        let mut layers = Vec::with_capacity(w.layers.len());
        for layer in &w.layers {
            layers.push(DevLayer {
                norm1: norm(&layer.norm1)?,
                in_proj: linear(&layer.in_proj)?,
                out_proj: linear(&layer.out_proj)?,
                norm2: norm(&layer.norm2)?,
                linear1: linear(&layer.linear1)?,
                linear2: linear(&layer.linear2)?,
            });
        }
        let weights = DevWeights {
            piece_embed: up(&w.piece_embed)?,
            square_embed: up(&w.square_embed)?,
            state_embed: up(&w.state_embed)?,
            castling_w: up(&w.castling_proj.weight)?,
            castling_b: up(&w.castling_proj.bias)?,
            en_passant_embed: up(&w.en_passant_embed)?,
            layers,
            final_norm: norm(&w.final_norm)?,
            policy_norm: norm(&w.policy_norm)?,
            policy_w: up(&w.policy_proj.weight)?,
            policy_b: up(&w.policy_proj.bias)?,
            value_norm: norm(&w.value_norm)?,
            value_w1: up(&w.value_ff1.weight)?,
            value_b1: up(&w.value_ff1.bias)?,
            value_w2: up(&w.value_ff2.weight)?,
            value_b2: up(&w.value_ff2.bias)?,
        };

        let (t, d) = (arch.tokens, arch.d_model);
        let rows = max_batch * t;
        let buffers = {
            Buffers {
                boards_host: Pinned::new(&ctx, max_batch * BOARD_BYTES)?,
                legal_host: Pinned::new(&ctx, max_batch * MAX_LEGAL)?,
                priors_host: Pinned::new(&ctx, max_batch * MAX_LEGAL)?,
                wdl_host: Pinned::new(&ctx, max_batch * WDL_OUTPUTS)?,
                boards: stream.alloc_zeros(max_batch * BOARD_BYTES)?,
                legal: stream.alloc_zeros(max_batch * MAX_LEGAL)?,
                x: stream.alloc_zeros(rows * d)?,
                h: stream.alloc_zeros(rows * d)?,
                qkv: stream.alloc_zeros(rows * 3 * d)?,
                attn: stream.alloc_zeros(rows * d)?,
                ff: stream.alloc_zeros(rows * arch.d_ff)?,
                h8: stream.alloc_zeros(if fp8 { rows * d } else { 1 })?,
                attn8: stream.alloc_zeros(if fp8 { rows * d } else { 1 })?,
                ff8: stream.alloc_zeros(if fp8 { rows * arch.d_ff } else { 1 })?,
                scales: {
                    // Static per-tensor scales from the card's calibration, with headroom.
                    let mut values = vec![1.0f32; arch.n_layers * SCALES_PER_LAYER];
                    if let Some(amax) = &arch.fp8_amax {
                        for (layer, row) in amax.iter().enumerate() {
                            let base = layer * SCALES_PER_LAYER;
                            for (i, &m) in row.iter().enumerate() {
                                values[base + i] = m * FP8_MARGIN / E4M3_MAX;
                            }
                        }
                    }
                    stream.clone_htod(&values)?
                },
                squares: stream.alloc_zeros(max_batch * 64 * d)?,
                priors: stream.alloc_zeros(max_batch * MAX_LEGAL)?,
                wdl: stream.alloc_zeros(max_batch * WDL_OUTPUTS)?,
            }
        };
        stream.synchronize()?;

        let mut network = Self {
            arch,
            ctx,
            stream,
            lt,
            workspace,
            _module: module,
            kernels,
            weights,
            state: Mutex::new(State {
                buffers,
                buckets: Vec::new(),
            }),
            device_name,
            max_batch,
            fp8,
        };
        network.build_buckets()?;
        Ok(network)
    }

    fn ptr<T>(&self, slice: &CudaSlice<T>) -> u64 {
        slice.device_ptr(&self.stream).0
    }

    /// Whether the linear layers run on e4m3 (asked for, and the device and card allow it).
    pub fn is_fp8(&self) -> bool {
        self.fp8
    }

    /// The GEMM's A operand: the e4m3 weight in FP8 builds, else the fp16 one.
    fn weight_ptr(&self, layer: &DevLinear) -> u64 {
        match &layer.weight8 {
            Some((codes, _)) => self.ptr(codes),
            None => self.ptr(&layer.weight),
        }
    }

    /// `h = LN(x)` as the next GEMM reads it: fp16 `h`, or e4m3 `h8` with the scale at
    /// `scale_index` in FP8 builds. `rows` blocks of `min(d, 256)` threads.
    fn layernorm(
        &self,
        b: &mut Buffers,
        norm: &DevNorm,
        scale_index: usize,
        rows: usize,
        e4m3: bool,
    ) -> Result<(), CudaError> {
        let d = self.arch.d_model;
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: ((d as u32).min(256), 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: kernel arguments match kernels.cu; every buffer covers `max_batch` rows.
        unsafe {
            if e4m3 {
                let scale = b.scales.slice(scale_index..scale_index + 1);
                self.stream
                    .launch_builder(&self.kernels.layernorm_e4m3)
                    .arg(&b.x)
                    .arg(&mut b.h8)
                    .arg(&norm.gamma)
                    .arg(&norm.beta)
                    .arg(&(d as i32))
                    .arg(&self.arch.layer_norm_eps)
                    .arg(&scale)
                    .launch(cfg)?;
            } else {
                self.stream
                    .launch_builder(&self.kernels.layernorm)
                    .arg(&b.x)
                    .arg(&mut b.h)
                    .arg(&norm.gamma)
                    .arg(&norm.beta)
                    .arg(&(d as i32))
                    .arg(&self.arch.layer_norm_eps)
                    .launch(cfg)?;
            }
        }
        Ok(())
    }

    /// `h = LN(x)` in fp16 whatever the build: the heads read fp16.
    fn layernorm_f16(&self, b: &mut Buffers, norm: &DevNorm, rows: usize) -> Result<(), CudaError> {
        let d = self.arch.d_model;
        // SAFETY: as `layernorm`.
        unsafe {
            self.stream
                .launch_builder(&self.kernels.layernorm)
                .arg(&b.x)
                .arg(&mut b.h)
                .arg(&norm.gamma)
                .arg(&norm.beta)
                .arg(&(d as i32))
                .arg(&self.arch.layer_norm_eps)
                .launch(LaunchConfig {
                    grid_dim: (rows as u32, 1, 1),
                    block_dim: ((d as u32).min(256), 1, 1),
                    shared_mem_bytes: 0,
                })?;
        }
        Ok(())
    }

    /// Plan for `out[rows, out_features] = a[rows, in_features] · Wᵀ + bias`.
    /// `residual` plans add their product into the fp16 residual stream (out-proj, FFN2);
    /// `gelu` plans apply the tanh GELU in the epilogue (FFN1 of fp16 builds).
    /// For an e4m3 layer `input_scale` is the index into `scales` of its input's scale.
    fn linear_plan(
        &self,
        layer: &DevLinear,
        rows: usize,
        residual: bool,
        gelu: bool,
        scales: &CudaSlice<f32>,
        input_scale: usize,
    ) -> Result<Plan, CudaError> {
        let fp8 = layer.weight8.as_ref().map(|(_, a_scale)| gemm::Fp8 {
            a_scale: self.ptr(a_scale),
            b_scale: self.ptr(scales) + (input_scale * std::mem::size_of::<f32>()) as u64,
        });
        Ok(Plan::new(
            self.lt,
            &gemm::Linear {
                rows: rows as u64,
                k: layer.in_features as u64,
                n: layer.out_features as u64,
                bias: self.ptr(&layer.bias),
                residual,
                gelu,
                fp8,
            },
            self.ptr(&self.workspace),
            WORKSPACE_BYTES,
        )?)
    }

    /// Build the plans and capture the graph for every bucket.
    fn build_buckets(&mut self) -> Result<(), CudaError> {
        let arch = self.arch.clone();
        let t = arch.tokens;
        // cuBLASLt fuses the GELU into FFN1 for fp16 operands only; FP8 builds run the
        // activation in the kernel that quantises for FFN2.
        let epilogue_gelu = !self.fp8;
        // Borrowed for the plans' scale pointers; the buffers outlive every plan.
        let scales = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .buffers
            .scales
            .clone();
        let mut buckets = Vec::with_capacity(BUCKETS.len());
        for &positions in BUCKETS.iter().filter(|&&b| b <= self.max_batch) {
            let rows = positions * t;
            let mut layers = Vec::with_capacity(self.weights.layers.len());
            for (i, layer) in self.weights.layers.iter().enumerate() {
                let s = i * SCALES_PER_LAYER;
                layers.push(LayerPlans {
                    in_proj: self.linear_plan(&layer.in_proj, rows, false, false, &scales, s)?,
                    out_proj: self.linear_plan(
                        &layer.out_proj,
                        rows,
                        true,
                        false,
                        &scales,
                        s + 1,
                    )?,
                    linear1: self.linear_plan(
                        &layer.linear1,
                        rows,
                        false,
                        epilogue_gelu,
                        &scales,
                        s + 2,
                    )?,
                    linear2: self.linear_plan(&layer.linear2, rows, true, false, &scales, s + 3)?,
                });
            }
            buckets.push(Bucket {
                positions,
                layers,
                graph: None,
            });
        }

        // Capture one graph per bucket. The host buffers hold zeros: valid input.
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // SAFETY: the stream is idle (synchronised above).
        unsafe {
            state.buffers.boards_host.slice_mut().fill(0);
            state.buffers.legal_host.slice_mut().fill(-1);
        }
        for bucket in &mut buckets {
            let captured = self.capture(&mut state.buffers, bucket);
            match captured {
                Ok(graph) => bucket.graph = Some(graph),
                Err(error) => {
                    tracing::warn!(%error, positions = bucket.positions, "graph capture failed; running eagerly");
                    // Leave the stream in a sane state if capture was left open.
                    let _ = self.stream.end_capture(
                        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                    );
                }
            }
        }
        self.stream.synchronize()?;
        state.buckets = buckets;
        Ok(())
    }

    fn capture(&self, buffers: &mut Buffers, bucket: &Bucket) -> Result<CudaGraph, CudaError> {
        self.stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        self.enqueue(buffers, bucket)?;
        let graph = self
            .stream
            .end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )?
            .ok_or(CudaError::CaptureFailed)?;
        Ok(graph)
    }

    /// Enqueue the whole forward pass for `bucket` on the stream (captured into its graph
    /// at load, or run directly when capture failed).
    fn enqueue(&self, b: &mut Buffers, bucket: &Bucket) -> Result<(), CudaError> {
        let arch = &self.arch;
        let (t, d, hd, heads) = (arch.tokens, arch.d_model, arch.head_dim(), arch.n_heads);
        let n = bucket.positions;
        let rows = n * t;
        let stream = &self.stream;
        let cu_stream = stream.cu_stream() as ltsys::cudaStream_t;
        let w = &self.weights;
        let eps = arch.layer_norm_eps;
        let block = |threads: u32, blocks: u32, shared: u32| LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: shared,
        };
        // `d` is a multiple of 32 (checked in `new`), so every warp the reductions shuffle
        // across is full.
        let ln_threads = (d as u32).min(256);
        // Eight warps per (position, head) block (ATT_THREADS in kernels.cu).
        let attn_threads = 256u32;
        let attn_shared = attention_shared_bytes(arch) as u32;

        // ---- upload
        // SAFETY: host buffers are pinned and outlive the stream work; device buffers
        // cover `max_batch` positions.
        unsafe {
            let cu = stream.cu_stream();
            cudarc::driver::result::memcpy_htod_async(
                self.ptr(&b.boards),
                &b.boards_host.slice()[..n * BOARD_BYTES],
                cu,
            )?;
            cudarc::driver::result::memcpy_htod_async(
                self.ptr(&b.legal),
                &b.legal_host.slice()[..n * MAX_LEGAL],
                cu,
            )?;
        }

        // ---- tokens
        // SAFETY: kernel arguments match kernels.cu; every buffer covers `max_batch` rows.
        unsafe {
            stream
                .launch_builder(&self.kernels.embed_tokens)
                .arg(&b.boards)
                .arg(&w.piece_embed)
                .arg(&w.square_embed)
                .arg(&w.state_embed)
                .arg(&w.castling_w)
                .arg(&w.castling_b)
                .arg(&w.en_passant_embed)
                .arg(&mut b.x)
                .arg(&(d as i32))
                .arg(&(t as i32))
                .arg(&(arch.square_token_start as i32))
                .launch(block(ln_threads, rows as u32, 0))?;
        }

        // ---- encoder
        // The residual stream `x` is fp16; out-proj and FFN2 add into it inside their GEMM
        // (β = 1), so the layer norms are pure normalisation passes. A linear layer that
        // runs on e4m3 reads the e4m3 buffer its producer (layer norm, attention, GELU)
        // writes with the layer's static scale; otherwise the producer writes fp16.
        let (x_p, qkv_p) = (self.ptr(&b.x), self.ptr(&b.qkv));
        let (h_p, h8_p) = (self.ptr(&b.h), self.ptr(&b.h8));
        let (attn_p, attn8_p) = (self.ptr(&b.attn), self.ptr(&b.attn8));
        let (ff_p, ff8_p) = (self.ptr(&b.ff), self.ptr(&b.ff8));
        // Matches `build_buckets`: the epilogue carries the GELU for fp16 operands only.
        // The kernels apply the same tanh form (see the module doc).
        let epilogue_gelu = !self.fp8;
        let tanh = 1i32;
        for (i, (layer, plans)) in w.layers.iter().zip(&bucket.layers).enumerate() {
            let s = i * SCALES_PER_LAYER;
            let (q_in, q_out, q_ff1, q_ff2) = (
                layer.in_proj.weight8.is_some(),
                layer.out_proj.weight8.is_some(),
                layer.linear1.weight8.is_some(),
                layer.linear2.weight8.is_some(),
            );
            self.layernorm(b, &layer.norm1, s, rows, q_in)?;
            unsafe {
                plans.in_proj.run(
                    self.weight_ptr(&layer.in_proj),
                    if q_in { h8_p } else { h_p },
                    qkv_p,
                    cu_stream,
                )?;
                if q_out {
                    let attn_scale = b.scales.slice(s + 1..s + 2);
                    stream
                        .launch_builder(&self.kernels.attention_e4m3)
                        .arg(&b.qkv)
                        .arg(&mut b.attn8)
                        .arg(&(t as i32))
                        .arg(&(d as i32))
                        .arg(&(heads as i32))
                        .arg(&(1.0 / (hd as f32).sqrt()))
                        .arg(&attn_scale)
                        .launch(block(attn_threads, (n * heads) as u32, attn_shared))?;
                } else {
                    stream
                        .launch_builder(&self.kernels.attention)
                        .arg(&b.qkv)
                        .arg(&mut b.attn)
                        .arg(&(t as i32))
                        .arg(&(d as i32))
                        .arg(&(heads as i32))
                        .arg(&(1.0 / (hd as f32).sqrt()))
                        .launch(block(attn_threads, (n * heads) as u32, attn_shared))?;
                }
                plans.out_proj.run(
                    self.weight_ptr(&layer.out_proj),
                    if q_out { attn8_p } else { attn_p },
                    x_p,
                    cu_stream,
                )?;
            }
            self.layernorm(b, &layer.norm2, s + 2, rows, q_ff1)?;
            unsafe {
                let count = rows * arch.d_ff;
                // FFN1 writes fp16 `ff`; the GELU is then either already applied (fused
                // epilogue), applied in place, or applied while quantising for an FP8 FFN2.
                plans.linear1.run(
                    self.weight_ptr(&layer.linear1),
                    if q_ff1 { h8_p } else { h_p },
                    ff_p,
                    cu_stream,
                )?;
                if q_ff2 {
                    let ff_scale = b.scales.slice(s + 3..s + 4);
                    stream
                        .launch_builder(&self.kernels.gelu_e4m3)
                        .arg(&b.ff)
                        .arg(&mut b.ff8)
                        .arg(&(count as i32))
                        .arg(&tanh)
                        .arg(&ff_scale)
                        .launch(block(256, count.div_ceil(256) as u32, 0))?;
                } else if !epilogue_gelu {
                    stream
                        .launch_builder(&self.kernels.gelu_inplace)
                        .arg(&mut b.ff)
                        .arg(&(count as i32))
                        .arg(&tanh)
                        .launch(block(256, count.div_ceil(256) as u32, 0))?;
                }
                plans.linear2.run(
                    self.weight_ptr(&layer.linear2),
                    if q_ff2 { ff8_p } else { ff_p },
                    x_p,
                    cu_stream,
                )?;
            }
        }
        self.layernorm_f16(b, &w.final_norm, rows)?;
        unsafe {
            // ---- heads
            stream
                .launch_builder(&self.kernels.square_layernorm)
                .arg(&b.h)
                .arg(&mut b.squares)
                .arg(&w.policy_norm.gamma)
                .arg(&w.policy_norm.beta)
                .arg(&(d as i32))
                .arg(&(t as i32))
                .arg(&(arch.square_token_start as i32))
                .arg(&eps)
                .launch(block(ln_threads, (n * 64) as u32, 0))?;
            let shared = ((MAX_LEGAL + 32) * std::mem::size_of::<f32>()) as u32;
            stream
                .launch_builder(&self.kernels.policy_legal)
                .arg(&b.squares)
                .arg(&w.policy_w)
                .arg(&w.policy_b)
                .arg(&b.legal)
                .arg(&mut b.priors)
                .arg(&(d as i32))
                .arg(&(MAX_LEGAL as i32))
                .launch(block(MAX_LEGAL as u32, n as u32, shared))?;
            let shared = ((2 * d + 32) * std::mem::size_of::<f32>()) as u32;
            stream
                .launch_builder(&self.kernels.value_head)
                .arg(&b.h)
                .arg(&w.value_norm.gamma)
                .arg(&w.value_norm.beta)
                .arg(&w.value_w1)
                .arg(&w.value_b1)
                .arg(&w.value_w2)
                .arg(&w.value_b2)
                .arg(&mut b.wdl)
                .arg(&(t as i32))
                .arg(&(d as i32))
                .arg(&eps)
                .launch(block(ln_threads, n as u32, shared))?;
        }

        // ---- download
        // SAFETY: as for the upload.
        unsafe {
            let cu = stream.cu_stream();
            cudarc::driver::result::memcpy_dtoh_async(
                &mut b.priors_host.slice_mut()[..n * MAX_LEGAL],
                self.ptr(&b.priors),
                cu,
            )?;
            cudarc::driver::result::memcpy_dtoh_async(
                &mut b.wdl_host.slice_mut()[..n * WDL_OUTPUTS],
                self.ptr(&b.wdl),
                cu,
            )?;
        }
        Ok(())
    }

    /// Run the network on `boards` with `legal` action lists and hand back the priors
    /// (`MAX_LEGAL` per position, in legal order) and WDL via `sink`.
    fn forward(
        &self,
        boards: &[&EncodedBoard],
        legal: &[&[usize]],
        sink: &mut dyn FnMut(&[f32], &[f32]),
    ) -> Result<(), CudaError> {
        let n = boards.len();
        if n > self.max_batch {
            return Err(CudaError::BatchTooLarge(n));
        }
        // The backend thread is new for every search; the context must be current on it
        // before the raw copies below (the graph launch and synchronize bind it themselves).
        self.ctx.bind_to_thread()?;
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let state = &mut *state;
        let index = state
            .buckets
            .iter()
            .position(|bucket| bucket.positions >= n)
            .unwrap_or(state.buckets.len() - 1);
        let bucket = &state.buckets[index];
        let padded = bucket.positions;
        let b = &mut state.buffers;

        // SAFETY: the previous forward synchronised the stream before returning, so no
        // copy is in flight.
        unsafe {
            let boards_host = b.boards_host.slice_mut();
            boards_host[..padded * BOARD_BYTES].fill(0);
            for (p, board) in boards.iter().enumerate() {
                let dst = &mut boards_host[p * BOARD_BYTES..(p + 1) * BOARD_BYTES];
                dst[..64].copy_from_slice(&board.pieces);
                dst[64] = board.castling;
                dst[65] = board.en_passant;
            }
            let legal_host = b.legal_host.slice_mut();
            legal_host[..padded * MAX_LEGAL].fill(-1);
            for (p, actions) in legal.iter().enumerate() {
                if actions.len() > MAX_LEGAL {
                    return Err(CudaError::TooManyMoves(actions.len()));
                }
                for (slot, &action) in legal_host[p * MAX_LEGAL..].iter_mut().zip(*actions) {
                    *slot = action as i32;
                }
            }
        }

        match &bucket.graph {
            Some(graph) => graph.launch()?,
            None => self.enqueue(b, bucket)?,
        }
        self.stream.synchronize()?;
        // SAFETY: synchronised: the downloads have landed and nothing else is queued.
        unsafe {
            sink(
                &b.priors_host.slice()[..n * MAX_LEGAL],
                &b.wdl_host.slice()[..n * WDL_OUTPUTS],
            );
        }
        Ok(())
    }
}

impl Network for CudaNetwork {
    fn evaluate(&self, batch: &[Leaf<'_>]) -> Result<Vec<Evaluation>, NetworkError> {
        let boards: Vec<&EncodedBoard> = batch.iter().map(|(board, _)| *board).collect();
        let actions: Vec<Vec<usize>> = batch
            .iter()
            .map(|(_, legal)| legal.moves.iter().map(|&(_, a)| a).collect())
            .collect();
        let action_refs: Vec<&[usize]> = actions.iter().map(Vec::as_slice).collect();
        let mut out = Vec::with_capacity(batch.len());
        let result = self.forward(&boards, &action_refs, &mut |priors, wdl| {
            out.extend(batch.iter().enumerate().map(|(p, (_, legal))| Evaluation {
                priors: priors[p * MAX_LEGAL..p * MAX_LEGAL + legal.len()].to_vec(),
                wdl: [wdl[p * 3], wdl[p * 3 + 1], wdl[p * 3 + 2]],
            }));
        });
        result.map_err(|error| NetworkError(format!("cuda forward failed: {error}")))?;
        Ok(out)
    }

    fn describe(&self) -> String {
        format!(
            "cuda {} on {}: {}",
            if self.fp8 { "fp8" } else { "fp16" },
            self.device_name,
            self.arch.describe()
        )
    }
}

// Used from one thread at a time under the state mutex; the raw cuBLASLt handle and the
// pinned host buffers' pointers are the non-Send members, and both are only touched
// under `state`.
unsafe impl Send for CudaNetwork {}
unsafe impl Sync for CudaNetwork {}

impl Drop for CudaNetwork {
    fn drop(&mut self) {
        let _ = self.ctx.synchronize();
        // Graphs and plans hold the stream and handle; drop them before the handle.
        if let Ok(mut state) = self.state.lock() {
            state.buckets.clear();
        }
        // SAFETY: created by `new`, not used after this.
        unsafe {
            let _ = lt::destroy_handle(self.lt);
        }
    }
}
