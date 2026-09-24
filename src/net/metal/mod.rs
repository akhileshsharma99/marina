//! Metal backend for Apple GPUs: Metal Performance Shaders for the linear layers
//! (`MPSMatrixMultiplication` in fp16, which Apple GPUs run at twice the fp32 rate) and
//! the kernels in `kernels.metal` for everything else (token embedding, layer norm,
//! attention, biases and GELU, the head gathers). The residual stream and every reduction
//! are fp32; only what a GEMM reads or writes is fp16, the same split as the CUDA fp16
//! path. The kernel source is compiled by the system's Metal compiler at load, so no
//! developer tools are needed to build or run.
//!
//! Batches are padded to one of [`BUCKETS`]; the activation buffers are sized once for the
//! largest, and the MPS multiplications (which fix their row count at creation) are built
//! per bucket the first time it is used. One command buffer per batch: every kernel and
//! GEMM of the forward pass is encoded, committed, and waited for; the results are read
//! straight from shared-storage buffers (unified memory, so no copies).

use std::collections::HashMap;
use std::sync::Mutex;

use objc2::AnyThread;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSUInteger};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice, MTLDevice,
    MTLLibrary, MTLResourceOptions, MTLSize,
};
use objc2_metal_performance_shaders::{
    MPSDataType, MPSMatrix, MPSMatrixDescriptor, MPSMatrixMultiplication,
};
use thiserror::Error;

use super::model::{Architecture, Gelu, POLICY_PLANES, WDL_OUTPUTS};
use super::weights::{Linear, Norm, Weights};
use super::{Evaluation, Leaf, Network, NetworkError};
use crate::encoding::EncodedBoard;
use crate::encoding::action::PLANES_PER_SQUARE;

const KERNELS: &str = include_str!("kernels.metal");
/// Bytes per position in the board upload: 64 piece tokens, castling, en passant.
const BOARD_BYTES: usize = 66;
/// Batch sizes the buffers and multiplications are built for; a batch pads up to the next.
const BUCKETS: [usize; 8] = [8, 16, 32, 64, 128, 256, 512, 1024];
pub const MAX_BATCH: usize = 1024;

#[derive(Debug, Error)]
pub enum MetalError {
    #[error("no Metal device")]
    NoDevice,
    #[error("Metal: {0}")]
    Metal(String),
    #[error("kernels.metal: {0}")]
    Compile(String),
    #[error("network has {0} tokens; the attention kernel holds at most 128")]
    Tokens(usize),
    #[error("head_dim {0} is not a multiple of 4; the attention kernel loads float4s")]
    HeadDim(usize),
    #[error("d_ff {0} is not a multiple of 4; the element-wise kernels work in float4s")]
    Dff(usize),
    #[error(
        "the attention kernel allows {max} threads per threadgroup; the network has {tokens} tokens"
    )]
    Threadgroup { tokens: usize, max: usize },
    #[error("the GPU did not complete the batch: {0}")]
    Gpu(String),
    #[error("batch of {got} positions exceeds the {max} the buffers hold")]
    BatchTooLarge { got: usize, max: usize },
    #[error(
        "network puts its square tokens at {0}; the embedding kernel expects one state token first"
    )]
    SquareStart(usize),
}

type Device = Retained<ProtocolObject<dyn MTLDevice>>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Whether this machine has a Metal device.
pub fn available() -> bool {
    MTLCreateSystemDefaultDevice().is_some()
}

/// Mirrors `struct Shape` in kernels.metal.
#[repr(C)]
#[derive(Clone, Copy)]
struct Shape {
    positions: u32,
    tokens: u32,
    d: u32,
    heads: u32,
    head_dim: u32,
    square_start: u32,
    d_ff: u32,
    eps: f32,
    scale: f32,
}

struct Pipelines {
    f32_to_f16: Pipeline,
    embed: Pipeline,
    layer_norm: Pipeline,
    layer_norm_f32: Pipeline,
    add_bias: Pipeline,
    add_bias4: Pipeline,
    add_bias_residual: Pipeline,
    bias_gelu: Pipeline,
    attention: Pipeline,
}

/// A linear layer on the device: `[out, in]` fp16 weights as an MPS matrix (used
/// transposed; MPS reads that as fast as the plain layout) and the fp32 bias.
struct DevLinear {
    /// Owns the weight storage `matrix` views; only read through the matrix.
    _weight: Buffer,
    matrix: Retained<MPSMatrix>,
    bias: Buffer,
    out: usize,
    inp: usize,
}

struct DevNorm {
    gamma: Buffer,
    beta: Buffer,
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
    piece_embed: Buffer,
    square_embed: Buffer,
    state_embed: Buffer,
    castling_w: Buffer,
    castling_b: Buffer,
    en_passant_embed: Buffer,
    layers: Vec<DevLayer>,
    final_norm: DevNorm,
    policy_norm: DevNorm,
    policy_proj: DevLinear,
    value_norm: DevNorm,
    value_ff1: DevLinear,
    value_ff2: DevLinear,
}

/// Activation buffers, sized for `MAX_BATCH` positions.
/// Per-batch working buffers. `x` (the residual stream), `hf` (the final norm, input to
/// the head norms) and the two head results are fp32; everything a GEMM reads or writes
/// is the GEMM type (`policy16`/`value16` are the heads' GEMM outputs in that type).
struct Activations {
    boards: Buffer,
    x: Buffer,
    h: Buffer,
    hf: Buffer,
    qkv: Buffer,
    attn: Buffer,
    proj: Buffer,
    ff: Buffer,
    squares: Buffer,
    policy16: Buffer,
    policy: Buffer,
    vnorm: Buffer,
    hidden: Buffer,
    value16: Buffer,
    value: Buffer,
}

/// Everything touched while a batch is in flight, behind one lock: Metal objects are
/// thread-safe, but the activation buffers are one set.
struct State {
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    act: Activations,
    /// `(rows, out, in)` → multiplication with those fixed dimensions.
    matmuls: HashMap<(usize, usize, usize), Retained<MPSMatrixMultiplication>>,
}

// SAFETY: Metal devices, queues, buffers, pipeline states and MPS kernels may be used from
// any thread (Apple's Metal threading guarantees); the mutable parts are behind `State`'s
// mutex and every batch waits for its command buffer before the lock is released.
unsafe impl Send for MetalNetwork {}
unsafe impl Sync for MetalNetwork {}

pub struct MetalNetwork {
    arch: Architecture,
    gemm: GemmType,
    device: Device,
    pipelines: Pipelines,
    weights: DevWeights,
    state: Mutex<State>,
    max_batch: usize,
    device_name: String,
}

fn ns(s: &str) -> Retained<NSString> {
    NSString::from_str(s)
}

fn buffer(device: &Device, bytes: usize) -> Result<Buffer, MetalError> {
    device
        .newBufferWithLength_options(
            bytes.max(16) as NSUInteger,
            MTLResourceOptions::StorageModeShared,
        )
        .ok_or_else(|| MetalError::Metal(format!("cannot allocate a {bytes}-byte buffer")))
}

fn upload(device: &Device, data: &[f32]) -> Result<Buffer, MetalError> {
    let buf = buffer(device, data.len() * 4)?;
    // SAFETY: shared storage; the buffer is at least `data.len()` floats long.
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            buf.contents().as_ptr().cast::<f32>(),
            data.len(),
        );
    }
    Ok(buf)
}

/// The GEMM element type: fp16 or fp32, chosen per network by `Precision`.
#[derive(Debug, Clone, Copy)]
struct GemmType {
    fp16: bool,
}

impl GemmType {
    fn bytes(self) -> usize {
        if self.fp16 { 2 } else { 4 }
    }
    fn mps(self) -> MPSDataType {
        if self.fp16 {
            MPSDataType::Float16
        } else {
            MPSDataType::Float32
        }
    }
}

fn matrix(buf: &Buffer, rows: usize, cols: usize, gemm: GemmType) -> Retained<MPSMatrix> {
    // SAFETY: the descriptor describes `rows × cols` elements of `gemm` within `buf`, which
    // every caller sizes for at least that.
    unsafe {
        let desc = MPSMatrixDescriptor::matrixDescriptorWithRows_columns_rowBytes_dataType(
            rows as NSUInteger,
            cols as NSUInteger,
            (cols * gemm.bytes()) as NSUInteger,
            gemm.mps(),
        );
        MPSMatrix::initWithBuffer_descriptor(MPSMatrix::alloc(), buf, &desc)
    }
}

/// Weights in the GEMM type: uploaded as fp32 and, for fp16, converted by one kernel
/// dispatch with the fp32 copy dropped.
struct Converter<'a> {
    device: &'a Device,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipeline: &'a Pipeline,
    gemm: GemmType,
}

impl Converter<'_> {
    fn to_gemm(&self, data: &[f32]) -> Result<Buffer, MetalError> {
        let src = upload(self.device, data)?;
        if !self.gemm.fp16 {
            return Ok(src);
        }
        let dst = buffer(self.device, data.len() * 2)?;
        let cb = self
            .queue
            .commandBuffer()
            .ok_or_else(|| MetalError::Metal("cannot create a command buffer".into()))?;
        let enc = cb.computeCommandEncoder().expect("compute encoder");
        enc.setComputePipelineState(self.pipeline);
        set_buffers(&enc, &[&src, &dst]);
        set_bytes(&enc, 2, &(data.len() as u32));
        let group = self.pipeline.maxTotalThreadsPerThreadgroup().min(256);
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: data.len().max(1) as NSUInteger,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: group as NSUInteger,
                height: 1,
                depth: 1,
            },
        );
        enc.endEncoding();
        cb.commit();
        cb.waitUntilCompleted();
        Ok(dst)
    }
}

impl DevLinear {
    fn new(conv: &Converter<'_>, linear: &Linear) -> Result<Self, MetalError> {
        let weight = conv.to_gemm(&linear.weight)?;
        let matrix = matrix(&weight, linear.out_features, linear.in_features, conv.gemm);
        Ok(Self {
            _weight: weight,
            matrix,
            bias: upload(conv.device, &linear.bias)?,
            out: linear.out_features,
            inp: linear.in_features,
        })
    }
}

impl DevNorm {
    fn new(device: &Device, norm: &Norm) -> Result<Self, MetalError> {
        Ok(Self {
            gamma: upload(device, &norm.weight)?,
            beta: upload(device, &norm.bias)?,
        })
    }
}

fn pipeline(
    device: &Device,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Result<Pipeline, MetalError> {
    let function = library
        .newFunctionWithName(&ns(name))
        .ok_or_else(|| MetalError::Compile(format!("kernel {name} is missing")))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| MetalError::Metal(format!("pipeline {name}: {e}")))
}

impl MetalNetwork {
    pub fn new(
        arch: Architecture,
        weights: &Weights,
        max_batch: usize,
        fp16: bool,
    ) -> Result<Self, MetalError> {
        if arch.tokens > 128 {
            return Err(MetalError::Tokens(arch.tokens));
        }
        if arch.square_token_start != 1 {
            return Err(MetalError::SquareStart(arch.square_token_start));
        }
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::NoDevice)?;
        let device_name = device.name().to_string();
        if !arch.head_dim().is_multiple_of(4) {
            return Err(MetalError::HeadDim(arch.head_dim()));
        }
        if !arch.d_ff.is_multiple_of(4) {
            return Err(MetalError::Dff(arch.d_ff));
        }
        // The attention kernel is specialised to the architecture: constant trip counts
        // let the compiler unroll and keep its state in registers.
        let gemm_t = if fp16 { "half" } else { "float" };
        let source = format!(
            "#define TOKENS {}u\n#define HEAD_DIM {}u\n#define GEMM_T {gemm_t}\n{KERNELS}",
            arch.tokens,
            arch.head_dim()
        );
        let library = device
            .newLibraryWithSource_options_error(&ns(&source), None)
            .map_err(|e| MetalError::Compile(e.to_string()))?;
        let pipelines = Pipelines {
            f32_to_f16: pipeline(&device, &library, "f32_to_f16")?,
            embed: pipeline(&device, &library, "embed")?,
            layer_norm: pipeline(&device, &library, "layer_norm")?,
            layer_norm_f32: pipeline(&device, &library, "layer_norm_f32")?,
            add_bias: pipeline(&device, &library, "add_bias")?,
            add_bias4: pipeline(&device, &library, "add_bias4")?,
            add_bias_residual: pipeline(&device, &library, "add_bias_residual")?,
            bias_gelu: pipeline(&device, &library, "bias_gelu")?,
            attention: pipeline(&device, &library, "attention")?,
        };
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| MetalError::Metal("cannot create a command queue".into()))?;
        let gemm = GemmType { fp16 };
        let conv = Converter {
            device: &device,
            queue: queue.clone(),
            pipeline: &pipelines.f32_to_f16,
            gemm,
        };
        let w = weights;
        let dev_weights = DevWeights {
            piece_embed: upload(&device, &w.piece_embed)?,
            square_embed: upload(&device, &w.square_embed)?,
            state_embed: upload(&device, &w.state_embed)?,
            castling_w: upload(&device, &w.castling_proj.weight)?,
            castling_b: upload(&device, &w.castling_proj.bias)?,
            en_passant_embed: upload(&device, &w.en_passant_embed)?,
            layers: w
                .layers
                .iter()
                .map(|l| {
                    Ok(DevLayer {
                        norm1: DevNorm::new(&device, &l.norm1)?,
                        in_proj: DevLinear::new(&conv, &l.in_proj)?,
                        out_proj: DevLinear::new(&conv, &l.out_proj)?,
                        norm2: DevNorm::new(&device, &l.norm2)?,
                        linear1: DevLinear::new(&conv, &l.linear1)?,
                        linear2: DevLinear::new(&conv, &l.linear2)?,
                    })
                })
                .collect::<Result<Vec<_>, MetalError>>()?,
            final_norm: DevNorm::new(&device, &w.final_norm)?,
            policy_norm: DevNorm::new(&device, &w.policy_norm)?,
            policy_proj: DevLinear::new(&conv, &w.policy_proj)?,
            value_norm: DevNorm::new(&device, &w.value_norm)?,
            value_ff1: DevLinear::new(&conv, &w.value_ff1)?,
            value_ff2: DevLinear::new(&conv, &w.value_ff2)?,
        };
        let max_batch = max_batch.clamp(1, MAX_BATCH);
        let cap = BUCKETS
            .iter()
            .copied()
            .find(|&b| b >= max_batch)
            .unwrap_or(MAX_BATCH);
        let (t, d, d_ff) = (arch.tokens, arch.d_model, arch.d_ff);
        let g = gemm.bytes();
        let act = Activations {
            boards: buffer(&device, cap * BOARD_BYTES)?,
            x: buffer(&device, cap * t * d * 4)?,
            h: buffer(&device, cap * t * d * g)?,
            hf: buffer(&device, cap * t * d * 4)?,
            qkv: buffer(&device, cap * t * 3 * d * g)?,
            attn: buffer(&device, cap * t * d * g)?,
            proj: buffer(&device, cap * t * d * g)?,
            ff: buffer(&device, cap * t * d_ff * g)?,
            squares: buffer(&device, cap * 64 * d * g)?,
            policy16: buffer(&device, cap * 64 * POLICY_PLANES * g)?,
            policy: buffer(&device, cap * 64 * POLICY_PLANES * 4)?,
            vnorm: buffer(&device, cap * d * g)?,
            hidden: buffer(&device, cap * d * g)?,
            value16: buffer(&device, cap * WDL_OUTPUTS * g)?,
            value: buffer(&device, cap * WDL_OUTPUTS * 4)?,
        };
        Ok(Self {
            arch,
            gemm,
            device,
            pipelines,
            weights: dev_weights,
            state: Mutex::new(State {
                queue,
                act,
                matmuls: HashMap::new(),
            }),
            max_batch: cap,
            device_name,
        })
    }

    fn shape(&self, positions: usize) -> Shape {
        Shape {
            positions: positions as u32,
            tokens: self.arch.tokens as u32,
            d: self.arch.d_model as u32,
            heads: self.arch.n_heads as u32,
            head_dim: self.arch.head_dim() as u32,
            square_start: self.arch.square_token_start as u32,
            d_ff: self.arch.d_ff as u32,
            eps: self.arch.layer_norm_eps,
            scale: 1.0 / (self.arch.head_dim() as f32).sqrt(),
        }
    }

    /// Raw policy and value logits for `boards` (`[n·64·73]`, `[n·3]`), as the CPU backend.
    pub fn logits(&self, boards: &[&EncodedBoard]) -> Result<(Vec<f32>, Vec<f32>), MetalError> {
        let n = boards.len();
        if n == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        // The activation buffers were sized for `max_batch` at load; a larger batch is a
        // caller error reported like any other, as the CUDA backend does.
        if n > self.max_batch {
            return Err(MetalError::BatchTooLarge {
                got: n,
                max: self.max_batch,
            });
        }
        let bucket = BUCKETS
            .iter()
            .copied()
            .find(|&b| b >= n)
            .unwrap_or(MAX_BATCH);
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        // SAFETY: shared-storage buffer of `max_batch · BOARD_BYTES`; we write `bucket` rows.
        unsafe {
            let dst = state.act.boards.contents().as_ptr().cast::<u8>();
            std::ptr::write_bytes(dst, 0, bucket * BOARD_BYTES);
            for (p, board) in boards.iter().enumerate() {
                let row = dst.add(p * BOARD_BYTES);
                std::ptr::copy_nonoverlapping(board.pieces.as_ptr(), row, 64);
                *row.add(64) = board.castling;
                *row.add(65) = board.en_passant;
            }
        }
        self.forward(&mut state, bucket)?;
        let policy_len = n * 64 * POLICY_PLANES;
        let value_len = n * WDL_OUTPUTS;
        // SAFETY: the buffers hold at least `bucket` positions' outputs, written by the
        // command buffer that has completed.
        let (policy, value) = unsafe {
            (
                std::slice::from_raw_parts(
                    state.act.policy.contents().as_ptr().cast::<f32>(),
                    policy_len,
                )
                .to_vec(),
                std::slice::from_raw_parts(
                    state.act.value.contents().as_ptr().cast::<f32>(),
                    value_len,
                )
                .to_vec(),
            )
        };
        Ok((policy, value))
    }

    fn forward(&self, state: &mut State, positions: usize) -> Result<(), MetalError> {
        let (t, d, d_ff) = (self.arch.tokens, self.arch.d_model, self.arch.d_ff);
        let rows = positions * t;
        let shape = self.shape(positions);
        let w = &self.weights;
        let cb = state
            .queue
            .commandBuffer()
            .ok_or_else(|| MetalError::Metal("cannot create a command buffer".into()))?;
        let State { act, matmuls, .. } = state;
        let matmuls = std::cell::RefCell::new(matmuls);

        // A compute encoder per stretch of kernels; MPS encodes its own between them.
        let encode = |f: &dyn Fn(&ProtocolObject<dyn MTLComputeCommandEncoder>)| {
            let enc = cb.computeCommandEncoder().expect("compute encoder");
            f(&enc);
            enc.endEncoding();
        };
        let gemm = |a: &Buffer, lin: &DevLinear, out: &Buffer, rows: usize| {
            self.gemm(&mut matmuls.borrow_mut(), &cb, a, lin, out, rows);
        };

        encode(&|enc| {
            self.dispatch(enc, &self.pipelines.embed, rows, |enc| {
                set_buffers(
                    enc,
                    &[
                        &act.boards,
                        &act.x,
                        &w.piece_embed,
                        &w.square_embed,
                        &w.state_embed,
                        &w.castling_w,
                        &w.castling_b,
                        &w.en_passant_embed,
                    ],
                );
                set_bytes(enc, 8, &shape);
            });
        });

        for layer in &w.layers {
            encode(&|enc| {
                self.layer_norm(enc, &act.x, &act.h, &layer.norm1, &shape, t, 0, positions)
            });
            gemm(&act.h, &layer.in_proj, &act.qkv, rows);
            encode(&|enc| {
                self.elementwise(
                    enc,
                    &self.pipelines.add_bias4,
                    &[&act.qkv, &layer.in_proj.bias],
                    3 * d,
                    rows * 3 * d,
                    None,
                );
                // One threadgroup per (position, head), a thread per query token.
                enc.setComputePipelineState(&self.pipelines.attention);
                set_buffers(enc, &[&act.qkv, &act.attn]);
                set_bytes(enc, 2, &shape);
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize {
                        width: (positions * self.arch.n_heads) as NSUInteger,
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: t as NSUInteger,
                        height: 1,
                        depth: 1,
                    },
                );
            });
            gemm(&act.attn, &layer.out_proj, &act.proj, rows);
            encode(&|enc| {
                self.elementwise(
                    enc,
                    &self.pipelines.add_bias_residual,
                    &[&act.x, &act.proj, &layer.out_proj.bias],
                    d,
                    rows * d,
                    None,
                );
                self.layer_norm(enc, &act.x, &act.h, &layer.norm2, &shape, t, 0, positions);
            });
            gemm(&act.h, &layer.linear1, &act.ff, rows);
            encode(&|enc| {
                let tanh = u32::from(self.arch.gelu == Gelu::Tanh);
                self.elementwise(
                    enc,
                    &self.pipelines.bias_gelu,
                    &[&act.ff, &layer.linear1.bias],
                    d_ff,
                    rows * d_ff,
                    Some(tanh),
                );
            });
            gemm(&act.ff, &layer.linear2, &act.proj, rows);
            encode(&|enc| {
                self.elementwise(
                    enc,
                    &self.pipelines.add_bias_residual,
                    &[&act.x, &act.proj, &layer.linear2.bias],
                    d,
                    rows * d,
                    None,
                );
            });
        }

        // Heads: the final norm over every token, then the square tokens through the
        // policy norm and the state token through the value norm.
        encode(&|enc| {
            self.layer_norm_f32(enc, &act.x, &act.hf, &w.final_norm, &shape, t, positions);
            self.layer_norm(
                enc,
                &act.hf,
                &act.squares,
                &w.policy_norm,
                &shape,
                64,
                self.arch.square_token_start,
                positions,
            );
            self.layer_norm(
                enc,
                &act.hf,
                &act.vnorm,
                &w.value_norm,
                &shape,
                1,
                0,
                positions,
            );
        });
        gemm(&act.squares, &w.policy_proj, &act.policy16, positions * 64);
        gemm(&act.vnorm, &w.value_ff1, &act.hidden, positions);
        encode(&|enc| {
            self.elementwise(
                enc,
                &self.pipelines.add_bias,
                &[&act.policy16, &w.policy_proj.bias, &act.policy],
                POLICY_PLANES,
                positions * 64 * POLICY_PLANES,
                None,
            );
            // The value head's GELU is the exact form whatever the encoder uses.
            self.elementwise(
                enc,
                &self.pipelines.bias_gelu,
                &[&act.hidden, &w.value_ff1.bias],
                d,
                positions * d,
                Some(0),
            );
        });
        gemm(&act.hidden, &w.value_ff2, &act.value16, positions);
        encode(&|enc| {
            self.elementwise(
                enc,
                &self.pipelines.add_bias,
                &[&act.value16, &w.value_ff2.bias, &act.value],
                WDL_OUTPUTS,
                positions * WDL_OUTPUTS,
                None,
            );
        });

        cb.commit();
        cb.waitUntilCompleted();
        // A faulted command buffer leaves the output buffers holding the previous batch;
        // nothing is ever substituted for the network's answer.
        if cb.status() == MTLCommandBufferStatus::Error {
            let error = cb
                .error()
                .map(|e| e.localizedDescription().to_string())
                .unwrap_or_else(|| "unknown error".into());
            return Err(MetalError::Gpu(error));
        }
        Ok(())
    }

    /// `out[rows × lin.out] = a[rows × lin.inp] · Wᵀ`, the multiplication cached per shape.
    fn gemm(
        &self,
        matmuls: &mut HashMap<(usize, usize, usize), Retained<MPSMatrixMultiplication>>,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
        a: &Buffer,
        lin: &DevLinear,
        out: &Buffer,
        rows: usize,
    ) {
        let key = (rows, lin.out, lin.inp);
        let mm = matmuls.entry(key).or_insert_with(|| {
            // SAFETY: plain object construction with the dimensions the encode calls honour.
            unsafe {
                MPSMatrixMultiplication::initWithDevice_transposeLeft_transposeRight_resultRows_resultColumns_interiorColumns_alpha_beta(
                    MPSMatrixMultiplication::alloc(),
                    &self.device,
                    false,
                    true,
                    rows as NSUInteger,
                    lin.out as NSUInteger,
                    lin.inp as NSUInteger,
                    1.0,
                    0.0,
                )
            }
        });
        let left = matrix(a, rows, lin.inp, self.gemm);
        let result = matrix(out, rows, lin.out, self.gemm);
        // SAFETY: the matrices lie within their buffers (sized for MAX_BATCH rows).
        unsafe {
            mm.encodeToCommandBuffer_leftMatrix_rightMatrix_resultMatrix(
                cb,
                &left,
                &lin.matrix,
                &result,
            )
        };
    }

    fn dispatch(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        pso: &Pipeline,
        threads: usize,
        bind: impl Fn(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    ) {
        enc.setComputePipelineState(pso);
        bind(enc);
        let group = pso
            .maxTotalThreadsPerThreadgroup()
            .min(threads.max(1))
            .min(256);
        enc.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: threads.max(1) as NSUInteger,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: group as NSUInteger,
                height: 1,
                depth: 1,
            },
        );
    }

    /// An element-wise kernel: buffers, then `cols`, `count`, and an optional flag.
    fn elementwise(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        pso: &Pipeline,
        buffers: &[&Buffer],
        cols: usize,
        count: usize,
        flag: Option<u32>,
    ) {
        // Every kernel but the scalar `add_bias` works in float4s.
        let threads = if std::ptr::eq(&**pso, &*self.pipelines.add_bias) {
            count
        } else {
            count / 4
        };
        self.dispatch(enc, pso, threads, |enc| {
            set_buffers(enc, buffers);
            let base = buffers.len();
            set_bytes(enc, base, &(cols as u32));
            set_bytes(enc, base + 1, &(count as u32));
            if let Some(flag) = flag {
                set_bytes(enc, base + 2, &flag);
            }
        });
    }

    /// The final norm: fp32 out, over every token, for the head norms to read.
    #[allow(clippy::too_many_arguments)]
    fn layer_norm_f32(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &Buffer,
        out: &Buffer,
        norm: &DevNorm,
        shape: &Shape,
        per_pos: usize,
        positions: usize,
    ) {
        self.dispatch(
            enc,
            &self.pipelines.layer_norm_f32,
            positions * per_pos * 32,
            |enc| {
                set_buffers(enc, &[input, out, &norm.gamma, &norm.beta]);
                set_bytes(enc, 4, shape);
                set_bytes(enc, 5, &(per_pos as u32));
                set_bytes(enc, 6, &0u32);
            },
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn layer_norm(
        &self,
        enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
        input: &Buffer,
        out: &Buffer,
        norm: &DevNorm,
        shape: &Shape,
        per_pos: usize,
        start: usize,
        positions: usize,
    ) {
        // One SIMD group of 32 lanes per row.
        self.dispatch(
            enc,
            &self.pipelines.layer_norm,
            positions * per_pos * 32,
            |enc| {
                set_buffers(enc, &[input, out, &norm.gamma, &norm.beta]);
                set_bytes(enc, 4, shape);
                set_bytes(enc, 5, &(per_pos as u32));
                set_bytes(enc, 6, &(start as u32));
            },
        );
    }
}

fn set_buffers(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, buffers: &[&Buffer]) {
    for (i, b) in buffers.iter().enumerate() {
        // SAFETY: binding a live buffer at an argument index the kernel declares.
        unsafe { enc.setBuffer_offset_atIndex(Some(b), 0, i as NSUInteger) };
    }
}

fn set_bytes<T: Copy>(enc: &ProtocolObject<dyn MTLComputeCommandEncoder>, index: usize, value: &T) {
    // SAFETY: `value` is a plain-old-data struct copied into the command stream.
    unsafe {
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::from(value).cast(),
            std::mem::size_of::<T>() as NSUInteger,
            index as NSUInteger,
        );
    }
}

impl Network for MetalNetwork {
    fn evaluate(&self, batch: &[Leaf<'_>]) -> Result<Vec<Evaluation>, NetworkError> {
        let boards: Vec<&EncodedBoard> = batch.iter().map(|(board, _)| *board).collect();
        let (policy, value) = self
            .logits(&boards)
            .map_err(|e| NetworkError(e.to_string()))?;
        let actions = 64 * PLANES_PER_SQUARE;
        Ok(batch
            .iter()
            .enumerate()
            .map(|(p, (_, legal))| {
                let logits = &policy[p * actions..(p + 1) * actions];
                let mut priors: Vec<f32> = legal.moves.iter().map(|&(_, a)| logits[a]).collect();
                super::cpu::softmax(&mut priors);
                let mut wdl = [value[p * 3], value[p * 3 + 1], value[p * 3 + 2]];
                super::cpu::softmax(&mut wdl);
                Evaluation { priors, wdl }
            })
            .collect())
    }

    fn describe(&self) -> String {
        format!(
            "metal {} ({}): {}",
            if self.gemm.fp16 { "fp16" } else { "fp32" },
            self.device_name,
            self.arch.describe()
        )
    }
}
