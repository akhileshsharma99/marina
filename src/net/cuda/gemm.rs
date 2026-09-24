//! cuBLASLt plans for the linear layers: descriptors and the chosen algorithm built once
//! per shape, so a forward pass only launches. (cudarc's safe `matmul` rebuilds the
//! descriptors and reruns the heuristic on every call, which costs host time and cannot be
//! captured cleanly.)
//!
//! Every plan is the same operation, `out[rows, n] = a[rows, k] · Wᵀ + bias` in fp16 with
//! fp32 accumulation, or with `residual` the in-place `x[rows, n] += a · Wᵀ + bias` (β = 1
//! with C = D = x, one fused kernel, since A, B, C and D are all fp16; the fp32-residual
//! variant of this split into two). cuBLASLt is column-major, so it is expressed as the
//! column-major `D[n, rows] = Wᵀ(trans of W-as-stored [k, n]) · A-as-stored [k, rows]`, with
//! the bias (one value per output feature, i.e. per row of `D`) added by the epilogue,
//! optionally followed by cuBLASLt's fused GELU (FFN1, for networks trained with the tanh
//! form).
//!
//! With [`Fp8`] the same plan runs on e4m3 tensor cores (Ada and newer): A (the weights)
//! and B (the activation) are e4m3 with per-tensor dequantisation scales read from device
//! memory, accumulation stays fp32, C and D are fp16 as before. The layouts are already
//! the TN form FP8 requires. (cuBLASLt on Ada does not combine an FP8 GEMM with the GELU
//! epilogue or an e4m3 D, so FP8 FFN1 writes fp16 and a kernel activates and quantises.)

use std::ffi::c_void;

use cudarc::cublaslt::result::{self, CublasError};
use cudarc::cublaslt::sys;

const CUBLAS_OP_N: i32 = 0;
const CUBLAS_OP_T: i32 = 1;

/// Shape of one linear layer's GEMM.
#[derive(Debug, Clone, Copy)]
pub struct Linear {
    /// Rows of the activation (`positions × tokens`).
    pub rows: u64,
    /// Input features.
    pub k: u64,
    /// Output features.
    pub n: u64,
    /// Device pointer of the bias vector (`n` fp16 values).
    pub bias: u64,
    /// Add the product into the output instead of overwriting it (the residual stream).
    pub residual: bool,
    /// Apply cuBLASLt's GELU (the tanh form) after the bias, inside the GEMM.
    pub gelu: bool,
    /// Run on e4m3 inputs.
    pub fp8: Option<Fp8>,
}

/// FP8 operands: device pointers to fp32 dequantisation scales (`value = code · scale`).
#[derive(Debug, Clone, Copy)]
pub struct Fp8 {
    /// Scale of the e4m3 weight matrix.
    pub a_scale: u64,
    /// Scale of the e4m3 activation.
    pub b_scale: u64,
}

pub struct Plan {
    handle: sys::cublasLtHandle_t,
    desc: sys::cublasLtMatmulDesc_t,
    a: sys::cublasLtMatrixLayout_t,
    b: sys::cublasLtMatrixLayout_t,
    c: sys::cublasLtMatrixLayout_t,
    d: sys::cublasLtMatrixLayout_t,
    algo: sys::cublasLtMatmulAlgo_t,
    workspace: u64,
    workspace_size: usize,
    beta: f32,
}

// Handles are only ever used from one thread at a time (the network's buffer mutex).
unsafe impl Send for Plan {}
unsafe impl Sync for Plan {}

impl Plan {
    /// Build layouts and descriptor for `shape` and pick the fastest algorithm that fits in
    /// the `workspace_size`-byte workspace at device address `workspace`.
    pub fn new(
        handle: sys::cublasLtHandle_t,
        shape: &Linear,
        workspace: u64,
        workspace_size: usize,
    ) -> Result<Self, CublasError> {
        let f16 = sys::cudaDataType_t::CUDA_R_16F;
        let e4m3 = sys::cudaDataType_t::CUDA_R_8F_E4M3;
        let ab = if shape.fp8.is_some() { e4m3 } else { f16 };
        // W as stored is [k, n] column-major (row-major [n, k]); transposed it is op(A) = n×k.
        let a = result::create_matrix_layout(ab, shape.k, shape.n, shape.k as i64)?;
        // The activation as stored is [k, rows] column-major: op(B) = k×rows.
        let b = result::create_matrix_layout(ab, shape.k, shape.rows, shape.k as i64)?;
        // C and D are n×rows column-major, i.e. the row-major [rows, n] output.
        let c = result::create_matrix_layout(f16, shape.n, shape.rows, shape.n as i64)?;
        let d = result::create_matrix_layout(f16, shape.n, shape.rows, shape.n as i64)?;

        let desc = result::create_matmul_desc(
            sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            sys::cudaDataType_t::CUDA_R_32F,
        )?;
        // SAFETY: attribute values are the documented types and sizes.
        unsafe {
            result::set_matmul_desc_attribute(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                (&CUBLAS_OP_T) as *const i32 as *const c_void,
                std::mem::size_of::<i32>(),
            )?;
            result::set_matmul_desc_attribute(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
                (&CUBLAS_OP_N) as *const i32 as *const c_void,
                std::mem::size_of::<i32>(),
            )?;
            let epilogue = if shape.gelu {
                sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS
            } else {
                sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS
            };
            result::set_matmul_desc_attribute(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
                (&epilogue) as *const sys::cublasLtEpilogue_t as *const c_void,
                std::mem::size_of::<sys::cublasLtEpilogue_t>(),
            )?;
            let bias = shape.bias as *const c_void;
            result::set_matmul_desc_attribute(
                desc,
                sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                (&bias) as *const *const c_void as *const c_void,
                std::mem::size_of::<*const c_void>(),
            )?;
            if let Some(fp8) = shape.fp8 {
                let pointer = |attr: sys::cublasLtMatmulDescAttributes_t,
                               value: u64|
                 -> Result<(), CublasError> {
                    let p = value as *const c_void;
                    // SAFETY: as above.
                    #[allow(unused_unsafe)]
                    unsafe {
                        result::set_matmul_desc_attribute(
                            desc,
                            attr,
                            (&p) as *const *const c_void as *const c_void,
                            std::mem::size_of::<*const c_void>(),
                        )
                    }
                };
                pointer(
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                    fp8.a_scale,
                )?;
                pointer(
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                    fp8.b_scale,
                )?;
                // The bias vectors are fp16 on the device whatever the operand type.
                result::set_matmul_desc_attribute(
                    desc,
                    sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                    (&f16) as *const sys::cudaDataType_t as *const c_void,
                    std::mem::size_of::<sys::cudaDataType_t>(),
                )?;
            }
        }

        let pref = result::create_matmul_pref()?;
        // SAFETY: as above; the preference is destroyed right after use.
        let heuristic = unsafe {
            result::set_matmul_pref_attribute(
                pref,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&workspace_size) as *const usize as *const c_void,
                std::mem::size_of::<usize>(),
            )?;
            let h = result::get_matmul_algo_heuristic(handle, desc, a, b, c, d, pref);
            result::destroy_matmul_pref(pref)?;
            h?
        };
        Ok(Self {
            handle,
            desc,
            a,
            b,
            c,
            d,
            algo: heuristic.algo,
            workspace,
            workspace_size,
            beta: if shape.residual { 1.0 } else { 0.0 },
        })
    }

    /// Launch on `stream`: `weight` is W as stored, `input` the activation, `output` the
    /// result (read as well when the plan is a residual one), all device pointers matching
    /// the plan's shape.
    ///
    /// # Safety
    /// The pointers must address buffers of the planned shapes on the plan's device, and
    /// the workspace must not be in use by another matmul on a different stream.
    pub unsafe fn run(
        &self,
        weight: u64,
        input: u64,
        output: u64,
        stream: sys::cudaStream_t,
    ) -> Result<(), CublasError> {
        let (alpha, beta) = (1.0f32, self.beta);
        unsafe {
            result::matmul(
                self.handle,
                self.desc,
                (&alpha) as *const f32 as *const c_void,
                (&beta) as *const f32 as *const c_void,
                weight as *const c_void,
                self.a,
                input as *const c_void,
                self.b,
                output as *const c_void,
                self.c,
                output as *mut c_void,
                self.d,
                &self.algo,
                self.workspace as *mut c_void,
                self.workspace_size,
                stream,
            )
        }
    }
}

impl Drop for Plan {
    fn drop(&mut self) {
        // SAFETY: handles were created by this plan and are not used after drop.
        unsafe {
            let _ = result::destroy_matmul_desc(self.desc);
            let _ = result::destroy_matrix_layout(self.a);
            let _ = result::destroy_matrix_layout(self.b);
            let _ = result::destroy_matrix_layout(self.c);
            let _ = result::destroy_matrix_layout(self.d);
        }
    }
}
