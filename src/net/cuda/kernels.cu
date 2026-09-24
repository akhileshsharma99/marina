// Kernels for the CUDA backend. Everything the network needs besides the GEMMs (the four
// linear layers and the two batched attention products per layer), which cuBLASLt runs.
// Activations and the residual stream are fp16; arithmetic inside a kernel is fp32.
// Compiled to PTX by build.rs.
//
// Layout: a batch of B positions is B*t token rows of d columns (t = 65 in board mode:
// one state token, 64 squares). Row r = pos*t + i.

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

// Must match the Rust side: encoding::board (the 66-byte board, PIECE_VOCAB, EN_PASSANT_VOCAB)
// and net::model::POLICY_PLANES.
#define BOARD_BYTES 66  // 64 piece tokens, castling bits, en passant
#define PLANES 73

__device__ __forceinline__ float h2f(__half h) { return __half2float(h); }
__device__ __forceinline__ __half f2h(float f) { return __float2half(f); }
// FP8 e4m3 with a per-tensor dequantisation scale s (value = code * s): quantise as v / s,
// saturating at the format's 448. `s` lives in device memory so the same graph serves
// every scale.
__device__ __forceinline__ uint8_t f2e4m3(float v, float inv_scale) {
    return __nv_cvt_float_to_fp8(v * inv_scale, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

// Sum over the block; `red` is shared scratch of 32 floats. All threads get the result.
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    int warps = (blockDim.x + 31) >> 5;
    float total = (threadIdx.x < warps) ? red[threadIdx.x] : 0.f;
    if (warp == 0) total = warp_sum(total);
    if (threadIdx.x == 0) red[0] = total;
    __syncthreads();
    return red[0];
}

__device__ __forceinline__ float block_max(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    int warps = (blockDim.x + 31) >> 5;
    float total = (threadIdx.x < warps) ? red[threadIdx.x] : -1e30f;
    if (warp == 0)
        for (int o = 16; o > 0; o >>= 1) total = fmaxf(total, __shfl_xor_sync(0xffffffff, total, o));
    if (threadIdx.x == 0) red[0] = total;
    __syncthreads();
    return red[0];
}

// Exact GELU as torch.nn.GELU(): 0.5 x (1 + erf(x / sqrt 2)).
__device__ __forceinline__ float gelu_exact(float x) {
    return 0.5f * x * (1.f + erff(x * 0.70710678118654752f));
}

// ---- tokens -------------------------------------------------------------------------
// One block per token row. boards: [B, BOARD_BYTES] uint8.
extern "C" __global__ void embed_tokens(
    const uint8_t* __restrict__ boards,
    const __half* __restrict__ piece_embed,   // [13, d]
    const __half* __restrict__ square_embed,  // [64, d]
    const __half* __restrict__ state_embed,   // [d]
    const __half* __restrict__ castling_w,    // [d, 4]
    const __half* __restrict__ castling_b,    // [d]
    const __half* __restrict__ ep_embed,      // [9, d]
    __half* __restrict__ x,                   // [B*t, d] the residual stream, fp16
    int d, int t, int square_start)
{
    int row = blockIdx.x;
    int pos = row / t, i = row % t;
    const uint8_t* b = boards + (size_t)pos * BOARD_BYTES;
    __half* out = x + (size_t)row * d;
    if (i == 0) {
        float bit0 = (b[64] >> 0) & 1, bit1 = (b[64] >> 1) & 1;
        float bit2 = (b[64] >> 2) & 1, bit3 = (b[64] >> 3) & 1;
        int ep = b[65] > 8 ? 8 : b[65];
        for (int c = threadIdx.x; c < d; c += blockDim.x) {
            const __half* cw = castling_w + (size_t)c * 4;
            out[c] = f2h(h2f(state_embed[c]) + h2f(castling_b[c])
                   + h2f(cw[0]) * bit0 + h2f(cw[1]) * bit1 + h2f(cw[2]) * bit2 + h2f(cw[3]) * bit3
                   + h2f(ep_embed[(size_t)ep * d + c]));
        }
    } else if (i >= square_start) {
        int s = i - square_start;
        int piece = b[s];
        for (int c = threadIdx.x; c < d; c += blockDim.x)
            out[c] = f2h(h2f(piece_embed[(size_t)piece * d + c]) + h2f(square_embed[(size_t)s * d + c]));
    }
}

// ---- layer norm ---------------------------------------------------------------------
// One block per row: out[row] = LN(x[row]). The residual adds happen in the GEMM epilogues
// (β = 1 into the fp16 stream), so this is a pure normalisation pass: one read, one write.
extern "C" __global__ void layernorm(
    const __half* __restrict__ x,
    __half* __restrict__ out,
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    int d, float eps)
{
    __shared__ float red[32];
    int row = blockIdx.x;
    const __half* xr = x + (size_t)row * d;
    float local = 0.f;
    for (int c = threadIdx.x; c < d; c += blockDim.x) local += h2f(xr[c]);
    float mean = block_sum(local, red) / d;
    float var_local = 0.f;
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        float dv = h2f(xr[c]) - mean;
        var_local += dv * dv;
    }
    float inv = rsqrtf(block_sum(var_local, red) / d + eps);
    __half* o = out + (size_t)row * d;
    for (int c = threadIdx.x; c < d; c += blockDim.x)
        o[c] = f2h((h2f(xr[c]) - mean) * inv * h2f(gamma[c]) + h2f(beta[c]));
}

// The same, quantised to e4m3 for an FP8 GEMM. `scale` points at the tensor's
// dequantisation scale.
extern "C" __global__ void layernorm_e4m3(
    const __half* __restrict__ x,
    uint8_t* __restrict__ out,
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    int d, float eps, const float* __restrict__ scale)
{
    __shared__ float red[32];
    int row = blockIdx.x;
    const __half* xr = x + (size_t)row * d;
    float local = 0.f;
    for (int c = threadIdx.x; c < d; c += blockDim.x) local += h2f(xr[c]);
    float mean = block_sum(local, red) / d;
    float var_local = 0.f;
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        float dv = h2f(xr[c]) - mean;
        var_local += dv * dv;
    }
    float inv = rsqrtf(block_sum(var_local, red) / d + eps);
    float inv_scale = 1.f / *scale;
    uint8_t* o = out + (size_t)row * d;
    for (int c = threadIdx.x; c < d; c += blockDim.x)
        o[c] = f2e4m3((h2f(xr[c]) - mean) * inv * h2f(gamma[c]) + h2f(beta[c]), inv_scale);
}

// Layer norm of the 64 square tokens of each position, fp16 in, fp16 out, compacted to
// [B*64, d] for the policy head. One block per (pos, square).
extern "C" __global__ void square_layernorm(
    const __half* __restrict__ h,       // [B*t, d]
    __half* __restrict__ out,           // [B*64, d]
    const __half* __restrict__ gamma,
    const __half* __restrict__ beta,
    int d, int t, int square_start, float eps)
{
    __shared__ float red[32];
    int idx = blockIdx.x;                // pos*64 + s
    int pos = idx >> 6, s = idx & 63;
    const __half* row = h + ((size_t)pos * t + square_start + s) * d;
    float local = 0.f;
    for (int c = threadIdx.x; c < d; c += blockDim.x) local += h2f(row[c]);
    float mean = block_sum(local, red) / d;
    float var_local = 0.f;
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        float dv = h2f(row[c]) - mean;
        var_local += dv * dv;
    }
    float inv = rsqrtf(block_sum(var_local, red) / d + eps);
    __half* o = out + (size_t)idx * d;
    for (int c = threadIdx.x; c < d; c += blockDim.x)
        o[c] = f2h((h2f(row[c]) - mean) * inv * h2f(gamma[c]) + h2f(beta[c]));
}

// ---- attention --------------------------------------------------------------------------
// One block of 8 warps per (position, head), on the tensor cores through WMMA (16x16x16
// fp16 fragments, fp32 accumulation). Q, K, V for the head are staged in shared memory as
// fp16 with rows padded to a multiple of 16 (zero rows past t). The queries are then
// processed 16 rows at a time so only a 16-row strip of scores is ever in shared memory:
// S = Q·Kᵀ for the strip, one warp per row turns it into normalised fp16 probabilities in
// place, O = P·V goes through the tensor cores into the same strip, and the strip is
// written out as fp16. The head dimension is a template parameter; one kernel per
// supported size below. blockDim.x = ATT_THREADS; launch with the dynamic shared memory
// the Rust side computes (3 padded fp16 matrices plus the fp32 strip).
#include <mma.h>
using namespace nvcuda;
#define ATT_THREADS 256
#define ATT_WARPS (ATT_THREADS / 32)
// Leading dimensions in elements: multiples of 8 halves / 4 floats as WMMA requires, padded
// by 8 to spread the fragment loads over the banks. The strip is 16 x (rows + 8) floats.
#define ATT_LD_QKV(HD) ((HD) + 8)
__device__ __forceinline__ int att_rows(int t) { return (t + 15) & ~15; }
template <int HD, bool E4M3>
__device__ __forceinline__ void attention_impl(
    const __half* __restrict__ qkv,   // [B*t, 3*d]: Q | K | V, head h at columns h*HD
    void* __restrict__ out,           // [B*t, d] fp16, or e4m3 when E4M3, head h at columns h*HD
    int t, int d, int heads, float scale,
    const float* __restrict__ out_scale)  // e4m3 dequantisation scale (E4M3 only)
{
    float inv_out_scale = E4M3 ? 1.f / *out_scale : 1.f;
    constexpr int LDQ = ATT_LD_QKV(HD);
    constexpr int OT = HD / 16;       // output tiles per 16-row strip
    extern __shared__ __align__(32) unsigned char smem_raw[];
    int rows = att_rows(t);
    int lds = rows + 8;
    __half* qs = reinterpret_cast<__half*>(smem_raw);
    __half* ks = qs + rows * LDQ;
    __half* vs = ks + rows * LDQ;
    float* sc = reinterpret_cast<float*>(vs + rows * LDQ);   // [16, lds] fp32 strip
    __half* ps = reinterpret_cast<__half*>(sc);               // P overlays S row for row
    int ldp = 2 * lds;

    int pos = blockIdx.x / heads;
    int head = blockIdx.x - pos * heads;
    size_t base = (size_t)pos * t * (3 * d);
    int stride = 3 * d;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    // Stage Q, K, V (rows < t) with 16-byte loads and zero the padding rows.
    constexpr int CH = HD / 8;
    for (int i = tid; i < rows * CH; i += ATT_THREADS) {
        int row = i / CH, ch = i - row * CH;
        int4 q4 = make_int4(0, 0, 0, 0), k4 = q4, v4 = q4;
        if (row < t) {
            const __half* r = qkv + base + (size_t)row * stride + head * HD;
            q4 = reinterpret_cast<const int4*>(r)[ch];
            k4 = reinterpret_cast<const int4*>(r + d)[ch];
            v4 = reinterpret_cast<const int4*>(r + 2 * d)[ch];
        }
        reinterpret_cast<int4*>(qs + row * LDQ)[ch] = q4;
        reinterpret_cast<int4*>(ks + row * LDQ)[ch] = k4;
        reinterpret_cast<int4*>(vs + row * LDQ)[ch] = v4;
    }
    __syncthreads();

    int tiles = rows / 16;
    for (int ti = 0; ti < tiles; ++ti) {
        // S strip = Q[ti] · Kᵀ: one 16x16 tile per key tile, spread across the warps.
        for (int tj = warp; tj < tiles; tj += ATT_WARPS) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
            wmma::fill_fragment(c, 0.f);
            #pragma unroll
            for (int k = 0; k < HD; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
                wmma::load_matrix_sync(a, qs + ti * 16 * LDQ + k, LDQ);
                // B = Kᵀ: element (k, j) is K[j][k], i.e. K row-major read as column-major.
                wmma::load_matrix_sync(b, ks + tj * 16 * LDQ + k, LDQ);
                wmma::mma_sync(c, a, b, c);
            }
            wmma::store_matrix_sync(sc + tj * 16, c, lds, wmma::mem_row_major);
        }
        __syncthreads();

        // Softmax, one warp per row with the lanes along it, in place: fp32 scores ->
        // normalised fp16 probabilities; rows past t and columns past t become zero so the
        // padded P·V adds nothing. Every lane reads its values before any lane writes.
        for (int rr = warp; rr < 16; rr += ATT_WARPS) {
            int r = ti * 16 + rr;
            float* srow = sc + rr * lds;
            __half* prow = ps + rr * ldp;
            if (r < t) {
                float v[3];
                float m = -1e30f;
                #pragma unroll
                for (int k = 0; k < 3; ++k) {
                    int j = lane + 32 * k;
                    v[k] = j < t ? srow[j] : -1e30f;
                    m = fmaxf(m, v[k]);
                }
                for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
                float l = 0.f;
                #pragma unroll
                for (int k = 0; k < 3; ++k) {
                    int j = lane + 32 * k;
                    v[k] = j < t ? __expf((v[k] - m) * scale) : 0.f;
                    l += v[k];
                }
                l = warp_sum(l);
                float inv = 1.f / l;
                __syncwarp();
                #pragma unroll
                for (int k = 0; k < 3; ++k) {
                    int j = lane + 32 * k;
                    if (j < rows) prow[j] = f2h(v[k] * inv);
                }
            } else {
                for (int j = lane; j < rows; j += 32) prow[j] = f2h(0.f);
            }
        }
        __syncthreads();

        // O strip = P · V, one 16x16 tile per 16 output features. The accumulators are
        // held until every warp is done reading P, then stored into the strip.
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> o;
        int tc = warp;
        if (tc < OT) {
            wmma::fill_fragment(o, 0.f);
            for (int k = 0; k < rows; k += 16) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
                wmma::load_matrix_sync(a, ps + k, ldp);
                wmma::load_matrix_sync(b, vs + k * LDQ + tc * 16, LDQ);
                wmma::mma_sync(o, a, b, o);
            }
        }
        __syncthreads();
        if (tc < OT) {
            wmma::store_matrix_sync(sc + tc * 16, o, lds, wmma::mem_row_major);
        }
        __syncthreads();
        // Strip out: 16 rows x HD values, 8 per chunk (16 bytes as fp16, 8 as e4m3).
        for (int i = tid; i < 16 * CH; i += ATT_THREADS) {
            int rr = i / CH, ch = i - rr * CH;
            int row = ti * 16 + rr;
            if (row < t) {
                const float* src = sc + rr * lds + ch * 8;
                size_t at = ((size_t)pos * t + row) * d + head * HD + ch * 8;
                if (E4M3) {
                    uint8_t q[8];
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) q[k] = f2e4m3(src[k], inv_out_scale);
                    *reinterpret_cast<uint2*>(static_cast<uint8_t*>(out) + at) = *reinterpret_cast<uint2*>(q);
                } else {
                    __half2 h[4];
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) h[k] = __floats2half2_rn(src[2 * k], src[2 * k + 1]);
                    *reinterpret_cast<int4*>(static_cast<__half*>(out) + at) = *reinterpret_cast<int4*>(h);
                }
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void attention_hd32(
    const __half* __restrict__ qkv, __half* __restrict__ out, int t, int d, int heads, float scale)
{
    attention_impl<32, false>(qkv, out, t, d, heads, scale, nullptr);
}

extern "C" __global__ void attention_hd64(
    const __half* __restrict__ qkv, __half* __restrict__ out, int t, int d, int heads, float scale)
{
    attention_impl<64, false>(qkv, out, t, d, heads, scale, nullptr);
}

extern "C" __global__ void attention_hd32_e4m3(
    const __half* __restrict__ qkv, uint8_t* __restrict__ out, int t, int d, int heads, float scale,
    const float* __restrict__ out_scale)
{
    attention_impl<32, true>(qkv, out, t, d, heads, scale, out_scale);
}

extern "C" __global__ void attention_hd64_e4m3(
    const __half* __restrict__ qkv, uint8_t* __restrict__ out, int t, int d, int heads, float scale,
    const float* __restrict__ out_scale)
{
    attention_impl<64, true>(qkv, out, t, d, heads, scale, out_scale);
}

// ---- GELU in place: exact, or the tanh form when `tanh_form` (networks trained with it
// get cuBLASLt's fused epilogue instead in fp16 builds; FP8 GEMMs cannot carry it) ------
__device__ __forceinline__ float gelu_tanh_form(float x) {
    return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
extern "C" __global__ void gelu_inplace(__half* __restrict__ v, int n, int tanh_form) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = h2f(v[i]);
        v[i] = f2h(tanh_form ? gelu_tanh_form(x) : gelu_exact(x));
    }
}

// The same from fp16 into e4m3 for the FP8 FFN2.
extern "C" __global__ void gelu_e4m3(
    const __half* __restrict__ v, uint8_t* __restrict__ out, int n, int tanh_form,
    const float* __restrict__ scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = h2f(v[i]);
        out[i] = f2e4m3(tanh_form ? gelu_tanh_form(x) : gelu_exact(x), 1.f / *scale);
    }
}

// ---- weights to e4m3 at load, one dequantisation scale per matrix ---------------------
extern "C" __global__ void quantize_e4m3(
    const __half* __restrict__ w, uint8_t* __restrict__ out, int n, const float* __restrict__ scale)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f2e4m3(h2f(w[i]), 1.f / *scale);
}

// ---- policy: logits for the legal actions only, then softmax over them ----------------
// One block per position. legal: [B, maxl] action indices, -1 padded. Each thread owns
// one legal action: its logit is a d-long dot product of the square's normed token with
// the plane's weight row. Writes priors [B, maxl] in the same order, 0 on padding.
extern "C" __global__ void policy_legal(
    const __half* __restrict__ squares,   // [B*64, d] policy-normed square tokens
    const __half* __restrict__ wp,        // [73, d]
    const __half* __restrict__ bp,        // [73]
    const int32_t* __restrict__ legal,
    float* __restrict__ priors,
    int d, int maxl)
{
    extern __shared__ float logit[];       // maxl floats, then 32 for reductions
    float* red = logit + maxl;
    int pos = blockIdx.x;
    for (int a = threadIdx.x; a < maxl; a += blockDim.x) {
        int action = legal[(size_t)pos * maxl + a];
        float v = -1e30f;
        if (action >= 0) {
            int square = action / PLANES, plane = action % PLANES;
            const __half* tok = squares + ((size_t)pos * 64 + square) * d;
            const __half* w = wp + (size_t)plane * d;
            float acc = h2f(bp[plane]);
            for (int c = 0; c < d; ++c) acc += h2f(tok[c]) * h2f(w[c]);
            v = acc;
        }
        logit[a] = v;
    }
    __syncthreads();
    float mx_local = -1e30f;
    for (int a = threadIdx.x; a < maxl; a += blockDim.x) mx_local = fmaxf(mx_local, logit[a]);
    float mx = block_max(mx_local, red);
    float sum_local = 0.f;
    for (int a = threadIdx.x; a < maxl; a += blockDim.x) {
        float e = logit[a] > -1e29f ? __expf(logit[a] - mx) : 0.f;
        logit[a] = e;
        sum_local += e;
    }
    float inv = 1.f / fmaxf(block_sum(sum_local, red), 1e-30f);
    for (int a = threadIdx.x; a < maxl; a += blockDim.x)
        priors[(size_t)pos * maxl + a] = logit[a] * inv;
}

// ---- value head: LN(state) -> Linear -> exact GELU -> Linear -> softmax ---------------
// One block per position, blockDim.x >= 32, shared: 2*d floats + 32.
extern "C" __global__ void value_head(
    const __half* __restrict__ h,        // [B*t, d] final-normed tokens; state token at row pos*t
    const __half* __restrict__ gamma, const __half* __restrict__ beta,   // value norm
    const __half* __restrict__ w1, const __half* __restrict__ b1,        // [d, d], [d]
    const __half* __restrict__ w2, const __half* __restrict__ b2,        // [3, d], [3]
    float* __restrict__ wdl,             // [B, 3]
    int t, int d, float eps)
{
    extern __shared__ float sm[];
    float* v = sm;            // d
    float* hid = sm + d;      // d
    float* red = sm + 2 * d;  // 32
    int pos = blockIdx.x;
    const __half* row = h + (size_t)pos * t * d;
    float local = 0.f;
    for (int c = threadIdx.x; c < d; c += blockDim.x) { v[c] = h2f(row[c]); local += v[c]; }
    float mean = block_sum(local, red) / d;
    float var_local = 0.f;
    for (int c = threadIdx.x; c < d; c += blockDim.x) { float dv = v[c] - mean; var_local += dv * dv; }
    float inv = rsqrtf(block_sum(var_local, red) / d + eps);
    for (int c = threadIdx.x; c < d; c += blockDim.x)
        v[c] = (v[c] - mean) * inv * h2f(gamma[c]) + h2f(beta[c]);
    __syncthreads();
    for (int o = threadIdx.x; o < d; o += blockDim.x) {
        const __half* w = w1 + (size_t)o * d;
        float acc = h2f(b1[o]);
        for (int c = 0; c < d; ++c) acc += h2f(w[c]) * v[c];
        hid[o] = gelu_exact(acc);
    }
    __syncthreads();
    __shared__ float logits[3];
    if (threadIdx.x < 3) {
        const __half* w = w2 + (size_t)threadIdx.x * d;
        float acc = h2f(b2[threadIdx.x]);
        for (int c = 0; c < d; ++c) acc += h2f(w[c]) * hid[c];
        logits[threadIdx.x] = acc;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float mx = fmaxf(logits[0], fmaxf(logits[1], logits[2]));
        float e0 = __expf(logits[0] - mx), e1 = __expf(logits[1] - mx), e2 = __expf(logits[2] - mx);
        float s = e0 + e1 + e2;
        wdl[pos * 3 + 0] = e0 / s;
        wdl[pos * 3 + 1] = e1 / s;
        wdl[pos * 3 + 2] = e2 / s;
    }
}
