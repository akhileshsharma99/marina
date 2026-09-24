// Metal kernels for everything but the GEMMs (those are MPS). The residual stream x is
// fp32; everything that enters or leaves a GEMM (weights, layer-norm outputs, projections)
// is GEMM_T, half or float, #defined by the host from the `Precision` option before
// compiling; reductions are fp32 either way. Sized for a 65-token, d ≤ 1024 network.
#include <metal_stdlib>
using namespace metal;

struct Shape {
    uint positions;     // batch rows in use (padded to the bucket)
    uint tokens;        // encoder tokens per position
    uint d;             // d_model
    uint heads;
    uint head_dim;
    uint square_start;  // index of the first square token
    uint d_ff;
    float eps;
    float scale;        // 1/sqrt(head_dim)
};

constant uint BOARD_BYTES = 66;   // 64 piece tokens, castling bits, en passant

typedef GEMM_T gemm_t;            // element type of every GEMM operand and result
typedef vec<GEMM_T, 4> gemm4_t;

// Weights arrive as fp32 and are kept as fp16 for the GEMMs; converted once at load.
kernel void f32_to_f16(device const float* in  [[buffer(0)]],
                       device half* out        [[buffer(1)]],
                       constant uint& count    [[buffer(2)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = half(in[gid]);
}

// x[p, i, :] for every (position, token): the state token from the state embedding,
// castling projection and en-passant embedding; square tokens from piece + square
// embeddings. One thread per (position, token).
kernel void embed(device const uchar* boards      [[buffer(0)]],
                  device float* x                 [[buffer(1)]],
                  device const float* piece_embed [[buffer(2)]],
                  device const float* square_embed[[buffer(3)]],
                  device const float* state_embed [[buffer(4)]],
                  device const float* castle_w    [[buffer(5)]],   // [d, 4]
                  device const float* castle_b    [[buffer(6)]],   // [d]
                  device const float* ep_embed    [[buffer(7)]],   // [9, d]
                  constant Shape& s               [[buffer(8)]],
                  uint gid [[thread_position_in_grid]]) {
    uint rows = s.positions * s.tokens;
    if (gid >= rows) return;
    uint p = gid / s.tokens, i = gid % s.tokens;
    device const uchar* board = boards + p * BOARD_BYTES;
    device float* out = x + gid * s.d;
    if (i < s.square_start) {
        // Only the single state token is supported (square_start == 1).
        uchar castling = board[64];
        uint ep = min((uint)board[65], 8u);
        float bits[4] = { float(castling & 1), float((castling >> 1) & 1),
                          float((castling >> 2) & 1), float((castling >> 3) & 1) };
        for (uint o = 0; o < s.d; o++) {
            float v = state_embed[o] + castle_b[o] + ep_embed[ep * s.d + o];
            device const float* w = castle_w + o * 4;
            v += w[0] * bits[0] + w[1] * bits[1] + w[2] * bits[2] + w[3] * bits[3];
            out[o] = v;
        }
    } else {
        uint sq = i - s.square_start;
        uint piece = board[sq];
        device const float* pe = piece_embed + piece * s.d;
        device const float* se = square_embed + sq * s.d;
        for (uint o = 0; o < s.d; o++) out[o] = pe[o] + se[o];
    }
}

// out[r, :] = LayerNorm(in[row(r), :]); one SIMD group (32 lanes) per output row, lanes
// striding over the columns and reducing with simd_sum. `per_pos` output rows per position
// map to input token rows `start + (r % per_pos)`: the whole sequence (per_pos = tokens,
// start = 0), the 64 squares, or the state token alone. d must be a multiple of 4. The
// output is gemm_t when it feeds a GEMM (`layer_norm`) and fp32 when it feeds another
// norm (`layer_norm_f32`, the final norm).
template <typename Out4>
inline void layer_norm_body(device const float* in, device Out4* out, device const float* gamma,
                            device const float* beta, constant Shape& s, uint per_pos, uint start,
                            uint gid, uint lane) {
    const uint r = gid / 32;
    if (r >= s.positions * per_pos) return;
    const uint p = r / per_pos, k = r % per_pos;
    device const float4* row = (device const float4*)(in + ((p * s.tokens) + start + k) * s.d);
    device Out4* o = out + r * (s.d / 4);
    const uint d4 = s.d / 4;
    float4 sum = float4(0.0f);
    for (uint c = lane; c < d4; c += 32) sum += row[c];
    const float mean = simd_sum(sum.x + sum.y + sum.z + sum.w) / float(s.d);
    float4 sq = float4(0.0f);
    for (uint c = lane; c < d4; c += 32) { float4 t = row[c] - mean; sq += t * t; }
    const float inv = rsqrt(simd_sum(sq.x + sq.y + sq.z + sq.w) / float(s.d) + s.eps);
    device const float4* g4 = (device const float4*)gamma;
    device const float4* b4 = (device const float4*)beta;
    for (uint c = lane; c < d4; c += 32) o[c] = Out4((row[c] - mean) * inv * g4[c] + b4[c]);
}
kernel void layer_norm(device const float* in   [[buffer(0)]],
                       device gemm4_t* out      [[buffer(1)]],
                       device const float* gamma[[buffer(2)]],
                       device const float* beta [[buffer(3)]],
                       constant Shape& s        [[buffer(4)]],
                       constant uint& per_pos   [[buffer(5)]],
                       constant uint& start     [[buffer(6)]],
                       uint gid [[thread_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    layer_norm_body<gemm4_t>(in, out, gamma, beta, s, per_pos, start, gid, lane);
}
kernel void layer_norm_f32(device const float* in   [[buffer(0)]],
                           device float4* out       [[buffer(1)]],
                           device const float* gamma[[buffer(2)]],
                           device const float* beta [[buffer(3)]],
                           constant Shape& s        [[buffer(4)]],
                           constant uint& per_pos   [[buffer(5)]],
                           constant uint& start     [[buffer(6)]],
                           uint gid [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    layer_norm_body<float4>(in, out, gamma, beta, s, per_pos, start, gid, lane);
}

// out[e] = float(y[e]) + bias[e % cols]: a head's GEMM output (gemm_t) to its fp32 result,
// one thread per element, for widths (73, 3) that are not multiples of 4.
kernel void add_bias(device const gemm_t* y       [[buffer(0)]],
                     device const float* bias   [[buffer(1)]],
                     device float* out          [[buffer(2)]],
                     constant uint& cols        [[buffer(3)]],
                     constant uint& count       [[buffer(4)]],
                     uint gid [[thread_position_in_grid]]) {
    if (gid >= count) return;
    out[gid] = float(y[gid]) + bias[gid % cols];
}

// y[r, c] += bias[c] in place on gemm_t rows, one thread per four elements (cols a
// multiple of 4): the QKV projection before attention.
kernel void add_bias4(device gemm4_t* y             [[buffer(0)]],
                      device const float4* bias   [[buffer(1)]],
                      constant uint& cols         [[buffer(2)]],
                      constant uint& count        [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid * 4 >= count) return;
    y[gid] = gemm4_t(float4(y[gid]) + bias[gid % (cols / 4)]);
}

// x[e] += proj[e] + bias[e % cols]: the residual add, fp32 stream, gemm_t projection.
kernel void add_bias_residual(device float4* x          [[buffer(0)]],
                              device const gemm4_t* proj  [[buffer(1)]],
                              device const float4* bias [[buffer(2)]],
                              constant uint& cols       [[buffer(3)]],
                              constant uint& count      [[buffer(4)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid * 4 >= count) return;
    x[gid] += float4(proj[gid]) + bias[gid % (cols / 4)];
}

// Abramowitz & Stegun 7.1.26, |error| < 1.5e-7: the exact-form GELU within fp32 noise.
inline float erf_approx(float x) {
    float sign = x < 0.0f ? -1.0f : 1.0f;
    x = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * x);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-x * x);
    return sign * y;
}

// y = gelu(y + bias), gemm_t in place with fp32 arithmetic; `tanh_form` selects the tanh
// approximation the network was trained with.
kernel void bias_gelu(device gemm4_t* y            [[buffer(0)]],
                      device const float4* bias  [[buffer(1)]],
                      constant uint& cols        [[buffer(2)]],
                      constant uint& count       [[buffer(3)]],
                      constant uint& tanh_form   [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid * 4 >= count) return;
    float4 v = float4(y[gid]) + bias[gid % (cols / 4)];
    float4 r;
    if (tanh_form != 0) {
        float4 u = 0.7978845608028654f * (v + 0.044715f * v * v * v);
        r = 0.5f * v * (1.0f + tanh(u));
    } else {
        const float4 z = v * 0.7071067811865476f;
        r = 0.5f * v * (1.0f + float4(erf_approx(z.x), erf_approx(z.y), erf_approx(z.z), erf_approx(z.w)));
    }
    y[gid] = gemm4_t(r);
}

// Multi-head self-attention over one position's tokens: one threadgroup per (position,
// head) with a thread per query token. qkv rows are [q | k | v], each d wide, with the
// in_proj bias already added. TOKENS and HEAD_DIM are #defined by the host from the
// architecture before compiling, so the loops unroll and the query and the running output
// stay in registers. The head's K is staged in threadgroup memory once; V rows are read in
// lockstep by the whole group, so both leave device memory once per head instead of once
// per query. The softmax is the online form (running max and sum): no score is stored.
#define HD4 (HEAD_DIM / 4)
kernel void attention(device const gemm_t* qkv  [[buffer(0)]],
                      device gemm_t* out        [[buffer(1)]],
                      constant Shape& s       [[buffer(2)]],
                      uint group [[threadgroup_position_in_grid]],
                      uint i [[thread_position_in_threadgroup]]) {
    threadgroup float4 K[TOKENS * HD4];
    const uint p = group / s.heads, head = group % s.heads;
    const uint row_stride = 3 * s.d;
    device const gemm_t* base = qkv + p * TOKENS * row_stride;
    for (uint idx = i; idx < TOKENS * HD4; idx += TOKENS) {
        const uint j = idx / HD4, c = idx % HD4;
        K[idx] = float4(((device const gemm4_t*)(base + j * row_stride + s.d + head * HEAD_DIM))[c]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const gemm4_t* qp = (device const gemm4_t*)(base + i * row_stride + head * HEAD_DIM);
    float4 q[HD4];
    float4 acc[HD4];
    for (uint c = 0; c < HD4; c++) { q[c] = float4(qp[c]) * s.scale; acc[c] = float4(0.0f); }
    float m = -INFINITY, l = 0.0f;
    for (uint j = 0; j < TOKENS; j++) {
        device const gemm4_t* v = (device const gemm4_t*)(base + j * row_stride + 2 * s.d + head * HEAD_DIM);
        float4 dot = float4(0.0f);
        for (uint c = 0; c < HD4; c++) dot += q[c] * K[j * HD4 + c];
        const float sc = dot.x + dot.y + dot.z + dot.w;
        const float m_new = max(m, sc);
        const float corr = exp(m - m_new);   // 0 on the first key (m = -inf)
        const float w = exp(sc - m_new);
        l = l * corr + w;
        for (uint c = 0; c < HD4; c++) acc[c] = acc[c] * corr + w * float4(v[c]);
        m = m_new;
    }
    const float inv = 1.0f / l;
    device gemm4_t* o = (device gemm4_t*)(out + (p * TOKENS + i) * s.d + head * HEAD_DIM);
    for (uint c = 0; c < HD4; c++) o[c] = gemm4_t(acc[c] * inv);
}
