//! fp32 forward pass on the CPU: the reference every other backend is checked against,
//! and the fallback where there is no GPU.
//!
//! Every linear layer is a GEMM through the platform BLAS ([`super::gemm`]), sliced by rows
//! across the thread pool with the BLAS pinned to one thread; the rest (embeddings, layer
//! norm, attention, GELU, softmax) is loops spread over the same pool. The CUDA backend is
//! the fast path; this one has to be right, and fast enough to play with.

use rayon::prelude::*;

use super::model::{Architecture, Gelu, POLICY_PLANES, WDL_OUTPUTS};
use super::weights::{Linear, Norm, Weights};
use super::{Evaluation, Leaf, Network, NetworkError};
use crate::encoding::EncodedBoard;
use crate::encoding::action::PLANES_PER_SQUARE;

pub struct CpuNetwork {
    arch: Architecture,
    weights: Weights,
    /// The forward pass runs on this pool, sized by the `Threads` option.
    pool: rayon::ThreadPool,
}

impl CpuNetwork {
    /// `threads` is the size of the pool the forward pass runs on (the `Threads` option).
    pub fn new(arch: Architecture, weights: Weights, threads: usize) -> Self {
        super::gemm::single_threaded();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.max(1))
            .thread_name(|i| format!("marina-cpu-{i}"))
            .build()
            .expect("rayon thread pool");
        Self {
            arch,
            weights,
            pool,
        }
    }

    /// Raw policy logits over all `64 × 73` actions and WDL logits for one batch, in
    /// action-index order (`square · 73 + plane`, the order of the golden vectors). What
    /// `verify` compares; `evaluate` gathers and normalises them. Policy logits are
    /// `[n, 64·73]`, value logits `[n, 3]`.
    pub fn logits(&self, boards: &[&EncodedBoard]) -> (Vec<f32>, Vec<f32>) {
        self.pool.install(|| self.forward(boards))
    }

    fn forward(&self, boards: &[&EncodedBoard]) -> (Vec<f32>, Vec<f32>) {
        let arch = &self.arch;
        let w = &self.weights;
        let n = boards.len();
        let t = arch.tokens;
        let d = arch.d_model;
        let rows = n * t;
        if n == 0 {
            return (Vec::new(), Vec::new());
        }

        // ---- tokens
        let mut x = vec![0.0f32; rows * d];
        x.par_chunks_mut(t * d)
            .zip(boards.par_iter())
            .for_each(|(tokens, board)| {
                let (state, squares) = tokens.split_at_mut(d);
                state.copy_from_slice(&w.state_embed);
                let bits: [f32; 4] = std::array::from_fn(|b| f32::from((board.castling >> b) & 1));
                for (o, out) in state.iter_mut().enumerate() {
                    let row = &w.castling_proj.weight[o * 4..o * 4 + 4];
                    *out += w.castling_proj.bias[o]
                        + row
                            .iter()
                            .zip(&bits)
                            .map(|(wgt, bit)| wgt * bit)
                            .sum::<f32>();
                }
                let ep = usize::from(board.en_passant).min(w.en_passant_embed.len() / d - 1);
                add(state, &w.en_passant_embed[ep * d..(ep + 1) * d]);
                let squares = &mut squares[(arch.square_token_start - 1) * d..];
                for (s, token) in squares.chunks_exact_mut(d).enumerate().take(64) {
                    let piece = usize::from(board.pieces[s]);
                    token.copy_from_slice(&w.piece_embed[piece * d..(piece + 1) * d]);
                    add(token, &w.square_embed[s * d..(s + 1) * d]);
                }
            });

        // ---- encoder
        let heads = arch.n_heads;
        let hd = arch.head_dim();
        let scale = 1.0 / (hd as f32).sqrt();
        let eps = arch.layer_norm_eps;
        let mut h = vec![0.0f32; rows * d];
        let mut qkv = vec![0.0f32; rows * 3 * d];
        let mut attn = vec![0.0f32; rows * d];
        let mut ff = vec![0.0f32; rows * arch.d_ff];
        let mut proj = vec![0.0f32; rows * d];
        for layer in &w.layers {
            layer_norm(&x, &mut h, &layer.norm1, d, eps);
            linear(&h, &mut qkv, &layer.in_proj, rows);
            attn.par_chunks_mut(t * d)
                .zip(qkv.par_chunks(t * 3 * d))
                .for_each(|(out, qkv)| {
                    let mut scores = vec![0.0f32; t];
                    for head in 0..heads {
                        let q_off = head * hd;
                        let k_off = d + head * hd;
                        let v_off = 2 * d + head * hd;
                        let row = |i: usize| i * 3 * d;
                        for i in 0..t {
                            let q = &qkv[row(i) + q_off..row(i) + q_off + hd];
                            let mut max = f32::NEG_INFINITY;
                            for (j, s) in scores.iter_mut().enumerate() {
                                let k = &qkv[row(j) + k_off..row(j) + k_off + hd];
                                *s = dot(q, k) * scale;
                                max = max.max(*s);
                            }
                            let inv = 1.0 / softmax_inplace(&mut scores, max);
                            let o = &mut out[i * d + q_off..i * d + q_off + hd];
                            o.fill(0.0);
                            for (j, &pij) in scores.iter().enumerate() {
                                let v = &qkv[row(j) + v_off..row(j) + v_off + hd];
                                axpy(pij * inv, v, o);
                            }
                        }
                    }
                });
            linear(&attn, &mut proj, &layer.out_proj, rows);
            add_par(&mut x, &proj);
            layer_norm(&x, &mut h, &layer.norm2, d, eps);
            linear(&h, &mut ff, &layer.linear1, rows);
            match self.arch.gelu {
                Gelu::Erf => ff.par_chunks_mut(4096).for_each(gelu_inplace),
                Gelu::Tanh => ff.par_chunks_mut(4096).for_each(gelu_tanh_inplace),
            }
            linear(&ff, &mut proj, &layer.linear2, rows);
            add_par(&mut x, &proj);
        }
        layer_norm(&x, &mut h, &w.final_norm, d, eps);

        // ---- heads
        let mut squares = vec![0.0f32; n * 64 * d];
        let mut state = vec![0.0f32; n * d];
        squares
            .par_chunks_mut(64 * d)
            .zip(state.par_chunks_mut(d))
            .zip(h.par_chunks(t * d))
            .for_each(|((sq, st), tokens)| {
                st.copy_from_slice(&tokens[..d]);
                let start = arch.square_token_start * d;
                sq.copy_from_slice(&tokens[start..start + 64 * d]);
            });
        let mut normed = vec![0.0f32; n * 64 * d];
        layer_norm(&squares, &mut normed, &w.policy_norm, d, eps);
        let mut policy = vec![0.0f32; n * 64 * POLICY_PLANES];
        linear(&normed, &mut policy, &w.policy_proj, n * 64);

        let mut vnorm = vec![0.0f32; n * d];
        layer_norm(&state, &mut vnorm, &w.value_norm, d, eps);
        let mut hidden = vec![0.0f32; n * d];
        linear(&vnorm, &mut hidden, &w.value_ff1, n);
        gelu_inplace(&mut hidden);
        let mut value = vec![0.0f32; n * WDL_OUTPUTS];
        linear(&hidden, &mut value, &w.value_ff2, n);
        (policy, value)
    }
}

impl Network for CpuNetwork {
    fn evaluate(&self, batch: &[Leaf<'_>]) -> Result<Vec<Evaluation>, NetworkError> {
        let boards: Vec<&EncodedBoard> = batch.iter().map(|(board, _)| *board).collect();
        let (policy, value) = self.logits(&boards);
        let actions = 64 * PLANES_PER_SQUARE;
        Ok(batch
            .iter()
            .enumerate()
            .map(|(p, (_, legal))| {
                let logits = &policy[p * actions..(p + 1) * actions];
                let mut priors: Vec<f32> = legal.moves.iter().map(|&(_, a)| logits[a]).collect();
                softmax(&mut priors);
                let mut wdl = [value[p * 3], value[p * 3 + 1], value[p * 3 + 2]];
                softmax(&mut wdl);
                Evaluation { priors, wdl }
            })
            .collect())
    }

    fn describe(&self) -> String {
        format!(
            "cpu fp32 ({}): {}",
            super::gemm::BACKEND,
            self.arch.describe()
        )
    }
}

#[inline]
fn add(target: &mut [f32], source: &[f32]) {
    for (t, s) in target.iter_mut().zip(source) {
        *t += s;
    }
}

/// Dot product with independent accumulators so the compiler can vectorise it (a
/// strict left-to-right sum cannot be).
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (a8, a_tail) = a.as_chunks::<8>();
    let (b8, b_tail) = b.as_chunks::<8>();
    for (x, y) in a8.iter().zip(b8) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in a_tail.iter().zip(b_tail) {
        sum += x * y;
    }
    sum
}

/// `y += a · x` over 8-wide chunks so it vectorises.
#[inline]
fn axpy(a: f32, x: &[f32], y: &mut [f32]) {
    let (x8, x_tail) = x.as_chunks::<8>();
    let (y8, y_tail) = y.as_chunks_mut::<8>();
    for (yc, xc) in y8.iter_mut().zip(x8) {
        for i in 0..8 {
            yc[i] += a * xc[i];
        }
    }
    for (yi, xi) in y_tail.iter_mut().zip(x_tail) {
        *yi += a * xi;
    }
}

/// `exp` to ~1e-7 relative error (Cephes `expf`), inlined so softmax and GELU loops
/// vectorise; `libm`'s `expf` is a scalar call and would otherwise dominate the softmax and GELU loops.
#[inline]
fn exp_fast(x: f32) -> f32 {
    const C1: f32 = 0.693_359_4;
    const C2: f32 = -2.121_944_4e-4;
    let x = x.clamp(-87.0, 88.0);
    let k = (x * std::f32::consts::LOG2_E).round();
    let r = x - k * C1 - k * C2;
    let poly = ((((1.987_569_2e-4 * r + 1.398_199_9e-3) * r + 8.333_452e-3) * r + 4.166_579_6e-2)
        * r
        + 1.666_666_5e-1)
        * r
        + 0.5;
    let e = poly * r * r + r + 1.0;
    let scale = f32::from_bits(((k as i32 + 127) as u32) << 23);
    e * scale
}

/// Exact GELU, `0.5·x·(1 + erf(x/√2))`, as `torch.nn.GELU()` computes it.
#[inline]
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

fn gelu_inplace(values: &mut [f32]) {
    for v in values {
        *v = gelu(*v);
    }
}

/// The tanh approximation, `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`, as
/// `torch.nn.GELU(approximate="tanh")` and cuBLASLt's GELU epilogue compute it.
#[inline]
fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_6;
    let u = SQRT_2_OVER_PI * (x + 0.044_715 * x * x * x);
    // tanh(u) = 1 - 2 / (exp(2u) + 1), with the fast exp and its overflow clamped.
    let e = exp_fast(2.0 * u.clamp(-20.0, 20.0));
    0.5 * x * (2.0 - 2.0 / (e + 1.0))
}

fn gelu_tanh_inplace(values: &mut [f32]) {
    for v in values {
        *v = gelu_tanh(*v);
    }
}

/// erf to within 1.5e-7 (Abramowitz & Stegun 7.1.26): below fp32 resolution for this
/// use and an order of magnitude faster than a libm call, which matters because the FFN
/// applies it a few hundred thousand times per position.
#[inline]
fn erf(x: f32) -> f32 {
    const P: f32 = 0.327_591_1;
    const A: [f32; 5] = [
        0.254_829_6,
        -0.284_496_72,
        1.421_413_7,
        -1.453_152,
        1.061_405_4,
    ];
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + P * x);
    let poly = ((((A[4] * t + A[3]) * t + A[2]) * t + A[1]) * t + A[0]) * t;
    sign * (1.0 - poly * exp_fast(-x * x))
}

/// Exponentiate `values - max` in place and return the sum.
#[inline]
fn softmax_inplace(values: &mut [f32], max: f32) -> f32 {
    let mut sum = 0.0f32;
    for v in values.iter_mut() {
        *v = exp_fast(*v - max);
        sum += *v;
    }
    sum
}

/// `x += y`, in parallel over large slices.
fn add_par(x: &mut [f32], y: &[f32]) {
    x.par_chunks_mut(4096)
        .zip(y.par_chunks(4096))
        .for_each(|(a, b)| add(a, b));
}

/// `LayerNorm` over the last dimension of `[rows, d]`, biased variance, as PyTorch.
fn layer_norm(input: &[f32], output: &mut [f32], norm: &Norm, d: usize, eps: f32) {
    input
        .par_chunks(d)
        .zip(output.par_chunks_mut(d))
        .for_each(|(row, out)| layer_norm_row(row, out, norm, d, eps));
}

#[inline]
fn layer_norm_row(row: &[f32], out: &mut [f32], norm: &Norm, d: usize, eps: f32) {
    debug_assert_eq!(row.len(), d);
    let mean = row.iter().sum::<f32>() / d as f32;
    let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
    let inv = 1.0 / (var + eps).sqrt();
    for ((o, v), (g, b)) in out
        .iter_mut()
        .zip(row)
        .zip(norm.weight.iter().zip(&norm.bias))
    {
        *o = (v - mean) * inv * g + b;
    }
}

/// `output[rows, out] = input[rows, in] · weightᵀ + bias`.
fn linear(input: &[f32], output: &mut [f32], layer: &Linear, rows: usize) {
    let (m, k, n) = (rows, layer.in_features, layer.out_features);
    debug_assert_eq!(input.len(), m * k);
    debug_assert_eq!(output.len(), m * n);
    // Row slices in parallel: the BLAS is fastest on a few hundred rows at a time (cache
    // resident) and does not thread these shapes itself.
    const ROWS_PER_CALL: usize = 512;
    output
        .par_chunks_mut(ROWS_PER_CALL * n)
        .zip(input.par_chunks(ROWS_PER_CALL * k))
        .for_each(|(out, inp)| {
            for row in out.chunks_exact_mut(n) {
                row.copy_from_slice(&layer.bias);
            }
            super::gemm::sgemm_nt(out.len() / n, k, n, inp, &layer.weight, out);
        });
}

pub(super) fn softmax(values: &mut [f32]) {
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum = softmax_inplace(values, max);
    let inv = 1.0 / sum.max(f32::MIN_POSITIVE);
    for v in values.iter_mut() {
        *v *= inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitives_match_definitions() {
        // GELU at a few points (torch.nn.GELU(), exact).
        assert!((gelu(0.0)).abs() < 1e-7);
        assert!((gelu(1.0) - 0.841_345).abs() < 1e-5);
        assert!((gelu(-1.0) + 0.158_655).abs() < 1e-5);
        assert!((gelu(3.0) - 2.995_95).abs() < 1e-5);
        // torch.nn.GELU(approximate="tanh") at the same points.
        assert!((gelu_tanh(0.0)).abs() < 1e-7);
        assert!((gelu_tanh(1.0) - 0.841_192).abs() < 1e-5);
        assert!((gelu_tanh(-1.0) + 0.158_808).abs() < 1e-5);
        assert!((gelu_tanh(3.0) - 2.996_36).abs() < 1e-5);

        // LayerNorm of a row with gamma 1, beta 0 has mean 0 and unit variance.
        let norm = Norm {
            weight: vec![1.0; 4],
            bias: vec![0.0; 4],
        };
        let mut out = vec![0.0; 4];
        layer_norm(&[1.0, 2.0, 3.0, 4.0], &mut out, &norm, 4, 1e-5);
        assert!((out.iter().sum::<f32>()).abs() < 1e-5);
        assert!((out[3] - 1.341_64).abs() < 1e-4);

        // Linear against a hand computation: y = x·Wᵀ + b.
        let layer = Linear {
            out_features: 2,
            in_features: 3,
            weight: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            bias: vec![0.5, -0.5],
        };
        let mut y = vec![0.0; 4];
        linear(&[1.0, 1.0, 1.0, 0.0, 1.0, 0.0], &mut y, &layer, 2);
        assert_eq!(y, vec![6.5, 14.5, 2.5, 4.5]);

        for &x in &[-80.0f32, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0, 20.0, 80.0] {
            let rel = (exp_fast(x) - x.exp()).abs() / x.exp();
            assert!(rel < 3e-7, "exp({x}): rel err {rel}");
        }

        let mut s = vec![1.0, 2.0, 3.0];
        softmax(&mut s);
        assert!((s.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((s[2] - 0.665_24).abs() < 1e-4);
    }
}
