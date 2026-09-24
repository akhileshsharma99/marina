//! Single-precision GEMM for the CPU backend: `C[m×n] += A[m×k] · W[n×k]ᵀ`, the shape of
//! every `Linear` (PyTorch stores weights as `[out, in]`).
//!
//! The platform BLAS is an order of magnitude faster than any pure-Rust kernel on typical
//! hardware (Accelerate reaches ~500 GFLOP/s on Apple silicon where `matrixmultiply`
//! reaches ~40), so it is used when available: Accelerate on macOS,
//! OpenBLAS with the `openblas` feature elsewhere, and `matrixmultiply` as the fallback
//! that always builds.

/// Which implementation this build uses, for `info string`.
pub const BACKEND: &str = if cfg!(target_os = "macos") {
    "accelerate"
} else if cfg!(feature = "openblas") {
    "openblas"
} else {
    "matrixmultiply"
};

/// Keep the BLAS on one thread: the CPU backend parallelises over row slices itself, and a
/// BLAS pool inside every rayon task oversubscribes the machine several times over (several
/// times slower with OpenBLAS on a many-core machine). Accelerate has no API for this and reads
/// `VECLIB_MAXIMUM_THREADS` at start-up instead, which `main` sets.
pub fn single_threaded() {
    imp::single_threaded();
}

/// `c += a · wᵀ` with `a` row-major `[m, k]`, `w` row-major `[n, k]`, `c` row-major `[m, n]`.
pub fn sgemm_nt(m: usize, k: usize, n: usize, a: &[f32], w: &[f32], c: &mut [f32]) {
    assert_eq!(a.len(), m * k, "a is [m, k]");
    assert_eq!(w.len(), n * k, "w is [n, k]");
    assert_eq!(c.len(), m * n, "c is [m, n]");
    if m == 0 || n == 0 {
        return;
    }
    imp::sgemm_nt(m, k, n, a, w, c);
}

#[cfg(any(target_os = "macos", feature = "openblas"))]
mod imp {
    use std::os::raw::c_int;

    const ROW_MAJOR: c_int = 101;
    const NO_TRANS: c_int = 111;
    const TRANS: c_int = 112;

    unsafe extern "C" {
        #[cfg(feature = "openblas")]
        fn openblas_set_num_threads(threads: c_int);
        fn cblas_sgemm(
            order: c_int,
            trans_a: c_int,
            trans_b: c_int,
            m: c_int,
            n: c_int,
            k: c_int,
            alpha: f32,
            a: *const f32,
            lda: c_int,
            b: *const f32,
            ldb: c_int,
            beta: f32,
            c: *mut f32,
            ldc: c_int,
        );
    }

    pub fn single_threaded() {
        #[cfg(feature = "openblas")]
        // SAFETY: plain FFI call with a scalar argument.
        unsafe {
            openblas_set_num_threads(1);
        }
    }

    pub fn sgemm_nt(m: usize, k: usize, n: usize, a: &[f32], w: &[f32], c: &mut [f32]) {
        let dim = |v: usize| c_int::try_from(v).expect("matrix dimension fits in c_int");
        // SAFETY: the caller checked the slice lengths against m, k, n; leading dimensions
        // are the row lengths of row-major storage.
        unsafe {
            cblas_sgemm(
                ROW_MAJOR,
                NO_TRANS,
                TRANS,
                dim(m),
                dim(n),
                dim(k),
                1.0,
                a.as_ptr(),
                dim(k),
                w.as_ptr(),
                dim(k),
                1.0,
                c.as_mut_ptr(),
                dim(n),
            );
        }
    }
}

#[cfg(not(any(target_os = "macos", feature = "openblas")))]
mod imp {
    pub fn single_threaded() {}

    pub fn sgemm_nt(m: usize, k: usize, n: usize, a: &[f32], w: &[f32], c: &mut [f32]) {
        // SAFETY: the caller checked the slice lengths; `w` is `[n, k]` row-major, read as
        // its transpose through strides (row stride 1, column stride k).
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                1.0,
                a.as_ptr(),
                k as isize,
                1,
                w.as_ptr(),
                1,
                k as isize,
                1.0,
                c.as_mut_ptr(),
                n as isize,
                1,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_a_hand_computation() {
        // a = [[1,1,1],[0,1,0]], w = [[1,2,3],[4,5,6]] -> a·wᵀ = [[6,15],[2,5]]; c starts at 0.5.
        let a = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0];
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut c = [0.5; 4];
        sgemm_nt(2, 3, 2, &a, &w, &mut c);
        assert_eq!(c, [6.5, 15.5, 2.5, 5.5]);
    }

    #[test]
    fn agrees_with_naive_on_a_larger_shape() {
        let (m, k, n) = (37, 53, 29);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7) % 11) as f32 / 11.0 - 0.5)
            .collect();
        let w: Vec<f32> = (0..n * k)
            .map(|i| ((i * 5) % 13) as f32 / 13.0 - 0.5)
            .collect();
        let mut c = vec![0.0f32; m * n];
        sgemm_nt(m, k, n, &a, &w, &mut c);
        for i in 0..m {
            for j in 0..n {
                let expect: f32 = (0..k).map(|x| a[i * k + x] * w[j * k + x]).sum();
                assert!((c[i * n + j] - expect).abs() < 1e-4, "({i},{j})");
            }
        }
    }
}
