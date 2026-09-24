//! Link the platform BLAS used by the CPU backend's GEMM (see `src/net/gemm.rs`) and, with
//! the `cuda` feature, compile the CUDA kernels to PTX with nvcc. The embedded networks are
//! the `marina-nets` crate's business (`nets/build.rs`).

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    } else if std::env::var_os("CARGO_FEATURE_OPENBLAS").is_some() {
        println!("cargo:rustc-link-lib=openblas");
    }
    if std::env::var_os("CARGO_FEATURE_CUDA").is_some() {
        compile_kernels();
    }
    println!("cargo:rerun-if-changed=build.rs");
}

/// `src/net/cuda/kernels.cu` -> `$OUT_DIR/kernels.ptx`. PTX for compute_80 runs on every
/// Ampere-or-newer GPU through the driver's JIT (cached by the driver).
fn compile_kernels() {
    let source = PathBuf::from("src/net/cuda/kernels.cu");
    let out = out_dir().join("kernels.ptx");
    println!("cargo:rerun-if-changed={}", source.display());
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    let nvcc = ["CUDA_ROOT", "CUDA_PATH"]
        .iter()
        .filter_map(std::env::var_os)
        .map(|root| PathBuf::from(root).join("bin").join("nvcc"))
        .find(|path| path.exists())
        .unwrap_or_else(|| PathBuf::from("nvcc"));
    let status = Command::new(&nvcc)
        .args(["-ptx", "-O3", "-arch=compute_80", "-std=c++17", "-o"])
        .arg(&out)
        .arg(&source)
        .status()
        .unwrap_or_else(|error| panic!("cannot run {}: {error}", nvcc.display()));
    assert!(
        status.success(),
        "nvcc failed compiling {}",
        source.display()
    );
}

fn out_dir() -> PathBuf {
    PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"))
}
