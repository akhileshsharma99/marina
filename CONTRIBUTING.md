# Contributing

Marina is developed in a larger repository and mirrored here; this repository is a read-only image of that directory, kept in sync on every change, and it is where releases are cut. Issues and pull requests here are welcome; a pull request is applied upstream and comes back through the mirror with your authorship intact.

## Building

```sh
cargo build --release                       # CPU backend, networks embedded; on macOS also Metal
cargo build --release --features cuda       # + CUDA (needs the CUDA toolkit to build)
cargo build --release --features openblas   # Linux: OpenBLAS for the CPU backend
```

Rust 1.88 or newer. The first build downloads the embedded networks (about 165 MB, plus 70 MB of golden vectors beside them) and checks their sha256 against `nets.toml`; set `MARINA_NETS_DIR` to build offline. `docker build --target dev .` gives an image with the CUDA toolkit and a Rust toolchain for building the CUDA backend (see `Dockerfile`).

## Checks

```sh
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test --release
marina verify --weights small --backend cpu     # a backend against the golden vectors
marina verify --weights small --backend metal   # macOS: the Metal kernels against the same
```

The Metal backend (`src/net/metal/`) uses MPS matrix multiplications for the linear layers, in fp16 by default or fp32 with `Precision=fp32` (weights converted on the device at load; the residual stream and every reduction stay fp32, as in the CUDA fp16 path) and Metal kernels, compiled at run time from `kernels.metal`, for everything else. It needs no feature flag and no SDK beyond macOS itself.

`cargo test` covers the unit tests, the UCI protocol, the encoder against the embedded networks' golden vectors, and a real search. The tablebase tests need `MARINA_SYZYGY_PATH=<dir>` and skip without it. `MARINA_LOG=marina::search=trace` turns on search diagnostics on stderr.

## Measuring a change

Every change to the search or a backend is gated on Elo, not on a benchmark:

```sh
marina bench --weights small --backend cuda      # nodes per second, for throughput work
tools/ab.sh --base "CPuct=175" --test "CPuct=150" --tc 10+0.1   # SPRT, one game at a time
```

`tools/ab.sh` needs [fastchess](https://github.com/Disservin/fastchess) on the path and uses the opening book in `tools/books`. Run one game at a time: concurrency changes what the engine can compute per move and the result stops meaning anything. A backend change must also pass `marina verify` against the golden vectors before it is measured.

## Commits and releases

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `perf:`, …); the changelog and version bumps are generated from them by release-please, and a release builds the binaries for every platform from the tag.
