# Two images in one file.
#
# runtime (the default): the released CUDA binary on the CUDA runtime base, networks
# embedded; cuBLASLt is in the base, the driver comes from the host through `--gpus all`,
# and without a GPU the engine runs on the CPU. Published on every release as
# ghcr.io/akhileshsharma99/marina:<version> and :latest.
#
#   docker run --rm -i --gpus all ghcr.io/akhileshsharma99/marina
#
# To build it yourself, put a Linux x64 CUDA binary named `marina` in the context:
#
#   curl -fsSL -o marina https://github.com/akhileshsharma99/marina/releases/latest/download/marina-linux-x64-cuda
#   docker build -t marina .
#
# dev: the CUDA toolkit (nvcc, cuBLASLt, NVRTC) and a Rust toolchain for building the CUDA
# backend on a machine without the toolkit; mount the source tree.
#
#   docker build --target dev -t marina-dev .
#   docker run --rm -it --gpus all -v "$PWD:/work" marina-dev cargo build --release --features cuda
#
# The host driver must be at least as new as this toolkit (CUDA 12.8 needs driver >= 570).

ARG CUDA_VERSION=12.8.1
# The images by digest, so a re-pushed tag cannot change what this builds; both are the
# -*-ubuntu24.04 tags as of 2026-09.
ARG CUDA_DEVEL=nvidia/cuda@sha256:4b9ed5fa8361736996499f64ecebf25d4ec37ff56e4d11323ccde10aa36e0c43
ARG CUDA_RUNTIME=nvidia/cuda@sha256:828c4d878adcaa4265d80c95d8ec877149b49bb2419a4cf3bb6aa889bbb7ca2e

FROM ${CUDA_DEVEL} AS dev
ARG RUST_VERSION=1.98.1
ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/opt/rustup \
    CARGO_HOME=/opt/cargo \
    PATH=/opt/cargo/bin:/usr/local/cuda/bin:${PATH} \
    CUDA_ROOT=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl build-essential libopenblas-dev git \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal \
        --default-toolchain "${RUST_VERSION}" --component rustfmt --component clippy \
    && chmod -R a+rwX /opt/cargo /opt/rustup
WORKDIR /work
CMD ["bash"]

FROM ${CUDA_RUNTIME} AS runtime
COPY --chmod=755 marina /usr/local/bin/marina
# Mount Syzygy tables here and `setoption name SyzygyPath value /syzygy`.
VOLUME ["/syzygy"]
# The engine needs stdin, stdout and read access to /syzygy; nothing that wants root.
USER 65534:65534
ENTRYPOINT ["marina"]
