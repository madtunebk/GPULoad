# Build StreamForge Rust binaries for Kaggle T4 (sm_75, Turing)
#
# Base: Ubuntu 20.04 + CUDA 12.2 devel
#   - glibc 2.31 → binaries run on any Ubuntu 20.04+ system (including Kaggle)
#   - CUDA devel image provides /usr/local/cuda/lib64/stubs/libcuda.so for linking
#
# To build and extract binaries:
#   bash scripts/docker_build.sh

FROM nvidia/cuda:12.2.2-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
        curl git pkg-config libssl-dev build-essential python3 \
    && rm -rf /var/lib/apt/lists/*

# Install Rust stable
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable --profile minimal
ENV PATH="/root/.cargo/bin:$PATH"

WORKDIR /build

# Copy manifests first — lets Docker cache the `cargo fetch` layer
COPY Cargo.toml Cargo.lock ./
COPY src/ src/

# Download all Cargo dependencies (pinned via Cargo.lock)
RUN cargo fetch --locked

# Patch candle flux model.rs to expose private items
COPY scripts/patch_candle.py scripts/patch_candle.py
RUN python3 scripts/patch_candle.py

# Build release binaries targeting Kaggle T4 (Turing sm_75)
ENV CUDA_COMPUTE_CAP=75
ENV CUDA_PATH=/usr/local/cuda
ENV CUDA_ROOT=/usr/local/cuda

RUN RUSTFLAGS="-C link-arg=-L/usr/local/cuda/lib64/stubs \
               -C link-arg=-L/usr/local/cuda/lib64" \
    cargo build --release --locked --features cuda
