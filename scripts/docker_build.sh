#!/usr/bin/env bash
# Build StreamForge Rust binaries inside Docker and copy them to dist/
#
# Usage:
#   bash scripts/docker_build.sh
#
# Requirements:
#   - Docker with BuildKit support (docker >= 20.10)
#   - NVIDIA Container Toolkit is NOT required for the build step
#     (linking uses stub libcuda.so from the devel image, not the real GPU)
#
# Output:
#   dist/text_encoder  dist/flux_gpu  dist/vae_decode  (ready for Kaggle T4)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
IMAGE="gpuload-builder:latest"

echo "==> Building Docker image ($IMAGE) ..."
docker build --progress=plain -t "$IMAGE" "$ROOT"

echo ""
echo "==> Extracting binaries to dist/ ..."
mkdir -p "$ROOT/dist"

docker run --rm \
    -v "$ROOT/dist:/out" \
    "$IMAGE" \
    sh -c "cp /build/target/release/text_encoder \
              /build/target/release/flux_gpu \
              /build/target/release/vae_decode \
              /out/"

echo ""
echo "==> Done."
ls -lh "$ROOT/dist/text_encoder" "$ROOT/dist/flux_gpu" "$ROOT/dist/vae_decode"
