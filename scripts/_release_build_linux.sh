#!/usr/bin/env bash
# In-container entrypoint for the Linux release archives: build the headless
# binary (all backends compiled in) then package it self-contained. Invoked by
# scripts/release-linux-x86_64.bat and scripts/release-linux-arm64.bat.
#
# Usage: scripts/_release_build_linux.sh <arch>     (arch: x86_64 | aarch64)
set -euo pipefail
cd "$(dirname "$0")/.."

ARCH="${1:?usage: _release_build_linux.sh <arch>}"

# CUDA target arch for the aarch64 (Grace-class) release. Ship NATIVE SASS for
# the two realistic ARM NVIDIA targets: Grace-Hopper (sm_90, GH200) and — the
# headline target — the DGX Spark's GB10 (sm_121, Blackwell, CUDA 13.0+).
# kernels-cuda/build.rs also embeds forward-compatible PTX for the highest arch
# (compute_121) for any newer GPU. Native GB10 SASS avoids the one-time PTX JIT
# at first load (the extra nvcc pass lengthens the QEMU build, accepted). Static
# cudart 13.0 matches the Spark's CUDA 13.0.2 / driver 580.159.03.
if [ "$ARCH" = "aarch64" ]; then
  export RUSTLLAMA_CUDA_ARCHS="90;121"
  echo ">> RUSTLLAMA_CUDA_ARCHS=$RUSTLLAMA_CUDA_ARCHS (native GH200 sm_90 + DGX Spark GB10 sm_121)"
fi

# 1. Build the headless server/CLI binary.
scripts/build.sh --headless

# 2. Resolve version from the workspace manifest.
VERSION="$(grep -m1 '^version = ' Cargo.toml | sed -E 's/.*"(.*)".*/\1/')"
[ -n "$VERSION" ] || { echo "ERROR: could not read version from Cargo.toml" >&2; exit 1; }

# 3. Package the self-contained archive.
BIN="${CARGO_TARGET_DIR:-target}/release/rustllama"
scripts/_release_package.sh "$BIN" "$VERSION" "$ARCH"
