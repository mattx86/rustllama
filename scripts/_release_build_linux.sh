#!/usr/bin/env bash
# In-container entrypoint for the Linux release archives: build the GUI binary
# (desktop GUI + CLI + server, all backends compiled in) then package it
# self-contained — the single binary also runs headless (`rustllama serve`),
# so the GUI artifact is a superset of the old headless one. Invoked by
# scripts/release-linux-x86_64.bat and scripts/release-linux-arm64.bat.
#
# The GUI is native egui/eframe (crates/rustllama-gui) — it links a small
# OpenGL + windowing graph (mesa GL/EGL, X11/Wayland, libxkbcommon,
# fontconfig/freetype, D-Bus for rfd file dialogs), NOT Tauri/webkit. The
# packaging step below bundles the ABI-stable, non-host .so's via $ORIGIN
# RPATH (dynamically linked + replaceable + accompanied by NOTICES); the
# host-provided GL/vendor-driver and xdg-desktop-portal libs are left on the
# target. Set HEADLESS=1 to build the server-only binary instead (no GUI
# graph bundled).
#
# Usage: scripts/_release_build_linux.sh <arch>     (arch: x86_64 | aarch64)
set -euo pipefail
cd "$(dirname "$0")/.."

ARCH="${1:?usage: _release_build_linux.sh <arch>}"

# CUDA target arch for the aarch64 (Grace-class) release. Ship NATIVE SASS for
# the two realistic ARM NVIDIA targets: Grace-Hopper (sm_90a, GH200) and — the
# headline target — the DGX Spark's GB10 (sm_121a, Blackwell, CUDA 13.0+).
# BOTH are the *accelerated* arches (the `a` suffix), NOT plain sm_90/sm_121: the
# tensor-core kernels in rustllama-kernels-cuda are gated on the suffix — Hopper
# wgmma FP8 + TMA (build.rs `hopper_tc`, sm_90a) and Blackwell FP4/FP8/FP6 + TMA
# (build.rs `blackwell_tc`, sm_121a) — so a plain sm_90/sm_121 release would ship
# them DISABLED. kernels-cuda/build.rs also embeds forward-compatible PTX for the
# highest arch for any newer GPU. Native SASS avoids the one-time PTX JIT at first
# load (the extra nvcc pass lengthens the QEMU build, accepted). Static cudart
# 13.0 matches the Spark's CUDA 13.0.2 / driver 580.159.03.
if [ "$ARCH" = "aarch64" ]; then
  export RUSTLLAMA_CUDA_ARCHS="90a;121a"
  echo ">> RUSTLLAMA_CUDA_ARCHS=$RUSTLLAMA_CUDA_ARCHS (GH200 sm_90a Hopper + DGX Spark GB10 sm_121a Blackwell, both tensor-core)"
fi

# 1. Build the binary. GUI (desktop + CLI + server) by default; HEADLESS=1
#    builds server-only. The native egui GUI needs no JS frontend — just the
#    egui/GL build deps in the image (present).
if [ "${HEADLESS:-0}" = "1" ]; then
  scripts/build.sh --headless
else
  scripts/build.sh
fi

# 2. Resolve version from the workspace manifest.
VERSION="$(grep -m1 '^version = ' Cargo.toml | sed -E 's/.*"(.*)".*/\1/')"
[ -n "$VERSION" ] || { echo "ERROR: could not read version from Cargo.toml" >&2; exit 1; }

# 3. Package the self-contained archive.
BIN="${CARGO_TARGET_DIR:-target}/release/rustllama"
scripts/_release_package.sh "$BIN" "$VERSION" "$ARCH"
