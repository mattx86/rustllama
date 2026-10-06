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

# CUDA target arches. GPU arch is independent of the host CPU, so BOTH releases
# ship the full realistic GPU set; only sm_121a (GB10 / DGX Spark) is ARM-only.
# The TC kernels in rustllama-kernels-cuda are gated on the *accelerated* (`a`)
# arches — Hopper wgmma FP8 on sm_90a, Blackwell FP4/FP6/FP8 on sm_12xa — so those
# must carry the `a` suffix to COMPILE the TC path (the prerequisite for
# `tune --validate-kernels` to see + auto-enable it). The other arches
# (75;80;86;89;100a) get the base path (the TC .cu is __CUDA_ARCH__-guarded, so
# those passes take the fallback). build.rs also embeds forward-compat PTX for the
# highest arch. Native SASS per arch avoids first-load PTX JIT but each adds an
# nvcc pass (notably slower under QEMU on aarch64). Static cudart 13.0 matches the
# Spark's CUDA 13.0.2 / driver 580.159.03.
if [ "$ARCH" = "aarch64" ]; then
  # Grace superchips (GH200 sm_90a, GB200 sm_100a, GB10 sm_121a) AND ARM-host
  # discrete GPUs (Ampere/Grace server + PCIe A100 80, A6000 86, L40 89, T4 75,
  # RTX 120a, ...). Jetson/Tegra (Orin sm_87) is a separate Tegra repo — not here.
  export RUSTLLAMA_CUDA_ARCHS="75;80;86;89;90a;100a;120a;121a"
  echo ">> RUSTLLAMA_CUDA_ARCHS=$RUSTLLAMA_CUDA_ARCHS (aarch64: full GPU set incl. GB10 sm_121a)"
else
  # x86_64: Turing (75), Ampere (80/86), Ada (89), Hopper (90a FP8 TC), DC
  # Blackwell (100a, HGX B200 base), consumer Blackwell (120a RTX 50xx FP4 TC).
  # The accelerated arches let the on-device self-check see + auto-enable the TC
  # paths; sm_90a is a superset of sm_90 so the base Hopper path still runs.
  export RUSTLLAMA_CUDA_ARCHS="75;80;86;89;90a;100a;120a"
  echo ">> RUSTLLAMA_CUDA_ARCHS=$RUSTLLAMA_CUDA_ARCHS (x86_64: Turing..consumer Blackwell + Hopper/Blackwell TC)"
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
