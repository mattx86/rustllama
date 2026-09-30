#!/usr/bin/env bash
# Standardized rustllama build/test for macOS — the counterpart to
# scripts/build.sh (Linux) and scripts/build.bat (Windows).
#
#   scripts/build-macos.sh                          GUI artifact (native egui)
#   scripts/build-macos.sh --headless               server-only build (no GUI)
#   scripts/build-macos.sh test                     run the CPU test suites
#   scripts/build-macos.sh --debug                  debug profile
#   scripts/build-macos.sh --target x86_64          cross-build the other arch
#   scripts/build-macos.sh -- <args>                pass extra args to cargo
#
# The macOS compute stack — WHY this wrapper has no oneAPI/CUDA setup:
#   * CPU (real): AVX2 on Intel Macs, NEON on Apple Silicon — always compiled.
#   * MLX / Metal (real, Apple Silicon only): the GPU tier, crates/
#     rustllama-kernels-mlx. Phase 0 is an inert stub; a live backend lands in
#     Phase 1 (set MLX_LIB_DIR / MLX_INCLUDE_DIR then — see below).
#   * SYCL and CUDA: STUBBED on every Mac. Intel oneAPI (icx/icpx) never
#     targeted macOS, and NVIDIA dropped macOS CUDA ~2019 and never shipped it
#     for Apple Silicon — so those kernel crates compile to inert no-op shims
#     (kernels-sycl/build.rs + kernels-cuda/build.rs detect TARGET_OS==macos).
#     There is therefore NOTHING to source here: no oneAPI setvars, no
#     CUDA_PATH. Just cargo + the Xcode Command Line Tools (clang/ld/otool).
#
# GUI is the default artifact (the single binary also runs the CLI/server).
# The GUI is native egui/eframe (crates/rustllama-gui) — eframe/winit/glow +
# rfd — which builds natively on macOS against Cocoa/Metal/QuartzCore system
# frameworks. No JS frontend, nothing to prebuild. --headless drops it.
set -euo pipefail
cd "$(dirname "$0")/.."

# Phase-1 MLX link passthrough: if the caller exports MLX_LIB_DIR /
# MLX_INCLUDE_DIR (the Apple mlx-c SDK), let them reach the kernels-mlx
# build.rs. Harmless in Phase 0 (the shim doesn't reference libmlx yet); we
# simply re-export whatever is already set so `cargo` inherits it.
export MLX_LIB_DIR="${MLX_LIB_DIR:-}"
export MLX_INCLUDE_DIR="${MLX_INCLUDE_DIR:-}"

cmd="build"; profile="release"; gui=1; arch=""; extra=()
while [ $# -gt 0 ]; do
  case "$1" in
    build|test|check|clean) cmd="$1"; shift ;;
    --debug)    profile="debug";   shift ;;
    --release)  profile="release"; shift ;;
    --gui)      gui=1;             shift ;;
    --headless) gui=0;             shift ;;
    # Accept the target arch as arm64|aarch64|x86_64 OR a full *-apple-darwin
    # triple. A cross-build needs the Rust std for that target installed
    # (`rustup target add {aarch64,x86_64}-apple-darwin`); the C stubs
    # cross-compile automatically because build.rs passes clang `-arch` to
    # match CARGO_CFG_TARGET_ARCH.
    --target)   arch="$2"; shift 2 ;;
    --target=*) arch="${1#--target=}"; shift ;;
    --) shift; extra+=("$@"); break ;;
    *) extra+=("$1"); shift ;;
  esac
done

# Resolve an explicit --target to a full Rust triple. Empty `arch` = native
# build (no --target flag → host triple, output under target/<profile>).
triple=""
if [ -n "$arch" ]; then
  case "$arch" in
    arm64|aarch64)              triple="aarch64-apple-darwin" ;;
    x86_64|x86-64|intel)        triple="x86_64-apple-darwin" ;;
    aarch64-apple-darwin|x86_64-apple-darwin) triple="$arch" ;;
    *) echo "ERROR: unknown --target '$arch' (want arm64|x86_64 or a *-apple-darwin triple)" >&2; exit 2 ;;
  esac
fi

pflag=(); [ "$profile" = "release" ] && pflag=(--release)
tflag=(); [ -n "$triple" ] && tflag=(--target "$triple")
# cargo nests the target dir under the triple when --target is passed.
base="${CARGO_TARGET_DIR:-target}"
if [ -n "$triple" ]; then
  outdir="$base/$triple/$([ "$profile" = release ] && echo release || echo debug)"
else
  outdir="$base/$([ "$profile" = release ] && echo release || echo debug)"
fi
APP=(--manifest-path app/desktop/Cargo.toml)

# SYCL + CUDA + CPU (+ MLX on Apple Silicon) are all compiled in (non-optional
# deps); no backend features are passed. GUI (the default) adds the native
# egui/eframe desktop shell (crates/rustllama-gui); --headless drops it for a
# server-only binary.
featflag=()
if [ "$gui" = "1" ]; then
  featflag=(--features gui)
fi

case "$cmd" in
  build)
    echo ">> rustllama build ($profile${triple:+, $triple})${featflag[*]:+ — ${featflag[*]}}"
    cargo build "${pflag[@]}" "${tflag[@]}" "${APP[@]}" "${featflag[@]}" "${extra[@]}"
    echo ">> built: $outdir/rustllama"
    file "$outdir/rustllama" 2>/dev/null || echo "WARN: binary not found"
    echo ">> shared-library dependencies (otool -L):"
    otool -L "$outdir/rustllama" 2>&1 | sed 's/^/     /' | head -20 || true
    ;;
  check)
    echo ">> rustllama check ($profile${triple:+, $triple})${featflag[*]:+ — ${featflag[*]}}"
    cargo check "${pflag[@]}" "${tflag[@]}" "${APP[@]}" "${featflag[@]}" "${extra[@]}"
    ;;
  test)
    echo ">> rustllama test ($profile, macOS CPU suites)"
    cargo test "${pflag[@]}" "${tflag[@]}" \
      -p rustllama-kernels-cpu -p rustllama-config -p rustllama-tensor \
      -p rustllama-tokenizer -p rustllama-safetensors \
      -p rustllama-models -p rustllama-engine -p rustllama-server \
      "${extra[@]}"
    ;;
  clean)
    cargo clean "${extra[@]}"
    ;;
esac
