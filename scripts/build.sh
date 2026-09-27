#!/usr/bin/env bash
# Standardized rustllama build/test for Linux — the counterpart to
# scripts/build.bat on Windows. All THREE compute backends (Intel SYCL +
# NVIDIA CUDA + CPU) are always compiled into the binary; there are no
# per-backend features. Startup detection picks which to run (or all):
# Intel via SYCL/Level-Zero, NVIDIA via CUDA, CPU always.
#
#   scripts/build.sh                 GUI artifact (desktop GUI + CLI + server)
#   scripts/build.sh --headless      server-only build (no Tauri GUI)
#   scripts/build.sh test            run the CPU test suites
#   scripts/build.sh --debug         debug profile
#   scripts/build.sh -- <args>       pass extra args to cargo
#
# GUI is the default artifact so users can launch the desktop GUI OR use
# the CLI/server from the same binary. The GUI links GTK/webkit2gtk-4.1
# (present in the build image) and embeds the prebuilt frontend at
# app/ui/dist. That dist is platform-independent — build it once with
# `pnpm -C app/ui install && pnpm -C app/ui build` (Node 20+); this script
# consumes it (the build image needs no Node). It errors if dist is absent.
#
# Because all backends are built in, this REQUIRES both GPU toolchains:
# Intel oneAPI (icx/icpx) and the CUDA Toolkit (nvcc). Run it inside the
# dedicated GPU image (scripts/Dockerfile.linux-gpu) — from Windows via
# scripts/build-linux-gpu-docker.bat — which has both.
#
# Linking: Rust links all crates + bundled C libs (sqlite, oniguruma,
# rustls crypto, tree-sitter) + the nvcc kernels + static cudart into the
# binary. Dynamic deps: glibc + core system libs, the SYCL runtime, the
# NVIDIA driver, and (unless --headless) GTK/webkit2gtk-4.1. No musl, no
# static glibc.
set -euo pipefail
cd "$(dirname "$0")/.."

# Both GPU backends compile in, so make both toolchains usable here:
#   - Intel oneAPI: source setvars.sh → puts icx/icpx on PATH and the SYCL
#     runtime on LD_LIBRARY_PATH (the SYCL build.rs invokes icpx directly).
#   - NVIDIA CUDA: nvcc is located via CUDA_PATH by the kernels-cuda
#     build.rs; the GPU image already exports CUDA_PATH + PATH.
# setvars.sh references many optional vars, so relax `nounset` around it.
if [ -f /opt/intel/oneapi/setvars.sh ]; then
  set +u
  . /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
  set -u
fi

cmd="build"; profile="release"; gui=1; extra=()
while [ $# -gt 0 ]; do
  case "$1" in
    build|test|check|clean) cmd="$1"; shift ;;
    --debug)    profile="debug";   shift ;;
    --release)  profile="release"; shift ;;
    --gui)      gui=1;             shift ;;
    --headless) gui=0;             shift ;;
    --) shift; extra+=("$@"); break ;;
    *) extra+=("$1"); shift ;;
  esac
done

pflag=(); [ "$profile" = "release" ] && pflag=(--release)
outdir="${CARGO_TARGET_DIR:-target}/$([ "$profile" = release ] && echo release || echo debug)"
APP=(--manifest-path app/src-tauri/Cargo.toml)

# SYCL + CUDA + CPU are always compiled in (non-optional deps), so no
# backend features are passed. GUI (the default) adds the Tauri desktop
# shell and embeds the prebuilt app/ui/dist; --headless drops it for a
# server-only binary.
featflag=()
if [ "$gui" = "1" ]; then
  featflag=(--features gui)
  if { [ "$cmd" = "build" ] || [ "$cmd" = "check" ]; } && [ ! -e app/ui/dist/index.html ]; then
    echo "ERROR: GUI build needs the prebuilt frontend at app/ui/dist." >&2
    echo "       Build it first (Node 20+): pnpm -C app/ui install && pnpm -C app/ui build" >&2
    echo "       Or build server-only:      scripts/build.sh --headless" >&2
    exit 1
  fi
fi

case "$cmd" in
  build)
    echo ">> rustllama build ($profile)${featflag[*]:+ — ${featflag[*]}}"
    cargo build "${pflag[@]}" "${APP[@]}" "${featflag[@]}" "${extra[@]}"
    echo ">> built: $outdir/rustllama"
    file "$outdir/rustllama" 2>/dev/null | cut -d, -f1-3 || echo "WARN: binary not found"
    echo ">> shared-library dependencies (ldd):"
    ldd "$outdir/rustllama" 2>&1 | sed 's/^/     /' | head -16 || true
    ;;
  check)
    echo ">> rustllama check ($profile)${featflag[*]:+ — ${featflag[*]}}"
    cargo check "${pflag[@]}" "${APP[@]}" "${featflag[@]}" "${extra[@]}"
    ;;
  test)
    echo ">> rustllama test ($profile, Linux CPU suites)"
    cargo test "${pflag[@]}" \
      -p rustllama-kernels-cpu -p rustllama-config -p rustllama-tensor \
      -p rustllama-tokenizer -p rustllama-safetensors \
      -p rustllama-models -p rustllama-engine -p rustllama-server \
      "${extra[@]}"
    ;;
  clean)
    cargo clean "${extra[@]}"
    ;;
esac
