# rustllama

A Rust LLM runtime with hybrid system-RAM + VRAM offload, an
OpenAI-compatible HTTP server, a CLI, and a native egui GUI — all shipped as
one binary, all owned end-to-end (no llama.cpp dependency). Runs on **Windows
and Linux on x86_64, and Linux on aarch64** (e.g. NVIDIA Grace / DGX
Spark). Compute backends are compiled in and selected at startup per the
hardware present: **Intel GPUs via SYCL** (x86_64 only), **NVIDIA GPUs via
CUDA**, and **CPU SIMD** (AVX2 on x86_64, NEON on aarch64).

> **Status:** experimental / pre-1.0. APIs, config keys, and on-disk
> formats may change between releases.

## Build prerequisites

| Tool                       | Version             | Required for                                 |
| -------------------------- | ------------------- | -------------------------------------------- |
| Rust                       | 1.83+ (stable)      | everything                                   |
| Intel oneAPI Base Toolkit  | 2025.0+             | SYCL kernels (`rustllama-kernels-sycl`, icx/icpx) — **x86_64 only** |
| NVIDIA CUDA Toolkit        | 13.x                | CUDA kernels (`rustllama-kernels-cuda`, nvcc; also targets aarch64) |

The GUI is a native **egui/eframe** desktop app (`crates/rustllama-gui`) — it
has **no JS frontend**, so no Node.js / pnpm is required. (The legacy Vite/React
tree at `app/ui/` is unused by the current build and slated for removal.)

The compute kernels are **real-only** — there are no `mock`/`sycl`/`cuda`
cargo features; the backends are always compiled in and startup detection
picks which to run. On **x86_64** this means building requires both Intel
oneAPI (`icx`/`icpx`) and the CUDA Toolkit (`nvcc`). On **aarch64** Intel
oneAPI does not exist, so `rustllama-kernels-sycl` compiles a no-op stub
and only the CUDA Toolkit (`nvcc`) is required — inference then runs on
CPU (NEON) + CUDA. CPU SIMD is chosen automatically per arch.

## Quickstart

```powershell
# Probe the environment
cargo xtask doctor                                 # build-time: are the toolchains present?
target\release\rustllama.exe doctor                # runtime: which backend/devices are live
target\release\rustllama.exe doctor --cuda-parity  # verify CUDA kernels vs the CPU reference (on an NVIDIA host)

# Build the GUI artifact (GUI + CLI + server, all backends). Windows:
scripts\build.bat
scripts\build.bat --headless                       # server-only, no GUI

# Linux builds run in Docker (or on a Linux host directly via scripts/build.sh):
#   scripts\build-linux-gpu-docker.bat             # x86_64 GPU image (oneAPI + CUDA)
#   scripts\build-linux-arm64-gpu-docker.bat       # aarch64 / DGX Spark (CUDA + NEON, SYCL stubbed)

# Run it
target\release\rustllama.exe serve --model model.gguf   # load + auto-tune + serve (OpenAI-compatible API)
target\release\rustllama.exe serve --model a.gguf --model b.gguf   # load several; the first is the default
target\release\rustllama.exe serve --ip 0.0.0.0 --port 11434   # bind host/port (expose on the LAN)
target\release\rustllama.exe gui                        # desktop GUI
target\release\rustllama.exe chat                       # CLI chat against a running `serve`
target\release\rustllama.exe chat --ip 192.168.1.10 --port 11434   # chat to a serve on another host
target\release\rustllama.exe chat --tui                 # full-screen ncurses-style TUI
```

**Zero-config first run.** `serve --model <file>` loads the model, and the
first time it sees a given model on a given device it runs a one-time
auto-tune sweep (kv-dtype coherence, flash-attention, KV layout, CPU/GPU
placement, batch size, threads) and caches the winners — exactly what the
desktop GUI does on first load. Progress streams to the console; later
runs read the cache and skip the sweep. Set `[tuning].auto_tune_on_first_load
= false` to opt out. Pass `--model` more than once to load several models at
startup (each is auto-tuned; the **first** is the server default, the rest are
reachable by id and via `rustllama model default`). `serve` with no `--model` uses
`[model].path` from the config (or starts empty, ready for the GUI / `POST
/v1/models/load`).

## Releases

Prebuilt, **self-contained** archives — unpack and run, no toolchain or
runtime install required. Each archive extracts to a single directory named
after the archive (minus extension) containing the `rustllama` binary, the
bundled runtime libraries it needs, `README.md`, both `LICENSE-*` files, and
`docs/`. Filename format: `rustllama-<version>-<os>-<arch>.<ext>`.

| Archive | OS / distros (x86-64 baseline: v2 / AVX2) | Arch | Contents | Compute backends |
| --- | --- | --- | --- | --- |
| `rustllama-<ver>-windows-x86_64.zip` | Windows 10 / 11 | x86_64 | **GUI** + CLI + server | Intel GPU (SYCL), NVIDIA GPU (CUDA), CPU (AVX2) |
| `rustllama-<ver>-linux-x86_64.tar.gz` | RHEL 10 / Rocky Linux 10, Ubuntu 24.04 (glibc ≥ 2.39) | x86_64 | **headless** (CLI + server) | Intel GPU (SYCL), NVIDIA GPU (CUDA), CPU (AVX2) |
| `rustllama-<ver>-linux-aarch64.tar.gz` | Ubuntu 24.04 (NVIDIA Grace / **DGX Spark**) | aarch64 (ARMv9) | **headless** (CLI + server) | NVIDIA GPU (CUDA), CPU (NEON) |

Notes:

- **Self-contained.** Windows archives bundle the Intel oneAPI SYCL runtime
  DLLs; Linux archives bundle the SYCL runtime `.so`s alongside the binary
  (loaded via an `$ORIGIN`-relative path). CUDA is statically linked. A
  compute backend still needs its hardware + driver at runtime (an NVIDIA
  driver for CUDA; an Intel GPU + its driver for SYCL); absent that, rustllama
  falls back to the CPU path automatically.
- **SYCL is x86-64 only** — the aarch64 build ships a no-op SYCL stub and runs
  on CUDA + CPU (NEON). See [docs/sycl-offload.md](docs/sycl-offload.md).
- **GUI is Windows-only** in the release archives; the Linux archives are
  headless (server + CLI). Run the server anywhere and point a browser-based
  or editor client at it (see [docs/editor-integrations.md](docs/editor-integrations.md)).
- Build these locally with `scripts/release-windows.bat`,
  `scripts/release-linux-x86_64.bat`, and `scripts/release-linux-arm64.bat`
  (Linux archives build in Docker); output lands in `release/`.

## Layout

```
crates/                Library crates (engine + supporting layers, incl. rustllama-gui)
app/desktop/           The shipped binary (CLI + GUI dispatcher)
app/ui/                Legacy Vite + React frontend (unused; from the old Tauri GUI)
xtask/                 Dev tasks (doctor, fetch-test-model, build)
docs/                  Architecture, quant-format reference, redist manifest
```

## License

Licensed under either of [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE) at your option.
