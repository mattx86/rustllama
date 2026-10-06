# rustllama

A Rust LLM runtime with hybrid system-RAM + VRAM offload, an
OpenAI-compatible HTTP server, a CLI, and a native egui GUI — all shipped as
one binary, all owned end-to-end (no llama.cpp dependency). Runs on **Windows
and Linux (x86_64), Linux on aarch64** (e.g. NVIDIA Grace / DGX Spark), **and
macOS** (Apple Silicon + Intel). Compute backends are compiled in and selected
at startup per the hardware present: **Intel GPUs via SYCL** (x86_64 only),
**NVIDIA GPUs via CUDA**, **Apple GPUs via Metal/MLX** (Apple Silicon), and
**CPU SIMD** (AVX2 on x86_64, NEON on aarch64). NVIDIA builds also carry
tensor-core GEMM paths (Blackwell FP4/FP6/FP8, Hopper FP8) and Intel builds an
XMX/DPAS bf16 path; these **auto-enable per device** — the first-load autotune
runs an on-device parity self-check and turns each specialized path on only
where it matches the CPU reference (no env vars, fail-closed).

> **Status:** experimental / pre-1.0. APIs, config keys, and on-disk
> formats may change between releases.

## Build prerequisites

| Tool                       | Version             | Required for                                 |
| -------------------------- | ------------------- | -------------------------------------------- |
| Rust                       | 1.83+ (stable)      | everything                                   |
| Intel oneAPI Base Toolkit  | 2025.0+             | SYCL kernels (`rustllama-kernels-sycl`, icx/icpx) — **x86_64 only** |
| NVIDIA CUDA Toolkit        | 13.x                | CUDA kernels (`rustllama-kernels-cuda`, nvcc; also targets aarch64) |
| Xcode command-line tools   | 15+                 | Metal kernels (`rustllama-kernels-mlx`, `xcrun metal`) — **macOS only** |

The GUI is a native **egui/eframe** desktop app (`crates/rustllama-gui`) — it
has **no JS frontend**, so no Node.js / pnpm is required. (The legacy Vite/React
tree at `app/ui/` is unused by the current build and slated for removal.)

The compute kernels are **real-only** — there are no `mock`/`sycl`/`cuda`
cargo features; the backends are always compiled in and startup detection
picks which to run. On **x86_64** this means building requires both Intel
oneAPI (`icx`/`icpx`) and the CUDA Toolkit (`nvcc`). On **aarch64** Intel
oneAPI does not exist, so `rustllama-kernels-sycl` compiles a no-op stub
and only the CUDA Toolkit (`nvcc`) is required — inference then runs on
CPU (NEON) + CUDA. On **macOS** neither oneAPI nor CUDA exists, so SYCL and
CUDA both compile to inert stubs and the Metal kernels (`rustllama-kernels-mlx`,
built by `xcrun metal`) carry the GPU path on Apple Silicon; the same crate
compiles to a no-op stub off Apple hardware. CPU SIMD is chosen automatically
per arch.

## Quickstart

```powershell
# Probe the environment
cargo xtask doctor                                 # build-time: are the toolchains present?
target\release\rustllama.exe doctor                # runtime: which backend/devices are live
target\release\rustllama.exe doctor --cuda-parity  # verify GPU kernels vs the CPU reference; per backend:
                                                   #   --cuda-parity (NVIDIA), --sycl-parity (Intel, incl. xmx:gemm),
                                                   #   --metal-parity (Apple), --cpu-parity (CPU SIMD self-check)

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
target\release\rustllama.exe serve --api-key sk-secret   # require `Authorization: Bearer sk-secret` on requests
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

## Binding & authentication

The server binds `127.0.0.1:11434` by default (the port matches Ollama's, so
Ollama clients swap in without reconfig). Change it with `serve --ip <addr>
--port <n>` or `[server].bind_addr` / `[server].port` in `config.toml`.

Authentication is **off by default**. Set an API key and the server requires a
matching `Authorization: Bearer <token>` on every request (constant-time
compared) — this is what lets coding agents like **OpenCode**, Continue, Aider,
etc. talk to rustllama over a shared network. Three ways to set it, highest
precedence first:

```powershell
rustllama serve --api-key sk-secret            # 1. CLI flag (wins)
$env:RUSTLLAMA_API_KEY = "sk-secret"; rustllama serve   # 2. env var
#   [server]                                   # 3. config.toml
#   api_key = "sk-secret"
```

`/healthz` is always reachable without a token. Loopback (`127.0.0.1`) requests
are exempt by default so local tools keep working; set
`[server].require_auth_loopback = true` to require the token from local clients
too. See [docs/editor-integrations.md](docs/editor-integrations.md). Coming from
llama.cpp? [docs/llama-cpp-migration.md](docs/llama-cpp-migration.md) maps the
common `llama-cli` / `llama-server` flags (`-fa`, `--temp`, `--top-k`,
`--top-p`, `-c`, …) to their rustllama equivalents.

## Releases

Prebuilt, **self-contained** archives — unpack and run, no toolchain or
runtime install required. Each archive extracts to a single directory named
after the archive (minus extension) containing the `rustllama` binary, the
bundled runtime libraries it needs, `README.md`, both `LICENSE-*` files, and
`docs/`. Filename format: `rustllama-<version>-<os>-<arch>.<ext>`.

| Archive | OS / distros (x86-64 baseline: v2 / AVX2) | Arch | Contents | Compute backends |
| --- | --- | --- | --- | --- |
| `rustllama-<ver>-windows-x86_64.zip` | Windows 10 / 11 | x86_64 | **GUI** + CLI + server | Intel GPU (SYCL), NVIDIA GPU (CUDA), CPU (AVX2) |
| `rustllama-<ver>-linux-x86_64.tar.gz` | RHEL 10 / Rocky Linux 10, Ubuntu 24.04 (glibc ≥ 2.39) | x86_64 | **GUI** + CLI + server | Intel GPU (SYCL), NVIDIA GPU (CUDA), CPU (AVX2) |
| `rustllama-<ver>-linux-aarch64.tar.gz` | Ubuntu 24.04 (NVIDIA Grace / **DGX Spark**) | aarch64 (ARMv9) | **GUI** + CLI + server | NVIDIA GPU (CUDA), CPU (NEON) |
| `rustllama-<ver>-macos-arm64.tar.gz` | macOS (Apple Silicon: M1 and later) | arm64 (aarch64) | **GUI** + CLI + server | Apple GPU (Metal/MLX), CPU (NEON) |
| `rustllama-<ver>-macos-x86_64.tar.gz` | macOS (Intel) | x86_64 | **GUI** + CLI + server | CPU (AVX2) |

Notes:

- **Self-contained.** Windows archives bundle the Intel oneAPI SYCL runtime
  DLLs; Linux archives bundle the SYCL runtime `.so`s alongside the binary
  (loaded via an `$ORIGIN`-relative path). CUDA is statically linked. A
  compute backend still needs its hardware + driver at runtime (an NVIDIA
  driver for CUDA; an Intel GPU + its driver for SYCL); absent that, rustllama
  falls back to the CPU path automatically.
- **SYCL is x86-64 only** — the aarch64 build ships a no-op SYCL stub and runs
  on CUDA + CPU (NEON). See [docs/sycl-offload.md](docs/sycl-offload.md).
- **macOS has neither SYCL nor CUDA** — Intel oneAPI never targeted macOS and
  NVIDIA dropped macOS CUDA ~2019, so both compile to inert stubs. Real compute
  on a Mac is **CPU** (both arches) plus **Metal via MLX on Apple Silicon**
  (Intel Macs are CPU-only). The macOS binary is ad-hoc code-signed (`-s -`);
  an unsigned/ad-hoc binary may need a Gatekeeper override on first run.
- **Every archive ships the GUI.** The single binary is a superset — it launches
  the native egui desktop GUI *and* runs headless as a server/CLI (`rustllama
  serve`), so the aarch64 / DGX Spark archive still works fine on a headless box.
  A GUI launch needs a display + GL at runtime (the host provides mesa/vendor GL,
  X11/Wayland, and an xdg-desktop-portal); absent that, use the server/CLI and
  point a browser-based or editor client at it (see
  [docs/editor-integrations.md](docs/editor-integrations.md)).
- Build these locally with `scripts/release-windows.bat`,
  `scripts/release-linux-x86_64.bat`, and `scripts/release-linux-arm64.bat`
  (Linux archives build in Docker), and `scripts/release-macos.sh {arm64,x86_64}`
  (run **on a Mac**, no Docker); output lands in `release/`.
- **Deploying to a GPU cloud** (RunPod, etc.)? See
  [docs/deploy-runpod.md](docs/deploy-runpod.md) — `scripts/Dockerfile.runpod`
  is a ready runtime image.

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
