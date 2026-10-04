# CLAUDE.md — rustllama

Guidance for Claude Code (and humans) working in this repo. Read this first; it
captures the non-obvious build rules, architecture, and gotchas that aren't
derivable from a quick look at the tree.

## What this is

**rustllama** is a from-scratch Rust LLM runtime — an OpenAI/Ollama-compatible
inference server + CLI + native egui GUI. It is **clean-room: no `llama.cpp`
dependency**. It owns its whole compute stack: a real-only kernel layer over
**Intel SYCL + NVIDIA CUDA + Apple Metal/MLX + CPU**, GGUF/safetensors loaders,
tokenizers, chat templating, a KV cache, sampling, an autotuner, and
quantization tooling.

- Targets: **Windows x64, Linux x64, Linux aarch64** (DGX Spark / GB10),
  **macOS arm64 (Apple Silicon) + x86_64 (Intel)**.
- License: **MIT OR Apache-2.0**, © 2026 Matt Smith (`LICENSE-MIT`,
  `LICENSE-APACHE`). Application workspace — every crate is `publish = false`.
- Status: experimental / pre-1.0 (`version = 0.1.0`). Edition 2021,
  `rust-version = 1.83`.

## Repo layout

```
crates/                         (21-crate Cargo workspace)
  rustllama-tensor              tensor types / dtype defs
  rustllama-gguf                GGUF read/write, quant encode/decode (26 formats)
  rustllama-safetensors         AWQ/GPTQ safetensors load path
  rustllama-tokenizer           tokenizers wrapper + chat detok (Utf8Stream)
  rustllama-kernels-cpu         SIMD CPU kernels (AVX2/AVX-512 x86, NEON aarch64, rayon matmul)
  rustllama-kernels-sycl        SYCL kernels (icx/icpx; x86 only — no-op stub on aarch64)
  rustllama-kernels-cuda        native CUDA kernels (nvcc; inert off-NVIDIA; Blackwell/Hopper TC)
  rustllama-kernels-mlx         Metal/MLX kernels (xcrun metal + ObjC++; inert off Apple)
  rustllama-l0-sys              Level-Zero Sysman FFI (power/energy telemetry)
  rustllama-models              model archs, forward pass, device dispatch (accel.rs)
  rustllama-tuner               device fingerprint + autotune cache
  rustllama-engine              generation loop, sampling, KV cache, placement
  rustllama-config              config.toml schema + load/validate
  rustllama-hub                 HuggingFace hub download/cache
  rustllama-runtime             paths, server-record lockfile, crash handling
  rustllama-server              axum HTTP server (OpenAI + Ollama APIs, bearer auth)
  rustllama-client              HTTP client used by chat/lsp
  rustllama-gui                 native egui/eframe GUI (glow + winit; rfd dialogs)
  rustllama-rag                 code-aware RAG indexer (tree-sitter, opt-in)
  rustllama-lsp                 LSP bridge (inline completion via FIM)
  rustllama-cli                 the `rustllama` binary — all subcommands
app/
  desktop/                      the `rustllama` binary — CLI/GUI dispatcher; embeds the server + native egui GUI
  ui/                           legacy React frontend (pnpm) — unused by the native egui GUI; slated for removal
scripts/                        build + release wrappers (see below)
docs/                           architecture.md, quant-formats.md, sycl-offload.md, ...
xtask/                          workspace automation
release/                        release artifact staging output
```

## Build & run  ← read this before building

The compute kernels are **real-only and always compiled** — there are **no
`sycl`/`cuda`/`mock` cargo features**. Backends are detected at startup. This
means:

- Building any crate that transitively touches the kernel layer requires
  **Intel oneAPI (`icx`/`icpx`) AND the CUDA Toolkit (`nvcc`) on PATH**.
- `scripts\build-env.bat <cargo args…>` sets up that env (MSVC vcvars + oneAPI
  setvars + auto-detected `CUDA_PATH`) and then runs cargo. **Always build the
  GPU-dependent crates through it**, never bare `cargo build`.

**Canonical commands** (run from repo root; on Windows invoke the `.bat`
wrappers via the PowerShell tool, not a long `cmd /c` string — the harness
mangles long batch strings):

| Goal | Command |
|------|---------|
| GUI release artifact (native egui) | `scripts\_build.bat` |
| Headless server/CLI build | `scripts\_build_headless.bat` |
| Arbitrary cargo cmd with the full toolchain env | `scripts\build-env.bat build --release …` |
| Check a **toolchain-free** crate (no env needed) | `cargo check -p rustllama-runtime` |

The **8 toolchain-free crates** (no transitive `kernels-sycl`/`kernels-cuda`
dep) can be built/checked/tested with plain cargo, and are the only ones CI
compiles: `rustllama-config`, `-tensor`, `-tokenizer`, `-gguf`,
`-kernels-cpu`, `-l0-sys`, `-runtime`, `-hub`. Everything else (models,
engine, server, cli, tuner, app, and the crates that pull them) needs the GPU
toolchains and is validated on a dev box with both installed + the
`scripts/Dockerfile.linux-gpu` image. Source of truth: `.github/workflows/ci.yml`.

**Reading build results:** PowerShell `... | Tee-Object | Select-Object` masks
cargo's exit code — don't trust `$LASTEXITCODE` through a pipe. Redirect to a
log (`cmd /c "scripts\_build.bat > build.log 2>&1"`) and scan for `error[`. The
8-line vcvars `'M' is not recognized` noise from `build-env.bat` is **benign**,
not a failure.

**Process hygiene:** before launching `rustllama.exe` to actually *run*
inference, kill stale `rustllama` + `cmd` processes — a leftover server holds
RAM and OOMs the new run. (This does **not** apply before a *build*; killing
`cmd.exe` mid-session also slows subsequent `cmd /c` spawns.)

**Profiles:** `release` keeps `debug = 1` (so `cdb.exe` can symbolize hangs);
`dist` (`cargo build --profile dist`) is the ship profile (fat LTO, stripped).
**Never set `panic = "abort"`** — the embedded server thread relies on
`catch_unwind`.

## Kernel layer & device model

- **SYCL, CUDA, and Metal GPUs are first-class equals at the surface.** CPU is
  the 2nd-place fallback used only when no usable GPU is present. The
  **tuner/placement layer decides** actual device use from there, weighted by
  measured **performance and VRAM** — including on mixed SYCL+CUDA or
  multi-CUDA-of-varying-perf systems, with true cross-GPU layer distribution as
  the target (see `memory`/plan for the multi-GPU roadmap).
- matvec dispatch (`rustllama-models/src/accel.rs`) currently routes to ONE
  active backend: CUDA (`cuda_active()`) first if a usable NVIDIA GPU exists,
  else the SYCL device, else the Metal/MLX device (Apple Silicon), else CPU —
  gated by `n_gpu_layers`. Opt-in tensor-core GEMM paths layer on top: CUDA
  Blackwell FP4/FP6/FP8 (`RUSTLLAMA_FP4_TC=1`) + Hopper FP8 `wgmma`
  (`RUSTLLAMA_FP8_WGMMA=1`), and SYCL Intel XMX/DPAS bf16 (`RUSTLLAMA_SYCL_XMX=1`,
  also a build-time define) — all off by default.
- VRAM-fit placement (`rustllama-engine/src/placement_auto.rs`,
  `auto_n_gpu_layers`) budgets **CUDA-first** (mirrors dispatch), else SYCL,
  else all-CPU.
- Prefer-L0-over-OpenCL is only a **backend-view collapse** for enumeration; it
  is NOT a device preference — placement chooses the device.
- Devices are fingerprinted by **driver-invariant UUID** (SYCL
  `ext::intel::info::device::uuid`, CUDA `cudaDeviceProp.uuid`).
  `rustllama_tuner::system_fingerprint()` dedups GPUs across backends + CPU
  cores + RAM into one FNV-1a slug used as the tuner-cache key.

## Paths & state (binary-relative by default)

`rustllama_runtime::paths()` resolves all state **relative to the running exe by
default** (self-contained folder):

- `config.toml` sits **beside** the binary; `models/`, `tuning/`, `sessions/`,
  `runtime/`, `logs/` are subdirectories.
- Opt out to OS standard dirs (`%APPDATA%`/`%LOCALAPPDATA%`, XDG) via the
  `RUSTLLAMA_SYSTEM_DIRS` env var or a `system.flag` file next to the exe.
- `--config` and individual config keys still override specific paths. A
  relative `[model].path` resolves against the process CWD (not `models/`), so
  prefer absolute or a hub ref for the default model.

## CLI surface (`rustllama <cmd>`)

Top-level commands (each parent auto-provides a `help` subcommand; `/`-prefixed
commands in the chat REPL/TUI mirror this verb set):

- `serve` — start the HTTP server. `--model <gguf>` (repeatable; first = default),
  `--ip`, `--port`, `--api-key <token>` (requires `Authorization: Bearer`; flag >
  `RUSTLLAMA_API_KEY` > `[server].api_key`). Auto-tunes each model on first load.
- `chat` — interactive REPL, or full-screen `--tui`. `--resume <id|name>`,
  `--model`, `--system`, `--base-url`/`--ip`/`--port`. History subcommands:
  `chat list|show|export|delete|search`.
- `generate` / `embed` — one-shot `/v1/completions` / `/v1/embeddings`.
- `model` — `list|pull|rm|use|inspect|load|unload|default|bench`. `use` sets the
  config default; `load`/`unload`/`default` manage the *running server's* loaded
  set (can't `unload` the current default). `bench` = synthetic prefill+decode.
- `config` — manage the on-disk config.
- `tune` — autotuner. `tune --all` runs every sweep (coordinate-descent);
  `tune --show` prints tuner-cache state. Individual sweeps: `--kv-dtype`,
  `--flash-attention`, `--kv-layout`, `--placement`, `--batch-size`,
  `--threads`, `--moe-placement`, `--flash-kv-min`, etc.
- `quantize` / `imatrix` / `kv-calibrate` — quantization + calibration tooling.
- `doctor` — diagnostics. `--sycl-smoke`, `--sycl-parity`, `--cuda-parity`,
  `--metal-parity`, `--cpu-parity` run the kernel parity/stability harnesses
  against the CPU reference (`--sycl-parity` subprocesses each probe — including
  an `xmx:gemm` probe — so a DEVICE_LOST only kills its child; `--cuda-parity`,
  `--metal-parity`, and `--cpu-parity` run in-process). Also reports per-device
  `xmx_capable` (Intel XMX/DPAS capability).
- `gui` — launch the native egui GUI. `lsp` — LSP bridge over stdio. `version`.

## Config & server

- `config.toml` sections: `[model]`, `[inference]` (n_gpu_layers, ctx_size,
  batch_size, kv_dtype/k_dtype/v_dtype, flash_attention, speculative_ngram,
  moe_placement, lock_ram_mb, `[inference.placement]`), `[server]`
  (bind_addr, **port 11434** default, api_key, require_auth_loopback,
  max_loaded_models), `[ui]`,
  `[hub]`, `[tuning]` (auto_tune_on_first_load, auto_apply_* flags),
  `[embeddings]`, `[reranker]`.
- The server speaks the **OpenAI** API (`/v1/chat/completions`, `/v1/completions`,
  `/v1/embeddings`, `/v1/models`), the **Anthropic Messages** API
  (`/v1/messages`, for Claude Code), and the **Ollama** API (`/api/*`), plus
  `/healthz`, `/v1/capabilities`, `/v1/tuning_summary`, and model
  load/default/unload endpoints. Server tracing goes to **stdout**, not stderr.
- The GUI embeds the server in-process; it sets `RUSTLLAMA_GUI_EMBEDDED=1` so
  the embedded start skips the blocking first-load autotune (that autotune runs
  as a separate subprocess with a progress modal instead).

## Autotuner

Per-(system, model) first-load autotune (`rustllama-server/src/autotune.rs`):
on a model's first load, if untuned, spawn `tune --all` as a subprocess (GUI
shows a progress modal; CLI echoes to console). A coherence guardrail can force
KV→f32. Winners persist to the tuner cache keyed by `system_fingerprint()` (so
they work on SYCL/CUDA/CPU hosts alike) and auto-apply on next load per the
`[tuning].auto_apply_*` flags.

## Testing & validation

- `cargo test --lib -p <crate>` for the 8 toolchain-free crates (what CI runs).
- Kernel correctness: `rustllama doctor --sycl-parity` (SYCL, per-probe
  subprocessed) / `--cuda-parity` (CUDA, in-process) / `--metal-parity` (Metal,
  in-process) — every GPU kernel family vs its CPU reference on identical
  inputs. `--cpu-parity` self-checks the CPU matvec SIMD / rayon-parallel paths
  against a naive scalar reference on this host (f32, f16, PTQ1_0 ternary
  fastdot/batched). The aarch64 NEON forward-pass ports (RMSNorm/RoPE/GQA/flash)
  carry their own scalar-parity `#[test]`s, validated under QEMU.
- GPU/multi-GPU code that can't be exercised on available hardware is validated
  by parity harness + review; on-device confirmation happens on the user's HW.
- Windows hang debugging: attach `cdb.exe` non-invasively for symbolized stacks;
  sample the CPU delta before bisecting. Never write a self-recursive
  `OnceLock` init (caused a past deadlock).

## Release builds

Five self-contained artifacts, `rustllama-{version}-{os}-{arch}.{ext}`:

- Windows x64 `.zip` (GUI) — `scripts\release-windows.bat`
- Linux x64 `.tar.gz` (headless) — `scripts\release-linux-x86_64.bat`
- Linux aarch64 `.tar.gz` (headless, DGX Spark / GB10) —
  `scripts\release-linux-arm64.bat` (Docker/QEMU, SBSA CUDA repo)
- macOS arm64 (Apple Silicon) `.tar.gz` (GUI) — `scripts/release-macos.sh arm64`
- macOS x86_64 (Intel) `.tar.gz` (GUI) — `scripts/release-macos.sh x86_64`

The **two macOS artifacts are built ON A MAC** (no Docker): `release-macos.sh`
drives `build-macos.sh` (native egui GUI; CPU real on both arches, MLX/Metal
real on Apple Silicon; **SYCL + CUDA compile to inert stubs** — neither oneAPI
nor nvcc exists on macOS). Cross-build the non-native arch with `--target`
(`rustup target add …`; the C stubs cross-compile via clang `-arch`).

Each bundles the binary + its non-system shared libs + README + LICENSE + docs
in a nested dir. Linux uses patchelf `$ORIGIN` RPATH so a single `rustllama`
binary finds its `.so`s; macOS uses the Mach-O analogue — `otool -L` to walk
dylib deps, `install_name_tool` to rewrite them to `@rpath` + add an
`@loader_path/../lib` rpath, then an ad-hoc `codesign -s -` (a real Developer
ID can replace `-` later). Packaging helpers: `_release_package.ps1` (bsdtar),
`_release_package.sh`, `_release_build_linux.sh`; macOS packaging is inline in
`release-macos.sh`.

## Useful environment variables

- `RUSTLLAMA_SYSTEM_DIRS` — use OS dirs instead of binary-relative paths.
- `RUSTLLAMA_API_KEY` — bearer token for `[server].api_key` (env tier: flag >
  env > config). Non-empty ⇒ `Authorization: Bearer` required (loopback exempt
  unless `[server].require_auth_loopback`).
- `RUSTLLAMA_GUI_EMBEDDED` — set by the GUI; skips the blocking startup autotune.
- `RUSTLLAMA_DISABLED_GPUS` — exclude specific GPUs from selection.
- `RUSTLLAMA_IQ_GPU=1` — run IQ1_S imatrix weighting on the GPU during quantize.
- Opt-in tensor-core / matrix-engine GEMM (all default-off): `RUSTLLAMA_FP4_TC=1`
  (CUDA Blackwell sm_120a FP4/FP6/FP8), `RUSTLLAMA_FP8_WGMMA=1`
  (+`RUSTLLAMA_FP8_WGMMA_TMA=1`; Hopper sm_90a FP8), `RUSTLLAMA_SYCL_XMX=1`
  (Intel XMX/DPAS bf16 — also a build-time define).
- Perf levers (production CPU-path recipe): `SYCL_DISPATCH=0`, `LOCK_RAM_MB`
  (also an `[inference]` key), `TERNARY_FASTDOT=1`, `PROFILE_HYBRID_PREFILL`.

## Conventions

- Match the surrounding code's comment density, naming, and idioms. This tree
  comments the *why* (hardware quirks, format constraints) heavily — keep that.
- Keep committed files free of personal paths / machine names (the repo was
  scrubbed for public release).
- No `llama.cpp`, no copyleft deps — the runtime is clean-room and permissively
  licensed; keep it that way.
