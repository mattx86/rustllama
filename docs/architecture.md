# rustllama Architecture (condensed)

This document is the in-repo architecture summary; update it as the
project evolves.

## Stack

```
┌──────────────────────────────────────────────────────────────┐
│ rustllama-cli  rustllama-server  rustllama-gui (Tauri)       │
└────────────────────────┬─────────────────────────────────────┘
                         │ Engine API
┌────────────────────────┴─────────────────────────────────────┐
│ rustllama-engine                                             │
│   generation loop, sampling, KV cache, chat templates,       │
│   hybrid CPU/GPU placement policy                            │
└──┬───────────────────────────────────┬──────────────────────┘
   │                                   │
┌──┴──────────────────┐    ┌───────────┴───────────────────────┐
│ rustllama-tensor    │    │ rustllama-models                  │
│  Tensor, Device,    │    │  llama_arch (Llama / Qwen2 /      │
│  Dtype, Storage     │    │   DeepSeek / Mistral / Phi3)      │
└──┬─────────────┬────┘    └───────────────────────────────────┘
   │             │
┌──┴──┐    ┌─────┴──────────────────────────────────────────────┐
│ CPU │    │ GPU                                                │
│ ker-│    │ rustllama-kernels-sycl  (our own C++/SYCL .cpp     │
│ nels│    │   built by icx/icpx; extern "C" shims — Intel)     │
│ Rust│    │ rustllama-kernels-cuda  (our own .cu built by      │
│ SIMD│    │   nvcc; extern "C" shims — NVIDIA)                 │
└─────┘    └───────────────────────────────────────────────────┘
   ▲           ▲
   │           │
┌──┴───────────┴────┐
│ rustllama-gguf    │
│  parser, mmap,    │
│  quant decoders   │
└───────────────────┘
```

The kernel layer (`rustllama-kernels-sycl`, `rustllama-kernels-cuda`) is
replaceable without touching anything above `rustllama-tensor`. Future
Vulkan / Metal / ROCm backends will sit beside it.

**Backend / arch notes.** SYCL is **x86_64-only** (Intel oneAPI has no
aarch64 build); on aarch64 `rustllama-kernels-sycl` compiles a no-op stub,
so inference runs on CPU + CUDA. CPU kernels use **AVX2 on x86_64** and
**NEON on aarch64** (scalar fallback elsewhere). The CUDA backend is wired
into the forward pass — packed matvecs dispatch to a device-resident
weight cache on the NVIDIA GPU (inert when none is present). Validate each
backend's kernels against the CPU reference with `rustllama doctor
--cuda-parity` (CUDA) or `--sycl-parity` (SYCL); `--cpu-parity` self-checks the
CPU SIMD / parallel matvec paths against a naive scalar reference.
One physical Intel GPU exposed via both Level Zero and OpenCL is reported
as a single device, and dispatch prefers Level Zero.

## Locked decisions

1. **No llama.cpp.** Inference engine is our own. Kernels are self-contained
   and hand-rolled: the SYCL TU is compiled by Intel's `icx`/`icpx`, the CUDA
   TU by NVIDIA's `nvcc` — no third-party math libraries (no oneMKL/oneDNN).
2. **GGUF v3.** Quantizations supported in v1: F16, Q8_0, Q4_K_M, Q5_K_M.
3. **Tokenizer**: HuggingFace `tokenizers` (Rust-native).
4. **Chat templates**: `minijinja`.
5. **GUI**: Tauri 2 + React + TypeScript + Vite.
6. **HTTP server**: Axum, bind 127.0.0.1, no auth (overridable).
7. **Single binary.** Subcommands: `serve` (`--model`/`--ip`/`--port`),
   `chat` (line REPL or `--tui`; `--resume`, plus `chat list/show/export/
   delete/search` for saved history), `gui`, `generate`, `embed`,
   `model` (a group: `list`/`pull`/`rm`/`use`/`inspect`/`load`/`unload`/
   `default`/`bench`), `config`, `tune` (`--show` for cached state),
   `quantize`, `imatrix`, `kv-calibrate`, `doctor` (`--sycl-parity` /
   `--cuda-parity` / `--cpu-parity` run per-backend kernel parity), `lsp`,
   `version`. Every command and
   subcommand also takes `--help` / a `help` subcommand. Run
   `rustllama <cmd> --help` for the authoritative set.
8. **Autotuner** is a first-class subsystem (`rustllama-tuner`).

## Process model

One shipped binary `rustllama.exe`. Each mode either hosts the engine in-process
or attaches to an already-running server via a port-file lockfile under
`%LOCALAPPDATA%\rustllama\runtime\`.

## Storage paths

| Purpose           | Path                                          |
| ----------------- | --------------------------------------------- |
| Config            | `%APPDATA%\rustllama\config.toml`             |
| Model cache       | `%LOCALAPPDATA%\rustllama\models\`            |
| Tuning cache      | `%LOCALAPPDATA%\rustllama\tuning\`            |
| Runtime / lock    | `%LOCALAPPDATA%\rustllama\runtime\`           |
| Chat history      | `%APPDATA%\rustllama\chat_history`            |
