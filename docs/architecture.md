# rustllama Architecture (condensed)

This document is the in-repo architecture summary; update it as the
project evolves.

## Stack

```
┌──────────────────────────────────────────────────────────────┐
│ rustllama-cli  rustllama-server  rustllama-gui (egui)        │
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
│AVX2/│    │ rustllama-kernels-mlx   (our own .metal + ObjC++   │
│NEON │    │   .mm built by xcrun metal; extern "C" — Apple)    │
└─────┘    └───────────────────────────────────────────────────┘
   ▲           ▲
   │           │
┌──┴───────────┴────┐
│ rustllama-gguf    │
│  parser, mmap,    │
│  quant decoders   │
└───────────────────┘
```

The kernel layer (`rustllama-kernels-sycl`, `rustllama-kernels-cuda`,
`rustllama-kernels-mlx`) is replaceable without touching anything above
`rustllama-tensor`. Each backend compiles to an inert no-op stub on a platform
where its toolchain is absent (SYCL off aarch64/macOS, CUDA off macOS, MLX off
Apple hardware), so the single binary builds everywhere and startup detection
picks what to run. Future Vulkan / ROCm backends will sit beside these.

**Backend / arch notes.** SYCL is **x86_64-only** (Intel oneAPI has no
aarch64 build); on aarch64 `rustllama-kernels-sycl` compiles a no-op stub,
so inference runs on CPU + CUDA. **Metal/MLX** (`rustllama-kernels-mlx`) is the
Apple-Silicon GPU path and is a no-op stub off Apple hardware; on macOS SYCL and
CUDA are both stubs, so real compute is CPU (both arches) + Metal (Apple
Silicon). CPU kernels use **AVX2 on x86_64** and **NEON on aarch64** (the
aarch64 NEON path covers the matvec quants *and* the forward-pass ops —
RMSNorm, RoPE, GQA/flash attention — validated under QEMU; scalar fallback
elsewhere). The CUDA and Metal backends are wired into the forward pass —
packed matvecs dispatch to a device-resident weight cache (inert when no such
GPU is present). NVIDIA builds carry **tensor-core GEMM** paths (Blackwell
sm_120a FP4/FP6/FP8, Hopper sm_90a FP8 `wgmma`) and Intel builds an **XMX/DPAS**
bf16 GEMM — all **auto-enabled per device, no env vars**: the `tune
--validate-kernels` stage runs the parity probes on-device and caches a pass/fail
verdict the dispatch gates read, so a path turns on only where it matched the CPU
reference (fail-closed). Device capability shows as `xmx_capable` in `doctor`.
Validate each backend's kernels against the CPU
reference with `rustllama doctor --cuda-parity` (CUDA), `--sycl-parity` (SYCL,
including an `xmx:gemm` probe), or `--metal-parity` (Metal); `--cpu-parity`
self-checks the CPU SIMD / parallel matvec paths against a naive scalar
reference. One physical Intel GPU exposed via both Level Zero and OpenCL is
reported as a single device, and dispatch prefers Level Zero.

## Locked decisions

1. **No llama.cpp.** Inference engine is our own. Kernels are self-contained
   and hand-rolled: the SYCL TU is compiled by Intel's `icx`/`icpx`, the CUDA
   TU by NVIDIA's `nvcc` — no third-party math libraries (no oneMKL/oneDNN).
2. **GGUF v3.** ~26 quant formats decode in-tree — F16/BF16, the `Q*_0`/`Q*_1`
   and `Q*_K` families, the IQ-family (IQ1/2/3/4), ternary TQ1_0/TQ2_0 plus
   packed PTQ1_0/PQ2_0, and the microscaling MXFP4/6/8 + NVFP4 + FP8 families.
   See [quant-formats.md](quant-formats.md).
3. **Tokenizer**: HuggingFace `tokenizers` (Rust-native).
4. **Chat templates**: `minijinja`.
5. **GUI**: native egui/eframe (glow renderer + winit; rfd file dialogs).
6. **HTTP server**: Axum, binds `127.0.0.1:11434` by default (configurable via
   `serve --ip/--port` or `[server]`). Optional `Authorization: Bearer <token>`
   auth — off unless a key is set (`serve --api-key`, `RUSTLLAMA_API_KEY`, or
   `[server].api_key`); loopback is exempt unless `require_auth_loopback`.
7. **Single binary.** Subcommands: `serve` (`--model`/`--ip`/`--port`),
   `chat` (line REPL or `--tui`; `--resume`, plus `chat list/show/export/
   delete/search` for saved history), `gui`, `generate`, `embed`,
   `model` (a group: `list`/`pull`/`rm`/`use`/`inspect`/`load`/`unload`/
   `default`/`bench`), `config`, `tune` (`--show` for cached state),
   `quantize`, `imatrix`, `kv-calibrate`, `doctor` (`--sycl-parity` /
   `--cuda-parity` / `--metal-parity` / `--cpu-parity` run per-backend kernel
   parity), `lsp`,
   `version`. Every command and
   subcommand also takes `--help` / a `help` subcommand. Run
   `rustllama <cmd> --help` for the authoritative set.
8. **Autotuner** is a first-class subsystem (`rustllama-tuner`).

## Process model

One shipped binary `rustllama`. Each mode either hosts the engine in-process or
attaches to an already-running server via a port-file lockfile in the `runtime/`
state directory (see below).

## Storage paths

State is **binary-relative by default** (`rustllama_runtime::paths()`): a
self-contained folder beside the executable. `config.toml` sits next to the
binary; everything else is a subdirectory of the binary's directory.

| Purpose        | Path (default, binary-relative)      |
| -------------- | ------------------------------------ |
| Config         | `config.toml` (beside the binary)    |
| Model cache    | `models/`                            |
| Tuning cache   | `tuning/`                            |
| Runtime / lock | `runtime/`                           |
| Chat history   | `sessions/`                          |
| Logs           | `logs/`                              |

Opt into OS-standard directories (`%APPDATA%`/`%LOCALAPPDATA%` on Windows, XDG
on Linux, `~/Library` on macOS) by setting the `RUSTLLAMA_SYSTEM_DIRS` env var
or dropping a `system.flag` file next to the binary. `--config` and individual
config keys still override specific paths; a relative `[model].path` resolves
against the process CWD, so prefer an absolute path or a hub ref for the default
model.
