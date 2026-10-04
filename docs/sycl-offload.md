# SYCL GPU offload — status & enablement

**TL;DR: SYCL is already wired into the forward pass.** It is *not* an
unintegrated scaffold. The matvec/GEMM hot path, RMSNorm, RoPE, SiLU·mul,
and attention all dispatch to USM-resident SYCL kernels with a transparent
CPU fallback. Getting the speedup is an **operational** task (runtime config
+ an Intel GPU), not a code-integration task — the SYCL backend is always
compiled in.

## Where the dispatch happens

All heavy projections route through `matvec_tensor_dispatch`
([`crates/rustllama-models/src/llama_arch.rs`](../crates/rustllama-models/src/llama_arch.rs)),
which tries, in order, then falls back to CPU:

```
try_matvec_tensor_f16in_usm_f32  ─┐
try_matvec_tensor_usm_f32         ├─ GPU (USM); first to succeed wins
try_matvec_f16_usm_f32           ─┘
k::matvec_tensor                  ── CPU fallback
```

Called for QKV, gate/up (incl. a fused gate-up variant), SSM projections,
`lm_head`, etc. Elementwise/norm ops dispatch the same way via
`crate::accel::try_rmsnorm_usm_f32` / `try_rope_usm_f32` /
`try_silu_mul_usm_f32` inside `forward_one_with_scratch_inner` (the body
`forward_one` and the backend forward delegate to). Attention uses the
`usm_attn_*` flash paths.

## The gates (why it may silently run on CPU)

`try_matvec_tensor_usm_f32` (accel.rs) returns `false` — i.e. falls back to
CPU — unless **all** of these hold:

1. `usm_attn_enabled()` — the USM stream is installed on the worker thread
   and weights are uploaded to USM (engine warmup does this). This requires
   a usable Intel GPU + Level Zero / OpenCL runtime; with none present the
   stream never opens and every dispatch falls back to CPU.
2. `gpu_active_for_current_layer()` — the layer index is below the
   `[inference].n_gpu_layers` cutoff. `n_gpu_layers = 0` ⇒ all-CPU.
3. The tensor is not pinned to CPU via `[inference].placement.overrides`.
4. The matvec clears a **min-FLOPs threshold** (`2·M·K`) — tiny matvecs stay
   on CPU because Level-Zero launch + USM marshaling would dominate.

## Device selection (Level Zero preferred; x86_64-only)

One physical Intel GPU is often exposed **twice** — once via Level Zero,
once via OpenCL. The inventory collapses these to a single device, and
dispatch explicitly **prefers Level Zero** (falling back to OpenCL only
when Level Zero is unavailable). SYCL itself is **x86_64-only** — Intel
oneAPI has no aarch64 build, so on aarch64 `rustllama-kernels-sycl`
compiles a no-op stub, there is no SYCL device, and inference runs on
CPU + CUDA.

## XMX / DPAS tensor-core GEMM (opt-in)

Intel Arc / Data Center GPU Max (Xe-HPG/HPC) expose **XMX** systolic arrays
(DPAS instructions) for matrix-engine GEMM. rustllama has a bf16-compute GEMM
path (`rsl_sycl_gemm_bf16_xmx_f32`, f32 in/out via `joint_matrix`) that runs on
them. It is **off by default** and double-gated:

1. **Build:** the kernel compiles only when the crate is built with
   `RUSTLLAMA_SYCL_XMX=1` (adds `-DRSL_SYCL_XMX`). A normal build omits it.
2. **Runtime:** the model dispatcher uses it only when `RUSTLLAMA_SYCL_XMX=1`
   is *also* set at runtime **and** the device reports XMX capability.

Device capability surfaces as `xmx_capable` in `rustllama doctor` (it is the
SYCL `ext::intel::info::device::uuid` sibling probe — Xe-LP iGPUs like Iris Xe
report `false`, having no XMX). Validate the kernel against the CPU reference
with the `xmx:gemm` probe under `rustllama doctor --sycl-parity`; it SKIPs
cleanly on a non-XMX device or a build without the define. Leave XMX off unless
you have the hardware and have confirmed parity on it.

## How to actually use it (faster without new hardware *if you have an Intel GPU*)

If this machine has an idle Intel iGPU (Xe/UHD) or Arc, offloading to it is a
real speedup with no hardware purchase. The SYCL backend is always compiled
in, so there's nothing to enable at build time — you just need the hardware +
runtime present and a runtime config that allows GPU layers:

1. Have an Intel GPU with its driver + the Level Zero (or OpenCL) runtime.
   Building from source additionally needs the oneAPI Base Toolkit
   (`icx`/`icpx`, via `scripts/build-env.bat`, which drops the
   `+crt-static` RUSTFLAG the SYCL link requires); a prebuilt binary already
   contains the kernels — see [`docs/oneapi-redist.md`](./oneapi-redist.md).
2. Runtime config:
   ```toml
   [inference]
   device = "sycl:0"
   n_gpu_layers = 999        # 999 (default) = AUTO placement; set a
                             # specific count only to override the planner
   ```
3. Validate the kernels on *this* GPU before trusting output — the project
   sequences **parity before integration** for exactly this reason. Run the
   accelerator parity tests (they compare each SYCL kernel against the CPU
   reference within tolerance) on the target hardware. A passing CPU
   fallback can otherwise mask a GPU kernel that produces wrong logits.

## Remaining gaps (minor)

- A few non-default forward variants and some prefill sub-paths still call
  `k::rmsnorm_f32_row` directly rather than through `try_rmsnorm_usm_f32`.
  The *heavy* compute (matvec) already dispatches everywhere via
  `matvec_tensor_dispatch`, so these are second-order. Filling them is a
  per-site `if !try_*_usm() { k::cpu() }` edit, gated behind the existing
  parity tests.
- The AUTO placement planner (the default, unless `n_gpu_layers` is set to
  an explicit override) decides the CPU/GPU layer split; verify its measured
  cutoff on the target box with `rustllama tune`.

_Probe performed 2026-06-23. Corrects an earlier assessment that claimed
SYCL was not dispatched from the forward pass._
