# NPU support + heterogeneous GPU/Metal + NPU + CPU scheduling — design

> **Audience:** rustllama maintainers. This is a design-of-record for an
> unbuilt, write-blind subsystem (no NPU hardware in the dev/validation set as
> of 2026-10-13). It consolidates the requirements and proposes an
> architecture to review *before* implementation. Nothing here is built yet.

## 1. Requirements (owner, 2026-10-13)

1. **Full NPU support** — Intel NPU (Core Ultra / Meteor–Arrow Lake) via
   **OpenVINO** (Apache-2.0) and AMD NPU (Ryzen AI / XDNA) via the **Ryzen AI /
   XRT** stack. Owner decision: **take the vendor-runtime dependency** (the
   clean-room "owns its whole compute stack" rule is relaxed for NPU only,
   because NPUs are not ISA-programmable — a vendor runtime is the only path).
   Feature-gated. Hardware: **none on hand → write-blind**, validated later on
   real NPUs.
2. **NPU device management, mirroring GPUs** — NPUs must be **enumerated**,
   **disable-able by index** (an analog of `RUSTLLAMA_DISABLED_GPUS` /
   `[inference.placement].disabled_gpus`), and **folded into
   `rustllama_tuner::system_fingerprint()`** so the tuner cache is keyed by the
   NPU set too.
3. **Full Intel + AMD CPU performance extensions** — AVX-VNNI / AVX-512-VNNI
   (int8) and AVX-512-BF16, plus AMX (Intel Xeon). *(In progress separately;
   see `vnni.rs`, `amx.rs`. Not part of this doc beyond being a scheduler tier.)*
4. **Cross-tier prioritization + spillover** across **GPU/Metal + NPU + CPU** —
   an explicit priority order with **spillover** to the next tier when one is
   saturated (VRAM / NPU memory / compute).

## 2. Why op/layer-offload, not whole-graph

The spillover requirement (4) is the deciding constraint. Vendor NPU runtimes
are *graph compilers*: the native-fit path is to compile and run a whole model
graph on the NPU. But a whole-graph offload is **all-or-nothing** — it cannot
move *part* of the work to another tier, so it cannot satisfy spillover, and it
would bypass rustllama's GGUF weights, KV cache, sampler, and kernels (a
parallel inference stack).

Therefore the NPU joins as **another device tier in the existing per-tensor /
per-layer placement model**, and we **offload individual ops** (matmul first,
then attention) to it. This reuses the machinery already in the tree:

- `rustllama_models::accel::MultiGpuPlan` + `device_for(tensor, layer)` — the
  per-tensor device router (today CUDA multi-device).
- The MoE **tiered-expert** promotion (device-resident experts) + the new
  column-parallel TP (`try_matvec_tp_cuda`).
- `placement_auto::auto_n_gpu_layers` — the VRAM-fit layer budgeter
  (CUDA-first → SYCL → CPU).

The cost to watch: vendor runtimes have **per-submission overhead** and compile
graphs, so a *per-op* call is only worth it when amortized. Mitigation: a
**compiled-op cache** (compile each distinct (op, shape, dtype) once, reuse the
executable), and route to the NPU only ops/layers large and sustained enough to
pay the submit cost — decided by the autotuner, like every other placement
choice here.

## 3. Architecture

### 3.1 NPU backend crate(s)
- `rustllama-npu-openvino` (Intel, feature `npu-openvino`) and
  `rustllama-npu-ryzenai` (AMD, feature `npu-ryzenai`), each FFI to its vendor
  runtime, each **inert / compiled-out** when its feature is off — mirroring how
  `rustllama-kernels-cuda` is inert off NVIDIA. Clean-room elsewhere is
  preserved; only these feature-gated crates link a vendor runtime.
- A backend-agnostic `NpuDevice` trait: `enumerate() -> Vec<NpuInfo>`,
  `matmul(...)`, (later) `attention(...)`, with a **compiled-op cache** keyed by
  (op, shapes, dtype). First op: the large packed/f32 matmul (LM head, FFN/attn
  projections) — the same ops the GPU tiers accelerate.

### 3.2 Device model (mirror GPUs — requirement 2)
- **Enumeration**: each NPU backend enumerates its devices; results fold into a
  single device inventory beside SYCL/CUDA/Metal GPUs + CPU.
- **Unified index space**: today `cuda_device_disabled` offsets CUDA indices by
  `distinct_sycl_gpu_count()`. Extend that scheme so NPUs occupy the next range
  after the GPUs; `[inference.placement].disabled_npus` + `RUSTLLAMA_DISABLED_NPUS`
  (comma-separated), exactly like `disabled_gpus`.
- **Fingerprint**: add NPU identity (vendor + model + count, driver-invariant)
  to `rustllama_tuner::system_fingerprint()` so a tuned cache is keyed by the
  NPU set — same treatment GPUs and `cpu_enabled` already get.
- **Config**: a `[inference.placement]` `npu_enabled` flag beside
  `cpu_enabled` / `gpu_enabled`; `disabled_npus` beside `disabled_gpus`.

### 3.3 N-tier heterogeneous scheduler (requirement 4)
Generalize placement from "GPU-or-CPU per layer" to an ordered tier list with
spillover:

- **Priority order — MEASURED by the autotuner, fastest→slowest.** The
  autotuner benches each tier's throughput (CPU + NPU + GPU/Metal, tok/s) and
  orders the tiers by measured speed (`tier::order_tiers_by_perf`), extending
  the existing `per_device_perf` stage to the NPU tier. So a weak iGPU that
  benches below the CPU drops under it, and an NPU slots wherever its measured
  speed lands — no static assumption. The order persists per `system_fingerprint`
  (now NPU-aware) and is applied as `[inference.placement].tier_order`. The
  static `GPU/Metal → NPU → CPU` (`tier::default_tier_order`) is only the
  cold-start fallback before the bench runs, and an explicit config `tier_order`
  overrides the measurement.
- **Spillover**: assign layers/ops down the priority list until each tier's
  **capacity** is exhausted, then spill the remainder to the next tier. Capacity
  = free VRAM (GPU), NPU memory (NPU), and "the rest" (CPU). This is the direct
  generalization of `auto_n_gpu_layers` (which already budgets GPU layers by
  VRAM then sends the rest to CPU) to ≥3 tiers, and of the MoE tier's
  promote-hottest-until-budget logic.
- **Decided by the autotuner**, persisted per `system_fingerprint()` (now
  NPU-aware), auto-applied on load — the established pattern.

## 4. Phasing (write-blind, incremental, feature-gated)

1. **Device model first** (buildable + testable here, no NPU): enumeration hook
   (returns empty without the feature/HW), `disabled_npus` config + env, unified
   index space, fingerprint inclusion. Unit-testable with a stub NPU inventory.
2. **N-tier scheduler** (buildable here): generalize placement to the ordered
   tier list + spillover, with the NPU tier present but empty on this box →
   behavior identical to today (GPU→CPU). Unit-testable with synthetic
   capacities.
3. **Intel NPU via OpenVINO** (feature `npu-openvino`, write-blind): FFI +
   compiled-op cache + the matmul offload. Validated on a Core Ultra by the
   owner.
4. **AMD NPU via Ryzen AI** (feature `npu-ryzenai`, write-blind): same shape.
5. **Attention offload + refinement** once matmul offload is validated.

Phases 1–2 are real, testable work that lands the device-management +
scheduler requirements **without** any NPU hardware or vendor runtime; 3–5 are
the write-blind vendor integrations behind their features.

## 5. Decisions (owner, 2026-10-13)

- **Approved**, with the remaining CPU extension (AVX-512-BF16 GEMM) landed
  **before** the NPU/scheduler phases.
- **First NPU target**: **both** Intel/OpenVINO and AMD/Ryzen AI scaffolded in
  parallel (Phases 3+4 together) behind their feature flags.
- **Default priority order**: **`GPU/Metal → NPU → CPU`** (throughput-first).
  Keep it a tunable `[inference.placement].tier_order`; a power-aware NPU-first
  mode can come later.
- **Op-offload v1 scope**: matmul-first (open; matmul + attention is the natural
  extension once matmul offload is validated).
