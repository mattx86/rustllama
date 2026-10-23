//! Autotuner — types, cache I/O, and the sweep coordinator entrypoint.
//!
//! Phase 0 ships only the public types and the on-disk cache schema so the
//! engine can already call `Tuner::load_cache` and the GUI can render an
//! "untuned kernel" state. The actual sweep logic lands in phase 5.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub mod placement;
pub mod sweep;
pub use placement::{ModelDims, WeightQuant};
pub use sweep::{sweep, CandidateResult, SweepConfig, SweepResult};

// v2: the cache file is keyed by the whole-SYSTEM fingerprint
// (`system_fingerprint()`), not a single SYCL device — so the tuner works on
// CUDA-only / CPU-only hosts and placement is per-(system, model). Old v1 files
// (keyed by SYCL device) are treated as a schema mismatch → a one-time re-tune.
pub const TUNING_SCHEMA_VERSION: u32 = 2;

// ============================================================
// Per-kernel tunable shape: Q4_K packed USM matvec (LWS sweep)
// ============================================================
//
// Stage A made the kernel runtime-selectable on local work-group
// size. Stage B (here) defines the cache shape + lookup helpers so
// the engine can read a pre-tuned LWS at dispatch time. The tuner
// CLI (Stage C) populates these entries by sweeping the candidate
// set.

/// Canonical kernel name strings used as keys in
/// `TuningResult.kernels`. Pinned here so the engine, the tuner CLI,
/// and the test suite agree on spelling without hand-coding it in
/// multiple places.
pub const KERNEL_Q4K_PACKED_USM: &str = "q4k_matvec_usm";
/// PrismML PTQ1_0 (Bonsai ternary) packed matvec — 28 B / 128-weight
/// blocks, K % 128 == 0.
pub const KERNEL_PTQ1_0_PACKED_USM: &str = "ptq1_0_matvec_usm";
pub const KERNEL_Q5K_PACKED_USM: &str = "q5k_matvec_usm";
pub const KERNEL_Q6K_PACKED_USM: &str = "q6k_matvec_usm";
pub const KERNEL_Q8_0_PACKED_USM: &str = "q8_0_matvec_usm";
/// F4: IQ4_NL packed USM matvec — 18 bytes/32-weight block, K%32==0.
pub const KERNEL_IQ4_NL_PACKED_USM: &str = "iq4_nl_matvec_usm";
/// F4: IQ4_XS packed USM matvec — 136 bytes/256-weight super-block, K%256==0.
pub const KERNEL_IQ4_XS_PACKED_USM: &str = "iq4_xs_matvec_usm";
/// F4 inference: IQ1_S packed USM matvec — 50 bytes/256-weight super-block, K%256==0.
pub const KERNEL_IQ1_S_PACKED_USM: &str = "iq1_s_matvec_usm";
/// F4 inference: IQ2_XXS packed USM matvec — 66 bytes/256-weight super-block, K%256==0.
pub const KERNEL_IQ2_XXS_PACKED_USM: &str = "iq2_xxs_matvec_usm";
/// F4 inference: IQ1_M packed USM matvec — 56 bytes/256-weight super-block, K%256==0.
pub const KERNEL_IQ1_M_PACKED_USM: &str = "iq1_m_matvec_usm";
/// F4 inference: IQ2_XS packed USM matvec — 74 bytes/256-weight super-block, K%256==0.
pub const KERNEL_IQ2_XS_PACKED_USM: &str = "iq2_xs_matvec_usm";
/// F4 inference: IQ2_S packed USM matvec — 82 bytes/256-weight super-block, K%256==0.
pub const KERNEL_IQ2_S_PACKED_USM: &str = "iq2_s_matvec_usm";
/// F4 inference: IQ3_XXS packed USM matvec — 98 bytes/256-weight super-block, K%256==0.
pub const KERNEL_IQ3_XXS_PACKED_USM: &str = "iq3_xxs_matvec_usm";
/// F4 inference: IQ3_S packed USM matvec — 110 bytes/256-weight super-block, K%256==0.
pub const KERNEL_IQ3_S_PACKED_USM: &str = "iq3_s_matvec_usm";

/// All packed single-row USM matvec kernels that share the same
/// LWS-sweep shape (compiled-in candidate set + `(M, K)` bucketing).
/// The tuner CLI iterates this to know which kernels to sweep; the
/// engine cache loader iterates this to know which entries to read.
/// Add a new kernel to extend coverage — no other Rust-side changes
/// needed besides templating the C++ side (mirror Q4_K) and adding
/// the FFI plumbing.
pub const PACKED_USM_KERNELS: &[&str] = &[
    KERNEL_Q4K_PACKED_USM,
    KERNEL_PTQ1_0_PACKED_USM,
    KERNEL_Q5K_PACKED_USM,
    KERNEL_Q6K_PACKED_USM,
    KERNEL_Q8_0_PACKED_USM,
    KERNEL_IQ4_NL_PACKED_USM,
    KERNEL_IQ4_XS_PACKED_USM,
    KERNEL_IQ1_S_PACKED_USM,
    KERNEL_IQ2_XXS_PACKED_USM,
    KERNEL_IQ1_M_PACKED_USM,
    KERNEL_IQ2_XS_PACKED_USM,
    KERNEL_IQ2_S_PACKED_USM,
    KERNEL_IQ3_XXS_PACKED_USM,
    KERNEL_IQ3_S_PACKED_USM,
];

/// Compiled-in candidate LWS values shared by every packed USM
/// matvec. Must match the `case` arms in each kernel's dispatcher in
/// `cpp/rsl_kernels.cpp`. The dispatcher falls back to the hand-
/// picked default (64) for any value outside this set.
pub const Q4K_USM_LWS_CANDIDATES: &[u32] = &[16, 32, 64, 128, 256];
/// Alias for the shared candidate set — every templated packed USM
/// matvec uses the same compiled-in LWS variants.
pub const PACKED_USM_LWS_CANDIDATES: &[u32] = Q4K_USM_LWS_CANDIDATES;

/// Canonical shape-bucket string for packed USM matvecs. Used by all
/// quants — the bucketing strategy is shared so the cache format
/// stays consistent across kernels.
///
/// Today we use exact (M, K) — most LLM architectures only emit a
/// handful of distinct shapes per forward pass (Q/K/V/O/gate/up/down),
/// so the cache stays small without lossy bucketing. Future tuning
/// of wider shape families (e.g. continuous-batched prefill with
/// variable N) may switch to log2 buckets here without changing the
/// cache format.
pub fn q4k_usm_shape_bucket(m: usize, k: usize) -> String {
    format!("M={m},K={k}")
}

/// Read the tuned LWS for `(kernel, M, K)` from a loaded
/// [`TuningResult`]. Returns `None` if no entry exists — engine then
/// passes `lws = 0` to the kernel, which selects the hand-picked
/// default.
///
/// Expected cache shape:
/// ```text
/// [kernels.q4k_matvec_usm]
/// kernel = "q4k_matvec_usm"
/// [kernels.q4k_matvec_usm.params.lws_by_shape]
/// "M=1536,K=1536" = 64
/// "M=8960,K=1536" = 128
/// ```
pub fn tuned_lws_for(result: &TuningResult, kernel: &str, m: usize, k: usize) -> Option<u32> {
    let entry = result.kernels.get(kernel)?;
    let by_shape = entry.params.get("lws_by_shape")?.as_object()?;
    let bucket = q4k_usm_shape_bucket(m, k);
    by_shape.get(&bucket)?.as_u64().map(|v| v as u32)
}

/// Insert or update the tuned LWS for `(kernel, M, K)`. Creates the
/// kernel entry on the fly if missing. Used by the tuner CLI to
/// record sweep winners.
pub fn set_tuned_lws_for(result: &mut TuningResult, kernel: &str, m: usize, k: usize, lws: u32) {
    let bucket = q4k_usm_shape_bucket(m, k);
    let entry = result
        .kernels
        .entry(kernel.to_string())
        .or_insert_with(|| KernelParams {
            kernel: kernel.to_string(),
            params: serde_json::json!({ "lws_by_shape": {} }),
        });
    let obj = entry
        .params
        .as_object_mut()
        .expect("KernelParams.params must be a JSON object");
    let by_shape = obj
        .entry("lws_by_shape")
        .or_insert_with(|| serde_json::json!({}));
    let by_shape_obj = by_shape
        .as_object_mut()
        .expect("lws_by_shape must be a JSON object");
    by_shape_obj.insert(bucket, serde_json::json!(lws));
}

/// Iterate every `(kernel, "M=<m>,K=<k>", lws)` entry in the cache
/// that has the `lws_by_shape` shape. Used by the engine's cache
/// loader to materialize a flat lookup map at load time. Returns an
/// owned vec so the caller can drop the `TuningResult` after.
pub fn iter_tuned_lws(result: &TuningResult) -> Vec<(String, String, u32)> {
    let mut out = Vec::new();
    for (kname, entry) in &result.kernels {
        if let Some(by_shape) = entry.params.get("lws_by_shape").and_then(|v| v.as_object()) {
            for (bucket, val) in by_shape {
                if let Some(lws) = val.as_u64() {
                    out.push((kname.clone(), bucket.clone(), lws as u32));
                }
            }
        }
    }
    out
}

// --- GPU-kernel validation verdicts -----------------------------------
//
// Parity-probe names for the specialized GPU kernels whose use is
// auto-enabled per device by the `tune --validate-kernels` stage (which
// runs these probes) and read at dispatch time. The names MUST match the
// probe names emitted by `rustllama doctor --cuda-parity` / the SYCL XMX
// probe so the verdict map keys line up. These replace the removed
// `RUSTLLAMA_FP4_TC` / `_FP8_WGMMA` / `_SYCL_XMX` env gates.

/// Blackwell FP4 tensor-core GEMM (NVFP4 weights, W4A4). Backs `fp4_tc`.
pub const VERDICT_GEMM_NVFP4_TC: &str = "gemm:nvfp4_tc";
/// Blackwell FP4 tensor-core GEMM (MXFP4 weights, W4A4). Backs `fp4_tc`.
pub const VERDICT_GEMM_MXFP4_TC: &str = "gemm:mxfp4_tc";
/// Blackwell MXFP8 (W8A8) block-scaled tensor-core GEMM. Probe-only
/// until wired into dispatch (Phase 3).
pub const VERDICT_GEMM_MXFP8_TC: &str = "gemm:mxfp8_tc";
/// Blackwell MXFP6 (W6A6) block-scaled tensor-core GEMM. Probe-only.
pub const VERDICT_GEMM_MXFP6_TC: &str = "gemm:mxfp6_tc";
/// Blackwell NVFP4 TMA-staged tensor-core GEMM variant. Probe-only.
pub const VERDICT_GEMM_NVFP4_TC_TMA: &str = "gemm:nvfp4_tc_tma";
/// Blackwell FP8 2:4 structured-sparse GEMM. Probe-only.
pub const VERDICT_GEMM_FP8_SP24: &str = "gemm:fp8_sp24";
/// Hopper sm_90a FP8 `wgmma` GEMM. Backs `hopper_tc` (no probe emits this
/// yet — stays off until a GH200 is available to validate).
pub const VERDICT_GEMM_FP8_WGMMA: &str = "gemm:fp8_wgmma";
/// Intel XMX/DPAS bf16 GEMM (SYCL). Backs `xmx`.
pub const VERDICT_XMX_GEMM: &str = "xmx:gemm";
/// Q4_K prefill GEMM (tiled, shared-mem weight reuse, f32 — BIT-EXACT). Backs
/// `q4k_gemm_f32`, wired onto the batched/prefill CUDA path. Graded by the
/// bit-exact batched parity probe; gated fail-closed (catches a write-blind
/// kernel bug on an unvalidated arch).
pub const VERDICT_GEMM_Q4K_F32: &str = "gemm:q4_k_f32";
/// Q8_0 prefill GEMM via shared-mem weight reuse, f32 accumulate (bit-exact,
/// LOSSLESS — the Q8_0 analog of [`VERDICT_GEMM_Q4K_F32`]). Backs `q8_0_gemm_f32`,
/// wired onto the batched/prefill CUDA path BELOW the lossy int8-TC Q8_0 GEMM, so
/// it is the lossless fast Q8_0 prefill path when the TC GEMM is perf-gated off.
/// Graded by the bit-exact batched parity probe; gated fail-closed.
pub const VERDICT_GEMM_Q8_0_F32: &str = "gemm:q8_0_f32";
/// Q4_K prefill GEMM via int8 tensor cores (W8A8, LOSSY). Backs `q4k_w8a8_tc`,
/// wired onto the batched/prefill CUDA path ABOVE the bit-exact f32 GEMM (it is
/// dispatched first). Graded by a FAIR W8A8 probe (int8 activations) + the perf gate.
pub const VERDICT_GEMM_Q4K_W8A8_TC: &str = "gemm:q4_k_w8a8_tc";

/// Q8_0 int8 tensor-core PREFILL GEMM (W8A8, lossy; T3c). Graded by a FAIR W8A8
/// probe (int8 activations) + the perf gate vs the f32 batched path, so it
/// auto-enables only where correct AND faster. The Q8_0 analog of
/// [`VERDICT_GEMM_Q4K_W8A8_TC`].
pub const VERDICT_GEMM_Q8_0_W8A8_TC: &str = "gemm:q8_0_w8a8_tc";
/// Q4_K W4A8 DECODE matvec (int8 activations, LOSSY). Backs `q4k_w4a8`, wired
/// onto the single-row CUDA decode path ABOVE the bit-exact f32 warp matvec (it
/// is dispatched first). Graded by a FAIR W8A8 probe (int8 activations) + the
/// perf gate, so it auto-enables only where correct AND faster than the f32
/// matvec. The decode twin of [`VERDICT_GEMM_Q4K_W8A8_TC`].
pub const VERDICT_MATVEC_Q4K_W4A8: &str = "matvec:q4_k_w4a8";
/// T3a — the Q8_0 W8A8 int-dot DECODE matvec (int8 activations, LOSSY; MMVQ-style,
/// symmetric so no min term). On where its probe passed: graded by a FAIR W8A8
/// reference (int8 activations) AND perf-gated vs the bit-exact f32 warp matvec,
/// so it auto-enables only where correct AND faster. Fail-closed. The Q8_0 analog
/// of [`VERDICT_MATVEC_Q4K_W4A8`].
pub const VERDICT_MATVEC_Q8_0_W8A8: &str = "matvec:q8_0_w8a8";
/// T3b int-dot fan-out: the Q4_0 / Q5_0 W8A8 DECODE matvecs (symmetric, MMVQ-style
/// int8 activations, lossy) — same probe + fail-closed story as the Q8_0 analog.
pub const VERDICT_MATVEC_Q4_0_W8A8: &str = "matvec:q4_0_w8a8";
pub const VERDICT_MATVEC_Q5_0_W8A8: &str = "matvec:q5_0_w8a8";
/// #2 on-device activation-residency DECODE attention block (whole attention
/// block GPU-resident on one stream: rmsnorm→qkv→qk-head-norm→rope→flash→o→
/// residual+norm, no per-op host round-trip). Unlike the other verdicts this is
/// a PERF verdict, not a parity one: the block is already bit-exact (coherence
/// validated), so the `tune --dev-attn` A/B stage enables it ONLY where a real
/// f32-KV decode measurement shows it beats the host path on THIS device+model
/// (fail-closed off otherwise — it's a no-win or slight loss on MoE/CPU-tail
/// configs where attention isn't the bottleneck, a win only on a dense model or
/// a fully-fitting GPU). Read by `accel::dev_attn_enabled`; f32-KV-only at dispatch.
pub const VERDICT_ATTN_DEV_RESIDENT_BLOCK: &str = "attn:dev_resident_block";
/// GPU paged attention (Phase C): the CUDA block-table flash-decode kernel
/// (`flash_attn_decode_paged_f32`) graded vs the CPU gather + flash-decode
/// reference over pass / SWA / soft-cap. Read by `accel::dev_attn_paged_enabled`;
/// f32-KV-only at dispatch (the paged KV store is F32). Fail-closed: absent /
/// `false` ⇒ the paged engine keeps the CPU gather attention path.
pub const VERDICT_ATTN_PAGED_DECODE_CUDA: &str = "attn:paged_decode_cuda";
/// GPU paged attention (Phase E — SYCL twin): the SYCL block-table flash-decode
/// kernel (`flash_attn_decode_paged_usm`) graded vs the SAME CPU gather +
/// flash-decode reference as the CUDA twin, over pass / SWA / soft-cap. Read by
/// `accel::dev_attn_paged_sycl_enabled`; f32-KV-only at dispatch (the paged KV
/// store + its USM mirror are F32, so the kernel is BIT-EXACT — no f16
/// round-trip). Fail-closed: absent / `false` ⇒ the paged engine keeps the CPU
/// gather attention path.
pub const VERDICT_ATTN_PAGED_DECODE_SYCL: &str = "attn:paged_decode_sycl";
/// GPU paged attention (Phase F — MLX/Metal twin): the Metal block-table
/// flash-decode kernel (`rsl_mlx_flash_attn_decode_paged_f32_kernel`) graded vs
/// the SAME CPU gather + flash-decode reference as the CUDA/SYCL twins, over
/// pass / SWA / soft-cap. Read by `accel::dev_attn_paged_mlx_enabled`;
/// f32-KV-only at dispatch (the paged KV store + its Metal mirror are F32, so
/// the kernel is BIT-EXACT). Fail-closed: absent / `false` ⇒ the paged engine
/// keeps the CPU gather attention path. WRITE-BLIND (validated on the user's
/// Mac via `doctor --metal-parity`).
pub const VERDICT_ATTN_PAGED_DECODE_MLX: &str = "attn:paged_decode_mlx";
/// SYCL (USM) twin of [`VERDICT_ATTN_DEV_RESIDENT_BLOCK`] — the #2 on-device
/// activation-residency DECODE attention block run on the SYCL in-order queue
/// (Feature 1a). Separate verdict because the win/loss profile is per-backend:
/// the SYCL block collapses a per-token FLOOD of independent `q.submit().wait()`
/// dispatches into ONE submit-chain that syncs once per layer, so the measured
/// saving is the SYCL dispatch/sync overhead (not a PCIe transfer like CUDA).
/// Same PERF-only semantics: the block is already correct (parity-validated via
/// `doctor --sycl-parity`), so `tune --dev-attn` enables it only where it beats
/// the per-op SYCL path on THIS device+model (fail-closed off otherwise). Read by
/// `accel::dev_attn_sycl_enabled`; f32-KV-only + no-Qwen3-head-norm at dispatch.
pub const VERDICT_ATTN_DEV_RESIDENT_BLOCK_SYCL: &str = "attn:dev_resident_block_sycl";
/// CUDA-graph twin of [`VERDICT_ATTN_DEV_RESIDENT_BLOCK`] (Feature 1b) — the #2
/// device-resident DECODE attention block captured into a `cudaGraph` and
/// replayed as ONE launch, instead of the eager block's per-op kernel
/// launch+sync. Separate verdict because it is a refinement OF the CUDA eager
/// block measured against it: `tune --dev-attn` A/Bs graph-vs-eager on CUDA and
/// enables the graph path only where it beats the eager device block on THIS
/// device+model (fail-closed off otherwise — the graph's per-launch saving is a
/// no-win when attention isn't the decode bottleneck, and capture carries its
/// own overhead). The captured chain runs the IDENTICAL kernels/math as the
/// eager block, so correctness is the eager block's (graded in-process by
/// `doctor --cuda-parity attn:dev_resident_block_cuda_graph`); this is a
/// PERF-only verdict. Read by `accel::dev_attn_cuda_graph_enabled`; at dispatch
/// the graph path is preferred over the eager CUDA block when ON (f32-KV-only).
pub const VERDICT_ATTN_DEV_RESIDENT_BLOCK_CUDA_GRAPH: &str = "attn:dev_resident_block_cuda_graph";
/// P7: CORRECTNESS anchor for the WHOLE-LAYER CUDA-graph capture — the decode
/// attention block AND the dense SwiGLU FFN (gate/up→silu→down→residual) captured
/// into ONE `cudaGraph` and replayed as a single launch per layer (the
/// `dense_ffn = Some(..)` arm of `CudaMatvecCache::decode_attn_block_f32_graph`,
/// dispatched from `accel::try_decode_attn_block_cuda_graph_f32`). The attention-
/// only graph is covered by [`VERDICT_ATTN_DEV_RESIDENT_BLOCK_CUDA_GRAPH`]; this
/// name is the `doctor --cuda-parity layer:dev_resident_cuda_graph` probe that
/// additionally grades the FFN-in-graph end to end vs a CPU attention+SwiGLU
/// reference. The PERF gate stays the eager-vs-graph `attn:...cuda_graph` verdict
/// (the FFN rides the same captured chain / dev-attn A/B), so this is a correctness
/// anchor, not a separate dispatch gate — the graph path is already fail-soft to
/// the eager block on any capture/launch error.
pub const VERDICT_LAYER_DEV_RESIDENT_CUDA_GRAPH: &str = "layer:dev_resident_cuda_graph";
/// T2 (B1): WHOLE-DECODE-STEP CUDA graph — EVERY dense layer's attention+FFN chain
/// captured into ONE `cudaGraph` and replayed with ONE launch + ONE `stream_sync`
/// per token (vs the per-layer graph's one of each PER layer), removing
/// `(n_layers-1)` syncs + per-layer host round-trips. The captured kernels are the
/// IDENTICAL `issue_layer_chain` the per-layer graph uses, so it is byte-identical
/// to (and correctness-anchored by) [`VERDICT_ATTN_DEV_RESIDENT_BLOCK_CUDA_GRAPH`]
/// / [`VERDICT_LAYER_DEV_RESIDENT_CUDA_GRAPH`]; this is a PERF-only verdict graded
/// by `doctor --cuda-parity step:dev_resident_cuda_graph`. Read by
/// `accel::dev_step_cuda_graph_enabled`; gated AND with the two per-layer graph
/// verdicts (pure-dense, F32-KV, no-bias/ring/YaRN/streaming), fail-closed OFF.
pub const VERDICT_STEP_DEV_RESIDENT_CUDA_GRAPH: &str = "step:dev_resident_cuda_graph";
/// Apple-Metal/MLX twin of [`VERDICT_ATTN_DEV_RESIDENT_BLOCK`] (Feature 1c) —
/// the #2 on-device activation-residency DECODE attention block encoded into ONE
/// Metal command buffer with ONE compute encoder (rmsnorm→qkv→rope→KV-mirror
/// write→flash→o→residual+norm), committed with a SINGLE `waitUntilCompleted`
/// instead of the per-op commit+wait flood. Separate verdict because the win/loss
/// profile is per-backend: the Metal block collapses the per-op command-buffer
/// submit+sync overhead on the unified-memory Apple GPU (no PCIe transfer like
/// CUDA, no in-order-queue submit like SYCL). Same PERF-only semantics — the
/// block is bit-exact f32 end to end (the KV mirror is F32 on Metal, so there is
/// no f16 round-trip at all), correctness validated by `doctor --metal-parity`
/// (`attn:dev_resident_block_mlx`), so `tune --dev-attn` enables it only where it
/// beats the per-op MLX path on THIS device+model (fail-closed off otherwise).
/// Read by `accel::dev_attn_mlx_enabled`; f32-KV-only + no-Qwen3-head-norm at
/// dispatch (there is no MLX qk-head-norm kernel).
pub const VERDICT_ATTN_DEV_RESIDENT_BLOCK_MLX: &str = "attn:dev_resident_block_mlx";

// --- P5b: verdict keys that replaced default-off env gates -----------
//
// Each key below is the auto-enable verdict for a GPU path that used to be
// gated by a `RUSTLLAMA_*` env var (default off). The dispatch reads the
// verdict (fail-closed: absent / `false` ⇒ OFF), and `tune --validate-kernels`
// banks it on-device — the same discipline as the tensor-core keys above.

/// Per-dtype fused gate+up matvec (one dispatch computing both FFN projections).
/// The actual keys are `fused:<dtype>` (e.g. `fused:q4_k`), banked by the SYCL
/// `probe_fused` parity probes; this is the shared PREFIX. The BARE-KERNEL
/// parity check — a single ~512×2048 GEMM graded vs the CPU matvec. Necessary
/// but NOT sufficient to turn the path on: `accel::moe_gate_up_fused_enabled`
/// ANDs this with [`VERDICT_MOE_GATE_UP_COHERENT_PREFIX`] (the real per-expert
/// MoE forward), because this generic shape does not reproduce the 2026-09-06
/// Iris-Xe/OpenCL token-soup incident the fused path was disabled for.
pub const VERDICT_FUSED_GATE_UP_PREFIX: &str = "fused:";
/// Per-dtype REAL-MoE-FORWARD coherence verdict for the fused gate+up path.
/// Keys are `moe:gate_up_coherent:<dtype>`, banked by the SYCL/CUDA/MLX
/// `probe_moe_gate_up_coherent` probes. Unlike [`VERDICT_FUSED_GATE_UP_PREFIX`]
/// (a bare ~512×2048 single GEMM graded vs the CPU matvec), these run the
/// per-expert FFN the model actually executes — several experts, realistic
/// NON-512 `[d_ff, d_model]` expert shapes, the SwiGLU `silu(gate)·up`
/// nonlinearity, routing-weighted summed — and grade the FUSED path against
/// BOTH the separate-matvec GPU path (does fusing change the answer?) AND the
/// CPU reference (is the fused math right at all?). This is the faithful
/// reproduction of the path that produced garbage logits on 2026-09-06 (Iris Xe,
/// Qwen3.6-35B IQ2_XXS experts); the later-found root cause (a since-removed
/// `(ptr,len)` activation dedup) is NOT reproduced by the bare GEMM probe, so
/// the dispatch requires THIS verdict too. In addition the SYCL dispatch
/// hard-excludes the OpenCL-fallback backend — the exact incident backend —
/// regardless of any verdict (fail-closed by construction on the incident
/// conditions). Fail-closed: a missing key ⇒ OFF.
pub const VERDICT_MOE_GATE_UP_COHERENT_PREFIX: &str = "moe:gate_up_coherent:";
/// Per-dtype F16-activation (mixed-precision) packed matvec. Keys are
/// `matvec_f16in:<dtype>`, banked by the SYCL `probe_matvec_f16in` probes —
/// graded vs an f16-activation-roundtrip reference (so the probe grades kernel
/// correctness, not the inherent f16 loss) AND perf-gated (`SLOW` ⇒ fail). Backs
/// `accel::mixed_precision_matvec_enabled`. Lossy, so it auto-enables ONLY where
/// correct AND faster than the f32-activation matvec.
pub const VERDICT_MATVEC_F16IN_PREFIX: &str = "matvec_f16in:";
/// GPU-resident sampler pipeline (penalty → temp+softmax → top-k → draw). A
/// correctness-AND-perf verdict banked by `collect_sampler_gpu_verdict`: ON only
/// where the GPU chain matches the CPU sampler AND is faster. Backs
/// `rustllama_engine::sampling` (replaces `RUSTLLAMA_SAMPLER_GPU`).
pub const VERDICT_SAMPLER_GPU: &str = "sampler:gpu";
/// USM-resident KV cache. A PERF verdict banked by `collect_kv_usm_verdict`
/// (USM write/read bandwidth vs a host buffer): ON only where USM is faster.
/// Backs `llama_arch::try_open_usm_kv_stream` (replaces `RUSTLLAMA_USM_KV`). On a
/// unified-memory iGPU (USM IS host memory) this is a wash → fail-closed OFF.
pub const VERDICT_KV_USM: &str = "kv:usm";
/// GPU dequant of K-quant sources during quantize (quantize-time, not inference).
/// A parity verdict read by the `quantize` command, which passes the decision
/// into the (toolchain-free) gguf pipeline. Backs `gguf::quantize` (replaces
/// `RUSTLLAMA_GPU_DEQUANT`). No on-device banking probe yet (P6 follow-up: a
/// dequant parity probe through `SyclIqEncoder`) → fail-closed OFF.
pub const VERDICT_DEQUANT_GPU: &str = "dequant:gpu";

/// P17: GPU DeltaNet chunked-prefill recurrence (the hybrid-model tok/s ceiling).
/// A parity-AND-perf verdict: `tune --validate-kernels` grades the SYCL (and,
/// write-blind, the CUDA/MLX) `delta_net_chunk` kernel against the CPU
/// `delta_net.rs` chunked reference on identical inputs (tolerance ≈1e-3, the
/// recurrence reduction-order budget) AND perf-gates it (the chunked GPU scan is
/// an alternative to the SIMD/rayon CPU path, so it auto-enables only where it
/// matches AND is faster). Backs `accel::delta_net_chunk_sycl_enabled` (+ the
/// CUDA/MLX twins); fail-closed: absent / `false` ⇒ the hybrid DeltaNet forward
/// stays byte-identical on the CPU path. No env gate.
pub const VERDICT_SSM_DELTA_NET: &str = "ssm:delta_net";
/// C1 — resident-state Gated-DeltaNet single-token DECODE step on the GPU (the
/// per-token twin of the chunk-prefill `ssm:delta_net`). The recurrence is
/// memory-bound on the per-head state S; the GPU step keeps S RESIDENT in VRAM
/// across tokens and runs the fused 2-pass recurrence at device bandwidth. Backs
/// `accel::delta_net_step_enabled`; graded vs the CPU scalar reference by
/// `doctor --cuda-parity ssm:delta_net_step`. Fail-closed: absent / `false` ⇒ the
/// DeltaNet decode stays on the SIMD/rayon CPU path. Pre-verdict A/B override:
/// `RUSTLLAMA_DELTA_NET_STEP=1`.
pub const VERDICT_SSM_DELTA_NET_STEP: &str = "ssm:delta_net_step";
/// PP-0 (multi-GPU) — the cross-GPU peer-copy transport is byte-correct. Graded by
/// `doctor --cuda-parity peer:copy` (a round-trip through `rsl_cuda_memcpy_peer`;
/// on a single-GPU box src==dst validates the API + byte fidelity, the only part
/// verifiable without a 2nd GPU — real P2P transport is confirmed on multi-GPU HW).
/// Foundation for pipeline-stage activation handoff + the TP collectives.
pub const VERDICT_PEER_COPY: &str = "peer:copy";
/// SYCL multi-GPU — the cross-device USM peer-copy transport is byte-correct. The
/// SYCL twin of [`VERDICT_PEER_COPY`], graded by `doctor --sycl-parity peer:copy_sycl`
/// (a round-trip through `rsl_sycl_memcpy_peer`; on a single SYCL GPU src==dst
/// validates the API + byte fidelity, the only part verifiable without a 2nd SYCL
/// GPU — real cross-device USM transfer confirmed on multi-Arc/PVC HW). Foundation
/// for the SYCL `GpuTransport` (per-device SYCL matvec routing + collectives).
pub const VERDICT_PEER_COPY_SYCL: &str = "peer:copy_sycl";
/// TP-1 (multi-GPU) — the cross-GPU all-reduce (SUM) collective recombines
/// correctly. Graded by `doctor --cuda-parity collective:all_reduce`: K partial
/// buffers are summed via a star reduce (peer-copy + on-device `add_f32`) +
/// broadcast, and every participant's result is checked against the CPU sum. On a
/// single-GPU box the K buffers live on one device, so this validates the
/// reduction ALGEBRA + the peer/add plumbing (self-copies) — the real P2P sum is
/// confirmed on multi-GPU HW. Foundation for TP-2 row-parallel matmul (partials →
/// all-reduce) + TP-3 attention. Hand-rolled (no NCCL).
pub const VERDICT_COLLECTIVE_ALL_REDUCE: &str = "collective:all_reduce";
/// TP-1 (multi-GPU) — the cross-GPU all-gather collective recombines correctly.
/// Graded by `doctor --cuda-parity collective:all_gather`: K chunks are
/// concatenated (chunk i at offset i*n) into every participant's slab via
/// peer-copy and checked bit-exact against the host concatenation. Single-GPU =
/// the concat-placement algebra + plumbing; real P2P gather on multi-GPU HW.
/// Foundation for TP-3 head-parallel attention (per-head outputs → all-gather).
pub const VERDICT_COLLECTIVE_ALL_GATHER: &str = "collective:all_gather";
/// TP-2 (multi-GPU) — the ROW-parallel (K-split) matvec recombines correctly.
/// Graded by `doctor --cuda-parity matvec:tp_row`: a packed weight's contraction
/// dim K is split into block-aligned slices, each device computes a partial over
/// its slice, and the summed partials are checked against the WHOLE single-device
/// matvec. On a single-GPU box the two logical slices run on device 0, so this
/// validates the strided-extraction + reduce ALGEBRA + plumbing; the real
/// cross-device split is confirmed on 2+ GPU HW. The complement of the column
/// split (output-row partition); the standard TP choice for O- / down-proj.
pub const VERDICT_MATVEC_TP_ROW: &str = "matvec:tp_row";
/// TP-3 (multi-GPU) — head-parallel attention recombines correctly. Graded by
/// `doctor --cuda-parity attn:head_parallel`: the Q-heads (and GQA-matched
/// KV-heads) are split into groups, flash-prefill runs per group, and the
/// concatenated per-head outputs are checked against the WHOLE single-call flash
/// attention. Each head attends only within itself, so the split + concat (an
/// all-gather recombine, no reduce) is bit-exact; on a single-GPU box the groups
/// run on device 0, validating the head-split + concat ALGEBRA + plumbing — the
/// real cross-device head shard is confirmed on 2+ GPU HW.
pub const VERDICT_ATTN_HEAD_PARALLEL: &str = "attn:head_parallel";

// --- P6-fusions: the three forward-path fusion verdicts ----------------
//
// Each is a PARITY verdict: `tune --validate-kernels` grades the fused SYCL
// path against the two-op CPU reference (bit-tolerant f32, ~1e-4) and banks a
// pass/fail. The dispatch gate reads it (fail-closed: absent / `false` ⇒ OFF),
// so the forward uses the existing separate ops BYTE-IDENTICAL until a device
// proves the fusion correct. No env gate — same discipline as the tensor-core
// keys above. The SYCL `collect_sycl_verdicts` probe banks the verdict on
// Level-Zero/OpenCL hosts — the only path that is BOTH validatable here AND
// consumed by the (SYCL-USM-only) forward fusion dispatch. The CUDA/MLX kernel
// mirrors (`rsl_cuda_{rmsnorm_matvec,down_residual}_f32` in `rsl_cuda.cu`,
// `rsl_mlx_*` in `rsl_mlx.mm`/`.metal`) are ported WRITE-BLIND as device-pointer
// twins of the existing `add_rmsnorm_f32` / `silu_mul_f32` fusion family — ready
// for a device-resident CUDA/MLX forward to adopt (and for a future
// `--cuda-parity` / `--metal-parity` probe once that forward lands); they do not
// bank these verdicts today.

/// P6: RMSNorm-of-the-input fused into the immediately-following matvec
/// (`y = matvec(W, rmsnorm(x)·w_norm)`) — the attention-input / FFN-input norm
/// folded into its matvec in one SYCL USM dispatch, no host round-trip of the
/// normalized activation. Backs `accel::rmsnorm_matvec_fused_enabled`; graded by
/// the `fused:rmsnorm_matvec` probe vs CPU `rmsnorm_f32_row` then `matvec_f32`.
pub const VERDICT_FUSED_RMSNORM_MATVEC: &str = "fused:rmsnorm_matvec";
/// P6: FFN down-projection fused with the residual add
/// (`out = residual + matvec(W_down, ff)`) in one SYCL USM dispatch. Backs
/// `accel::down_residual_fused_enabled`; graded by the `fused:down_residual`
/// probe vs CPU `matvec_f32` then `add_inplace_f32`.
pub const VERDICT_FUSED_DOWN_RESIDUAL: &str = "fused:down_residual";
/// P6: attention output-projection + residual-add + post-attn RMSNorm fused
/// (`y_norm = rmsnorm(hidden + matvec(W_o, attn))·w_norm`). Re-introduces the
/// gate P5b deleted (`RUSTLLAMA_FUSED_OUT_PROJ_NORM`) as a verdict. Backs
/// `accel::fused_out_proj_norm_enabled` (and the already-wired
/// `try_matvec_out_proj_add_rmsnorm_usm_f32` chained dispatcher); graded by the
/// `fused:out_proj_norm` probe — the exact chained kernels (packed matvec →
/// `add_rmsnorm_f32`) vs CPU `matvec` then `add` then `rmsnorm_f32_row`.
pub const VERDICT_FUSED_OUT_PROJ_NORM: &str = "fused:out_proj_norm";
/// B2 (Kindi rmsfold): the Prism block-Hadamard activation rotation run ON the
/// GPU and kept device-resident with the preceding RMSNorm, so a folded Bonsai
/// projection does `rmsnorm → hadamard → matvec` entirely in USM — no CPU
/// rotation, no host round-trip of the rotated activation. Backs
/// `accel::rmsfold_fused_enabled`; graded by the `fused:rmsfold` probe (device
/// `rmsnorm_f32_usm` → `hadamard_forward_usm` vs CPU `rmsnorm_f32_row` then
/// `hadamard::hadamard_forward`).
pub const VERDICT_FUSED_RMSFOLD: &str = "fused:rmsfold";

/// B2 (Kindi smfold): the dense-FFN down-projection's Hadamard rotation run ON
/// the GPU and kept device-resident with the preceding SiLU-gate, so a folded
/// Bonsai FFN does `rmsnorm → hadamard(gate/up) → gate/up matvec → silu_mul →
/// hadamard(down) → down matvec` entirely in USM. The FFN device path needs BOTH
/// this and [`VERDICT_FUSED_RMSFOLD`] (the gate/up-input rotation). Backs
/// `accel::smfold_fused_enabled`; graded by the `fused:smfold` probe (device
/// `silu_mul_usm` → `hadamard_forward_usm` vs CPU `silu_mul_f32` then
/// `hadamard::hadamard_forward`).
pub const VERDICT_FUSED_SMFOLD: &str = "fused:smfold";

/// Read the on-device verdict for a kernel probe name, if one has been
/// recorded. `None` = never validated (or the probe SKIPped because the
/// device lacks the capability) — callers treat that as "off".
pub fn verdict_for(result: &TuningResult, name: &str) -> Option<bool> {
    result.kernel_verdicts.get(name).copied()
}

/// Record (insert or overwrite) a kernel's on-device verdict. Used by the
/// `tune --validate-kernels` stage.
pub fn set_verdict(result: &mut TuningResult, name: &str, pass: bool) {
    result.kernel_verdicts.insert(name.to_string(), pass);
}

/// T4 — build identity that changes whenever the crate VERSION *or any GPU
/// kernel source* changes. Format: `"{version}+{sycl}+{cuda}+{mlx}"`, where each
/// `{backend}` is the FNV-1a hash that the backend's `build.rs` computed over its
/// kernel sources and exposed as `rustllama_kernels_*::KERNEL_HASH`.
///
/// Stamped into [`TuningResult::kernel_build_id`] by `tune --validate-kernels`
/// and compared (not just `rustllama_version`) at the first-load verdict
/// self-heal sites, so an intra-version kernel fix re-validates on next load
/// without a manual `tune --validate-kernels` — the old check keyed on the crate
/// version alone, which never changes within a release. An old cache (field
/// absent → serde default `""`) never equals a real build id, so it is treated
/// as stale and re-validated (safe).
pub fn kernel_build_id() -> String {
    format!(
        "{}+{}+{}+{}",
        env!("CARGO_PKG_VERSION"),
        rustllama_kernels_sycl::KERNEL_HASH,
        rustllama_kernels_cuda::KERNEL_HASH,
        rustllama_kernels_mlx::KERNEL_HASH,
    )
}

/// Read this host's banked kernel verdict for `name` straight from the on-disk
/// tuner cache (keyed by [`system_fingerprint`]), fail-closed: a missing cache
/// dir / cache file / key ⇒ `false`. The verdict reader for callers that CANNOT
/// link `rustllama-models` and so can't reach `accel::kernel_verdict` — notably
/// `rustllama-cli`, which decides whether to pass `gpu_dequant` into the
/// toolchain-free gguf quantize pipeline (`dequant:gpu` verdict). Uncached (each
/// call re-reads), which is fine for the one-shot quantize command.
pub fn cached_kernel_verdict(name: &str) -> bool {
    let Some(dir) = default_cache_dir() else {
        return false;
    };
    let key = system_fingerprint();
    match load_cache(&dir, &key) {
        Ok(Some(t)) => t.kernel_verdicts.get(name).copied().unwrap_or(false),
        _ => false,
    }
}

/// Read the autotuned CacheBlend recompute ratio for a model (keyed by file
/// stem), if one has been fit. `Some(0.0)` means the stage ran and disabled
/// CacheBlend (incoherent at every ratio); `None` means untuned.
pub fn cacheblend_ratio_for(result: &TuningResult, model_stem: &str) -> Option<f32> {
    result.cacheblend_recompute_ratio.get(model_stem).copied()
}

/// Record a model's autotuned CacheBlend recompute ratio. Used by the
/// `tune --cacheblend` stage.
pub fn set_cacheblend_ratio(result: &mut TuningResult, model_stem: &str, ratio: f32) {
    result
        .cacheblend_recompute_ratio
        .insert(model_stem.to_string(), ratio);
}

/// Read the autotuned best spec-decoding strategy for a model (keyed by file
/// stem), if the `tune --spec-strategy` stage has run. `None` = untuned (the
/// serve path then falls back to the legacy global `speculative_mtp` /
/// `winner_tree_widths` winners).
pub fn spec_strategy_for(result: &TuningResult, model_stem: &str) -> Option<SpecStrategy> {
    result.spec_strategy.get(model_stem).copied()
}

/// Record a model's autotuned best spec strategy. Used by `tune --spec-strategy`.
pub fn set_spec_strategy(result: &mut TuningResult, model_stem: &str, s: SpecStrategy) {
    result.spec_strategy.insert(model_stem.to_string(), s);
}

// Compatibility shim: keep the original Q4_K-specific names that
// existing tests + engine code reference. Both delegate to the new
// kernel-agnostic helpers above.

/// Q4_K-specific shim over [`tuned_lws_for`].
pub fn tuned_q4k_usm_lws(result: &TuningResult, m: usize, k: usize) -> Option<u32> {
    tuned_lws_for(result, KERNEL_Q4K_PACKED_USM, m, k)
}

/// Q4_K-specific shim over [`set_tuned_lws_for`].
pub fn set_tuned_q4k_usm_lws(result: &mut TuningResult, m: usize, k: usize, lws: u32) {
    set_tuned_lws_for(result, KERNEL_Q4K_PACKED_USM, m, k, lws)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct DeviceFingerprint {
    pub pci_id: u32,
    pub driver_ver: String,
    pub name: String,
    pub vram_mb: u64,
}

impl DeviceFingerprint {
    /// Stable, filesystem-safe string used as the cache filename.
    pub fn slug(&self) -> String {
        let safe_name = self.name.replace(|c: char| !c.is_ascii_alphanumeric(), "_");
        let safe_drv = self
            .driver_ver
            .replace(|c: char| !c.is_ascii_alphanumeric() && c != '.', "_");
        format!("{:08x}-{}-{}", self.pci_id, safe_name, safe_drv)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ProblemClass {
    pub kernel: String,
    /// Bucketed problem shape (e.g. "M=4096,K=4096,N<=32"). The bucketing
    /// strategy lives next to each kernel's tunable definition.
    pub shape_bucket: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KernelParams {
    pub kernel: String,
    pub params: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ModelId {
    pub arch: String,
    pub n_params: u64,
    pub quantization: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlacementPlan {
    pub n_gpu_layers: u32,
    pub overrides: Vec<TensorOverride>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TensorOverride {
    pub pattern: String,
    pub device: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadPlan {
    pub prefill_threads: u32,
    pub gen_threads: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningResult {
    pub schema_version: u32,
    /// The whole-system fingerprint this cache file is keyed by (see
    /// [`system_fingerprint`]). This is the CACHE KEY — placement, threads,
    /// batch, kv_dtype etc. all live under it, so a topology change refreshes
    /// them. Works on SYCL / CUDA / CPU-only hosts alike.
    #[serde(default)]
    pub system_fp: String,
    /// The SYCL device the per-device *kernel-LWS* tuning was measured on
    /// (default/empty on a non-SYCL host — CUDA/CPU-only — where there is no
    /// SYCL kernel to tune). Informational + keys the `kernels` map's device.
    pub device: DeviceFingerprint,
    pub kernels: HashMap<String, KernelParams>,
    pub placement: HashMap<String, PlacementPlan>,
    /// Per-model winning MoE tiered-expert VRAM **margin** in BYTES (the headroom
    /// reserved below free VRAM when projecting the expert-promotion budget),
    /// keyed by model file stem — from `tune --moe-vram-margin` / `--all`. A
    /// smaller margin promotes more experts to GPU (faster decode) until paging
    /// thrash; the sweep measures decode tok/s per margin and keeps the best.
    /// `load_inner` reads this (else the static `kv_mirror.max(1 GiB)` heuristic).
    /// `#[serde(default)]` → old cache files still load, no schema-version bump.
    #[serde(default)]
    pub moe_vram_margin: HashMap<String, u64>,
    pub threads: Option<ThreadPlan>,
    pub batch_size: Option<u32>,
    /// Winning `[inference].kv_dtype` from the comprehensive sweep
    /// (`rustllama tune --kv-dtype` / `--all`). `None` = no entry,
    /// engine falls back to the config value. Persisted as the raw
    /// config string (e.g. `"q8_0"`, `"tq4"`, `"nvfp4"`) so cache
    /// entries survive a `KvDtype` enum bump.
    #[serde(default)]
    pub kv_dtype: Option<String>,
    /// F4: optional PER-LAYER-GROUP KV dtype spec — one raw config
    /// string per layer (e.g. `["f32", "f32", "q4_0", "q4_0", …]` for an
    /// attention-sink-lossless / rest-cheaper split). `None` (the serde
    /// default, so old caches load unchanged) means "all layers use
    /// `kv_dtype`" — the uniform case. The current `--kv-dtype` sweep
    /// still picks a single global `kv_dtype`; this slot carries a
    /// per-group spec once a per-group search lands, and the engine load
    /// builds the cache via `KvCache::new_with_layer_dtypes` when it is
    /// `Some`.
    #[serde(default)]
    pub kv_dtype_groups: Option<Vec<String>>,
    /// Winning `[inference].flash_attention` setting. `None` = no
    /// entry, engine uses the config value.
    #[serde(default)]
    pub flash_attention: Option<bool>,
    /// Winning `[inference].kv_cache_layout` value (`"contiguous"`
    /// or `"paged"`). `None` = no entry.
    #[serde(default)]
    pub kv_cache_layout: Option<String>,
    /// P9: single-stream serving-engine winner from `tune --batching` —
    /// `"paged"` (the continuous-batching `PagedBatchEngine` is
    /// parity-or-faster for the single-user case, so default-on is safe)
    /// or `"contiguous"` (the contiguous `ContiguousEngine` is measurably faster
    /// single-stream, so the serve path falls back to it). `None` =
    /// untuned; the serve path then uses the default-on batching engine.
    /// Consumed when `[server].fused_decode` is unset +
    /// `[tuning].auto_apply_batching` is on (an explicit `fused_decode`
    /// always wins). `#[serde(default)]` → old caches still load.
    #[serde(default)]
    pub batching_engine: Option<String>,
    /// Winning `[inference].flash_attention_kv_min` threshold from
    /// `rustllama tune --flash-kv-min`. Surfaces to the kernel
    /// dispatch via the `RUSTLLAMA_FLASH_KV_LEN_MIN` env var the
    /// engine-load path sets. `None` = no entry; engine uses the
    /// config value (default 256).
    #[serde(default)]
    pub flash_kv_min: Option<u32>,
    /// Winning `[inference].prefix_cache_max_snapshots` pool depth
    /// from `rustllama tune --prefix-snapshots`. `None` = no entry;
    /// engine uses the config value.
    #[serde(default)]
    pub prefix_cache_max_snapshots: Option<u32>,
    /// Winning `[inference].kv_page_size` from
    /// `rustllama tune --kv-page-size`. Only applies to the paged
    /// KV layout — contiguous-layout configs ignore this slot.
    /// `None` = no entry; engine uses the config value.
    #[serde(default)]
    pub kv_page_size: Option<u32>,
    /// Winning `KV_TILE` for the SYCL flash-attn-v3 decode kernel
    /// from `rustllama tune --flash-v3-kv-tile`. Surfaces to the
    /// kernel via the `RUSTLLAMA_FLASH_V3_KV_TILE` env var the
    /// engine-load path sets. Supported values: 16, 32, 64.
    /// `None` = no entry; kernel uses the built-in default (32).
    #[serde(default)]
    pub flash_v3_kv_tile: Option<u32>,
    /// P19: ADAPTIVE per-`kv_len`-bucket `KV_TILE` winners for the SYCL flash-
    /// attn-v3 decode kernel — `[short, mid, long]` (buckets `kv_len < 512`,
    /// `512..4096`, `>= 4096`), each 16/32/64. From `tune --flash-v3-kv-tile`,
    /// which now sweeps the candidate set at a SHORT and a LONG context and keeps
    /// the fastest tile per bucket (the single global `flash_v3_kv_tile` above is
    /// still written for back-compat / the mid bucket). Surfaced to the kernel
    /// via the explicit `tile_size` param (`accel::set_flash_v3_kv_tiles` +
    /// `fa_v3_tile_for`), NOT the env var. `None` (serde default → old caches
    /// load) ⇒ the kernel keeps its built-in env+kv_len heuristic, byte-identical.
    #[serde(default)]
    pub flash_v3_kv_tile_by_bucket: Option<Vec<u32>>,
    /// Winning MoE placement mode from `rustllama tune
    /// --moe-placement` — `"uniform"` (experts follow the layer's
    /// device) or `"experts_cpu"` (routed experts + router pinned to
    /// CPU, attention/shared expert stay GPU-eligible). Consumed when
    /// `[inference].moe_placement = "auto"` +
    /// `[tuning].auto_apply_moe_placement`. `None` = no entry.
    #[serde(default)]
    pub moe_placement: Option<String>,
    /// Measured multi-GPU mode winner from `rustllama tune --multi-gpu` — the
    /// fastest of single-device (`"off"`), pipeline (`"pp"`), tensor (`"tp"`) at a
    /// representative decode on THIS (system, model). Consumed at load when
    /// `[inference].multi_gpu` is unset AND `[tuning].auto_apply_multi_gpu`. `None` =
    /// no entry (never measured, or <2 GPUs so there was no choice).
    #[serde(default)]
    pub multi_gpu_mode: Option<String>,
    /// Winning CPU/GPU co-execution split for the routed-expert loop
    /// (q★-style), in permille of the top-K picks dispatched to the
    /// GPU while the rest run on CPU concurrently. `0` = co-execution
    /// off (the measured optimum on shared-DRAM iGPUs, where the GPU
    /// gains nothing from racing the CPU to the same memory bus).
    /// Only meaningful with `moe_placement = "uniform"`. `None` = no
    /// entry.
    #[serde(default)]
    pub moe_gpu_split_permille: Option<u32>,
    /// Autotune winner: enable MTP/NextN self-speculation decode for this
    /// model. Some(true)=faster with MTP, Some(false)=faster without, None=untuned.
    #[serde(default)]
    pub speculative_mtp: Option<bool>,
    /// Autotune winner: the best per-depth branching WIDTH vector for
    /// tree-structured MTP speculation (`[inference].speculative_tree_widths`),
    /// picked by the `tune --tree-widths` sweep as the candidate with the highest
    /// measured effective decode tok/s. `None` = untuned OR the model carries no
    /// MTP/NextN draft head (tree spec inert — the sweep SKIPs and writes
    /// nothing). Applied on load over the config default when
    /// `[tuning].auto_apply_tree_widths` is on. Stored as the raw width vector so
    /// it round-trips the TOML cache unchanged. `#[serde(default)]` → old caches
    /// (which lack the key) still load, no schema-version bump.
    #[serde(default)]
    pub winner_tree_widths: Option<Vec<u32>>,
    /// Autotune winner: the fastest speculative-decoding DRAFTER for this model,
    /// picked by `tune --spec-strategy` (a pure decode-tok/s argmax — spec is
    /// LOSSLESS, so there is no coherence gate). Keyed by model file STEM
    /// (PER-MODEL, unlike the legacy system-global `speculative_mtp` /
    /// `winner_tree_widths` which a second model on the same host would
    /// overwrite). Applied on load when `[tuning].auto_apply_spec_strategy` is on:
    /// the winner's drafter is enabled and the other five disabled.
    /// `#[serde(default)]` → old caches load, no schema bump. Empty map / missing
    /// stem ⇒ fall back to the legacy global fields.
    #[serde(default)]
    pub spec_strategy: HashMap<String, SpecStrategy>,
    /// Autotune winner: use the chunked-parallel SSM prefill scan (hybrid
    /// models). Some(true)=chunked prefill was faster, None=untuned.
    #[serde(default)]
    pub ssm_prefill_chunked: Option<bool>,
    /// P16 autotune axis: the per-host parallel-matvec crossover (M rows) — the
    /// threshold above which a quant/f16 CPU matvec goes rayon-parallel. Picked
    /// by `tune --parallel-matvec-crossover` (decode tok/s per candidate, best
    /// per `optimize_for`). `None` (serde default, so old caches load) → the
    /// engine keeps the historical 256 default
    /// (`rustllama_kernels_cpu::PARALLEL_MATVEC_CROSSOVER_DEFAULT`), byte-identical.
    #[serde(default)]
    pub parallel_matvec_crossover: Option<u32>,
    /// P16 autotune axis: the per-host GPU-vs-CPU matvec FLOP crossover — a
    /// matvec below this FLOP count stays on CPU rather than dispatching to the
    /// GPU. Picked by `tune --matvec-min-flops`. `None` (serde default) → the
    /// engine keeps the historical `0` default (no threshold — every matvec is
    /// GPU-eligible), byte-identical.
    #[serde(default)]
    pub matvec_min_flops: Option<u64>,
    /// P16 autotune axis: the per-host DeltaNet prefill chunk LENGTH (the tuner
    /// previously only toggled the chunked path on/off via `ssm_prefill_chunked`;
    /// this picks its length). Picked by `tune --ssm-chunk` (prefill tok/s per
    /// candidate). `None` (serde default) → the engine keeps the historical 64
    /// default (`rustllama_kernels_cpu::delta_net::SSM_CHUNK_DEFAULT`),
    /// byte-identical.
    #[serde(default)]
    pub ssm_chunk: Option<u32>,
    /// Per-model decision-probability calibration, keyed by model file
    /// stem like `placement`. Fit by the decision-calibration stage of
    /// `tune --all`: the temperature that best calibrates the model's
    /// Choice/Score/Boolean confidences against a labeled corpus.
    /// Applied by the `/v1/decide/*` endpoints when
    /// `[tuning].auto_apply_decision_calibration` is on.
    #[serde(default)]
    pub decision_calibration: HashMap<String, DecisionCalibration>,
    /// Phase 3 (multi-GPU): measured short-synthetic-decode throughput
    /// per COMPUTE DEVICE, keyed by a stable device slug and valued in
    /// decode tok/s (higher = faster). Populated by the mandatory
    /// per-device-perf stage of the autotune sweep (`tune --all` /
    /// first-load autotune). The keys are:
    ///   - each usable GPU → its UUID slug ([`sycl_gpu_key`] /
    ///     [`cuda_gpu_key`]), the SAME slug [`system_fingerprint`] dedups
    ///     on, so a device's perf survives an enumeration reorder;
    ///   - the CPU tier → [`cpu_perf_key`] (CPU package identity + the
    ///     enabled-core set), so a different CPU or a different
    ///     `disabled_cpus` set re-measures.
    /// The Phase 5 heat-placement planner reads this map to RANK the
    /// tiers (fastest first) and WEIGHT their budgets. Empty = not yet
    /// measured; the planner then declines and the engine keeps the
    /// existing VRAM-fit `auto_n_gpu_layers` loader until a tune runs.
    /// `#[serde(default)]` → old cache files (which lack the key) still
    /// load, so no schema-version bump is needed.
    #[serde(default)]
    pub per_device_perf: HashMap<String, f32>,
    /// On-device GPU-kernel validation verdicts, keyed by parity-probe
    /// name (e.g. `"gemm:mxfp4_tc"`, `"xmx:gemm"`, `"attn:decode_mxfp8"`)
    /// and valued `true` = the kernel matched its CPU reference on THIS
    /// machine, `false` = miscomputed. Populated by the `tune
    /// --validate-kernels` stage (runs the parity probes). The dispatch
    /// layer reads this to AUTO-ENABLE the specialized tensor-core / XMX
    /// / microscaling-KV paths only where they pass — replacing the old
    /// `RUSTLLAMA_FP4_TC` / `_FP8_WGMMA` / `_SYCL_XMX` env gating.
    /// Fail-closed: a probe absent from this map (never validated, or the
    /// device lacks the capability so the probe SKIPped) reads as "off".
    /// `#[serde(default)]` → old caches still load, no schema bump.
    #[serde(default)]
    pub kernel_verdicts: HashMap<String, bool>,
    /// Per-model autotuned CacheBlend recompute ratio (the fraction of chunk
    /// tokens recomputed when blending cached RAG-chunk KV into a prompt),
    /// keyed by model file stem like [`Self::moe_vram_margin`]. Fit by `tune
    /// --cacheblend`: the SMALLEST swept ratio whose blended next-token
    /// distribution stays coherent (cosine ≥ threshold) vs a full prefill on a
    /// synthetic multi-chunk context. `0.0` = the model stayed incoherent at
    /// every ratio → CacheBlend disabled for it. Absent = untuned. Applied when
    /// `[tuning].auto_apply_cacheblend` is on, overriding
    /// `[inference].cacheblend_recompute_ratio`.
    /// `#[serde(default)]` → old caches still load, no schema bump.
    #[serde(default)]
    pub cacheblend_recompute_ratio: HashMap<String, f32>,
    /// ISO-8601 timestamp of the last successful tune.
    pub last_tuned: Option<String>,
    pub rustllama_version: String,
    /// T4 — the [`kernel_build_id`] (`"{version}+{sycl}+{cuda}+{mlx}"`) stamped
    /// by the last `tune --validate-kernels`. The first-load verdict self-heal
    /// compares THIS (not just `rustllama_version`) against the current build's
    /// id, so a kernel-source edit WITHIN a crate version re-validates
    /// automatically. `#[serde(default)]` → old caches (which lack the key) load
    /// with `""`, which never equals a real build id ⇒ treated as stale ⇒
    /// re-validated (safe).
    #[serde(default)]
    pub kernel_build_id: String,
}

/// A model's fitted decision-calibration: a single temperature
/// applied to the per-option decision logits before softmax, plus the
/// before/after NLL as a quality record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionCalibration {
    pub temperature: f32,
    pub nll_before: f32,
    pub nll_after: f32,
    pub n_items: u32,
    /// Split-conformal nonconformity quantiles, keyed by target coverage
    /// formatted as `"{:.2}"` (e.g. `"0.80"`, `"0.90"`, `"0.95"`). The
    /// value `q` is the empirical ⌈(n+1)·coverage⌉/n quantile of the
    /// per-item nonconformity score `1 − calibrated P(true label)`, fit
    /// on the same labeled corpus as the temperature. The `/v1/decide/
    /// choice` endpoint uses it to build a prediction set: every option
    /// whose calibrated probability ≥ `1 − q` — the smallest set that
    /// meets the coverage guarantee. Empty (serde default) for caches
    /// written before this stage existed, so old files still load.
    #[serde(default)]
    pub conformal_q: HashMap<String, f32>,
}

impl TuningResult {
    pub fn empty(system_fp: String, device: DeviceFingerprint) -> Self {
        Self {
            schema_version: TUNING_SCHEMA_VERSION,
            system_fp,
            device,
            kernels: HashMap::new(),
            placement: HashMap::new(),
            moe_vram_margin: HashMap::new(),
            threads: None,
            batch_size: None,
            kv_dtype: None,
            kv_dtype_groups: None,
            flash_attention: None,
            kv_cache_layout: None,
            batching_engine: None,
            flash_kv_min: None,
            prefix_cache_max_snapshots: None,
            kv_page_size: None,
            flash_v3_kv_tile: None,
            flash_v3_kv_tile_by_bucket: None,
            moe_placement: None,
            multi_gpu_mode: None,
            moe_gpu_split_permille: None,
            speculative_mtp: None,
            winner_tree_widths: None,
            ssm_prefill_chunked: None,
            parallel_matvec_crossover: None,
            matvec_min_flops: None,
            ssm_chunk: None,
            decision_calibration: HashMap::new(),
            per_device_perf: HashMap::new(),
            kernel_verdicts: HashMap::new(),
            cacheblend_recompute_ratio: HashMap::new(),
            spec_strategy: HashMap::new(),
            last_tuned: None,
            rustllama_version: env!("CARGO_PKG_VERSION").to_string(),
            kernel_build_id: kernel_build_id(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuneDepth {
    Quick,
    Thorough,
}

#[derive(Debug, thiserror::Error)]
pub enum TunerError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("toml decode: {0}")]
    TomlDe(#[from] toml::de::Error),
    #[error("toml encode: {0}")]
    TomlSer(#[from] toml::ser::Error),
    #[error("tuning cache schema mismatch (file = {file_version}, expected = {expected})")]
    SchemaMismatch { file_version: u32, expected: u32 },
}

pub type Result<T> = std::result::Result<T, TunerError>;

/// On-disk cache directory — `%LOCALAPPDATA%\rustllama\tuning\` in
/// the default install, or `<exe>/rustllama-data/tuning/` in portable
/// mode. Sources truth from [`rustllama_runtime::paths`].
pub fn default_cache_dir() -> Option<PathBuf> {
    Some(rustllama_runtime::paths().tuning_dir.clone())
}

/// The tuner cache is keyed by the whole-SYSTEM fingerprint string
/// ([`system_fingerprint`]) — NOT a single SYCL device — so it resolves on
/// SYCL / CUDA / CPU-only hosts alike. `key` is that string.
pub fn cache_path_for(dir: &Path, key: &str) -> PathBuf {
    dir.join(format!("{key}.toml"))
}

pub fn load_cache(dir: &Path, key: &str) -> Result<Option<TuningResult>> {
    let p = cache_path_for(dir, key);
    if !p.exists() {
        return Ok(None);
    }
    let s = std::fs::read_to_string(&p)?;
    let parsed: TuningResult = toml::from_str(&s)?;
    if parsed.schema_version != TUNING_SCHEMA_VERSION {
        return Err(TunerError::SchemaMismatch {
            file_version: parsed.schema_version,
            expected: TUNING_SCHEMA_VERSION,
        });
    }
    Ok(Some(parsed))
}

pub fn save_cache(dir: &Path, result: &TuningResult) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    let p = cache_path_for(dir, &result.system_fp);
    let s = toml::to_string_pretty(result)?;
    std::fs::write(p, s)?;
    Ok(())
}

/// The tuner cache context every read/write site needs: the cache KEY (the
/// whole-system fingerprint, which always resolves) plus the SYCL
/// [`DeviceFingerprint`] to stamp into a freshly-created [`TuningResult`]
/// (default/empty on a non-SYCL host). Callers no longer bail when there's no
/// SYCL device — the system fingerprint keys the cache on CUDA/CPU hosts too.
pub fn cache_context() -> (String, DeviceFingerprint) {
    (
        system_fingerprint(),
        fingerprint_active_device().unwrap_or_default(),
    )
}

/// Probe the live SYCL device at `device_index` and produce a
/// [`DeviceFingerprint`] suitable for keying a [`TuningResult`].
///
/// The fingerprint hashes `(pci_id, driver_ver, name, vram_mb)`:
///   - `pci_id` from the SYCL `vendor_id` field. NB: SYCL's
///     `vendor_id` is the *vendor* ID (e.g. `0x8086` for Intel), not
///     the full PCI device ID. That's coarser than the plan's wording
///     suggests but is what the SYCL runtime exposes portably; a
///     vendor bump on a fresh card is rare enough that the
///     name + driver_ver pair carries the rest of the uniqueness.
///   - `driver_ver` as reported by `sycl::info::device::driver_version`.
///     A driver bump invalidates the cache entry automatically.
///   - `name` is the human-readable device name (e.g.
///     "Intel(R) Arc(TM) A380 Graphics"). Differentiates models with
///     the same vendor ID.
///   - `vram_mb` rounds VRAM to MiB; absorbs the small wobble that
///     `global_mem_size` can show between driver versions.
///
/// Returns `None` when SYCL is unavailable (mock build, no Intel GPU,
/// driver missing) — callers fall back to a CPU-only tune.
pub fn fingerprint_device(device_index: u32) -> Option<DeviceFingerprint> {
    let info = rustllama_kernels_sycl::device_info(device_index).ok()?;
    let vram_mb = info.vram_mb();
    Some(DeviceFingerprint {
        pci_id: info.vendor_id,
        driver_ver: info.driver_version,
        name: info.name,
        vram_mb,
    })
}

/// The SYCL device index the app actually dispatches on: among the backend
/// "views" of the (first) physical GPU, prefer **Level Zero**, else
/// **OpenCL**, else the first view. This mirrors the backend preference in
/// `rustllama_models::accel::first_enabled_sycl_device_index` (the Iris Xe
/// exposes L0 at one index and OpenCL at another) — WITHOUT the multi-GPU /
/// disable-list bookkeeping the tuner doesn't need. Keying the tuner cache
/// off this index (rather than a bare `0`) keeps the fingerprint — and thus
/// the cache file — stable regardless of SYCL enumeration order, which was
/// letting the L0 (`1.6.x`) vs OpenCL (`32.0.x`) `driver_version` flip
/// silently create a second cache and re-run first-load tuning.
pub fn preferred_device_index() -> u32 {
    let n = rustllama_kernels_sycl::device_count().unwrap_or(0);
    if n == 0 {
        return 0;
    }
    let mut opencl: Option<u32> = None;
    for i in 0..n {
        match rustllama_kernels_sycl::current_backend_name(i) {
            Some("level_zero") => return i,
            Some("opencl") if opencl.is_none() => opencl = Some(i),
            _ => {}
        }
    }
    opencl.unwrap_or(0)
}

/// Fingerprint the device the engine dispatches on ([`preferred_device_index`]).
/// Prefer this over `fingerprint_device(0)` everywhere the cache is keyed, so a
/// Level-Zero-first collapse (or any enumeration reorder) doesn't change the key.
pub fn fingerprint_active_device() -> Option<DeviceFingerprint> {
    fingerprint_device(preferred_device_index())
}

/// A stable, whole-SYSTEM fingerprint for keying **placement** — how a model is
/// split across ALL compute devices + CPU + RAM. Unlike [`DeviceFingerprint`]
/// (one SYCL device, for per-device *kernel* tuning), placement is a system
/// property: a 1-GPU box and a 3-mixed-GPU + big-RAM box need different splits.
///
/// It enumerates every GPU — all SYCL GPUs (deduped by device **UUID**, which
/// collapses the L0 + OpenCL views of one physical GPU) and all CUDA GPUs —
/// plus the CPU (logical cores) and total physical RAM (bucketed to GiB). Any
/// real topology change (add/remove/swap a GPU by UUID, change CPU or RAM)
/// changes the fingerprint → placement is re-tuned for the new system. A mere
/// driver update does NOT change it (UUIDs are driver-invariant). Falls back to
/// vendor/name/vram for any GPU that reports no UUID.
pub fn system_fingerprint() -> String {
    // BTreeSet → order-independent (enumeration order can't change the key) and
    // deduped (one physical GPU counts once across its backend views).
    let mut gpus: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

    let n_sycl = rustllama_kernels_sycl::device_count().unwrap_or(0);
    for i in 0..n_sycl {
        if let Ok(d) = rustllama_kernels_sycl::device_info(i) {
            gpus.insert(gpu_key("sycl", d.vendor_id, &d.name, d.vram_mb(), &d.uuid));
        }
    }
    let n_cuda = rustllama_kernels_cuda::device_count();
    for i in 0..n_cuda {
        if let Ok(d) = rustllama_kernels_cuda::device_info(i) {
            let vram_mb = d.total_mem_bytes / (1024 * 1024);
            // CUDA GPUs are NVIDIA (vendor 0x10de).
            gpus.insert(gpu_key("cuda", 0x10de, &d.name, vram_mb, &d.uuid));
        }
    }
    // Apple Metal GPUs — the 4th tier. Inert off Apple Silicon
    // (`device_count()` returns 0), so the fingerprint is unchanged on
    // Windows/Linux/Intel-mac. Keyed by `mlx_gpu_key` (registry-id-aware),
    // deduped into the same BTreeSet as the SYCL/CUDA GPUs.
    let n_mlx = rustllama_kernels_mlx::device_count();
    for i in 0..n_mlx {
        if let Ok(d) = rustllama_kernels_mlx::device_info(i) {
            gpus.insert(mlx_gpu_key(&d));
        }
    }

    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    let ram_gib = rustllama_runtime::memory_info().total_bytes / (1024 * 1024 * 1024);

    // CPU is a first-class tuning tier: fold in the package identity (brand
    // + CPUID family/model/stepping) AND the ENABLED-core set (all logical
    // procs minus `disabled_cpus`). A different CPU, OR the same CPU with a
    // different disabled-set, therefore yields a different fingerprint, so
    // tuning is CPU-config-aware — pinning work off the E-cores re-tunes.
    let topo = rustllama_runtime::cpu_topology();
    let disabled = rustllama_runtime::disabled_cpu_indices();
    let enabled: Vec<String> = topo
        .logical
        .iter()
        .map(|p| p.index)
        .filter(|i| !disabled.contains(i))
        .map(|i| i.to_string())
        .collect();
    let pkg = &topo.package;

    let mut parts: Vec<String> = gpus.into_iter().collect();
    parts.push(format!("cpu:{cores}"));
    parts.push(format!(
        "cpuid:{}:{:x}:{:x}:{:x}",
        pkg.brand, pkg.family, pkg.model, pkg.stepping
    ));
    parts.push(format!("cpu_enabled:{}", enabled.join(",")));
    parts.push(format!("ram:{ram_gib}GiB"));
    // NPUs (Intel/AMD) join the fingerprint exactly as GPUs do — but ONLY when
    // one is actually present. Adding "npu:none" unconditionally would change
    // every existing GPU/CPU-only key and force a needless re-tune, so the slug
    // is appended only for a non-empty usable NPU set (env-disabled applied).
    let npu_fp = rustllama_npu::npu_fingerprint(&[]);
    if npu_fp != "npu:none" {
        parts.push(npu_fp);
    }
    format!("{:016x}", fnv1a(&parts.join("|")))
}

/// One GPU's stable key: its UUID when the device reports one (driver-
/// invariant, and identical across the L0 + OpenCL views of one physical GPU),
/// else vendor/name/vram (still stable across driver + backend-view flips).
///
/// Public so the Phase 3 per-device-perf measurement records tok/s under the
/// SAME slug this module dedups on in [`system_fingerprint`], and the Phase 5
/// planner looks the score back up by the same key. Prefer the typed
/// [`sycl_gpu_key`] / [`cuda_gpu_key`] wrappers at call sites.
pub fn gpu_key(backend: &str, vendor: u32, name: &str, vram_mb: u64, uuid: &[u8; 16]) -> String {
    if uuid.iter().any(|&b| b != 0) {
        let hex: String = uuid.iter().map(|b| format!("{b:02x}")).collect();
        format!("{backend}:uuid:{hex}")
    } else {
        format!("{backend}:{vendor:04x}:{name}:{vram_mb}MiB")
    }
}

/// Stable per-device slug for a SYCL device — the exact key
/// [`system_fingerprint`] uses for this device. Use as the
/// `per_device_perf` map key for the Intel/SYCL tier.
pub fn sycl_gpu_key(d: &rustllama_kernels_sycl::DeviceInfo) -> String {
    gpu_key("sycl", d.vendor_id, &d.name, d.vram_mb(), &d.uuid)
}

/// Stable per-device slug for a CUDA device — the exact key
/// [`system_fingerprint`] uses for this device (CUDA GPUs are NVIDIA,
/// vendor `0x10de`). Use as the `per_device_perf` map key for the
/// NVIDIA/CUDA tier.
pub fn cuda_gpu_key(d: &rustllama_kernels_cuda::CudaDeviceInfo) -> String {
    let vram_mb = d.total_mem_bytes / (1024 * 1024);
    gpu_key("cuda", 0x10de, &d.name, vram_mb, &d.uuid)
}

/// Stable per-device slug for an Apple Metal GPU — the exact key
/// [`system_fingerprint`] uses for this device. Mirror of [`cuda_gpu_key`]
/// (Apple GPUs use vendor `0x106b`, Apple Inc.'s PCI vendor ID).
///
/// Metal has no CUDA-style 16-byte device UUID, so `MlxDeviceInfo.uuid` is
/// all-zero until Phase 1 synthesizes one from `registry_id`. To keep the
/// key stable + per-device-unique even in Phase 0, fold the driver-invariant
/// `registry_id` into the name so the name/vram FALLBACK in [`gpu_key`]
/// distinguishes devices (two identically-named Metal GPUs — never on
/// today's single-GPU Apple Silicon — would otherwise collide). Once Phase 1
/// sets a real UUID, `gpu_key` takes the UUID path and ignores the name.
pub fn mlx_gpu_key(d: &rustllama_kernels_mlx::MlxDeviceInfo) -> String {
    let vram_mb = d.total_mem_bytes / (1024 * 1024);
    let name = format!("{}#{:x}", d.name, d.registry_id);
    gpu_key("mlx", 0x106b, &name, vram_mb, &d.uuid)
}

/// Stable per-device slug for the CPU compute tier: the CPU package
/// identity (brand + CPUID family/model/stepping — the same `cpuid:…`
/// piece [`system_fingerprint`] folds in) PLUS the ENABLED-core set
/// (all logical procs minus `disabled_cpus`). A different CPU, OR the
/// same CPU with a different disabled-set (fewer/other cores pinned),
/// therefore yields a different key, so the CPU tier's measured tok/s
/// re-measures when the pinned tier changes. Non-alphanumeric chars in
/// the brand are collapsed to `_` so the slug stays a clean map/TOML key.
pub fn cpu_perf_key() -> String {
    let topo = rustllama_runtime::cpu_topology();
    let disabled = rustllama_runtime::disabled_cpu_indices();
    let enabled: Vec<String> = topo
        .logical
        .iter()
        .map(|p| p.index)
        .filter(|i| !disabled.contains(i))
        .map(|i| i.to_string())
        .collect();
    let pkg = &topo.package;
    let safe_brand: String = pkg
        .brand
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    format!(
        "cpu:cpuid:{}:{:x}:{:x}:{:x}:enabled:{}",
        safe_brand,
        pkg.family,
        pkg.model,
        pkg.stepping,
        enabled.join(",")
    )
}

/// Stable per-device slug for an NPU tier's measured perf — the NPU analog of
/// [`cpu_perf_key`] / the GPU `*_gpu_key`s. Keyed by vendor + name + index so
/// the Stage-1b NPU bench and the heat planner agree on the `per_device_perf`
/// entry (Phase 3). Non-alphanumeric chars in the name collapse to `_`.
pub fn npu_perf_key(npu: &rustllama_npu::NpuInfo) -> String {
    let safe_name: String = npu
        .name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    format!("npu:{:?}:{}:{}", npu.vendor, safe_name, npu.index)
}

/// One tree-width candidate's measured result, as handed to [`pick_tree_widths`].
///
/// The `tune --tree-widths` sweep drives tree-structured MTP speculation once
/// per candidate width vector and fills one of these per candidate. `tok_s` is
/// the median EFFECTIVE decode throughput (the real objective — accepted tokens
/// per wall-second, so a wider tree that drafts more but gets rejected more is
/// correctly penalized); `accept` is the speculative accept rate (drafted →
/// accepted) for that candidate — REPORTED for insight, NOT used to rank. A
/// candidate whose measurement failed carries `tok_s == None` and is skipped by
/// the picker.
///
/// The type (and the picker below) live in this toolchain-free crate so the
/// selection logic can be unit-tested with plain `cargo test -p rustllama-tuner`
/// — the on-device measurement itself is write-blind (no small MTP-head GGUF is
/// public), so the DECISION logic is what the tests pin.
#[derive(Debug, Clone)]
pub struct TreeWidthMeasurement {
    /// The per-depth branching width vector this result measured.
    pub widths: Vec<u32>,
    /// Median effective decode tok/s, or `None` if every run for this candidate
    /// failed.
    pub tok_s: Option<f64>,
    /// Speculative accept rate (accepted / drafted) for this candidate, or
    /// `None` if it could not be computed. Reported only.
    pub accept: Option<f64>,
}

/// Pick the winning tree-width vector from a sweep's per-candidate measurements.
///
/// Objective: MAXIMIZE measured effective decode tok/s (`tok_s`) — the real
/// end-to-end speed the user feels, already net of wasted verify compute on
/// rejected branches. Candidates whose measurement failed (`tok_s == None`) are
/// ignored. Returns `None` when NO candidate produced a measurement (the caller
/// then leaves the config default in place).
///
/// Ties are broken DETERMINISTICALLY so the same inputs always yield the same
/// winner regardless of candidate ordering or sub-noise-floor tok/s wobble: when
/// two candidates are within [`TREE_WIDTHS_TPS_EPS`] tok/s of each other, prefer
/// the SMALLER tree (fewer total nodes = `sum(widths)` = less verify compute and
/// KV footprint per round), and if still tied prefer the candidate that appears
/// EARLIER in the input slice. A strictly faster candidate (beyond the epsilon)
/// always wins outright — the tie-break only decides a statistical dead heat.
pub fn pick_tree_widths(candidates: &[TreeWidthMeasurement]) -> Option<Vec<u32>> {
    let mut best: Option<(&TreeWidthMeasurement, f64, u32)> = None; // (cand, tok_s, node_sum)
    for cand in candidates {
        let Some(tps) = cand.tok_s else { continue };
        let nodes: u32 = cand.widths.iter().copied().sum();
        best = Some(match best {
            None => (cand, tps, nodes),
            Some((b_cand, b_tps, b_nodes)) => {
                if tps > b_tps + TREE_WIDTHS_TPS_EPS {
                    // Strictly faster → new winner.
                    (cand, tps, nodes)
                } else if tps + TREE_WIDTHS_TPS_EPS < b_tps {
                    // Strictly slower → keep the incumbent.
                    (b_cand, b_tps, b_nodes)
                } else if nodes < b_nodes {
                    // Dead heat on tok/s → prefer the cheaper (smaller) tree.
                    (cand, tps, nodes)
                } else {
                    // Dead heat AND not smaller → keep the earlier candidate
                    // (iteration order == input order, so this is stable).
                    (b_cand, b_tps, b_nodes)
                }
            }
        });
    }
    best.map(|(c, _, _)| c.widths.clone())
}

/// A speculative-decoding drafter strategy. The `tune --spec-strategy` stage
/// measures each AVAILABLE strategy's decode tok/s on a model and picks the
/// fastest. Spec decoding is LOSSLESS (every draft is target-verified, so output
/// is byte-identical to classic decode), so the choice is a pure speed argmax —
/// no coherence gate. Persisted per model in [`TuningResult::spec_strategy`] and
/// auto-applied on load. Serialized as a snake_case string so it round-trips the
/// TOML cache unchanged (same rationale as the kv_dtype winner).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpecStrategy {
    /// Classic decode, no drafter (the baseline — wins when no drafter helps).
    None,
    /// Prompt-lookup n-gram drafter (self-contained, always available).
    Ngram,
    /// Lookahead / Jacobi drafter (self-contained, always available).
    Lookahead,
    /// MTP / NextN self-speculation (needs the model's own NextN head).
    Mtp,
    /// Tree-structured MTP speculation (needs the NextN head + tuned widths).
    Tree,
    /// EAGLE draft head (only when a companion head GGUF is already attached).
    Eagle,
    /// External paired draft model (only when a tokenizer-compatible draft is
    /// already paired).
    Draft,
}

impl SpecStrategy {
    /// Complexity/cost rank for the dead-heat tie-break (lower = simpler/cheaper,
    /// preferred when tok/s is a statistical tie): no drafter < self-contained <
    /// NextN self-spec < companion model. This avoids paying drafter overhead +
    /// configuration complexity unless a drafter is CLEARLY faster.
    fn cost_rank(self) -> u8 {
        match self {
            SpecStrategy::None => 0,
            SpecStrategy::Ngram => 1,
            SpecStrategy::Lookahead => 2,
            SpecStrategy::Mtp => 3,
            SpecStrategy::Tree => 4,
            SpecStrategy::Eagle => 5,
            SpecStrategy::Draft => 6,
        }
    }
}

/// One strategy's measured decode tok/s from the `--spec-strategy` sweep.
/// `tok_s == None` = the candidate failed or was skipped (not capable).
#[derive(Debug, Clone)]
pub struct SpecStrategyMeasurement {
    pub strategy: SpecStrategy,
    pub tok_s: Option<f64>,
    pub error: Option<String>,
}

/// Epsilon (tok/s) for the spec-strategy dead-heat tie-break — same role as
/// [`TREE_WIDTHS_TPS_EPS`].
pub const SPEC_STRATEGY_TPS_EPS: f64 = TREE_WIDTHS_TPS_EPS;

/// Pick the fastest spec strategy: MAXIMIZE measured decode tok/s. On a dead heat
/// (within [`SPEC_STRATEGY_TPS_EPS`]) prefer the LOWER-cost strategy (see
/// [`SpecStrategy::cost_rank`]), then the earlier candidate. `None` when no
/// candidate produced a measurement (caller keeps the config default). Because
/// spec is lossless, the winner is purely the fastest — no coherence check.
pub fn pick_spec_strategy(candidates: &[SpecStrategyMeasurement]) -> Option<SpecStrategy> {
    let mut best: Option<(SpecStrategy, f64, u8)> = None; // (strategy, tok_s, cost)
    for cand in candidates {
        let Some(tps) = cand.tok_s else { continue };
        let cost = cand.strategy.cost_rank();
        best = Some(match best {
            None => (cand.strategy, tps, cost),
            Some((b_s, b_tps, b_cost)) => {
                if tps > b_tps + SPEC_STRATEGY_TPS_EPS {
                    (cand.strategy, tps, cost)
                } else if tps + SPEC_STRATEGY_TPS_EPS < b_tps {
                    (b_s, b_tps, b_cost)
                } else if cost < b_cost {
                    (cand.strategy, tps, cost)
                } else {
                    (b_s, b_tps, b_cost)
                }
            }
        });
    }
    best.map(|(s, _, _)| s)
}

/// Tok/s dead-heat threshold for [`pick_tree_widths`]'s tie-break. Two
/// candidates within this of each other are treated as equal and decided by the
/// deterministic smaller-tree / earlier-index rule rather than by measurement
/// noise. Small but non-zero so bit-identical floats and sub-noise wobble both
/// resolve the same way.
pub const TREE_WIDTHS_TPS_EPS: f64 = 1e-6;

// ============================================================
// P16: autotune objective + numeric perf-knob picker
// ============================================================

/// The objective the autotune sweeps optimize against. NOT a swept value — it
/// is the MODE the other sweeps rank candidates by. `Throughput` maximizes
/// decode/prefill tok/s (the default, byte-identical to the historical tuner);
/// `Watts` maximizes energy efficiency (tokens-per-joule) for laptop / battery-
/// sensitive hosts. Resolved flag > env (`RUSTLLAMA_TUNE_OPTIMIZE_FOR`) > config
/// (`[tuning].optimize_for`) by the CLI; threaded into the measurement loop in
/// place of the former inline env read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizeFor {
    /// Maximize tok/s (default).
    #[default]
    Throughput,
    /// Maximize tokens-per-joule (energy efficiency).
    Watts,
}

impl OptimizeFor {
    /// Parse a config/flag/env string. Accepts `"throughput"`/`"tps"`/`"speed"`
    /// and `"watts"`/`"joule"`/`"energy"`/`"efficiency"` (case-insensitive).
    /// Returns `None` on an unrecognized value so the caller can keep the default.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "throughput" | "tps" | "tok/s" | "toks" | "speed" | "" => Some(OptimizeFor::Throughput),
            "watts" | "watt" | "joule" | "joules" | "energy" | "efficiency" | "tok/joule" => {
                Some(OptimizeFor::Watts)
            }
            _ => None,
        }
    }

    /// Canonical config string (round-trips through [`Self::parse`]).
    pub fn as_str(&self) -> &'static str {
        match self {
            OptimizeFor::Throughput => "throughput",
            OptimizeFor::Watts => "watts",
        }
    }

    /// The score this objective ranks a candidate by, given its measured tok/s
    /// and tokens-per-joule. For `Watts`, fall back to tok/s when no energy
    /// reading is available (non-Intel host / no power domain), so the picker
    /// still ranks sensibly rather than discarding every candidate.
    pub fn score(&self, tok_s: Option<f64>, tokens_per_joule: Option<f64>) -> Option<f64> {
        match self {
            OptimizeFor::Throughput => tok_s,
            OptimizeFor::Watts => tokens_per_joule.or(tok_s),
        }
    }
}

/// One numeric perf-knob candidate's measurement (shared by the
/// `parallel_matvec_crossover` / `matvec_min_flops` / `ssm_chunk` sweeps). The
/// `value` is the knob setting measured; `tok_s` / `tokens_per_joule` are its
/// median decode-or-prefill throughput and energy efficiency (the latter `None`
/// off a power-instrumented host). A candidate whose every run failed carries
/// `tok_s == None` and is skipped by the picker.
///
/// The type + the picker below live in this toolchain-free crate so the
/// selection logic is unit-testable with plain `cargo test -p rustllama-tuner`
/// — the on-device measurement itself is HW/model-gated, so the DECISION logic
/// is what the tests pin.
#[derive(Debug, Clone)]
pub struct PerfCandidate {
    /// The knob value this candidate measured.
    pub value: u64,
    /// Median throughput (decode or prefill tok/s), or `None` if every run
    /// failed.
    pub tok_s: Option<f64>,
    /// Tokens-per-joule for this candidate, or `None` when no power reading was
    /// available. Ranked only under `OptimizeFor::Watts`.
    pub tokens_per_joule: Option<f64>,
}

/// Relative dead-heat threshold for [`pick_perf_knob`]. Two candidates whose
/// scores are within this FRACTION of each other are a statistical tie and
/// decided by the deterministic default-preference tie-break rather than by
/// measurement noise. 0.5% is well under run-to-run decode-tok/s wobble on these
/// hosts while still letting a real win decide outright.
pub const PERF_KNOB_REL_EPS: f64 = 0.005;

/// Pick the winning numeric perf-knob value from a sweep's per-candidate
/// measurements, maximizing the score for `objective` (tok/s for `Throughput`,
/// tokens-per-joule for `Watts`).
///
/// Candidates whose measurement failed (score `None`) are ignored. Returns
/// `None` when NO candidate produced a usable score — the caller then keeps the
/// historical hard-coded default, so the knob stays byte-identical to pre-P16.
///
/// Ties are broken DETERMINISTICALLY toward SAFETY: when two candidates are
/// within [`PERF_KNOB_REL_EPS`] of each other, prefer the one EQUAL to
/// `default_value` (so a dead heat keeps today's constant → byte-identical),
/// else the one CLOSEST to `default_value`, else the candidate appearing EARLIER
/// in the slice. A candidate that beats the field beyond the epsilon always wins
/// outright — the tie-break only decides a statistical dead heat, so the sweep
/// never ships a worse-than-default value on noise alone.
pub fn pick_perf_knob(
    candidates: &[PerfCandidate],
    objective: OptimizeFor,
    default_value: u64,
) -> Option<u64> {
    // (value, score, is_default, distance-to-default)
    let mut best: Option<(u64, f64)> = None;
    for cand in candidates {
        let Some(score) = objective.score(cand.tok_s, cand.tokens_per_joule) else {
            continue;
        };
        best = Some(match best {
            None => (cand.value, score),
            Some((b_val, b_score)) => {
                // Relative epsilon around the incumbent's score.
                let eps = b_score.abs().max(1.0) * PERF_KNOB_REL_EPS;
                if score > b_score + eps {
                    // Strictly better → new winner.
                    (cand.value, score)
                } else if score + eps < b_score {
                    // Strictly worse → keep incumbent.
                    (b_val, b_score)
                } else {
                    // Dead heat → prefer the default, else the value closest to
                    // the default, else keep the earlier (incumbent) candidate.
                    let cand_is_def = cand.value == default_value;
                    let inc_is_def = b_val == default_value;
                    if cand_is_def && !inc_is_def {
                        (cand.value, score)
                    } else if inc_is_def && !cand_is_def {
                        (b_val, b_score)
                    } else {
                        let cand_dist = cand.value.abs_diff(default_value);
                        let inc_dist = b_val.abs_diff(default_value);
                        if cand_dist < inc_dist {
                            (cand.value, score)
                        } else {
                            (b_val, b_score)
                        }
                    }
                }
            }
        });
    }
    best.map(|(v, _)| v)
}

/// FNV-1a 64-bit — deterministic across processes (unlike `DefaultHasher`,
/// whose seed is randomized), so the system-fingerprint slug is stable.
fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    /// T4: the build id is well-formed — four `+`-separated parts
    /// (`version+sycl+cuda+mlx`), the first being the crate version, and the
    /// kernel-hash parts are the real 16-hex FNV digests the three kernel
    /// `build.rs` scripts emitted (never the `env!` default). Proves the build
    /// wiring reached the tuner.
    #[test]
    fn kernel_build_id_is_well_formed() {
        let id = kernel_build_id();
        let parts: Vec<&str> = id.split('+').collect();
        assert_eq!(parts.len(), 4, "expected version+sycl+cuda+mlx, got {id:?}");
        assert_eq!(parts[0], env!("CARGO_PKG_VERSION"));
        for h in &parts[1..] {
            assert_eq!(h.len(), 16, "kernel hash {h:?} is not 16 hex chars");
            assert!(
                h.chars().all(|c| c.is_ascii_hexdigit()),
                "kernel hash {h:?} is not hex"
            );
        }
        // An empty `TuningResult` stamps the current build id; an old cache
        // (serde default `""`) never matches it, so it self-heals (re-validates).
        let empty = TuningResult::empty("fp".into(), DeviceFingerprint::default());
        assert_eq!(empty.kernel_build_id, id);
        assert_ne!(String::new(), id);
    }

    /// T4 anchor (the plan's "a unit test comparing two hashes"): the same
    /// FNV-1a-64 fold the kernel `build.rs` scripts use is edit-sensitive — a
    /// one-byte whitespace change to a kernel source flips the digest, so the
    /// build id changes and the first-load self-heal re-validates on next load.
    #[test]
    fn kernel_source_hash_changes_on_whitespace_edit() {
        // Mirrors `fnv1a_kernel_sources` in each kernel crate's build.rs: fold
        // each file's NAME then its BYTES into one FNV-1a-64 (name-then-bytes so
        // a rename also changes the digest).
        fn fnv1a(files: &[(&str, &[u8])]) -> String {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for (name, bytes) in files {
                // name-then-bytes, matching `fnv1a_kernel_sources` in build.rs.
                for &b in name.as_bytes().iter().chain(bytes.iter()) {
                    h ^= b as u64;
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
            format!("{h:016x}")
        }
        let before = fnv1a(&[("rsl_kernels.cpp", b"// kernel source\nint x = 1;\n")]);
        // One extra whitespace char (a no-op comment edit the plan describes).
        let after = fnv1a(&[("rsl_kernels.cpp", b"// kernel source \nint x = 1;\n")]);
        assert_ne!(before, after, "FNV digest must change on a whitespace edit");
        assert_eq!(before.len(), 16);
    }

    /// C-A: the fused gate+up dispatch is gated on TWO verdict families — the
    /// bare-GEMM parity (`fused:`) AND the real-MoE coherence (`moe:gate_up_
    /// coherent:`). Pin that both prefixes exist, are distinct, and that a
    /// per-dtype coherence verdict round-trips through the cache helpers so
    /// `verdict_for` reads back what `set_verdict` wrote (fail-closed on absence).
    #[test]
    fn moe_gate_up_coherence_verdict_roundtrips() {
        assert_eq!(VERDICT_FUSED_GATE_UP_PREFIX, "fused:");
        assert_eq!(VERDICT_MOE_GATE_UP_COHERENT_PREFIX, "moe:gate_up_coherent:");
        assert_ne!(
            VERDICT_FUSED_GATE_UP_PREFIX,
            VERDICT_MOE_GATE_UP_COHERENT_PREFIX
        );
        let mut t = TuningResult::empty("fp".into(), DeviceFingerprint::default());
        let key = format!("{VERDICT_MOE_GATE_UP_COHERENT_PREFIX}iq2_xxs");
        // Absent ⇒ unset (the dispatch gate treats this as OFF).
        assert_eq!(verdict_for(&t, &key), None);
        set_verdict(&mut t, &key, true);
        assert_eq!(verdict_for(&t, &key), Some(true));
        set_verdict(&mut t, &key, false);
        assert_eq!(verdict_for(&t, &key), Some(false));
    }

    /// P6-fusions + P19: the three fusion verdict keys are stable strings, a
    /// fresh cache has them UNSET (fail-closed OFF — the forward stays
    /// byte-identical), and the new per-bucket FA-tile field defaults to `None`
    /// (so the kernel keeps its built-in heuristic, byte-identical pre-P19).
    #[test]
    fn p6_fusion_verdicts_and_fa_bucket_defaults() {
        assert_eq!(VERDICT_FUSED_RMSNORM_MATVEC, "fused:rmsnorm_matvec");
        assert_eq!(VERDICT_FUSED_DOWN_RESIDUAL, "fused:down_residual");
        assert_eq!(VERDICT_FUSED_OUT_PROJ_NORM, "fused:out_proj_norm");
        let mut t = TuningResult::empty("fp".into(), DeviceFingerprint::default());
        for k in [
            VERDICT_FUSED_RMSNORM_MATVEC,
            VERDICT_FUSED_DOWN_RESIDUAL,
            VERDICT_FUSED_OUT_PROJ_NORM,
        ] {
            assert_eq!(
                verdict_for(&t, k),
                None,
                "{k} must be unset on a fresh cache"
            );
        }
        // P19: per-bucket FA tile winners default to None (byte-identical).
        assert!(t.flash_v3_kv_tile_by_bucket.is_none());
        // Round-trips through the verdict helpers like the others.
        set_verdict(&mut t, VERDICT_FUSED_OUT_PROJ_NORM, true);
        assert_eq!(verdict_for(&t, VERDICT_FUSED_OUT_PROJ_NORM), Some(true));
    }

    #[test]
    fn slug_is_filesystem_safe() {
        let fp = DeviceFingerprint {
            pci_id: 0x1234_5678,
            driver_ver: "32.0.101.5333".into(),
            name: "Intel(R) Arc(TM) A380 Graphics".into(),
            vram_mb: 6144,
        };
        let s = fp.slug();
        assert!(s.starts_with("12345678-"));
        assert!(!s.contains('('));
        assert!(!s.contains(' '));
    }

    #[test]
    fn fingerprint_device_probe_does_not_panic() {
        // Host-agnostic: on a no-SYCL host (CI) the live probe returns
        // None; on a host with a real SYCL GPU (a dev box with an Iris Xe
        // or Arc) it returns Some. Either way the probe must not panic,
        // and any fingerprint it returns must be well-formed — a named
        // device. (We can't assert concrete field values without pinning
        // to specific hardware.)
        if let Some(fp) = fingerprint_device(0) {
            let DeviceFingerprint {
                pci_id: _,
                driver_ver: _,
                name,
                vram_mb: _,
            } = &fp;
            assert!(
                !name.is_empty(),
                "a probed SYCL device must report a non-empty name: {fp:?}"
            );
        }
    }

    #[test]
    fn roundtrip_empty() {
        let fp = DeviceFingerprint {
            pci_id: 1,
            driver_ver: "v".into(),
            name: "test".into(),
            vram_mb: 0,
        };
        let r = TuningResult::empty("test-sys".to_string(), fp);
        let s = toml::to_string(&r).unwrap();
        let back: TuningResult = toml::from_str(&s).unwrap();
        assert_eq!(back.schema_version, TUNING_SCHEMA_VERSION);
    }

    #[test]
    fn q4k_shape_bucket_is_stable() {
        // Pin the format string so cached entries from older builds
        // keep matching after a tuner version bump.
        assert_eq!(q4k_usm_shape_bucket(1536, 1536), "M=1536,K=1536");
        assert_eq!(q4k_usm_shape_bucket(256, 1024), "M=256,K=1024");
    }

    #[test]
    fn tuned_q4k_lws_round_trip_through_cache() {
        let fp = DeviceFingerprint {
            pci_id: 0x8086,
            driver_ver: "32.0.101.7076".into(),
            name: "Iris Xe".into(),
            vram_mb: 0,
        };
        let mut r = TuningResult::empty("test-sys".to_string(), fp);
        // Empty cache → no tuned LWS.
        assert_eq!(tuned_q4k_usm_lws(&r, 1536, 1536), None);

        // Record winners for two distinct (M, K) shapes.
        set_tuned_q4k_usm_lws(&mut r, 1536, 1536, 64);
        set_tuned_q4k_usm_lws(&mut r, 8960, 1536, 128);

        assert_eq!(tuned_q4k_usm_lws(&r, 1536, 1536), Some(64));
        assert_eq!(tuned_q4k_usm_lws(&r, 8960, 1536), Some(128));
        // Unknown shape stays None.
        assert_eq!(tuned_q4k_usm_lws(&r, 999, 999), None);
    }

    #[test]
    fn tuned_q4k_lws_survives_toml_roundtrip() {
        // The whole point of the cache is that it persists across
        // process restarts. Verify the JSON-in-TOML serialization
        // doesn't lose the lws values.
        let fp = DeviceFingerprint {
            pci_id: 0x8086,
            driver_ver: "32.0.101.7076".into(),
            name: "Iris Xe".into(),
            vram_mb: 0,
        };
        let mut r = TuningResult::empty("test-sys".to_string(), fp);
        set_tuned_q4k_usm_lws(&mut r, 1536, 1536, 64);
        set_tuned_q4k_usm_lws(&mut r, 8960, 1536, 128);
        set_tuned_q4k_usm_lws(&mut r, 256, 1536, 32);

        let s = toml::to_string(&r).expect("toml encode");
        let back: TuningResult = toml::from_str(&s).expect("toml decode");
        assert_eq!(tuned_q4k_usm_lws(&back, 1536, 1536), Some(64));
        assert_eq!(tuned_q4k_usm_lws(&back, 8960, 1536), Some(128));
        assert_eq!(tuned_q4k_usm_lws(&back, 256, 1536), Some(32));
    }

    #[test]
    fn pick_tree_widths_maximizes_tok_s() {
        // The picker's whole job: return the candidate with the highest measured
        // effective decode tok/s. `[4, 2]` is fastest here, so it wins outright.
        let cands = vec![
            TreeWidthMeasurement {
                widths: vec![2],
                tok_s: Some(10.0),
                accept: Some(0.5),
            },
            TreeWidthMeasurement {
                widths: vec![2, 2],
                tok_s: Some(12.0),
                accept: Some(0.6),
            },
            TreeWidthMeasurement {
                widths: vec![4, 2],
                tok_s: Some(15.0),
                accept: Some(0.55),
            },
            TreeWidthMeasurement {
                widths: vec![2, 2, 2, 2],
                tok_s: Some(11.0),
                accept: Some(0.7),
            },
        ];
        assert_eq!(pick_tree_widths(&cands), Some(vec![4, 2]));
    }

    #[test]
    fn pick_tree_widths_skips_failed_and_handles_empty() {
        // A failed measurement (`tok_s == None`) is ignored, even if it would
        // otherwise be "first". With every candidate failed the picker yields
        // None (caller keeps the config default).
        let cands = vec![
            TreeWidthMeasurement {
                widths: vec![2],
                tok_s: None,
                accept: None,
            },
            TreeWidthMeasurement {
                widths: vec![2, 2],
                tok_s: Some(9.5),
                accept: Some(0.4),
            },
        ];
        assert_eq!(pick_tree_widths(&cands), Some(vec![2, 2]));

        let all_failed = vec![
            TreeWidthMeasurement {
                widths: vec![2],
                tok_s: None,
                accept: None,
            },
            TreeWidthMeasurement {
                widths: vec![4, 2],
                tok_s: None,
                accept: None,
            },
        ];
        assert_eq!(pick_tree_widths(&all_failed), None);
        assert_eq!(pick_tree_widths(&[]), None);
    }

    #[test]
    fn pick_tree_widths_breaks_ties_deterministically() {
        // Dead heat on tok/s (identical values): the tie-break prefers the
        // SMALLER tree (fewer total nodes). `[2,2]` (sum 4) beats `[4,2,2]`
        // (sum 8) despite appearing later — a hand-checkable, order-independent
        // outcome.
        let cands = vec![
            TreeWidthMeasurement {
                widths: vec![4, 2, 2],
                tok_s: Some(13.0),
                accept: Some(0.6),
            },
            TreeWidthMeasurement {
                widths: vec![2, 2],
                tok_s: Some(13.0),
                accept: Some(0.5),
            },
        ];
        assert_eq!(pick_tree_widths(&cands), Some(vec![2, 2]));

        // Equal tok/s AND equal node sum → the EARLIER candidate wins (stable).
        let equal_size = vec![
            TreeWidthMeasurement {
                widths: vec![3, 1],
                tok_s: Some(7.0),
                accept: Some(0.3),
            },
            TreeWidthMeasurement {
                widths: vec![2, 2],
                tok_s: Some(7.0),
                accept: Some(0.9),
            },
        ];
        assert_eq!(pick_tree_widths(&equal_size), Some(vec![3, 1]));

        // A strictly faster (beyond-eps) candidate wins regardless of its size,
        // so the tie-break never overrides a real speed difference.
        let real_diff = vec![
            TreeWidthMeasurement {
                widths: vec![2],
                tok_s: Some(10.0),
                accept: Some(0.5),
            },
            TreeWidthMeasurement {
                widths: vec![4, 4, 2],
                tok_s: Some(20.0),
                accept: Some(0.8),
            },
        ];
        assert_eq!(pick_tree_widths(&real_diff), Some(vec![4, 4, 2]));
    }

    // ---- P16: OptimizeFor + pick_perf_knob ----

    #[test]
    fn optimize_for_parse_roundtrips() {
        assert_eq!(
            OptimizeFor::parse("throughput"),
            Some(OptimizeFor::Throughput)
        );
        assert_eq!(OptimizeFor::parse("TPS"), Some(OptimizeFor::Throughput));
        assert_eq!(OptimizeFor::parse("watts"), Some(OptimizeFor::Watts));
        assert_eq!(OptimizeFor::parse("Energy"), Some(OptimizeFor::Watts));
        assert_eq!(OptimizeFor::parse("nonsense"), None);
        // default is Throughput; canonical strings round-trip.
        assert_eq!(OptimizeFor::default(), OptimizeFor::Throughput);
        assert_eq!(
            OptimizeFor::parse(OptimizeFor::Throughput.as_str()),
            Some(OptimizeFor::Throughput)
        );
        assert_eq!(
            OptimizeFor::parse(OptimizeFor::Watts.as_str()),
            Some(OptimizeFor::Watts)
        );
    }

    /// Throughput objective: the picker returns the value with the highest
    /// tok/s (a clear win beyond the relative epsilon decides outright).
    #[test]
    fn pick_perf_knob_throughput_maximizes_tps() {
        let cands = vec![
            PerfCandidate {
                value: 64,
                tok_s: Some(10.0),
                tokens_per_joule: Some(5.0),
            },
            PerfCandidate {
                value: 128,
                tok_s: Some(12.0),
                tokens_per_joule: Some(4.0),
            },
            PerfCandidate {
                value: 256,
                tok_s: Some(15.0),
                tokens_per_joule: Some(3.0),
            },
            PerfCandidate {
                value: 512,
                tok_s: Some(11.0),
                tokens_per_joule: Some(2.0),
            },
        ];
        assert_eq!(
            pick_perf_knob(&cands, OptimizeFor::Throughput, 256),
            Some(256)
        );
    }

    /// Watts objective picks a DIFFERENT winner than throughput on the same
    /// candidate set — proving the objective actually steers the pick.
    #[test]
    fn pick_perf_knob_watts_picks_per_joule_winner() {
        let cands = vec![
            PerfCandidate {
                value: 64,
                tok_s: Some(10.0),
                tokens_per_joule: Some(5.0),
            },
            PerfCandidate {
                value: 128,
                tok_s: Some(12.0),
                tokens_per_joule: Some(4.0),
            },
            PerfCandidate {
                value: 256,
                tok_s: Some(15.0),
                tokens_per_joule: Some(3.0),
            },
        ];
        // Throughput → 256 (15 tok/s); Watts → 64 (5 tok/J).
        assert_eq!(
            pick_perf_knob(&cands, OptimizeFor::Throughput, 256),
            Some(256)
        );
        assert_eq!(pick_perf_knob(&cands, OptimizeFor::Watts, 256), Some(64));
    }

    /// A cached winner OVERRIDES the default: when a non-default value is
    /// strictly fastest, the picker returns it (not the default).
    #[test]
    fn pick_perf_knob_winner_overrides_default() {
        let cands = vec![
            PerfCandidate {
                value: 256,
                tok_s: Some(10.0),
                tokens_per_joule: None,
            }, // default
            PerfCandidate {
                value: 1024,
                tok_s: Some(18.0),
                tokens_per_joule: None,
            },
        ];
        assert_eq!(
            pick_perf_knob(&cands, OptimizeFor::Throughput, 256),
            Some(1024)
        );
    }

    /// Absent winner → None, so the caller keeps the historical constant
    /// (byte-identical). Also: a statistical dead heat resolves TOWARD the
    /// default, so noise alone never ships a worse-than-default value.
    #[test]
    fn pick_perf_knob_absent_and_dead_heat_keep_default() {
        // No measurement at all → None (caller uses the hard-coded default).
        assert_eq!(pick_perf_knob(&[], OptimizeFor::Throughput, 256), None);
        let all_failed = vec![
            PerfCandidate {
                value: 64,
                tok_s: None,
                tokens_per_joule: None,
            },
            PerfCandidate {
                value: 256,
                tok_s: None,
                tokens_per_joule: None,
            },
        ];
        assert_eq!(
            pick_perf_knob(&all_failed, OptimizeFor::Throughput, 256),
            None
        );

        // Dead heat (all within the 0.5% relative epsilon): the DEFAULT value is
        // preferred even though it appears last, so the pick == default ⇒
        // byte-identical. 100.0 vs 100.2 vs 100.1 are all within eps.
        let dead_heat = vec![
            PerfCandidate {
                value: 64,
                tok_s: Some(100.2),
                tokens_per_joule: None,
            },
            PerfCandidate {
                value: 128,
                tok_s: Some(100.1),
                tokens_per_joule: None,
            },
            PerfCandidate {
                value: 256,
                tok_s: Some(100.0),
                tokens_per_joule: None,
            },
        ];
        assert_eq!(
            pick_perf_knob(&dead_heat, OptimizeFor::Throughput, 256),
            Some(256)
        );

        // Dead heat with NO candidate equal to the default → the CLOSEST to the
        // default wins (128 is nearer 200 than 64 is).
        let no_default = vec![
            PerfCandidate {
                value: 64,
                tok_s: Some(100.2),
                tokens_per_joule: None,
            },
            PerfCandidate {
                value: 128,
                tok_s: Some(100.0),
                tokens_per_joule: None,
            },
        ];
        assert_eq!(
            pick_perf_knob(&no_default, OptimizeFor::Throughput, 200),
            Some(128)
        );
    }

    /// The three P16 numeric winners persist across a TOML round-trip, and old
    /// caches lacking the keys still load (serde default → None).
    #[test]
    fn perf_knob_winners_survive_toml_roundtrip() {
        let mut t = TuningResult::empty("fp".into(), DeviceFingerprint::default());
        t.parallel_matvec_crossover = Some(512);
        t.matvec_min_flops = Some(1 << 20);
        t.ssm_chunk = Some(128);
        let s = toml::to_string(&t).unwrap();
        let back: TuningResult = toml::from_str(&s).unwrap();
        assert_eq!(back.parallel_matvec_crossover, Some(512));
        assert_eq!(back.matvec_min_flops, Some(1 << 20));
        assert_eq!(back.ssm_chunk, Some(128));

        // Old-cache compatibility: serialize a cache WITHOUT the P16 keys (strip
        // them from the TOML) and confirm it still loads, with the three winners
        // defaulting to None (the `#[serde(default)]` contract — no schema bump).
        let empty = TuningResult::empty("fp".into(), DeviceFingerprint::default());
        let full = toml::to_string(&empty).unwrap();
        let stripped: String = full
            .lines()
            .filter(|l| {
                !l.starts_with("parallel_matvec_crossover")
                    && !l.starts_with("matvec_min_flops")
                    && !l.starts_with("ssm_chunk")
            })
            .collect::<Vec<_>>()
            .join("\n");
        let old: TuningResult = toml::from_str(&stripped).unwrap();
        assert_eq!(old.parallel_matvec_crossover, None);
        assert_eq!(old.matvec_min_flops, None);
        assert_eq!(old.ssm_chunk, None);
    }

    #[test]
    fn winner_tree_widths_survives_toml_roundtrip() {
        // The width winner must persist across process restarts like every other
        // cached winner (serde default keeps old caches loadable).
        let fp = DeviceFingerprint {
            pci_id: 0x8086,
            driver_ver: "32.0.101.7076".into(),
            name: "Iris Xe".into(),
            vram_mb: 0,
        };
        let mut r = TuningResult::empty("test-sys".to_string(), fp);
        assert_eq!(r.winner_tree_widths, None);
        r.winner_tree_widths = Some(vec![4, 2]);
        let s = toml::to_string(&r).expect("toml encode");
        let back: TuningResult = toml::from_str(&s).expect("toml decode");
        assert_eq!(back.winner_tree_widths, Some(vec![4, 2]));
    }

    #[test]
    fn set_tuned_q4k_lws_overwrites_existing() {
        // Re-tuning the same shape should replace, not duplicate.
        let fp = DeviceFingerprint {
            pci_id: 0,
            driver_ver: "".into(),
            name: "x".into(),
            vram_mb: 0,
        };
        let mut r = TuningResult::empty("test-sys".to_string(), fp);
        set_tuned_q4k_usm_lws(&mut r, 1536, 1536, 64);
        set_tuned_q4k_usm_lws(&mut r, 1536, 1536, 128);
        assert_eq!(tuned_q4k_usm_lws(&r, 1536, 1536), Some(128));
    }
}
