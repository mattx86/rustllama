//! Phase 5: tensor-heat-aware multi-GPU / CPU placement planner.
//!
//! This is the placement brain that GENERALIZES + subsumes both the flat
//! `n_gpu_layers` layer cutoff and the MoE experts→CPU class split. It ranks
//! every model weight by HEAT (compute-per-token × activation frequency) and
//! assigns the hottest tensors to the FASTEST measured tier first, filling each
//! tier's budget before moving to the next, coldest → the slowest tier / CPU.
//!
//! # Inputs (measured-perf is PRIMARY, not a tiebreaker)
//! The per-device decode tok/s from the Phase 3 measurement stage
//! (`TuningResult.per_device_perf`) is what RANKS the tiers (which is
//! "fastest") and WEIGHTS their budgets. Heat only sets the ORDER tensors are
//! assigned (hottest first); measured perf + free-VRAM decide WHERE and HOW
//! MUCH each tier gets. A slow iGPU measured slower than the CPU therefore gets
//! LESS than the CPU; a fast dGPU measured faster gets MORE.
//!
//! # Gating (two code paths, no third heuristic path)
//! Autotune is a hard requirement before a model is served, so at planner time
//! measured `per_device_perf` is ALWAYS present. Concretely:
//!   - **measured perf present** ⇒ compute + apply the heat plan (this module).
//!   - **measured perf absent** ⇒ only ever happens INSIDE the tune's own
//!     measurement loads (which must load the model to measure tok/s); those
//!     use the existing VRAM-fit [`crate::placement_auto::auto_n_gpu_layers`].
//!     [`plan_heat_placement`] returns `None` in that case so the caller keeps
//!     the VRAM-fit loader. There is NO speculative perf-prior heuristic.
//!
//! # Tier set from the device-tier flags
//! Target tiers = {ENABLED GPUs, only if `gpu_enabled`} ∪ ({CPU} only if
//! `cpu_enabled`):
//!   - `gpu_enabled == false` ⇒ NO GPU tier is offered: all weights land on
//!     the CPU tier (the symmetric mirror of `cpu_enabled == false`). Both
//!     flags false is rejected by config validation (no compute tier).
//!   - Disabled GPUs (`RUSTLLAMA_DISABLED_GPUS` ∪ `[inference].disabled_gpus`)
//!     are excluded from GPU targets.
//!   - `cpu_enabled == false` ⇒ CPU is NOT a target: all tensors must fit the
//!     enabled GPU(s), else a `placement_error` (caller refuses the load).
//!   - `vram_only == true` (only real on a discrete GPU) ⇒ no host-RAM/CPU
//!     residency: same overflow → `placement_error`. On unified memory it is
//!     the no-op+warn path, handled by the caller.
//!   - `disabled_cpus` shrinks the CPU tier's compute; the Phase 3 CPU tok/s is
//!     measured on the affinity-pinned enabled-core pool, so the measured score
//!     already reflects it — no separate core-count weighting is needed.
//!
//! # Cross-GPU vs single-GPU
//! The per-tensor CPU-vs-GPU split (a single GPU + CPU) is fully expressed
//! through the engine's existing residency gate
//! ([`rustllama_models::accel::tensor_forced_to_cpu`]): a tensor the plan does
//! not assign to a GPU is a CPU-tier tensor. Cross-GPU routing (a tensor to a
//! specific non-active GPU) is the only part that needs Phase 4's per-device
//! caches and only occurs with >1 usable GPU. On the single-iGPU box every
//! GPU-assigned tensor targets the one active device, so the plan reduces to a
//! heat-ranked GPU/CPU split — validatable here.

use std::collections::{HashMap, HashSet};

use rustllama_models::accel::{DeviceTarget, GpuBackend, MultiGpuPlan};
use rustllama_models::llama_arch::{HybridFfn, HybridLayer, LlamaWeights};
use rustllama_tensor::Tensor;

use crate::placement_auto::{AutoPlacementDecision, AutoPlacementOpts};

/// A placement tier the planner can assign tensors to.
#[derive(Debug, Clone)]
pub struct Tier {
    pub kind: TierKind,
    /// Measured decode tok/s (Phase 3). Higher = faster = filled first.
    pub perf_score: f32,
    /// Residency budget in bytes. For a GPU: free VRAM × safety − reserve. For
    /// CPU: the system-RAM budget (the CPU is the spill sink).
    pub budget_bytes: u64,
    /// Total device memory (GPU) or total RAM (CPU) — for the decision log.
    pub total_bytes: u64,
    pub label: String,
}

/// Which tier a tensor is assigned to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TierKind {
    Cpu,
    Gpu(DeviceTarget),
}

/// Heat class of a weight tensor. `Ord`: later variants are HOTTER, so a
/// descending sort puts the hottest first. Ranking = compute-per-token ×
/// activation frequency:
///   Hot (attn Q/K/V/O + dense FFN gate/up/down + shared expert + router —
///   every token) > LmHead (big vocab matvec/token) > Embedding (lookup/token,
///   memory-bound) >> RoutedExpert (sparse top-K — coldest).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Heat {
    RoutedExpert = 0,
    Embedding = 1,
    LmHead = 2,
    Hot = 3,
}

/// Classify a GGUF tensor name into a heat class. Substring rules mirror the
/// engine's existing `is_moe_expert_weight` conventions across Mixtral /
/// Qwen-MoE / DeepSeek-MoE.
pub fn classify_heat(name: &str) -> Heat {
    // Routed experts: sparse (only top-K of N fire per token) → coldest.
    if name.contains("ffn_gate_exps")
        || name.contains("ffn_up_exps")
        || name.contains("ffn_down_exps")
    {
        return Heat::RoutedExpert;
    }
    // Input embedding table: one row lookup per token, memory-bound.
    if name.contains("token_embd") {
        return Heat::Embedding;
    }
    // LM head: the top-level output projection (big vocab matvec every token).
    // Exactly `output.weight` (guard against `blk.N.attn_output.weight`, which
    // contains "output" but is an attention projection = Hot).
    // `.output.weight` (dot-prefixed) matches an LM head under a namespace but
    // NOT `blk.N.attn_output.weight` (which ends with `_output.weight`).
    if name == "output.weight" || name.ends_with(".output.weight") {
        return Heat::LmHead;
    }
    // Everything else that reaches the placer — attn Q/K/V/O, dense FFN
    // gate/up/down, the router (`ffn_gate_inp`), and shared-expert FFN
    // (`ffn_*_shexp`) — fires every token: hottest.
    Heat::Hot
}

/// Is this an attention-projection tensor? Used to pin the KV cache to the GPU
/// that runs attention.
fn is_attn_tensor(name: &str) -> bool {
    name.contains("attn_q")
        || name.contains("attn_k")
        || name.contains("attn_v")
        || name.contains("attn_output")
        || name.contains("attn_qkv")
}

/// One placeable weight tensor.
#[derive(Debug, Clone)]
struct TensorInfo {
    name: String,
    bytes: u64,
    heat: Heat,
    is_attn: bool,
}

/// Collect every GPU-placeable weight tensor (name + byte size + heat) from the
/// model. Norms (`Vec<f32>`, never USM-uploaded) are skipped, matching
/// [`crate::placement_auto::per_layer_weight_bytes`].
fn collect_tensors(weights: &LlamaWeights) -> Vec<TensorInfo> {
    // Plain fn (not a closure) so multiple call sites don't fight over a `&mut
    // out` borrow.
    fn info_of(t: &Tensor) -> TensorInfo {
        TensorInfo {
            name: t.name.clone(),
            bytes: t.dtype.byte_size(t.element_count()),
            heat: classify_heat(&t.name),
            is_attn: is_attn_tensor(&t.name),
        }
    }
    // Emit every placeable Tensor of a hybrid layer's FFN (routed-expert MoE
    // or dense SwiGLU). `classify_heat` sorts them (experts → RoutedExpert;
    // shared / router / dense gate-up-down → Hot). Sibling of `info_of`.
    fn push_hybrid_ffn(out: &mut Vec<TensorInfo>, ffn: &HybridFfn) {
        match ffn {
            HybridFfn::Moe {
                router,
                w_gate_exps,
                w_up_exps,
                w_down_exps,
                w_gate_shared,
                w_up_shared,
                w_down_shared,
                shared_router,
                ..
            } => {
                out.push(info_of(router));
                out.push(info_of(w_gate_exps));
                out.push(info_of(w_up_exps));
                out.push(info_of(w_down_exps));
                for t in [w_gate_shared, w_up_shared, w_down_shared, shared_router]
                    .into_iter()
                    .flatten()
                {
                    out.push(info_of(t));
                }
            }
            HybridFfn::Dense { w_gate, w_up, w_down } => {
                out.push(info_of(w_gate));
                out.push(info_of(w_up));
                out.push(info_of(w_down));
            }
        }
    }
    let mut out: Vec<TensorInfo> = Vec::new();
    out.push(info_of(&weights.token_embd));
    // Hybrid (transformer+SSM) models store layers in `hybrid_layers`, not
    // `blocks`/`moe_blocks`. Enumerate every layer's placeable tensors so the
    // heat plan covers them — otherwise they're absent from the plan and the
    // dispatch gate strands every hybrid layer on CPU (the Qwen3.5 / Ornith
    // all-CPU regression).
    if let Some(hyb) = weights.hybrid_layers.as_ref() {
        for layer in hyb {
            match layer {
                HybridLayer::FullAttention(b) => {
                    out.push(info_of(&b.w_q));
                    out.push(info_of(&b.w_k));
                    out.push(info_of(&b.w_v));
                    out.push(info_of(&b.w_o));
                    push_hybrid_ffn(&mut out, &b.ffn);
                }
                HybridLayer::Ssm(b) => {
                    // Linear-attention (Gated-DeltaNet) projections + gate +
                    // short conv + B/C/out — all fire every token = Hot.
                    out.push(info_of(&b.attn_qkv));
                    out.push(info_of(&b.attn_gate));
                    out.push(info_of(&b.ssm_conv1d));
                    out.push(info_of(&b.ssm_alpha));
                    out.push(info_of(&b.ssm_beta));
                    out.push(info_of(&b.ssm_out));
                    push_hybrid_ffn(&mut out, &b.ffn);
                }
            }
        }
    } else if let Some(moe) = weights.moe_blocks.as_ref() {
        for b in moe {
            out.push(info_of(&b.w_q));
            out.push(info_of(&b.w_k));
            out.push(info_of(&b.w_v));
            out.push(info_of(&b.w_o));
            out.push(info_of(&b.router));
            out.push(info_of(&b.w_gate_exps));
            out.push(info_of(&b.w_up_exps));
            out.push(info_of(&b.w_down_exps));
            if let Some(t) = b.w_gate_shared.as_ref() {
                out.push(info_of(t));
            }
            if let Some(t) = b.w_up_shared.as_ref() {
                out.push(info_of(t));
            }
            if let Some(t) = b.w_down_shared.as_ref() {
                out.push(info_of(t));
            }
        }
    } else {
        for b in &weights.blocks {
            out.push(info_of(&b.w_q));
            out.push(info_of(&b.w_k));
            out.push(info_of(&b.w_v));
            out.push(info_of(&b.w_o));
            out.push(info_of(&b.w_gate));
            out.push(info_of(&b.w_up));
            out.push(info_of(&b.w_down));
        }
    }
    if let Some(t) = weights.output.as_ref() {
        out.push(info_of(t));
    }
    out
}

/// The output of the pure planner.
#[derive(Debug, Clone)]
pub struct HeatPlan {
    /// tensor name → assigned tier. GPU-assigned entries become the
    /// [`MultiGpuPlan`] table; CPU entries are omitted from it (the dispatch
    /// gate treats "absent" as CPU).
    pub assignments: HashMap<String, TierKind>,
    /// GPU that runs attention (holds the KV cache), if attention is on a GPU.
    pub attn_device: Option<DeviceTarget>,
    /// Set when a hard device-tier constraint failed (`cpu_enabled == false` /
    /// enforced `vram_only`) and tensors overflow the enabled GPU(s). The
    /// caller REFUSES the load and surfaces this instead of applying the plan.
    pub placement_error: Option<String>,
    /// Total weight bytes assigned to GPU tiers.
    pub gpu_resident_bytes: u64,
    /// Total weight bytes assigned to the CPU tier.
    pub cpu_resident_bytes: u64,
    /// Human summary for the load log.
    pub reason: String,
    /// Number of distinct GPU devices the plan routes to. `> 1` ⇒ cross-GPU
    /// routing is in play (needs Phase 4's per-device caches).
    pub distinct_gpu_count: usize,
}

impl HeatPlan {
    /// Lower to the runtime [`MultiGpuPlan`] the dispatch hot path consults.
    /// Only GPU-assigned tensors are entered; CPU tensors are absent (⇒ the
    /// dispatch gate routes them to CPU).
    pub fn to_multi_gpu_plan(&self) -> MultiGpuPlan {
        let mut by_name: HashMap<String, DeviceTarget> = HashMap::new();
        for (name, tier) in &self.assignments {
            if let TierKind::Gpu(t) = tier {
                by_name.insert(name.clone(), *t);
            }
        }
        MultiGpuPlan::new(by_name, self.attn_device)
    }
}

/// PURE heat-ranked first-fit placement over pre-built, pre-measured tiers.
///
/// Contract:
///   - Tensors are sorted hottest-first (then largest-first within a class).
///   - Tiers are ranked by MEASURED `perf_score` (fastest first).
///   - Each tensor lands on the fastest tier that still has budget; when the
///     fastest fills, the next-fastest takes over, coldest → slowest / CPU.
///   - `cpu_allowed == false` ⇒ no CPU tier is present in `tiers`; a tensor
///     that fits no GPU sets `placement_error`.
///
/// `tiers` must be non-empty. Returns a [`HeatPlan`]; check `placement_error`
/// before applying.
pub fn plan_from_tiers(
    weights: &LlamaWeights,
    tiers: &[Tier],
    cpu_allowed: bool,
) -> HeatPlan {
    let mut tensors = collect_tensors(weights);
    // Hottest first; within a heat class, largest first (pack big hot tensors
    // onto the fast tier before small ones).
    tensors.sort_by(|a, b| {
        b.heat
            .cmp(&a.heat)
            .then_with(|| b.bytes.cmp(&a.bytes))
            .then_with(|| a.name.cmp(&b.name))
    });

    // Tier indices ranked by measured perf, fastest first. Ties broken by
    // larger budget (prefer the roomier tier).
    let mut order: Vec<usize> = (0..tiers.len()).collect();
    order.sort_by(|&i, &j| {
        tiers[j]
            .perf_score
            .partial_cmp(&tiers[i].perf_score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| tiers[j].budget_bytes.cmp(&tiers[i].budget_bytes))
    });

    let mut used: Vec<u64> = vec![0; tiers.len()];
    let mut assignments: HashMap<String, TierKind> = HashMap::new();
    let mut attn_device: Option<DeviceTarget> = None;
    let mut gpu_resident: u64 = 0;
    let mut cpu_resident: u64 = 0;
    let mut overflow: Vec<String> = Vec::new();

    for t in &tensors {
        let mut placed: Option<usize> = None;
        for &ti in &order {
            if used[ti].saturating_add(t.bytes) <= tiers[ti].budget_bytes {
                placed = Some(ti);
                break;
            }
        }
        match placed {
            Some(ti) => {
                used[ti] += t.bytes;
                assignments.insert(t.name.clone(), tiers[ti].kind);
                match tiers[ti].kind {
                    TierKind::Gpu(dt) => {
                        gpu_resident += t.bytes;
                        if t.is_attn && attn_device.is_none() {
                            attn_device = Some(dt);
                        }
                    }
                    TierKind::Cpu => cpu_resident += t.bytes,
                }
            }
            None => {
                // No tier (GPU, and CPU when disallowed) had room.
                if cpu_allowed {
                    // A CPU tier exists but even it is full — the CPU budget is
                    // sized to system RAM, so this means the model exceeds RAM.
                    overflow.push(t.name.clone());
                    // Best-effort: still record it as CPU so a name lookup is
                    // deterministic; the caller surfaces the overflow.
                    assignments.insert(t.name.clone(), TierKind::Cpu);
                    cpu_resident += t.bytes;
                } else {
                    overflow.push(t.name.clone());
                }
            }
        }
    }

    let distinct_gpu_count = {
        let mut set: HashSet<DeviceTarget> = HashSet::new();
        for v in assignments.values() {
            if let TierKind::Gpu(dt) = v {
                set.insert(*dt);
            }
        }
        set.len()
    };

    let placement_error = if !overflow.is_empty() {
        if cpu_allowed {
            Some(format!(
                "{} weight tensor(s) exceed total RAM+VRAM budget (e.g. {}); \
                 free memory or lower ctx_size/kv_dtype",
                overflow.len(),
                overflow.first().cloned().unwrap_or_default()
            ))
        } else {
            Some(format!(
                "cpu_enabled=false / vram_only: {} weight tensor(s) do not fit the \
                 enabled GPU(s) (e.g. {}) and the CPU tier is disabled — free VRAM, \
                 lower ctx_size/kv_dtype, or re-enable the CPU tier",
                overflow.len(),
                overflow.first().cloned().unwrap_or_default()
            ))
        }
    } else {
        None
    };

    let reason = format!(
        "heat placement: {} GPU-resident bytes across {} device(s), {} CPU-resident bytes \
         over {} tier(s) ranked by measured tok/s",
        gpu_resident,
        distinct_gpu_count,
        cpu_resident,
        tiers.len()
    );

    HeatPlan {
        assignments,
        attn_device,
        placement_error,
        gpu_resident_bytes: gpu_resident,
        cpu_resident_bytes: cpu_resident,
        reason,
        distinct_gpu_count,
    }
}

/// A usable GPU discovered on the host, with its measured perf + budget.
#[derive(Debug, Clone)]
struct GpuCatalogEntry {
    target: DeviceTarget,
    perf_score: f32,
    free_vram_bytes: u64,
    total_vram_bytes: u64,
    label: String,
}

/// Build the measured tier set for the planner, or `None` to DECLINE heat
/// placement (⇒ the caller keeps the VRAM-fit loader). Declines when:
///   - no usable GPU has a measured perf entry (the tune-measurement path,
///     before Phase 3 has run), or
///   - a required tier (an enabled GPU we kept, or the CPU tier when allowed)
///     has no measured perf yet.
///
/// Impure: reads `per_device_perf`, live free-VRAM, total RAM, and the
/// disabled-GPU set.
fn build_tiers(
    per_device_perf: &HashMap<String, f32>,
    opts: &AutoPlacementOpts,
    cpu_allowed: bool,
) -> Option<Vec<Tier>> {
    if per_device_perf.is_empty() {
        return None;
    }
    let disabled = disabled_gpu_set();
    let safety = opts.safety_factor;
    let reserve = opts.reserve_bytes;

    let mut catalog: Vec<GpuCatalogEntry> = Vec::new();

    // --- NVIDIA / CUDA devices ---
    // `gpu_enabled == false` ⇒ CPU-only: offer NO GPU tiers (skip discovery
    // of both backends), so the planner falls through to the CPU tier below.
    let n_cuda = if opts.gpu_enabled {
        rustllama_kernels_cuda::device_count()
    } else {
        0
    };
    let sycl_physical = distinct_sycl_gpu_count();
    for i in 0..n_cuda {
        // Unified index = (# physical SYCL GPUs) + CUDA index.
        let unified = sycl_physical.saturating_add(i);
        if disabled.contains(&unified) {
            continue;
        }
        let Ok(info) = rustllama_kernels_cuda::device_info(i) else {
            continue;
        };
        let slug = rustllama_tuner::cuda_gpu_key(&info);
        // Measured perf is mandatory — a GPU with no entry is excluded.
        let Some(&perf) = per_device_perf.get(&slug) else {
            continue;
        };
        // Free VRAM via the driver (dlopen'd NVML/CUDA); fall back to total.
        let free = rustllama_runtime::gpu_detect::nvidia_free_vram_bytes(i)
            .unwrap_or(info.total_mem_bytes);
        catalog.push(GpuCatalogEntry {
            target: DeviceTarget { backend: GpuBackend::Cuda, device_index: i },
            perf_score: perf,
            free_vram_bytes: free,
            total_vram_bytes: info.total_mem_bytes,
            label: format!("CUDA {i} ({})", info.name),
        });
    }

    // --- Intel / SYCL active device ---
    // Only the device the engine dispatches to is a usable SYCL target today
    // (the per-thread USM stream binds one device). Its unified index is the
    // first non-disabled physical SYCL GPU, already resolved by
    // `first_enabled_sycl_device_index`.
    if let Some(idx) = opts
        .gpu_enabled
        .then(|| rustllama_models::accel::first_enabled_sycl_device_index())
        .flatten()
    {
        if let Ok(info) = rustllama_kernels_sycl::device_info(idx) {
            let slug = rustllama_tuner::sycl_gpu_key(&info);
            if let Some(&perf) = per_device_perf.get(&slug) {
                // SYCL free VRAM: use total × safety (Sysman free-VRAM
                // refinement is a follow-up; on the unified-memory iGPU the
                // "VRAM" is a shared-RAM aperture anyway).
                let _ = &slug; // consumed above by the perf lookup
                catalog.push(GpuCatalogEntry {
                    target: DeviceTarget { backend: GpuBackend::Sycl, device_index: idx },
                    perf_score: perf,
                    free_vram_bytes: info.vram_bytes,
                    total_vram_bytes: info.vram_bytes,
                    label: format!("SYCL {idx} ({})", info.name),
                });
            }
        }
    }

    if catalog.is_empty() && opts.gpu_enabled {
        // GPU tier requested but no GPU has a measured perf entry → decline
        // (the tune-measurement path). When `gpu_enabled == false` an empty
        // catalog is EXPECTED — fall through to the CPU-only tier set.
        return None;
    }

    let mut tiers: Vec<Tier> = Vec::new();
    for g in &catalog {
        let budget = ((g.free_vram_bytes as f64) * safety).floor() as u64;
        let budget = budget.saturating_sub(reserve);
        tiers.push(Tier {
            kind: TierKind::Gpu(g.target),
            perf_score: g.perf_score,
            budget_bytes: budget,
            total_bytes: g.total_vram_bytes,
            label: g.label.clone(),
        });
    }

    if cpu_allowed {
        let cpu_slug = rustllama_tuner::cpu_perf_key();
        // CPU perf is mandatory when the CPU is a target.
        let cpu_perf = per_device_perf.get(&cpu_slug).copied()?;
        let ram = rustllama_runtime::memory_info().total_bytes;
        // CPU is the spill sink: give it the RAM budget minus the same reserve.
        let cpu_budget = ((ram as f64) * safety).floor() as u64;
        let cpu_budget = cpu_budget.saturating_sub(reserve);
        tiers.push(Tier {
            kind: TierKind::Cpu,
            perf_score: cpu_perf,
            budget_bytes: cpu_budget,
            total_bytes: ram,
            label: "CPU (enabled cores)".to_string(),
        });
    }

    // No tier at all (e.g. gpu_enabled=false AND cpu_allowed=false — a
    // degenerate both-disabled combo config validation normally rejects):
    // decline so the caller falls back to the VRAM-fit planner, which
    // surfaces the "no compute tier" placement_error.
    if tiers.is_empty() {
        return None;
    }

    Some(tiers)
}

/// `RUSTLLAMA_DISABLED_GPUS` as a set of unified GPU indices. The CLI promotes
/// `[inference].disabled_gpus` into this env, so it is the single source of
/// truth for both.
fn disabled_gpu_set() -> HashSet<u32> {
    rustllama_runtime::disabled_gpu_indices()
}

/// Count of distinct physical SYCL GPUs (deduped by name), used to map a CUDA
/// device's index into the unified enumeration for the disable-list check.
fn distinct_sycl_gpu_count() -> u32 {
    let n = rustllama_kernels_sycl::device_count().unwrap_or(0);
    let mut names: HashSet<String> = HashSet::new();
    for i in 0..n {
        if let Ok(info) = rustllama_kernels_sycl::device_info(i) {
            names.insert(info.name);
        }
    }
    names.len() as u32
}

/// The measured-perf heat placement, computed end to end (impure).
///
/// Returns `Some(outcome)` when a heat plan was produced (check
/// `outcome.decision.placement_error` before applying), or `None` to DECLINE —
/// meaning measured perf isn't available yet, so the caller must keep the
/// existing VRAM-fit [`crate::placement_auto::auto_n_gpu_layers`] loader (the
/// tune-measurement path).
pub fn plan_heat_placement(
    weights: &LlamaWeights,
    per_device_perf: &HashMap<String, f32>,
    opts: &AutoPlacementOpts,
) -> Option<HeatPlacementOutcome> {
    let total_layers = if let Some(hyb) = weights.hybrid_layers.as_ref() {
        hyb.len() as u32
    } else if let Some(moe) = weights.moe_blocks.as_ref() {
        moe.len() as u32
    } else {
        weights.blocks.len() as u32
    };

    // CPU is a target only when it is enabled AND vram_only is not being
    // enforced on real dedicated VRAM. Unified-memory vram_only is a no-op, so
    // the CPU tier stays (the caller sets `vram_only_noop`).
    let vram_only_enforced = opts.vram_only && host_has_dedicated_vram(opts.device_index);
    let cpu_allowed = opts.cpu_enabled && !vram_only_enforced;

    let tiers = build_tiers(per_device_perf, opts, cpu_allowed)?;

    let plan = plan_from_tiers(weights, &tiers, cpu_allowed);

    let effective_budget: u64 = tiers
        .iter()
        .filter(|t| matches!(t.kind, TierKind::Gpu(_)))
        .map(|t| t.budget_bytes)
        .sum();
    let device_global: u64 = tiers
        .iter()
        .filter(|t| matches!(t.kind, TierKind::Gpu(_)))
        .map(|t| t.total_bytes)
        .sum();

    // The layer cutoff also gates the LAYER-level non-matvec ops (rmsnorm,
    // rope, silu, the attention core) — which the per-tensor plan does NOT
    // cover. Keep the two coherent via the attention placement:
    //   - attention on a GPU  ⇒ `n_gpu_layers = total` (attention core + the
    //     GPU-assigned matvecs run on GPU; the plan splits the matvecs).
    //   - attention on CPU (the "CPU measured faster than the iGPU" outcome,
    //     where the plan routes everything to CPU) ⇒ `n_gpu_layers = 0`, so the
    //     non-matvec ops land on CPU too. That case is byte-identical to
    //     today's all-CPU path (and defends in depth: even if the plan were
    //     ignored, the cutoff alone yields all-CPU).
    let applied_n_gpu_layers = if plan.attn_device.is_some() {
        total_layers
    } else {
        0
    };

    let decision = AutoPlacementDecision {
        n_gpu_layers: applied_n_gpu_layers,
        total_layers,
        effective_budget_bytes: effective_budget,
        gpu_resident_bytes: plan.gpu_resident_bytes,
        device_global_mem_bytes: device_global,
        reason: plan.reason.clone(),
        placement_error: plan.placement_error.clone(),
        vram_only_noop: opts.vram_only && !host_has_dedicated_vram(opts.device_index),
    };

    Some(HeatPlacementOutcome {
        decision,
        plan,
        applied_n_gpu_layers,
    })
}

/// Everything the engine needs to APPLY (or refuse) a heat placement.
#[derive(Debug, Clone)]
pub struct HeatPlacementOutcome {
    /// The placement decision (for logging + the device-tier refuse path).
    pub decision: AutoPlacementDecision,
    /// The heat plan. Lower to a runtime plan via [`HeatPlan::to_multi_gpu_plan`].
    pub plan: HeatPlan,
    /// The `n_gpu_layers` value the engine should set (always `total_layers` —
    /// the plan is per-tensor authoritative).
    pub applied_n_gpu_layers: u32,
}

/// Re-export of the placement_auto dedicated-VRAM probe so the tier logic and
/// the caller agree on what `vram_only` means.
fn host_has_dedicated_vram(fallback_device_index: u32) -> bool {
    if rustllama_models::accel::cuda_active() {
        return true;
    }
    let idx = rustllama_models::accel::first_enabled_sycl_device_index()
        .unwrap_or(fallback_device_index);
    match rustllama_kernels_sycl::device_info(idx) {
        Ok(info) => !info.is_integrated,
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_tier(perf: f32, budget: u64) -> Tier {
        Tier {
            kind: TierKind::Cpu,
            perf_score: perf,
            budget_bytes: budget,
            total_bytes: budget,
            label: "cpu".into(),
        }
    }
    fn gpu_tier(idx: u32, perf: f32, budget: u64) -> Tier {
        Tier {
            kind: TierKind::Gpu(DeviceTarget { backend: GpuBackend::Cuda, device_index: idx }),
            perf_score: perf,
            budget_bytes: budget,
            total_bytes: budget,
            label: format!("gpu{idx}"),
        }
    }

    #[test]
    fn heat_ordering_is_hot_gt_lmhead_gt_embed_gt_expert() {
        assert!(Heat::Hot > Heat::LmHead);
        assert!(Heat::LmHead > Heat::Embedding);
        assert!(Heat::Embedding > Heat::RoutedExpert);
    }

    #[test]
    fn classify_distinguishes_lm_head_from_attn_output() {
        assert_eq!(classify_heat("output.weight"), Heat::LmHead);
        assert_eq!(classify_heat("blk.3.attn_output.weight"), Heat::Hot);
        assert_eq!(classify_heat("token_embd.weight"), Heat::Embedding);
        assert_eq!(classify_heat("blk.7.ffn_gate_exps.weight"), Heat::RoutedExpert);
        assert_eq!(classify_heat("blk.7.ffn_gate.weight"), Heat::Hot);
        assert_eq!(classify_heat("blk.2.ffn_gate_inp.weight"), Heat::Hot);
    }

    /// Build a minimal 1-layer dense LlamaWeights with the given per-tensor
    /// element counts, so `plan_from_tiers` can be exercised without a GPU.
    fn dense_weights() -> LlamaWeights {
        use rustllama_models::llama_arch::LlamaBlockWeights;
        use rustllama_tensor::{Dtype, Tensor};
        let mk = |name: &str, n: u64| -> Tensor {
            let mut t = Tensor::zeros_cpu(Dtype::F32, vec![1]);
            t.dtype = Dtype::F32;
            t.shape = vec![n];
            t.name = name.into();
            t
        };
        LlamaWeights {
            token_embd: mk("token_embd.weight", 100),
            blocks: vec![LlamaBlockWeights {
                attn_norm: vec![],
                w_q: mk("blk.0.attn_q.weight", 10),
                w_k: mk("blk.0.attn_k.weight", 10),
                w_v: mk("blk.0.attn_v.weight", 10),
                w_o: mk("blk.0.attn_output.weight", 10),
                b_q: None,
                b_k: None,
                b_v: None,
                ffn_norm: vec![],
                w_gate: mk("blk.0.ffn_gate.weight", 20),
                w_up: mk("blk.0.ffn_up.weight", 20),
                w_down: mk("blk.0.ffn_down.weight", 20),
                w_qkv_fused: None,
            }],
            moe_blocks: None,
            hybrid_layers: None,
            output_norm: vec![],
            output: Some(mk("output.weight", 100)),
            mtp_heads: None,
            nextn_head: None,
            hadamard: None,
        }
    }

    #[test]
    fn faster_cpu_takes_hot_tensors_over_slow_igpu() {
        // CPU measured faster than the iGPU, both roomy: the hottest tensors
        // land on the CPU (the fastest tier). Demonstrates measured-perf, not
        // heat, decides WHERE.
        let w = dense_weights();
        let tiers = vec![
            gpu_tier(0, 10.0, 1 << 30), // slow iGPU
            cpu_tier(50.0, 1 << 30),    // fast CPU
        ];
        let plan = plan_from_tiers(&w, &tiers, true);
        // Attn (Hot) must be on the CPU (fastest), so attn_device is None.
        assert_eq!(plan.assignments.get("blk.0.attn_q.weight"), Some(&TierKind::Cpu));
        assert!(plan.attn_device.is_none());
        assert!(plan.placement_error.is_none());
    }

    #[test]
    fn fast_gpu_gets_hot_tensors_first_then_cpu_spill() {
        // Fast GPU with a TINY budget: only the single hottest/biggest tensor
        // fits; the rest spill to the CPU.
        let w = dense_weights();
        // F32 bytes: attn tensors = 40 each, ffn = 80 each, embd/head = 400.
        // Budget 400 fits exactly one 400-byte tensor (embd or head).
        let tiers = vec![
            gpu_tier(0, 100.0, 400),
            cpu_tier(10.0, 1 << 30),
        ];
        let plan = plan_from_tiers(&w, &tiers, true);
        let gpu_bytes = plan.gpu_resident_bytes;
        assert!(gpu_bytes > 0 && gpu_bytes <= 400, "gpu got {gpu_bytes}");
        assert!(plan.cpu_resident_bytes > 0);
        assert!(plan.placement_error.is_none());
    }

    #[test]
    fn cpu_disallowed_overflow_sets_placement_error() {
        // Small GPU, no CPU tier (cpu_enabled=false): tensors overflow → error.
        let w = dense_weights();
        let tiers = vec![gpu_tier(0, 100.0, 100)]; // too small for all weights
        let plan = plan_from_tiers(&w, &tiers, false);
        assert!(plan.placement_error.is_some(), "expected overflow error");
    }

    #[test]
    fn everything_fits_one_big_gpu_all_on_gpu() {
        let w = dense_weights();
        let tiers = vec![gpu_tier(0, 100.0, 1 << 30), cpu_tier(10.0, 1 << 30)];
        let plan = plan_from_tiers(&w, &tiers, true);
        assert_eq!(plan.cpu_resident_bytes, 0, "all weights should fit the fast GPU");
        assert!(plan.attn_device.is_some(), "attention is on the GPU");
        assert_eq!(plan.distinct_gpu_count, 1);
    }

    #[test]
    fn two_gpus_hottest_on_fastest() {
        // Two GPUs; the faster one (idx 1) fills first with the hottest.
        let w = dense_weights();
        let tiers = vec![
            gpu_tier(0, 20.0, 400),  // slower, small
            gpu_tier(1, 90.0, 400),  // faster, small
            cpu_tier(5.0, 1 << 30),
        ];
        let plan = plan_from_tiers(&w, &tiers, true);
        // The single biggest Hot tensor is an FFN (80B) — but embd/head (400B,
        // colder) are bigger. Hottest-first means attn+ffn (Hot) go first onto
        // gpu1 until its 400B budget fills.
        let attn_q = plan.assignments.get("blk.0.attn_q.weight").copied();
        assert_eq!(
            attn_q,
            Some(TierKind::Gpu(DeviceTarget { backend: GpuBackend::Cuda, device_index: 1 })),
            "hottest attn tensor should be on the faster GPU (idx 1)"
        );
        assert_eq!(plan.attn_device.unwrap().device_index, 1);
    }
}
