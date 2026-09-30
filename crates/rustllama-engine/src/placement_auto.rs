//! Automatic `n_gpu_layers` placement.
//!
//! Reads the SYCL device's reported global memory, counts weight
//! bytes per transformer layer (honoring per-tensor CPU-force
//! patterns), reserves a safety margin for activations + KV cache,
//! and picks the largest cutoff such that cumulative GPU residency
//! fits the device's budget.
//!
//! Trigger: the default placement path. Unless `[inference].n_gpu_layers`
//! is set to an explicit override, the engine ignores that field and calls
//! [`auto_n_gpu_layers`] right after model load. This VRAM-fit planner is
//! also the pre-tune fallback under [`crate::multi_gpu::plan_heat_placement`].
//!
//! Fallback: if SYCL isn't healthy (mock build, no Intel GPU,
//! `device_info` query fails), the function returns the
//! conservative `0` (all CPU) and the caller logs a warning. A
//! more aggressive fallback (return `u32::MAX` = all GPU) would
//! be silently wrong on memory-constrained boxes.

use rustllama_models::llama_arch::{HybridFfn, HybridLayer, LlamaWeights};
use rustllama_tensor::Tensor;

/// Inputs that shape the placement decision.
#[derive(Debug, Clone)]
pub struct AutoPlacementOpts {
    /// SYCL device index to query. Engine uses the same index it
    /// passes to `create_stream`; mismatched indices would silently
    /// plan against the wrong device's budget.
    pub device_index: u32,
    /// Patterns from `[inference].placement.overrides` filtered to
    /// `device = "cpu"`. Tensors whose name contains any of these
    /// substrings are excluded from per-layer GPU byte counts.
    /// V1 matching is substring (matches `accel::tensor_forced_to_cpu`).
    pub cpu_force_patterns: Vec<String>,
    /// Bytes reserved upfront for KV cache + activations + sampler
    /// scratch. Default: 1 GiB, which covers a 4K-context Qwen-style
    /// KV slab + per-token activation buffers + sampler USM.
    /// Increase for larger contexts (KV scales with `ctx_size *
    /// n_kv_heads * head_dim * 2 (K+V) * dtype_bytes`).
    pub reserve_bytes: u64,
    /// Multiplicative safety factor applied to the device's
    /// reported `global_mem_size`. Default `0.85` — keeps 15%
    /// headroom for the SYCL runtime + driver allocations that
    /// don't show up in our byte count.
    pub safety_factor: f64,
    /// Whether the CPU is an eligible compute/residency tier. `true`
    /// (default) = GPU-primary + CPU spill (the largest GPU-resident
    /// prefix that fits, remainder on CPU/RAM). `false` = **GPU-only**:
    /// the decision carries a [`AutoPlacementDecision::placement_error`]
    /// when the model doesn't fit GPU memory in full, so the caller
    /// refuses the load instead of silently spilling to the CPU.
    /// From `[inference].cpu_enabled` / `RUSTLLAMA_CPU_ENABLED` /
    /// `serve --no-cpu`.
    pub cpu_enabled: bool,
    /// Whether the GPU is an eligible compute/residency tier. `true`
    /// (default) = normal GPU-primary placement. `false` = **CPU-only**:
    /// no GPU tier is offered, so every layer is placed on the CPU/RAM
    /// tier (`n_gpu_layers = 0`) regardless of any GPU present — the
    /// symmetric mirror of `cpu_enabled == false`. If BOTH `cpu_enabled`
    /// and `gpu_enabled` are `false` the decision carries a
    /// [`AutoPlacementDecision::placement_error`] (no compute tier). From
    /// `[inference].gpu_enabled` / `RUSTLLAMA_GPU_ENABLED` /
    /// `serve --no-gpu`.
    pub gpu_enabled: bool,
    /// Whether weights must live in DEDICATED VRAM only (no host-RAM
    /// residency / mmap). `true` requires a full GPU fit on a device with
    /// separate dedicated VRAM, else a `placement_error`. DEGENERATE on
    /// unified-memory GPUs (no separate dedicated VRAM): treated as a
    /// no-op, [`AutoPlacementDecision::vram_only_noop`] is set so the
    /// caller can warn, and placement proceeds on the normal path. From
    /// `[inference].vram_only` / `RUSTLLAMA_VRAM_ONLY` / `serve --vram-only`.
    pub vram_only: bool,
}

impl Default for AutoPlacementOpts {
    fn default() -> Self {
        Self {
            device_index: 0,
            cpu_force_patterns: Vec::new(),
            reserve_bytes: 1 << 30, // 1 GiB
            safety_factor: 0.85,
            cpu_enabled: true,
            gpu_enabled: true,
            vram_only: false,
        }
    }
}

/// Result of an auto-placement decision. Returned for logging /
/// telemetry so users can see *why* the planner picked a given
/// cutoff.
#[derive(Debug, Clone)]
pub struct AutoPlacementDecision {
    /// Chosen `n_gpu_layers` cutoff. `0` means "all CPU" (returned
    /// when SYCL is unavailable or the device budget is smaller
    /// than `reserve_bytes`).
    pub n_gpu_layers: u32,
    /// Total layer count in the model (= `weights.blocks.len()`
    /// or `moe_blocks.len()`).
    pub total_layers: u32,
    /// Effective device budget after `safety_factor` + `reserve_bytes`.
    pub effective_budget_bytes: u64,
    /// Cumulative weight bytes the chosen cutoff puts on GPU.
    pub gpu_resident_bytes: u64,
    /// Device's reported `global_mem_size`. `0` when the query
    /// failed and we fell back to the conservative all-CPU path.
    pub device_global_mem_bytes: u64,
    /// Human-readable explanation. Logged at engine load.
    pub reason: String,
    /// `Some(msg)` = a **hard** device-tier constraint failed
    /// (`cpu_enabled = false` with no full GPU fit, or `vram_only` with no
    /// dedicated-VRAM fit). The caller must REFUSE the load and surface
    /// `msg` rather than proceeding. `None` = placement is usable.
    pub placement_error: Option<String>,
    /// `true` when `vram_only` was requested but the host has no separate
    /// dedicated VRAM (unified-memory GPU / CPU-only), so the flag was
    /// treated as a no-op. The caller should warn but proceed.
    pub vram_only_noop: bool,
}

/// Plan an `n_gpu_layers` value automatically, then apply the device-tier
/// policy (`cpu_enabled` / `vram_only`). See module docs.
///
/// Under the defaults (`cpu_enabled = true`, `vram_only = false`) this is
/// byte-identical to the historical VRAM-fit planner — the policy layer is
/// a pure no-op, so existing behavior is preserved.
pub fn auto_n_gpu_layers(
    weights: &LlamaWeights,
    opts: &AutoPlacementOpts,
) -> AutoPlacementDecision {
    let mut d = auto_n_gpu_layers_inner(weights, opts);

    // `gpu_enabled == false` ⇒ CPU-only: no GPU tier, so every layer lands
    // on the CPU/RAM tier. Short-circuit the GPU-tier policy below (which
    // assumes a GPU is a target). The symmetric mirror of the GPU-only path.
    if !opts.gpu_enabled {
        d.n_gpu_layers = 0;
        d.gpu_resident_bytes = 0;
        d.reason = format!(
            "gpu_enabled=false (CPU-only): all {} layers placed on CPU/RAM",
            d.total_layers
        );
        // Both tiers disabled leaves nowhere to place weights (config
        // validation catches this, but a CLI flag override can reach here).
        if !opts.cpu_enabled {
            d.placement_error = Some(
                "cpu_enabled=false and gpu_enabled=false: no compute tier available — \
                 enable the CPU or GPU tier"
                    .to_string(),
            );
        }
        return d;
    }

    // `cpu_enabled == false` ⇒ GPU-only: every layer must be GPU-resident.
    // (`n_gpu_layers < total_layers` also covers the no-GPU case, where the
    // inner planner returns 0.)
    if !opts.cpu_enabled && d.n_gpu_layers < d.total_layers {
        d.placement_error = Some(format!(
            "cpu_enabled=false (GPU-only): only {}/{} layers fit GPU memory — free VRAM, \
             lower ctx_size/kv_dtype, or re-enable the CPU tier",
            d.n_gpu_layers, d.total_layers
        ));
    }

    // `vram_only == true` ⇒ weights must live in dedicated VRAM. On a
    // unified-memory GPU there is no separate dedicated VRAM to pin into, so
    // it degenerates to a no-op (+ a warning surfaced via `vram_only_noop`).
    if opts.vram_only {
        if host_has_dedicated_vram(opts.device_index) {
            if d.n_gpu_layers < d.total_layers {
                d.placement_error = Some(format!(
                    "vram_only=true: model does not fit dedicated VRAM ({}/{} layers fit) — \
                     free VRAM or disable vram_only",
                    d.n_gpu_layers, d.total_layers
                ));
            }
        } else {
            d.vram_only_noop = true;
            d.reason.push_str(
                "; vram_only: no separate dedicated VRAM on this host \
                 (unified-memory GPU or CPU-only) — vram_only is a no-op",
            );
        }
    }

    d
}

/// Does this host expose SEPARATE dedicated VRAM (as opposed to a
/// unified-memory / integrated GPU whose "VRAM" is a shared-RAM aperture)?
///
/// - A usable CUDA (NVIDIA discrete) GPU always means real dedicated VRAM.
/// - Otherwise, consult the ACTIVE SYCL device: the engine dispatches on
///   the first non-disabled (prefer-L0) SYCL device, so we query that one's
///   `is_integrated` flag (from SYCL `host_unified_memory`). A discrete
///   Intel Arc reports `is_integrated == false` ⇒ dedicated VRAM present
///   (so `vram_only` is enforced); the integrated Iris Xe reports
///   `is_integrated == true` ⇒ unified memory (so `vram_only` is a no-op).
/// - No queryable GPU of either kind ⇒ no dedicated VRAM.
///
/// `fallback_device_index` is the planner's SYCL device index, used only
/// when the active-device selector yields nothing.
fn host_has_dedicated_vram(fallback_device_index: u32) -> bool {
    if rustllama_models::accel::cuda_active() {
        return true;
    }
    // Apple Metal is UNIFIED MEMORY: the CPU and GPU share one physical
    // pool, so there is NO separate dedicated VRAM to fit weights into —
    // `vram_only` degenerates to a no-op (weights live in the shared pool
    // either way), exactly as for the integrated Iris Xe. Return false.
    // Checked before the SYCL probe below because an Apple host has no SYCL
    // device for the fallback `device_info` query to consult. Inert off
    // Apple Silicon (`mlx_active` == false → this is skipped).
    if rustllama_models::accel::mlx_active() {
        return false;
    }
    let idx = rustllama_models::accel::first_enabled_sycl_device_index()
        .unwrap_or(fallback_device_index);
    match rustllama_kernels_sycl::device_info(idx) {
        Ok(info) => !info.is_integrated,
        // No queryable SYCL device (all disabled, stub build, query fail) ⇒
        // treat as no dedicated VRAM (vram_only degenerates to a no-op).
        Err(_) => false,
    }
}

/// The historical VRAM-fit planner (GPU-primary + CPU spill). Split out so
/// [`auto_n_gpu_layers`] can layer the device-tier policy on top without
/// touching this logic.
fn auto_n_gpu_layers_inner(
    weights: &LlamaWeights,
    opts: &AutoPlacementOpts,
) -> AutoPlacementDecision {
    let total_layers = if let Some(hyb) = weights.hybrid_layers.as_ref() {
        hyb.len() as u32
    } else if let Some(moe) = weights.moe_blocks.as_ref() {
        moe.len() as u32
    } else {
        weights.blocks.len() as u32
    };

    // Query the GPU we'll dispatch on for its VRAM budget. Prefer the SYCL
    // device the engine uses; if there is NO SYCL device (e.g. an NVIDIA-only
    // host / the DGX Spark, where SYCL is a no-op stub), fall back to the CUDA
    // GPU's total memory. Without this fallback the planner returned 0 layers
    // and — since the CUDA matvec dispatch is gated by `n_gpu_layers` — the
    // whole model ran on CPU while the NVIDIA card sat idle. On mock / no-GPU
    // hosts (no SYCL and no CUDA) we still route everything to CPU.
    // Budget for the GPU the engine ACTUALLY dispatches on. SYCL and CUDA are
    // equal first-class GPU backends; the matvec path is CUDA-first when a
    // usable NVIDIA GPU is present (`accel::cuda_active`), then SYCL, then CPU.
    // Mirror that precedence here so a mixed Intel+NVIDIA box sizes for the
    // NVIDIA card CUDA runs on (not the small Intel iGPU), and an NVIDIA-only
    // host (the DGX Spark) uses its GPU instead of collapsing to all-CPU. CPU
    // is the fallback only when there's no usable GPU of either kind. (A tuner
    // that *measures + chooses* the device per system is the next step; today
    // the dispatch precedence is the choice.)
    let gpu_budget: Option<(u64, String)> = {
        let cuda = if rustllama_models::accel::cuda_active() {
            cuda_vram_budget().map(|(b, n)| (b, format!("CUDA GPU ({n})")))
        } else {
            None
        };
        // Apple Metal — the 4th GPU tier. Size against the unified-memory
        // pool (MLX device 0's total). Mutually exclusive with CUDA/SYCL in
        // practice (a host has Metal xor CUDA/SYCL GPUs); inert off Apple
        // (`mlx_active` == false → stays None, chain byte-identical). Placed
        // after CUDA, before SYCL, mirroring the matvec-dispatch precedence.
        let mlx = if rustllama_models::accel::mlx_active() {
            mlx_vram_budget().map(|(b, n)| (b, format!("Metal GPU ({n})")))
        } else {
            None
        };
        cuda.or(mlx).or_else(|| {
            rustllama_kernels_sycl::device_info(opts.device_index)
                .ok()
                .map(|i| (i.vram_bytes, format!("SYCL device {}", opts.device_index)))
        })
    };
    let (raw_budget, device_src): (u64, String) = match gpu_budget {
        Some(b) => b,
        None => {
            return AutoPlacementDecision {
                n_gpu_layers: 0,
                total_layers,
                effective_budget_bytes: 0,
                gpu_resident_bytes: 0,
                device_global_mem_bytes: 0,
                reason: "no usable SYCL or CUDA GPU; all-CPU".to_string(),
                placement_error: None,
                vram_only_noop: false,
            };
        }
    };
    let after_safety =
        ((raw_budget as f64) * opts.safety_factor).floor() as u64;
    let effective_budget = after_safety.saturating_sub(opts.reserve_bytes);

    if effective_budget == 0 {
        return AutoPlacementDecision {
            n_gpu_layers: 0,
            total_layers,
            effective_budget_bytes: 0,
            gpu_resident_bytes: 0,
            device_global_mem_bytes: raw_budget,
            reason: format!(
                "{device_src}: budget {raw_budget} bytes × {:.2} - {} reserve = 0; \
                 all-CPU",
                opts.safety_factor, opts.reserve_bytes
            ),
            placement_error: None,
            vram_only_noop: false,
        };
    }

    // Walk layers in order, accumulating bytes until we'd exceed
    // the budget. The cutoff is the largest k such that
    // sum(per_layer_bytes[0..k]) <= effective_budget.
    let per_layer = per_layer_weight_bytes(weights, &opts.cpu_force_patterns);
    let mut cumulative: u64 = 0;
    let mut cutoff: u32 = 0;
    for (i, bytes) in per_layer.iter().enumerate() {
        let next = cumulative.saturating_add(*bytes);
        if next > effective_budget {
            break;
        }
        cumulative = next;
        cutoff = (i + 1) as u32;
    }

    let reason = if cutoff == total_layers {
        format!(
            "all {total_layers} layers fit in {effective_budget} effective bytes \
             (used {cumulative}, budget after {:.2}× safety + {} reserve)",
            opts.safety_factor, opts.reserve_bytes
        )
    } else if cutoff == 0 {
        format!(
            "even layer 0 ({} bytes) exceeds effective budget {effective_budget}; \
             all-CPU",
            per_layer.first().copied().unwrap_or(0)
        )
    } else {
        format!(
            "{cutoff}/{total_layers} layers fit ({cumulative} bytes used; \
             next layer would push past {effective_budget})"
        )
    };

    AutoPlacementDecision {
        n_gpu_layers: cutoff,
        total_layers,
        effective_budget_bytes: effective_budget,
        gpu_resident_bytes: cumulative,
        device_global_mem_bytes: raw_budget,
        reason: format!("{device_src}: {reason}"),
        placement_error: None,
        vram_only_noop: false,
    }
}

/// The active CUDA GPU's total VRAM (device 0) + name, or `None` when no
/// NVIDIA GPU is present. Returns TOTAL memory (matching SYCL's reported
/// `global_mem_size`); the `safety_factor` + `reserve_bytes` above cover the
/// non-weight allocations (activations, KV, runtime). Free-VRAM accounting for
/// CUDA (subtracting already-reserved memory) is a later refinement.
fn cuda_vram_budget() -> Option<(u64, String)> {
    if rustllama_kernels_cuda::device_count() == 0 {
        return None;
    }
    let info = rustllama_kernels_cuda::device_info(0).ok()?;
    Some((info.total_mem_bytes, info.name))
}

/// The active Apple Metal GPU's total (unified) memory (device 0) + name, or
/// `None` when no Metal GPU is present. Mirror of [`cuda_vram_budget`].
/// UNIFIED MEMORY: this "VRAM" is the shared CPU/GPU pool, so the
/// `safety_factor` + `reserve_bytes` (which already cover activations / KV /
/// runtime) matter even more here than on a discrete card — the same pool
/// also holds the OS and everything else. Inert off Apple Silicon
/// (`device_count()` returns 0 → `None`).
fn mlx_vram_budget() -> Option<(u64, String)> {
    if rustllama_kernels_mlx::device_count() == 0 {
        return None;
    }
    let info = rustllama_kernels_mlx::device_info(0).ok()?;
    Some((info.total_mem_bytes, info.name))
}

/// Per-layer weight bytes the GPU path would have to upload. Skips
/// tensors whose name matches any `cpu_force_patterns` entry (those
/// stay on CPU even when the layer is GPU-resident, so they don't
/// count against the budget).
///
/// The dense norms (`attn_norm`, `ffn_norm`) are `Vec<f32>` not
/// `Tensor`, so they're never USM-uploaded — the GPU dispatch
/// helpers consume the CPU view directly. They contribute zero
/// bytes here.
pub fn per_layer_weight_bytes(
    weights: &LlamaWeights,
    cpu_force_patterns: &[String],
) -> Vec<u64> {
    let count_tensor = |t: &Tensor| -> u64 {
        if cpu_force_patterns
            .iter()
            .any(|p| t.name.contains(p.as_str()))
        {
            return 0;
        }
        t.dtype.byte_size(t.element_count())
    };
    let count_opt = |t: &Option<Tensor>| -> u64 {
        t.as_ref().map(count_tensor).unwrap_or(0)
    };
    // Bytes of a hybrid layer's FFN (routed-expert MoE or dense SwiGLU).
    let ffn_bytes = |ffn: &HybridFfn| -> u64 {
        match ffn {
            HybridFfn::Moe {
                router,
                w_gate_exps,
                w_up_exps,
                w_down_exps,
                w_gate_shared,
                w_up_shared,
                w_down_shared,
                ..
            } => {
                count_tensor(router)
                    + count_tensor(w_gate_exps)
                    + count_tensor(w_up_exps)
                    + count_tensor(w_down_exps)
                    + count_opt(w_gate_shared)
                    + count_opt(w_up_shared)
                    + count_opt(w_down_shared)
            }
            HybridFfn::Dense { w_gate, w_up, w_down } => {
                count_tensor(w_gate) + count_tensor(w_up) + count_tensor(w_down)
            }
        }
    };

    // Hybrid (transformer+SSM) models store per-layer weights in
    // `hybrid_layers`. Sum each layer's Tensor bytes so the VRAM-fit cutoff
    // isn't computed against an empty `blocks` (which yielded n_gpu_layers=0
    // → all-CPU for Qwen3.5 / Ornith).
    if let Some(hyb) = weights.hybrid_layers.as_ref() {
        hyb.iter()
            .map(|layer| match layer {
                HybridLayer::FullAttention(b) => {
                    count_tensor(&b.w_q)
                        + count_tensor(&b.w_k)
                        + count_tensor(&b.w_v)
                        + count_tensor(&b.w_o)
                        + ffn_bytes(&b.ffn)
                }
                HybridLayer::Ssm(b) => {
                    count_tensor(&b.attn_qkv)
                        + count_tensor(&b.attn_gate)
                        + count_tensor(&b.ssm_conv1d)
                        + count_tensor(&b.ssm_alpha)
                        + count_tensor(&b.ssm_beta)
                        + count_tensor(&b.ssm_out)
                        + ffn_bytes(&b.ffn)
                }
            })
            .collect()
    } else if let Some(moe) = weights.moe_blocks.as_ref() {
        moe.iter()
            .map(|b| {
                count_tensor(&b.w_q)
                    + count_tensor(&b.w_k)
                    + count_tensor(&b.w_v)
                    + count_tensor(&b.w_o)
                    + count_tensor(&b.router)
                    + count_tensor(&b.w_gate_exps)
                    + count_tensor(&b.w_up_exps)
                    + count_tensor(&b.w_down_exps)
                    + count_opt(&b.w_gate_shared)
                    + count_opt(&b.w_up_shared)
                    + count_opt(&b.w_down_shared)
            })
            .collect()
    } else {
        weights
            .blocks
            .iter()
            .map(|b| {
                count_tensor(&b.w_q)
                    + count_tensor(&b.w_k)
                    + count_tensor(&b.w_v)
                    + count_tensor(&b.w_o)
                    + count_tensor(&b.w_gate)
                    + count_tensor(&b.w_up)
                    + count_tensor(&b.w_down)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustllama_tensor::{Dtype, Shape};

    fn mk_tensor(name: &str, n_elements: u64, dtype: Dtype) -> Tensor {
        // Minimal tensor with the right name + dtype + element count;
        // storage doesn't matter for byte_size accounting.
        let mut t = Tensor::zeros_cpu(Dtype::F32, vec![1]);
        t.dtype = dtype;
        t.shape = vec![n_elements];
        t.name = name.into();
        t
    }

    #[test]
    fn dense_per_layer_bytes_sums_seven_projection_tensors() {
        // Build a minimal dense LlamaWeights with two layers and
        // verify per_layer_weight_bytes counts the seven projection
        // tensors (Q, K, V, O, gate, up, down).
        let mk_block = |layer: usize| {
            use rustllama_models::llama_arch::LlamaBlockWeights;
            LlamaBlockWeights {
                attn_norm: vec![],
                w_q: mk_tensor(&format!("blk.{layer}.attn_q.weight"), 100, Dtype::F32),
                w_k: mk_tensor(&format!("blk.{layer}.attn_k.weight"), 100, Dtype::F32),
                w_v: mk_tensor(&format!("blk.{layer}.attn_v.weight"), 100, Dtype::F32),
                w_o: mk_tensor(&format!("blk.{layer}.attn_output.weight"), 100, Dtype::F32),
                b_q: None, b_k: None, b_v: None,
                ffn_norm: vec![],
                w_gate: mk_tensor(&format!("blk.{layer}.ffn_gate.weight"), 200, Dtype::F32),
                w_up: mk_tensor(&format!("blk.{layer}.ffn_up.weight"), 200, Dtype::F32),
                w_down: mk_tensor(&format!("blk.{layer}.ffn_down.weight"), 200, Dtype::F32),
                w_qkv_fused: None,
            }
        };
        let weights = LlamaWeights {
            token_embd: mk_tensor("token_embd.weight", 1, Dtype::F32),
            blocks: vec![mk_block(0), mk_block(1)],
            moe_blocks: None,
            hybrid_layers: None,
            output_norm: vec![],
            output: None,
            mtp_heads: None,
            nextn_head: None,
            hadamard: None,
        };
        let bytes = per_layer_weight_bytes(&weights, &[]);
        // F32 = 4 bytes/element. Per layer: 4*(100*4 + 200*3) = 4*1000 = 4000 bytes.
        assert_eq!(bytes, vec![4000, 4000]);
    }

    #[test]
    fn cpu_force_patterns_exclude_matching_tensors() {
        use rustllama_models::llama_arch::LlamaBlockWeights;
        let block = LlamaBlockWeights {
            attn_norm: vec![],
            w_q: mk_tensor("blk.0.attn_q.weight", 100, Dtype::F32),
            w_k: mk_tensor("blk.0.attn_k.weight", 100, Dtype::F32),
            w_v: mk_tensor("blk.0.attn_v.weight", 100, Dtype::F32),
            w_o: mk_tensor("blk.0.attn_output.weight", 100, Dtype::F32),
            b_q: None, b_k: None, b_v: None,
            ffn_norm: vec![],
            w_gate: mk_tensor("blk.0.ffn_gate.weight", 200, Dtype::F32),
            w_up: mk_tensor("blk.0.ffn_up.weight", 200, Dtype::F32),
            w_down: mk_tensor("blk.0.ffn_down.weight", 200, Dtype::F32),
            w_qkv_fused: None,
        };
        let weights = LlamaWeights {
            token_embd: mk_tensor("token_embd.weight", 1, Dtype::F32),
            blocks: vec![block],
            moe_blocks: None,
            hybrid_layers: None,
            output_norm: vec![],
            output: None,
            mtp_heads: None,
            nextn_head: None,
            hadamard: None,
        };
        // Pinning "ffn" to CPU removes the 3 ffn_* tensors (600 elements × 4 B = 2400 B).
        let bytes = per_layer_weight_bytes(&weights, &["ffn".to_string()]);
        // Remaining: 4 attn tensors × 100 × 4 = 1600 bytes.
        assert_eq!(bytes, vec![1600]);
    }

    #[test]
    fn auto_n_gpu_layers_returns_zero_on_unavailable_sycl() {
        // No mock setup beyond what default Tensor::zeros_cpu gives:
        // on this host SYCL is mock → device_info returns Err.
        use rustllama_models::llama_arch::LlamaBlockWeights;
        let block = LlamaBlockWeights {
            attn_norm: vec![],
            w_q: mk_tensor("w_q", 1, Dtype::F32),
            w_k: mk_tensor("w_k", 1, Dtype::F32),
            w_v: mk_tensor("w_v", 1, Dtype::F32),
            w_o: mk_tensor("w_o", 1, Dtype::F32),
            b_q: None, b_k: None, b_v: None,
            ffn_norm: vec![],
            w_gate: mk_tensor("w_gate", 1, Dtype::F32),
            w_up: mk_tensor("w_up", 1, Dtype::F32),
            w_down: mk_tensor("w_down", 1, Dtype::F32),
            w_qkv_fused: None,
        };
        let weights = LlamaWeights {
            token_embd: mk_tensor("token_embd", 1, Dtype::F32),
            blocks: vec![block],
            moe_blocks: None,
            hybrid_layers: None,
            output_norm: vec![],
            output: None,
            mtp_heads: None,
            nextn_head: None,
            hadamard: None,
        };
        let decision = auto_n_gpu_layers(&weights, &AutoPlacementOpts::default());
        // SYCL not available in mock builds → fall through to all-CPU.
        // On a real-SYCL test host this might return a non-zero count;
        // the test asserts the fallback contract for mock builds.
        if decision.device_global_mem_bytes == 0 {
            assert_eq!(decision.n_gpu_layers, 0);
            assert!(decision.reason.contains("no usable SYCL or CUDA GPU"));
        }
    }
}
