//! T3: typed, process-local load options that replace the ~41 config→env
//! `std::env::set_var("RUSTLLAMA_…", …)` bridges the CLI/engine/server used
//! to shuttle `[inference]` knobs into the models/engine forward path.
//!
//! The old shape was: the CLI promoted each config value to a process-global
//! env var (only when the user hadn't already set it — "env wins"), and the
//! models/engine read it back with `std::env::var`. That write is racy across
//! concurrent loads and becomes `unsafe` in edition 2024.
//!
//! The new shape:
//!   1. The CLI builds a [`LoadOptions`] ONCE from the resolved config, with
//!      the env-overrides-win precedence applied THERE (see
//!      [`LoadOptions::from_input`]) and documented in one place.
//!   2. [`LoadOptions::apply`] pushes the values into the `accel`-level
//!      settable cells (the same mechanism as `accel::set_flash_attention` /
//!      `set_n_gpu_layers`) — a plain atomic store, no `set_var`.
//!   3. The models/engine read `accel::zerocopy_weights()` etc. Those cells
//!      still initialize from the legacy `RUSTLLAMA_*` env var on first read,
//!      so a bare `LlamaModel::load`, a unit test, or `doctor` (paths that
//!      never build a `LoadOptions`) keep honoring a user-set env var.
//!
//! Precedence is therefore preserved exactly: a user-set env var wins (it is
//! what `from_input` resolves to, and what the cell would read anyway), else
//! the typed config value applies.
//!
//! The autotune `tune --all` subprocess (spawned by the server on first load)
//! does NOT share this process's `accel` cells, so it must still receive the
//! knobs it needs through its environment — [`LoadOptions::tune_child_env`]
//! produces exactly those pairs, set on the child `Command` at the single
//! spawn site (NOT a process-global `set_var`).

use rustllama_models::accel;

/// Plain config scalars the CLI/server read off `[inference]` (+ the tuner
/// cache for the MoE-placement / SSM knobs). Kept dependency-free (no
/// `rustllama_config` type) so this crate needn't depend on the config crate;
/// the caller fills it from its own `InferenceConfig`.
#[derive(Clone, Debug, Default)]
pub struct LoadOptionsInput {
    pub zerocopy_weights: bool,
    /// Raw `lock_ram_mb` directive (`"auto"` | MB | `""`/`"0"` = off).
    pub lock_ram_mb: String,
    pub prefill_batched: bool,
    pub prefill_batched_hybrid: bool,
    /// Raw `memory_budget` directive (`"auto"` enables the planner).
    pub memory_budget: String,
    pub ssm_prefill_chunked: bool,
    pub kv_persist_mb: u64,
    pub moe_expert_cache_mb: u64,
    /// Already resolved from the tuner cache by the caller.
    pub moe_experts_cpu: bool,
    /// Already resolved from the tuner cache by the caller (permille).
    pub moe_gpu_split_permille: u32,
    pub hybrid_kv_any: bool,
}

/// Typed, env-wins-resolved snapshot of the config-mirroring feature/sizing
/// knobs. Built once per load; applied via [`LoadOptions::apply`].
#[derive(Clone, Debug, Default)]
pub struct LoadOptions {
    pub zerocopy_weights: bool,
    /// `None` = unset (fall through to the legacy env / off).
    pub lock_ram_mb: Option<String>,
    pub prefill_batched: bool,
    pub prefill_batched_hybrid: bool,
    /// `None` = unset.
    pub memory_budget: Option<String>,
    pub ssm_prefill_chunked: bool,
    pub kv_persist_mb: u64,
    pub moe_expert_cache_mb: u64,
    pub moe_experts_cpu: bool,
    pub moe_gpu_split_permille: u32,
    pub hybrid_kv_any: bool,
}

/// `true` when `RUSTLLAMA_<name>` is set to a truthy value (non-empty, not
/// `"0"`, not `"false"`) — matches the legacy accessors for the
/// `PREFILL_BATCHED*` knobs.
fn env_truthy(name: &str) -> Option<bool> {
    std::env::var(name)
        .ok()
        .map(|v| !v.is_empty() && v != "0" && v.to_ascii_lowercase() != "false")
}

impl LoadOptions {
    /// Resolve the env-overrides-win precedence ONCE, here. For each knob: if
    /// the legacy `RUSTLLAMA_*` env var is set it wins (so a power user or the
    /// tuner can sweep a knob without rewriting config); otherwise the typed
    /// config value (already folded with any tuner-cache winner by the
    /// caller) applies.
    pub fn from_input(input: &LoadOptionsInput) -> Self {
        // `ZEROCOPY_WEIGHTS` truthiness historically = `"1"` / `"true"`.
        let zerocopy_weights = match std::env::var("RUSTLLAMA_ZEROCOPY_WEIGHTS") {
            Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
            Err(_) => input.zerocopy_weights,
        };
        let lock_ram_mb = match std::env::var("RUSTLLAMA_LOCK_RAM_MB") {
            Ok(v) => Some(v),
            Err(_) => {
                let t = input.lock_ram_mb.trim();
                if t.is_empty() || t == "0" {
                    None
                } else {
                    Some(input.lock_ram_mb.clone())
                }
            }
        };
        let prefill_batched = env_truthy("RUSTLLAMA_PREFILL_BATCHED").unwrap_or(input.prefill_batched);
        let prefill_batched_hybrid =
            env_truthy("RUSTLLAMA_PREFILL_BATCHED_HYBRID").unwrap_or(input.prefill_batched_hybrid);
        let memory_budget = match std::env::var("RUSTLLAMA_MEMORY_BUDGET") {
            Ok(v) => Some(v),
            Err(_) => {
                let t = input.memory_budget.trim();
                if t.is_empty() {
                    None
                } else {
                    Some(input.memory_budget.clone())
                }
            }
        };
        // `SSM_PREFILL_CHUNKED` / `HYBRID_KV_ANY` legacy gate = PRESENT (any
        // value, even `"0"`). So env present ⇒ on, else the config value.
        let ssm_prefill_chunked =
            std::env::var_os("RUSTLLAMA_SSM_PREFILL_CHUNKED").is_some() || input.ssm_prefill_chunked;
        let hybrid_kv_any =
            std::env::var_os("RUSTLLAMA_HYBRID_KV_ANY").is_some() || input.hybrid_kv_any;
        let kv_persist_mb = std::env::var("RUSTLLAMA_KV_PERSIST_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(input.kv_persist_mb);
        let moe_expert_cache_mb = std::env::var("RUSTLLAMA_MOE_EXPERT_CACHE_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(input.moe_expert_cache_mb);
        let moe_experts_cpu = match std::env::var("RUSTLLAMA_MOE_EXPERTS_CPU") {
            Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
            Err(_) => input.moe_experts_cpu,
        };
        let moe_gpu_split_permille = std::env::var("RUSTLLAMA_MOE_GPU_SPLIT_PERMILLE")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .map(|p| p.min(1000))
            .unwrap_or(input.moe_gpu_split_permille.min(1000));

        LoadOptions {
            zerocopy_weights,
            lock_ram_mb,
            prefill_batched,
            prefill_batched_hybrid,
            memory_budget,
            ssm_prefill_chunked,
            kv_persist_mb,
            moe_expert_cache_mb,
            moe_experts_cpu,
            moe_gpu_split_permille,
            hybrid_kv_any,
        }
    }

    /// Push the resolved values into the process-local `accel` cells, which
    /// the models/engine read at load + forward. Call BEFORE loading the
    /// model (the readers fire during load). A plain atomic store per knob —
    /// no `set_var`. The MoE expert-cache budget and zero-copy flag may be
    /// further overridden IN-PROCESS later (the budget planner, the
    /// pin-cache auto-enable); those run after this and win, matching the
    /// old env-then-auto-enable ordering.
    pub fn apply(&self) {
        accel::set_zerocopy_weights(self.zerocopy_weights);
        accel::set_prefill_batched(self.prefill_batched);
        accel::set_prefill_batched_hybrid(self.prefill_batched_hybrid);
        accel::set_ssm_prefill_chunked(self.ssm_prefill_chunked);
        accel::set_hybrid_kv_any(self.hybrid_kv_any);
        accel::set_kv_persist_mb(self.kv_persist_mb);
        accel::set_moe_experts_cpu(self.moe_experts_cpu);
        accel::set_moe_gpu_split_permille(self.moe_gpu_split_permille);
        if self.moe_expert_cache_mb > 0 {
            accel::set_moe_expert_cache_max_bytes(self.moe_expert_cache_mb * 1024 * 1024);
        }
        if let Some(v) = &self.lock_ram_mb {
            accel::set_lock_ram_mb_directive(v.clone());
        }
        if let Some(v) = &self.memory_budget {
            accel::set_memory_budget_directive(v.clone());
        }
    }

    /// The `(env_key, value)` pairs a spawned `tune --all` subprocess needs to
    /// measure under the same conditions as the serve load. The autotune
    /// child is a separate process (it does NOT share this process's `accel`
    /// cells), so these are set on its `Command` — a child-scoped env set, NOT
    /// a process-global `std::env::set_var`. Only the "on"/non-default knobs
    /// are emitted (mirrors the old promote, which set env only when the knob
    /// was enabled); the MoE-placement/hybrid-KV knobs are re-measured by the
    /// sweep itself and so are intentionally omitted.
    pub fn tune_child_env(&self) -> Vec<(&'static str, String)> {
        let mut out = Vec::new();
        if self.zerocopy_weights {
            out.push(("RUSTLLAMA_ZEROCOPY_WEIGHTS", "1".to_string()));
        }
        if let Some(v) = &self.lock_ram_mb {
            out.push(("RUSTLLAMA_LOCK_RAM_MB", v.clone()));
        }
        if self.prefill_batched {
            out.push(("RUSTLLAMA_PREFILL_BATCHED", "1".to_string()));
        }
        if self.prefill_batched_hybrid {
            out.push(("RUSTLLAMA_PREFILL_BATCHED_HYBRID", "1".to_string()));
        }
        if let Some(v) = &self.memory_budget {
            out.push(("RUSTLLAMA_MEMORY_BUDGET", v.clone()));
        }
        if self.ssm_prefill_chunked {
            out.push(("RUSTLLAMA_SSM_PREFILL_CHUNKED", "1".to_string()));
        }
        if self.kv_persist_mb > 0 {
            out.push(("RUSTLLAMA_KV_PERSIST_MB", self.kv_persist_mb.to_string()));
        }
        if self.moe_expert_cache_mb > 0 {
            out.push((
                "RUSTLLAMA_MOE_EXPERT_CACHE_MB",
                self.moe_expert_cache_mb.to_string(),
            ));
        }
        out
    }
}
