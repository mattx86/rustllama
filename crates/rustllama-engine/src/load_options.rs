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
    /// F12 periodic KV-checkpoint cadence (seconds); `0` = off (drop-only save).
    pub kv_persist_interval_secs: u64,
    pub moe_expert_cache_mb: u64,
    /// Already resolved from the tuner cache by the caller.
    pub moe_experts_cpu: bool,
    /// Already resolved from the tuner cache by the caller (permille).
    pub moe_gpu_split_permille: u32,
    pub hybrid_kv_any: bool,
    // --- P16 autotune axes (already resolved from the tuner cache by the
    // caller; `None` = no winner ⇒ keep the kernel's historical default). ---
    /// Parallel-matvec crossover (M rows).
    pub parallel_matvec_crossover: Option<u32>,
    /// GPU-vs-CPU matvec FLOP crossover.
    pub matvec_min_flops: Option<u64>,
    /// DeltaNet prefill chunk length.
    pub ssm_chunk: Option<u32>,
    // --- P16 config keys (from `[inference]`; `None`/default = off). ---
    /// Paged-KV sliding-window retention (token positions); `None` = off.
    pub kv_sliding_window: Option<u32>,
    /// Paged-KV heavy-hitter recent-keep (token positions); `None` = off.
    pub kv_heavy_hitter: Option<u32>,
    /// Paged-KV heavy-hitter sink block size (default 4).
    pub kv_heavy_sink: u32,
    /// VirtualLock SYCL USM into the working set (Windows).
    pub lock_usm: bool,
    /// USM-pin budget cap (MB).
    pub lock_usm_mb: u64,
    /// P1: generate past `ctx_size` via llama.cpp-style context shift
    /// (`[inference].context_shift`). `false` (default) keeps the historic
    /// `PromptTooLong` / hard-stop behavior byte-identical.
    pub context_shift: bool,
    /// P1: leading positions retained across a context shift
    /// (`[inference].context_shift_keep`).
    pub context_shift_keep: u32,
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
    /// F12 periodic KV-checkpoint cadence (seconds); `0` = off.
    pub kv_persist_interval_secs: u64,
    pub moe_expert_cache_mb: u64,
    pub moe_experts_cpu: bool,
    pub moe_gpu_split_permille: u32,
    pub hybrid_kv_any: bool,
    /// P16 autotune axes (`None` = keep the kernel default — byte-identical).
    pub parallel_matvec_crossover: Option<u32>,
    pub matvec_min_flops: Option<u64>,
    pub ssm_chunk: Option<u32>,
    /// P16 config keys.
    pub kv_sliding_window: Option<u32>,
    pub kv_heavy_hitter: Option<u32>,
    pub kv_heavy_sink: u32,
    pub lock_usm: bool,
    pub lock_usm_mb: u64,
    /// P1 context shift (generate past `ctx_size`).
    pub context_shift: bool,
    pub context_shift_keep: u32,
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
        let prefill_batched =
            env_truthy("RUSTLLAMA_PREFILL_BATCHED").unwrap_or(input.prefill_batched);
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
        let ssm_prefill_chunked = std::env::var_os("RUSTLLAMA_SSM_PREFILL_CHUNKED").is_some()
            || input.ssm_prefill_chunked;
        let hybrid_kv_any =
            std::env::var_os("RUSTLLAMA_HYBRID_KV_ANY").is_some() || input.hybrid_kv_any;
        let kv_persist_mb = std::env::var("RUSTLLAMA_KV_PERSIST_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(input.kv_persist_mb);
        let kv_persist_interval_secs = std::env::var("RUSTLLAMA_KV_PERSIST_INTERVAL_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(input.kv_persist_interval_secs);
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

        // P16: autotune axes — env (the legacy per-knob override) wins; else the
        // caller's cache-resolved winner (`None` ⇒ keep the kernel default).
        let parallel_matvec_crossover = std::env::var("RUSTLLAMA_PARALLEL_MATVEC_M")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .map(Some)
            .unwrap_or(input.parallel_matvec_crossover);
        let matvec_min_flops = std::env::var("RUSTLLAMA_MATVEC_MIN_FLOPS")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .map(Some)
            .unwrap_or(input.matvec_min_flops);
        let ssm_chunk = std::env::var("RUSTLLAMA_SSM_CHUNK")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .filter(|&c| c > 0)
            .map(Some)
            .unwrap_or(input.ssm_chunk);

        // P16: config keys — env wins, else the typed config value.
        let kv_sliding_window = std::env::var("RUSTLLAMA_KV_SLIDING_WINDOW")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|&n| n > 0)
            .map(Some)
            .unwrap_or(input.kv_sliding_window);
        let kv_heavy_hitter = std::env::var("RUSTLLAMA_KV_HEAVY_HITTER")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .filter(|&n| n > 0)
            .map(Some)
            .unwrap_or(input.kv_heavy_hitter);
        let kv_heavy_sink = std::env::var("RUSTLLAMA_KV_HEAVY_SINK")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(input.kv_heavy_sink);
        let lock_usm = match std::env::var("RUSTLLAMA_LOCK_USM") {
            Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
            Err(_) => input.lock_usm,
        };
        let lock_usm_mb = std::env::var("RUSTLLAMA_LOCK_USM_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(input.lock_usm_mb);
        // P1 context shift — env wins, else the typed config value.
        let context_shift = match std::env::var("RUSTLLAMA_CONTEXT_SHIFT") {
            Ok(v) => v == "1" || v.eq_ignore_ascii_case("true"),
            Err(_) => input.context_shift,
        };
        let context_shift_keep = std::env::var("RUSTLLAMA_CONTEXT_SHIFT_KEEP")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(input.context_shift_keep);

        LoadOptions {
            zerocopy_weights,
            lock_ram_mb,
            prefill_batched,
            prefill_batched_hybrid,
            memory_budget,
            ssm_prefill_chunked,
            kv_persist_mb,
            kv_persist_interval_secs,
            moe_expert_cache_mb,
            moe_experts_cpu,
            moe_gpu_split_permille,
            hybrid_kv_any,
            parallel_matvec_crossover,
            matvec_min_flops,
            ssm_chunk,
            kv_sliding_window,
            kv_heavy_hitter,
            kv_heavy_sink,
            lock_usm,
            lock_usm_mb,
            context_shift,
            context_shift_keep,
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
        accel::set_kv_persist_interval_secs(self.kv_persist_interval_secs);
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
        // P16 autotune axes: apply a winner only when present; `None` leaves the
        // kernel's env/historical default untouched (byte-identical).
        if let Some(m) = self.parallel_matvec_crossover {
            accel::set_parallel_matvec_crossover(m);
        }
        if let Some(f) = self.matvec_min_flops {
            accel::set_matvec_min_flops(f);
        }
        if let Some(c) = self.ssm_chunk {
            accel::set_ssm_chunk(c);
        }
        // P16 config keys: applied unconditionally (resolved from config, env
        // already folded in above) so a multi-model reload resets them cleanly.
        accel::set_kv_sliding_window(self.kv_sliding_window);
        accel::set_kv_heavy_hitter(self.kv_heavy_hitter);
        accel::set_kv_heavy_sink(Some(self.kv_heavy_sink));
        accel::set_lock_usm(self.lock_usm);
        accel::set_lock_usm_mb(self.lock_usm_mb);
        // P1 context shift (resolved from config + env above; applied
        // unconditionally so a multi-model reload resets it cleanly).
        accel::set_context_shift(self.context_shift);
        accel::set_context_shift_keep(self.context_shift_keep);
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
        if self.kv_persist_interval_secs > 0 {
            out.push((
                "RUSTLLAMA_KV_PERSIST_INTERVAL_SECS",
                self.kv_persist_interval_secs.to_string(),
            ));
        }
        if self.moe_expert_cache_mb > 0 {
            out.push((
                "RUSTLLAMA_MOE_EXPERT_CACHE_MB",
                self.moe_expert_cache_mb.to_string(),
            ));
        }
        // P16: the USM lock + paged-KV evictor affect residency/perf, so a tune
        // child should measure under the same config-resolved state. The three
        // swept axes (crossover / min_flops / ssm_chunk) are intentionally
        // OMITTED — the sweep RE-MEASURES them, so pinning them would defeat it
        // (mirrors the MoE-placement / hybrid-KV omission above).
        if self.lock_usm {
            out.push(("RUSTLLAMA_LOCK_USM", "1".to_string()));
            out.push(("RUSTLLAMA_LOCK_USM_MB", self.lock_usm_mb.to_string()));
        }
        if let Some(n) = self.kv_sliding_window {
            out.push(("RUSTLLAMA_KV_SLIDING_WINDOW", n.to_string()));
        }
        if let Some(n) = self.kv_heavy_hitter {
            out.push(("RUSTLLAMA_KV_HEAVY_HITTER", n.to_string()));
            out.push(("RUSTLLAMA_KV_HEAVY_SINK", self.kv_heavy_sink.to_string()));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The P16-relevant env vars this crate reads in `from_input`. Cleared at the
    /// top of each test so resolution is deterministic (config/input is the
    /// source), regardless of the host's environment.
    const P16_ENVS: &[&str] = &[
        "RUSTLLAMA_PARALLEL_MATVEC_M",
        "RUSTLLAMA_MATVEC_MIN_FLOPS",
        "RUSTLLAMA_SSM_CHUNK",
        "RUSTLLAMA_KV_SLIDING_WINDOW",
        "RUSTLLAMA_KV_HEAVY_HITTER",
        "RUSTLLAMA_KV_HEAVY_SINK",
        "RUSTLLAMA_LOCK_USM",
        "RUSTLLAMA_LOCK_USM_MB",
    ];
    fn clear_p16_envs() {
        for k in P16_ENVS {
            std::env::remove_var(k);
        }
    }

    /// P16 resolution, all three scenarios in ONE test so the process-global env
    /// mutations are serialized (cargo runs `#[test]`s in parallel threads, and
    /// these knobs read `std::env`, so splitting them would race).
    ///
    /// 1. Byte-identical default: an all-default Input (no cached winners, config
    ///    defaults, no env) resolves the three swept axes to `None` — so
    ///    `LoadOptions::apply` leaves the kernels at their historical constants.
    /// 2. A cached winner / config value flows through when no env override.
    /// 3. Env wins over the config/cache value (back-compat override).
    #[test]
    fn p16_resolution_defaults_config_and_env_wins() {
        // --- (1) defaults → None / off (byte-identical) ---
        clear_p16_envs();
        let input = LoadOptionsInput {
            kv_heavy_sink: 4,
            lock_usm_mb: 1024,
            ..LoadOptionsInput::default()
        };
        let opts = LoadOptions::from_input(&input);
        assert_eq!(opts.parallel_matvec_crossover, None);
        assert_eq!(opts.matvec_min_flops, None);
        assert_eq!(opts.ssm_chunk, None);
        assert_eq!(opts.kv_sliding_window, None);
        assert_eq!(opts.kv_heavy_hitter, None);
        assert!(!opts.lock_usm);
        assert_eq!(opts.lock_usm_mb, 1024);

        // --- (2) cached winner / config value flows through (no env) ---
        clear_p16_envs();
        let input = LoadOptionsInput {
            parallel_matvec_crossover: Some(512),
            matvec_min_flops: Some(1 << 22),
            ssm_chunk: Some(128),
            kv_sliding_window: Some(4096),
            kv_heavy_hitter: Some(1024),
            kv_heavy_sink: 8,
            lock_usm: true,
            lock_usm_mb: 2048,
            ..LoadOptionsInput::default()
        };
        let opts = LoadOptions::from_input(&input);
        assert_eq!(opts.parallel_matvec_crossover, Some(512));
        assert_eq!(opts.matvec_min_flops, Some(1 << 22));
        assert_eq!(opts.ssm_chunk, Some(128));
        assert_eq!(opts.kv_sliding_window, Some(4096));
        assert_eq!(opts.kv_heavy_hitter, Some(1024));
        assert_eq!(opts.kv_heavy_sink, 8);
        assert!(opts.lock_usm);
        assert_eq!(opts.lock_usm_mb, 2048);

        // --- (3) env overrides win for every P16 knob ---
        std::env::set_var("RUSTLLAMA_PARALLEL_MATVEC_M", "64");
        std::env::set_var("RUSTLLAMA_MATVEC_MIN_FLOPS", "1048576");
        std::env::set_var("RUSTLLAMA_SSM_CHUNK", "32");
        std::env::set_var("RUSTLLAMA_KV_SLIDING_WINDOW", "777");
        std::env::set_var("RUSTLLAMA_KV_HEAVY_HITTER", "555");
        std::env::set_var("RUSTLLAMA_KV_HEAVY_SINK", "9");
        std::env::set_var("RUSTLLAMA_LOCK_USM", "1");
        std::env::set_var("RUSTLLAMA_LOCK_USM_MB", "333");
        let input = LoadOptionsInput {
            parallel_matvec_crossover: Some(512),
            matvec_min_flops: Some(1 << 30),
            ssm_chunk: Some(256),
            kv_sliding_window: Some(4096),
            kv_heavy_hitter: Some(2048),
            kv_heavy_sink: 4,
            lock_usm: false,
            lock_usm_mb: 1024,
            ..LoadOptionsInput::default()
        };
        let opts = LoadOptions::from_input(&input);
        assert_eq!(opts.parallel_matvec_crossover, Some(64));
        assert_eq!(opts.matvec_min_flops, Some(1_048_576));
        assert_eq!(opts.ssm_chunk, Some(32));
        assert_eq!(opts.kv_sliding_window, Some(777));
        assert_eq!(opts.kv_heavy_hitter, Some(555));
        assert_eq!(opts.kv_heavy_sink, 9);
        assert!(opts.lock_usm);
        assert_eq!(opts.lock_usm_mb, 333);
        clear_p16_envs();
    }
}
