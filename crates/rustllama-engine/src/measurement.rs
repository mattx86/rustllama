//! Engine-driven measurement loops for the tuner's dynamic sweeps.
//!
//! Lives in `rustllama-engine` rather than `rustllama-cli` so the
//! HTTP server (and any future scripted-tune flow) can drive
//! measurements without going through the CLI binary. Outputs are
//! pure structured data — callers format for their own surface
//! (the CLI prints a tabular view; the server returns JSON; the
//! GUI's eventual "Run tune" button would render a progress card).
//!
//! Each measurement function loads the model exactly once and
//! reconfigures the engine between candidates via the relevant
//! setter (`set_n_gpu_layers` for placement, `set_prefill_chunk_size`
//! for batch-size). Prefix cache is disabled so each candidate's
//! run starts from a comparable cold state.

use std::path::Path;

use rustllama_tuner::{OptimizeFor, PerfCandidate, PlacementPlan};

use crate::inference::ContiguousEngine;
use crate::SamplingParams;
use rustllama_models::llama_arch::KvDtype;

/// Resolve `configured` into the engine's device placement, the SAME way the
/// serve / `model bench` load paths do, then install it.
///
/// `configured` comes from `[inference].n_gpu_layers`, whose default is the
/// AUTO sentinel `999` (`rustllama_config::N_GPU_LAYERS_AUTO`). That sentinel is
/// **not a layer count** — the engine's per-tensor device plan (measured-perf
/// heat / VRAM-fit) is only installed when AUTO is resolved through
/// `auto_place_heat`. Handing `999` straight to `set_n_gpu_layers` left every
/// measurement running **all-CPU** (the same AUTO-sentinel trap that made
/// `model bench` silently measure the CPU: see the T0a note in
/// `rustllama-cli`). On a 9B/27B model that all-CPU crawl looks like a hang.
///
/// An explicit non-sentinel override is honored as-is. The AUTO sentinel
/// resolves to the measured-perf heat plan under default placement opts
/// (`cpu_enabled`/`gpu_enabled` default on); a CPU-only / GPU-unavailable host
/// resolves to all-CPU as before, so this is a no-op there.
fn apply_measurement_placement(cpu: &mut ContiguousEngine, configured: u32) {
    // Mirrors `rustllama_config::N_GPU_LAYERS_AUTO` (kept inline so the engine
    // crate need not depend on `rustllama-config`).
    const N_GPU_LAYERS_AUTO: u32 = 999;
    let n = if configured == N_GPU_LAYERS_AUTO {
        cpu.auto_place_heat(&crate::placement_auto::AutoPlacementOpts::default())
            .n_gpu_layers
    } else {
        configured
    };
    cpu.set_n_gpu_layers(n);
}

/// Collect `repeats` timed decode runs (clearing the prefix cache before each)
/// and return the (UNSORTED) tok/s runs + the last error seen — the shared
/// inner loop of the tuner's decode A/B sweeps. Each run is one
/// `generate_token_ids` whose `last_request_stats()` gives `tokens_generated /
/// (decode_ms/1000)`. Factored out of ~half a dozen copy-pasted sweep loops;
/// the caller keeps its own warmup + sort + median/max + result shaping (which
/// differ between sweeps — simple vs even-averaged median, etc.).
fn collect_decode_tok_s(
    cpu: &mut ContiguousEngine,
    prompt_ids: &[i32],
    decode_tokens: u32,
    repeats: u32,
    sampling: &SamplingParams,
) -> (Vec<f64>, Option<String>) {
    let mut runs: Vec<f64> = Vec::with_capacity(repeats as usize);
    let mut last_err: Option<String> = None;
    for _ in 0..repeats {
        cpu.clear_prefix_cache();
        match cpu.generate_token_ids(prompt_ids, decode_tokens, sampling) {
            Ok(_) => {
                let stats = cpu.last_request_stats();
                if stats.decode_ms > 0.0 && stats.tokens_generated > 0 {
                    runs.push((stats.tokens_generated as f64) / (stats.decode_ms / 1000.0));
                }
            }
            Err(e) => last_err = Some(e.to_string()),
        }
    }
    (runs, last_err)
}

/// Per-candidate result from a placement sweep. `median_tps` is
/// `None` when every measurement run for this candidate failed
/// (warmup error, generate error, etc.) — `error` carries the
/// first failure message in that case.
///
/// Power readings (`watts`, `tokens_per_joule`) are `None` on non-
/// Intel hosts, or when Sysman doesn't expose
/// a power domain on the device (typical on Iris Xe — Arc parts
/// usually surface it). When `Some`, they're derived from the
/// cumulative energy counter sampled before + after each candidate's
/// run sequence: `watts = ΔE / Δt`, `tokens_per_joule = generated_tokens / (ΔE × 1e-6)`.
#[derive(Debug, Clone)]
pub struct PlacementCandidateResult {
    pub n_gpu_layers: u32,
    pub warmup_ms: f64,
    pub median_tps: Option<f64>,
    pub max_tps: Option<f64>,
    pub error: Option<String>,
    /// Average GPU power draw (watts) over the measured runs.
    pub watts: Option<f64>,
    /// Energy efficiency: tokens generated per joule. Higher = more
    /// efficient. Useful on laptops where the user cares more about
    /// battery life than peak throughput; pair with
    /// `RUSTLLAMA_TUNE_OPTIMIZE_FOR=watts` to pick the per-joule
    /// winner instead of the tok/s winner.
    pub tokens_per_joule: Option<f64>,
}

/// Outcome of `measure_placement_candidates`. The `winner` is the
/// candidate with the highest median tok/s; `None` when no
/// candidate produced a successful run.
#[derive(Debug, Clone)]
pub struct PlacementMeasurementReport {
    pub load_ms: f64,
    pub winner: Option<u32>,
    pub winner_tps: f64,
    pub candidates: Vec<PlacementCandidateResult>,
}

/// Per-candidate result from a batch-size sweep. `median_tps` is
/// prefill tok/s, not decode.
#[derive(Debug, Clone)]
pub struct BatchSizeCandidateResult {
    pub batch_size: usize,
    pub warmup_ms: f64,
    pub median_tps: Option<f64>,
    pub max_tps: Option<f64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct BatchSizeMeasurementReport {
    pub load_ms: f64,
    pub winner: Option<usize>,
    pub winner_tps: f64,
    pub candidates: Vec<BatchSizeCandidateResult>,
}

/// Knobs the measurement loop needs from the user's config. Kept
/// as a flat struct so callers can construct without depending on
/// the heavier `rustllama_config::Config` shape (which would force
/// an `rustllama-config` dep on this crate just for these few
/// fields).
#[derive(Debug, Clone)]
pub struct MeasurementConfig {
    pub kv_dtype: KvDtype,
    /// String form of `[inference].kv_cache_layout` — passed verbatim
    /// to `ContiguousEngine::load_with_options_and_layout`.
    pub kv_cache_layout: String,
    pub flash_attention: bool,
    /// Initial `n_gpu_layers` used for the batch-size sweep load.
    /// Ignored by the placement sweep (it overwrites per candidate).
    pub n_gpu_layers: u32,
    /// P16: the objective the sweep ranks candidates by (`Throughput` =
    /// tok/s, the default; `Watts` = tokens-per-joule). Resolved by the
    /// caller from `tune --optimize-for` / `RUSTLLAMA_TUNE_OPTIMIZE_FOR` /
    /// `[tuning].optimize_for`, and threaded here in place of the former
    /// inline env read in `measure_placement_candidates`.
    pub optimize_for: OptimizeFor,
}

/// Drive the engine through each placement candidate, time decode
/// tok/s, pick the candidate with the highest median.
///
/// Loads the model once. Between candidates: `set_n_gpu_layers(n)`,
/// `clear_prefix_cache()`, run one untimed warmup, then `repeats`
/// timed runs.
pub fn measure_placement_candidates(
    model_path: &Path,
    candidates: &[PlacementPlan],
    ctx_size: usize,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
    cfg: &MeasurementConfig,
) -> crate::Result<PlacementMeasurementReport> {
    let load_start = std::time::Instant::now();
    let mut cpu = ContiguousEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        true,
        cfg.kv_dtype,
        &cfg.kv_cache_layout,
    )
    .map_err(|e| crate::EngineError::Engine(format!("load failed: {e}")))?;
    cpu.set_prefix_cache(false);
    cpu.set_flash_attention(cfg.flash_attention);
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let vocab = cpu.vocab_size() as i32;
    let prompt_ids: Vec<i32> = (0..prompt_tokens as i32)
        .map(|i| 1 + (i % vocab.max(2).saturating_sub(1)))
        .collect();
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        typical_p: 1.0,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 0,
        max_tokens: decode_tokens,
        stop: Vec::new(),
        ..SamplingParams::default()
    };

    let mut report_candidates: Vec<PlacementCandidateResult> = Vec::with_capacity(candidates.len());
    let mut best_tps = 0.0f64;
    let mut best_n: Option<u32> = None;

    // Sysman probe for power-aware tuning. Constructed once and
    // reused across candidates; the probe call itself is cheap (~µs)
    // so per-candidate sampling is fine. `None` on non-Intel hosts
    // and in mock mode → power readings stay None throughout.
    let sysman_probe = open_sysman_probe();

    for plan in candidates {
        let n = plan.n_gpu_layers;
        cpu.set_n_gpu_layers(n);
        cpu.clear_prefix_cache();
        let warmup_t = std::time::Instant::now();
        let warmup_result = cpu.generate_token_ids(&prompt_ids, decode_tokens, &sampling);
        let warmup_ms = warmup_t.elapsed().as_secs_f64() * 1000.0;
        if let Err(e) = warmup_result {
            report_candidates.push(PlacementCandidateResult {
                n_gpu_layers: n,
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(format!("warmup failed: {e}")),
                watts: None,
                tokens_per_joule: None,
            });
            continue;
        }

        // Sample the cumulative energy counter at the start of the
        // measured-runs sequence. `None` when Sysman can't read it.
        let energy_before = sysman_probe.as_ref().and_then(|p| p.sample_energy());

        let mut runs: Vec<f64> = Vec::with_capacity(repeats as usize);
        let mut last_err: Option<String> = None;
        let mut total_generated: u64 = 0;
        for _ in 0..repeats {
            cpu.clear_prefix_cache();
            match cpu.generate_token_ids(&prompt_ids, decode_tokens, &sampling) {
                Ok(_) => {
                    let stats = cpu.last_request_stats();
                    if stats.decode_ms > 0.0 && stats.tokens_generated > 0 {
                        runs.push((stats.tokens_generated as f64) / (stats.decode_ms / 1000.0));
                        total_generated += stats.tokens_generated as u64;
                    }
                }
                Err(e) => last_err = Some(e.to_string()),
            }
        }
        let energy_after = sysman_probe.as_ref().and_then(|p| p.sample_energy());

        // Compute watts + tok/joule when both endpoints succeeded.
        // µJ → J: divide by 1e6; µs → s: divide by 1e6.
        let (watts, tokens_per_joule) = match (energy_before, energy_after) {
            (Some((e0, t0)), Some((e1, t1))) if t1 > t0 && e1 >= e0 => {
                let de_uj = (e1 - e0) as f64;
                let dt_us = (t1 - t0) as f64;
                let watts = de_uj / dt_us; // µJ/µs == W
                let tpj = if de_uj > 0.0 {
                    total_generated as f64 / (de_uj * 1e-6)
                } else {
                    0.0
                };
                (Some(watts), Some(tpj))
            }
            _ => (None, None),
        };

        if runs.is_empty() {
            report_candidates.push(PlacementCandidateResult {
                n_gpu_layers: n,
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(last_err.unwrap_or_else(|| "every measurement run failed".into())),
                watts,
                tokens_per_joule,
            });
            continue;
        }
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if runs.len() % 2 == 1 {
            runs[runs.len() / 2]
        } else {
            (runs[runs.len() / 2 - 1] + runs[runs.len() / 2]) / 2.0
        };
        let max = runs[runs.len() - 1];
        report_candidates.push(PlacementCandidateResult {
            n_gpu_layers: n,
            warmup_ms,
            median_tps: Some(median),
            max_tps: Some(max),
            error: None,
            watts,
            tokens_per_joule,
        });

        // Winning metric. P16: the objective is a resolved config/flag setting
        // (`cfg.optimize_for`), no longer an inline env read. `Watts` flips the
        // picker to tokens-per-joule (laptop / battery sensitive); the default
        // `Throughput` ranks by tok/s.
        let score = match cfg.optimize_for {
            OptimizeFor::Watts => tokens_per_joule.unwrap_or(median),
            OptimizeFor::Throughput => median,
        };
        if score > best_tps {
            best_tps = score;
            best_n = Some(n);
        }
    }

    Ok(PlacementMeasurementReport {
        load_ms,
        winner: best_n,
        winner_tps: best_tps,
        candidates: report_candidates,
    })
}

/// Open the Level Zero Sysman probe once for the duration of the
/// sweep. Returns `None` on non-Intel hosts, in mock-mode builds, or
/// when the loader DLL isn't present. The returned probe captures
/// the loader + first device handle so per-candidate energy reads
/// reuse the same context.
fn open_sysman_probe() -> Option<SysmanProbe> {
    let sysman = rustllama_l0_sys::Sysman::load().ok()?;
    let l0 = rustllama_l0_sys::LevelZero::load().ok()?;
    let drivers = l0.drivers().ok()?;
    if drivers.is_empty() {
        return None;
    }
    let devices = l0.devices(drivers[0]).ok()?;
    if devices.is_empty() {
        return None;
    }
    Some(SysmanProbe {
        sysman,
        device: devices[0],
    })
}

struct SysmanProbe {
    sysman: &'static rustllama_l0_sys::Sysman,
    device: rustllama_l0_sys::ZeDeviceHandle,
}

impl SysmanProbe {
    /// Sample (energy_uj, timestamp_us). `None` when Sysman doesn't
    /// expose a power domain on this device.
    fn sample_energy(&self) -> Option<(u64, u64)> {
        let reading = self.sysman.probe(self.device);
        reading
            .energy_counter
            .map(|c| (c.energy_uj, c.timestamp_us))
    }
}

// ============================================================
// Phase 3: per-device (per-tier) performance measurement
// ============================================================

/// One compute tier's measured decode throughput.
#[derive(Debug, Clone)]
pub struct PerDeviceTierResult {
    /// Stable device slug (map key): a GPU UUID slug
    /// ([`rustllama_tuner::sycl_gpu_key`] / [`rustllama_tuner::cuda_gpu_key`])
    /// or the CPU-tier slug ([`rustllama_tuner::cpu_perf_key`]).
    pub slug: String,
    /// Human label for logs (e.g. "CPU (enabled cores)", "SYCL active
    /// device", "CUDA device 0").
    pub label: String,
    /// Median decode tok/s across the timed runs. `None` when every run
    /// failed.
    pub median_tps: Option<f64>,
    /// First failure message, when `median_tps` is `None`.
    pub error: Option<String>,
}

/// Outcome of [`measure_per_device_perf`]. `scores` is the flat
/// slug→tok/s map the tuner persists into `TuningResult.per_device_perf`.
#[derive(Debug, Clone)]
pub struct PerDevicePerfReport {
    pub load_ms: f64,
    pub scores: std::collections::HashMap<String, f32>,
    pub tiers: Vec<PerDeviceTierResult>,
}

/// Measure short-synthetic-decode throughput (decode tok/s) for each
/// compute TIER observable on this host, keyed by the tier's stable slug
/// so the Phase 5 heat planner can rank tiers by MEASURED perf and weight
/// their budgets. Loads the model ONCE and reuses it across tiers,
/// mirroring [`measure_placement_candidates`].
///
/// Tiers measured (all in the SAME unit — decode tok/s — so CPU and GPU
/// are directly comparable, which is what the planner needs to decide
/// "slow iGPU gets less than the CPU"):
///   - **CPU tier** (`set_n_gpu_layers(0)`): pure-CPU decode on the
///     affinity-pinned enabled-core pool (so the score reflects the real
///     `disabled_cpus` set). Keyed by [`rustllama_tuner::cpu_perf_key`].
///   - **Active GPU tier** (`set_n_gpu_layers(u32::MAX)`): decode on the
///     GPU the engine dispatches to (CUDA device 0 when a usable NVIDIA
///     GPU is present, else the active SYCL device). Keyed by that
///     device's UUID slug. Recorded only when a GPU is actually active.
///
/// SCAFFOLD NOTE: on a >1-GPU host, isolating NON-active GPUs for
/// measurement needs per-device dispatch selection the engine does not yet
/// expose (the SYCL/CUDA active device is fixed at load). Measuring the
/// active GPU + CPU is exactly what's observable + validatable on the
/// single-iGPU box here; extending to each additional GPU is the
/// combined-build owner's follow-up (see [`measure_extra_gpu_stub`]). The
/// planner already tolerates a GPU with no measured entry by excluding it.
pub fn measure_per_device_perf(
    model_path: &Path,
    ctx_size: usize,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
    cfg: &MeasurementConfig,
) -> crate::Result<PerDevicePerfReport> {
    let load_start = std::time::Instant::now();
    let mut cpu = ContiguousEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        true,
        cfg.kv_dtype,
        &cfg.kv_cache_layout,
    )
    .map_err(|e| crate::EngineError::Engine(format!("load failed: {e}")))?;
    cpu.set_prefix_cache(false);
    cpu.set_flash_attention(cfg.flash_attention);
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let vocab = cpu.vocab_size() as i32;
    let prompt_ids: Vec<i32> = (0..prompt_tokens as i32)
        .map(|i| 1 + (i % vocab.max(2).saturating_sub(1)))
        .collect();
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        typical_p: 1.0,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 0,
        max_tokens: decode_tokens,
        stop: Vec::new(),
        ..SamplingParams::default()
    };

    // Time `repeats` decode runs at the given layer cutoff; return the
    // median decode tok/s, or an error string if every run failed.
    let bench = |cpu: &mut ContiguousEngine, n_gpu: u32| -> std::result::Result<f64, String> {
        cpu.set_n_gpu_layers(n_gpu);
        cpu.clear_prefix_cache();
        // One untimed warmup (JIT / USM upload / page-in).
        cpu.generate_token_ids(&prompt_ids, decode_tokens, &sampling)
            .map_err(|e| format!("warmup failed: {e}"))?;
        // `cpu` is already `&mut ContiguousEngine` here (closure param) — reborrow.
        let (mut runs, last_err) =
            collect_decode_tok_s(cpu, &prompt_ids, decode_tokens, repeats, &sampling);
        if runs.is_empty() {
            return Err(last_err.unwrap_or_else(|| "every measurement run failed".into()));
        }
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok(runs[runs.len() / 2])
    };

    let mut scores: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    let mut tiers: Vec<PerDeviceTierResult> = Vec::new();

    // --- CPU tier (all-CPU decode on the pinned enabled-core pool) ---
    {
        let slug = rustllama_tuner::cpu_perf_key();
        match bench(&mut cpu, 0) {
            Ok(tps) => {
                scores.insert(slug.clone(), tps as f32);
                tiers.push(PerDeviceTierResult {
                    slug,
                    label: "CPU (enabled cores)".to_string(),
                    median_tps: Some(tps),
                    error: None,
                });
            }
            Err(e) => tiers.push(PerDeviceTierResult {
                slug,
                label: "CPU (enabled cores)".to_string(),
                median_tps: None,
                error: Some(e),
            }),
        }
    }

    // --- Active GPU tier (decode on the device the engine dispatches to) ---
    if let Some((slug, label)) = active_gpu_slug() {
        match bench(&mut cpu, u32::MAX) {
            Ok(tps) => {
                scores.insert(slug.clone(), tps as f32);
                tiers.push(PerDeviceTierResult {
                    slug,
                    label,
                    median_tps: Some(tps),
                    error: None,
                });
            }
            Err(e) => tiers.push(PerDeviceTierResult {
                slug,
                label,
                median_tps: None,
                error: Some(e),
            }),
        }
    }

    // --- F14: extra (non-active) CUDA GPUs ---
    // Inert on a single-GPU / non-CUDA host (incl. this box): `scores`/`tiers`
    // are left exactly as the active-tier + CPU-tier blocks above produced them,
    // so the report is byte-identical to the single-device path. Only a genuine
    // multi-CUDA-GPU host records additional per-device entries (write-blind).
    measure_extra_cuda_gpus(&mut scores, &mut tiers);

    Ok(PerDevicePerfReport {
        load_ms,
        scores,
        tiers,
    })
}

/// One multi-GPU mode's measured decode throughput.
#[derive(Debug, Clone)]
pub struct MultiGpuModeResult {
    pub mode: String, // "off" | "pp" | "tp"
    pub median_tps: Option<f64>,
    pub error: Option<String>,
}

/// Result of the measured-autotune multi-GPU sweep.
#[derive(Debug, Clone)]
pub struct MultiGpuMeasurementReport {
    pub candidates: Vec<MultiGpuModeResult>,
    /// Fastest mode, or `None` when skipped / every run failed.
    pub winner: Option<String>,
    pub winner_tps: f64,
    /// The single-device (`"off"`) baseline, for the speedup readout.
    pub baseline_tps: f64,
    /// Set (and `winner == None`) when there was no multi-GPU choice to measure.
    pub skipped: Option<String>,
}

/// Measured autotune: time a representative decode under each multi-GPU mode
/// (single-device `"off"`, pipeline `"pp"`, tensor `"tp"`) and pick the fastest for
/// this (system, model). Each mode is driven by `RUSTLLAMA_MULTI_GPU` (read at load
/// by the placement planner + the TP dispatch), so every mode gets a FRESH load
/// with its own placement, then a warmup + `repeats` timed decode runs (median
/// tok/s). SKIPs — no winner — on < 2 usable CUDA GPUs, where single-device is the
/// only mode. WRITE-BLIND: only a >= 2-GPU host observes a real difference between
/// the modes; on one GPU every mode collapses to single-device. Run AFTER the
/// per-device-perf stage has banked (pp/tp placement reads it); `tune --all`
/// orders it so.
pub fn measure_multi_gpu_candidates(
    model_path: &Path,
    ctx_size: usize,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
    cfg: &MeasurementConfig,
) -> crate::Result<MultiGpuMeasurementReport> {
    if rustllama_models::accel::usable_cuda_device_count() < 2 {
        return Ok(MultiGpuMeasurementReport {
            candidates: Vec::new(),
            winner: None,
            winner_tps: 0.0,
            baseline_tps: 0.0,
            skipped: Some(
                "fewer than 2 usable CUDA GPUs — single-device is the only mode".to_string(),
            ),
        });
    }
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        typical_p: 1.0,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 0,
        max_tokens: decode_tokens,
        stop: Vec::new(),
        ..SamplingParams::default()
    };

    let saved_env = std::env::var("RUSTLLAMA_MULTI_GPU").ok();
    let mut candidates: Vec<MultiGpuModeResult> = Vec::new();
    for mode in ["off", "pp", "tp"] {
        // The mode is read at load by `auto_place_heat` (pipeline branch) + the TP
        // dispatch arms, so each mode needs its own fresh load + placement.
        std::env::set_var("RUSTLLAMA_MULTI_GPU", mode);
        let run = (|| -> std::result::Result<f64, String> {
            let mut eng = ContiguousEngine::load_with_options_and_layout(
                model_path,
                ctx_size,
                true,
                cfg.kv_dtype,
                &cfg.kv_cache_layout,
            )
            .map_err(|e| format!("load failed: {e}"))?;
            eng.set_prefix_cache(false);
            eng.set_flash_attention(cfg.flash_attention);
            // Install this mode's placement plan (reads RUSTLLAMA_MULTI_GPU). An
            // error (e.g. no banked per-device perf) leaves the single-device plan,
            // so the mode still benches — just without its multi-GPU split.
            let _ = eng.auto_place_heat(&crate::placement_auto::AutoPlacementOpts::default());
            let vocab = eng.vocab_size() as i32;
            let prompt_ids: Vec<i32> = (0..prompt_tokens as i32)
                .map(|i| 1 + (i % vocab.max(2).saturating_sub(1)))
                .collect();
            // One untimed warmup (JIT / upload / page-in).
            eng.generate_token_ids(&prompt_ids, decode_tokens, &sampling)
                .map_err(|e| format!("warmup failed: {e}"))?;
            let (mut runs, last_err) =
                collect_decode_tok_s(&mut eng, &prompt_ids, decode_tokens, repeats, &sampling);
            if runs.is_empty() {
                return Err(last_err.unwrap_or_else(|| "every measurement run failed".into()));
            }
            runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            Ok(runs[runs.len() / 2])
        })();
        match run {
            Ok(tps) => candidates.push(MultiGpuModeResult {
                mode: mode.to_string(),
                median_tps: Some(tps),
                error: None,
            }),
            Err(e) => candidates.push(MultiGpuModeResult {
                mode: mode.to_string(),
                median_tps: None,
                error: Some(e),
            }),
        }
    }
    // Restore the caller's environment.
    match saved_env {
        Some(v) => std::env::set_var("RUSTLLAMA_MULTI_GPU", v),
        None => std::env::remove_var("RUSTLLAMA_MULTI_GPU"),
    }

    let baseline_tps = candidates
        .iter()
        .find(|c| c.mode == "off")
        .and_then(|c| c.median_tps)
        .unwrap_or(0.0);
    let mut winner: Option<String> = None;
    let mut winner_tps = 0.0f64;
    for c in &candidates {
        if let Some(t) = c.median_tps {
            if t > winner_tps {
                winner_tps = t;
                winner = Some(c.mode.clone());
            }
        }
    }
    Ok(MultiGpuMeasurementReport {
        candidates,
        winner,
        winner_tps,
        baseline_tps,
        skipped: None,
    })
}

/// The slug + label of the GPU the engine actually dispatches to (CUDA
/// device 0 first, else the active SYCL device), or `None` when no GPU is
/// usable. Uses the SAME slug scheme as `system_fingerprint`, so the score
/// lands under the key the planner looks up.
fn active_gpu_slug() -> Option<(String, String)> {
    if rustllama_models::accel::cuda_active() {
        if let Ok(info) = rustllama_kernels_cuda::device_info(0) {
            let slug = rustllama_tuner::cuda_gpu_key(&info);
            return Some((slug, format!("CUDA device 0 ({})", info.name)));
        }
    }
    let idx = rustllama_models::accel::first_enabled_sycl_device_index()?;
    let info = rustllama_kernels_sycl::device_info(idx).ok()?;
    let slug = rustllama_tuner::sycl_gpu_key(&info);
    Some((slug, format!("SYCL device {idx} ({})", info.name)))
}

/// F14 — real per-device micro-bench for a NON-active GPU. Runs a
/// representative packed (Q8_0) matvec on CUDA device `device_index` and returns
/// its throughput in **matvecs/sec** (median of `reps` timed runs, after a
/// warmup). This replaces the former `measure_extra_gpu_stub`: a secondary GPU
/// now gets a real measured number instead of being silently excluded.
///
/// Returns `None` when no such CUDA device exists, the per-device cache can't be
/// created, or a run fails — so on a **non-CUDA or single-GPU host this is inert**
/// (`CudaMatvecCache::new` yields `None` off-NVIDIA, and `device_count()` is 0/1),
/// and the single-device measurement path is byte-identical to before. The
/// multi-GPU numbers themselves are WRITE-BLIND (no second GPU on this box);
/// the structure is exercised by the unit tests + the inert single-GPU path.
///
/// Why a micro-bench rather than full decode: end-to-end decode on an arbitrary
/// device needs a per-device dispatch seam the engine doesn't expose (the active
/// SYCL/CUDA device is fixed at load). A per-device `CudaMatvecCache` DOES let us
/// time real device work in isolation, which [`scale_extra_gpu_tps`] converts
/// into a decode-tok/s-equivalent score by anchoring to the active GPU's
/// measured end-to-end tps.
fn measure_matvec_throughput_cuda(device_index: u32, reps: u32) -> Option<f64> {
    use rustllama_kernels_cuda::{CudaMatvecCache, CudaPackedKind};
    // Representative decode-ish projection shape (square 4096×4096 Q8_0).
    const M: usize = 4096;
    const K: usize = 4096;
    let kind = CudaPackedKind::Q8_0;
    let row_bytes = kind.row_bytes(K);
    // Budget enough to resident-cache the one synthetic weight on the device.
    let budget = M * row_bytes + (1 << 20);
    let mut cache = CudaMatvecCache::new(device_index, budget)?;
    // Shape-valid synthetic weight (contents arbitrary — this is a timing
    // bench, not a correctness probe) + input / output scratch.
    let w = vec![0u8; M * row_bytes];
    let x = vec![1.0f32; K];
    let mut out = vec![0.0f32; M];
    let weight_key = w.as_ptr() as usize;
    // Warmup: JIT + weight upload + first device touch (not timed).
    if !cache.matvec_packed(kind, weight_key, &w, &x, &mut out, M, K) {
        return None;
    }
    let mut rates: Vec<f64> = Vec::with_capacity(reps.max(1) as usize);
    for _ in 0..reps.max(1) {
        let t = std::time::Instant::now();
        if !cache.matvec_packed(kind, weight_key, &w, &x, &mut out, M, K) {
            return None;
        }
        let secs = t.elapsed().as_secs_f64();
        if secs > 0.0 {
            rates.push(1.0 / secs); // matvecs/sec
        }
    }
    if rates.is_empty() {
        return None;
    }
    rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(rates[rates.len() / 2])
}

/// F14 — convert an extra GPU's raw micro-bench throughput into a
/// decode-tok/s-equivalent score by scaling the ACTIVE GPU's measured
/// end-to-end decode tps by the throughput ratio. This keeps every per-device
/// score in the SAME unit (decode tok/s) so the heat planner can rank a
/// secondary GPU directly against the active one and the CPU. `None` when any
/// input is non-positive (no valid ratio to form). Pure — unit-tested without a
/// GPU.
fn scale_extra_gpu_tps(active_decode_tps: f64, active_mv: f64, extra_mv: f64) -> Option<f32> {
    if active_decode_tps <= 0.0 || active_mv <= 0.0 || extra_mv <= 0.0 {
        return None;
    }
    Some((active_decode_tps * (extra_mv / active_mv)) as f32)
}

/// F14 — measure each non-active CUDA GPU and fold a decode-tok/s-equivalent
/// score into `scores` + `tiers`. No-op unless a CUDA GPU is active AND the host
/// has >= 2 CUDA devices AND the active tier already recorded a decode tps to
/// anchor the scaling — so on a single-GPU / non-CUDA host (incl. this box) this
/// leaves `scores`/`tiers` untouched and the report is byte-identical to the
/// single-device path. The active device is index 0 (matching
/// [`active_gpu_slug`]); extras are `1..device_count()`.
fn measure_extra_cuda_gpus(
    scores: &mut std::collections::HashMap<String, f32>,
    tiers: &mut Vec<PerDeviceTierResult>,
) {
    if !rustllama_models::accel::cuda_active() {
        return;
    }
    let count = rustllama_kernels_cuda::device_count();
    if count < 2 {
        return;
    }
    // Anchor on the active CUDA device's already-measured end-to-end decode tps.
    let Some((active_slug, _)) = active_gpu_slug() else {
        return;
    };
    let Some(active_tps) = scores.get(&active_slug).copied() else {
        return;
    };
    const REPS: u32 = 5;
    let Some(active_mv) = measure_matvec_throughput_cuda(0, REPS) else {
        return;
    };
    for idx in 1..count {
        let Ok(info) = rustllama_kernels_cuda::device_info(idx) else {
            continue;
        };
        let slug = rustllama_tuner::cuda_gpu_key(&info);
        let label = format!("CUDA device {idx} ({})", info.name);
        match measure_matvec_throughput_cuda(idx, REPS) {
            Some(extra_mv) => match scale_extra_gpu_tps(active_tps as f64, active_mv, extra_mv) {
                Some(tps) => {
                    scores.insert(slug.clone(), tps);
                    tiers.push(PerDeviceTierResult {
                        slug,
                        label,
                        median_tps: Some(tps as f64),
                        error: None,
                    });
                }
                None => tiers.push(PerDeviceTierResult {
                    slug,
                    label,
                    median_tps: None,
                    error: Some("extra-GPU scaling produced no score".into()),
                }),
            },
            None => tiers.push(PerDeviceTierResult {
                slug,
                label,
                median_tps: None,
                error: Some("extra-GPU micro-bench failed".into()),
            }),
        }
    }
}

/// Per-candidate result from a KV-dtype sweep. `median_tps` is
/// decode tok/s; KV-dtype's main effect is on attention throughput
/// + memory footprint, both visible in decode timing.
///
/// `decode_tokens` carries the deterministic (temp=0) greedy
/// generation produced by the warmup run. Used by the coherence-
/// aware picker to compute top-1 agreement against the highest-
/// precision candidate's sequence — that's how the autotuner picks
/// the smallest KV dtype that still produces the same tokens.
#[derive(Debug, Clone)]
pub struct KvDtypeCandidateResult {
    pub kv_dtype: KvDtype,
    pub warmup_ms: f64,
    pub median_tps: Option<f64>,
    pub max_tps: Option<f64>,
    pub error: Option<String>,
    pub decode_tokens: Option<Vec<u32>>,
}

#[derive(Debug, Clone)]
pub struct KvDtypeMeasurementReport {
    pub total_load_ms: f64,
    pub winner: Option<KvDtype>,
    pub winner_tps: f64,
    pub candidates: Vec<KvDtypeCandidateResult>,
}

impl KvDtypeMeasurementReport {
    /// Coherence-first KV dtype selection.
    ///
    /// Picks the smallest-memory KV dtype whose deterministic
    /// (temp=0) greedy decode agrees with the highest-precision
    /// candidate's decode on at least `agreement_threshold`
    /// fraction of tokens. Ties on memory are broken by tok/s.
    ///
    /// The reference is the highest-precision candidate present —
    /// F32 if it ran, else the candidate with the largest
    /// `approx_bits_per_element` that produced tokens. If only
    /// one candidate produced tokens, that candidate wins (no
    /// reference to compare against).
    ///
    /// Returns `None` only if no candidate produced tokens at all.
    ///
    /// `agreement_threshold` is in `[0, 1]`. A typical value of
    /// `0.90` says "the chosen dtype must agree with F32 on ≥90%
    /// of decoded tokens." Setting it to `1.0` requires bit-exact
    /// agreement (rare for any quant); setting it to `0.0`
    /// disables the coherence gate entirely and reduces to pure
    /// memory-first selection.
    pub fn pick_coherence_first(&self, agreement_threshold: f32) -> Option<KvDtype> {
        // Find the reference: highest-bits candidate with tokens.
        let mut reference: Option<(KvDtype, &[u32])> = None;
        for c in &self.candidates {
            if let Some(toks) = c.decode_tokens.as_deref() {
                let take = match reference {
                    None => true,
                    Some((dt, _)) => {
                        c.kv_dtype.approx_bits_per_element() > dt.approx_bits_per_element()
                    }
                };
                if take {
                    reference = Some((c.kv_dtype, toks));
                }
            }
        }
        let (ref_dt, ref_toks) = reference?;

        // Rank eligible candidates: (bits_per_element ASC, tps DESC).
        let mut eligible: Vec<(KvDtype, f32, f64)> = Vec::new();
        for c in &self.candidates {
            let Some(toks) = c.decode_tokens.as_deref() else {
                continue;
            };
            let Some(med) = c.median_tps else { continue };
            // Agreement vs reference (across the shorter of the two).
            let agree = top1_agreement(toks, ref_toks);
            // The reference candidate itself always passes (agree
            // with itself = 1.0). All others must clear the bar.
            if c.kv_dtype != ref_dt && agree < agreement_threshold {
                continue;
            }
            eligible.push((c.kv_dtype, c.kv_dtype.approx_bits_per_element(), med));
        }
        if eligible.is_empty() {
            return Some(ref_dt);
        }
        eligible.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
        });
        Some(eligible[0].0)
    }

    /// Speed-first-if-it-fits KV dtype selection.
    ///
    /// Among candidates that stay coherent (top-1 agreement ≥
    /// `agreement_threshold` with the highest-precision reference —
    /// same reference + coherence gate as [`Self::pick_coherence_first`]),
    /// pick the **fastest** (highest median tok/s) whose KV cache fits
    /// the memory budget. `kv_bytes(dt)` returns the estimated total
    /// K+V cache size at the deployment context for dtype `dt`; a
    /// candidate is eligible only when `kv_bytes(dt) <= budget_bytes`.
    ///
    /// Under memory pressure — when no coherent candidate fits — fall
    /// back to the smallest coherent dtype (identical to
    /// [`Self::pick_coherence_first`]), so a tight box still gets a
    /// coherent, memory-frugal pick rather than one that would OOM.
    ///
    /// `budget_bytes == 0` disables the fit gate (everything is treated
    /// as fitting): callers pass `0` when they cannot determine a budget
    /// and would rather honor the speed-first intent than guess.
    ///
    /// Returns `None` only if no candidate produced tokens at all.
    pub fn pick_speed_first_if_fits(
        &self,
        agreement_threshold: f32,
        budget_bytes: u64,
        kv_bytes: impl Fn(KvDtype) -> u64,
    ) -> Option<KvDtype> {
        // Reference: highest-bits candidate that produced tokens
        // (shared definition with `pick_coherence_first`).
        let mut reference: Option<(KvDtype, &[u32])> = None;
        for c in &self.candidates {
            if let Some(toks) = c.decode_tokens.as_deref() {
                let take = match reference {
                    None => true,
                    Some((dt, _)) => {
                        c.kv_dtype.approx_bits_per_element() > dt.approx_bits_per_element()
                    }
                };
                if take {
                    reference = Some((c.kv_dtype, toks));
                }
            }
        }
        let (ref_dt, ref_toks) = reference?;

        // Coherent candidates that produced a median tps:
        // (kv_dtype, bits_per_element, median_tps).
        let mut coherent: Vec<(KvDtype, f32, f64)> = Vec::new();
        for c in &self.candidates {
            let Some(toks) = c.decode_tokens.as_deref() else {
                continue;
            };
            let Some(med) = c.median_tps else { continue };
            let agree = top1_agreement(toks, ref_toks);
            // The reference passes trivially (agrees with itself);
            // others must clear the coherence bar.
            if c.kv_dtype != ref_dt && agree < agreement_threshold {
                continue;
            }
            coherent.push((c.kv_dtype, c.kv_dtype.approx_bits_per_element(), med));
        }
        if coherent.is_empty() {
            return Some(ref_dt);
        }

        // Fastest coherent that fits the budget (0 = no gate).
        let fits = |dt: KvDtype| budget_bytes == 0 || kv_bytes(dt) <= budget_bytes;
        if let Some((dt, _, _)) = coherent
            .iter()
            .filter(|(dt, _, _)| fits(*dt))
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
        {
            return Some(*dt);
        }

        // Memory pressure: nothing coherent fits. Fall back to the
        // smallest coherent dtype (bits ASC, tps DESC) — the same
        // memory-frugal pick `pick_coherence_first` would make.
        coherent.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal))
        });
        Some(coherent[0].0)
    }

    /// Compute top-1 agreement (in `[0, 1]`) between each candidate's
    /// decode and a reference KV dtype's decode. Returns the pairs
    /// in `(kv_dtype, agreement)` order. Useful for diagnostic
    /// printing.
    pub fn agreement_vs(&self, reference: KvDtype) -> Vec<(KvDtype, Option<f32>)> {
        let ref_toks = self
            .candidates
            .iter()
            .find(|c| c.kv_dtype == reference)
            .and_then(|c| c.decode_tokens.as_deref());
        self.candidates
            .iter()
            .map(|c| match (c.decode_tokens.as_deref(), ref_toks) {
                (Some(toks), Some(r)) => (c.kv_dtype, Some(top1_agreement(toks, r))),
                _ => (c.kv_dtype, None),
            })
            .collect()
    }
}

fn top1_agreement(a: &[u32], b: &[u32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let matches: usize = a
        .iter()
        .zip(b.iter())
        .take(n)
        .filter(|(x, y)| x == y)
        .count();
    matches as f32 / n as f32
}

/// Drive the engine through each KV-dtype candidate. B1: the model is loaded
/// ONCE (for the first candidate) and the KV backend is HOT-RE-INIT per
/// subsequent candidate via [`ContiguousEngine::reinit_kv_backend`] — the
/// weights stay resident, so an N-candidate sweep costs one full load instead of
/// N. The first candidate stays a fresh load (F32 is always `candidates[0]`, the
/// coherence reference — never skewed by a re-init), and a re-init failure falls
/// back to a fresh load so a candidate is never silently skipped.
///
/// `candidates` typically covers `{F32, Q8_0, Tq(1|2|4|8), Nvfp4}`;
/// the picker selects the winner by median decode tok/s.
pub fn measure_kv_dtype_candidates(
    model_path: &Path,
    candidates: &[KvDtype],
    ctx_size: usize,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
    cfg: &MeasurementConfig,
) -> crate::Result<KvDtypeMeasurementReport> {
    let mut report_candidates: Vec<KvDtypeCandidateResult> = Vec::with_capacity(candidates.len());
    let mut best_tps = 0.0f64;
    let mut best_dt: Option<KvDtype> = None;
    let mut total_load_ms = 0.0f64;

    // B1: load the model ONCE for the first candidate and HOT-RE-INIT the KV
    // backend for every subsequent candidate — the weights (the dominant
    // per-candidate load cost) stay resident, cutting the sweep from N full loads
    // to one. The FIRST candidate stays a FRESH load so the coherence reference
    // (F32 is always candidates[0]) is never skewed by a re-init; a re-init
    // FAILURE falls back to a fresh load so a candidate is never silently skipped.
    let mut engine: Option<ContiguousEngine> = None;
    for (cand_idx, &dt) in candidates.iter().enumerate() {
        let load_start = std::time::Instant::now();
        let fresh_load = |engine: &mut Option<ContiguousEngine>| -> Option<String> {
            match ContiguousEngine::load_with_options_and_layout(
                model_path,
                ctx_size,
                true,
                dt,
                &cfg.kv_cache_layout,
            ) {
                Ok(mut c) => {
                    c.set_prefix_cache(false);
                    c.set_flash_attention(cfg.flash_attention);
                    apply_measurement_placement(&mut c, cfg.n_gpu_layers);
                    *engine = Some(c);
                    None
                }
                Err(e) => Some(format!("load failed: {e}")),
            }
        };
        let acquire_err: Option<String> = if cand_idx == 0 || engine.is_none() {
            fresh_load(&mut engine)
        } else if engine
            .as_mut()
            .unwrap()
            .reinit_kv_backend(dt, &cfg.kv_cache_layout)
            .is_ok()
        {
            None
        } else {
            // Re-init failed → fall back to a fresh load so this candidate is
            // still measured (never silently skipped).
            fresh_load(&mut engine)
        };
        total_load_ms += load_start.elapsed().as_secs_f64() * 1000.0;
        if let Some(e) = acquire_err {
            report_candidates.push(KvDtypeCandidateResult {
                kv_dtype: dt,
                warmup_ms: 0.0,
                median_tps: None,
                max_tps: None,
                error: Some(e),
                decode_tokens: None,
            });
            continue;
        }
        let cpu = engine.as_mut().unwrap();

        let vocab = cpu.vocab_size() as i32;
        let prompt_ids: Vec<i32> = (0..prompt_tokens as i32)
            .map(|i| 1 + (i % vocab.max(2).saturating_sub(1)))
            .collect();
        let sampling = SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            typical_p: 1.0,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: 0,
            max_tokens: decode_tokens,
            stop: Vec::new(),
            ..SamplingParams::default()
        };

        cpu.clear_prefix_cache();
        let warmup_t = std::time::Instant::now();
        let warmup_result = cpu.generate_token_ids(&prompt_ids, decode_tokens, &sampling);
        let warmup_ms = warmup_t.elapsed().as_secs_f64() * 1000.0;
        let decode_tokens_seq: Option<Vec<u32>> = match warmup_result {
            Ok(toks) => Some(toks),
            Err(e) => {
                report_candidates.push(KvDtypeCandidateResult {
                    kv_dtype: dt,
                    warmup_ms,
                    median_tps: None,
                    max_tps: None,
                    error: Some(format!("warmup failed: {e}")),
                    decode_tokens: None,
                });
                continue;
            }
        };

        let (mut runs, last_err) =
            collect_decode_tok_s(&mut *cpu, &prompt_ids, decode_tokens, repeats, &sampling);
        if runs.is_empty() {
            report_candidates.push(KvDtypeCandidateResult {
                kv_dtype: dt,
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(last_err.unwrap_or_else(|| "every measurement run failed".into())),
                decode_tokens: decode_tokens_seq,
            });
            continue;
        }
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if runs.len() % 2 == 1 {
            runs[runs.len() / 2]
        } else {
            (runs[runs.len() / 2 - 1] + runs[runs.len() / 2]) / 2.0
        };
        let max = runs[runs.len() - 1];
        report_candidates.push(KvDtypeCandidateResult {
            kv_dtype: dt,
            warmup_ms,
            median_tps: Some(median),
            max_tps: Some(max),
            error: None,
            decode_tokens: decode_tokens_seq,
        });
        if median > best_tps {
            best_tps = median;
            best_dt = Some(dt);
        }
    }

    Ok(KvDtypeMeasurementReport {
        total_load_ms,
        winner: best_dt,
        winner_tps: best_tps,
        candidates: report_candidates,
    })
}

/// Per-candidate result from a flash-attention on/off sweep.
#[derive(Debug, Clone)]
pub struct FlashAttentionCandidateResult {
    pub flash_attention: bool,
    pub warmup_ms: f64,
    pub median_tps: Option<f64>,
    pub max_tps: Option<f64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FlashAttentionMeasurementReport {
    pub load_ms: f64,
    pub winner: Option<bool>,
    pub winner_tps: f64,
    pub candidates: Vec<FlashAttentionCandidateResult>,
}

/// Sweep `flash_attention = true` vs `false`. Reuses one loaded
/// engine — flash-attention is a hot toggle via
/// [`ContiguousEngine::set_flash_attention`].
pub fn measure_flash_attention_candidates(
    model_path: &Path,
    ctx_size: usize,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
    cfg: &MeasurementConfig,
) -> crate::Result<FlashAttentionMeasurementReport> {
    let load_start = std::time::Instant::now();
    let mut cpu = ContiguousEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        true,
        cfg.kv_dtype,
        &cfg.kv_cache_layout,
    )
    .map_err(|e| crate::EngineError::Engine(format!("load failed: {e}")))?;
    cpu.set_prefix_cache(false);
    apply_measurement_placement(&mut cpu, cfg.n_gpu_layers);
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let vocab = cpu.vocab_size() as i32;
    let prompt_ids: Vec<i32> = (0..prompt_tokens as i32)
        .map(|i| 1 + (i % vocab.max(2).saturating_sub(1)))
        .collect();
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        typical_p: 1.0,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 0,
        max_tokens: decode_tokens,
        stop: Vec::new(),
        ..SamplingParams::default()
    };

    let mut report_candidates: Vec<FlashAttentionCandidateResult> = Vec::with_capacity(2);
    let mut best_tps = 0.0f64;
    let mut best_flag: Option<bool> = None;

    for &flag in &[true, false] {
        cpu.set_flash_attention(flag);
        cpu.clear_prefix_cache();
        let warmup_t = std::time::Instant::now();
        let warmup_result = cpu.generate_token_ids(&prompt_ids, decode_tokens, &sampling);
        let warmup_ms = warmup_t.elapsed().as_secs_f64() * 1000.0;
        if let Err(e) = warmup_result {
            report_candidates.push(FlashAttentionCandidateResult {
                flash_attention: flag,
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(format!("warmup failed: {e}")),
            });
            continue;
        }

        let (mut runs, last_err) =
            collect_decode_tok_s(&mut cpu, &prompt_ids, decode_tokens, repeats, &sampling);
        if runs.is_empty() {
            report_candidates.push(FlashAttentionCandidateResult {
                flash_attention: flag,
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(last_err.unwrap_or_else(|| "every measurement run failed".into())),
            });
            continue;
        }
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if runs.len() % 2 == 1 {
            runs[runs.len() / 2]
        } else {
            (runs[runs.len() / 2 - 1] + runs[runs.len() / 2]) / 2.0
        };
        let max = runs[runs.len() - 1];
        report_candidates.push(FlashAttentionCandidateResult {
            flash_attention: flag,
            warmup_ms,
            median_tps: Some(median),
            max_tps: Some(max),
            error: None,
        });
        if median > best_tps {
            best_tps = median;
            best_flag = Some(flag);
        }
    }

    Ok(FlashAttentionMeasurementReport {
        load_ms,
        winner: best_flag,
        winner_tps: best_tps,
        candidates: report_candidates,
    })
}

/// Per-candidate result from a `kv_cache_layout` sweep
/// (`"contiguous"` vs `"paged"`).
#[derive(Debug, Clone)]
pub struct KvLayoutCandidateResult {
    pub kv_cache_layout: String,
    pub warmup_ms: f64,
    pub median_tps: Option<f64>,
    pub max_tps: Option<f64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct KvLayoutMeasurementReport {
    pub total_load_ms: f64,
    pub winner: Option<String>,
    pub winner_tps: f64,
    pub candidates: Vec<KvLayoutCandidateResult>,
}

/// Sweep `kv_cache_layout`. Loads per candidate since the cache
/// shape changes with the layout choice.
///
/// `candidates` typically `&["contiguous", "paged"]`. Paged requires
/// `kv_dtype = F32` today — paged + non-F32 fails to load with a
/// clear error and the candidate is marked failed (the sweep
/// continues to the next).
pub fn measure_kv_layout_candidates(
    model_path: &Path,
    candidates: &[String],
    ctx_size: usize,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
    cfg: &MeasurementConfig,
) -> crate::Result<KvLayoutMeasurementReport> {
    let mut report_candidates: Vec<KvLayoutCandidateResult> = Vec::with_capacity(candidates.len());
    let mut best_tps = 0.0f64;
    let mut best_layout: Option<String> = None;
    let mut total_load_ms = 0.0f64;

    for layout in candidates {
        let load_start = std::time::Instant::now();
        let load_result = ContiguousEngine::load_with_options_and_layout(
            model_path,
            ctx_size,
            true,
            cfg.kv_dtype,
            layout,
        );
        total_load_ms += load_start.elapsed().as_secs_f64() * 1000.0;
        let mut cpu = match load_result {
            Ok(c) => c,
            Err(e) => {
                report_candidates.push(KvLayoutCandidateResult {
                    kv_cache_layout: layout.clone(),
                    warmup_ms: 0.0,
                    median_tps: None,
                    max_tps: None,
                    error: Some(format!("load failed: {e}")),
                });
                continue;
            }
        };
        cpu.set_prefix_cache(false);
        cpu.set_flash_attention(cfg.flash_attention);
        apply_measurement_placement(&mut cpu, cfg.n_gpu_layers);

        let vocab = cpu.vocab_size() as i32;
        let prompt_ids: Vec<i32> = (0..prompt_tokens as i32)
            .map(|i| 1 + (i % vocab.max(2).saturating_sub(1)))
            .collect();
        let sampling = SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            typical_p: 1.0,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: 0,
            max_tokens: decode_tokens,
            stop: Vec::new(),
            ..SamplingParams::default()
        };

        cpu.clear_prefix_cache();
        let warmup_t = std::time::Instant::now();
        let warmup_result = cpu.generate_token_ids(&prompt_ids, decode_tokens, &sampling);
        let warmup_ms = warmup_t.elapsed().as_secs_f64() * 1000.0;
        if let Err(e) = warmup_result {
            report_candidates.push(KvLayoutCandidateResult {
                kv_cache_layout: layout.clone(),
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(format!("warmup failed: {e}")),
            });
            continue;
        }

        let (mut runs, last_err) =
            collect_decode_tok_s(&mut cpu, &prompt_ids, decode_tokens, repeats, &sampling);
        if runs.is_empty() {
            report_candidates.push(KvLayoutCandidateResult {
                kv_cache_layout: layout.clone(),
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(last_err.unwrap_or_else(|| "every measurement run failed".into())),
            });
            continue;
        }
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if runs.len() % 2 == 1 {
            runs[runs.len() / 2]
        } else {
            (runs[runs.len() / 2 - 1] + runs[runs.len() / 2]) / 2.0
        };
        let max = runs[runs.len() - 1];
        report_candidates.push(KvLayoutCandidateResult {
            kv_cache_layout: layout.clone(),
            warmup_ms,
            median_tps: Some(median),
            max_tps: Some(max),
            error: None,
        });
        if median > best_tps {
            best_tps = median;
            best_layout = Some(layout.clone());
        }
    }

    Ok(KvLayoutMeasurementReport {
        total_load_ms,
        winner: best_layout,
        winner_tps: best_tps,
        candidates: report_candidates,
    })
}

/// Per-engine result from the P9 `tune --batching` A/B. `median_tps` is the
/// **single-stream effective throughput** — generated tokens ÷ the full
/// (prefill + decode) wall time of one request — the user-perceived number the
/// perf gate must not regress.
#[derive(Debug, Clone)]
pub struct BatchingEngineResult {
    /// `"paged"` (the continuous-batching `PagedBatchEngine`) or
    /// `"contiguous"` (the single-stream `ContiguousEngine`).
    pub engine: String,
    pub warmup_ms: f64,
    pub median_tps: Option<f64>,
    pub max_tps: Option<f64>,
    pub error: Option<String>,
}

/// Outcome of [`measure_batching_candidates`]. `winner` is `"paged"` when the
/// batching engine is parity-or-faster single-stream (so default-on is safe),
/// else `"contiguous"`. `None` when neither engine produced a run.
#[derive(Debug, Clone)]
pub struct BatchingMeasurementReport {
    pub total_load_ms: f64,
    pub winner: Option<String>,
    pub paged_tps: Option<f64>,
    pub contiguous_tps: Option<f64>,
    pub candidates: Vec<BatchingEngineResult>,
}

/// P9 batching A/B: a batching winner is `"paged"` as long as the paged engine
/// is within `(1 - PARITY) ×` the contiguous engine's single-stream tok/s —
/// i.e. "parity-or-better" keeps continuous batching default-ON (its
/// concurrency win dominates once >1 request is in flight), and only a CLEAR
/// single-stream loss flips the serve path back to contiguous.
const BATCHING_PARITY_MARGIN: f64 = 0.97;

/// P9 `tune --batching` A/B: measure SINGLE-STREAM (one request, no
/// concurrency) prefill+decode throughput on the continuous-batching
/// `PagedBatchEngine` vs the contiguous `ContiguousEngine`, and decide whether
/// default-on continuous batching regresses the single-user case.
///
/// Both engines run the SAME synthetic prompt (shifted per repeat so neither
/// the contiguous prefix cache nor the paged APC short-circuits the prefill)
/// under greedy sampling, and each request's full wall time (prefill + decode)
/// is measured; the per-engine score is generated-tokens ÷ wall-seconds,
/// median over `repeats`. The winner is `"paged"` when it's parity-or-faster
/// (see [`BATCHING_PARITY_MARGIN`]), else `"contiguous"`.
pub fn measure_batching_candidates(
    model_path: &Path,
    ctx_size: usize,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
    cfg: &MeasurementConfig,
) -> crate::Result<BatchingMeasurementReport> {
    use crate::paged_batch::PagedBatchEngine;

    let mut total_load_ms = 0.0f64;
    let mut candidates: Vec<BatchingEngineResult> = Vec::with_capacity(2);

    // Greedy, deterministic sampling. No string stops / grammar — this
    // measures the raw forward throughput, which is what both engines share.
    let sampling = |n: u32| SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        typical_p: 1.0,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 0,
        max_tokens: n,
        stop: Vec::new(),
        ..SamplingParams::default()
    };
    // A prompt of `prompt_tokens` ids, shifted by `salt` so a different call
    // never content-matches a prior one (dodges prefix-cache / APC reuse).
    let make_prompt = |vocab: i32, salt: i64| -> Vec<i32> {
        let v = vocab.max(2) as i64;
        (0..prompt_tokens as i64)
            .map(|i| (1 + (i + salt).rem_euclid(v - 1)) as i32)
            .collect()
    };
    let median_max = |mut runs: Vec<f64>| -> (Option<f64>, Option<f64>) {
        if runs.is_empty() {
            return (None, None);
        }
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if runs.len() % 2 == 1 {
            runs[runs.len() / 2]
        } else {
            (runs[runs.len() / 2 - 1] + runs[runs.len() / 2]) / 2.0
        };
        (Some(median), Some(runs[runs.len() - 1]))
    };

    // ---- Contiguous ContiguousEngine (the single-stream reference) ----
    let contiguous_tps = {
        let load_start = std::time::Instant::now();
        let loaded = ContiguousEngine::load_with_options_and_layout(
            model_path,
            ctx_size,
            true,
            cfg.kv_dtype,
            "contiguous",
        );
        total_load_ms += load_start.elapsed().as_secs_f64() * 1000.0;
        match loaded {
            Err(e) => {
                candidates.push(BatchingEngineResult {
                    engine: "contiguous".into(),
                    warmup_ms: 0.0,
                    median_tps: None,
                    max_tps: None,
                    error: Some(format!("load failed: {e}")),
                });
                None
            }
            Ok(mut cpu) => {
                cpu.set_prefix_cache(false);
                cpu.set_flash_attention(cfg.flash_attention);
                apply_measurement_placement(&mut cpu, cfg.n_gpu_layers);
                let vocab = cpu.vocab_size() as i32;
                let warmup_t = std::time::Instant::now();
                let _ = cpu.generate_token_ids(
                    &make_prompt(vocab, -1),
                    decode_tokens,
                    &sampling(decode_tokens),
                );
                let warmup_ms = warmup_t.elapsed().as_secs_f64() * 1000.0;
                let mut runs: Vec<f64> = Vec::new();
                let mut last_err: Option<String> = None;
                for r in 0..repeats {
                    cpu.clear_prefix_cache();
                    let p = make_prompt(vocab, r as i64);
                    let t = std::time::Instant::now();
                    match cpu.generate_token_ids(&p, decode_tokens, &sampling(decode_tokens)) {
                        Ok(out) if !out.is_empty() => {
                            let secs = t.elapsed().as_secs_f64();
                            if secs > 0.0 {
                                runs.push(out.len() as f64 / secs);
                            }
                        }
                        Ok(_) => {}
                        Err(e) => last_err = Some(e.to_string()),
                    }
                }
                let (median, max) = median_max(runs);
                candidates.push(BatchingEngineResult {
                    engine: "contiguous".into(),
                    warmup_ms,
                    median_tps: median,
                    max_tps: max,
                    error: if median.is_none() { last_err } else { None },
                });
                median
            }
        }
    };

    // ---- Paged PagedBatchEngine (the continuous-batching engine), M=1 ----
    let paged_tps = {
        let load_start = std::time::Instant::now();
        let loaded = PagedBatchEngine::load(model_path, ctx_size, 1);
        total_load_ms += load_start.elapsed().as_secs_f64() * 1000.0;
        match loaded {
            Err(e) => {
                candidates.push(BatchingEngineResult {
                    engine: "paged".into(),
                    warmup_ms: 0.0,
                    median_tps: None,
                    max_tps: None,
                    error: Some(format!("load failed: {e}")),
                });
                None
            }
            Ok(paged) => {
                let vocab = paged.vocab_size() as i32;
                let warmup_t = std::time::Instant::now();
                let _ = paged.generate_batched_token_ids(vec![(
                    make_prompt(vocab, -1),
                    sampling(decode_tokens),
                )]);
                let warmup_ms = warmup_t.elapsed().as_secs_f64() * 1000.0;
                let mut runs: Vec<f64> = Vec::new();
                let mut last_err: Option<String> = None;
                for r in 0..repeats {
                    let p = make_prompt(vocab, r as i64);
                    let t = std::time::Instant::now();
                    match paged.generate_batched_token_ids(vec![(p, sampling(decode_tokens))]) {
                        Ok(outs) if outs.first().map(|o| !o.is_empty()).unwrap_or(false) => {
                            let secs = t.elapsed().as_secs_f64();
                            if secs > 0.0 {
                                runs.push(outs[0].len() as f64 / secs);
                            }
                        }
                        Ok(_) => {}
                        Err(e) => last_err = Some(e.to_string()),
                    }
                }
                let (median, max) = median_max(runs);
                candidates.push(BatchingEngineResult {
                    engine: "paged".into(),
                    warmup_ms,
                    median_tps: median,
                    max_tps: max,
                    error: if median.is_none() { last_err } else { None },
                });
                median
            }
        }
    };

    // Decide the winner. Paged wins on parity-or-better (its real payoff is
    // multi-request concurrency, which this single-stream A/B can't see, so a
    // near-tie keeps continuous batching on). A clear single-stream loss flips
    // the serve path back to contiguous.
    let winner = match (paged_tps, contiguous_tps) {
        (Some(p), Some(c)) => {
            if p >= c * BATCHING_PARITY_MARGIN {
                Some("paged".to_string())
            } else {
                Some("contiguous".to_string())
            }
        }
        (Some(_), None) => Some("paged".to_string()),
        (None, Some(_)) => Some("contiguous".to_string()),
        (None, None) => None,
    };

    Ok(BatchingMeasurementReport {
        total_load_ms,
        winner,
        paged_tps,
        contiguous_tps,
        candidates,
    })
}

/// Drive the engine through each batch-size candidate, time the
/// prefill phase of a synthetic prompt, pick the highest-throughput
/// candidate. Decode tokens are pinned to 1 so the measurement
/// isolates prefill (decode is the placement-sweep's territory).
pub fn measure_batch_size_candidates(
    model_path: &Path,
    candidates: &[usize],
    prompt_tokens: u32,
    repeats: u32,
    cfg: &MeasurementConfig,
) -> crate::Result<BatchSizeMeasurementReport> {
    let load_start = std::time::Instant::now();
    // Pick a context size that comfortably holds the synthetic
    // prompt + 1 decode token, regardless of what the user configured.
    let ctx_size = (prompt_tokens as usize + 8).max(64);
    let mut cpu = ContiguousEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        true,
        cfg.kv_dtype,
        &cfg.kv_cache_layout,
    )
    .map_err(|e| crate::EngineError::Engine(format!("load failed: {e}")))?;
    cpu.set_prefix_cache(false);
    cpu.set_flash_attention(cfg.flash_attention);
    apply_measurement_placement(&mut cpu, cfg.n_gpu_layers);
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let vocab = cpu.vocab_size() as i32;
    let prompt_ids: Vec<i32> = (0..prompt_tokens as i32)
        .map(|i| 1 + (i % vocab.max(2).saturating_sub(1)))
        .collect();
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        typical_p: 1.0,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 0,
        max_tokens: 1,
        stop: Vec::new(),
        ..SamplingParams::default()
    };

    let mut report_candidates: Vec<BatchSizeCandidateResult> = Vec::with_capacity(candidates.len());
    let mut best_tps = 0.0f64;
    let mut best_b: Option<usize> = None;

    for &b in candidates {
        cpu.set_prefill_chunk_size(b);
        cpu.clear_prefix_cache();
        let warmup_t = std::time::Instant::now();
        let warmup_result = cpu.generate_token_ids(&prompt_ids, 1, &sampling);
        let warmup_ms = warmup_t.elapsed().as_secs_f64() * 1000.0;
        if let Err(e) = warmup_result {
            report_candidates.push(BatchSizeCandidateResult {
                batch_size: b,
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(format!("warmup failed: {e}")),
            });
            continue;
        }

        let mut runs: Vec<f64> = Vec::with_capacity(repeats as usize);
        let mut last_err: Option<String> = None;
        for _ in 0..repeats {
            cpu.clear_prefix_cache();
            match cpu.generate_token_ids(&prompt_ids, 1, &sampling) {
                Ok(_) => {
                    let stats = cpu.last_request_stats();
                    let pf_ms = stats.prefill_ms;
                    let prefilled = stats.tokens_prefilled.max(prompt_tokens);
                    if pf_ms > 0.0 && prefilled > 0 {
                        runs.push((prefilled as f64) / (pf_ms / 1000.0));
                    }
                }
                Err(e) => last_err = Some(e.to_string()),
            }
        }
        if runs.is_empty() {
            report_candidates.push(BatchSizeCandidateResult {
                batch_size: b,
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(last_err.unwrap_or_else(|| "every measurement run failed".into())),
            });
            continue;
        }
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if runs.len() % 2 == 1 {
            runs[runs.len() / 2]
        } else {
            (runs[runs.len() / 2 - 1] + runs[runs.len() / 2]) / 2.0
        };
        let max = runs[runs.len() - 1];
        report_candidates.push(BatchSizeCandidateResult {
            batch_size: b,
            warmup_ms,
            median_tps: Some(median),
            max_tps: Some(max),
            error: None,
        });
        if median > best_tps {
            best_tps = median;
            best_b = Some(b);
        }
    }

    Ok(BatchSizeMeasurementReport {
        load_ms,
        winner: best_b,
        winner_tps: best_tps,
        candidates: report_candidates,
    })
}

// ============================================================
// E2 autotuner candidates: flash_attention_kv_min, prefix_cache_max_snapshots
// ============================================================
//
// Both share the existing single-load-many-candidates pattern: load
// the engine once, mutate the relevant runtime knob between
// candidates, time decode tok/s, pick the highest median.

/// Per-candidate result from a `flash_attention_kv_min` threshold
/// sweep. The threshold gates flash-decode vs standard attention;
/// the break-even crossover is host-specific (AVX-2 CPU, AVX-512 CPU,
/// Iris Xe iGPU, Arc dGPU all sit at different sweet spots).
#[derive(Debug, Clone)]
pub struct FlashKvMinCandidateResult {
    pub kv_min: u32,
    pub warmup_ms: f64,
    pub median_tps: Option<f64>,
    pub max_tps: Option<f64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FlashKvMinMeasurementReport {
    pub load_ms: f64,
    pub winner: Option<u32>,
    pub winner_tps: f64,
    pub candidates: Vec<FlashKvMinCandidateResult>,
}

/// Sweep `flash_attention_kv_min` across candidate thresholds. The
/// kernel-dispatch site at `llama_arch.rs` reads
/// `RUSTLLAMA_FLASH_KV_LEN_MIN` from the environment; we set it per
/// candidate, run a synthetic decode pass, and pick the winner.
///
/// The decode length must EXCEED the candidate threshold for the
/// threshold to actually affect dispatch (otherwise both flash and
/// non-flash paths reach the same code). The caller should pick
/// `decode_tokens` accordingly — at least 2× the largest candidate
/// is a safe default.
pub fn measure_flash_kv_min_candidates(
    model_path: &Path,
    candidates: &[u32],
    ctx_size: usize,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
    cfg: &MeasurementConfig,
) -> crate::Result<FlashKvMinMeasurementReport> {
    let load_start = std::time::Instant::now();
    let mut cpu = ContiguousEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        true,
        cfg.kv_dtype,
        &cfg.kv_cache_layout,
    )
    .map_err(|e| crate::EngineError::Engine(format!("load failed: {e}")))?;
    cpu.set_prefix_cache(false);
    cpu.set_flash_attention(cfg.flash_attention);
    apply_measurement_placement(&mut cpu, cfg.n_gpu_layers);
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let vocab = cpu.vocab_size() as i32;
    let prompt_ids: Vec<i32> = (0..prompt_tokens as i32)
        .map(|i| 1 + (i % vocab.max(2).saturating_sub(1)))
        .collect();
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        typical_p: 1.0,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 0,
        max_tokens: decode_tokens,
        stop: Vec::new(),
        ..SamplingParams::default()
    };

    let saved_env = std::env::var_os("RUSTLLAMA_FLASH_KV_LEN_MIN");
    let mut report_candidates: Vec<FlashKvMinCandidateResult> =
        Vec::with_capacity(candidates.len());
    let mut best_tps = 0.0f64;
    let mut best_kv_min: Option<u32> = None;

    for &kv_min in candidates {
        // Mutate the env var the kernel-dispatch site reads. Restored
        // after the sweep so we don't leak state to the rest of the
        // process.
        std::env::set_var("RUSTLLAMA_FLASH_KV_LEN_MIN", kv_min.to_string());
        cpu.clear_prefix_cache();
        let warmup_t = std::time::Instant::now();
        let warmup_result = cpu.generate_token_ids(&prompt_ids, decode_tokens, &sampling);
        let warmup_ms = warmup_t.elapsed().as_secs_f64() * 1000.0;
        if let Err(e) = warmup_result {
            report_candidates.push(FlashKvMinCandidateResult {
                kv_min,
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(format!("warmup failed: {e}")),
            });
            continue;
        }

        let (mut runs, last_err) =
            collect_decode_tok_s(&mut cpu, &prompt_ids, decode_tokens, repeats, &sampling);
        if runs.is_empty() {
            report_candidates.push(FlashKvMinCandidateResult {
                kv_min,
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(last_err.unwrap_or_else(|| "every measurement run failed".into())),
            });
            continue;
        }
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if runs.len() % 2 == 1 {
            runs[runs.len() / 2]
        } else {
            (runs[runs.len() / 2 - 1] + runs[runs.len() / 2]) / 2.0
        };
        let max = runs[runs.len() - 1];
        report_candidates.push(FlashKvMinCandidateResult {
            kv_min,
            warmup_ms,
            median_tps: Some(median),
            max_tps: Some(max),
            error: None,
        });
        if median > best_tps {
            best_tps = median;
            best_kv_min = Some(kv_min);
        }
    }

    // Restore env var to pre-sweep state.
    match saved_env {
        Some(v) => std::env::set_var("RUSTLLAMA_FLASH_KV_LEN_MIN", v),
        None => std::env::remove_var("RUSTLLAMA_FLASH_KV_LEN_MIN"),
    }

    Ok(FlashKvMinMeasurementReport {
        load_ms,
        winner: best_kv_min,
        winner_tps: best_tps,
        candidates: report_candidates,
    })
}

/// Per-candidate result from a `prefix_cache_max_snapshots` sweep.
/// The pool depth controls how many distinct prompt prefixes the
/// engine keeps warm across requests; deeper pools help interactive
/// chat with many concurrent conversations, but cost memory.
#[derive(Debug, Clone)]
pub struct PrefixSnapshotsCandidateResult {
    pub max_snapshots: u32,
    pub warmup_ms: f64,
    pub median_tps: Option<f64>,
    pub max_tps: Option<f64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PrefixSnapshotsMeasurementReport {
    pub load_ms: f64,
    pub winner: Option<u32>,
    pub winner_tps: f64,
    pub candidates: Vec<PrefixSnapshotsCandidateResult>,
}

/// Sweep `prefix_cache_max_snapshots` — measures decode tok/s across
/// pool depths. Pool semantics: at depth N, the engine retains up
/// to N most-recently-used prompt-prefix snapshots for LCP reuse on
/// future requests. The optimal depth depends on conversation
/// pattern: single-turn batch favors 1; interactive chat with
/// multiple long-lived threads favors 4-8.
///
/// For the synthetic sweep we drive a single conversation, so the
/// measured signal here is the pool-management overhead (which
/// caps the per-request gain). Real-world workload measurement is
/// up to the user; this sweep validates "no regression from
/// changing the depth".
pub fn measure_prefix_snapshots_candidates(
    model_path: &Path,
    candidates: &[u32],
    ctx_size: usize,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
    cfg: &MeasurementConfig,
) -> crate::Result<PrefixSnapshotsMeasurementReport> {
    let load_start = std::time::Instant::now();
    let mut cpu = ContiguousEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        true,
        cfg.kv_dtype,
        &cfg.kv_cache_layout,
    )
    .map_err(|e| crate::EngineError::Engine(format!("load failed: {e}")))?;
    cpu.set_prefix_cache(true);
    cpu.set_flash_attention(cfg.flash_attention);
    apply_measurement_placement(&mut cpu, cfg.n_gpu_layers);
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let vocab = cpu.vocab_size() as i32;
    let prompt_ids: Vec<i32> = (0..prompt_tokens as i32)
        .map(|i| 1 + (i % vocab.max(2).saturating_sub(1)))
        .collect();
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        typical_p: 1.0,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 0,
        max_tokens: decode_tokens,
        stop: Vec::new(),
        ..SamplingParams::default()
    };

    let mut report_candidates: Vec<PrefixSnapshotsCandidateResult> =
        Vec::with_capacity(candidates.len());
    let mut best_tps = 0.0f64;
    let mut best_depth: Option<u32> = None;

    for &depth in candidates {
        cpu.set_prefix_cache_max_snapshots(depth as usize);
        cpu.clear_prefix_cache();
        let warmup_t = std::time::Instant::now();
        let warmup_result = cpu.generate_token_ids(&prompt_ids, decode_tokens, &sampling);
        let warmup_ms = warmup_t.elapsed().as_secs_f64() * 1000.0;
        if let Err(e) = warmup_result {
            report_candidates.push(PrefixSnapshotsCandidateResult {
                max_snapshots: depth,
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(format!("warmup failed: {e}")),
            });
            continue;
        }

        let mut runs: Vec<f64> = Vec::with_capacity(repeats as usize);
        let mut last_err: Option<String> = None;
        for _ in 0..repeats {
            // Don't clear prefix cache between repeats — that's the
            // whole point of measuring pool depth.
            match cpu.generate_token_ids(&prompt_ids, decode_tokens, &sampling) {
                Ok(_) => {
                    let stats = cpu.last_request_stats();
                    if stats.decode_ms > 0.0 && stats.tokens_generated > 0 {
                        runs.push((stats.tokens_generated as f64) / (stats.decode_ms / 1000.0));
                    }
                }
                Err(e) => last_err = Some(e.to_string()),
            }
        }
        if runs.is_empty() {
            report_candidates.push(PrefixSnapshotsCandidateResult {
                max_snapshots: depth,
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(last_err.unwrap_or_else(|| "every measurement run failed".into())),
            });
            continue;
        }
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if runs.len() % 2 == 1 {
            runs[runs.len() / 2]
        } else {
            (runs[runs.len() / 2 - 1] + runs[runs.len() / 2]) / 2.0
        };
        let max = runs[runs.len() - 1];
        report_candidates.push(PrefixSnapshotsCandidateResult {
            max_snapshots: depth,
            warmup_ms,
            median_tps: Some(median),
            max_tps: Some(max),
            error: None,
        });
        if median > best_tps {
            best_tps = median;
            best_depth = Some(depth);
        }
    }

    Ok(PrefixSnapshotsMeasurementReport {
        load_ms,
        winner: best_depth,
        winner_tps: best_tps,
        candidates: report_candidates,
    })
}

/// Per-candidate result from a `kv_page_size` sweep. The page size
/// only affects the paged-KV layout; on contiguous configs every
/// candidate measures the same path and the sweep is a no-op.
#[derive(Debug, Clone)]
pub struct KvPageSizeCandidateResult {
    pub page_size: u32,
    pub warmup_ms: f64,
    pub median_tps: Option<f64>,
    pub max_tps: Option<f64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct KvPageSizeMeasurementReport {
    pub total_load_ms: f64,
    pub winner: Option<u32>,
    pub winner_tps: f64,
    pub candidates: Vec<KvPageSizeCandidateResult>,
}

/// Sweep `kv_page_size` across candidate token-counts-per-page.
/// Reloads the engine per candidate via
/// [`ContiguousEngine::load_with_options_layout_and_page_size`] — page size
/// is structural (changes `PagedKvStore` geometry) so we can't
/// mutate it on a loaded engine. Caller should run this only when
/// `kv_cache_layout = "paged"` is in effect; on contiguous configs
/// every candidate measures the same path and the sweep emits a
/// degenerate winner.
pub fn measure_kv_page_size_candidates(
    model_path: &Path,
    candidates: &[u32],
    ctx_size: usize,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
    cfg: &MeasurementConfig,
) -> crate::Result<KvPageSizeMeasurementReport> {
    let mut report_candidates: Vec<KvPageSizeCandidateResult> =
        Vec::with_capacity(candidates.len());
    let mut best_tps = 0.0f64;
    let mut best_page: Option<u32> = None;
    let mut total_load_ms = 0.0f64;

    for &page_size in candidates {
        let load_start = std::time::Instant::now();
        let load_result = ContiguousEngine::load_with_options_layout_and_page_size(
            model_path,
            ctx_size,
            true,
            cfg.kv_dtype,
            &cfg.kv_cache_layout,
            page_size,
        );
        total_load_ms += load_start.elapsed().as_secs_f64() * 1000.0;
        let mut cpu = match load_result {
            Ok(c) => c,
            Err(e) => {
                report_candidates.push(KvPageSizeCandidateResult {
                    page_size,
                    warmup_ms: 0.0,
                    median_tps: None,
                    max_tps: None,
                    error: Some(format!("load failed: {e}")),
                });
                continue;
            }
        };
        cpu.set_prefix_cache(false);
        cpu.set_flash_attention(cfg.flash_attention);
        apply_measurement_placement(&mut cpu, cfg.n_gpu_layers);

        let vocab = cpu.vocab_size() as i32;
        let prompt_ids: Vec<i32> = (0..prompt_tokens as i32)
            .map(|i| 1 + (i % vocab.max(2).saturating_sub(1)))
            .collect();
        let sampling = SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            typical_p: 1.0,
            repeat_penalty: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: 0,
            max_tokens: decode_tokens,
            stop: Vec::new(),
            ..SamplingParams::default()
        };

        cpu.clear_prefix_cache();
        let warmup_t = std::time::Instant::now();
        let warmup_result = cpu.generate_token_ids(&prompt_ids, decode_tokens, &sampling);
        let warmup_ms = warmup_t.elapsed().as_secs_f64() * 1000.0;
        if let Err(e) = warmup_result {
            report_candidates.push(KvPageSizeCandidateResult {
                page_size,
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(format!("warmup failed: {e}")),
            });
            continue;
        }

        let (mut runs, last_err) =
            collect_decode_tok_s(&mut cpu, &prompt_ids, decode_tokens, repeats, &sampling);
        if runs.is_empty() {
            report_candidates.push(KvPageSizeCandidateResult {
                page_size,
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(last_err.unwrap_or_else(|| "every measurement run failed".into())),
            });
            continue;
        }
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if runs.len() % 2 == 1 {
            runs[runs.len() / 2]
        } else {
            (runs[runs.len() / 2 - 1] + runs[runs.len() / 2]) / 2.0
        };
        let max = runs[runs.len() - 1];
        report_candidates.push(KvPageSizeCandidateResult {
            page_size,
            warmup_ms,
            median_tps: Some(median),
            max_tps: Some(max),
            error: None,
        });
        if median > best_tps {
            best_tps = median;
            best_page = Some(page_size);
        }
    }

    Ok(KvPageSizeMeasurementReport {
        total_load_ms,
        winner: best_page,
        winner_tps: best_tps,
        candidates: report_candidates,
    })
}

/// Per-candidate result from a `flash_v3_kv_tile` sweep. Affects
/// only the SYCL flash-attn-v3 decode kernel; CPU and host-pointer
/// SYCL paths are unaffected. On mock-feature builds (no SYCL)
/// every candidate measures the same CPU code path and the sweep
/// emits a degenerate winner.
#[derive(Debug, Clone)]
pub struct FlashV3KvTileCandidateResult {
    pub kv_tile: u32,
    pub warmup_ms: f64,
    pub median_tps: Option<f64>,
    pub max_tps: Option<f64>,
    pub error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct FlashV3KvTileMeasurementReport {
    pub load_ms: f64,
    pub winner: Option<u32>,
    pub winner_tps: f64,
    pub candidates: Vec<FlashV3KvTileCandidateResult>,
}

/// Sweep `KV_TILE` across `{16, 32, 64}` for the SYCL flash-attn-v3
/// decode kernel. The kernel reads `RUSTLLAMA_FLASH_V3_KV_TILE`
/// at dispatch time; we set it per candidate and time decode tok/s.
/// Restores the env var to its pre-sweep state on exit.
pub fn measure_flash_v3_kv_tile_candidates(
    model_path: &Path,
    candidates: &[u32],
    ctx_size: usize,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
    cfg: &MeasurementConfig,
) -> crate::Result<FlashV3KvTileMeasurementReport> {
    let load_start = std::time::Instant::now();
    let mut cpu = ContiguousEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        true,
        cfg.kv_dtype,
        &cfg.kv_cache_layout,
    )
    .map_err(|e| crate::EngineError::Engine(format!("load failed: {e}")))?;
    cpu.set_prefix_cache(false);
    cpu.set_flash_attention(cfg.flash_attention);
    apply_measurement_placement(&mut cpu, cfg.n_gpu_layers);
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let vocab = cpu.vocab_size() as i32;
    let prompt_ids: Vec<i32> = (0..prompt_tokens as i32)
        .map(|i| 1 + (i % vocab.max(2).saturating_sub(1)))
        .collect();
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        typical_p: 1.0,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 0,
        max_tokens: decode_tokens,
        stop: Vec::new(),
        ..SamplingParams::default()
    };

    let saved_env = std::env::var_os("RUSTLLAMA_FLASH_V3_KV_TILE");
    let mut report_candidates: Vec<FlashV3KvTileCandidateResult> =
        Vec::with_capacity(candidates.len());
    let mut best_tps = 0.0f64;
    let mut best_tile: Option<u32> = None;

    for &kv_tile in candidates {
        std::env::set_var("RUSTLLAMA_FLASH_V3_KV_TILE", kv_tile.to_string());
        cpu.clear_prefix_cache();
        let warmup_t = std::time::Instant::now();
        let warmup_result = cpu.generate_token_ids(&prompt_ids, decode_tokens, &sampling);
        let warmup_ms = warmup_t.elapsed().as_secs_f64() * 1000.0;
        if let Err(e) = warmup_result {
            report_candidates.push(FlashV3KvTileCandidateResult {
                kv_tile,
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(format!("warmup failed: {e}")),
            });
            continue;
        }

        let (mut runs, last_err) =
            collect_decode_tok_s(&mut cpu, &prompt_ids, decode_tokens, repeats, &sampling);
        if runs.is_empty() {
            report_candidates.push(FlashV3KvTileCandidateResult {
                kv_tile,
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(last_err.unwrap_or_else(|| "every measurement run failed".into())),
            });
            continue;
        }
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if runs.len() % 2 == 1 {
            runs[runs.len() / 2]
        } else {
            (runs[runs.len() / 2 - 1] + runs[runs.len() / 2]) / 2.0
        };
        let max = runs[runs.len() - 1];
        report_candidates.push(FlashV3KvTileCandidateResult {
            kv_tile,
            warmup_ms,
            median_tps: Some(median),
            max_tps: Some(max),
            error: None,
        });
        if median > best_tps {
            best_tps = median;
            best_tile = Some(kv_tile);
        }
    }

    match saved_env {
        Some(v) => std::env::set_var("RUSTLLAMA_FLASH_V3_KV_TILE", v),
        None => std::env::remove_var("RUSTLLAMA_FLASH_V3_KV_TILE"),
    }

    Ok(FlashV3KvTileMeasurementReport {
        load_ms,
        winner: best_tile,
        winner_tps: best_tps,
        candidates: report_candidates,
    })
}

/// Per-candidate result from a MoE-placement sweep. A candidate is a
/// (placement mode, q★ GPU split) pair; `median_tps` is decode tok/s.
#[derive(Debug, Clone)]
pub struct MoePlacementCandidateResult {
    /// Human label: `uniform`, `experts_cpu`, `split_250`, ...
    pub label: String,
    pub experts_cpu: bool,
    pub gpu_split_permille: u32,
    pub warmup_ms: f64,
    pub median_tps: Option<f64>,
    pub max_tps: Option<f64>,
    pub error: Option<String>,
}

/// Outcome of [`measure_moe_placement_candidates`]. `winner` is
/// `(placement, gpu_split_permille)` — the tuner-cache shape.
#[derive(Debug, Clone)]
pub struct MoePlacementMeasurementReport {
    pub load_ms: f64,
    pub winner: Option<(String, u32)>,
    pub winner_tps: f64,
    /// The `uniform` baseline's median tok/s (the do-nothing option
    /// every alternative must beat by [`MOE_PLACEMENT_MIN_WIN`]).
    pub baseline_tps: f64,
    pub candidates: Vec<MoePlacementCandidateResult>,
}

/// Minimum relative decode-throughput win a non-uniform placement /
/// co-execution candidate needs over the uniform baseline to be
/// declared winner (the roadmap's ">5% median decode win" gate —
/// below that, the added dispatch complexity isn't worth run-to-run
/// variance).
pub const MOE_PLACEMENT_MIN_WIN: f64 = 1.05;

/// A/B the MoE placement modes and q★ co-execution splits by measured
/// decode throughput (FreeToken's bandwidth-adaptive policy, resolved
/// end-to-end rather than by the analytic `m·BP/BH` formula — the
/// measured decode rate folds transfer, dispatch overhead, and
/// memory-bus contention into one honest number; on shared-DRAM
/// iGPUs `uniform`/`experts_cpu` beating every split is an expected
/// and valid outcome). Refuses non-MoE models.
///
/// Loads the model once; toggles placement via the process-global
/// accel knobs between candidates; restores `uniform` + split 0
/// before returning.
pub fn measure_moe_placement_candidates(
    model_path: &Path,
    ctx_size: usize,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
    cfg: &MeasurementConfig,
) -> crate::Result<MoePlacementMeasurementReport> {
    use rustllama_models::accel;

    let load_start = std::time::Instant::now();
    let mut cpu = ContiguousEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        true,
        cfg.kv_dtype,
        &cfg.kv_cache_layout,
    )
    .map_err(|e| crate::EngineError::Engine(format!("load failed: {e}")))?;
    if !cpu.is_moe_model() {
        return Err(crate::EngineError::Engine(
            "moe-placement sweep needs a MoE model (no routed experts in this GGUF)".into(),
        ));
    }
    cpu.set_prefix_cache(false);
    cpu.set_flash_attention(cfg.flash_attention);
    apply_measurement_placement(&mut cpu, cfg.n_gpu_layers);
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

    let vocab = cpu.vocab_size() as i32;
    let prompt_ids: Vec<i32> = (0..prompt_tokens as i32)
        .map(|i| 1 + (i % vocab.max(2).saturating_sub(1)))
        .collect();
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        typical_p: 1.0,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 0,
        max_tokens: decode_tokens,
        stop: Vec::new(),
        ..SamplingParams::default()
    };

    // (label, experts_cpu, split‰). Uniform first — it's the baseline
    // the win gate compares against.
    let candidates: &[(&str, bool, u32)] = &[
        ("uniform", false, 0),
        ("experts_cpu", true, 0),
        ("split_250", false, 250),
        ("split_500", false, 500),
    ];

    let mut report_candidates: Vec<MoePlacementCandidateResult> = Vec::new();
    let mut baseline_tps = 0.0f64;
    let mut best: Option<(String, bool, u32, f64)> = None;

    // Restore whatever placement the process started with — the
    // sweep must not leave its last candidate applied.
    let saved_experts_cpu = accel::moe_experts_cpu_enabled();
    let saved_split = accel::moe_gpu_split_permille();

    for (label, experts_cpu, split) in candidates {
        accel::set_moe_experts_cpu(*experts_cpu);
        accel::set_moe_gpu_split_permille(*split);
        cpu.clear_prefix_cache();
        let warmup_t = std::time::Instant::now();
        let warmup_result = cpu.generate_token_ids(&prompt_ids, decode_tokens, &sampling);
        let warmup_ms = warmup_t.elapsed().as_secs_f64() * 1000.0;
        if let Err(e) = warmup_result {
            report_candidates.push(MoePlacementCandidateResult {
                label: label.to_string(),
                experts_cpu: *experts_cpu,
                gpu_split_permille: *split,
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(format!("warmup failed: {e}")),
            });
            continue;
        }
        let (mut runs, last_err) =
            collect_decode_tok_s(&mut cpu, &prompt_ids, decode_tokens, repeats, &sampling);
        if runs.is_empty() {
            report_candidates.push(MoePlacementCandidateResult {
                label: label.to_string(),
                experts_cpu: *experts_cpu,
                gpu_split_permille: *split,
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(last_err.unwrap_or_else(|| "every measurement run failed".into())),
            });
            continue;
        }
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if runs.len() % 2 == 1 {
            runs[runs.len() / 2]
        } else {
            (runs[runs.len() / 2 - 1] + runs[runs.len() / 2]) / 2.0
        };
        let max = runs[runs.len() - 1];
        if *label == "uniform" {
            baseline_tps = median;
        }
        report_candidates.push(MoePlacementCandidateResult {
            label: label.to_string(),
            experts_cpu: *experts_cpu,
            gpu_split_permille: *split,
            warmup_ms,
            median_tps: Some(median),
            max_tps: Some(max),
            error: None,
        });
        if best
            .as_ref()
            .map(|(_, _, _, t)| median > *t)
            .unwrap_or(true)
        {
            best = Some((label.to_string(), *experts_cpu, *split, median));
        }
    }

    accel::set_moe_experts_cpu(saved_experts_cpu);
    accel::set_moe_gpu_split_permille(saved_split);

    // Win gate: a non-uniform candidate must beat uniform by ≥5%
    // median, else the winner is uniform (split 0).
    let winner = match best {
        None => None,
        Some((label, experts_cpu, split, tps)) => {
            if label != "uniform" && tps < baseline_tps * MOE_PLACEMENT_MIN_WIN {
                Some((("uniform".to_string(), 0), baseline_tps))
            } else {
                let placement = if experts_cpu {
                    "experts_cpu"
                } else {
                    "uniform"
                };
                Some(((placement.to_string(), split), tps))
            }
        }
    };
    let (winner, winner_tps) = match winner {
        Some((w, t)) => (Some(w), t),
        None => (None, 0.0),
    };

    Ok(MoePlacementMeasurementReport {
        load_ms,
        winner,
        winner_tps,
        baseline_tps,
        candidates: report_candidates,
    })
}

// ============================================================
// MTP self-speculation + chunked-SSM-prefill autotune sweeps
// ============================================================
//
// Both are simple A/B toggles (not multi-candidate grids), so they
// return a two-arm report rather than the `{ <axis>, ..., median_tps }`
// candidate-vec convention. Each loads the model ONCE and flips the
// relevant knob between arms with NO reload:
//   - MTP is a hot engine flag (`set_mtp_speculative`);
//   - chunked SSM prefill is gated by an env var the model reads
//     per-forward (`RUSTLLAMA_SSM_PREFILL_CHUNKED`).

/// Outcome of [`measure_speculative_mtp`].
///
/// When `capable` is `false` the loaded model carries no NextN head
/// (not hybrid, or the head is absent), so MTP self-speculation is a
/// silent no-op — `off_tps` / `on_tps` / `winner` stay `None` and the
/// CLI skips the axis (nothing to tune). When `capable` is `true`,
/// `off_tps` / `on_tps` are the median DECODE tok/s with MTP disabled /
/// enabled on the SAME loaded engine, and `winner = Some(on_tps >
/// off_tps)` (`true` ⇒ enable MTP). `error` carries the first arm's
/// failure message when a median is `None`.
#[derive(Debug, Clone, Default)]
pub struct SpecMtpReport {
    pub capable: bool,
    pub off_tps: Option<f64>,
    pub on_tps: Option<f64>,
    pub winner: Option<bool>,
    pub error: Option<String>,
}

/// Outcome of [`measure_ssm_prefill_chunked`].
///
/// `off_prefill_tps` / `on_prefill_tps` are the median PREFILL tok/s
/// with the chunked-parallel SSM (DeltaNet) prefill path disabled /
/// enabled, measured on the SAME loaded engine (the model reads
/// `RUSTLLAMA_SSM_PREFILL_CHUNKED` per-forward, so the toggle needs no
/// reload). `winner = Some(on > off)` (`true` ⇒ enable the chunked
/// path). `error` carries the first arm's failure message when a
/// median is `None`.
#[derive(Debug, Clone, Default)]
pub struct SsmPrefillChunkedReport {
    pub off_prefill_tps: Option<f64>,
    pub on_prefill_tps: Option<f64>,
    pub winner: Option<bool>,
    pub error: Option<String>,
}

/// A/B MTP (NextN) self-speculative decode by measured DECODE tok/s.
///
/// Loads the model ONCE (same load call the decode sweeps use). MTP
/// self-speculation is a HOT toggle ([`ContiguousEngine::set_mtp_speculative`]),
/// so both arms run on that single loaded engine with NO reload between
/// them.
///
/// Capability gate: [`ContiguousEngine::mtp_speculative`] returns
/// `mtp_spec && model_supports_mtp()` (hybrid + NextN head present), so
/// enabling the flag and reading the effective getter back tells us
/// whether the loaded model can actually use MTP — no reload required,
/// since the capability is a property of the already-loaded weights.
/// When not capable we return `SpecMtpReport { capable: false, .. }`
/// with no measurements and the CLI skips the axis.
///
/// Deterministic (temp=0) synthetic decode, mirroring the decode
/// sweeps' quick-by-default prompt/token sizing (`tune`'s
/// `--prompt-tokens=64` / `--decode-tokens=16` defaults).
///
/// The two arms exercise the paths they name so the tok/s delta is real:
/// the OFF arm drives `generate_token_ids` (the classic single-token
/// decode loop, which never routes through MTP), while the ON arm drives
/// the actual MTP self-speculation stream
/// (`speculate_mtp_stream_from_ids`). The MTP flag does NOT gate either
/// call — `generate_token_ids` is always classic and the stream is the
/// explicit MTP entry — so `set_mtp_speculative` toggling is irrelevant
/// to what's measured; only the capability probe below needs it.
///
/// Comparability: both arms report a DECODE-only tok/s that excludes
/// prefill. OFF uses the engine's forward-only `decode_ms`; ON times
/// wall-clock from the first yielded token to the last
/// (`(generated - 1) / (t_last - t_first)`), which excludes prefill but
/// includes MTP's verify + sampling overhead — a conservative (against
/// ON) estimate, so a declared MTP win is a genuine one.
pub fn measure_speculative_mtp(
    model_path: &std::path::Path,
    cfg: &MeasurementConfig,
    repeats: usize,
) -> Result<SpecMtpReport, String> {
    // Mirror the decode sweeps' quick-by-default sizing. ctx holds
    // prompt + decode + slack (see `measure_end_to_end_tok_s` in the
    // CLI). Decode tok/s stabilizes within a few tokens, so 16 decode
    // tokens ranks the MTP on/off arms reliably at a fraction of the
    // wall-time of the old 32.
    const PROMPT_TOKENS: u32 = 64;
    const DECODE_TOKENS: u32 = 16;
    let ctx_size = (PROMPT_TOKENS as usize + DECODE_TOKENS as usize + 8).max(128);

    let mut cpu = ContiguousEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        true,
        cfg.kv_dtype,
        &cfg.kv_cache_layout,
    )
    .map_err(|e| format!("load failed: {e}"))?;
    cpu.set_prefix_cache(false);
    cpu.set_flash_attention(cfg.flash_attention);
    apply_measurement_placement(&mut cpu, cfg.n_gpu_layers);

    // Capability probe: enable the flag, then read the EFFECTIVE getter
    // (`mtp_spec && model_supports_mtp()`). No reload needed — the flag
    // is a plain field and the capability is a property of the loaded
    // model. Not capable ⇒ report it and let the CLI skip the axis.
    cpu.set_mtp_speculative(true);
    let capable = cpu.mtp_speculative();
    if !capable {
        cpu.set_mtp_speculative(false);
        return Ok(SpecMtpReport {
            capable: false,
            ..Default::default()
        });
    }

    let vocab = cpu.vocab_size() as i32;
    let prompt_ids: Vec<i32> = (0..PROMPT_TOKENS as i32)
        .map(|i| 1 + (i % vocab.max(2).saturating_sub(1)))
        .collect();
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        typical_p: 1.0,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 0,
        max_tokens: DECODE_TOKENS,
        stop: Vec::new(),
        ..SamplingParams::default()
    };

    // Median CLASSIC decode tok/s (the OFF arm): `generate_token_ids` is
    // the single-token decode loop and never routes through MTP, so this
    // measures the classic path regardless of the `mtp_speculative` flag.
    // One untimed warmup + `repeats` timed runs; returns `(None,
    // Some(err))` when every run failed. Mirrors the decode-metric pattern
    // used by `measure_placement_candidates` / `measure_per_device_perf`.
    let bench = |cpu: &mut ContiguousEngine| -> (Option<f64>, Option<String>) {
        cpu.clear_prefix_cache();
        if let Err(e) = cpu.generate_token_ids(&prompt_ids, DECODE_TOKENS, &sampling) {
            return (None, Some(format!("warmup failed: {e}")));
        }
        let mut runs: Vec<f64> = Vec::with_capacity(repeats);
        let mut last_err: Option<String> = None;
        for _ in 0..repeats {
            cpu.clear_prefix_cache();
            match cpu.generate_token_ids(&prompt_ids, DECODE_TOKENS, &sampling) {
                Ok(_) => {
                    let stats = cpu.last_request_stats();
                    if stats.decode_ms > 0.0 && stats.tokens_generated > 0 {
                        runs.push((stats.tokens_generated as f64) / (stats.decode_ms / 1000.0));
                    }
                }
                Err(e) => last_err = Some(e.to_string()),
            }
        }
        if runs.is_empty() {
            return (
                None,
                Some(last_err.unwrap_or_else(|| "every measurement run failed".into())),
            );
        }
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if runs.len() % 2 == 1 {
            runs[runs.len() / 2]
        } else {
            (runs[runs.len() / 2 - 1] + runs[runs.len() / 2]) / 2.0
        };
        (Some(median), None)
    };

    // Decode wall-clock throughput for the REAL MTP stream: one untimed
    // warmup drain + `repeats` timed drains of
    // `speculate_mtp_stream_from_ids`. The stream is async, so we drive it
    // with `futures::executor::block_on` (runtime-agnostic — the SYCL
    // worker's `run_blocking` returns a plain `tokio::sync::oneshot`
    // receiver that resolves without a tokio runtime context; the same
    // `futures::StreamExt` drain the engine's own streaming callers use).
    // Timing runs from the first yielded token to the last so prefill is
    // excluded — comparable to the OFF arm's decode-only tok/s.
    let bench_mtp = |cpu: &mut ContiguousEngine| -> (Option<f64>, Option<String>) {
        use futures::StreamExt;
        // Untimed warmup: drain one full stream.
        cpu.clear_prefix_cache();
        match cpu.speculate_mtp_stream_from_ids(prompt_ids.clone(), sampling.clone(), None) {
            Ok(mut stream) => {
                let warm = futures::executor::block_on(async {
                    while let Some(item) = stream.next().await {
                        if let Err(e) = item {
                            return Err(e.to_string());
                        }
                    }
                    Ok(())
                });
                if let Err(e) = warm {
                    return (None, Some(format!("warmup failed: {e}")));
                }
            }
            Err(e) => return (None, Some(format!("warmup stream init failed: {e}"))),
        }

        let mut runs: Vec<f64> = Vec::with_capacity(repeats);
        let mut last_err: Option<String> = None;
        for _ in 0..repeats {
            cpu.clear_prefix_cache();
            let mut stream =
                match cpu.speculate_mtp_stream_from_ids(prompt_ids.clone(), sampling.clone(), None)
                {
                    Ok(s) => s,
                    Err(e) => {
                        last_err = Some(e.to_string());
                        continue;
                    }
                };
            let drained = futures::executor::block_on(async {
                let mut generated = 0usize;
                let mut t_first: Option<std::time::Instant> = None;
                let mut t_last = std::time::Instant::now();
                while let Some(item) = stream.next().await {
                    match item {
                        Ok(_) => {
                            let now = std::time::Instant::now();
                            if t_first.is_none() {
                                t_first = Some(now);
                            }
                            t_last = now;
                            generated += 1;
                        }
                        Err(e) => return Err(e.to_string()),
                    }
                }
                Ok((generated, t_first, t_last))
            });
            match drained {
                Ok((generated, Some(t_first), t_last)) if generated >= 2 => {
                    let decode_s = (t_last - t_first).as_secs_f64();
                    if decode_s > 0.0 {
                        runs.push((generated - 1) as f64 / decode_s);
                    }
                }
                // Fewer than 2 tokens (e.g. immediate EOS on the synthetic
                // prompt) leaves no inter-token interval to time — skip.
                Ok(_) => {}
                Err(e) => last_err = Some(e),
            }
        }
        if runs.is_empty() {
            return (
                None,
                Some(last_err.unwrap_or_else(|| "every measurement run failed".into())),
            );
        }
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if runs.len() % 2 == 1 {
            runs[runs.len() / 2]
        } else {
            (runs[runs.len() / 2 - 1] + runs[runs.len() / 2]) / 2.0
        };
        (Some(median), None)
    };

    // OFF arm: classic single-token decode via `generate_token_ids` (never
    // routes through MTP). ON arm: the real MTP self-speculation stream.
    // The `mtp_speculative` flag gates neither call, so both arms
    // genuinely exercise the paths they name.
    cpu.set_mtp_speculative(false);
    let (off_tps, off_err) = bench(&mut cpu);

    let (on_tps, on_err) = bench_mtp(&mut cpu);

    // Restore the engine's default state (classic decode).
    cpu.set_mtp_speculative(false);

    let winner = match (off_tps, on_tps) {
        (Some(off), Some(on)) => Some(on > off),
        _ => None,
    };
    let error = off_err.or(on_err);

    Ok(SpecMtpReport {
        capable: true,
        off_tps,
        on_tps,
        winner,
        error,
    })
}

/// Median of a run list (sorted in place), or `(None, err)` when empty. Shared by
/// the spec-strategy benches.
fn median_of(runs: &mut [f64], last_err: Option<String>) -> (Option<f64>, Option<String>) {
    if runs.is_empty() {
        return (
            None,
            Some(last_err.unwrap_or_else(|| "every measurement run failed".into())),
        );
    }
    runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if runs.len() % 2 == 1 {
        runs[runs.len() / 2]
    } else {
        (runs[runs.len() / 2 - 1] + runs[runs.len() / 2]) / 2.0
    };
    (Some(median), None)
}

/// Whether stage-9b (spec-strategy) progress markers are enabled — opt-in via
/// `RUSTLLAMA_DEBUG_SPEC_STAGE` (any non-empty, non-"0" value). Cached.
fn spec_dbg_enabled() -> bool {
    static EN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EN.get_or_init(|| {
        std::env::var("RUSTLLAMA_DEBUG_SPEC_STAGE")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    })
}

/// Emit a stage-9b progress marker to STDERR (unbuffered → flushed before any
/// subsequent stall) when `RUSTLLAMA_DEBUG_SPEC_STAGE` is set; silent otherwise. On
/// a HANG, the LAST marker with no matching "… done" names the wedged piece — a
/// debugger-free way to bisect which candidate/phase of `measure_spec_strategy`
/// never returns.
macro_rules! spec_dbg {
    ($($arg:tt)*) => {
        if spec_dbg_enabled() {
            eprintln!("[spec-9b] {}", format!($($arg)*));
        }
    };
}

/// Median decode tok/s for a speculative stream, timed first-token→last-token
/// (prefill-excluded). `init` re-initializes the stream per run; it is the only
/// strategy-specific part. One untimed warmup drain + `repeats` timed drains.
/// Mirrors `measure_speculative_mtp`'s `bench_mtp`, generalized over the stream.
/// Per-candidate wall-clock budget for the spec-strategy sweep (stage 9b). A
/// candidate whose warmup/timed decode exceeds this is recorded as "no winner" and
/// the sweep continues — a slow hybrid (CPU DeltaNet recurrence under spec) or a
/// wedged drafter can't stall the whole autotune. Generous vs the ~1 s a healthy
/// 16-token spec decode takes, so it only trips on genuinely-too-slow candidates.
const SPEC_CAND_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20);

fn bench_spec_stream<F>(
    label: &str,
    repeats: usize,
    timeout: std::time::Duration,
    mut init: F,
) -> (Option<f64>, Option<String>)
where
    F: FnMut() -> crate::Result<crate::TokenStream>,
{
    use futures::StreamExt;
    // Warmup — wall-clock BOUNDED so a slow/wedged candidate (e.g. a hybrid whose
    // CPU DeltaNet recurrence under spec runs tens of seconds per measurement) bails
    // and is recorded as "no winner" instead of stalling the whole sweep (stage 9b).
    spec_dbg!("{label}: warmup init stream");
    match init() {
        Ok(mut stream) => {
            spec_dbg!("{label}: warmup draining");
            let deadline = std::time::Instant::now() + timeout;
            let warm = futures::executor::block_on(async {
                while let Some(item) = stream.next().await {
                    if let Err(e) = item {
                        return Err(e.to_string());
                    }
                    if std::time::Instant::now() > deadline {
                        return Err("timeout".to_string());
                    }
                }
                Ok(())
            });
            spec_dbg!("{label}: warmup drained ({warm:?})");
            if let Err(e) = warm {
                return (None, Some(format!("warmup failed: {e}")));
            }
        }
        Err(e) => return (None, Some(format!("warmup stream init failed: {e}"))),
    }
    let mut runs: Vec<f64> = Vec::with_capacity(repeats);
    let mut last_err: Option<String> = None;
    for i in 0..repeats {
        spec_dbg!("{label}: run {i} init stream");
        let mut stream = match init() {
            Ok(s) => s,
            Err(e) => {
                last_err = Some(e.to_string());
                continue;
            }
        };
        spec_dbg!("{label}: run {i} draining");
        let deadline = std::time::Instant::now() + timeout;
        let drained = futures::executor::block_on(async {
            let mut generated = 0usize;
            let mut t_first: Option<std::time::Instant> = None;
            let mut t_last = std::time::Instant::now();
            while let Some(item) = stream.next().await {
                match item {
                    Ok(_) => {
                        let now = std::time::Instant::now();
                        if t_first.is_none() {
                            t_first = Some(now);
                        }
                        t_last = now;
                        generated += 1;
                    }
                    Err(e) => return Err(e.to_string()),
                }
                if std::time::Instant::now() > deadline {
                    return Err("timeout".to_string());
                }
            }
            Ok((generated, t_first, t_last))
        });
        spec_dbg!("{label}: run {i} drained ({drained:?})");
        match drained {
            Ok((generated, Some(t_first), t_last)) if generated >= 2 => {
                let decode_s = (t_last - t_first).as_secs_f64();
                if decode_s > 0.0 {
                    runs.push((generated - 1) as f64 / decode_s);
                }
            }
            Ok(_) => {}
            Err(e) => last_err = Some(e),
        }
    }
    median_of(&mut runs, last_err)
}

/// Outcome of [`measure_spec_strategy`]: one measured decode tok/s per AVAILABLE
/// speculative-decoding strategy. Spec decoding is LOSSLESS (every draft is
/// target-verified), so the CLI picks the fastest via
/// [`rustllama_tuner::pick_spec_strategy`] with NO coherence gate.
#[derive(Debug, Clone, Default)]
pub struct SpecStrategyReport {
    pub candidates: Vec<rustllama_tuner::SpecStrategyMeasurement>,
    pub error: Option<String>,
}

/// Measure decode tok/s for every self-contained speculative-decoding strategy on
/// `model_path`, for selection by `pick_spec_strategy`. Candidates: classic
/// `None`, `Ngram`, `Lookahead` (always available), plus `Mtp` and `Tree` when the
/// model carries a NextN head (`Tree` uses `tree_widths`, the tuned winner). EAGLE
/// and external-draft strategies are omitted — they need a companion model that
/// the autotune subprocess does not load. One model load, then each candidate's
/// own stream is drained + timed; subsumes `measure_speculative_mtp`.
pub fn measure_spec_strategy(
    model_path: &std::path::Path,
    cfg: &MeasurementConfig,
    tree_widths: Option<&[u32]>,
    repeats: usize,
) -> Result<SpecStrategyReport, String> {
    use rustllama_tuner::{SpecStrategy, SpecStrategyMeasurement};

    const PROMPT_TOKENS: u32 = 64;
    const DECODE_TOKENS: u32 = 16;
    let ctx_size = (PROMPT_TOKENS as usize + DECODE_TOKENS as usize + 8).max(128);

    spec_dbg!("load start ({})", model_path.display());
    let mut cpu = ContiguousEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        true,
        cfg.kv_dtype,
        &cfg.kv_cache_layout,
    )
    .map_err(|e| format!("load failed: {e}"))?;
    // Prefix cache ON for the spec sweep — unlike the other measurement commands,
    // which disable it to force a fresh prefill. Speculative decode's verify
    // primitive (`verify_and_commit_speculation` → `spec_prefill_and_forward`)
    // re-presents the whole running sequence each round and relies on live-prefix
    // KV reuse (`prepare_prefix_reuse`) to skip re-prefilling everything but the new
    // tokens — exactly as production serves it (config default `prefix_cache = true`).
    // With it OFF, every round `kv.reset()`s and re-prefills the full growing context
    // (the trace shows `effective=0 seq_len=0` each round), making every spec
    // candidate ~80x slower than the classic `None` baseline (which decodes
    // incrementally via `generate_token_ids` and never re-prefills). That handicap is
    // pure measurement artifact: it made the sweep ALWAYS pick `None`, defeating its
    // purpose. Matching production here lets the candidates be compared on real decode
    // throughput.
    cpu.set_prefix_cache(true);
    cpu.set_flash_attention(cfg.flash_attention);
    apply_measurement_placement(&mut cpu, cfg.n_gpu_layers);
    spec_dbg!("load done");

    // Capability probes (cheap flag flips on the loaded engine, reset after): the
    // Mtp / Tree candidates only apply to a model with a NextN head.
    spec_dbg!("probe mtp/tree capability");
    cpu.set_mtp_speculative(true);
    let mtp_capable = cpu.mtp_speculative();
    cpu.set_mtp_speculative(false);
    let widths_vec: Vec<u32> = tree_widths
        .map(|w| w.to_vec())
        .unwrap_or_else(|| vec![2, 2, 2, 2]);
    cpu.set_tree_speculative(Some(widths_vec.clone()));
    let tree_capable = cpu.tree_speculative();
    cpu.set_tree_speculative(None);
    spec_dbg!("capability: mtp={mtp_capable} tree={tree_capable}");

    let vocab = cpu.vocab_size() as i32;
    let prompt_ids: Vec<i32> = (0..PROMPT_TOKENS as i32)
        .map(|i| 1 + (i % vocab.max(2).saturating_sub(1)))
        .collect();
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        typical_p: 1.0,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 0,
        max_tokens: DECODE_TOKENS,
        stop: Vec::new(),
        ..SamplingParams::default()
    };

    // `None` baseline: classic `generate_token_ids` (&mut; measured FIRST, before
    // the &self stream closures borrow the engine).
    spec_dbg!("baseline None start (warmup + {repeats} runs)");
    let none_tps = {
        cpu.clear_prefix_cache();
        if let Err(e) = cpu.generate_token_ids(&prompt_ids, DECODE_TOKENS, &sampling) {
            (None, Some(format!("warmup failed: {e}")))
        } else {
            let mut runs: Vec<f64> = Vec::with_capacity(repeats);
            let mut last_err: Option<String> = None;
            for _ in 0..repeats {
                cpu.clear_prefix_cache();
                match cpu.generate_token_ids(&prompt_ids, DECODE_TOKENS, &sampling) {
                    Ok(_) => {
                        let st = cpu.last_request_stats();
                        if st.decode_ms > 0.0 && st.tokens_generated > 0 {
                            runs.push((st.tokens_generated as f64) / (st.decode_ms / 1000.0));
                        }
                    }
                    Err(e) => last_err = Some(e.to_string()),
                }
            }
            median_of(&mut runs, last_err)
        }
    };
    spec_dbg!(
        "baseline None done (tps={:?} err={:?})",
        none_tps.0,
        none_tps.1
    );

    // Per-candidate wall-clock timeout, AWARE of this (model, config)'s REAL decode
    // speed rather than a fixed 20s. A large model, a CPU / low `n_gpu_layers`
    // placement, or a slow KV dtype legitimately makes a 16-token spec decode take
    // tens of seconds — far longer than the ~1s a fast GPU decode takes — so a flat
    // `SPEC_CAND_TIMEOUT` would false-"timeout" a slow-but-HEALTHY candidate and drop
    // a valid winner. Instead bound each candidate at a generous multiple of the
    // measured `None` baseline (which already ran above), floored at
    // `SPEC_CAND_TIMEOUT` for fast models and capped so an anomalous baseline can't
    // make it effectively unbounded. A failed/zero baseline ⇒ a conservative
    // fallback. NOTE: a truly WEDGED candidate — one forward that never returns —
    // escapes this stream-level deadline (it is checked only between emitted items),
    // regardless of its value; the CLI tune-stage subprocess backstop is the catch
    // for that case.
    let cand_timeout: std::time::Duration = {
        let cap = std::time::Duration::from_secs(300);
        match none_tps.0.filter(|t| *t > 0.0) {
            Some(tps) => std::time::Duration::from_secs_f64((DECODE_TOKENS as f64 / tps) * 10.0)
                .clamp(SPEC_CAND_TIMEOUT, cap),
            // Baseline failed/unmeasured: assume a slow model and give each candidate
            // a generous fixed budget rather than the tight 20s default.
            None => std::time::Duration::from_secs(120),
        }
    };
    spec_dbg!("per-candidate timeout = {cand_timeout:?}");

    let mut candidates: Vec<SpecStrategyMeasurement> = Vec::new();
    candidates.push(SpecStrategyMeasurement {
        strategy: SpecStrategy::None,
        tok_s: none_tps.0,
        error: none_tps.1,
    });

    // Self-contained drafters (always available).
    let ng = crate::speculative::NgramDrafterConfig::default();
    let r = bench_spec_stream("Ngram", repeats, cand_timeout, || {
        cpu.clear_prefix_cache();
        cpu.speculate_ngram_stream_from_ids(prompt_ids.clone(), ng.clone(), sampling.clone())
    });
    candidates.push(SpecStrategyMeasurement {
        strategy: SpecStrategy::Ngram,
        tok_s: r.0,
        error: r.1,
    });

    let la = crate::speculative::LookaheadDrafterConfig::default();
    let r = bench_spec_stream("Lookahead", repeats, cand_timeout, || {
        cpu.clear_prefix_cache();
        cpu.speculate_lookahead_stream_from_ids(prompt_ids.clone(), la.clone(), sampling.clone())
    });
    candidates.push(SpecStrategyMeasurement {
        strategy: SpecStrategy::Lookahead,
        tok_s: r.0,
        error: r.1,
    });

    // NextN-head drafters.
    if mtp_capable {
        let r = bench_spec_stream("Mtp", repeats, cand_timeout, || {
            cpu.clear_prefix_cache();
            cpu.speculate_mtp_stream_from_ids(prompt_ids.clone(), sampling.clone(), None)
        });
        candidates.push(SpecStrategyMeasurement {
            strategy: SpecStrategy::Mtp,
            tok_s: r.0,
            error: r.1,
        });
    }
    if tree_capable {
        let widths = widths_vec.clone();
        let r = bench_spec_stream("Tree", repeats, cand_timeout, || {
            cpu.clear_prefix_cache();
            cpu.speculate_mtp_stream_from_ids(
                prompt_ids.clone(),
                sampling.clone(),
                Some(widths.clone()),
            )
        });
        candidates.push(SpecStrategyMeasurement {
            strategy: SpecStrategy::Tree,
            tok_s: r.0,
            error: r.1,
        });
    }

    Ok(SpecStrategyReport {
        candidates,
        error: None,
    })
}

/// Outcome of [`measure_tree_widths`].
///
/// `capable` is `false` when the loaded model carries no MTP/NextN draft head,
/// so tree-structured speculation is inert (a silent fall-back to classic
/// decode) — `candidates` is then empty and the CLI SKIPs the axis, writing no
/// winner. This is the write-blind reality on every publicly available GGUF (no
/// small MTP-head model exists), so the sweep must degrade to this SKIP cleanly.
///
/// When `capable` is `true`, `candidates` holds one
/// [`rustllama_tuner::TreeWidthMeasurement`] per swept width vector — its median
/// EFFECTIVE decode tok/s (the objective) and its speculative accept rate
/// (reported). The CLI feeds `candidates` to [`rustllama_tuner::pick_tree_widths`]
/// to choose + persist the winner.
#[derive(Debug, Clone, Default)]
pub struct TreeWidthsReport {
    pub capable: bool,
    pub candidates: Vec<rustllama_tuner::TreeWidthMeasurement>,
    pub error: Option<String>,
}

/// Measure tree-structured MTP speculation decode throughput across a set of
/// per-depth branching-WIDTH candidate vectors, so the tuner can pick the best
/// `[inference].speculative_tree_widths`.
///
/// Loads the model ONCE. Tree spec is driven directly by the `tree_widths`
/// argument to [`ContiguousEngine::speculate_mtp_stream_from_ids`] (NOT a hot engine
/// flag), so each candidate just drains that stream with its own width vector —
/// no reload, no setter. The capability probe uses
/// [`ContiguousEngine::set_tree_speculative`] + [`ContiguousEngine::tree_speculative`]: a model
/// with no draft head reads back `false`, and we return `capable: false` with no
/// measurements (the CLI SKIPs).
///
/// Objective = EFFECTIVE decode tok/s, timed first-yield → last-yield so prefill
/// is excluded (comparable to the MTP sweep's ON arm) and the wall-clock already
/// nets out tree verify + rejected-branch waste — a wider tree that drafts more
/// but accepts less is correctly ranked lower. Accept rate is read from the
/// engine's lifetime speculation counters (delta over the candidate's timed
/// runs) and REPORTED alongside; it does not rank.
///
/// Deterministic (temp=0) synthetic decode, mirroring the MTP sweep's
/// quick-by-default sizing.
pub fn measure_tree_widths(
    model_path: &Path,
    cfg: &MeasurementConfig,
    candidates: &[Vec<u32>],
    repeats: usize,
) -> Result<TreeWidthsReport, String> {
    use futures::StreamExt;

    // Same quick-by-default sizing as `measure_speculative_mtp`.
    const PROMPT_TOKENS: u32 = 64;
    const DECODE_TOKENS: u32 = 16;
    let ctx_size = (PROMPT_TOKENS as usize + DECODE_TOKENS as usize + 8).max(128);

    let mut cpu = ContiguousEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        true,
        cfg.kv_dtype,
        &cfg.kv_cache_layout,
    )
    .map_err(|e| format!("load failed: {e}"))?;
    cpu.set_prefix_cache(false);
    cpu.set_flash_attention(cfg.flash_attention);
    apply_measurement_placement(&mut cpu, cfg.n_gpu_layers);

    // Capability probe: set a trivial width vector and read the EFFECTIVE getter
    // (`tree_spec.is_some() && model_supports_tree()`). No reload needed — the
    // capability is a property of the loaded weights. Not capable ⇒ report it and
    // let the CLI skip the axis (the write-blind SKIP on every public model).
    cpu.set_tree_speculative(Some(vec![2]));
    let capable = cpu.tree_speculative();
    cpu.set_tree_speculative(None);
    if !capable {
        return Ok(TreeWidthsReport {
            capable: false,
            ..Default::default()
        });
    }

    let vocab = cpu.vocab_size() as i32;
    let prompt_ids: Vec<i32> = (0..PROMPT_TOKENS as i32)
        .map(|i| 1 + (i % vocab.max(2).saturating_sub(1)))
        .collect();
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        typical_p: 1.0,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 0,
        max_tokens: DECODE_TOKENS,
        stop: Vec::new(),
        ..SamplingParams::default()
    };

    // Drain the tree-spec stream for one width vector and return the median
    // effective decode tok/s + the accept rate over the timed runs. Mirrors
    // `measure_speculative_mtp`'s `bench_mtp`, parameterized on the widths and
    // additionally sampling the engine's cumulative speculation counters to
    // derive this candidate's accept rate (drafted → accepted).
    let bench_widths = |cpu: &mut ContiguousEngine,
                        widths: &[u32]|
     -> (Option<f64>, Option<f64>, Option<String>) {
        let tw = Some(widths.to_vec());
        // Untimed warmup drain.
        cpu.clear_prefix_cache();
        match cpu.speculate_mtp_stream_from_ids(prompt_ids.clone(), sampling.clone(), tw.clone()) {
            Ok(mut stream) => {
                let warm = futures::executor::block_on(async {
                    while let Some(item) = stream.next().await {
                        if let Err(e) = item {
                            return Err(e.to_string());
                        }
                    }
                    Ok(())
                });
                if let Err(e) = warm {
                    return (None, None, Some(format!("warmup failed: {e}")));
                }
            }
            Err(e) => return (None, None, Some(format!("warmup stream init failed: {e}"))),
        }

        // Snapshot the lifetime speculation counters AFTER warmup so the accept
        // rate reflects only this candidate's timed runs.
        let spec_before = cpu.cumulative_request_stats();
        let mut runs: Vec<f64> = Vec::with_capacity(repeats);
        let mut last_err: Option<String> = None;
        for _ in 0..repeats {
            cpu.clear_prefix_cache();
            let mut stream = match cpu.speculate_mtp_stream_from_ids(
                prompt_ids.clone(),
                sampling.clone(),
                tw.clone(),
            ) {
                Ok(s) => s,
                Err(e) => {
                    last_err = Some(e.to_string());
                    continue;
                }
            };
            let drained = futures::executor::block_on(async {
                let mut generated = 0usize;
                let mut t_first: Option<std::time::Instant> = None;
                let mut t_last = std::time::Instant::now();
                while let Some(item) = stream.next().await {
                    match item {
                        Ok(_) => {
                            let now = std::time::Instant::now();
                            if t_first.is_none() {
                                t_first = Some(now);
                            }
                            t_last = now;
                            generated += 1;
                        }
                        Err(e) => return Err(e.to_string()),
                    }
                }
                Ok((generated, t_first, t_last))
            });
            match drained {
                Ok((generated, Some(t_first), t_last)) if generated >= 2 => {
                    let decode_s = (t_last - t_first).as_secs_f64();
                    if decode_s > 0.0 {
                        runs.push((generated - 1) as f64 / decode_s);
                    }
                }
                Ok(_) => {}
                Err(e) => last_err = Some(e),
            }
        }
        let spec_after = cpu.cumulative_request_stats();
        let drafted = spec_after
            .total_spec_drafted
            .saturating_sub(spec_before.total_spec_drafted);
        let accepted = spec_after
            .total_spec_accepted
            .saturating_sub(spec_before.total_spec_accepted);
        let accept = if drafted > 0 {
            Some(accepted as f64 / drafted as f64)
        } else {
            None
        };
        if runs.is_empty() {
            return (
                None,
                accept,
                Some(last_err.unwrap_or_else(|| "every measurement run failed".into())),
            );
        }
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if runs.len() % 2 == 1 {
            runs[runs.len() / 2]
        } else {
            (runs[runs.len() / 2 - 1] + runs[runs.len() / 2]) / 2.0
        };
        (Some(median), accept, None)
    };

    let mut results: Vec<rustllama_tuner::TreeWidthMeasurement> =
        Vec::with_capacity(candidates.len());
    let mut first_err: Option<String> = None;
    for widths in candidates {
        let (tok_s, accept, err) = bench_widths(&mut cpu, widths);
        if let Some(e) = err {
            if first_err.is_none() {
                first_err = Some(e);
            }
        }
        results.push(rustllama_tuner::TreeWidthMeasurement {
            widths: widths.clone(),
            tok_s,
            accept,
        });
    }

    // Restore the engine's default (no tree spec).
    cpu.set_tree_speculative(None);

    Ok(TreeWidthsReport {
        capable: true,
        candidates: results,
        error: first_err,
    })
}

/// A/B the chunked-parallel SSM (DeltaNet) prefill path by measured
/// PREFILL tok/s.
///
/// Loads the model ONCE. The chunked path is gated by the
/// `RUSTLLAMA_SSM_PREFILL_CHUNKED` env var, which `llama_arch.rs` reads
/// PER-FORWARD (once at the top of the prefill pass, via `is_some()`),
/// so both arms run on the same loaded engine by toggling the env var
/// between runs — NO reload. Because the gate uses `is_some()`, the OFF
/// arm must REMOVE the var entirely (setting it to `"0"` would still
/// read as on); the pre-sweep value is restored before returning.
///
/// Prefill is isolated exactly like [`measure_batch_size_candidates`]:
/// `max_tokens = 1`, and prefill tok/s = `tokens_prefilled /
/// (prefill_ms / 1000)`. A synthetic prompt (256 tokens — the
/// quick-by-default size, mirroring the batch-size sweep's quick
/// prompt) keeps prefill the dominant, measurable cost while keeping
/// the sweep fast; prefill tok/s ranking is size-independent, so the
/// two arms rank the same as at the old 2048. On a non-SSM /
/// non-DeltaNet model the env var changes nothing and both arms measure
/// the same path (a degenerate, harmless near-tie).
pub fn measure_ssm_prefill_chunked(
    model_path: &std::path::Path,
    cfg: &MeasurementConfig,
    repeats: usize,
) -> Result<SsmPrefillChunkedReport, String> {
    // Quick-by-default synthetic prompt so prefill still dominates
    // (mirror the batch-size sweep's quick prompt). Prefill tok/s
    // ranking is size-independent, so 256 tokens ranks the chunked
    // on/off arms the same as the old 2048 at ~8× less work. ctx holds
    // the prompt + the single decode token + slack.
    const PROMPT_TOKENS: u32 = 256;
    let ctx_size = (PROMPT_TOKENS as usize + 8).max(64);

    let mut cpu = ContiguousEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        true,
        cfg.kv_dtype,
        &cfg.kv_cache_layout,
    )
    .map_err(|e| format!("load failed: {e}"))?;
    cpu.set_prefix_cache(false);
    cpu.set_flash_attention(cfg.flash_attention);
    apply_measurement_placement(&mut cpu, cfg.n_gpu_layers);

    let vocab = cpu.vocab_size() as i32;
    let prompt_ids: Vec<i32> = (0..PROMPT_TOKENS as i32)
        .map(|i| 1 + (i % vocab.max(2).saturating_sub(1)))
        .collect();
    // Pin decode to 1 token so the measurement isolates prefill.
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        typical_p: 1.0,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 0,
        max_tokens: 1,
        stop: Vec::new(),
        ..SamplingParams::default()
    };

    // Median prefill tok/s for the current env-var state: one untimed
    // warmup + `repeats` timed runs. Mirrors the prefill-metric pattern
    // in `measure_batch_size_candidates` (incl. the
    // `tokens_prefilled.max(prompt)` guard).
    let bench = |cpu: &mut ContiguousEngine| -> (Option<f64>, Option<String>) {
        cpu.clear_prefix_cache();
        if let Err(e) = cpu.generate_token_ids(&prompt_ids, 1, &sampling) {
            return (None, Some(format!("warmup failed: {e}")));
        }
        let mut runs: Vec<f64> = Vec::with_capacity(repeats);
        let mut last_err: Option<String> = None;
        for _ in 0..repeats {
            cpu.clear_prefix_cache();
            match cpu.generate_token_ids(&prompt_ids, 1, &sampling) {
                Ok(_) => {
                    let stats = cpu.last_request_stats();
                    let pf_ms = stats.prefill_ms;
                    let prefilled = stats.tokens_prefilled.max(PROMPT_TOKENS);
                    if pf_ms > 0.0 && prefilled > 0 {
                        runs.push((prefilled as f64) / (pf_ms / 1000.0));
                    }
                }
                Err(e) => last_err = Some(e.to_string()),
            }
        }
        if runs.is_empty() {
            return (
                None,
                Some(last_err.unwrap_or_else(|| "every measurement run failed".into())),
            );
        }
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if runs.len() % 2 == 1 {
            runs[runs.len() / 2]
        } else {
            (runs[runs.len() / 2 - 1] + runs[runs.len() / 2]) / 2.0
        };
        (Some(median), None)
    };

    // Save the pre-sweep state so the toggle doesn't leak to the rest of
    // the process. T3: the SSM-prefill flag is the typed accel knob (read
    // per-forward), toggled here via the setter instead of a process-global
    // `set_var`.
    let saved = rustllama_models::accel::ssm_prefill_chunked();

    // OFF arm.
    rustllama_models::accel::set_ssm_prefill_chunked(false);
    let (off_prefill_tps, off_err) = bench(&mut cpu);

    // ON arm — read per-forward, so it takes effect on the next generate
    // with no reload.
    rustllama_models::accel::set_ssm_prefill_chunked(true);
    let (on_prefill_tps, on_err) = bench(&mut cpu);

    // Restore the pre-sweep state.
    rustllama_models::accel::set_ssm_prefill_chunked(saved);

    let winner = match (off_prefill_tps, on_prefill_tps) {
        (Some(off), Some(on)) => Some(on > off),
        _ => None,
    };
    let error = off_err.or(on_err);

    Ok(SsmPrefillChunkedReport {
        off_prefill_tps,
        on_prefill_tps,
        winner,
        error,
    })
}

/// P19: criterion-free FlashAttention bench. Reported numbers for the FA-v3 and
/// FA-v2 generations: PREFILL tok/s and FLASH-DECODE tok/s. `None` when every
/// run for that cell failed.
#[derive(Debug, Clone, Default)]
pub struct FlashAttnBenchReport {
    pub prompt_tokens: u32,
    pub decode_tokens: u32,
    pub v3_prefill_tps: Option<f64>,
    pub v3_decode_tps: Option<f64>,
    pub v2_prefill_tps: Option<f64>,
    pub v2_decode_tps: Option<f64>,
    pub error: Option<String>,
}

/// P19: criterion-free FlashAttention bench. Loads the model ONCE and, for each
/// FA generation (FA-v3 then FA-v2), measures PREFILL tok/s (`max_tokens = 1` →
/// `prefill_ms`, exercising the FA prefill kernels) and FLASH-DECODE tok/s
/// (`decode_tokens` → `decode_ms`, the FA decode kernels). Reuses the same
/// `collect_decode_tok_s` + prefill-metric timing as the tuner sweeps — no
/// criterion. The FA generation is forced per arm via the accel override
/// ([`rustllama_models::accel::set_fa_v3`] / `set_fa_v2`), since the env-latched
/// defaults can't be toggled in-process; both overrides are restored to the env
/// default (`None`) on exit so nothing leaks to the rest of the process. On a
/// CPU-only host or a model below the flash threshold both arms measure the same
/// path (a harmless near-tie) — the numbers still report.
pub fn measure_flash_attn_bench(
    model_path: &std::path::Path,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: usize,
    cfg: &MeasurementConfig,
) -> Result<FlashAttnBenchReport, String> {
    let ctx_size = (prompt_tokens as usize + decode_tokens as usize + 8).max(64);
    let mut cpu = ContiguousEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        true,
        cfg.kv_dtype,
        &cfg.kv_cache_layout,
    )
    .map_err(|e| format!("load failed: {e}"))?;
    cpu.set_prefix_cache(false);
    cpu.set_flash_attention(true); // force the FA path for the bench
    apply_measurement_placement(&mut cpu, cfg.n_gpu_layers);

    let vocab = cpu.vocab_size() as i32;
    let prompt_ids: Vec<i32> = (0..prompt_tokens as i32)
        .map(|i| 1 + (i % vocab.max(2).saturating_sub(1)))
        .collect();
    let decode_sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        typical_p: 1.0,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 0,
        max_tokens: decode_tokens,
        stop: Vec::new(),
        ..SamplingParams::default()
    };
    let prefill_sampling = SamplingParams {
        max_tokens: 1,
        ..decode_sampling.clone()
    };

    let median = |mut runs: Vec<f64>| -> Option<f64> {
        if runs.is_empty() {
            return None;
        }
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(if runs.len() % 2 == 1 {
            runs[runs.len() / 2]
        } else {
            (runs[runs.len() / 2 - 1] + runs[runs.len() / 2]) / 2.0
        })
    };

    // Median prefill tok/s (max_tokens=1 → prefill_ms), one untimed warmup +
    // `repeats` timed runs. Mirrors `measure_ssm_prefill_chunked`.
    let bench_prefill = |cpu: &mut ContiguousEngine| -> Option<f64> {
        cpu.clear_prefix_cache();
        if cpu
            .generate_token_ids(&prompt_ids, 1, &prefill_sampling)
            .is_err()
        {
            return None;
        }
        let mut runs = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            cpu.clear_prefix_cache();
            if cpu
                .generate_token_ids(&prompt_ids, 1, &prefill_sampling)
                .is_ok()
            {
                let stats = cpu.last_request_stats();
                let prefilled = stats.tokens_prefilled.max(prompt_tokens);
                if stats.prefill_ms > 0.0 && prefilled > 0 {
                    runs.push((prefilled as f64) / (stats.prefill_ms / 1000.0));
                }
            }
        }
        median(runs)
    };
    let bench_decode = |cpu: &mut ContiguousEngine| -> Option<f64> {
        // untimed warmup, then the shared decode-tok/s collector.
        cpu.clear_prefix_cache();
        let _ = cpu.generate_token_ids(&prompt_ids, decode_tokens, &decode_sampling);
        let (runs, _) = collect_decode_tok_s(
            cpu,
            &prompt_ids,
            decode_tokens,
            repeats as u32,
            &decode_sampling,
        );
        median(runs)
    };

    // FA-v3 arm: force v3 on, v2 off (so the decode dispatch tries v3 first and
    // never falls through to v2).
    rustllama_models::accel::set_fa_v3(Some(true));
    rustllama_models::accel::set_fa_v2(Some(false));
    let v3_prefill_tps = bench_prefill(&mut cpu);
    let v3_decode_tps = bench_decode(&mut cpu);

    // FA-v2 arm: force v3 off, v2 on.
    rustllama_models::accel::set_fa_v3(Some(false));
    rustllama_models::accel::set_fa_v2(Some(true));
    let v2_prefill_tps = bench_prefill(&mut cpu);
    let v2_decode_tps = bench_decode(&mut cpu);

    // Restore the env defaults — nothing leaks to the rest of the process.
    rustllama_models::accel::set_fa_v3(None);
    rustllama_models::accel::set_fa_v2(None);

    let error = if v3_prefill_tps.is_none()
        && v3_decode_tps.is_none()
        && v2_prefill_tps.is_none()
        && v2_decode_tps.is_none()
    {
        Some("every FlashAttention bench run failed".to_string())
    } else {
        None
    };

    Ok(FlashAttnBenchReport {
        prompt_tokens,
        decode_tokens,
        v3_prefill_tps,
        v3_decode_tps,
        v2_prefill_tps,
        v2_decode_tps,
        error,
    })
}

// ============================================================
// P16: numeric perf-knob sweeps (parallel_matvec_crossover / matvec_min_flops /
// ssm_chunk). Each loads the model ONCE and, for each candidate value, applies
// the knob via its `accel` setter (a hot, per-forward-read cell — no reload),
// then measures median throughput + energy so the tuner's `pick_perf_knob` can
// rank per the resolved `optimize_for` objective. The three axes differ only in
// (a) which setter the candidate toggles and (b) whether the metric is DECODE
// (crossover / min_flops) or PREFILL (ssm_chunk) tok/s.
// ============================================================

/// Which throughput metric a numeric perf-knob sweep ranks by.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KnobMetric {
    /// Median decode tok/s (`generate_token_ids` → `decode_ms`).
    Decode,
    /// Median prefill tok/s (`max_tokens = 1` → `prefill_ms`).
    Prefill,
}

/// Outcome of a numeric perf-knob sweep. `candidates` holds one
/// [`PerfCandidate`] per swept value (median tok/s + tokens-per-joule); the CLI
/// feeds it to [`rustllama_tuner::pick_perf_knob`] with the resolved objective to
/// choose + persist the winner. `error` carries the first arm's failure message
/// when some candidate failed.
#[derive(Debug, Clone, Default)]
pub struct PerfKnobReport {
    pub candidates: Vec<PerfCandidate>,
    pub error: Option<String>,
}

/// Measure one candidate's median throughput + energy on an already-loaded
/// engine (one untimed warmup + `repeats` timed runs). Mirrors the decode/
/// prefill metric patterns used by the placement + ssm-prefill sweeps, with the
/// placement sweep's Sysman energy sampling folded in so `optimize_for = watts`
/// has a tokens-per-joule score to rank by.
fn measure_one_knob_candidate(
    cpu: &mut ContiguousEngine,
    prompt_ids: &[i32],
    sampling: &SamplingParams,
    metric: KnobMetric,
    repeats: usize,
    sysman: Option<&SysmanProbe>,
    prompt_tokens: u32,
    decode_tokens: u32,
) -> (Option<f64>, Option<f64>, Option<String>) {
    let gen_tokens = match metric {
        KnobMetric::Decode => decode_tokens,
        KnobMetric::Prefill => 1,
    };
    // Untimed warmup.
    cpu.clear_prefix_cache();
    if let Err(e) = cpu.generate_token_ids(prompt_ids, gen_tokens, sampling) {
        return (None, None, Some(format!("warmup failed: {e}")));
    }
    let energy_before = sysman.and_then(|p| p.sample_energy());
    let mut runs: Vec<f64> = Vec::with_capacity(repeats);
    let mut total_generated: u64 = 0;
    let mut last_err: Option<String> = None;
    for _ in 0..repeats {
        cpu.clear_prefix_cache();
        match cpu.generate_token_ids(prompt_ids, gen_tokens, sampling) {
            Ok(_) => {
                let stats = cpu.last_request_stats();
                match metric {
                    KnobMetric::Decode => {
                        if stats.decode_ms > 0.0 && stats.tokens_generated > 0 {
                            runs.push((stats.tokens_generated as f64) / (stats.decode_ms / 1000.0));
                            total_generated += stats.tokens_generated as u64;
                        }
                    }
                    KnobMetric::Prefill => {
                        let pf_ms = stats.prefill_ms;
                        let prefilled = stats.tokens_prefilled.max(prompt_tokens);
                        if pf_ms > 0.0 && prefilled > 0 {
                            runs.push((prefilled as f64) / (pf_ms / 1000.0));
                            total_generated += prefilled as u64;
                        }
                    }
                }
            }
            Err(e) => last_err = Some(e.to_string()),
        }
    }
    let energy_after = sysman.and_then(|p| p.sample_energy());
    let tokens_per_joule = match (energy_before, energy_after) {
        (Some((e0, t0)), Some((e1, t1))) if t1 > t0 && e1 > e0 => {
            let de_uj = (e1 - e0) as f64;
            Some(total_generated as f64 / (de_uj * 1e-6))
        }
        _ => None,
    };
    if runs.is_empty() {
        return (
            None,
            tokens_per_joule,
            Some(last_err.unwrap_or_else(|| "every measurement run failed".into())),
        );
    }
    runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if runs.len() % 2 == 1 {
        runs[runs.len() / 2]
    } else {
        (runs[runs.len() / 2 - 1] + runs[runs.len() / 2]) / 2.0
    };
    (Some(median), tokens_per_joule, None)
}

/// Shared driver for the three numeric perf-knob sweeps. Loads the model once,
/// then for each candidate: `apply(value)` (the knob's `accel` setter) → measure
/// → record. Restores the knob to `restore_value` before returning so the sweep
/// doesn't leak state to the rest of the process.
fn measure_perf_knob_generic(
    model_path: &Path,
    cfg: &MeasurementConfig,
    candidates: &[u64],
    repeats: usize,
    metric: KnobMetric,
    prompt_tokens: u32,
    decode_tokens: u32,
    apply: impl Fn(u64),
    restore_value: u64,
) -> Result<PerfKnobReport, String> {
    let ctx_size = (prompt_tokens as usize + decode_tokens as usize + 8).max(128);
    let mut cpu = ContiguousEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        true,
        cfg.kv_dtype,
        &cfg.kv_cache_layout,
    )
    .map_err(|e| format!("load failed: {e}"))?;
    cpu.set_prefix_cache(false);
    cpu.set_flash_attention(cfg.flash_attention);
    apply_measurement_placement(&mut cpu, cfg.n_gpu_layers);

    let vocab = cpu.vocab_size() as i32;
    let prompt_ids: Vec<i32> = (0..prompt_tokens as i32)
        .map(|i| 1 + (i % vocab.max(2).saturating_sub(1)))
        .collect();
    let gen_tokens = match metric {
        KnobMetric::Decode => decode_tokens,
        KnobMetric::Prefill => 1,
    };
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        typical_p: 1.0,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 0,
        max_tokens: gen_tokens,
        stop: Vec::new(),
        ..SamplingParams::default()
    };

    let sysman = if cfg.optimize_for == OptimizeFor::Watts {
        open_sysman_probe()
    } else {
        None
    };

    let mut results: Vec<PerfCandidate> = Vec::with_capacity(candidates.len());
    let mut first_err: Option<String> = None;
    for &value in candidates {
        apply(value);
        let (tok_s, tokens_per_joule, err) = measure_one_knob_candidate(
            &mut cpu,
            &prompt_ids,
            &sampling,
            metric,
            repeats,
            sysman.as_ref(),
            prompt_tokens,
            decode_tokens,
        );
        if let Some(e) = err {
            if first_err.is_none() {
                first_err = Some(e);
            }
        }
        results.push(PerfCandidate {
            value,
            tok_s,
            tokens_per_joule,
        });
    }
    // Restore the pre-sweep value.
    apply(restore_value);

    Ok(PerfKnobReport {
        candidates: results,
        error: first_err,
    })
}

/// A/B the CPU parallel-matvec crossover (M rows) by measured DECODE tok/s.
/// Toggles `accel::set_parallel_matvec_crossover` per candidate on one loaded
/// engine. On a fully-GPU-resident model the CPU matvec barely runs, so the
/// candidates tie and `pick_perf_knob` keeps the default — fail-safe.
pub fn measure_parallel_matvec_crossover(
    model_path: &Path,
    cfg: &MeasurementConfig,
    candidates: &[u64],
    repeats: usize,
    prompt_tokens: u32,
    decode_tokens: u32,
) -> Result<PerfKnobReport, String> {
    let saved = rustllama_models::accel::parallel_matvec_crossover() as u64;
    measure_perf_knob_generic(
        model_path,
        cfg,
        candidates,
        repeats,
        KnobMetric::Decode,
        prompt_tokens,
        decode_tokens,
        |v| rustllama_models::accel::set_parallel_matvec_crossover(v as u32),
        saved,
    )
}

/// A/B the GPU-vs-CPU matvec FLOP crossover by measured DECODE tok/s. Toggles
/// `accel::set_matvec_min_flops` per candidate on one loaded engine. On a
/// CPU-only host (no GPU dispatch) the threshold is inert, so candidates tie and
/// the default is kept — fail-safe.
pub fn measure_matvec_min_flops(
    model_path: &Path,
    cfg: &MeasurementConfig,
    candidates: &[u64],
    repeats: usize,
    prompt_tokens: u32,
    decode_tokens: u32,
) -> Result<PerfKnobReport, String> {
    let saved = rustllama_models::accel::matvec_min_flops();
    measure_perf_knob_generic(
        model_path,
        cfg,
        candidates,
        repeats,
        KnobMetric::Decode,
        prompt_tokens,
        decode_tokens,
        rustllama_models::accel::set_matvec_min_flops,
        saved,
    )
}

/// A/B the DeltaNet prefill chunk length by measured PREFILL tok/s. Toggles
/// `accel::set_ssm_chunk` per candidate on one loaded engine. Only meaningful on
/// a hybrid (DeltaNet) model with the chunked path enabled; the CLI gates the
/// call on `gguf_is_hybrid`, so on anything else it is never invoked.
pub fn measure_ssm_chunk(
    model_path: &Path,
    cfg: &MeasurementConfig,
    candidates: &[u64],
    repeats: usize,
    prompt_tokens: u32,
) -> Result<PerfKnobReport, String> {
    let saved = rustllama_models::accel::ssm_chunk() as u64;
    // The chunked SSM prefill path must be ON for the chunk-length knob to take
    // effect; force it for the sweep and restore after.
    let saved_chunked = rustllama_models::accel::ssm_prefill_chunked();
    rustllama_models::accel::set_ssm_prefill_chunked(true);
    let out = measure_perf_knob_generic(
        model_path,
        cfg,
        candidates,
        repeats,
        KnobMetric::Prefill,
        prompt_tokens,
        1,
        |v| rustllama_models::accel::set_ssm_chunk(v as u32),
        saved,
    );
    rustllama_models::accel::set_ssm_prefill_chunked(saved_chunked);
    out
}

/// Outcome of [`measure_dev_attn`] — the #2 device-resident attention A/B.
///
/// `capable` is `true` only when CUDA is active AND the ON arm actually ran the
/// device block (the hit counter advanced); otherwise the measurement is
/// meaningless (no NVIDIA GPU, non-packed q/k/v/o, qkv bias, unsupported shape)
/// and the CLI persists NO verdict so the path stays off. `off_tps` / `on_tps`
/// are median DECODE tok/s with the block forced off / on on the SAME f32-KV
/// engine; `winner = Some(on > off * margin)` — a real margin past run-to-run
/// noise, since the block is a no-win on MoE / CPU-expert-tail configs (attention
/// isn't the bottleneck there) and should auto-enable only on a clear win.
#[derive(Debug, Clone, Default)]
pub struct DevAttnReport {
    pub capable: bool,
    pub off_tps: Option<f64>,
    pub on_tps: Option<f64>,
    pub winner: Option<bool>,
    pub error: Option<String>,
    /// Which device-resident block the A/B drove, so the CLI persists the right
    /// (disjoint) verdict key: `true` = the Feature-1a SYCL block (SYCL/USM GPU,
    /// no CUDA), `false` = the #2 CUDA block OR the Feature-1c MLX block (see
    /// `is_mlx`). `rustllama-cli` doesn't link `rustllama-models`, so it can't
    /// read the backend itself.
    pub is_sycl: bool,
    /// `true` when the A/B drove the Feature-1c Apple-Metal/MLX block (MLX active,
    /// not CUDA, not SYCL) — the CLI then persists `attn:dev_resident_block_mlx`.
    /// Disjoint from `is_sycl`: at most one of the two is set (CUDA ⇒ both false).
    pub is_mlx: bool,
    /// Feature 1b (CUDA only): median DECODE tok/s with the CUDA-GRAPH block
    /// forced on (vs the eager device block = `on_tps`). `None` when not measured
    /// (not CUDA, or the graph arm never engaged).
    pub graph_tps: Option<f64>,
    /// `Some(true)` when the graph arm engaged AND beat the eager device block by
    /// the win margin — the CLI then persists `attn:dev_resident_block_cuda_graph`.
    /// `None` = not measured / graph never engaged (verdict left unset → off).
    pub graph_winner: Option<bool>,
}

/// A/B the #2 on-device activation-residency attention block by measured DECODE
/// tok/s, forcing f32 KV (the block is F32-KV-only). Loads the model ONCE; the
/// block is a process-global force toggle
/// ([`rustllama_models::accel::set_dev_attn_force`]), so both arms run on the
/// single loaded engine with NO reload. CUDA-only — off CUDA the block never
/// runs, so we skip (`capable=false`, no verdict). Mirrors
/// [`measure_speculative_mtp`]'s shape; both arms drive `generate_token_ids`
/// (the classic decode loop the block hooks per-layer), so the tok/s delta is
/// exactly the host↔device round-trip saving. Timing is content-independent
/// (the flash kernel runs over the same range regardless of mirror contents), so
/// the A/B is a valid SPEED measurement even though the block's CORRECTNESS is
/// validated separately (coherence run).
pub fn measure_dev_attn(
    model_path: &std::path::Path,
    cfg: &MeasurementConfig,
    repeats: usize,
) -> Result<DevAttnReport, String> {
    // A device-resident block runs on CUDA (#2) OR SYCL/USM (Feature 1a); nothing
    // to measure on a host with neither (return BEFORE loading so it wastes no
    // model load). The A/B itself is backend-agnostic — it forces the shared
    // `dev_attn` override on/off and times decode either way; whichever block is
    // live engages, confirmed by the `dev_attn_hits` engagement check below.
    if !rustllama_models::accel::dev_attn_capable() {
        return Ok(DevAttnReport {
            capable: false,
            error: Some(
                "no active NVIDIA GPU, SYCL/USM GPU, or Apple Metal/MLX GPU — \
                 device-resident attention cannot run"
                    .into(),
            ),
            ..Default::default()
        });
    }
    // Clear-win margin: the block is ~flat/slightly-negative on MoE / CPU-tail
    // configs and wins only on a dense model or a fully-fitting GPU, so require
    // ON to beat OFF by >3% to clear same-process back-to-back noise before
    // auto-enabling. Fail-closed otherwise.
    const WIN_MARGIN: f64 = 1.03;
    const PROMPT_TOKENS: u32 = 64;
    // A few more decode tokens than the MTP probe (16): the per-token transfer
    // saving must dominate the one-time prefill→decode mirror seed.
    const DECODE_TOKENS: u32 = 48;
    let ctx_size = (PROMPT_TOKENS as usize + DECODE_TOKENS as usize + 8).max(128);

    // Force f32 KV regardless of cfg.kv_dtype — the block is F32-KV-only, so the
    // A/B measures exactly the scenario where it can engage. (At serve time the
    // dispatch gate is f32-KV-only too, so a quant-KV winner leaves it dormant.)
    let mut cpu = ContiguousEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        true,
        KvDtype::F32,
        &cfg.kv_cache_layout,
    )
    .map_err(|e| format!("load failed: {e}"))?;
    cpu.set_prefix_cache(false);
    cpu.set_flash_attention(cfg.flash_attention);
    apply_measurement_placement(&mut cpu, cfg.n_gpu_layers);

    let vocab = cpu.vocab_size() as i32;
    let prompt_ids: Vec<i32> = (0..PROMPT_TOKENS as i32)
        .map(|i| 1 + (i % vocab.max(2).saturating_sub(1)))
        .collect();
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        typical_p: 1.0,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 0,
        max_tokens: DECODE_TOKENS,
        stop: Vec::new(),
        ..SamplingParams::default()
    };

    // One untimed warmup + `repeats` timed decode runs → median tok/s (the exact
    // decode-metric pattern the kv_dtype / MTP sweeps use).
    let bench = |cpu: &mut ContiguousEngine| -> (Option<f64>, Option<String>) {
        cpu.clear_prefix_cache();
        if let Err(e) = cpu.generate_token_ids(&prompt_ids, DECODE_TOKENS, &sampling) {
            return (None, Some(format!("warmup failed: {e}")));
        }
        let mut runs: Vec<f64> = Vec::with_capacity(repeats);
        let mut last_err: Option<String> = None;
        for _ in 0..repeats {
            cpu.clear_prefix_cache();
            match cpu.generate_token_ids(&prompt_ids, DECODE_TOKENS, &sampling) {
                Ok(_) => {
                    let stats = cpu.last_request_stats();
                    if stats.decode_ms > 0.0 && stats.tokens_generated > 0 {
                        runs.push((stats.tokens_generated as f64) / (stats.decode_ms / 1000.0));
                    }
                }
                Err(e) => last_err = Some(e.to_string()),
            }
        }
        if runs.is_empty() {
            return (
                None,
                Some(last_err.unwrap_or_else(|| "every measurement run failed".into())),
            );
        }
        runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if runs.len() % 2 == 1 {
            runs[runs.len() / 2]
        } else {
            (runs[runs.len() / 2 - 1] + runs[runs.len() / 2]) / 2.0
        };
        (Some(median), None)
    };

    // OFF arm: the host attention path (current production decode). Pin the
    // Feature-1b graph override OFF for the OFF and eager-ON arms so a cached
    // graph verdict can't engage and skew either baseline.
    rustllama_models::accel::set_dev_attn_graph_force(Some(false));
    rustllama_models::accel::set_dev_attn_force(Some(false));
    let (off_tps, off_err) = bench(&mut cpu);

    // ON arm: the #2 device-resident block (EAGER). Snapshot the hit counter
    // around it to confirm the block actually fired — if it never did, the A/B is
    // meaningless.
    let hits_before = rustllama_models::accel::dev_attn_hits();
    rustllama_models::accel::set_dev_attn_force(Some(true));
    let (on_tps, on_err) = bench(&mut cpu);
    let engaged = rustllama_models::accel::dev_attn_hits() > hits_before;

    // GRAPH arm (Feature 1b, CUDA only): keep the eager device override on (so the
    // CUDA path is live) and force the graph path on; it is PREFERRED over the
    // eager block at the dispatch site, so this measures the graph block. SYCL has
    // no graph path — the override is a no-op there, so only measure on CUDA.
    let is_cuda = rustllama_models::accel::cuda_active();
    let (graph_tps, graph_engaged) = if engaged && is_cuda {
        let gh_before = rustllama_models::accel::dev_attn_graph_hits();
        rustllama_models::accel::set_dev_attn_graph_force(Some(true));
        let (gtps, _gerr) = bench(&mut cpu);
        let gengaged = rustllama_models::accel::dev_attn_graph_hits() > gh_before;
        rustllama_models::accel::set_dev_attn_graph_force(Some(false));
        (gtps, gengaged)
    } else {
        (None, false)
    };

    // Clear both overrides → the verdicts decide at serve time.
    rustllama_models::accel::set_dev_attn_force(None);
    rustllama_models::accel::set_dev_attn_graph_force(None);

    if !engaged {
        return Ok(DevAttnReport {
            capable: false,
            off_tps,
            on_tps,
            winner: None,
            error: Some(
                "device-resident attention never engaged (non-packed q/k/v/o, qkv \
                 bias, Qwen3 head-norm, or unsupported shape) — verdict left unset"
                    .into(),
            ),
            is_sycl: false,
            is_mlx: false,
            graph_tps: None,
            graph_winner: None,
        });
    }

    let winner = match (off_tps, on_tps) {
        (Some(off), Some(on)) if off > 0.0 => Some(on > off * WIN_MARGIN),
        _ => None,
    };
    // Graph winner: it must have engaged AND beaten the EAGER device block (its
    // baseline — the graph only ever runs where the eager device block already
    // runs) by the same clear-win margin. Fail-closed otherwise (verdict unset).
    let graph_winner = match (graph_engaged, on_tps, graph_tps) {
        (true, Some(on), Some(gr)) if on > 0.0 => Some(gr > on * WIN_MARGIN),
        _ => None,
    };
    // Which backend's block engaged: CUDA runs the #2 block; off CUDA it was the
    // Feature-1c MLX block when Metal is active, else the Feature-1a SYCL block.
    // `is_sycl` / `is_mlx` are disjoint (both false on CUDA) so the CLI persists
    // exactly one verdict key.
    let is_mlx = !is_cuda && rustllama_models::accel::mlx_active();
    Ok(DevAttnReport {
        capable: true,
        off_tps,
        on_tps,
        winner,
        error: off_err.or(on_err),
        is_sycl: !is_cuda && !is_mlx,
        is_mlx,
        graph_tps,
        graph_winner,
    })
}

/// Outcome of [`measure_moe_vram_margin_candidates`] — the MoE VRAM-margin A/B.
///
/// `capable` is `true` only when a reload actually engaged the tiered-expert
/// promotion (CUDA / dedicated-SYCL MoE model); otherwise the margin has no
/// effect and the CLI persists nothing. `candidates` are `(margin_bytes, median
/// decode tok/s)` per swept margin; `winner` is the margin with the best median
/// decode (ties → the LARGER / safer margin). A smaller margin promotes more
/// experts (faster) until UVM paging thrash drags decode back down, so the
/// best-decode margin is the sweet spot — and it's safe to probe aggressively
/// because managed memory oversubscribes rather than OOMing.
#[derive(Debug, Clone, Default)]
pub struct MoeVramMarginReport {
    pub capable: bool,
    pub winner: Option<u64>,
    pub candidates: Vec<(u64, Option<f64>)>,
    pub error: Option<String>,
}

/// A/B the MoE tiered-expert VRAM margin by measured DECODE tok/s. Reloads the
/// model once per margin (the budget is applied at LOAD, so unlike the placement
/// A/B this can't toggle in place) with the accel margin override set, confirms
/// the tier promoted (`moe_tier_active`), and keeps the best-decode margin.
/// CUDA / dedicated-SYCL MoE only — inert elsewhere (`capable=false`, no winner).
pub fn measure_moe_vram_margin_candidates(
    model_path: &Path,
    margins: &[u64],
    ctx_size: usize,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
    cfg: &MeasurementConfig,
) -> crate::Result<MoeVramMarginReport> {
    use rustllama_models::accel;

    // The margin only matters where the tier can promote into device VRAM.
    let tier_possible = accel::cuda_active() || accel::sycl_dedicated_vram_bytes().is_some();
    if !tier_possible {
        return Ok(MoeVramMarginReport {
            capable: false,
            error: Some("no CUDA / dedicated-SYCL GPU — the MoE VRAM tier is inert".into()),
            ..Default::default()
        });
    }
    if margins.is_empty() {
        return Ok(MoeVramMarginReport {
            capable: false,
            error: Some("no margin candidates".into()),
            ..Default::default()
        });
    }

    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        typical_p: 1.0,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 0,
        max_tokens: decode_tokens,
        stop: Vec::new(),
        ..SamplingParams::default()
    };

    let mut candidates: Vec<(u64, Option<f64>)> = Vec::with_capacity(margins.len());
    let mut best: Option<(u64, f64)> = None;
    let mut any_tier = false;
    let mut last_err: Option<String> = None;

    for &margin in margins {
        accel::set_moe_vram_margin_override(Some(margin));
        let mut cpu = match ContiguousEngine::load_with_options_and_layout(
            model_path,
            ctx_size,
            true,
            cfg.kv_dtype,
            &cfg.kv_cache_layout,
        ) {
            Ok(c) => c,
            Err(e) => {
                last_err = Some(format!("load failed (margin {margin}): {e}"));
                candidates.push((margin, None));
                continue;
            }
        };
        // Did this margin actually engage the tier? `load_inner` sets the flag
        // (reset per load) only when it promoted >=1 expert into device memory.
        let promoted = accel::moe_tier_active();
        any_tier |= promoted;
        cpu.set_prefix_cache(false);
        cpu.set_flash_attention(cfg.flash_attention);
        apply_measurement_placement(&mut cpu, cfg.n_gpu_layers);

        let vocab = cpu.vocab_size() as i32;
        let prompt_ids: Vec<i32> = (0..prompt_tokens as i32)
            .map(|i| 1 + (i % vocab.max(2).saturating_sub(1)))
            .collect();

        // Untimed warmup + `repeats` timed decode runs → median tok/s.
        cpu.clear_prefix_cache();
        let tps = if cpu
            .generate_token_ids(&prompt_ids, decode_tokens, &sampling)
            .is_err()
        {
            None
        } else {
            let mut runs: Vec<f64> = Vec::with_capacity(repeats as usize);
            for _ in 0..repeats {
                cpu.clear_prefix_cache();
                if cpu
                    .generate_token_ids(&prompt_ids, decode_tokens, &sampling)
                    .is_ok()
                {
                    let stats = cpu.last_request_stats();
                    if stats.decode_ms > 0.0 && stats.tokens_generated > 0 {
                        runs.push((stats.tokens_generated as f64) / (stats.decode_ms / 1000.0));
                    }
                }
            }
            if runs.is_empty() {
                None
            } else {
                runs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                Some(if runs.len() % 2 == 1 {
                    runs[runs.len() / 2]
                } else {
                    (runs[runs.len() / 2 - 1] + runs[runs.len() / 2]) / 2.0
                })
            }
        };
        candidates.push((margin, tps));
        if let Some(t) = tps {
            // Best decode; on a (near-)tie prefer the LARGER margin already held
            // (safer headroom) — only replace when strictly faster.
            match best {
                Some((_, bt)) if t <= bt => {}
                _ => best = Some((margin, t)),
            }
        }
        // Drop the engine (frees the promoted device memory) before the next
        // reload; the load path's reset_device_caches_for_new_model clears any
        // stale device-cache/KV-mirror state on the following load.
        drop(cpu);
        accel::set_moe_vram_margin_override(None);
    }

    accel::set_moe_vram_margin_override(None);
    if !any_tier {
        return Ok(MoeVramMarginReport {
            capable: false,
            candidates,
            error: last_err.or_else(|| {
                Some("tier never promoted (not a MoE model, or no device VRAM budget)".into())
            }),
            ..Default::default()
        });
    }
    Ok(MoeVramMarginReport {
        capable: true,
        winner: best.map(|(m, _)| m),
        candidates,
        error: last_err,
    })
}

/// Result of the CacheBlend recompute-ratio sweep.
pub struct CacheblendMeasurement {
    /// The smallest swept ratio whose blended next-token distribution stayed
    /// coherent (mean cosine ≥ the bar) vs a full prefill. `0.0` = none cleared
    /// the bar (or the model is hybrid / unsupported) → CacheBlend off.
    pub chosen_ratio: f32,
    /// `(ratio, mean cosine vs full prefill)` for every candidate — for the
    /// CLI to print.
    pub per_ratio_cos: Vec<(f32, f32)>,
}

/// Autotune the CacheBlend recompute ratio for a model: the SMALLEST `ratio`
/// whose blended next-token distribution stays coherent (mean cosine ≥
/// `coherence_min_cos`) against a full prefill, measured on synthetic
/// multi-chunk contexts. This is a quality calibration, not a speed sweep —
/// a smaller coherent ratio recomputes fewer tokens (faster RAG prefill) while
/// preserving the output. `0.0` means the model was too sensitive at every
/// candidate (CacheBlend stays off). Hybrid models are unsupported → `0.0`.
///
/// Mirrors the KV-dtype coherence gate: the forward is run on a forced
/// contiguous F32-KV load (CacheBlend's requirement); `candidates` should be
/// ascending.
pub fn measure_cacheblend_ratio(
    model_path: &Path,
    ctx_size: usize,
    candidates: &[f32],
    coherence_min_cos: f32,
) -> std::result::Result<CacheblendMeasurement, String> {
    use rustllama_models::cacheblend::{CacheBlendConfig, ChunkPlacement};
    use rustllama_models::llama_arch::KvCache;
    use std::sync::Arc;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0f32;
        let mut na = 0f32;
        let mut nb = 0f32;
        for (&x, &y) in a.iter().zip(b.iter()) {
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        dot / (na.sqrt() * nb.sqrt() + 1e-12)
    }

    if candidates.is_empty() {
        return Err("measure_cacheblend_ratio: empty candidate list".into());
    }

    // CacheBlend requires contiguous F32 KV — force it for the measurement
    // regardless of the model's tuned serve-time KV dtype.
    let cpu = ContiguousEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        false,
        KvDtype::F32,
        "contiguous",
    )
    .map_err(|e| format!("load failed: {e}"))?;
    let model = cpu.model_arc();
    if model.weights.is_hybrid() {
        return Ok(CacheblendMeasurement {
            chosen_ratio: 0.0,
            per_ratio_cos: Vec::new(),
        });
    }
    let vocab = model.cfg.vocab_size as u64;
    if vocab < 2 {
        return Err("measure_cacheblend_ratio: vocab too small".into());
    }

    // Deterministic synthetic token in [1, vocab).
    let tok = |seed: u64| -> i32 {
        let mut h = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= h >> 29;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 32;
        (1 + (h % (vocab - 1))) as i32
    };

    const N_CHUNKS: usize = 3;
    const CHUNK_LEN: usize = 24;
    const SUFFIX_LEN: usize = 12;
    const N_CONTEXTS: usize = 2;

    let mut sums = vec![0f32; candidates.len()];
    let mut counted = 0usize;
    for ctx_i in 0..N_CONTEXTS {
        let base = (ctx_i as u64 + 1).wrapping_mul(1_000_003);
        let mut chunks: Vec<Vec<i32>> = Vec::with_capacity(N_CHUNKS);
        for c in 0..N_CHUNKS {
            let mut v = Vec::with_capacity(CHUNK_LEN);
            for j in 0..CHUNK_LEN {
                v.push(tok(base + (c as u64) * 101 + j as u64));
            }
            chunks.push(v);
        }
        let mut assembled: Vec<i32> = Vec::new();
        for c in &chunks {
            assembled.extend_from_slice(c);
        }
        let suffix_start = assembled.len();
        for j in 0..SUFFIX_LEN {
            assembled.push(tok(base + 90_001 + j as u64));
        }
        if assembled.len() > ctx_size {
            continue;
        }

        // Full-prefill reference. P2: force a NON-ring (full-slab) cache —
        // CacheBlend builds isolated chunk snapshots + injects at absolute
        // offsets, which a SWA ring (cap-strided) can't represent; the full
        // slab still masks the window correctly on an SWA model.
        let mut kv_ref = KvCache::new_with_dtype_no_ring(&model.cfg, ctx_size, KvDtype::F32);
        let ref_logits = model.forward_prefill_batched_f32(&assembled, 0, &mut kv_ref);

        // Isolated chunk snapshots → placements.
        let mut placements: Vec<ChunkPlacement> = Vec::with_capacity(N_CHUNKS);
        let mut off = 0usize;
        for c in &chunks {
            let mut scratch =
                KvCache::new_with_dtype_no_ring(&model.cfg, c.len().max(1), KvDtype::F32);
            let _ = model.forward_prefill_batched_f32(c, 0, &mut scratch);
            placements.push(ChunkPlacement {
                offset: off,
                snapshot: Arc::new(scratch.snapshot_prefix(c.len())),
            });
            off += c.len();
        }

        for (ri, &ratio) in candidates.iter().enumerate() {
            let mut kv_cb = KvCache::new_with_dtype_no_ring(&model.cfg, ctx_size, KvDtype::F32);
            let cb_logits = model.forward_prefill_cacheblend(
                &assembled,
                &placements,
                suffix_start,
                CacheBlendConfig {
                    recompute_ratio: ratio,
                    selection_layer: 1,
                },
                &mut kv_cb,
            );
            sums[ri] += cosine(&cb_logits, &ref_logits);
        }
        counted += 1;
    }

    let per_ratio_cos: Vec<(f32, f32)> = candidates
        .iter()
        .enumerate()
        .map(|(ri, &r)| {
            (
                r,
                if counted > 0 {
                    sums[ri] / counted as f32
                } else {
                    0.0
                },
            )
        })
        .collect();
    // Smallest ratio that cleared the coherence bar.
    let chosen = per_ratio_cos
        .iter()
        .filter(|(_, c)| *c >= coherence_min_cos)
        .map(|(r, _)| *r)
        .fold(f32::INFINITY, f32::min);
    let chosen_ratio = if chosen.is_finite() { chosen } else { 0.0 };
    Ok(CacheblendMeasurement {
        chosen_ratio,
        per_ratio_cos,
    })
}

#[cfg(test)]
mod kv_coherence_tests {
    use super::*;

    fn mk(dt: KvDtype, tps: f64, toks: Option<Vec<u32>>) -> KvDtypeCandidateResult {
        KvDtypeCandidateResult {
            kv_dtype: dt,
            warmup_ms: 0.0,
            median_tps: Some(tps),
            max_tps: Some(tps),
            error: None,
            decode_tokens: toks,
        }
    }

    #[test]
    fn pick_coherence_first_prefers_smaller_when_coherent() {
        // F32 reference; Q8_0 fully agrees; Tq(2) fully agrees;
        // Tq(2) is smaller, so it wins on memory.
        let report = KvDtypeMeasurementReport {
            total_load_ms: 0.0,
            winner: Some(KvDtype::F32),
            winner_tps: 100.0,
            candidates: vec![
                mk(KvDtype::F32, 100.0, Some(vec![1, 2, 3, 4])),
                mk(KvDtype::Q8_0, 110.0, Some(vec![1, 2, 3, 4])),
                mk(KvDtype::Tq(2), 105.0, Some(vec![1, 2, 3, 4])),
            ],
        };
        assert_eq!(report.pick_coherence_first(0.90), Some(KvDtype::Tq(2)));
    }

    #[test]
    fn pick_coherence_first_skips_incoherent_smaller() {
        // F32 reference; Tq(2) tokens are mostly wrong (1/4); Q8_0
        // matches perfectly. Coherence gate cuts Tq(2) — pick Q8_0.
        let report = KvDtypeMeasurementReport {
            total_load_ms: 0.0,
            winner: Some(KvDtype::F32),
            winner_tps: 100.0,
            candidates: vec![
                mk(KvDtype::F32, 100.0, Some(vec![1, 2, 3, 4])),
                mk(KvDtype::Q8_0, 110.0, Some(vec![1, 2, 3, 4])),
                mk(KvDtype::Tq(2), 130.0, Some(vec![1, 5, 6, 7])),
            ],
        };
        assert_eq!(report.pick_coherence_first(0.90), Some(KvDtype::Q8_0));
    }

    #[test]
    fn pick_coherence_first_breaks_memory_ties_on_tps() {
        // Two candidates with identical bits_per_element (Tq(8) and
        // Q8_0 are 8.3 vs 8.5 — different; let's craft a real tie
        // by using two Tq(4) candidates with different tps — only
        // one will appear, so test ties via two configs sharing a
        // bit width. Easier: just confirm Q8_0 wins over a same-bits
        // peer by tps. Here we use Tq(8) (bpe 8.3) which IS smaller
        // than Q8_0 (8.5), so it wins on memory rather than tps.
        let report = KvDtypeMeasurementReport {
            total_load_ms: 0.0,
            winner: None,
            winner_tps: 0.0,
            candidates: vec![
                mk(KvDtype::F32, 100.0, Some(vec![1, 2, 3, 4])),
                mk(KvDtype::Q8_0, 200.0, Some(vec![1, 2, 3, 4])),
                mk(KvDtype::Tq(8), 110.0, Some(vec![1, 2, 3, 4])),
            ],
        };
        assert_eq!(report.pick_coherence_first(0.90), Some(KvDtype::Tq(8)));
    }

    #[test]
    fn pick_coherence_first_threshold_zero_picks_smallest() {
        // With the coherence gate disabled (threshold=0), pure
        // memory-first selection wins — even if Tq(1) produces
        // garbage tokens, it's picked because it's smallest.
        let report = KvDtypeMeasurementReport {
            total_load_ms: 0.0,
            winner: Some(KvDtype::F32),
            winner_tps: 100.0,
            candidates: vec![
                mk(KvDtype::F32, 100.0, Some(vec![1, 2, 3, 4])),
                mk(KvDtype::Tq(1), 90.0, Some(vec![99, 99, 99, 99])),
            ],
        };
        assert_eq!(report.pick_coherence_first(0.0), Some(KvDtype::Tq(1)));
    }

    #[test]
    fn pick_coherence_first_falls_back_to_reference_when_all_quants_fail() {
        // No quant candidate clears the bar; the reference (F32)
        // wins by default — we always return *something* when at
        // least one candidate produced tokens.
        let report = KvDtypeMeasurementReport {
            total_load_ms: 0.0,
            winner: Some(KvDtype::F32),
            winner_tps: 100.0,
            candidates: vec![
                mk(KvDtype::F32, 100.0, Some(vec![1, 2, 3, 4])),
                mk(KvDtype::Tq(1), 90.0, Some(vec![9, 9, 9, 9])),
                mk(KvDtype::Tq(2), 95.0, Some(vec![8, 8, 8, 8])),
            ],
        };
        assert_eq!(report.pick_coherence_first(0.90), Some(KvDtype::F32));
    }

    // Estimated K+V bytes for a dtype: an f16 base scaled by
    // bits/16, matching how the CLI derives the fit estimate from
    // `kv_cache_vram_bytes` (which is f16-based).
    fn kv_bytes_est(dt: KvDtype, f16_base: u64) -> u64 {
        ((f16_base as f64) * (dt.approx_bits_per_element() as f64) / 16.0) as u64
    }

    #[test]
    fn speed_first_picks_fastest_coherent_when_all_fit() {
        // F32 reference; Q8_0 + Q4_0 both coherent. F32 is fastest.
        // With a roomy budget every candidate fits, so speed wins:
        // pick F32 even though it is the largest KV footprint.
        let report = KvDtypeMeasurementReport {
            total_load_ms: 0.0,
            winner: Some(KvDtype::F32),
            winner_tps: 100.0,
            candidates: vec![
                mk(KvDtype::F32, 130.0, Some(vec![1, 2, 3, 4])),
                mk(KvDtype::Q8_0, 110.0, Some(vec![1, 2, 3, 4])),
                mk(KvDtype::Q4_0, 90.0, Some(vec![1, 2, 3, 4])),
            ],
        };
        let base = 1_000_000; // 1 MB f16 base → F32 ≈ 2 MB
        let budget = 1_000_000_000; // 1 GB: everything fits
        assert_eq!(
            report.pick_speed_first_if_fits(0.90, budget, |dt| kv_bytes_est(dt, base)),
            Some(KvDtype::F32)
        );
    }

    #[test]
    fn speed_first_downgrades_under_memory_pressure() {
        // Same candidates, but the budget only admits a ~4.5-bit KV
        // cache. F32 (fastest) and Q8_0 don't fit; Q4_0 does — so the
        // fastest *fitting* coherent candidate (Q4_0) is chosen.
        let report = KvDtypeMeasurementReport {
            total_load_ms: 0.0,
            winner: Some(KvDtype::F32),
            winner_tps: 100.0,
            candidates: vec![
                mk(KvDtype::F32, 130.0, Some(vec![1, 2, 3, 4])),
                mk(KvDtype::Q8_0, 110.0, Some(vec![1, 2, 3, 4])),
                mk(KvDtype::Q4_0, 90.0, Some(vec![1, 2, 3, 4])),
            ],
        };
        let base = 1_000_000;
        // Q4_0 ≈ 281 KB, Q8_0 ≈ 531 KB, F32 ≈ 2 MB. Budget 400 KB
        // admits only Q4_0.
        let budget = 400_000;
        assert_eq!(
            report.pick_speed_first_if_fits(0.90, budget, |dt| kv_bytes_est(dt, base)),
            Some(KvDtype::Q4_0)
        );
    }

    #[test]
    fn speed_first_falls_back_to_smallest_when_nothing_fits() {
        // Budget is below even the smallest coherent candidate: fall
        // back to the smallest coherent dtype (never OOM), matching
        // `pick_coherence_first`.
        let report = KvDtypeMeasurementReport {
            total_load_ms: 0.0,
            winner: Some(KvDtype::F32),
            winner_tps: 100.0,
            candidates: vec![
                mk(KvDtype::F32, 130.0, Some(vec![1, 2, 3, 4])),
                mk(KvDtype::Q4_0, 90.0, Some(vec![1, 2, 3, 4])),
            ],
        };
        let base = 1_000_000;
        let budget = 1; // nothing fits
        assert_eq!(
            report.pick_speed_first_if_fits(0.90, budget, |dt| kv_bytes_est(dt, base)),
            Some(KvDtype::Q4_0)
        );
    }

    #[test]
    fn speed_first_zero_budget_disables_fit_gate() {
        // budget == 0 means "cannot determine budget": honor the
        // speed-first intent and pick the fastest coherent candidate.
        let report = KvDtypeMeasurementReport {
            total_load_ms: 0.0,
            winner: Some(KvDtype::F32),
            winner_tps: 100.0,
            candidates: vec![
                mk(KvDtype::F32, 130.0, Some(vec![1, 2, 3, 4])),
                mk(KvDtype::Q4_0, 90.0, Some(vec![1, 2, 3, 4])),
            ],
        };
        assert_eq!(
            report.pick_speed_first_if_fits(0.90, 0, |_dt| u64::MAX),
            Some(KvDtype::F32)
        );
    }

    #[test]
    fn speed_first_respects_coherence_gate() {
        // F32 fastest but Q4_0 is incoherent (wrong tokens). Even
        // with a roomy budget, the incoherent fast-but-smaller
        // candidate is excluded; F32 (coherent) wins.
        let report = KvDtypeMeasurementReport {
            total_load_ms: 0.0,
            winner: Some(KvDtype::F32),
            winner_tps: 100.0,
            candidates: vec![
                mk(KvDtype::F32, 100.0, Some(vec![1, 2, 3, 4])),
                mk(KvDtype::Q4_0, 200.0, Some(vec![9, 9, 9, 9])),
            ],
        };
        let base = 1_000_000;
        assert_eq!(
            report.pick_speed_first_if_fits(0.90, 1_000_000_000, |dt| kv_bytes_est(dt, base)),
            Some(KvDtype::F32)
        );
    }
}

#[cfg(test)]
mod f14_per_device_tests {
    use super::*;

    #[test]
    fn scale_extra_gpu_tps_anchors_to_active_decode() {
        // Active GPU decodes at 25 tok/s and micro-benches at 1000 matvecs/s.
        // An extra GPU that micro-benches at 2000 is ~2× → ~50 tok/s-equivalent.
        assert_eq!(scale_extra_gpu_tps(25.0, 1000.0, 2000.0), Some(50.0));
        // A slower extra GPU (500) → ~12.5 tok/s-equivalent.
        assert_eq!(scale_extra_gpu_tps(25.0, 1000.0, 500.0), Some(12.5));
        // Non-positive inputs form no ratio.
        assert_eq!(scale_extra_gpu_tps(0.0, 1000.0, 2000.0), None);
        assert_eq!(scale_extra_gpu_tps(25.0, 0.0, 2000.0), None);
        assert_eq!(scale_extra_gpu_tps(25.0, 1000.0, 0.0), None);
    }

    #[test]
    fn extra_gpu_measurement_is_inert_without_cuda() {
        // On this (non-CUDA) box the per-device micro-bench can't build a cache,
        // so it returns None — the inert / byte-identical fallback the active
        // single-device path relies on.
        assert_eq!(measure_matvec_throughput_cuda(0, 2), None);
    }

    #[test]
    fn measure_extra_cuda_gpus_leaves_report_unchanged_on_single_gpu() {
        // F14 byte-identical guard: with no active CUDA GPU (or < 2 devices), the
        // extra-GPU pass must not touch `scores`/`tiers`.
        let mut scores: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
        scores.insert("cpu-slug".into(), 7.5);
        let mut tiers: Vec<PerDeviceTierResult> = Vec::new();
        measure_extra_cuda_gpus(&mut scores, &mut tiers);
        assert_eq!(
            scores.len(),
            1,
            "no extra-GPU entries added on a single-GPU host"
        );
        assert_eq!(scores.get("cpu-slug"), Some(&7.5));
        assert!(
            tiers.is_empty(),
            "no extra-GPU tiers added on a single-GPU host"
        );
    }
}
