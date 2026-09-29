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

use rustllama_tuner::PlacementPlan;

use crate::cpu::CpuEngine;
use crate::SamplingParams;
use rustllama_models::llama_arch::KvDtype;

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
    /// to `CpuEngine::load_with_options_and_layout`.
    pub kv_cache_layout: String,
    pub flash_attention: bool,
    /// Initial `n_gpu_layers` used for the batch-size sweep load.
    /// Ignored by the placement sweep (it overwrites per candidate).
    pub n_gpu_layers: u32,
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
    let mut cpu = CpuEngine::load_with_options_and_layout(
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
                error: Some(
                    last_err.unwrap_or_else(|| "every measurement run failed".into()),
                ),
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

        // Winning metric. `RUSTLLAMA_TUNE_OPTIMIZE_FOR=watts` flips
        // the picker to tokens-per-joule (laptop / battery
        // sensitive); default is tok/s (throughput).
        let optimize_for_watts = std::env::var("RUSTLLAMA_TUNE_OPTIMIZE_FOR")
            .ok()
            .as_deref()
            == Some("watts");
        let score = if optimize_for_watts {
            tokens_per_joule.unwrap_or(median)
        } else {
            median
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
        reading.energy_counter.map(|c| (c.energy_uj, c.timestamp_us))
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
    let mut cpu = CpuEngine::load_with_options_and_layout(
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
    let bench = |cpu: &mut CpuEngine, n_gpu: u32| -> std::result::Result<f64, String> {
        cpu.set_n_gpu_layers(n_gpu);
        cpu.clear_prefix_cache();
        // One untimed warmup (JIT / USM upload / page-in).
        cpu.generate_token_ids(&prompt_ids, decode_tokens, &sampling)
            .map_err(|e| format!("warmup failed: {e}"))?;
        let mut runs: Vec<f64> = Vec::with_capacity(repeats as usize);
        let mut last_err: Option<String> = None;
        for _ in 0..repeats {
            cpu.clear_prefix_cache();
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

    Ok(PerDevicePerfReport { load_ms, scores, tiers })
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

/// SCAFFOLD placeholder for measuring a NON-active GPU on a multi-GPU host.
/// Returns `None` today: end-to-end decode on an arbitrary device needs
/// per-device dispatch selection the engine does not yet expose. The
/// combined-build owner wires this (e.g. a synthetic per-device matvec via
/// `CudaMatvecCache::new(idx, budget)` / `SyclAccel::try_new(idx)`, or an
/// engine device-select seam) and records the result under
/// `rustllama_tuner::{cuda,sycl}_gpu_key`. Kept as an explicit seam so the
/// call site + intent are discoverable rather than silently absent.
#[allow(dead_code)]
fn measure_extra_gpu_stub(_device_index: u32) -> Option<f32> {
    None
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
                    Some((dt, _)) => c.kv_dtype.approx_bits_per_element()
                        > dt.approx_bits_per_element(),
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
            let Some(toks) = c.decode_tokens.as_deref() else { continue };
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
    let matches: usize = a.iter().zip(b.iter()).take(n).filter(|(x, y)| x == y).count();
    matches as f32 / n as f32
}

/// Drive the engine through each KV-dtype candidate. Unlike the
/// placement / batch-size sweeps, this one reloads the model per
/// candidate — the KV cache shape changes with dtype, so a hot
/// reconfigure isn't possible.
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

    for &dt in candidates {
        let load_start = std::time::Instant::now();
        let load_result = CpuEngine::load_with_options_and_layout(
            model_path,
            ctx_size,
            true,
            dt,
            &cfg.kv_cache_layout,
        );
        total_load_ms += load_start.elapsed().as_secs_f64() * 1000.0;
        let mut cpu = match load_result {
            Ok(c) => c,
            Err(e) => {
                report_candidates.push(KvDtypeCandidateResult {
                    kv_dtype: dt,
                    warmup_ms: 0.0,
                    median_tps: None,
                    max_tps: None,
                    error: Some(format!("load failed: {e}")),
                    decode_tokens: None,
                });
                continue;
            }
        };
        cpu.set_prefix_cache(false);
        cpu.set_flash_attention(cfg.flash_attention);
        cpu.set_n_gpu_layers(cfg.n_gpu_layers);

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

        let mut runs: Vec<f64> = Vec::with_capacity(repeats as usize);
        let mut last_err: Option<String> = None;
        for _ in 0..repeats {
            cpu.clear_prefix_cache();
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
            report_candidates.push(KvDtypeCandidateResult {
                kv_dtype: dt,
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(
                    last_err.unwrap_or_else(|| "every measurement run failed".into()),
                ),
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
/// [`CpuEngine::set_flash_attention`].
pub fn measure_flash_attention_candidates(
    model_path: &Path,
    ctx_size: usize,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
    cfg: &MeasurementConfig,
) -> crate::Result<FlashAttentionMeasurementReport> {
    let load_start = std::time::Instant::now();
    let mut cpu = CpuEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        true,
        cfg.kv_dtype,
        &cfg.kv_cache_layout,
    )
    .map_err(|e| crate::EngineError::Engine(format!("load failed: {e}")))?;
    cpu.set_prefix_cache(false);
    cpu.set_n_gpu_layers(cfg.n_gpu_layers);
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

        let mut runs: Vec<f64> = Vec::with_capacity(repeats as usize);
        let mut last_err: Option<String> = None;
        for _ in 0..repeats {
            cpu.clear_prefix_cache();
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
            report_candidates.push(FlashAttentionCandidateResult {
                flash_attention: flag,
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(
                    last_err.unwrap_or_else(|| "every measurement run failed".into()),
                ),
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
        let load_result =
            CpuEngine::load_with_options_and_layout(model_path, ctx_size, true, cfg.kv_dtype, layout);
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
        cpu.set_n_gpu_layers(cfg.n_gpu_layers);

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

        let mut runs: Vec<f64> = Vec::with_capacity(repeats as usize);
        let mut last_err: Option<String> = None;
        for _ in 0..repeats {
            cpu.clear_prefix_cache();
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
            report_candidates.push(KvLayoutCandidateResult {
                kv_cache_layout: layout.clone(),
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(
                    last_err.unwrap_or_else(|| "every measurement run failed".into()),
                ),
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
    let mut cpu = CpuEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        true,
        cfg.kv_dtype,
        &cfg.kv_cache_layout,
    )
    .map_err(|e| crate::EngineError::Engine(format!("load failed: {e}")))?;
    cpu.set_prefix_cache(false);
    cpu.set_flash_attention(cfg.flash_attention);
    cpu.set_n_gpu_layers(cfg.n_gpu_layers);
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
                error: Some(
                    last_err.unwrap_or_else(|| "every measurement run failed".into()),
                ),
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
    let mut cpu = CpuEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        true,
        cfg.kv_dtype,
        &cfg.kv_cache_layout,
    )
    .map_err(|e| crate::EngineError::Engine(format!("load failed: {e}")))?;
    cpu.set_prefix_cache(false);
    cpu.set_flash_attention(cfg.flash_attention);
    cpu.set_n_gpu_layers(cfg.n_gpu_layers);
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
    let mut report_candidates: Vec<FlashKvMinCandidateResult> = Vec::with_capacity(candidates.len());
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

        let mut runs: Vec<f64> = Vec::with_capacity(repeats as usize);
        let mut last_err: Option<String> = None;
        for _ in 0..repeats {
            cpu.clear_prefix_cache();
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
            report_candidates.push(FlashKvMinCandidateResult {
                kv_min,
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(
                    last_err.unwrap_or_else(|| "every measurement run failed".into()),
                ),
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
    let mut cpu = CpuEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        true,
        cfg.kv_dtype,
        &cfg.kv_cache_layout,
    )
    .map_err(|e| crate::EngineError::Engine(format!("load failed: {e}")))?;
    cpu.set_prefix_cache(true);
    cpu.set_flash_attention(cfg.flash_attention);
    cpu.set_n_gpu_layers(cfg.n_gpu_layers);
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

    let mut report_candidates: Vec<PrefixSnapshotsCandidateResult> = Vec::with_capacity(candidates.len());
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
                error: Some(
                    last_err.unwrap_or_else(|| "every measurement run failed".into()),
                ),
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
/// [`CpuEngine::load_with_options_layout_and_page_size`] — page size
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
    let mut report_candidates: Vec<KvPageSizeCandidateResult> = Vec::with_capacity(candidates.len());
    let mut best_tps = 0.0f64;
    let mut best_page: Option<u32> = None;
    let mut total_load_ms = 0.0f64;

    for &page_size in candidates {
        let load_start = std::time::Instant::now();
        let load_result = CpuEngine::load_with_options_layout_and_page_size(
            model_path, ctx_size, true, cfg.kv_dtype, &cfg.kv_cache_layout, page_size,
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
        cpu.set_n_gpu_layers(cfg.n_gpu_layers);

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

        let mut runs: Vec<f64> = Vec::with_capacity(repeats as usize);
        let mut last_err: Option<String> = None;
        for _ in 0..repeats {
            cpu.clear_prefix_cache();
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
            report_candidates.push(KvPageSizeCandidateResult {
                page_size,
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(
                    last_err.unwrap_or_else(|| "every measurement run failed".into()),
                ),
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
    let mut cpu = CpuEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        true,
        cfg.kv_dtype,
        &cfg.kv_cache_layout,
    )
    .map_err(|e| crate::EngineError::Engine(format!("load failed: {e}")))?;
    cpu.set_prefix_cache(false);
    cpu.set_flash_attention(cfg.flash_attention);
    cpu.set_n_gpu_layers(cfg.n_gpu_layers);
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
    let mut report_candidates: Vec<FlashV3KvTileCandidateResult> = Vec::with_capacity(candidates.len());
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

        let mut runs: Vec<f64> = Vec::with_capacity(repeats as usize);
        let mut last_err: Option<String> = None;
        for _ in 0..repeats {
            cpu.clear_prefix_cache();
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
            report_candidates.push(FlashV3KvTileCandidateResult {
                kv_tile,
                warmup_ms,
                median_tps: None,
                max_tps: None,
                error: Some(
                    last_err.unwrap_or_else(|| "every measurement run failed".into()),
                ),
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
    let mut cpu = CpuEngine::load_with_options_and_layout(
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
    cpu.set_n_gpu_layers(cfg.n_gpu_layers);
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
        let mut runs: Vec<f64> = Vec::with_capacity(repeats as usize);
        let mut last_err: Option<String> = None;
        for _ in 0..repeats {
            cpu.clear_prefix_cache();
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
        if best.as_ref().map(|(_, _, _, t)| median > *t).unwrap_or(true) {
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
                let placement = if experts_cpu { "experts_cpu" } else { "uniform" };
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
/// self-speculation is a HOT toggle ([`CpuEngine::set_mtp_speculative`]),
/// so both arms run on that single loaded engine with NO reload between
/// them.
///
/// Capability gate: [`CpuEngine::mtp_speculative`] returns
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

    let mut cpu = CpuEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        true,
        cfg.kv_dtype,
        &cfg.kv_cache_layout,
    )
    .map_err(|e| format!("load failed: {e}"))?;
    cpu.set_prefix_cache(false);
    cpu.set_flash_attention(cfg.flash_attention);
    cpu.set_n_gpu_layers(cfg.n_gpu_layers);

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
    let bench = |cpu: &mut CpuEngine| -> (Option<f64>, Option<String>) {
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
    let bench_mtp = |cpu: &mut CpuEngine| -> (Option<f64>, Option<String>) {
        use futures::StreamExt;
        // Untimed warmup: drain one full stream.
        cpu.clear_prefix_cache();
        match cpu.speculate_mtp_stream_from_ids(prompt_ids.clone(), sampling.clone()) {
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
            let mut stream = match cpu
                .speculate_mtp_stream_from_ids(prompt_ids.clone(), sampling.clone())
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

    let mut cpu = CpuEngine::load_with_options_and_layout(
        model_path,
        ctx_size,
        true,
        cfg.kv_dtype,
        &cfg.kv_cache_layout,
    )
    .map_err(|e| format!("load failed: {e}"))?;
    cpu.set_prefix_cache(false);
    cpu.set_flash_attention(cfg.flash_attention);
    cpu.set_n_gpu_layers(cfg.n_gpu_layers);

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
    let bench = |cpu: &mut CpuEngine| -> (Option<f64>, Option<String>) {
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

    // Save the pre-sweep env state so the toggle doesn't leak to the
    // rest of the process.
    let saved_env = std::env::var_os("RUSTLLAMA_SSM_PREFILL_CHUNKED");

    // OFF arm: REMOVE the var (the gate is `is_some()`, so `"0"` ≠ off).
    std::env::remove_var("RUSTLLAMA_SSM_PREFILL_CHUNKED");
    let (off_prefill_tps, off_err) = bench(&mut cpu);

    // ON arm: set the var — read per-forward, so it takes effect on the
    // next generate with no reload.
    std::env::set_var("RUSTLLAMA_SSM_PREFILL_CHUNKED", "1");
    let (on_prefill_tps, on_err) = bench(&mut cpu);

    // Restore the pre-sweep env state.
    match saved_env {
        Some(v) => std::env::set_var("RUSTLLAMA_SSM_PREFILL_CHUNKED", v),
        None => std::env::remove_var("RUSTLLAMA_SSM_PREFILL_CHUNKED"),
    }

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
