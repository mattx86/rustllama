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
/// Q4_K W4A8/DP4A matvec (int8 activations, HW `__dp4a`). Backs `q4k_dp4a`,
/// wired ONLY onto the batched/prefill CUDA path (decode stays bit-exact).
/// Graded by a FAIR W4A8-vs-W4A8 parity probe (not the bit-exact probe).
pub const VERDICT_MATVEC_Q4K_DP4A: &str = "matvec:q4_k_dp4a";
/// Q4_K prefill GEMM (tiled, shared-mem weight reuse, f32 — BIT-EXACT). Backs
/// `q4k_gemm_f32`, wired onto the batched/prefill CUDA path. Graded by the
/// bit-exact batched parity probe; gated fail-closed (catches a write-blind
/// kernel bug on an unvalidated arch).
pub const VERDICT_GEMM_Q4K_F32: &str = "gemm:q4_k_f32";
/// Q4_K prefill GEMM via int8 tensor cores (W8A8, LOSSY). Backs `q4k_w8a8_tc`,
/// wired onto the batched/prefill CUDA path ABOVE DP4A, BELOW the bit-exact f32
/// GEMM. Graded by a FAIR W8A8 probe (int8 activations) + the perf gate.
pub const VERDICT_GEMM_Q4K_W8A8_TC: &str = "gemm:q4_k_w8a8_tc";

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
    pub threads: Option<ThreadPlan>,
    pub batch_size: Option<u32>,
    /// Winning `[inference].kv_dtype` from the comprehensive sweep
    /// (`rustllama tune --kv-dtype` / `--all`). `None` = no entry,
    /// engine falls back to the config value. Persisted as the raw
    /// config string (e.g. `"q8_0"`, `"tq4"`, `"nvfp4"`) so cache
    /// entries survive a `KvDtype` enum bump.
    #[serde(default)]
    pub kv_dtype: Option<String>,
    /// Winning `[inference].flash_attention` setting. `None` = no
    /// entry, engine uses the config value.
    #[serde(default)]
    pub flash_attention: Option<bool>,
    /// Winning `[inference].kv_cache_layout` value (`"contiguous"`
    /// or `"paged"`). `None` = no entry.
    #[serde(default)]
    pub kv_cache_layout: Option<String>,
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
    /// Winning MoE placement mode from `rustllama tune
    /// --moe-placement` — `"uniform"` (experts follow the layer's
    /// device) or `"experts_cpu"` (routed experts + router pinned to
    /// CPU, attention/shared expert stay GPU-eligible). Consumed when
    /// `[inference].moe_placement = "auto"` +
    /// `[tuning].auto_apply_moe_placement`. `None` = no entry.
    #[serde(default)]
    pub moe_placement: Option<String>,
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
    /// Autotune winner: use the chunked-parallel SSM prefill scan (hybrid
    /// models). Some(true)=chunked prefill was faster, None=untuned.
    #[serde(default)]
    pub ssm_prefill_chunked: Option<bool>,
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
    /// ISO-8601 timestamp of the last successful tune.
    pub last_tuned: Option<String>,
    pub rustllama_version: String,
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
            threads: None,
            batch_size: None,
            kv_dtype: None,
            flash_attention: None,
            kv_cache_layout: None,
            flash_kv_min: None,
            prefix_cache_max_snapshots: None,
            kv_page_size: None,
            flash_v3_kv_tile: None,
            moe_placement: None,
            moe_gpu_split_permille: None,
            speculative_mtp: None,
            ssm_prefill_chunked: None,
            decision_calibration: HashMap::new(),
            per_device_perf: HashMap::new(),
            kernel_verdicts: HashMap::new(),
            last_tuned: None,
            rustllama_version: env!("CARGO_PKG_VERSION").to_string(),
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
    fn fingerprint_device_returns_none_in_mock_build() {
        // When no SYCL device is visible at runtime — the usual
        // `cargo test` host — the live probe should produce None
        // rather than panic or yield garbage.
        // Real-GPU builds verify the success path via the engine's
        // sycl_resources tests instead.
        assert!(fingerprint_device(0).is_none());
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
