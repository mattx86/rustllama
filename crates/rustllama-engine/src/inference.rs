//! CPU-only generation engine: pure-Rust Llama-family inference.
//!
//! Two API levels:
//!   - low-level: [`CpuEngine::generate_token_ids`] takes/returns token IDs
//!     (used by tests and direct callers that already have a tokenizer).
//!   - high-level: [`Engine::generate`] / [`Engine::chat`] take/return text;
//!     requires a tokenizer (loaded via [`CpuEngine::load_with_tokenizer`]).

use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures::Stream;
use rustllama_gguf::Gguf;
use rustllama_models::llama_arch::{KvDtype, LlamaModel};
use rustllama_tokenizer::{ChatMessage as TokChat, Tokenizer};

use crate::grammar::GrammarMask;
use crate::kv_backend::KvBackend;
use crate::prefix_cache::PrefixCachePool;
use crate::sampling::{compute_logprobs, Rng, Sampler};
use crate::speculative::{
    accept_reject, accept_reject_dispatch, accept_reject_greedy, AcceptMode, DraftToken,
    LookaheadDrafter, LookaheadDrafterConfig, MtpDrafter, NgramDrafter, NgramDrafterConfig,
};
use crate::{
    ChatMessage, Engine, GrammarKind, Metrics, RequestStats, Result as EngineResult,
    SamplingParams, Token, TokenStream,
};

#[derive(Debug, thiserror::Error)]
pub enum CpuEngineError {
    #[error("gguf: {0}")]
    Gguf(#[from] rustllama_gguf::GgufError),
    #[error("model load: {0}")]
    Model(#[from] rustllama_models::llama_arch::LlamaLoadError),
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] rustllama_tokenizer::TokenizerError),
    #[error("prompt too long: {prompt_len} tokens, max_ctx is {max_ctx}")]
    PromptTooLong { prompt_len: usize, max_ctx: usize },
    #[error("tokenizer required: {0}")]
    NoTokenizer(&'static str),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, CpuEngineError>;

/// Owns a loaded model, an open mmap of the source GGUF, and a per-engine
/// KV cache. Internal state is `Arc`-wrapped so streaming methods can move
/// shared state into a `spawn_blocking` task.
pub struct CpuEngine {
    model: Arc<LlamaModel>,
    /// F9-tail — a CANONICAL Mamba/SSM selective-scan model (arch
    /// `mamba`/`mamba2`/`jamba`/`falcon-mamba`: the `in_proj`/`x_proj`/
    /// `dt_proj`/`A_log`/`D`/`conv1d` recurrence, NOT the DeltaNet hybrid
    /// layout). `Some` when the loaded GGUF is canonical Mamba: the `model`
    /// field then holds a minimal transformer SHELL (correct cfg, empty
    /// weights, never executed — see [`LlamaModel::mamba_shell`]) and
    /// generation dispatches the forward to this selective-scan model
    /// ([`rustllama_models::mamba_arch::MambaModel`]). `None` (the default) on
    /// every transformer / hybrid model, so those paths are byte-identical.
    mamba: Option<Arc<rustllama_models::mamba_arch::MambaModel>>,
    tokenizer: Option<Arc<Tokenizer>>,
    state: Arc<Mutex<EngineState>>,
    /// Dedicated SYCL worker thread. All streaming generation and
    /// warmup work routes through this thread so the per-thread USM
    /// weight cache + SyclStreamGuard installed at engine load time
    /// persist for every subsequent request. Without this, tokio's
    /// blocking pool would land requests on arbitrary threads and
    /// each thread would re-upload model weights to USM on first
    /// generation (1-2s waste per cold thread). The worker is
    /// shut down in `Drop` with a best-effort join.
    sycl_worker: Arc<SyclWorker>,
    max_ctx: usize,
    model_id: String,
    /// On-disk path the GGUF was loaded from. Lets callers locate
    /// sidecar files (e.g., HF model-card JSON saved by `rustllama
    /// pull`) without re-walking the cache.
    source_path: std::path::PathBuf,
    prefix_cache: bool,
    /// Number of tokens to forward per prefill chunk. Drives:
    ///   - Granularity of partial-prefix-cache snapshots: after every
    ///     chunk we update `state.last_ids` so a cancellation mid-prefill
    ///     leaves the work usable for the next request's LCP.
    ///   - Chrome-trace span granularity for prefill.
    /// `forward_one` is still called per-token within a chunk; chunking
    /// does NOT change the per-token compute cost in phase 1. A future
    /// GPU backend will fuse the chunk into a single batched forward.
    prefill_chunk_size: usize,
    /// How many distinct prompt-prefix snapshots to retain across
    /// requests. Set to 0 to disable the multi-snapshot pool entirely
    /// (the engine still does single-snapshot LCP against the most
    /// recent request's tokens via `EngineState.last_ids`).
    prefix_cache_max_snapshots: usize,
    /// When `Some`, text generation routes through the n-gram
    /// (prompt-lookup) speculative-decoding driver instead of the plain
    /// single-token path. `None` (default) keeps the classic decode.
    /// Set at load time from `[inference].speculative_ngram` via
    /// [`Self::set_ngram_speculative`]. Forks inherit the parent's
    /// setting. The drafter needs no second model — it proposes tokens
    /// by matching the trailing n-gram against earlier history, so it is
    /// a pure win on repetitive / code-like outputs and a no-op (one
    /// token per round, same as classic decode) when nothing matches.
    ngram_spec: Option<NgramDrafterConfig>,
    /// F10 — lookahead / Jacobi decoding. When `Some`, grammar-free text
    /// generation routes through the lookahead (n-gram-pool) speculative
    /// driver ([`Self::speculate_lookahead_stream_from_ids`]) instead of
    /// the classic single-token path. `None` (default) keeps classic
    /// decode, so leaving it off is byte-identical. Set at load time from
    /// `[inference].speculative_lookahead` via
    /// [`Self::set_lookahead_speculative`]; forks inherit the parent's
    /// setting. Like the n-gram drafter it needs no second model — it
    /// harvests an n-gram pool from the realized history and walks it for a
    /// Jacobi window of guess tokens — but the chained pool-walk reaches
    /// deeper on loops / structured output. Placed just ABOVE `ngram_spec`
    /// in dispatch precedence (`eagle > tree > MTP > lookahead > n-gram`),
    /// so the spec autotune can pick lookahead over plain n-gram per model.
    lookahead_spec: Option<LookaheadDrafterConfig>,
    /// When `Some((draft, k))`, grammar-free text generation routes
    /// through [`Engine::speculate`] with the paired draft engine and
    /// `k` candidates per round. Takes precedence over `ngram_spec`.
    /// Set from `[inference].speculative_draft_path` at serve time
    /// after a tokenizer-compatibility check.
    draft_spec: Option<(std::sync::Arc<dyn Engine>, u32)>,
    /// When `true`, grammar-free text generation on a hybrid model that
    /// carries a NextN head routes through the MTP / NextN self-
    /// speculative driver ([`Self::speculate_mtp_stream_from_ids`])
    /// instead of the classic single-token path. Set at load time from
    /// `[inference].speculative_mtp` via [`Self::set_mtp_speculative`].
    /// Forks inherit the parent's setting (a plain `Copy` bool). Takes
    /// PRECEDENCE over `ngram_spec` when both are enabled — the NextN
    /// head is a learned drafter and beats prompt-lookup on free prose.
    /// A no-op (falls back to classic decode) when the loaded model has
    /// no NextN head or isn't hybrid, so leaving it on for a non-MTP
    /// model costs nothing.
    mtp_spec: bool,
    /// Tree-structured MTP speculation (Feature 4). `Some(widths)` enables it on
    /// a non-hybrid model with MTP heads; `None` (default) off. Per-depth
    /// branching widths from `[inference].speculative_tree_widths`. Takes
    /// precedence over `mtp_spec` / `ngram_spec`; a no-op fallback on models
    /// without MTP heads or on the hybrid path.
    tree_spec: Option<Vec<u32>>,
    /// P10b — a loaded EAGLE draft head (companion GGUF, from
    /// `[model].eagle_head` / `serve --eagle-head`). `Some` enables EAGLE
    /// tree speculation on a dense/MoE model with the contiguous KV backend:
    /// the head drafts autoregressively and the base verifies via the shared
    /// tree-spec stack ([`tree_eagle_round`]). Takes PRECEDENCE over the
    /// in-base MTP/NextN + n-gram drafters (`eagle > MTP > n-gram > none`).
    /// `None` (default) leaves every other decode path byte-identical — the
    /// fail-closed contract. Forks inherit the parent's head (`Arc`-shared).
    eagle_head: Option<Arc<rustllama_models::llama_arch::EagleHead>>,
    /// Per-request performance + cache-hit stats from the most recent
    /// generation. Reset to defaults at the start of each request and
    /// committed at the end. Reads are race-free as long as the caller
    /// holds the model's gate (Semaphore(1) serializes requests).
    last_stats: Arc<Mutex<RequestStats>>,
    /// Exponential moving average of decode tok/s, bit-cast f64 in a
    /// shared atomic so the metrics handler can read without locking.
    /// `f64::NAN` encodes "no samples yet". Updated alongside
    /// `last_stats` at each successful request completion via
    /// [`update_ema_tok_s`]. Skipped for 0-token / cancelled
    /// requests so a single broken request can't pin the EMA to 0.
    ema_tok_s_bits: Arc<std::sync::atomic::AtomicU64>,
    /// Cumulative-since-startup counters: total requests, total
    /// prefill / cache-hit / generated tokens, total prefill +
    /// decode time. Bumped on every successful request commit
    /// (both `commit_request_stats` and the `drive_generation`
    /// direct-write path). Atomic counters → lockless reads from
    /// the metrics handler.
    cumulative_stats: Arc<crate::CumulativeStats>,
    /// Immutable copy of the KV backend's dtype, cached at
    /// construction. `kv_dtype()` is polled by `/v1/metrics` (~1 Hz
    /// from the GUI); reading it off the state mutex parked a tokio
    /// worker for the entire duration of any in-flight generation
    /// (the mutex is held from prefill through commit). The backend
    /// is built exactly once per engine, so a plain copy is safe.
    kv_dtype_cached: KvDtype,
    /// Last-observed context/paged-pool numbers, published on every
    /// successful locked read so `metrics()` can fall back to a
    /// lock-free snapshot when a generation holds the state mutex.
    metrics_cache: Arc<MetricsCacheAtomics>,
    /// Cap on complete tool-call bodies the streaming grammar will
    /// accept per response. `0` disables the cap. Drives the
    /// recursion-guard in [`crate::tool_grammar::ToolCallStreamGrammar`].
    /// See [`Self::set_max_tool_iterations`].
    max_tool_iterations: u32,
    /// Whether to use the FlashAttention-decode kernel (online
    /// softmax, no scores scratch) for the F32 KV path. Default
    /// `true` to match `[inference].flash_attention`. The engine's
    /// generate paths propagate this to the model via the per-thread
    /// `rustllama_models::accel::set_flash_attention` hook before
    /// each request.
    flash_attention: bool,
    /// KV-cache layout chosen at engine load via
    /// `[inference].kv_cache_layout`. Held here so
    /// `fork_for_concurrent_use` can rebuild a matching backend on
    /// each fork. Values mirror the config field:
    /// `"contiguous"` | `"paged"`.
    kv_layout: String,
    /// `[inference].kv_page_size` — paged-KV page size. Held here
    /// so forks inherit the same page size as the parent. `0` =
    /// `KvBackend` default (matches the historical behavior).
    /// Ignored when `kv_layout == "contiguous"`.
    kv_page_size: u32,
    /// `[inference].n_gpu_layers` — the layer-count cutoff for
    /// hybrid CPU/GPU placement. Transformer layers `0..n_gpu_layers`
    /// route through the SYCL/USM dispatch ladder; layers at or
    /// beyond the cutoff fall through to CPU. Set from config at
    /// load and pushed to the per-thread TLS slot
    /// `accel::N_GPU_LAYERS` before each generate call.
    ///
    /// Default `u32::MAX` (all GPU). `0` puts every layer on CPU
    /// even when SYCL is healthy — useful for debugging the GPU
    /// path or as a fallback when VRAM is exhausted.
    n_gpu_layers: u32,
    /// `[inference].placement.overrides` entries with
    /// `device = "cpu"`, extracted to their pattern strings.
    /// Pushed to `accel::CPU_FORCE_PATTERNS` before each generate
    /// call. Empty (the default) means no overrides — every
    /// dispatch falls through to the scalar `n_gpu_layers` cutoff.
    ///
    /// V1 patterns are matched as substrings against tensor
    /// names; e.g. `"ffn"` matches `blk.5.ffn_gate.weight` and
    /// every other tensor containing "ffn". A future regex
    /// upgrade is a one-line swap on the accel side.
    cpu_force_patterns: Vec<String>,
    /// Optional vision tower (mmproj GGUF) loaded alongside the text
    /// decoder via [`CpuEngine::load_with_mmproj`]. `None` for the
    /// usual text-only configuration; `Some(_)` enables
    /// [`Engine::supports_vision`] and unlocks the V-6b-3 splice
    /// path for image-bearing chat messages.
    ///
    /// `Arc` to share with `fork_for_concurrent_use` clones without
    /// duplicating the projected weights (the vision tower is
    /// stateless beyond its loaded tensors — every call to
    /// `forward_image_bytes` produces fresh output from inputs).
    vision: Option<Arc<rustllama_models::vision_arch::VisionModel>>,
    /// Token id the **text** tokenizer assigns to the image-placeholder
    /// string the active VLM uses (e.g. `<image>` for LLaVA-1.5,
    /// `<|image_pad|>` for Qwen2-VL). `None` when no mmproj is loaded
    /// or when the text tokenizer doesn't know the model's image
    /// placeholder — the latter is a load-time error rather than a
    /// silent fallback (V-6b-2 fails fast with
    /// [`CpuEngineError::Other`]).
    image_token_id: Option<u32>,
    /// How the prompt's image-placeholder tokens map to vision-pipeline
    /// patch outputs. `OnePerImage` (LLaVA-1.5 style) is the default
    /// for backward compatibility with the original
    /// `load_with_mmproj`; `OnePerPatch` (Qwen2-VL style) is selected
    /// via [`load_with_mmproj_with_mode`]. `None` when no mmproj is
    /// loaded.
    placeholder_mode: Option<rustllama_models::vision_arch::PlaceholderMode>,
    /// Injectable placeholder text for image-bearing chat messages
    /// (D1: the engine, not the template, inserts the vision markers
    /// — even HF's real Qwen3-VL template emits ONE `<|image_pad|>`
    /// per image and lets the processor expand it). One copy per
    /// attached image is prepended to the message content before
    /// template render, idempotently. Set by `attach_mmproj`.
    image_wrapper: Option<String>,
    /// V-7 vision-feature memo: projected features of the most recent
    /// image set, keyed by the chained FNV image hash. A repeated
    /// image (multi-turn chat about one picture) skips the whole ViT
    /// + projector. One entry is enough — chats revolve around the
    /// latest image; the KV prefix cache handles deeper reuse for
    /// non-hybrid models.
    vision_feature_memo: Arc<Mutex<Option<(u64, Vec<Vec<f32>>)>>>,
    /// Windows memory-residency: VirtualLock'd hot-weight ranges that
    /// keep the model resident in RAM (off the pagefile). `Some` when
    /// `RUSTLLAMA_LOCK_RAM_MB` is set and at least one range was locked;
    /// `None` otherwise (disabled, non-Windows, or nothing lockable).
    /// `Arc`-shared so `fork_for_concurrent_use` clones inherit the same
    /// locks (the forked engines share the parent's `Arc<LlamaModel>`,
    /// so the pages are already pinned) and the ranges unlock only when
    /// the last engine handle drops.
    lock_registry: Option<Arc<crate::pagelock::LockRegistry>>,
    /// MoE expert-cache learning driver (`Some` only on the engine
    /// that loaded a MoE model with the expert cache enabled; forks
    /// get `None` — one sidecar flusher per model). A background
    /// thread holding a `Weak` flushes routing stats periodically;
    /// [`Drop`] does the final flush.
    expert_usage: Option<Arc<Mutex<crate::expert_cache::UsageLearner>>>,
    /// Warm-restart persistence owner (roadmap Phase 5): `true` only
    /// on the engine that loaded the GGUF — its [`Drop`] writes the
    /// prefix pool to `<model>.rlkv`. Forks share the parent's pool
    /// concept but must not each write the sidecar.
    kv_persist_owner: bool,
    /// CacheBlend recompute config for RAG `documents` requests. `None` =
    /// CacheBlend off (the default; requests fall back to a normal prefill).
    /// Set at serve load from the autotuned ratio (`tune --cacheblend` winner,
    /// when `[tuning].auto_apply_cacheblend`) or the
    /// `[inference].cacheblend_recompute_ratio` fallback, via
    /// [`Self::set_cacheblend`]. Forks inherit it (plain `Copy`).
    cacheblend: Option<rustllama_models::cacheblend::CacheBlendConfig>,
    /// F12 — periodic KV-checkpoint clock. Gates how often the warm-restart
    /// sidecar is re-written mid-session (every `kv_persist_interval_secs`). A
    /// zero interval (the default) leaves it inert → save-on-drop only, exactly
    /// the pre-F12 behavior. Only the sidecar-owning engine
    /// (`kv_persist_owner`) ever fires it; forks carry an inert clock.
    checkpoint_clock: Mutex<crate::kv_persist::CheckpointClock>,
    /// P11 Phase 2/3 — resident multi-LoRA adapter registry. Adapters are
    /// attached UNMERGED (held as low-rank deltas) via
    /// [`Self::attach_lora_adapter`] and selected per request through
    /// [`SamplingParams::lora_adapter`]; the selected adapter's delta is applied
    /// around the base matvecs by scoping [`rustllama_models::lora::ACTIVE_LORA`]
    /// for the request's forwards (see [`Self::lora_adapter_for`] +
    /// `drive_generation`). Empty (the default) ⇒ no adapters resident ⇒ every
    /// request is byte-identical to the pre-LoRA path.
    ///
    /// `Arc<RwLock<..>>` so `fork_for_concurrent_use` clones SHARE one registry
    /// (an adapter attached at runtime is visible to every fork) and per-request
    /// lookups (reads) don't contend with the rare attach (write).
    lora: Arc<std::sync::RwLock<rustllama_models::lora::LoraRegistry>>,
}

impl Drop for CpuEngine {
    fn drop(&mut self) {
        // Final learning-cache flush: fold this session's routing
        // stats into the model's usage sidecar so the next load
        // pre-pins a warmer set. The periodic flusher thread exits
        // on its own once its `Weak` fails to upgrade.
        if let Some(usage) = &self.expert_usage {
            if let Ok(mut learner) = usage.lock() {
                learner.flush(true);
            }
        }
        // Warm restart (Phase 5): persist the prefix pool so the next
        // session's matching prompts skip re-prefill.
        self.save_kv_persist();
    }
}

/// Default number of prefill tokens to process before snapshotting the
/// partial prefix cache. Matches the typical `batch_size` config knob.
const DEFAULT_PREFILL_CHUNK: usize = 512;

/// Default size of the cross-request prompt prefix cache pool. Each
/// snapshot is clipped to the seq_len that was actually filled, so the
/// memory cost scales with conversation length rather than `max_ctx`.
/// 4 entries comfortably covers "two concurrent chat threads + a
/// scratch generation" without exploding RAM on typical hosts.
const DEFAULT_PREFIX_CACHE_MAX_SNAPSHOTS: usize = 4;

/// Default cap on complete tool-call bodies per response. 8 covers
/// "single agent turn that picks a tool, sees the result, picks another"
/// patterns with room for a couple of corrective retries; well-behaved
/// templates emit at most 1-2 per turn anyway.
const DEFAULT_MAX_TOOL_ITERATIONS: u32 = 8;

/// EMA weight on the newest tok/s sample. Lower = more smoothing,
/// slower convergence; higher = more responsive, more jitter.
/// 0.2 means each new sample contributes 20% — a single outlier
/// can only move the visible number by ~20%, but a sustained step
/// change (driver clock state, model swap) still reaches
/// steady-state within ~7 requests. Pin the value here rather than
/// expose a knob — the tuner will sweep the kernel-level params,
/// and we don't want EMA tuning entangled with that.
const TOK_S_EMA_ALPHA: f64 = 0.2;

/// Fold a freshly-completed request's decode throughput into the
/// shared EMA. Skips zero-token / zero-time requests (cancelled,
/// rejected, failed) so the EMA reflects "what generation actually
/// looks like" rather than "average including aborts".
///
/// Bit-casts f64 through u64 to keep the slot lock-free for the
/// `/v1/metrics` reader. The write isn't synchronized against
/// concurrent writers (we trust the single-flight gate to serialize
/// stats updates per engine).
fn update_ema_tok_s(slot: &std::sync::atomic::AtomicU64, stats: &RequestStats) {
    use std::sync::atomic::Ordering;
    if stats.decode_ms <= 0.0 || stats.tokens_generated == 0 {
        return;
    }
    let sample = stats.tokens_generated as f64 / (stats.decode_ms / 1000.0);
    let prev = f64::from_bits(slot.load(Ordering::Relaxed));
    let next = if prev.is_nan() {
        // First sample seeds the EMA — no smoothing yet.
        sample
    } else {
        TOK_S_EMA_ALPHA * sample + (1.0 - TOK_S_EMA_ALPHA) * prev
    };
    slot.store(next.to_bits(), Ordering::Relaxed);
}

/// If `RUSTLLAMA_SYCL_DISPATCH=1` is set, install a SYCL stream on the
/// current thread so the forward pass routes its SYCL-aware ops
/// through the GPU (with CPU fallback per-call on `Unavailable` /
/// shape rejection). Currently RMSNorm and SwiGLU consult the TLS
/// stream; other kernels follow as their parity checks pass on real
/// hardware.
///
/// Returns `None` when no SYCL device is visible, or the user
/// explicitly disabled dispatch via `RUSTLLAMA_SYCL_DISPATCH=0`.
/// Otherwise installs the SYCL guard so kernel calls have a TLS
/// stream to dispatch through; the per-thread circuit breaker
/// (`SYCL_FAIL_BUDGET` failures) flips the path back to CPU if the
/// driver rejects them.
fn install_sycl_dispatch_if_requested() -> Option<rustllama_models::accel::SyclStreamGuard> {
    let explicit_off = |name: &str| -> bool {
        std::env::var(name)
            .map(|v| v == "0" || v.eq_ignore_ascii_case("false"))
            .unwrap_or(false)
    };
    if explicit_off("RUSTLLAMA_SYCL_DISPATCH") {
        tracing::info!("SYCL dispatch disabled via env var; CPU path only");
        return None;
    }
    // Pick the first non-disabled SYCL device (respects the GUI/config
    // disable-list). `None` => every SYCL GPU is disabled → CPU path.
    let Some(dev) = rustllama_models::accel::first_enabled_sycl_device_index() else {
        tracing::info!(
            "all SYCL GPUs are in the disable-list (RUSTLLAMA_DISABLED_GPUS); CPU path only"
        );
        return None;
    };
    let g = rustllama_models::accel::SyclStreamGuard::install(dev);
    if g.is_some() {
        tracing::info!(
            device = dev,
            "SYCL dispatch enabled — RMSNorm + SwiGLU + matvec routing via SYCL device {dev}"
        );
    } else {
        tracing::debug!("no SYCL device visible; CPU path only");
    }
    g
}

// ============================================================
// Dedicated SYCL worker thread
// ============================================================
//
// Every CpuEngine instance owns one dedicated OS thread that processes
// all streaming generation + USM warmup work. The thread installs a
// SyclStreamGuard once at startup and keeps it alive for its entire
// lifetime, so the per-thread USM weight cache + the SYCL queue
// persist across every request without re-creation.
//
// Why this exists: tokio's blocking pool dispatches `spawn_blocking`
// tasks to a thread pool of variable size; consecutive requests can
// (and do) land on different threads. Each fresh thread's TLS context
// is empty → it must re-prepare the USM context AND re-upload model
// weights (1-2s for a 7B Q4_K_M model on Iris Xe). With the dedicated
// worker, that cost is paid exactly once per engine load, at warmup
// time.
//
// Concurrency model: serial. The worker processes work items in
// submission order. Single-flight at the server gate already
// serializes concurrent chats, so the worker's queue rarely has more
// than one pending item.

/// Single unit of work for the dedicated SYCL worker thread.
enum SyclWorkItem {
    /// Stream a full chat/generate response. The worker calls
    /// `drive_generation` and yields tokens via the embedded
    /// `response_tx`. Boxed so the enum stays small.
    DriveGeneration(Box<DriveGenJob>),
    /// Pre-upload packed-quant weights to USM. Sends the upload
    /// count back via `done_tx` on completion. Boxed for parity
    /// with `DriveGeneration`.
    Warmup(Box<WarmupJob>),
    /// Run an arbitrary closure on the worker thread and drop it when
    /// done. The closure owns its own result channel (a oneshot), so
    /// the item itself carries no return path. Used by the speculative
    /// / n-gram decode paths: their per-round verify-forward would
    /// otherwise run on a tokio blocking thread with no SYCL guard /
    /// USM context installed → 100% CPU dispatch regardless of config.
    /// Running it here reuses the worker's persistent SYCL stream +
    /// warmed USM weight cache, exactly like `DriveGeneration`.
    RunOnWorker(Box<dyn FnOnce() + Send>),
    /// Terminate the worker loop. Sent from `SyclWorker::Drop`.
    Shutdown,
}

/// Captured arguments for a `drive_generation` call dispatched
/// through the worker thread. All `Arc`-wrapped so the worker can
/// own them for the duration of the job without borrowing back to
/// the engine.
/// Lock-free mirror of the state-mutex-guarded numbers `/v1/metrics`
/// wants. Updated opportunistically whenever `metrics()` wins a
/// `try_lock`; read when it doesn't. Staleness is bounded by the
/// polling interval and is exactly what a 1 Hz status page wants —
/// the alternative was blocking a reactor thread for a whole
/// generation.
#[derive(Default)]
struct MetricsCacheAtomics {
    seq_len: std::sync::atomic::AtomicU32,
    paged_total: std::sync::atomic::AtomicU32,
    paged_free: std::sync::atomic::AtomicU32,
}

struct DriveGenJob {
    model: Arc<LlamaModel>,
    state: Arc<Mutex<EngineState>>,
    tokenizer: Arc<Tokenizer>,
    max_ctx: usize,
    model_eos: Option<u32>,
    prefix_cache: bool,
    max_tool_iterations: u32,
    prompt: String,
    sampling: SamplingParams,
    last_stats: Arc<Mutex<RequestStats>>,
    ema_tok_s_bits: Arc<std::sync::atomic::AtomicU64>,
    cumulative_stats: Arc<crate::CumulativeStats>,
    response_tx: tokio::sync::mpsc::Sender<EngineResult<Token>>,
    /// Per-thread dispatch state the generation must install before
    /// its first forward. The worker thread's TLS defaults are
    /// "flash on, ALL layers GPU, no CPU-force patterns" — without
    /// these fields the streaming path silently ignored
    /// `[inference].n_gpu_layers`, auto-placement, and
    /// `placement.overrides` (the sync `generate_token_ids*` paths
    /// always installed them; this path did not).
    flash_attention: bool,
    n_gpu_layers: u32,
    cpu_force_patterns: Vec<String>,
    /// P11 — the resident LoRA adapter selected for this request (resolved from
    /// `sampling.lora_adapter` against the engine's registry in `spawn_stream`),
    /// or `None` for the un-adapted base. `drive_generation` scopes it as the
    /// thread-active adapter for the whole generation on the worker thread, so
    /// `None` keeps the streaming path byte-identical.
    lora_adapter: Option<Arc<rustllama_models::lora::LoraAdapter>>,
}

/// Captured arguments for a USM weight pre-upload dispatched through
/// the worker thread. `done_tx` is signaled with the upload count
/// once the work completes (0 on no-op or failure).
struct WarmupJob {
    model: Arc<LlamaModel>,
    max_ctx: usize,
    /// Hybrid-placement cutoff at warmup time. `u32::MAX` keeps
    /// the historic "upload everything" behavior; smaller values
    /// skip per-block weight uploads for transformer layers at
    /// or above the cutoff, matching the per-call dispatch
    /// gating in `accel::gpu_active_for_current_layer`.
    n_gpu_layers: u32,
    done_tx: tokio::sync::oneshot::Sender<usize>,
}

/// Handle to the dedicated worker thread. Created in
/// `CpuEngine::load_inner`, shut down in `CpuEngine::Drop`.
///
/// `tx` is `std::sync::mpsc::Sender` (synchronous) because:
///   - submitters are sometimes inside `tokio::task::spawn_blocking`
///     already (where async sends would deadlock the pool)
///   - the channel never blocks: send is non-blocking on an unbounded
///     channel, and we never need backpressure here (single-flight
///     at the server gate already throttles).
pub(crate) struct SyclWorker {
    tx: std::sync::mpsc::Sender<SyclWorkItem>,
    /// Inside a `Mutex<Option<...>>` so `Drop` can `take()` the
    /// handle and `join()` it without needing `&mut self`.
    join: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl SyclWorker {
    /// Spawn the worker thread. The thread runs until it receives a
    /// `Shutdown` work item (sent from `Drop`) or the sender is
    /// dropped.
    fn spawn() -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<SyclWorkItem>();
        let join = std::thread::Builder::new()
            .name("rustllama-sycl-worker".into())
            .spawn(move || run_worker(rx))
            .expect("spawn rustllama-sycl-worker thread");
        Self {
            tx,
            join: std::sync::Mutex::new(Some(join)),
        }
    }

    /// Submit a streaming generation. If the worker has already shut
    /// down (engine dropped mid-request), the `response_tx` inside
    /// the job is silently dropped — the receiver sees a closed
    /// channel and the SSE stream ends cleanly.
    fn submit_generation(&self, job: DriveGenJob) {
        let _ = self.tx.send(SyclWorkItem::DriveGeneration(Box::new(job)));
    }

    /// Submit a warmup. Same drop semantics as `submit_generation`:
    /// if the worker is gone, `done_tx` is dropped and the awaiter
    /// receives an error which the caller treats as "0 uploaded".
    fn submit_warmup(&self, job: WarmupJob) {
        let _ = self.tx.send(SyclWorkItem::Warmup(Box::new(job)));
    }

    /// A cheap, cloneable handle for submitting `RunOnWorker` closures
    /// from inside an `async_stream` generator, which can't hold a
    /// `&SyclWorker` across its await points. See [`WorkerSubmitter`].
    fn submitter(&self) -> WorkerSubmitter {
        WorkerSubmitter {
            tx: self.tx.clone(),
        }
    }
}

/// Cloneable submitter for one-off closures that must run on the SYCL
/// worker thread (see [`SyclWorkItem::RunOnWorker`]). Holds a clone of
/// the worker's channel sender, so it can be moved into a generator and
/// used across await points where `&SyclWorker` cannot.
#[derive(Clone)]
struct WorkerSubmitter {
    tx: std::sync::mpsc::Sender<SyclWorkItem>,
}

impl WorkerSubmitter {
    /// Run `f` on the worker thread and return a oneshot receiver for
    /// its result; `await` the receiver from async code. If the worker
    /// has shut down (engine dropped mid-request) the send fails and
    /// the sender half is dropped, so the receiver resolves to
    /// `Err(RecvError)` — callers surface that as an engine error and
    /// end the stream. If the receiver is dropped first (stream
    /// cancelled), the result send is a silent no-op; the work still
    /// ran to completion, which keeps the worker's KV/DN state
    /// consistent.
    fn run_blocking<T, F>(&self, f: F) -> tokio::sync::oneshot::Receiver<T>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let (done_tx, done_rx) = tokio::sync::oneshot::channel();
        let job: Box<dyn FnOnce() + Send> = Box::new(move || {
            let _ = done_tx.send(f());
        });
        let _ = self.tx.send(SyclWorkItem::RunOnWorker(job));
        done_rx
    }
}

impl Drop for SyclWorker {
    fn drop(&mut self) {
        // Tell the worker to wind down after any in-flight item, then
        // join. Best-effort: if the channel is already closed (worker
        // panicked) the send fails silently and we still join.
        let _ = self.tx.send(SyclWorkItem::Shutdown);
        if let Some(j) = self.join.lock().ok().and_then(|mut g| g.take()) {
            // Engine drops are rare (process shutdown, model swap), so
            // briefly waiting for the worker to finish its current item
            // is acceptable. We don't have a stdlib timeout primitive,
            // so a stuck worker would hang Drop — but the only way for
            // that to happen is a kernel hanging, which is its own bug.
            let _ = j.join();
        }
    }
}

/// Worker thread main loop. Installs the SYCL stream + flash-attn
/// flag once at startup; thereafter just dispatches work items.
fn run_worker(rx: std::sync::mpsc::Receiver<SyclWorkItem>) {
    // Install once for the worker's lifetime. All subsequent
    // `drive_generation` calls reuse this guard — the helper inside
    // `drive_generation` skips its own install when an outer one is
    // already present (see `has_sycl_stream` check there).
    let _guard = install_sycl_dispatch_if_requested();
    rustllama_models::accel::set_flash_attention(true);
    while let Ok(item) = rx.recv() {
        match item {
            SyclWorkItem::DriveGeneration(job) => {
                let job = *job;
                let DriveGenJob {
                    model,
                    state,
                    tokenizer,
                    max_ctx,
                    model_eos,
                    prefix_cache,
                    max_tool_iterations,
                    prompt,
                    sampling,
                    last_stats,
                    ema_tok_s_bits,
                    cumulative_stats,
                    response_tx,
                    flash_attention,
                    n_gpu_layers,
                    cpu_force_patterns,
                    lora_adapter,
                } = job;
                if let Err(e) = drive_generation(
                    model,
                    state,
                    tokenizer,
                    max_ctx,
                    model_eos,
                    prefix_cache,
                    max_tool_iterations,
                    prompt,
                    sampling,
                    last_stats,
                    ema_tok_s_bits,
                    cumulative_stats,
                    &response_tx,
                    flash_attention,
                    n_gpu_layers,
                    cpu_force_patterns,
                    lora_adapter,
                ) {
                    let _ = response_tx.blocking_send(Err(e));
                }
            }
            SyclWorkItem::Warmup(job) => {
                let job = *job;
                let n = run_warmup(&job.model, job.max_ctx, job.n_gpu_layers);
                let _ = job.done_tx.send(n);
            }
            SyclWorkItem::RunOnWorker(f) => f(),
            SyclWorkItem::Shutdown => break,
        }
    }
}

/// Body of a warmup work item. Prepares the per-thread USM context
/// and pre-uploads every packed-quant weight tensor for the
/// transformer layers below `n_gpu_layers`. Returns the upload
/// count (0 on no-op / SYCL unavailable).
fn run_warmup(model: &LlamaModel, max_ctx: usize, n_gpu_layers: u32) -> usize {
    let cfg = &model.cfg;
    let head_dim = if cfg.head_dim > 0 {
        cfg.head_dim
    } else {
        cfg.d_model / cfg.n_heads.max(1)
    };
    if !rustllama_models::accel::prepare_usm_context(
        cfg.n_layers as u32,
        cfg.n_heads as u32,
        cfg.n_kv_heads as u32,
        head_dim as u32,
        max_ctx as u32,
    ) {
        return 0;
    }
    // One-time-per-process probe: does the L0 driver accept Win32
    // file-mapping HANDLE imports via `zeMemAllocHost`? Result goes
    // to `gui.log` as info/warn. Gates whether the GGUF zero-copy
    // refactor (Step 1c) is worth doing on this driver.
    rustllama_models::accel::probe_l0_import_once();
    let (uploaded, _skipped, _bytes) =
        model.preload_packed_weights_to_usm_with_cutoff(n_gpu_layers);
    rustllama_models::accel::mark_packed_weights_preloaded();
    uploaded
}

/// Coupled KV cache + the token sequence it currently reflects. Tracking
/// `last_ids` alongside the cache lets us reuse cached K/V entries across
/// independent generation calls when the new prompt shares a prefix with
/// the previous one — a huge win for chat / coding-assistant flows where
/// the system prompt + prior turns are resent on each request.
///
/// Invariants:
///   - `kv.seq_len == last_ids.len()` whenever the cache is in a clean
///     post-generation state.
///   - `last_ids[i]` is the token id that was processed at position `i`.
///
/// The `pool` holds clipped snapshots of past conversations so a request
/// that doesn't match `last_ids` can still skip prefill if a different
/// past prompt shared a prefix. See [`crate::prefix_cache`].
/// Build a `DeltaNetCache` for the model, or `None` for non-hybrid
/// models. Centralized so all three `EngineState` construction sites
/// (load, load_with_options, fork_for_concurrent_use) stay consistent.
fn build_delta_net_cache(
    model: &LlamaModel,
) -> Result<Option<rustllama_models::llama_arch::DeltaNetCache>> {
    // Non-hybrid models carry no DeltaNet layers → no cache (not an error).
    let Some(hybrid) = model.weights.hybrid_layers.as_ref() else {
        return Ok(None);
    };
    // A hybrid model with inconsistent DeltaNet geometry now returns a
    // clean `UnsupportedHybridGeometry` error (surfaced here as the load
    // error) instead of panicking inside the cache constructor.
    Ok(Some(
        rustllama_models::llama_arch::DeltaNetCache::new_for_hybrid(&model.cfg, hybrid)?,
    ))
}

/// Per-layer KV keep-mask for sparse contiguous allocation (memory
/// reclaim): hybrid models keep only the full-attention layers the
/// main forward actually runs — DeltaNet layers never write KV, and
/// the MTP block is excluded from the forward entirely (unless
/// `RUSTLLAMA_INCLUDE_MTP_LAYERS` opts it back in). On qwen35moe-35B
/// that skips 31 of 41 slabs (~992 MiB at ctx 8192 / F32). Dense
/// models return `None` — every layer allocates as before.
fn hybrid_kv_keep_mask(model: &LlamaModel) -> Option<Vec<bool>> {
    let layers = model.weights.hybrid_layers.as_ref()?;
    let n_mtp = model
        .cfg
        .hybrid
        .as_ref()
        .map(|h| h.nextn_predict_layers as usize)
        .unwrap_or(0);
    let main = if std::env::var("RUSTLLAMA_INCLUDE_MTP_LAYERS").is_ok() {
        layers.len()
    } else {
        layers.len().saturating_sub(n_mtp)
    };
    Some(
        layers
            .iter()
            .enumerate()
            .map(|(i, l)| {
                matches!(
                    l,
                    rustllama_models::llama_arch::HybridLayer::FullAttention(_)
                ) && i < main
            })
            .collect(),
    )
}

struct EngineState {
    kv_backend: KvBackend,
    last_ids: Vec<u32>,
    /// Prefix cache pool for the `Contiguous` backend — snapshot/LCP reuse
    /// (the contiguous KV has no per-position block addressing). The `Paged`
    /// backends reuse via `apc` (block-level vLLM automatic prefix caching)
    /// instead; see `prepare_prefix_reuse`. The pool stays empty on paged.
    pool: PrefixCachePool,
    /// DeltaNet recurrent state for `qwen35moe`-family hybrid
    /// models. `Some` when the loaded model carries
    /// `hybrid_layers`; `None` for standard transformer + dense /
    /// MoE models. Phase 3.7b plumbing — forward routes here for
    /// SSM layers via `forward_one_hybrid`. Prefix-cache restore
    /// is not yet wired for hybrid models; restoring resets this
    /// cache to its initial state.
    delta_net_cache: Option<rustllama_models::llama_arch::DeltaNetCache>,
}

impl EngineState {
    fn reset(&mut self) {
        self.kv_backend.reset();
        self.last_ids.clear();
        if let Some(dn) = self.delta_net_cache.as_mut() {
            dn.reset();
        }
    }

    // (kv_contiguous / kv_contiguous_mut helpers removed — no
    // callers in 3.6e. Re-add when spec decode or prefix cache
    // paths need direct `&KvCache` access on a contiguous backend.)

    /// Pick the best prefix to reuse for `prompt_u32` across both the
    /// live state (`last_ids`) and the pool, restoring K/V data from
    /// the pool if it wins. Sets `kv.seq_len` to the chosen LCP and
    /// updates `last_ids` to match. Returns the LCP that ended up
    /// reflected in the live state (== new `kv.seq_len`).
    ///
    /// `min_reuse` is the threshold below which we treat the match as
    /// "not worth the overhead" and start fresh from seq_len=0.
    /// `prompt_max` caps the LCP at `prompt_u32.len() - 1` so prefill
    /// always processes at least the last input token (its logits
    /// drive the first sample). Pass `prompt_u32.len().saturating_sub(1)`.
    fn prepare_prefix_reuse(
        &mut self,
        prompt_u32: &[u32],
        enabled: bool,
        min_reuse: usize,
        prompt_max: usize,
    ) -> usize {
        // The Contiguous backend reuses via the snapshot `pool` (below); the
        // Paged backends reuse via the vLLM APC index (`self.apc`) — block-level,
        // content-addressed reuse that revives even a completed request's freed
        // KV by content hash. Same mechanism the paged serving engine uses.
        let kv = match &mut self.kv_backend {
            KvBackend::Contiguous(kv) => kv,
            KvBackend::Paged { cache, table, .. }
            | KvBackend::PagedQ8_0 { cache, table, .. }
            | KvBackend::PagedTQ { cache, table, .. }
            | KvBackend::PagedNvfp4 { cache, table, .. }
            | KvBackend::PagedMxfp4 { cache, table, .. }
            | KvBackend::PagedMxfp6 { cache, table, .. }
            | KvBackend::PagedMxfp8 { cache, table, .. }
            | KvBackend::PagedF16 { cache, table, .. } => {
                use rustllama_models::apc;
                let page_size = table.page_size() as usize;
                if !enabled || prompt_max == 0 || page_size == 0 {
                    cache.release(table);
                    self.last_ids.clear();
                    return 0;
                }
                // Register the just-finished request's full blocks (prompt +
                // generated tokens, in `last_ids`) against the pages that still
                // hold their KV — block i ↔ page i — so a later prompt can
                // reacquire them by content hash. BEFORE release (pages valid).
                if !self.last_ids.is_empty() {
                    let chain = apc::block_hash_chain(&self.last_ids, page_size);
                    let pages = cache.pages();
                    let n = chain.len().min(pages.len());
                    for i in 0..n {
                        table.apc_register(chain[i], pages[i]);
                    }
                }
                // Free the finished request's pages: they drop to refcount 0 but
                // keep their contents + index entries, so the reacquire below can
                // revive the matching prefix instead of recomputing it.
                cache.release(table);
                self.last_ids.clear();
                // vLLM APC: reacquire the longest matching block prefix of the
                // NEW prompt into the now-fresh cache; prefill then starts at
                // that boundary and the caller's ensure_capacity grows the
                // suffix pages exactly as it would without reuse. The index lives
                // in the table, so the suffix's `alloc` auto-prunes stale hashes.
                let chain = apc::block_hash_chain(prompt_u32, page_size);
                let shared_pages = table.acquire_prefix(&chain);
                let shared_lcp = shared_pages.len() * page_size;
                if !shared_pages.is_empty() {
                    cache.pin_shared_prefix(shared_pages, shared_lcp as u32);
                }
                return shared_lcp;
            }
        };
        // P2: a contiguous SWA ring cache only physically holds the last
        // `window` positions per ring layer, so the snapshot-pool prefix
        // reuse (which assumes a full `max_ctx` history) can't restore it.
        // Treat it like "reuse disabled": reset + full re-prefill. Correct
        // (never serves stale/overwritten rows); the only cost is a missed
        // prefix-cache optimization on SWA models.
        //
        // M1: MLA (DeepSeek-V2/V3) stores its KV in `kv.mla` (the compressed
        // latent), NOT in `kv.layers` — the snapshot pool's `snapshot_prefix`/
        // `restore_prefix` only walk `kv.layers` (empty placeholders for MLA),
        // so a restored "prefix" would carry no latents. Same treatment: reset
        // + full re-prefill (correct; prefix-cache reuse for MLA is a follow-up).
        if !enabled || prompt_max == 0 || kv.has_ring_layers() || kv.mla.is_some() {
            kv.reset();
            if let Some(dn) = self.delta_net_cache.as_mut() {
                dn.reset();
            }
            self.last_ids.clear();
            return 0;
        }
        // Hybrid (DeltaNet) models: the recurrent state has no
        // per-position addressing, so reuse is **anchor-based**
        // (roadmap Phase 5, FreeToken's semantic checkpoints). Two
        // sound paths: (a) the new prompt exactly extends the live
        // state — continue in place; (b) the new prompt fully
        // contains a pooled anchor (KV + recurrent state captured
        // together at a turn boundary) — restore the anchor whole
        // and re-prefill only the suffix. Anything else re-prefills
        // from scratch with reset recurrent state. This is what
        // makes agent context edits (truncate to an earlier turn,
        // then extend differently) cheap on hybrid models: the
        // per-turn anchors in the pool survive the edit.
        if self.delta_net_cache.is_some() {
            let live_lcp = compute_lcp(prompt_u32, &self.last_ids);
            let live_exact = !self.last_ids.is_empty()
                && live_lcp == self.last_ids.len()
                && live_lcp <= prompt_max
                && live_lcp >= min_reuse;
            let live_len = if live_exact { live_lcp } else { 0 };
            // Stash the live state as an anchor before any restore
            // clobbers it (mirrors the dense branch's stash) — the
            // current conversation stays warm if a different thread's
            // request lands in between.
            if self.last_ids.len() >= PREFIX_REUSE_MIN_TOKENS && kv.seq_len == self.last_ids.len() {
                if let Some(dn) = self.delta_net_cache.as_ref() {
                    let ids = self.last_ids.clone();
                    self.pool.snapshot_and_insert_hybrid(ids, kv, dn);
                }
            }
            let pool_best = self
                .pool
                .find_full_match_hybrid(prompt_u32)
                .filter(|(_, len)| *len <= prompt_max && *len >= min_reuse && *len > live_len);
            self.pool.record_lookup(pool_best.is_some());
            if let Some((idx, len)) = pool_best {
                let snap = self.pool.touch(idx);
                kv.restore_prefix(&snap.kv);
                kv.seq_len = len;
                let dn = self
                    .delta_net_cache
                    .as_mut()
                    .expect("hybrid branch requires a DeltaNetCache");
                dn.restore(snap.dn.as_ref().expect("anchor entries always carry dn"));
                self.last_ids = snap.ids.clone();
                return len;
            }
            if live_exact {
                kv.seq_len = live_lcp;
                return live_lcp;
            }
            kv.reset();
            if let Some(dn) = self.delta_net_cache.as_mut() {
                dn.reset();
            }
            self.last_ids.clear();
            return 0;
        }
        // Live (current state) LCP.
        let live_lcp = compute_lcp(prompt_u32, &self.last_ids).min(prompt_max);
        // Best pool LCP (if any).
        let pool_best = self.pool.find_best(prompt_u32);
        // H5: telemetry — record hit/miss for prefix-cache hit-rate metric.
        self.pool.record_lookup(pool_best.is_some());
        let (pool_idx, pool_lcp) = match pool_best {
            Some((i, l)) => (Some(i), l.min(prompt_max)),
            None => (None, 0),
        };

        if pool_lcp > live_lcp {
            // The pool has a longer-matching prefix than the live state.
            // Stash the current live state into the pool (if it's worth
            // keeping — a non-trivial prefix) so we don't lose it, then
            // restore the pool's winning entry into the live state.
            if self.last_ids.len() >= PREFIX_REUSE_MIN_TOKENS && kv.seq_len > 0 {
                let ids = self.last_ids.clone();
                self.pool.snapshot_and_insert(ids, kv);
            }
            let _ = pool_idx;
            // Re-find after the stash (insert dedup can reorder or
            // replace entries), then restore by BORROW — the previous
            // `.clone()` of the whole snapshot was a full-KV transient
            // (+1.28 GiB at max ctx on a 41-layer dense model). The
            // stash can only have replaced the winner with a superset
            // of it, so the re-found LCP is ≥ pool_lcp.
            let refound = self
                .pool
                .find_best(prompt_u32)
                .map(|(i, l)| (i, l.min(prompt_max)))
                .filter(|(_, l)| *l >= min_reuse);
            if let Some((idx, lcp)) = refound {
                let snap = self.pool.touch(idx);
                kv.restore_prefix(&snap.kv);
                let mut ids = snap.ids.clone();
                ids.truncate(lcp);
                kv.seq_len = lcp;
                self.last_ids = ids;
                return lcp;
            }
            kv.reset();
            self.last_ids.clear();
            0
        } else if live_lcp >= min_reuse {
            kv.seq_len = live_lcp;
            // Keep last_ids as-is; the LCP guarantees the relevant
            // prefix matches prompt_u32, and downstream code overwrites
            // last_ids after generation anyway.
            live_lcp
        } else {
            kv.reset();
            self.last_ids.clear();
            0
        }
    }

    /// Stamp the post-generation state. Updates `last_ids` and pushes a
    /// clipped snapshot into the pool so the next request from a
    /// different conversation thread can still benefit from this work.
    fn commit_snapshot(&mut self, ids: Vec<u32>) {
        self.last_ids = ids.clone();
        // Only contiguous KV is snapshottable today. For paged we
        // still update last_ids so live-LCP-only optimizations in
        // later turns can still match — but skip the pool snapshot.
        if ids.len() >= PREFIX_REUSE_MIN_TOKENS {
            if let KvBackend::Contiguous(kv) = &self.kv_backend {
                // P2: don't pool-snapshot a SWA ring cache — its ring layers
                // don't hold the full prefix, so a later restore would be
                // wrong (prefix reuse is disabled for ring models in
                // `prepare_prefix_reuse`; this keeps the pool free of them).
                // M1: same for MLA — its KV lives in `kv.mla`, not the
                // (empty) `kv.layers` the pool snapshots; reuse is disabled
                // for MLA too, so keep its useless empty snapshots out.
                if kv.has_ring_layers() || kv.mla.is_some() {
                    return;
                }
                match self.delta_net_cache.as_ref() {
                    // Hybrid: a turn end is a semantic anchor —
                    // capture KV + recurrent state together (Phase 5)
                    // so later requests can restore to this boundary.
                    Some(dn) => self.pool.snapshot_and_insert_hybrid(ids, kv, dn),
                    None => self.pool.snapshot_and_insert(ids, kv),
                }
            }
        }
    }
}

/// Minimum LCP length required to actually reuse cached K/V entries.
/// Below this, the overhead of the LCP check + cache lookup isn't worth
/// the few skipped prefill forward passes.
const PREFIX_REUSE_MIN_TOKENS: usize = 8;

/// Chrome-tracing event recorder. Active only when `RUSTLLAMA_CHROME_TRACE`
/// is set to a path; otherwise all `span` calls compile down to a single
/// branch on the `enabled` flag.
///
/// Output is an array of "X" (complete-duration) events that loads in
/// chrome://tracing or perfetto.dev. The first event of the run defines
/// the epoch; subsequent events report `ts` (microseconds since epoch)
/// and `dur` (microseconds) so the viewer can render a flame graph of
/// prefill / per-token-forward / sample / detokenize phases.
struct ChromeTracer {
    enabled: bool,
    path: std::path::PathBuf,
    epoch: std::time::Instant,
    events: Vec<serde_json::Value>,
    pid: u32,
}

impl ChromeTracer {
    fn from_env() -> Self {
        let (enabled, path) = match std::env::var("RUSTLLAMA_CHROME_TRACE") {
            Ok(p) if !p.is_empty() => (true, std::path::PathBuf::from(p)),
            _ => (false, std::path::PathBuf::new()),
        };
        Self {
            enabled,
            path,
            epoch: std::time::Instant::now(),
            events: Vec::new(),
            pid: std::process::id(),
        }
    }

    /// Record a duration event named `name` that started at `start` and
    /// ended at the moment of this call. No-op when the tracer is off.
    fn span(&mut self, name: &str, start: std::time::Instant) {
        if !self.enabled {
            return;
        }
        let ts = start.duration_since(self.epoch).as_micros() as u64;
        let dur = start.elapsed().as_micros() as u64;
        self.events.push(serde_json::json!({
            "name": name,
            "ph": "X",
            "ts": ts,
            "dur": dur,
            "pid": self.pid,
            "tid": 1u32,
        }));
    }

    /// Write the accumulated trace as a JSON array. Silently skips when
    /// the tracer is off or no events were recorded. Best-effort: write
    /// errors are logged via `tracing::warn!` but never bubble up.
    fn flush(self) {
        if !self.enabled || self.events.is_empty() {
            return;
        }
        match serde_json::to_string(&self.events) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&self.path, json) {
                    tracing::warn!(
                        path = %self.path.display(),
                        error = %e,
                        "chrome trace write failed"
                    );
                }
            }
            Err(e) => tracing::warn!(error = %e, "chrome trace serialize failed"),
        }
    }
}

impl CpuEngine {
    /// Load a model from a GGUF, without a tokenizer. Only the token-ID API
    /// is available; calls to [`Engine::generate`] / [`Engine::chat`] error.
    pub fn load(path: &Path, max_ctx: usize) -> Result<Self> {
        Self::load_inner(path, max_ctx, false, KvDtype::F32, "contiguous", 0, None)
    }

    /// Load a model **and** a tokenizer from the same GGUF.
    pub fn load_with_tokenizer(path: &Path, max_ctx: usize) -> Result<Self> {
        Self::load_inner(path, max_ctx, true, KvDtype::F32, "contiguous", 0, None)
    }

    /// Same as [`load_with_tokenizer`] but with an explicit KV-cache dtype.
    /// `kv_dtype = "q8_0"` cuts KV memory ~4x at a small attention-time cost.
    pub fn load_with_options(
        path: &Path,
        max_ctx: usize,
        with_tokenizer: bool,
        kv_dtype: KvDtype,
    ) -> Result<Self> {
        Self::load_inner(
            path,
            max_ctx,
            with_tokenizer,
            kv_dtype,
            "contiguous",
            0,
            None,
        )
    }

    /// Same as [`load_with_options`] but with an explicit
    /// `kv_cache_layout` (matching `[inference].kv_cache_layout` —
    /// `"contiguous"` (default) | `"paged"`). Used by the CLI /
    /// server / GUI wiring to honor the user's config. Fails fast on
    /// invalid layout or paged + non-F32 combos
    /// (see [`KvBackend::from_inference_config`]).
    pub fn load_with_options_and_layout(
        path: &Path,
        max_ctx: usize,
        with_tokenizer: bool,
        kv_dtype: KvDtype,
        kv_layout: &str,
    ) -> Result<Self> {
        Self::load_inner(path, max_ctx, with_tokenizer, kv_dtype, kv_layout, 0, None)
    }

    /// Variant of [`Self::load_with_options_and_layout`] that takes
    /// an explicit `kv_page_size`. Ignored when `kv_layout` is
    /// `"contiguous"`. `kv_page_size = 0` falls back to the
    /// `KvBackend` default. Added for the E2 autotuner — the tuner
    /// sweep loads the engine multiple times with different page
    /// sizes via this entry.
    pub fn load_with_options_layout_and_page_size(
        path: &Path,
        max_ctx: usize,
        with_tokenizer: bool,
        kv_dtype: KvDtype,
        kv_layout: &str,
        kv_page_size: u32,
    ) -> Result<Self> {
        Self::load_inner(
            path,
            max_ctx,
            with_tokenizer,
            kv_dtype,
            kv_layout,
            kv_page_size,
            None,
        )
    }

    /// P11: like [`Self::load_with_options_layout_and_page_size`] but merges a
    /// LoRA adapter GGUF (`lora`) into the model at load (Phase 1 merge). `None`
    /// is the un-adapted base, byte-identical to the plain constructor.
    pub fn load_with_options_layout_page_size_and_lora(
        path: &Path,
        max_ctx: usize,
        with_tokenizer: bool,
        kv_dtype: KvDtype,
        kv_layout: &str,
        kv_page_size: u32,
        lora: Option<&Path>,
    ) -> Result<Self> {
        Self::load_inner(
            path,
            max_ctx,
            with_tokenizer,
            kv_dtype,
            kv_layout,
            kv_page_size,
            lora,
        )
    }

    /// Read the autotuned MoE VRAM margin (bytes) for `model_id` from the tuner
    /// cache, if `tune --moe-vram-margin` / `--all` recorded one on this system
    /// (keyed by the system fingerprint + model file stem). Best-effort: any
    /// cache miss / load error yields `None`, so `load_inner` falls back to the
    /// static `kv_mirror.max(1 GiB)` heuristic.
    fn moe_vram_margin_tuned(model_id: &str) -> Option<u64> {
        let dir = rustllama_tuner::default_cache_dir()?;
        let (key, _device) = rustllama_tuner::cache_context();
        rustllama_tuner::load_cache(&dir, &key)
            .ok()
            .flatten()?
            .moe_vram_margin
            .get(model_id)
            .copied()
    }

    fn load_inner(
        path: &Path,
        max_ctx: usize,
        with_tokenizer: bool,
        kv_dtype: KvDtype,
        kv_layout: &str,
        kv_page_size: u32,
        lora: Option<&Path>,
    ) -> Result<Self> {
        // Model DIRECTORY (an MLX affine checkpoint, or an AWQ/GPTQ folder
        // export): there is no GGUF tensor table to open, so route it through
        // the content-classifying auto-dispatcher (`load_mlx` /
        // `load_safetensors`) exactly as the server's own `use_load_auto` load
        // branch does. This is what lets every `load_with_options*` caller —
        // notably the autotuner's measurement loads (`tune --all` on an MLX
        // dir) — work on a directory instead of failing the `Gguf::open` below.
        // The `.gguf`-file path is unchanged: a file is never `is_dir()`, so
        // this guard is strictly additive. (The dir path uses `load_auto`'s
        // F32/contiguous KV defaults; the GGUF-specific kv_dtype / kv_layout /
        // kv_page_size options don't apply to the affine dir loader yet — the
        // same limitation the server documents on its dir load branch.)
        if path.is_dir() {
            return Self::load_auto(path, max_ctx);
        }
        // MoE expert cache: reset any previous model's pins + range
        // registry before the new model registers its experts (stale
        // pins must never outlive their mmap), and couple zero-copy
        // on — pinning only works on file-backed (`MmapBorrowed`)
        // expert views. The env nudge only helps before the first
        // model load in this process (`zerocopy_weights_enabled` is
        // read once); later loads inherit whatever the first saw.
        rustllama_models::accel::expert_pin_clear();
        rustllama_models::accel::expert_registry_clear();
        // Free the PREVIOUS model's GPU device state before loading this one:
        // the matvec weight caches key by host byte address (reused across
        // reloads → stale entry = wrong-sized device buffer = illegal memory
        // access, or accumulation → OOM), and the flash-attention KV mirrors
        // hold the old shape's device buffers. No-op on a first/only load.
        rustllama_models::accel::reset_device_caches_for_new_model();
        if (rustllama_models::accel::moe_expert_cache_max_bytes() > 0
            || crate::memory_budget::auto_memory_budget_enabled())
            && !rustllama_models::accel::zerocopy_weights()
        {
            // T3: in-process auto-enable via the typed accel knob (was a
            // process-global `set_var("RUSTLLAMA_ZEROCOPY_WEIGHTS", "1")`).
            rustllama_models::accel::set_zerocopy_weights(true);
            tracing::info!(
                "moe expert cache enabled — auto-enabling zero-copy weights \
                 (packed quant tensors borrowed from the mmap; pinning requires \
                 file-backed experts)"
            );
        }
        let gguf = Gguf::open(path)?;
        // F9 / F9-tail — Mamba / SSM routing.
        //
        // A `mamba`/`mamba2`/`jamba`/`falcon-mamba` GGUF that ALSO carries the
        // hybrid DeltaNet layout (`full_attention_interval` + `ssm_conv1d`) is a
        // gated-delta-rule recurrence the hybrid forward already runs (F9) — let
        // it fall straight through to `LlamaModel::load`. A CANONICAL Mamba-1/2
        // checkpoint (the `in_proj`/`x_proj`/`dt_proj`/`A_log`/`D`/`conv1d`
        // selective scan, `s ← exp(dt·A)·s + dt·B·x`, SiLU(z) gate) is NOT that
        // forward — a different parameterization AND tensor set — so F9-tail now
        // WIRES its tested CPU forward (`mamba_arch::MambaModel`) into the engine
        // instead of refusing. The engine carries the `MambaModel` in `mamba` and
        // a minimal transformer SHELL in `model` (correct cfg, empty weights,
        // never executed); `generate_token_ids` dispatches to the scan.
        let mut mamba_model: Option<Arc<rustllama_models::mamba_arch::MambaModel>> = None;
        if let Some(arch) = rustllama_models::mamba_arch::MambaConfig::detect(&gguf) {
            let hybrid_delta_layout = gguf
                .metadata_get(&format!("{arch}.full_attention_interval"))
                .is_some()
                && gguf.tensor("blk.0.ssm_conv1d.weight").is_some();
            if hybrid_delta_layout {
                tracing::info!(
                    arch = %arch,
                    "Mamba-named GGUF carries the hybrid DeltaNet layout \
                     (full_attention_interval + ssm_conv1d) — routing through the \
                     pure-SSM hybrid forward (F9)"
                );
            } else {
                let mm = rustllama_models::mamba_arch::MambaModel::load(&gguf).map_err(|e| {
                    CpuEngineError::Other(format!("canonical Mamba / SSM load failed: {e}"))
                })?;
                tracing::info!(
                    arch = %arch,
                    n_layers = mm.cfg.n_layers,
                    d_model = mm.cfg.d_model,
                    d_inner = mm.cfg.d_inner,
                    vocab = mm.cfg.vocab_size,
                    "canonical Mamba / SSM selective-scan model loaded — wired into \
                     CpuEngine's generation loop (F9-tail)"
                );
                mamba_model = Some(Arc::new(mm));
            }
        }
        let mut model = if let Some(mm) = mamba_model.as_ref() {
            // Minimal transformer shell so the engine's `LlamaModel`-shaped
            // accessors (vocab / n_layers / eos / KV sizing) read the right
            // dims; the forward never runs on it (see `generate_token_ids_mamba`).
            let eos = match gguf.metadata_get("tokenizer.ggml.eos_token_id") {
                Some(rustllama_gguf::MetadataValue::U32(v)) => Some(*v),
                Some(rustllama_gguf::MetadataValue::U64(v)) => Some(*v as u32),
                Some(rustllama_gguf::MetadataValue::I32(v)) if *v >= 0 => Some(*v as u32),
                _ => None,
            };
            rustllama_models::llama_arch::LlamaModel::mamba_shell(&mm.cfg, eos)
        } else {
            LlamaModel::load(&gguf)?
        };
        // P11: merge a LoRA adapter at load (Phase 1). F32/F16 base
        // projections get `W += (alpha/rank)·B·A`; quantized bases are left
        // packed (reported as skipped) for the Phase-2 delta path. A missing
        // or malformed adapter is a hard load error — the user asked for it.
        if let Some(lora_path) = lora {
            let lgguf = Gguf::open(lora_path)?;
            let adapter = rustllama_models::lora::LoraAdapter::from_gguf(&lgguf, "default")
                .map_err(|e| {
                    CpuEngineError::Other(format!(
                        "LoRA adapter load failed ({}): {e}",
                        lora_path.display()
                    ))
                })?;
            let stats = model.weights.merge_lora(&adapter);
            tracing::info!(
                adapter = %lora_path.display(),
                merged = stats.merged,
                skipped_quant = stats.skipped_quant,
                shape_mismatch = stats.shape_mismatch,
                "LoRA adapter merged at load (P11 Phase 1)"
            );
        }
        // Hybrid transformer+DeltaNet (qwen35moe-family) models now
        // load AND run a forward pass — Phase 3.7b plumbing routes
        // these to `forward_one_hybrid` instead of the dense forward.
        // Caveats per the Phase 3.7b doc: F32-only, decode-only, no
        // multi-section RoPE for the full-attn layers, and no KV
        // history. Output execution is real but correctness vs
        // reference is pending Phase 8 parity gate.
        if model.weights.is_hybrid() {
            tracing::warn!(
                arch = %model.cfg.arch,
                n_layers = model.cfg.n_layers,
                "hybrid attention+DeltaNet model loaded — the forward pass \
                 executes (F32, decode-first) but its output has not yet been \
                 validated against a reference implementation."
            );
        }
        let ctx = max_ctx.min(model.cfg.ctx_train.max(max_ctx));
        // Hybrid KV dtype. The hybrid full-attention forward now
        // implements EVERY KV dtype: F32 and Q4_0 have dedicated fast
        // arms (Q4_0 with whitening + kv-bias calibration), and Q8_0 /
        // TurboQuant / NVFP4 route through the same per-dtype flash
        // decode/prefill kernels the dense path uses.
        //
        // We still DEFAULT hybrids to F32 for coherence: the low-bit
        // Q8_0/TQ/NVFP4/MXFP arms lack the Q4_0 arm's whitening/
        // calibration and are not individually validated on the
        // SSM+attention hybrid. So an UNVALIDATED low-bit request on a
        // hybrid is coerced to F32 (Q4_0 is honored as-is — it has the
        // tuned arm).
        //
        // `RUSTLLAMA_HYBRID_KV_ANY` lifts that coercion — but it is now
        // driven by the AUTOTUNE, not the user. The `tune --kv-dtype`
        // sweep measures the FULL candidate grid on hybrids and its
        // coherence gate rejects dtypes that diverge from the f32
        // reference (e.g. 1-bit tq1) while adopting coherent ones. The
        // sweep sets this flag during its own measurement, and the serve
        // load sets it when it applies a coherence-VALIDATED cache winner
        // — so a validated quant hybrid KV is honored verbatim here while
        // an unqualified one still falls to F32. (A user may still set it
        // manually to force an un-swept dtype.)
        // T3: typed `LoadOptions`-backed accel knob (env-initialized for
        // back-compat; set by the autotune kv-dtype sweep + the serve/HTTP
        // load when it applies a coherence-validated cache winner).
        let hybrid_kv_any = rustllama_models::accel::hybrid_kv_any();
        let kv_dtype = if model.weights.is_hybrid()
            && !hybrid_kv_any
            && kv_dtype != KvDtype::F32
            && kv_dtype != KvDtype::Q4_0
        {
            tracing::warn!(
                requested = ?kv_dtype,
                "hybrid model: defaulting KV cache to f32 for coherence \
                 (set RUSTLLAMA_HYBRID_KV_ANY=1 to honor a quantized \
                 hybrid KV cache verbatim)"
            );
            KvDtype::F32
        } else {
            kv_dtype
        };
        // F1: the CONTIGUOUS quant-KV flash kernels (CPU `*_capped_windowed` +
        // the SYCL/CUDA/MLX quant flash) now carry the per-layer sliding-window
        // span + Gemma2 attention-logit soft-cap, so an SWA / softcap model may
        // KEEP these quant KV dtypes instead of being coerced to F32. f16/bf16
        // stay fail-closed → F32 (their windowed path is not wired yet), as does
        // any future dtype — matching "KEEP fail-closed for any dtype whose quant
        // flash still lacks window/softcap". NOTE the PAGED quant-KV forwards
        // (`PagedKvStoreOps`, `llama_arch.rs`) still IGNORE window/softcap, so the
        // `kv_layout` coercion below pins an SWA/softcap + quant-KV model to the
        // CONTIGUOUS layout (where the windowed/capped kernels live).
        let quant_kv_windowed = matches!(
            kv_dtype,
            KvDtype::Q8_0
                | KvDtype::Tq(_)
                | KvDtype::Nvfp4
                | KvDtype::Q4_0
                | KvDtype::Mxfp4
                | KvDtype::Mxfp6
                | KvDtype::Mxfp8
        );
        // Sliding-window attention: F32 KV always windows (CPU + F32-contiguous
        // GPU flash/decode-block + the F32 PAGED forwards, D1b). A quant KV dtype
        // whose flash now carries the window uniform (F1, `quant_kv_windowed`) is
        // ALSO honored verbatim on the contiguous path; every other non-F32 dtype
        // is still coerced to F32 for masking correctness. Non-SWA untouched.
        let kv_dtype =
            if model.cfg.sliding_window.is_some() && kv_dtype != KvDtype::F32 && !quant_kv_windowed
            {
                tracing::warn!(
                    requested = ?kv_dtype,
                    "sliding-window model: defaulting KV cache to f32 (this dtype's \
                     windowed attention kernels are not available; F1 covers q4_0 / \
                     q8_0 / nvfp4 / tq / mxfp4-6-8)"
                );
                KvDtype::F32
            } else {
                kv_dtype
            };
        // Gemma2 attention-logit soft-capping (`s = c·tanh(s/c)` on every
        // attention score, before softmax): same rule as the sliding-window
        // coercion just above — F32 and the F1 `quant_kv_windowed` quant dtypes
        // carry the per-layer softcap on the contiguous path and are honored;
        // other non-F32 dtypes are coerced to F32 (the output is WRONG without
        // the cap). Non-capping models are untouched.
        let kv_dtype = if model.cfg.attn_logit_softcapping.is_some()
            && kv_dtype != KvDtype::F32
            && !quant_kv_windowed
        {
            tracing::warn!(
                requested = ?kv_dtype,
                "attention-softcap model (Gemma2): defaulting KV cache to f32 (this \
                 dtype's soft-capping attention kernels are not available; F1 covers \
                 q4_0 / q8_0 / nvfp4 / tq / mxfp4-6-8)"
            );
            KvDtype::F32
        } else {
            kv_dtype
        };
        // D1b LIFTED the SWA / softcap `kv_layout → contiguous` coercion for F32
        // KV: the F32 paged forwards thread the per-layer sliding-window span +
        // Gemma2 attention-logit soft-cap (`forward_one_paged_f32`,
        // `forward_decode_paged_batched_f32`, `forward_ragged_paged_f32`, and the
        // inline `forward_prefill_paged_f32`), so an SWA / softcap model may use a
        // PAGED **F32** KV layout. F1 then lifted the SWA/softcap kv_dtype coercion
        // for the quant dtypes (above) — but the PAGED quant-KV forwards
        // (`PagedKvStoreOps`) still IGNORE window/softcap (they take `_window` /
        // `_softcap`), so an SWA/softcap model with a quant KV dtype is pinned to
        // the CONTIGUOUS layout here, where the windowed/capped quant kernels run.
        // (Lifting this needs the paged quant forwards to carry window/softcap too
        // — a T2/F-series follow-up.) The HYBRID layout coercion is KEPT: no paged
        // forward walks hybrid (DeltaNet/SSM) layers yet (lifted by F8).
        // M1: MLA (DeepSeek-V2/V3) stores a compressed latent, not per-head
        // K/V — the paged backends have no MLA store, so MLA is fail-closed to
        // the contiguous layout + an f32 latent (the MLA cache ignores the
        // requested kv_dtype, but coerce it for honest logging).
        let kv_dtype = if model.weights.is_mla() && kv_dtype != KvDtype::F32 {
            tracing::warn!(
                requested = ?kv_dtype,
                "MLA model (DeepSeek-V2/V3): the compressed-latent KV cache is \
                 f32-only — ignoring the requested KV dtype"
            );
            KvDtype::F32
        } else {
            kv_dtype
        };
        let swa_or_softcap =
            model.cfg.sliding_window.is_some() || model.cfg.attn_logit_softcapping.is_some();
        let kv_layout: &str =
            if (model.weights.is_hybrid() || model.weights.is_mla()) && kv_layout != "contiguous" {
                tracing::warn!(
                    requested = %kv_layout,
                    is_mla = model.weights.is_mla(),
                    "hybrid / MLA model: paged KV layouts are not supported by this \
                     forward path — coercing to contiguous for this load"
                );
                "contiguous"
            } else if swa_or_softcap && kv_dtype != KvDtype::F32 && kv_layout != "contiguous" {
                tracing::warn!(
                    requested = %kv_layout,
                    kv_dtype = ?kv_dtype,
                    "sliding-window / softcap model with quant KV: coercing to the \
                     contiguous layout (the paged quant-KV forwards do not yet carry \
                     the window/softcap uniforms; the contiguous quant kernels do)"
                );
                "contiguous"
            } else {
                kv_layout
            };
        // Sparse KV for hybrid models: only full-attention layers get
        // real slabs (see `hybrid_kv_keep_mask`).
        let kv_keep = hybrid_kv_keep_mask(&model);
        let kv_backend = KvBackend::from_inference_config_with_page_size_and_keep(
            kv_layout,
            &model.cfg,
            ctx as u32,
            kv_dtype,
            kv_page_size,
            kv_keep.as_deref(),
        )
        .map_err(|e| CpuEngineError::Other(format!("kv_backend init: {e}")))?;

        let tokenizer = if with_tokenizer {
            Some(Arc::new(Tokenizer::from_gguf(&gguf)?))
        } else {
            None
        };

        let model_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "rustllama-cpu".into());

        // Post-load summary: one info-level line carrying the
        // dimensions clients care about when grepping logs (arch,
        // layer count, hidden dim, vocab, context, KV dtype) plus
        // MoE info when applicable. Single line so log aggregators
        // can pin "this server is serving this model" with one match.
        let moe_desc = model.cfg.moe.as_ref().map(|m| {
            if m.n_experts_shared > 0 {
                format!(
                    "moe={}routed+{}shared(top-{})",
                    m.n_experts, m.n_experts_shared, m.n_experts_used
                )
            } else {
                format!("moe={}routed(top-{})", m.n_experts, m.n_experts_used)
            }
        });
        tracing::info!(
            model_id = %model_id,
            arch = %model.cfg.arch,
            n_layers = model.cfg.n_layers,
            d_model = model.cfg.d_model,
            n_heads = model.cfg.n_heads,
            n_kv_heads = model.cfg.n_kv_heads,
            vocab_size = model.cfg.vocab_size,
            ctx_train = model.cfg.ctx_train,
            ctx = ctx,
            kv_dtype = ?kv_dtype,
            moe = moe_desc.as_deref().unwrap_or("none"),
            "model loaded — engine ready"
        );

        // Drop the GGUF mmap now that `model` and `tokenizer` have
        // taken their owned copies of every byte they need. Holding
        // it past this point would keep ~model-size of file-mapped
        // pages in the process working set, duplicated against the
        // owned `Tensor` storage and (when SYCL is up) the USM
        // weight cache. Releasing here yields ~model-size visible
        // RAM reduction.
        //
        // NOTE (Stage 8 update): with zero-copy weights, `LlamaModel`
        // holds `Arc` clones of the mmap backing — dropping the
        // `Gguf` struct does NOT unmap the file, and the borrowed
        // weight pages remain live and needed. The comment below
        // predates zero-copy and stays true only for the
        // everything-copied configuration.
        //
        // Safety note for future maintainers: no public API on
        // `CpuEngine` consults the GGUF after load. `LlamaModel` and
        // `Tokenizer` both take `&Gguf` and produce fully owned
        // outputs (except zero-copy borrows, which keep the backing
        // alive via `Arc`).
        drop(gguf);
        // Force Windows to actually release the file-cache pages we
        // just unmapped — but ONLY when nothing was borrowed
        // zero-copy. With `RUSTLLAMA_ZEROCOPY_WEIGHTS` engaged, the
        // "unmapped" pages are ~the whole model's live weights;
        // trimming them evicted ~11 GB that pagelock + expert prepin
        // then re-read from disk seconds later (~6 GB of redundant
        // I/O per load, log-verified). `zerocopy_borrow_stats()` is
        // populated during `LlamaModel::load`, so it is authoritative
        // here regardless of how the env var was set.
        let (zc_borrowed_tensors, _) = rustllama_models::llama_arch::zerocopy_borrow_stats();
        #[cfg(target_os = "windows")]
        if zc_borrowed_tensors == 0 {
            // SAFETY: GetCurrentProcess returns a pseudo-handle
            // that's always valid; EmptyWorkingSet on the current
            // process is a documented no-side-effect operation
            // beyond trimming the WS. Errors are non-fatal — log
            // and continue.
            extern "system" {
                fn GetCurrentProcess() -> *mut std::ffi::c_void;
                fn EmptyWorkingSet(h_process: *mut std::ffi::c_void) -> i32;
            }
            let ok = unsafe { EmptyWorkingSet(GetCurrentProcess()) };
            if ok != 0 {
                tracing::info!(
                    model_id = %model_id,
                    "released GGUF mmap + trimmed process working set \
                     (EmptyWorkingSet) — Task Manager RAM should drop \
                     by ~model size"
                );
            } else {
                tracing::warn!(
                    model_id = %model_id,
                    "released GGUF mmap but EmptyWorkingSet returned 0; \
                     OS may keep the unmapped pages in standby cache until \
                     memory pressure"
                );
            }
        } else {
            tracing::info!(
                model_id = %model_id,
                borrowed_tensors = zc_borrowed_tensors,
                "zero-copy weights live — skipping the post-load working-set trim \
                 (trimming would evict the borrowed weight pages just to re-read them)"
            );
        }
        #[cfg(not(target_os = "windows"))]
        {
            let _ = zc_borrowed_tensors;
            tracing::info!(
                model_id = %model_id,
                "released GGUF mmap after load (non-Windows: OS reclaims unmapped pages lazily)"
            );
        }

        let delta_net_cache = build_delta_net_cache(&model)?;
        // Memory-budget planner (Phase 3): with `[inference].memory_budget
        // = "auto"`, derive the expert-cache budget from measured
        // available RAM minus an honest projection of what the engine
        // still allocates (KV, DeltaNet state, forward scratch, OS
        // reserve). Must run before `lock_model_into_ram` so both the
        // working-set floor and the pagelock clamp see the final budget.
        if crate::memory_budget::auto_memory_budget_enabled() {
            if let Some((total_phys, avail_phys)) = crate::pagelock::memory_status() {
                // The user's explicit moe_expert_cache_mb (if any) caps
                // the computed budget. Capture it once — after the first
                // plan the atomic holds the *computed* value, which must
                // not ratchet later loads.
                static EXPLICIT_CAP: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                let explicit_cap_bytes =
                    *EXPLICIT_CAP.get_or_init(rustllama_models::accel::moe_expert_cache_max_bytes);
                // Worst-case prefix-pool growth: N snapshots × one
                // full-context anchor (KV slabs + recurrent state).
                let per_anchor = kv_backend.approx_host_bytes()
                    + delta_net_cache
                        .as_ref()
                        .map(|d| d.approx_bytes() as u64)
                        .unwrap_or(0);
                let facts = crate::memory_budget::MemoryFacts {
                    total_phys,
                    avail_phys,
                    kv_bytes: kv_backend.approx_host_bytes(),
                    dn_bytes: delta_net_cache
                        .as_ref()
                        .map(|d| d.approx_bytes() as u64)
                        .unwrap_or(0),
                    scratch_bytes: crate::memory_budget::scratch_projection(
                        model.cfg.vocab_size,
                        model.cfg.d_model,
                        model.cfg.d_ff,
                    ),
                    expert_pool_bytes: rustllama_models::accel::expert_registry_total_bytes(),
                    pool_reserve_bytes: DEFAULT_PREFIX_CACHE_MAX_SNAPSHOTS as u64 * per_anchor,
                    // M7: RAM the pagelock tier is about to pin (`lock_ram_mb`).
                    // Reserve it CROSS-PLATFORM so the pinned expert cache isn't
                    // sized onto RAM the loader is about to lock. `0` when
                    // locking is off ⇒ byte-identical plan.
                    locked_bytes: crate::pagelock::projected_lock_bytes(&model),
                    explicit_cap_bytes,
                };
                let plan = crate::memory_budget::plan(&facts);
                rustllama_models::accel::set_moe_expert_cache_max_bytes(plan.expert_cache_bytes);
                if plan.exhausted {
                    tracing::warn!(
                        avail_mb = avail_phys / (1024 * 1024),
                        reserve_mb = plan.reserve_bytes / (1024 * 1024),
                        kv_mb = facts.kv_bytes / (1024 * 1024),
                        "memory_budget=auto: no headroom after reserve + projections — \
                         expert cache disabled; the box is at its limit for this model/ctx"
                    );
                } else {
                    tracing::info!(
                        avail_mb = avail_phys / (1024 * 1024),
                        kv_mb = facts.kv_bytes / (1024 * 1024),
                        scratch_mb = facts.scratch_bytes / (1024 * 1024),
                        reserve_mb = plan.reserve_bytes / (1024 * 1024),
                        locked_mb = facts.locked_bytes / (1024 * 1024),
                        expert_pool_mb = facts.expert_pool_bytes / (1024 * 1024),
                        expert_cache_mb = plan.expert_cache_bytes / (1024 * 1024),
                        "memory_budget=auto: expert-cache budget planned from measured RAM"
                    );
                }
            } else {
                tracing::warn!(
                    "memory_budget=auto: physical-memory query unavailable; keeping manual budgets"
                );
            }
        }
        // Automatic MoE disk-spill cold-tier store (GGUF-native, no env var).
        // Arm it exactly when the routed-expert pool exceeds the now-finalized
        // LRU expert-cache budget — i.e. experts WILL evict, so a compact,
        // sequential, GGUF-native spill copy of the cold tail beats re-faulting
        // the scattered GGUF mmap. Pool <= budget (all experts fit resident) or
        // budget == 0 (tier/exhausted path) ⇒ no eviction ⇒ spill stays OFF, so
        // the default box and the validated device-tier path are untouched.
        // This runs on every load — including the autotuner's — so the enable
        // is part of the per-(system, model) autotune determination.
        {
            let cache_budget = rustllama_models::accel::moe_expert_cache_max_bytes();
            let pool = rustllama_models::accel::expert_registry_total_bytes();
            let spill = cache_budget > 0 && pool > cache_budget;
            rustllama_models::accel::set_moe_spill_plan(spill);
            if spill {
                // Point the spill file at the runtime cache dir (idempotent;
                // no-op when disarmed). Independent of the usage learner.
                crate::expert_cache::init_moe_spill_store();
                tracing::info!(
                    pool_mb = pool / (1024 * 1024),
                    cache_budget_mb = cache_budget / (1024 * 1024),
                    "moe disk-spill: GGUF-native cold-tier store armed \
                     (expert pool exceeds the LRU cache budget; experts will evict)"
                );
            }
        }
        // MoE tiered-expert engine (Phase 2/4 trigger): AUTO-ENABLED on CUDA.
        // On a CUDA box with a MoE model this promotes the hottest experts into
        // CUDA device (managed) memory so they run GPU-resident regardless of the
        // `n_gpu_layers` layer cutoff (the lifted dispatch in `accel.rs`). It now
        // runs with the `auto` VRAM-expert budget (free VRAM at load −
        // non-expert weights − the KV-mirror-aware margin, capped at expert
        // bytes) — NO env var, no knob. The tier self-limits: a failed VRAM
        // query or a tiny/unified GPU yields budget 0 → no promotion.
        // Must happen HERE — `model` is still uniquely owned (before the `Arc`),
        // and before `lock_model_into_ram` (which skips `Storage::Device` via
        // `cpu_backing_ptr_len` → `None`, so promoted experts are never
        // VirtualLock'd). Safe-by-construction: managed memory oversubscribes to
        // host-paged UVM instead of hard-OOMing, and a failed promotion keeps the
        // CPU bytes, so an over-large budget degrades to "slower", never a crash
        // or a wrong answer. Validated on-device (RTX 2000 Ada, Qwen3-30B-A3B);
        // inert on unified-memory GPUs (promotion to managed is a no-op there).
        // SYCL/MLX device-tier (P20): the `auto` projection + enablement now
        // covers a DEDICATED SYCL GPU too (Intel Arc/PVC), using
        // `sycl_dedicated_vram_bytes()` as the total-VRAM proxy (SYCL has no free
        // query; promotion happens before the weight upload so ~all VRAM is free).
        // MLX (Apple unified memory) and integrated iGPUs (Iris Xe) have NO
        // dedicated VRAM to promote into, so the projection yields `None` → budget
        // 0 → no promotion (inert by construction, not a forced no-op). CUDA is
        // unchanged (managed memory). The budget formula itself is the shared,
        // unit-tested `project_moe_tier_budget_bytes` below.
        // AUTO — NO ENV VAR. Tier target VRAM: CUDA free VRAM, else a DEDICATED
        // SYCL GPU (Intel Arc/PVC — `None` on integrated Iris Xe where USM
        // promotion is a host-backed no-op, and on a box with no usable GPU).
        // The promotion routes to the active backend (CUDA managed / SYCL USM)
        // via `upload_bytes_to_device`. The tier self-limits with no knob: a
        // failed VRAM query / tiny / unified GPU yields budget 0 and the `> 0`
        // guard below skips promotion. `promote_experts_to_device` is NO-DOUBLE
        // (promote a block, free its host parent → transient host ≈ one block),
        // so an over-estimate degrades to "slower", never OOM/wrong-answer.
        let tier_avail: Option<u64> = if rustllama_models::accel::cuda_active() {
            Some(rustllama_runtime::gpu_detect::nvidia_free_vram_bytes(0).unwrap_or(0))
        } else {
            rustllama_models::accel::sycl_dedicated_vram_bytes()
        };
        if let Some(avail) = tier_avail {
            if model.weights.is_moe() {
                let vram_expert_bytes: u64 = {
                    // Projected budget: free VRAM at load (weights upload lazily, so
                    // ~nothing is resident yet) − the non-expert weights (assumed all
                    // GPU-resident, conservative) − a margin for the device context,
                    // KV mirror, and fragmentation; capped at the total expert bytes.
                    let (expert_b, non_expert_b) = model.weights.vram_byte_breakdown();
                    // Margin = the decode KV mirror + a floor for the CUDA context,
                    // Q/out scratch, and fragmentation. The canonical KV cache is
                    // host RAM (`KvBuf::Host`), BUT the CUDA/Metal flash-DECODE path
                    // keeps a persistent per-layer device KV mirror in VRAM (packed,
                    // so its footprint scales with kv_dtype: f32 = 4 B/elem, q8_0 ≈
                    // 1.06, nvfp4 ≈ 0.56). Reserving the actual mirror means a
                    // quantized KV auto-frees that VRAM for more promoted experts.
                    // On-pod (Qwen3-30B-A3B, RTX 2000 Ada 16 GB): f32 → ~1.6 GB
                    // mirror → budget ~13.2 GB (~38 layers, 9.96 tok/s); q8_0 →
                    // ~0.43 GB mirror (floor dominates) → budget ~13.8 GB (~40) AND
                    // a 4× lighter attention read (q8_0 alone measured 9.96→11.37
                    // tok/s). The old flat 20% over-reserved ~1.8 GB; budget sweeps
                    // up to ~13.5 GB promote cleanly (fb steady, no thrash) since
                    // promoted experts are managed memory that pages to host under
                    // pressure — an under-reserve degrades to "slower", never OOM.
                    let kv_bytes_per_elem = (kv_dtype.approx_bits_per_element() / 8.0) as f64;
                    let kv_mirror = (model.cfg.n_layers as f64
                    * 2.0 // K + V
                    * model.cfg.n_kv_heads as f64
                    * ctx as f64
                    * model.cfg.head_dim as f64
                    * kv_bytes_per_elem) as u64;
                    // Margin precedence: the tuner A/B override (set per candidate by
                    // the in-process `tune --moe-vram-margin` sweep) → the persisted
                    // per-model winner → the static `kv_mirror.max(1 GiB)` heuristic.
                    // An autotuned margin lets each host promote the decode-optimal
                    // expert count instead of the conservative default.
                    let heuristic_margin = kv_mirror.max(1024 * 1024 * 1024);
                    let margin = rustllama_models::accel::moe_vram_margin_override()
                        .or_else(|| Self::moe_vram_margin_tuned(&model_id))
                        .unwrap_or(heuristic_margin);
                    // Shared, unit-tested budget projection (P20). `Some(avail)` + a
                    // MoE model here → the same `avail − non_expert − margin` capped
                    // at expert bytes as before (byte-identical); the `None` /
                    // non-MoE arms (budget 0, inert) are exercised by the unit tests.
                    project_moe_tier_budget_bytes(Some(avail), true, expert_b, non_expert_b, margin)
                };
                if vram_expert_bytes > 0 {
                    // Usage ranking (hottest-first) from the persisted per-model
                    // sidecar, so a tight budget buys the most-routed experts. Cold
                    // start (no sidecar) ⇒ empty ⇒ promotion falls back to index
                    // order. Opening a throwaway learner just reads the sidecar; the
                    // session's own learner is initialized separately below.
                    let ranked = crate::expert_cache::UsageLearner::open(path)
                        .map(|l| l.ranked_keys())
                        .unwrap_or_default();
                    let (experts, bytes) = model
                        .weights
                        .promote_experts_to_device(vram_expert_bytes, &ranked);
                    if experts > 0 {
                        // Arm MoE-aware placement + the non-promoted-expert CPU
                        // routing (per_layer_weight_bytes excludes experts; the
                        // dispatch forces un-promoted expert views to CPU).
                        rustllama_models::accel::set_moe_tier_active(true);
                        tracing::info!(
                            experts,
                            promoted_mb = bytes / (1024 * 1024),
                            budget_mb = vram_expert_bytes / (1024 * 1024),
                            ranked = ranked.len(),
                            "moe tiered-expert: promoted hottest experts to GPU device memory"
                        );
                    }
                }
            }
        }
        // Wrap the model in its Arc, then (Windows) VirtualLock the hot
        // weight tiers into RAM per RUSTLLAMA_LOCK_RAM_MB. Done after the
        // GGUF mmap drop + EmptyWorkingSet above so we re-resident only
        // the live owned weights, not the just-trimmed file pages.
        let model = Arc::new(model);
        let lock_registry = crate::pagelock::lock_model_into_ram(Arc::clone(&model)).map(Arc::new);
        // Authoritative residency check: QueryWorkingSetEx reports the
        // Locked bit per page — confirms VirtualLock actually pinned the
        // ranges (vs. SetProcessWorkingSetSizeEx silently failing).
        #[cfg(windows)]
        if let Some(reg) = lock_registry.as_ref() {
            let (locked, probed) = reg.verify_locked();
            tracing::info!(
                model_id = %model_id,
                locked_regions = locked,
                probed_regions = probed,
                total_regions = reg.region_count() as u64,
                locked_mb = reg.locked_bytes() / (1024 * 1024),
                "pagelock: QueryWorkingSetEx Locked-bit verification"
            );
        }
        // Zero-copy confirmation: how many weight tensors were borrowed
        // straight from the GGUF mmap (Stage 8) vs. copied to owned heap.
        let (zc_tensors, zc_bytes) = rustllama_models::llama_arch::zerocopy_borrow_stats();
        if zc_tensors > 0 {
            tracing::info!(
                model_id = %model_id,
                borrowed_tensors = zc_tensors,
                borrowed_mb = zc_bytes / (1024 * 1024),
                "zero-copy weights: borrowed raw tensors from GGUF mmap (no owned-heap copy)"
            );
        } else {
            tracing::info!(
                model_id = %model_id,
                "zero-copy weights: not engaged (disabled or no raw-passthrough tensors)"
            );
        }
        // MoE expert cache: pre-pin the learned-hottest experts
        // before the first prefill (Colibrì-style learning cache) and
        // start the usage flusher. Runs after `lock_model_into_ram`
        // so the working-set floor already covers the cache budget.
        let expert_usage = Self::init_expert_usage(path, &model_id);
        // Prefix-pool byte cap: bound the pool at its worst case (N
        // snapshots × one full-context anchor) so it can never
        // outgrow what the memory planner accounts for.
        let pool_byte_cap = DEFAULT_PREFIX_CACHE_MAX_SNAPSHOTS as u64
            * (kv_backend.approx_host_bytes()
                + delta_net_cache
                    .as_ref()
                    .map(|d| d.approx_bytes() as u64)
                    .unwrap_or(0));
        let kv_dtype_cached = kv_backend.kv_dtype();
        let engine = Self {
            model,
            tokenizer,
            kv_dtype_cached,
            metrics_cache: Arc::new(MetricsCacheAtomics::default()),
            state: Arc::new(Mutex::new(EngineState {
                kv_backend,
                last_ids: Vec::new(),
                pool: PrefixCachePool::new(DEFAULT_PREFIX_CACHE_MAX_SNAPSHOTS),
                delta_net_cache,
            })),
            sycl_worker: Arc::new(SyclWorker::spawn()),
            max_ctx: ctx,
            model_id,
            source_path: path.to_path_buf(),
            prefix_cache: true,
            prefill_chunk_size: DEFAULT_PREFILL_CHUNK,
            prefix_cache_max_snapshots: DEFAULT_PREFIX_CACHE_MAX_SNAPSHOTS,
            last_stats: Arc::new(Mutex::new(RequestStats::default())),
            ema_tok_s_bits: Arc::new(std::sync::atomic::AtomicU64::new(f64::NAN.to_bits())),
            cumulative_stats: Arc::new(crate::CumulativeStats::default()),
            max_tool_iterations: DEFAULT_MAX_TOOL_ITERATIONS,
            flash_attention: true,
            kv_layout: kv_layout.to_string(),
            kv_page_size,
            n_gpu_layers: u32::MAX,
            cpu_force_patterns: Vec::new(),
            vision: None,
            image_token_id: None,
            placeholder_mode: None,
            ngram_spec: None,
            lookahead_spec: None,
            draft_spec: None,
            mtp_spec: false,
            tree_spec: None,
            eagle_head: None,
            image_wrapper: None,
            vision_feature_memo: Arc::new(Mutex::new(None)),
            lock_registry,
            mamba: mamba_model,
            expert_usage,
            kv_persist_owner: true,
            cacheblend: None,
            checkpoint_clock: Mutex::new(crate::kv_persist::CheckpointClock::new(
                crate::kv_persist::kv_persist_interval(),
                std::time::Instant::now(),
            )),
            lora: Arc::new(std::sync::RwLock::new(
                rustllama_models::lora::LoraRegistry::new(),
            )),
        };
        if let Ok(mut st) = engine.state.lock() {
            st.pool.set_max_bytes(pool_byte_cap);
        }
        // Warm restart (Phase 5): pull any persisted prefix snapshots
        // into the pool before the first request.
        engine.load_kv_persist();
        Ok(engine)
    }

    /// Elastic expert-cache budget hot-apply (roadmap Phase 3):
    /// revise the budget on a live engine without a model reload.
    /// The caller must ensure no generation is in flight (hold every
    /// serving-gate permit). Sequence: persist the learning stats
    /// (the clear below wipes the counters), install the new budget,
    /// raise the working-set floor for growth (raise-only — locked
    /// pages forbid lowering), drop every pin, and re-pin the
    /// learned-hottest set under the new cap using baseline + live
    /// session counts. Returns `(experts_pinned, bytes_pinned)`.
    ///
    /// Only fully effective when the model loaded with file-backed
    /// (zero-copy) experts — otherwise there is nothing pinnable and
    /// a reload is required for the new budget to matter.
    pub fn apply_expert_cache_budget(&self, budget_mb: u64) -> (usize, u64) {
        use rustllama_models::accel;
        let bytes = budget_mb.saturating_mul(1024 * 1024);
        if let Some(usage) = &self.expert_usage {
            if let Ok(mut learner) = usage.lock() {
                learner.flush(true);
            }
        }
        accel::set_moe_expert_cache_max_bytes(bytes);
        if !crate::pagelock::refresh_working_set_floor() {
            tracing::warn!(
                "expert cache: working-set floor raise failed; pins under the new budget may no-op"
            );
        }
        accel::expert_pin_clear();
        if bytes == 0 {
            tracing::info!("expert cache: budget hot-applied as 0 — cache disabled, pins released");
            return (0, 0);
        }
        if accel::expert_registry_len() == 0 {
            tracing::warn!(
                "expert cache: budget set but no file-backed experts are registered \
                 (dense model, or zero-copy weights were off at load) — reload the \
                 model for the budget to take effect"
            );
            return (0, 0);
        }
        let re_pinned = match &self.expert_usage {
            Some(usage) => match usage.lock() {
                Ok(learner) => learner.prepin_merged(bytes),
                Err(_) => (0, 0),
            },
            // No learner (learning disabled at load): the cache still
            // works touch-driven; it just starts cold under the new cap.
            None => (0, 0),
        };
        tracing::info!(
            budget_mb,
            prepinned_experts = re_pinned.0,
            prepinned_mb = re_pinned.1 / (1024 * 1024),
            "expert cache: budget hot-applied, learned-hot set re-pinned"
        );
        re_pinned
    }

    /// Warm-restart load (roadmap Phase 5): pull persisted prefix
    /// snapshots from `<model>.rlkv` into the pool so the first
    /// request whose prompt extends one skips straight to decode.
    /// No-op unless `RUSTLLAMA_KV_PERSIST_MB > 0` and the backend is
    /// contiguous F32 KV (the v1 persistence format).
    fn load_kv_persist(&self) {
        if crate::kv_persist::kv_persist_cap_bytes() == 0 {
            return;
        }
        let mut state = self.state.lock().expect("state lock");
        let s = &mut *state;
        let KvBackend::Contiguous(kv) = &s.kv_backend else {
            return;
        };
        if !kv.is_uniform_f32() {
            tracing::info!("kv_persist: only F32 KV is persisted in v1; skipping load");
            return;
        }
        let (n_layers, n_kv_heads, head_dim, max_ctx) =
            (kv.layers.len(), kv.n_kv_heads, kv.head_dim, kv.max_ctx);
        let Some(snaps) =
            crate::kv_persist::load_pool(&self.source_path, n_layers, n_kv_heads, head_dim)
        else {
            return;
        };
        let mut loaded = 0usize;
        // Saved most-recently-used first; insert in reverse so the
        // freshest entry gets the newest LRU stamp.
        for snap in snaps.into_iter().rev() {
            if snap.kv.prefix_len > max_ctx {
                continue; // saved under a bigger ctx_size than now
            }
            s.pool.insert_snapshot(snap);
            loaded += 1;
        }
        if loaded > 0 {
            tracing::info!(
                model_id = %self.model_id,
                snapshots = loaded,
                "kv persist: warm-restart snapshots loaded — matching prompts skip re-prefill"
            );
        }
    }

    /// Warm-restart save — see [`Self::load_kv_persist`]. Called from
    /// [`Drop`] on the owning engine. Locks `state` itself (Drop holds no
    /// lock), then delegates to the lock-free [`Self::write_kv_persist_sidecar`].
    fn save_kv_persist(&self) {
        if !self.kv_persist_owner {
            return;
        }
        let cap = crate::kv_persist::kv_persist_cap_bytes();
        if cap == 0 {
            return;
        }
        let Ok(state) = self.state.lock() else {
            return;
        };
        if let Some(n) = self.write_kv_persist_sidecar(&state, cap) {
            if n > 0 {
                tracing::info!(
                    snapshots = n,
                    "kv persist: session snapshots saved for warm restart"
                );
            }
        }
    }

    /// F12 — periodic mid-session checkpoint. Re-writes the warm-restart sidecar
    /// from the live pool once `kv_persist_interval_secs` has elapsed, so a hard
    /// kill loses at most the last interval instead of the whole conversation.
    ///
    /// Takes the ALREADY-LOCKED `state` (called from the generation path right
    /// after `commit_snapshot`, so the pool already carries this request's full
    /// sequence) and never re-locks `self.state` — re-locking the held guard
    /// would deadlock. No-op unless this engine owns the sidecar AND the clock
    /// fires; the clock is inert when `kv_persist_interval_secs == 0`, so the
    /// default path is byte-identical to pre-F12.
    fn maybe_checkpoint_kv_persist(&self, state: &EngineState) {
        if !self.kv_persist_owner {
            return;
        }
        let cap = crate::kv_persist::kv_persist_cap_bytes();
        if cap == 0 {
            return;
        }
        // Cheap clock gate before touching the backend.
        {
            let Ok(mut clk) = self.checkpoint_clock.lock() else {
                return;
            };
            if !clk.should_checkpoint(std::time::Instant::now()) {
                return;
            }
        }
        if let Some(n) = self.write_kv_persist_sidecar(state, cap) {
            tracing::debug!(
                snapshots = n,
                "kv persist: periodic checkpoint written (F12)"
            );
        }
    }

    /// Serialize the current prefix pool to `<model>.rlkv`. Lock-free: the
    /// caller provides the locked `state`. Returns `Some(n)` snapshots written
    /// on a successful save path (the contiguous-F32 case), `None` when the
    /// backend can't be persisted (non-contiguous / non-uniform-F32). Shared by
    /// the Drop save and the F12 periodic checkpoint.
    fn write_kv_persist_sidecar(&self, state: &EngineState, cap: u64) -> Option<usize> {
        let KvBackend::Contiguous(kv) = &state.kv_backend else {
            return None;
        };
        if !kv.is_uniform_f32() {
            return None;
        }
        match crate::kv_persist::save_pool(
            &self.source_path,
            &state.pool,
            kv.layers.len(),
            kv.n_kv_heads,
            kv.head_dim,
            cap,
        ) {
            Ok(n) => Some(n),
            Err(e) => {
                tracing::debug!(error = %e, "kv persist save failed (non-fatal)");
                Some(0)
            }
        }
    }

    /// Set up the MoE expert-cache learning driver: warn when the
    /// cache is on but nothing registered as pinnable, pre-pin the
    /// sidecar's hottest experts, and spawn the periodic flusher.
    /// Returns `None` whenever the learning cache shouldn't run
    /// (cache off, dense model, learning disabled, fingerprint
    /// failure) — the pin cache itself still works touch-driven.
    fn init_expert_usage(
        path: &Path,
        model_id: &str,
    ) -> Option<Arc<Mutex<crate::expert_cache::UsageLearner>>> {
        use rustllama_models::accel;
        let budget = accel::moe_expert_cache_max_bytes();
        let tier = accel::moe_tier_active();
        // Learn routing usage when EITHER the LRU pin-cache is on (budget > 0) OR
        // the tiered-expert engine is active — the tier's hottest-first promotion
        // needs the usage sidecar too, and without this it never fills (ranked=0).
        if budget == 0 && !tier {
            return None;
        }
        let registered = accel::expert_registry_len();
        // The LRU pin-cache needs file-backed registered experts; the tier does
        // NOT (its experts are device-resident via promotion, not the mmap pin
        // registry), so only bail on an empty registry for the LRU-only path.
        if budget > 0 && registered == 0 {
            tracing::warn!(
                model_id = %model_id,
                "moe expert cache enabled but no file-backed experts registered — \
                 dense model, or zero-copy weights disabled before the first load \
                 (RUSTLLAMA_ZEROCOPY_WEIGHTS). Experts will stream unpinned."
            );
            return None;
        }
        let learning = std::env::var("RUSTLLAMA_MOE_EXPERT_LEARNING")
            .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
            .unwrap_or(true);
        if !learning {
            tracing::info!(
                model_id = %model_id,
                registered_experts = registered,
                "expert cache active (learning sidecar disabled) — LRU only"
            );
            return None;
        }
        let learner = crate::expert_cache::UsageLearner::open(path)?;
        if budget > 0 {
            // LRU pin-cache: pre-pin the hottest experts from the sidecar.
            let t0 = std::time::Instant::now();
            let (pinned, bytes) = learner.prepin(budget);
            tracing::info!(
                model_id = %model_id,
                registered_experts = registered,
                budget_mb = budget / (1024 * 1024),
                prepinned_experts = pinned,
                prepinned_mb = bytes / (1024 * 1024),
                elapsed_ms = t0.elapsed().as_millis() as u64,
                "expert cache: learning sidecar loaded, hot experts pre-pinned"
            );
        } else {
            // Tiered-expert engine (no LRU cache): no pre-pin — the sidecar just
            // records routed-expert usage so the NEXT load promotes hottest-first.
            tracing::info!(
                model_id = %model_id,
                "moe tiered-expert: usage-learning sidecar active — routed-expert \
                 counts persist so promotion is hottest-first on the next load"
            );
        }
        let learner = Arc::new(Mutex::new(learner));
        let weak = Arc::downgrade(&learner);
        // Periodic flusher: folds routing stats into the sidecar
        // while the server runs (long-lived processes rarely Drop the
        // engine). Exits once the engine drops its Arc.
        std::thread::Builder::new()
            .name("rl-expert-usage-flush".into())
            .spawn(move || loop {
                std::thread::sleep(std::time::Duration::from_secs(60));
                match weak.upgrade() {
                    Some(l) => {
                        if let Ok(mut g) = l.lock() {
                            g.flush(false);
                        }
                        let s = rustllama_models::accel::expert_cache_stats();
                        if s.hits + s.misses > 0 {
                            tracing::info!(
                                hit_rate_pct = format!("{:.1}", s.hit_rate() * 100.0),
                                hits = s.hits,
                                misses = s.misses,
                                pinned_mb = s.pinned_bytes / (1024 * 1024),
                                resident_experts = s.resident_experts,
                                "expert cache telemetry"
                            );
                        }
                    }
                    None => break,
                }
            })
            .ok();
        Some(learner)
    }

    /// Load the text decoder + a paired mmproj GGUF (the LLaVA-style
    /// vision tower file) so the engine can serve VLM requests.
    ///
    /// Validates at load time that the projector's output dimension
    /// (`d_text`) matches the text decoder's hidden dim (`d_model`);
    /// a mismatch means the caller paired an mmproj from one model
    /// with the text decoder from another (e.g. a Qwen2-VL mmproj
    /// against a LLaVA-1.5 text decoder), which would silently produce
    /// garbage at inference. Failing here gives the user a clear
    /// "these two files don't go together" error.
    ///
    /// `image_placeholder` is the literal string the text tokenizer
    /// will encounter at image positions in the rendered prompt — the
    /// engine tokenizes it once at load and stores the resulting
    /// single token id. v1 requires the placeholder to tokenize to
    /// exactly one token; multi-token placeholders (Qwen2-VL's
    /// `<|vision_start|><|image_pad|><|vision_end|>` sequence) are
    /// a v2 surgery on the placeholder-scan side and rejected here.
    pub fn load_with_mmproj(
        text_path: &Path,
        mmproj_path: &Path,
        max_ctx: usize,
        image_placeholder: &str,
    ) -> Result<Self> {
        Self::load_with_mmproj_with_mode(
            text_path,
            mmproj_path,
            max_ctx,
            image_placeholder,
            rustllama_models::vision_arch::PlaceholderMode::OnePerImage,
        )
    }

    /// Same as [`load_with_mmproj`] but with an explicit
    /// [`PlaceholderMode`](rustllama_models::vision_arch::PlaceholderMode).
    /// Use `OnePerImage` for LLaVA-1.5 / LLaVA-Next family (one
    /// placeholder per image; vision pipeline's full patch sequence
    /// expands at that position) and `OnePerPatch` for Qwen2-VL
    /// (each image emits `num_patches` consecutive copies of
    /// `<|image_pad|>`; the splice is 1:1 per placeholder).
    pub fn load_with_mmproj_with_mode(
        text_path: &Path,
        mmproj_path: &Path,
        max_ctx: usize,
        image_placeholder: &str,
        placeholder_mode: rustllama_models::vision_arch::PlaceholderMode,
    ) -> Result<Self> {
        // Reuse the text-only load path; loading the mmproj second
        // means a malformed text GGUF surfaces first (faster, more
        // common failure) rather than after the mmproj cost.
        let mut engine = Self::load_inner(
            text_path,
            max_ctx,
            true,
            KvDtype::F32,
            "contiguous",
            0,
            None,
        )?;
        engine.attach_mmproj(mmproj_path, image_placeholder, placeholder_mode)?;
        Ok(engine)
    }

    /// Attach a vision tower (mmproj GGUF) to an ALREADY-LOADED
    /// engine — composable with every real load path (`serve` loads
    /// via `load_with_options_*` with the user's KV dtype/layout; the
    /// old `load_with_mmproj*` entry points hardcoded their own F32
    /// load and were only reachable from tests). Validates the
    /// projector↔decoder pairing, resolves the single-token image
    /// placeholder, and installs the chat-side injection wrapper
    /// (vision start/end markers for the Qwen3-VL merger family).
    pub fn attach_mmproj(
        &mut self,
        mmproj_path: &Path,
        image_placeholder: &str,
        placeholder_mode: rustllama_models::vision_arch::PlaceholderMode,
    ) -> Result<()> {
        let engine = self;
        let mmproj_gguf = Gguf::open(mmproj_path)?;
        let vision = rustllama_models::vision_arch::VisionModel::load(&mmproj_gguf)
            .map_err(|e| CpuEngineError::Other(format!("mmproj load: {e}")))?;

        // Pairing validation: projector.d_text must equal text decoder
        // d_model. A common foot-gun is to grab any LLaVA mmproj off
        // HuggingFace and pair it with a non-LLaVA text model — same
        // arch class but different hidden dims silently feed garbage
        // into the transformer body. Reject loudly.
        let d_text_proj = vision.projector.d_text();
        let d_text_model = engine.model.cfg.d_model;
        if d_text_proj != d_text_model {
            return Err(CpuEngineError::Other(format!(
                "mmproj projector output dim {d_text_proj} does not match text \
                 decoder d_model {d_text_model} — this mmproj GGUF was trained \
                 against a different text decoder; pair files from the same model"
            )));
        }

        // Tokenize the placeholder string and demand a single-token
        // result. Both `OnePerImage` (LLaVA: 1 placeholder per image)
        // and `OnePerPatch` (Qwen2-VL: N consecutive placeholders
        // per image) need a single token id to scan for — the modes
        // differ only in how many runs of that id they expect in the
        // prompt.
        let tokenizer = engine.tokenizer.as_ref().ok_or_else(|| {
            CpuEngineError::NoTokenizer("load_with_mmproj requires a tokenizer in the text GGUF")
        })?;
        let ids = tokenizer.encode(image_placeholder, false).map_err(|e| {
            CpuEngineError::Other(format!(
                "tokenize image placeholder '{image_placeholder}': {e}"
            ))
        })?;
        if ids.len() != 1 {
            return Err(CpuEngineError::Other(format!(
                "image placeholder '{image_placeholder}' tokenizes to {} tokens; \
                 a single-token placeholder is required (Qwen2-VL uses \
                 `<|image_pad|>` repeated N times — that's a single token \
                 id repeated, not a multi-token string)",
                ids.len()
            )));
        }
        engine.image_token_id = Some(ids[0]);
        // Chat-side injection wrapper: the Qwen3-VL family brackets
        // the pad token with vision start/end markers (single special
        // tokens in the paired text tokenizer); classic CLIP/LLaVA
        // paths inject the bare placeholder.
        engine.image_wrapper = Some(
            if vision.cfg.projector_type
                == rustllama_models::vision_arch::ProjectorType::Qwen3VlMerger
            {
                format!("<|vision_start|>{image_placeholder}<|vision_end|>")
            } else {
                image_placeholder.to_string()
            },
        );
        engine.vision = Some(Arc::new(vision));
        engine.placeholder_mode = Some(placeholder_mode);

        tracing::info!(
            model_id = %engine.model_id,
            d_text = d_text_proj,
            image_token_id = ids[0],
            placeholder = %image_placeholder,
            wrapper = %engine.image_wrapper.as_deref().unwrap_or(""),
            mode = ?placeholder_mode,
            "vision tower (mmproj) attached — VLM enabled"
        );
        Ok(())
    }

    /// Read-only access to the loaded vision tower, if any. Used by
    /// the chat path to call into [`prepare_vlm_inputs`] when image
    /// bytes are attached. `None` for the text-only configuration.
    pub fn vision(&self) -> Option<&Arc<rustllama_models::vision_arch::VisionModel>> {
        self.vision.as_ref()
    }

    /// Token id of the image placeholder in the text tokenizer's vocab,
    /// resolved at [`load_with_mmproj`] time. `None` when no mmproj
    /// is loaded.
    pub fn image_token_id(&self) -> Option<u32> {
        self.image_token_id
    }

    /// Auto-detect dispatcher: route a model path to the right loader.
    ///
    /// - A **directory** is an mlx-lm / mlx-community MLX model layout
    ///   (`config.json` + `*.safetensors` + `tokenizer.json`) → [`load_mlx`]
    ///   when it detects as MLX; an AWQ/GPTQ export handed as a directory
    ///   falls back to its `model.safetensors`.
    /// - `.gguf` → [`load_with_tokenizer`]; `.safetensors` →
    ///   [`load_safetensors`] (or [`load_mlx`] on the parent dir when the
    ///   sibling `config.json` + `.scales`/`.biases` mark it an MLX affine
    ///   checkpoint, not AWQ/GPTQ).
    ///
    /// Returns an error for unrecognized extensions rather than guessing —
    /// silently mis-classifying a checkpoint would surface as a cryptic
    /// parse failure deep in the loader.
    pub fn load_auto(path: &Path, max_ctx: usize) -> Result<Self> {
        // A model DIRECTORY has no meaningful extension; classify it by
        // contents. MLX first (its `.scales`/`.biases` triple is the
        // discriminator vs AWQ's `.scales`/`.qzeros`), then fall back to a
        // single `model.safetensors` for AWQ/GPTQ exported as a folder.
        if path.is_dir() {
            if Self::is_mlx_dir(path) {
                return Self::load_mlx(path, max_ctx);
            }
            let st = path.join("model.safetensors");
            if st.is_file() {
                return Self::load_safetensors(&st, max_ctx);
            }
            return Err(CpuEngineError::Other(format!(
                "model directory `{}` is neither an MLX affine checkpoint \
                 (a `quantization` block in config.json + `.scales`/`.biases` \
                 tensors) nor an AWQ/GPTQ export (no `model.safetensors`)",
                path.display()
            )));
        }
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase());
        match ext.as_deref() {
            Some("gguf") => Self::load_with_tokenizer(path, max_ctx),
            Some("safetensors") => {
                // An MLX affine checkpoint pointed at by its `.safetensors`
                // file (not the directory): route to the MLX loader on the
                // parent dir. Plain AWQ/GPTQ keep the existing path.
                if let Some(parent) = path.parent() {
                    if Self::is_mlx_dir(parent) {
                        return Self::load_mlx(parent, max_ctx);
                    }
                }
                Self::load_safetensors(path, max_ctx)
            }
            _ => Err(CpuEngineError::Other(format!(
                "unrecognized model file extension for `{}`: rustllama v1 \
                 supports `.gguf` (GGUF), `.safetensors` (HuggingFace \
                 AWQ / GPTQ / fp16 checkpoints), and an MLX model directory \
                 (mlx-lm / mlx-community)",
                path.display()
            ))),
        }
    }

    /// MLX-directory probe: does `dir` hold a `config.json` with a
    /// `quantization` block AND — across ALL its `*.safetensors` shards
    /// merged — a packed uint32 `.weight` next to a `.scales` sibling? That
    /// uint32-weight + scales pair distinguishes an MLX checkpoint (affine OR
    /// non-affine mxfp4/mxfp8/nvfp4) from AWQ/GPTQ (whose packed tensor is
    /// `.qweight`) and from a plain fp16 HF dump. Delegates to the
    /// shard-aware [`rustllama_safetensors::is_mlx_dir`], which merges every
    /// shard's HEADER (via mmap — no weight data read) before matching, so a
    /// sharded model whose `.weight` and `.scales` land in different shards
    /// is still detected. Any IO / parse error is swallowed as `false`.
    fn is_mlx_dir(dir: &Path) -> bool {
        rustllama_safetensors::is_mlx_dir(dir)
    }

    /// Load a HuggingFace `.safetensors` checkpoint (AWQ, GPTQ, or
    /// unquantized fp16). The path must point at the safetensors
    /// file; rustllama looks for the sibling `config.json` and
    /// `tokenizer.json` in the same directory:
    ///
    /// ```text
    ///   path/to/qwen2.5-coder-7b-awq/
    ///     config.json             ← architecture metadata
    ///     model.safetensors       ← packed weights (`path` argument)
    ///     tokenizer.json          ← HF tokenizer
    /// ```
    ///
    /// Sharded models (`model-00001-of-00004.safetensors`) are not
    /// supported in v1 — convert via `safetensors merge` or use a
    /// single-file release.
    pub fn load_safetensors(path: &Path, max_ctx: usize) -> Result<Self> {
        let parent = path.parent().ok_or_else(|| {
            CpuEngineError::Other(format!(
                "safetensors path `{}` has no parent directory",
                path.display()
            ))
        })?;
        let config_path = parent.join("config.json");
        let tokenizer_path = parent.join("tokenizer.json");
        if !config_path.exists() {
            return Err(CpuEngineError::Other(format!(
                "missing `config.json` next to `{}` — HuggingFace safetensors \
                 layouts require it for architecture metadata",
                path.display()
            )));
        }

        // Read + parse config.json. `parse_hf_config` does the
        // architecture-name normalization + field mapping.
        let config_json = std::fs::read_to_string(&config_path)
            .map_err(|e| CpuEngineError::Other(format!("read config.json: {e}")))?;
        let cfg = rustllama_safetensors::parse_hf_config(&config_json)
            .map_err(|e| CpuEngineError::Other(format!("config.json: {e}")))?;

        // Read + parse the safetensors blob. v1 reads the full file
        // into memory (the converter walks every tensor anyway); a
        // future mmap-based pass can shave peak RSS for very large
        // models.
        let blob = std::fs::read(path)
            .map_err(|e| CpuEngineError::Other(format!("read safetensors: {e}")))?;
        let converted = rustllama_safetensors::convert_safetensors_to_gguf_tensors(&blob)
            .map_err(|e| CpuEngineError::Other(format!("safetensors convert: {e}")))?;
        let model = rustllama_safetensors::build_llama_model_from_safetensors(&cfg, converted)
            .map_err(|e| CpuEngineError::Other(format!("safetensors build: {e}")))?;

        // Tokenizer: optional. Most HF releases ship `tokenizer.json`;
        // a small subset use the older `tokenizer.model` (SentencePiece)
        // which v1 doesn't read. Warn rather than fail if absent so the
        // engine can still serve a non-chat workload (e.g. unit-tests).
        let tokenizer = if tokenizer_path.exists() {
            Some(Arc::new(Tokenizer::from_file(&tokenizer_path).map_err(
                |e| CpuEngineError::Other(format!("tokenizer.json: {e}")),
            )?))
        } else {
            tracing::warn!(
                path = %tokenizer_path.display(),
                "tokenizer.json not present alongside safetensors — `chat` / \
                 `generate` will require a tokenizer; loading model only"
            );
            None
        };

        // KV backend: contiguous F32 in v1 (same default as the GGUF
        // load path). VLM + safetensors needs a future loader on the
        // vision side.
        let ctx = max_ctx.min(cfg.ctx_train.max(max_ctx));
        let kv_backend =
            KvBackend::from_inference_config("contiguous", &cfg, ctx as u32, KvDtype::F32)
                .map_err(|e| CpuEngineError::Other(format!("kv_backend init: {e}")))?;

        let model_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "rustllama-safetensors".into());

        tracing::info!(
            model_id = %model_id,
            arch = %cfg.arch,
            n_layers = cfg.n_layers,
            d_model = cfg.d_model,
            n_heads = cfg.n_heads,
            n_kv_heads = cfg.n_kv_heads,
            vocab_size = cfg.vocab_size,
            ctx_train = cfg.ctx_train,
            ctx = ctx,
            "safetensors model loaded — engine ready"
        );

        let delta_net_cache = build_delta_net_cache(&model)?;
        let model = Arc::new(model);
        let lock_registry = crate::pagelock::lock_model_into_ram(Arc::clone(&model)).map(Arc::new);
        let kv_dtype_cached = kv_backend.kv_dtype();
        Ok(Self {
            model,
            tokenizer,
            kv_dtype_cached,
            metrics_cache: Arc::new(MetricsCacheAtomics::default()),
            state: Arc::new(Mutex::new(EngineState {
                kv_backend,
                last_ids: Vec::new(),
                pool: PrefixCachePool::new(DEFAULT_PREFIX_CACHE_MAX_SNAPSHOTS),
                delta_net_cache,
            })),
            sycl_worker: Arc::new(SyclWorker::spawn()),
            max_ctx: ctx,
            model_id,
            source_path: path.to_path_buf(),
            prefix_cache: true,
            prefill_chunk_size: DEFAULT_PREFILL_CHUNK,
            prefix_cache_max_snapshots: DEFAULT_PREFIX_CACHE_MAX_SNAPSHOTS,
            last_stats: Arc::new(Mutex::new(RequestStats::default())),
            ema_tok_s_bits: Arc::new(std::sync::atomic::AtomicU64::new(f64::NAN.to_bits())),
            cumulative_stats: Arc::new(crate::CumulativeStats::default()),
            max_tool_iterations: DEFAULT_MAX_TOOL_ITERATIONS,
            flash_attention: true,
            kv_layout: "contiguous".to_string(),
            kv_page_size: 0, // unused on contiguous; KvBackend defaults
            n_gpu_layers: u32::MAX,
            cpu_force_patterns: Vec::new(),
            vision: None,
            image_token_id: None,
            placeholder_mode: None,
            ngram_spec: None,
            lookahead_spec: None,
            draft_spec: None,
            mtp_spec: false,
            tree_spec: None,
            eagle_head: None,
            image_wrapper: None,
            vision_feature_memo: Arc::new(Mutex::new(None)),
            lock_registry,
            // Safetensors VLM path is never a canonical Mamba model.
            mamba: None,
            // Safetensors weights are owned-heap (nothing mmap-backed
            // to pin), so the expert learning cache never applies.
            expert_usage: None,
            kv_persist_owner: false,
            cacheblend: None,
            checkpoint_clock: Mutex::new(crate::kv_persist::CheckpointClock::new(
                crate::kv_persist::kv_persist_interval(),
                std::time::Instant::now(),
            )),
            lora: Arc::new(std::sync::RwLock::new(
                rustllama_models::lora::LoraRegistry::new(),
            )),
        })
    }

    /// Load an Apple **MLX** affine-quantized model directory (mlx-lm /
    /// mlx-community layout): a `config.json` carrying the HF architecture
    /// hyperparameters **and** an MLX `quantization` block, one or more
    /// `*.safetensors` shards whose quantized linears/embeddings are stored
    /// as the `<module>.weight` (uint32-packed) + `.scales` + `.biases`
    /// affine triple, and a `tokenizer.json`:
    ///
    /// ```text
    ///   path/to/Qwen2.5-0.5B-Instruct-4bit/
    ///     config.json         ← HF arch metadata + `quantization` block
    ///     model.safetensors   ← packed MLX affine weights (+ scales/biases)
    ///     tokenizer.json      ← HF tokenizer
    /// ```
    ///
    /// **Strategy — load-time transcode to GGUF block-quant.** Each affine
    /// weight is remapped to the SAME GGUF slot the AWQ/GPTQ path targets
    /// (`model.layers.0.self_attn.q_proj.weight` → `blk.0.attn_q.weight`) and
    /// re-encoded to a standard GGUF quant (4-bit → `Q4_1`, 8-bit → `Q8_0`;
    /// see [`mlx_model_to_converted`]), producing the same
    /// [`ConvertedTensor`] shape `convert_safetensors_to_gguf_tensors` yields
    /// so [`build_llama_model_from_safetensors`] and the whole arch/forward
    /// stack are reused unchanged — and, crucially, so the mature, autotuned
    /// CPU/SYCL/CUDA/MLX-Metal quant kernels run the model with zero
    /// MLX-specific dispatch (MLX on the GPU "for free"). The earlier
    /// native-packed-residency step (`MlxAffineQuant` blobs + the CPU
    /// `matvec_mlx_affine_*` path) survives only as the fallback for odd
    /// bit-widths, `RUSTLLAMA_MLX_NATIVE=1`, and slim (`encoder`-off) builds.
    ///
    /// [`mlx_model_to_converted`]: Self::mlx_model_to_converted
    ///
    /// KV cache is F32 contiguous (same default as the GGUF + AWQ paths).
    ///
    /// [`ConvertedTensor`]: rustllama_safetensors::ConvertedTensor
    /// [`build_llama_model_from_safetensors`]: rustllama_safetensors::build_llama_model_from_safetensors
    pub fn load_mlx(dir: &Path, max_ctx: usize) -> Result<Self> {
        let config_path = dir.join("config.json");
        let tokenizer_path = dir.join("tokenizer.json");
        if !config_path.exists() {
            return Err(CpuEngineError::Other(format!(
                "missing `config.json` in MLX model directory `{}` — mlx-lm \
                 layouts require it for architecture + quantization metadata",
                dir.display()
            )));
        }

        // config.json feeds two parsers: the HF-arch hyperparams (via
        // `parse_hf_config`, shared with the AWQ path) and the MLX
        // `quantization` block (parsed inside `load_mlx_dir`).
        let config_json = std::fs::read_to_string(&config_path)
            .map_err(|e| CpuEngineError::Other(format!("read config.json: {e}")))?;
        let cfg = rustllama_safetensors::parse_hf_config(&config_json)
            .map_err(|e| CpuEngineError::Other(format!("config.json: {e}")))?;

        // Read every shard, split into quantized affine weights + full
        // tensors, then dequant + remap to the GGUF-named ConvertedTensor
        // list the llama builder consumes.
        let mlx = rustllama_safetensors::load_mlx_dir(dir)
            .map_err(|e| CpuEngineError::Other(format!("mlx safetensors load: {e}")))?;
        let n_quant = mlx.quant.len();
        let n_micro = mlx.micro.len();
        let n_full = mlx.full.len();
        let converted = Self::mlx_model_to_converted(&mlx)?;
        let model = rustllama_safetensors::build_llama_model_from_safetensors(&cfg, converted)
            .map_err(|e| CpuEngineError::Other(format!("mlx model build: {e}")))?;

        // Tokenizer: optional, same posture as `load_safetensors` — most
        // mlx-community releases ship `tokenizer.json`; warn (don't fail)
        // when it's absent so the model still loads for the token-ID API.
        let tokenizer = if tokenizer_path.exists() {
            Some(Arc::new(Tokenizer::from_file(&tokenizer_path).map_err(
                |e| CpuEngineError::Other(format!("tokenizer.json: {e}")),
            )?))
        } else {
            tracing::warn!(
                path = %tokenizer_path.display(),
                "tokenizer.json not present in MLX model directory — `chat` / \
                 `generate` will require a tokenizer; loading model only"
            );
            None
        };

        let ctx = max_ctx.min(cfg.ctx_train.max(max_ctx));
        let kv_backend =
            KvBackend::from_inference_config("contiguous", &cfg, ctx as u32, KvDtype::F32)
                .map_err(|e| CpuEngineError::Other(format!("kv_backend init: {e}")))?;

        let model_id = dir
            .file_name()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "rustllama-mlx".into());

        tracing::info!(
            model_id = %model_id,
            arch = %cfg.arch,
            n_layers = cfg.n_layers,
            d_model = cfg.d_model,
            n_heads = cfg.n_heads,
            n_kv_heads = cfg.n_kv_heads,
            vocab_size = cfg.vocab_size,
            ctx_train = cfg.ctx_train,
            ctx = ctx,
            quant_weights = n_quant,
            micro_weights = n_micro,
            full_tensors = n_full,
            "MLX model loaded (affine + mxfp4/mxfp8/nvfp4 → GGUF-quant transcode; \
             GPU-capable) — engine ready"
        );

        let delta_net_cache = build_delta_net_cache(&model)?;
        let model = Arc::new(model);
        let lock_registry = crate::pagelock::lock_model_into_ram(Arc::clone(&model)).map(Arc::new);
        let kv_dtype_cached = kv_backend.kv_dtype();
        Ok(Self {
            model,
            tokenizer,
            kv_dtype_cached,
            metrics_cache: Arc::new(MetricsCacheAtomics::default()),
            state: Arc::new(Mutex::new(EngineState {
                kv_backend,
                last_ids: Vec::new(),
                pool: PrefixCachePool::new(DEFAULT_PREFIX_CACHE_MAX_SNAPSHOTS),
                delta_net_cache,
            })),
            sycl_worker: Arc::new(SyclWorker::spawn()),
            max_ctx: ctx,
            model_id,
            source_path: dir.to_path_buf(),
            prefix_cache: true,
            prefill_chunk_size: DEFAULT_PREFILL_CHUNK,
            prefix_cache_max_snapshots: DEFAULT_PREFIX_CACHE_MAX_SNAPSHOTS,
            last_stats: Arc::new(Mutex::new(RequestStats::default())),
            ema_tok_s_bits: Arc::new(std::sync::atomic::AtomicU64::new(f64::NAN.to_bits())),
            cumulative_stats: Arc::new(crate::CumulativeStats::default()),
            max_tool_iterations: DEFAULT_MAX_TOOL_ITERATIONS,
            flash_attention: true,
            kv_layout: "contiguous".to_string(),
            kv_page_size: 0, // unused on contiguous; KvBackend defaults
            n_gpu_layers: u32::MAX,
            cpu_force_patterns: Vec::new(),
            vision: None,
            image_token_id: None,
            placeholder_mode: None,
            ngram_spec: None,
            lookahead_spec: None,
            draft_spec: None,
            mtp_spec: false,
            tree_spec: None,
            eagle_head: None,
            image_wrapper: None,
            vision_feature_memo: Arc::new(Mutex::new(None)),
            lock_registry,
            // Packed MLX-affine VLM path is never a canonical Mamba model.
            mamba: None,
            // Packed MLX-affine weights are owned-heap blobs (nothing
            // mmap-backed to pin), so the expert learning cache never applies.
            expert_usage: None,
            kv_persist_owner: false,
            cacheblend: None,
            checkpoint_clock: Mutex::new(crate::kv_persist::CheckpointClock::new(
                crate::kv_persist::kv_persist_interval(),
                std::time::Instant::now(),
            )),
            lora: Arc::new(std::sync::RwLock::new(
                rustllama_models::lora::LoraRegistry::new(),
            )),
        })
    }

    /// Remap a loaded [`MlxModel`](rustllama_safetensors::MlxModel) into the
    /// GGUF-named [`ConvertedTensor`](rustllama_safetensors::ConvertedTensor)
    /// list `build_llama_model_from_safetensors` consumes. Quantized linears
    /// are **transcoded at load** to a GGUF block-quant (4-bit → `Q4_1`,
    /// 8-bit → `Q8_0`) via [`mlx_quant_to_converted_bytes`], so the model
    /// runs on the shared, autotuned CPU/SYCL/CUDA/MLX-Metal quant kernels
    /// (MLX on the GPU "for free"); odd bit-widths, `RUSTLLAMA_MLX_NATIVE=1`,
    /// and slim builds keep the native packed `MlxAffineRaw` blob instead.
    /// Full-precision tensors (norms, biases, unquantized embeddings) pass
    /// through as f16/f32.
    ///
    /// [`mlx_quant_to_converted_bytes`]: Self::mlx_quant_to_converted_bytes
    ///
    /// Two naming families are accepted so both real mlx-community
    /// checkpoints AND rustllama's own `quantize --to-mlx` exports load:
    ///
    /// 1. **HF module paths** (`model.layers.0.self_attn.q_proj`,
    ///    `model.embed_tokens`, `lm_head`, `model.norm`) — real mlx-lm /
    ///    mlx-community models. Routed via
    ///    [`map_hf_to_gguf`](rustllama_safetensors::map_hf_to_gguf), exactly
    ///    as the AWQ/GPTQ path maps its HF names.
    /// 2. **GGUF names already** (`blk.0.attn_q`, `token_embd`, `output`,
    ///    `output_norm.weight`) — what `quantize --to-mlx` emits (it keeps
    ///    the source GGUF tensor names verbatim). Passed through unchanged,
    ///    so the encoder round-trip loads without a separate GGUF-name path.
    ///
    /// **Orientation (the one thing to get right):** MLX affine is
    /// row-major `[out_features, in_features]` — element `(r, c)` is at
    /// packed bit `(r*in_features + c)*bits` — which is byte-identical to
    /// the row-major `[out_features, in_features]` layout the llama builder
    /// expects (the same layout the AWQ converter *transposes into*). So
    /// **no transpose** is applied; the packed blob maps straight to a
    /// `ConvertedTensor` and the matvec `m`/`k` fall out of the logical
    /// shape.
    fn mlx_model_to_converted(
        mlx: &rustllama_safetensors::MlxModel,
    ) -> Result<Vec<rustllama_safetensors::ConvertedTensor>> {
        use rustllama_safetensors::{ConvertedDtype, ConvertedTensor, MlxFullDtype};

        // Quantized-weight module path → GGUF slot. HF name via the shared
        // table first; else GGUF-name passthrough (encoder round-trip).
        fn quant_name_to_gguf(module: &str) -> Option<String> {
            let hf_weight = format!("{module}.weight");
            if let Some(m) = rustllama_safetensors::map_hf_to_gguf(&hf_weight) {
                return Some(m.gguf_name);
            }
            if module.starts_with("blk.") || matches!(module, "token_embd" | "output") {
                return Some(hf_weight);
            }
            None
        }
        // Full-tensor name → GGUF slot (norms, biases, un-quantized
        // embeddings / lm_head). HF name via the table; else already-GGUF.
        fn full_name_to_gguf(name: &str) -> Option<String> {
            if let Some(m) = rustllama_safetensors::map_hf_to_gguf(name) {
                return Some(m.gguf_name);
            }
            if name.starts_with("blk.")
                || matches!(
                    name,
                    "token_embd.weight" | "output.weight" | "output_norm.weight"
                )
            {
                return Some(name.to_string());
            }
            None
        }

        // MoE detection + per-weight skip. An MLX MoE checkpoint
        // (Qwen2-MoE / Qwen3-MoE / OLMoE / Mixtral) names its FFN
        // experts `...mlp.experts.{E}.{gate,up,down}_proj`, its router
        // `...mlp.gate`, and (Qwen2-MoE) a shared expert
        // `...mlp.shared_expert.*` + `...mlp.shared_expert_gate`. Those
        // are stacked into the GGUF MoE tensor set by
        // `mlx_moe_to_converted`, so the per-weight generic loops below
        // must SKIP them (they'd otherwise have no GGUF mapping and be
        // dropped). Dense MLX models have no `.mlp.experts.` modules and
        // take the unchanged path.
        fn mlx_mlp_moe_module(module: &str) -> bool {
            let Some(rest) = module.strip_prefix("model.layers.") else {
                return false;
            };
            let Some(dot) = rest.find('.') else {
                return false;
            };
            if rest[..dot].parse::<usize>().is_err() {
                return false;
            }
            let Some(mlp) = rest[dot + 1..].strip_prefix("mlp.") else {
                return false;
            };
            mlp.starts_with("experts.")
                || mlp == "gate"
                || mlp.starts_with("shared_expert.")
                || mlp == "shared_expert_gate"
        }
        let is_moe = mlx
            .quant
            .keys()
            .chain(mlx.micro.keys())
            .any(|m| m.contains(".mlp.experts."));

        let mut out = Vec::with_capacity(mlx.quant.len() + mlx.full.len());

        // --- Quantized affine weights → load-time GGUF-quant transcode ---
        // "Transcode everywhere but Metal": each quantized MLX-affine linear
        // is dequantized to f32 and RE-ENCODED to a standard GGUF block-quant
        // (`mlx_quant_to_converted_bytes`) whose CPU / SYCL / CUDA / MLX-Metal
        // matvec + embedding kernels are already mature and autotuned. The
        // payoff: an MLX model then loads + runs as an ordinary quantized
        // model on EVERY backend with ZERO MLX-specific dispatch — so MLX
        // runs on the GPU "for free" via the existing Q4_1/Q8_0 kernels (the
        // weight is `Dtype::Q4_1Raw`/`Q8_0Raw` from here on, indistinguishable
        // from a GGUF model's).
        //
        // The native packed-affine blob (`MlxAffineRaw` + the CPU
        // `matvec_mlx_affine_*` path, the B2 residency format) is kept only
        // as the FALLBACK for: odd bit-widths (2/3/5/6, no faithful GGUF
        // analog), `RUSTLLAMA_MLX_NATIVE=1` (force native for ALL weights —
        // A/B the requant delta, exact-value parity, and the future Apple
        // Metal `quantized_matmul` fast path), and slim builds without the
        // `encoder` feature (no gguf encode tables). The target-format policy
        // + requant rationale live on `mlx_quant_to_converted_bytes`. No
        // transpose — MLX affine is already row-major
        // `[out_features, in_features]` (see fn docs).
        let force_native = std::env::var("RUSTLLAMA_MLX_NATIVE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let (mut n_q4_1, mut n_q4_k, mut n_q8_0, mut n_native) = (0usize, 0usize, 0usize, 0usize);
        for (module, q) in &mlx.quant {
            if is_moe && mlx_mlp_moe_module(module) {
                continue; // handled by mlx_moe_to_converted (stacked)
            }
            let Some(gguf_name) = quant_name_to_gguf(module) else {
                tracing::debug!(
                    module = %module,
                    "mlx load: quantized module has no GGUF mapping — skipping"
                );
                continue;
            };
            q.validate()
                .map_err(|e| CpuEngineError::Other(format!("mlx weight `{module}`: {e}")))?;
            let (dtype, bytes) = Self::mlx_quant_to_converted_bytes(q, force_native);
            match dtype {
                ConvertedDtype::Q4_1Raw => n_q4_1 += 1,
                ConvertedDtype::Q4_KRaw => n_q4_k += 1,
                ConvertedDtype::Q8_0Raw => n_q8_0 += 1,
                _ => n_native += 1,
            }
            out.push(ConvertedTensor {
                gguf_name,
                shape: q.shape.clone(),
                dtype,
                bytes,
            });
        }
        tracing::info!(
            transcoded_q4_1 = n_q4_1,
            transcoded_q4_k = n_q4_k,
            transcoded_q8_0 = n_q8_0,
            kept_native_affine = n_native,
            force_native,
            "mlx load: quantized-linear transcode complete (affine → GGUF \
             block-quant; native kept for odd-bit / RUSTLLAMA_MLX_NATIVE / \
             slim build)"
        );

        // --- Non-affine microscaling weights (MXFP4 / MXFP8 / NVFP4) ----
        // Real mlx-lm also ships microscaling checkpoints (no `.biases`;
        // E8M0/E4M3 uint8 block scales). Each is repacked from MLX's packed
        // layout into our GGUF-style block bytes, dequantized via the
        // already-parity-checked Wave-2 reference, and RE-ENCODED to the SAME
        // GGUF block-quant the affine path targets (4-bit → Q4_K/Q4_1, 8-bit
        // → Q8_0). So MXFP/NVFP MLX models run on the identical mature,
        // autotuned CPU/SYCL/CUDA/MLX-Metal quant kernels with zero
        // format-specific dispatch — the FP4→Q4_K requant is near-lossless
        // (Q4_K carries more per-weight precision than FP4). We emit the
        // existing ConvertedDtype variants rather than a native
        // `Mxfp4Raw`/… tensor to reuse the proven affine downstream (the
        // ConvertedDtype → Tensor bridge has no micro-raw variant); a native
        // no-requant path would need new ConvertedDtype + builder wiring.
        if !mlx.micro.is_empty() {
            #[cfg(not(feature = "encoder"))]
            {
                return Err(CpuEngineError::Other(format!(
                    "MLX non-affine (mxfp4/mxfp8/nvfp4) weights need the \
                     `encoder` feature to transcode to a GGUF block-quant; \
                     this slim build cannot load {} such weight(s)",
                    mlx.micro.len()
                )));
            }
            #[cfg(feature = "encoder")]
            {
                let (mut nm_q4_1, mut nm_q4_k, mut nm_q8_0) = (0usize, 0usize, 0usize);
                for (module, m) in &mlx.micro {
                    if is_moe && mlx_mlp_moe_module(module) {
                        continue; // handled by mlx_moe_to_converted (stacked)
                    }
                    let Some(gguf_name) = quant_name_to_gguf(module) else {
                        tracing::debug!(
                            module = %module,
                            "mlx load: micro module has no GGUF mapping — skipping"
                        );
                        continue;
                    };
                    m.validate().map_err(|e| {
                        CpuEngineError::Other(format!("mlx micro weight `{module}`: {e}"))
                    })?;
                    let (dtype, bytes) = Self::mlx_micro_to_converted_bytes(m)?;
                    match dtype {
                        ConvertedDtype::Q4_1Raw => nm_q4_1 += 1,
                        ConvertedDtype::Q4_KRaw => nm_q4_k += 1,
                        ConvertedDtype::Q8_0Raw => nm_q8_0 += 1,
                        _ => {}
                    }
                    out.push(ConvertedTensor {
                        gguf_name,
                        shape: m.shape.clone(),
                        dtype,
                        bytes,
                    });
                }
                tracing::info!(
                    micro_weights = mlx.micro.len(),
                    transcoded_q4_1 = nm_q4_1,
                    transcoded_q4_k = nm_q4_k,
                    transcoded_q8_0 = nm_q8_0,
                    "mlx load: non-affine (mxfp4/mxfp8/nvfp4) transcode complete \
                     (repack → Wave-2 dequant → GGUF block-quant)"
                );
            }
        }

        // --- Full-precision tensors → pass through ----------------------
        for (name, full) in &mlx.full {
            let Some(gguf_name) = full_name_to_gguf(name) else {
                tracing::debug!(
                    name = %name,
                    "mlx load: full tensor has no GGUF mapping — skipping"
                );
                continue;
            };
            let (dtype, bytes) = match full.dtype {
                MlxFullDtype::F32 => (ConvertedDtype::F32, full.bytes.clone()),
                MlxFullDtype::F16 => (ConvertedDtype::F16, full.bytes.clone()),
                MlxFullDtype::Bf16 => {
                    // BF16 = high 16 bits of an f32; widen then round to f16
                    // — the same conversion `convert::convert_plain` applies,
                    // so a bf16 norm reaches the builder as the f16 it wants.
                    let mut b = Vec::with_capacity(full.bytes.len());
                    for c in full.bytes.chunks_exact(2) {
                        let bf = u16::from_le_bytes([c[0], c[1]]);
                        let f = f32::from_bits((bf as u32) << 16);
                        b.extend_from_slice(&half::f16::from_f32(f).to_le_bytes());
                    }
                    (ConvertedDtype::F16, b)
                }
            };
            out.push(ConvertedTensor {
                gguf_name,
                shape: full.shape.clone(),
                dtype,
                bytes,
            });
        }

        // --- MoE: stack experts + router + shared expert ---------------
        // Appended after the per-weight (attention / norm / embedding /
        // lm_head) tensors so a MoE model carries BOTH its dense-shaped
        // slots (handled above) and its stacked expert set (here).
        if is_moe {
            let moe_tensors = Self::mlx_moe_to_converted(mlx)?;
            tracing::info!(
                moe_tensors = moe_tensors.len(),
                "mlx load: MoE experts/router/shared stacked into GGUF MoE \
                 tensor set (per-expert transcode → stacked block-quant)"
            );
            out.extend(moe_tensors);
        }

        Ok(out)
    }

    /// Choose the on-model representation for one quantized MLX-affine
    /// linear and produce its bytes — the core of the load-time transcoder.
    ///
    /// Dequantizes the packed affine weight to f32 and re-encodes it to the
    /// GGUF block-quant whose shared CPU/SYCL/CUDA/MLX-Metal matvec +
    /// embedding kernels are already mature and autotuned:
    ///
    /// **Target-format policy** (the one requant choice that matters):
    ///   - **4-bit → `Q4_K` when `in_features % 256 == 0`, else `Q4_1`.**
    ///     Q4_K (256-weight super-blocks: 6-bit sub-scales under an f16 super
    ///     `d`/`dmin`) is higher quality-per-bit than Q4_1 — its two-level
    ///     scale structure spends fewer bits on per-group metadata — so it's
    ///     the preferred 4-bit target. But Q4_K quantizes each row in
    ///     super-blocks of 256, so the row length (`in_features`, the
    ///     quantized/contraction dim) must be a multiple of 256 or a
    ///     super-block would straddle a weight-row boundary. Rows that aren't
    ///     256-aligned fall back to **Q4_1**: block-32 with an f16 scale `d` +
    ///     f16 min `m` — the *faithful affine analog* of MLX's group `scale·q
    ///     + bias`, and block-32 fits any column count (`in_features` is
    ///     always a multiple of the 32/64/128 MLX group size, hence of 32, so
    ///     its blocks never cross a row boundary either). Both 4→4-bit
    ///     requants are near-lossless.
    ///   - **8-bit → `Q8_0`** (block-32, symmetric int8 — the GGUF analog).
    ///
    /// Falls back to the native packed `MlxAffineRaw` blob (decoded by the
    /// CPU `matvec_mlx_affine_*` path) when: `force_native` is set
    /// (`RUSTLLAMA_MLX_NATIVE=1`), the bit-width is one of 2/3/5/6 (no
    /// faithful GGUF analog), or — defensively — the element count isn't a
    /// multiple of 32 (which a real MLX weight never hits). The
    /// `#[cfg(not(feature = "encoder"))]` twin below handles slim builds that
    /// ship no gguf encode tables: there every weight stays native.
    #[cfg(feature = "encoder")]
    fn mlx_quant_to_converted_bytes(
        q: &rustllama_tensor::MlxAffineQuant,
        force_native: bool,
    ) -> (rustllama_safetensors::ConvertedDtype, Vec<u8>) {
        use rustllama_safetensors::ConvertedDtype;

        let n = q.n_elements() as usize;
        // `0` = keep native affine. `force_native` or a non-block-32 count
        // short-circuits before any dequant work.
        let target_bits = if force_native || n % 32 != 0 {
            0
        } else {
            q.bits
        };

        match target_bits {
            4 | 8 => {
                let f32buf = Self::mlx_dequant_affine_f32(q, n);
                // Dequantized f32 → the matching GGUF block-quant (4-bit →
                // Q4_K/Q4_1, 8-bit → Q8_0). Shared with the micro path.
                Self::transcode_f32_to_gguf(&f32buf, target_bits, q.in_features()).unwrap_or_else(
                    || {
                        // Unreachable for a real affine weight (in_features is
                        // a multiple of group_size ⇒ of 32); keep native as a
                        // defensive fallback rather than panicking.
                        (ConvertedDtype::MlxAffineRaw, q.to_blob())
                    },
                )
            }
            // Native fallback: odd bit-width (2/3/5/6), RUSTLLAMA_MLX_NATIVE=1,
            // or a non-block-32 geometry.
            _ => (ConvertedDtype::MlxAffineRaw, q.to_blob()),
        }
    }

    /// Slim-build twin of [`mlx_quant_to_converted_bytes`]: without the
    /// `encoder` feature there are no gguf encode-side tables, so every MLX
    /// affine weight stays packed as native `MlxAffineRaw` (CPU-only affine
    /// matvec) — the pre-transcode behavior.
    #[cfg(not(feature = "encoder"))]
    fn mlx_quant_to_converted_bytes(
        q: &rustllama_tensor::MlxAffineQuant,
        _force_native: bool,
    ) -> (rustllama_safetensors::ConvertedDtype, Vec<u8>) {
        (
            rustllama_safetensors::ConvertedDtype::MlxAffineRaw,
            q.to_blob(),
        )
    }

    /// Dequantize one packed MLX-affine weight to a flat f32 buffer in
    /// `[out_features, in_features]` row-major order (the layout the GGUF
    /// encoders + llama builder consume — no transpose). Shared by the Q4_1
    /// and Q8_0 transcode arms of [`mlx_quant_to_converted_bytes`].
    #[cfg(feature = "encoder")]
    fn mlx_dequant_affine_f32(q: &rustllama_tensor::MlxAffineQuant, n: usize) -> Vec<f32> {
        let mut f32buf = vec![0f32; n];
        rustllama_kernels_cpu::mlx_affine::dequantize_mlx_affine(
            &q.packed,
            &q.scales,
            &q.biases,
            q.group_size,
            q.bits,
            &mut f32buf,
        );
        f32buf
    }

    /// Re-encode a dequantized f32 weight (row-major `[out_features,
    /// in_features]`) to the matching GGUF block-quant. Shared by the MLX
    /// affine and non-affine (micro) transcode paths.
    ///
    /// **Target-format policy** (same as the affine path's docs):
    ///   - **4-bit → `Q4_K`** when `in_features % 256 == 0` (256-weight
    ///     super-blocks must stay inside one weight row), **else `Q4_1`**
    ///     (block-32, fits any multiple of 32).
    ///   - **8-bit → `Q8_0`** (block-32 symmetric int8).
    ///
    /// Returns `None` when the geometry can't host the block-quant without a
    /// block straddling a weight-row boundary (`in_features` not a multiple
    /// of 32 — impossible for affine/mxfp4/mxfp8, only reachable by an
    /// exotic nvfp4 shape), or for an unsupported `bits`. Callers decide the
    /// fallback (affine keeps native; micro errors).
    #[cfg(feature = "encoder")]
    fn transcode_f32_to_gguf(
        f32buf: &[f32],
        bits: u32,
        in_features: u64,
    ) -> Option<(rustllama_safetensors::ConvertedDtype, Vec<u8>)> {
        use rustllama_safetensors::ConvertedDtype;
        let n = f32buf.len();
        match bits {
            4 => {
                if in_features % 256 == 0 {
                    // Q4_K: 144 B / 256-weight super-block.
                    let mut enc = vec![0u8; (n / 256) * 144];
                    rustllama_gguf::encode_k::encode_q4_k(f32buf, &mut enc);
                    Some((ConvertedDtype::Q4_KRaw, enc))
                } else if in_features % 32 == 0 {
                    // Q4_1: 20 B / 32-weight block (f16 d + f16 min + 16 B codes).
                    let mut enc = vec![0u8; (n / 32) * 20];
                    rustllama_gguf::encode::encode_q4_1(f32buf, &mut enc);
                    Some((ConvertedDtype::Q4_1Raw, enc))
                } else {
                    None
                }
            }
            8 => {
                if in_features % 32 == 0 {
                    // Q8_0: 34 B / 32-weight block (f16 d + 32 × i8 codes).
                    let mut enc = vec![0u8; (n / 32) * 34];
                    rustllama_gguf::encode::encode_q8_0(f32buf, &mut enc);
                    Some((ConvertedDtype::Q8_0Raw, enc))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Transcode one non-affine microscaling MLX weight (`mxfp4` / `mxfp8` /
    /// `nvfp4`) to a GGUF block-quant [`ConvertedTensor`] payload.
    ///
    /// Path: repack MLX's `(packed uint32 codes, uint8 block scales)` into
    /// our GGUF-style block bytes ([`MlxMicroQuant::to_gguf_blocks`]) →
    /// dequantize via the already-parity-checked Wave-2 reference
    /// (`dequant_mxfp4/mxfp8/nvfp4`, byte-exact with the SYCL/CUDA kernels)
    /// → re-encode to Q4_K/Q4_1 (4-bit) or Q8_0 (8-bit). The FP4→Q4_K
    /// requant is near-lossless (Q4_K carries more per-weight precision than
    /// FP4). Reusing the affine downstream (shared `transcode_f32_to_gguf`)
    /// means MXFP/NVFP MLX models run on every backend's mature quant
    /// kernels with zero format-specific dispatch.
    #[cfg(feature = "encoder")]
    fn mlx_micro_to_converted_bytes(
        m: &rustllama_safetensors::MlxMicroQuant,
    ) -> Result<(rustllama_safetensors::ConvertedDtype, Vec<u8>)> {
        use rustllama_safetensors::MlxQuantMode;
        let n = m.n_elements() as usize;
        let blocks = m.to_gguf_blocks();
        let mut f32buf = vec![0f32; n];
        match m.mode {
            MlxQuantMode::Mxfp4 => rustllama_gguf::dequant::dequant_mxfp4(&blocks, &mut f32buf),
            MlxQuantMode::Mxfp8 => rustllama_gguf::dequant::dequant_mxfp8(&blocks, &mut f32buf),
            MlxQuantMode::Nvfp4 => rustllama_gguf::dequant::dequant_nvfp4(&blocks, &mut f32buf),
            ref other => {
                return Err(CpuEngineError::Other(format!(
                    "mlx micro weight `{}`: non-micro mode {other:?} reached the \
                     micro transcoder",
                    m.name
                )));
            }
        }
        Self::transcode_f32_to_gguf(&f32buf, m.bits, m.in_features()).ok_or_else(|| {
            CpuEngineError::Other(format!(
                "mlx micro weight `{}`: geometry (in_features {}, {} bits) has no \
                 faithful GGUF block-quant target",
                m.name,
                m.in_features(),
                m.bits
            ))
        })
    }

    /// Group an MLX MoE checkpoint's per-expert / router / shared-expert
    /// linears into the GGUF MoE tensor set the engine's MoE forward
    /// already consumes:
    ///
    /// - `blk.{N}.ffn_{gate,up,down}_exps.weight` — ONE stacked tensor
    ///   per layer+projection holding every expert's transcoded
    ///   block-quant bytes back-to-back, in expert-index order. That is
    ///   exactly the byte layout [`rustllama_models::moe::expert_view`]
    ///   slices (offset = `expert * byte_size(d_out*d_in)`), so the
    ///   builder's per-expert views land on the right bytes.
    /// - `blk.{N}.ffn_gate_inp.weight` — the router (`mlp.gate`).
    /// - `blk.{N}.ffn_{gate,up,down}_shexp.weight` +
    ///   `blk.{N}.ffn_gate_inp_shexp.weight` — Qwen2-MoE's always-on
    ///   shared expert and its sigmoid gate (absent on Qwen3-MoE /
    ///   OLMoE / Mixtral).
    ///
    /// Every weight runs through the SAME per-weight transcoder the
    /// dense path uses (affine or mxfp*/nvfp* → Q4_K/Q4_1/Q8_0), so a
    /// MoE MLX model becomes an ordinary stacked-quant MoE model that
    /// runs on every backend with no MLX-specific dispatch. Experts must
    /// transcode to a block-quant (never native `MlxAffineRaw`) so the
    /// stacked bytes stay uniform + sliceable; an odd-bit / force-native
    /// expert is rejected rather than silently mis-stacked.
    #[cfg(feature = "encoder")]
    fn mlx_moe_to_converted(
        mlx: &rustllama_safetensors::MlxModel,
    ) -> Result<Vec<rustllama_safetensors::ConvertedTensor>> {
        use rustllama_safetensors::{ConvertedDtype, ConvertedTensor};
        use std::collections::BTreeMap;

        #[derive(Clone, Copy)]
        enum Proj {
            Gate,
            Up,
            Down,
        }
        enum Role {
            Expert {
                layer: usize,
                proj: Proj,
                expert: usize,
            },
            Router {
                layer: usize,
            },
            Shared {
                layer: usize,
                proj: Proj,
            },
            SharedGate {
                layer: usize,
            },
        }
        fn proj_from(s: &str) -> Option<Proj> {
            match s {
                "gate_proj" => Some(Proj::Gate),
                "up_proj" => Some(Proj::Up),
                "down_proj" => Some(Proj::Down),
                _ => None,
            }
        }
        fn classify(module: &str) -> Option<Role> {
            let rest = module.strip_prefix("model.layers.")?;
            let dot = rest.find('.')?;
            let layer: usize = rest[..dot].parse().ok()?;
            let mlp = rest[dot + 1..].strip_prefix("mlp.")?;
            if let Some(e) = mlp.strip_prefix("experts.") {
                let edot = e.find('.')?;
                let expert: usize = e[..edot].parse().ok()?;
                let proj = proj_from(&e[edot + 1..])?;
                return Some(Role::Expert {
                    layer,
                    proj,
                    expert,
                });
            }
            if mlp == "gate" {
                return Some(Role::Router { layer });
            }
            if mlp == "shared_expert_gate" {
                return Some(Role::SharedGate { layer });
            }
            if let Some(sp) = mlp.strip_prefix("shared_expert.") {
                return Some(Role::Shared {
                    layer,
                    proj: proj_from(sp)?,
                });
            }
            None
        }
        let proj_code = |p: Proj| -> u8 {
            match p {
                Proj::Gate => 0,
                Proj::Up => 1,
                Proj::Down => 2,
            }
        };

        // Stage every MoE weight (transcoded) with its classified role,
        // from BOTH the affine and the microscaling maps.
        let mut staged: Vec<(Role, ConvertedDtype, Vec<u8>, Vec<u64>, String)> = Vec::new();
        for (module, q) in &mlx.quant {
            let Some(role) = classify(module) else {
                continue;
            };
            let (dtype, bytes) = Self::mlx_quant_to_converted_bytes(q, false);
            staged.push((role, dtype, bytes, q.shape.clone(), module.clone()));
        }
        for (module, m) in &mlx.micro {
            let Some(role) = classify(module) else {
                continue;
            };
            let (dtype, bytes) = Self::mlx_micro_to_converted_bytes(m)?;
            staged.push((role, dtype, bytes, m.shape.clone(), module.clone()));
        }

        // Route: experts accumulate into per-(layer, proj) maps keyed by
        // expert index (BTreeMap → ascending order); router / shared
        // emit single GGUF-named tensors immediately.
        // (layer, proj_code) -> expert_idx -> (dtype, bytes, [out, in]).
        let mut experts: BTreeMap<
            (usize, u8),
            BTreeMap<usize, (ConvertedDtype, Vec<u8>, Vec<u64>)>,
        > = BTreeMap::new();
        let mut out: Vec<ConvertedTensor> = Vec::new();
        for (role, dtype, bytes, shape, module) in staged {
            if matches!(dtype, ConvertedDtype::MlxAffineRaw) {
                return Err(CpuEngineError::Other(format!(
                    "mlx MoE weight `{module}` did not transcode to a GGUF \
                     block-quant (kept native affine — odd bit-width or \
                     RUSTLLAMA_MLX_NATIVE set); MoE needs stackable uniform \
                     expert blocks"
                )));
            }
            match role {
                Role::Expert {
                    layer,
                    proj,
                    expert,
                } => {
                    experts
                        .entry((layer, proj_code(proj)))
                        .or_default()
                        .insert(expert, (dtype, bytes, shape));
                }
                Role::Router { layer } => {
                    out.push(ConvertedTensor {
                        gguf_name: format!("blk.{layer}.ffn_gate_inp.weight"),
                        shape,
                        dtype,
                        bytes,
                    });
                }
                Role::Shared { layer, proj } => {
                    let nm = match proj {
                        Proj::Gate => "ffn_gate_shexp",
                        Proj::Up => "ffn_up_shexp",
                        Proj::Down => "ffn_down_shexp",
                    };
                    out.push(ConvertedTensor {
                        gguf_name: format!("blk.{layer}.{nm}.weight"),
                        shape,
                        dtype,
                        bytes,
                    });
                }
                Role::SharedGate { layer } => {
                    out.push(ConvertedTensor {
                        gguf_name: format!("blk.{layer}.ffn_gate_inp_shexp.weight"),
                        shape,
                        dtype,
                        bytes,
                    });
                }
            }
        }

        // Stack each (layer, proj) expert group: concat expert 0's bytes,
        // then 1, … (BTreeMap iterates in ascending index order). Enforce
        // a dense 0..n index set + a single uniform dtype so the stacked
        // blob slices cleanly.
        for ((layer, pc), per_expert) in experts {
            let n = per_expert.len();
            let mut dtype: Option<ConvertedDtype> = None;
            let mut dims: Option<(u64, u64)> = None;
            let mut expect = 0usize;
            let mut all_bytes: Vec<u8> = Vec::new();
            for (idx, (dt, b, sh)) in per_expert {
                if idx != expect {
                    return Err(CpuEngineError::Other(format!(
                        "mlx MoE layer {layer} proj {pc}: experts not a dense \
                         0..n index set (saw {idx}, expected {expect})"
                    )));
                }
                expect += 1;
                match dtype {
                    None => dtype = Some(dt),
                    Some(d) if d == dt => {}
                    Some(_) => {
                        return Err(CpuEngineError::Other(format!(
                            "mlx MoE layer {layer} proj {pc}: experts transcoded \
                             to mixed dtypes — cannot stack"
                        )))
                    }
                }
                if dims.is_none() && sh.len() == 2 {
                    dims = Some((sh[0], sh[1]));
                }
                all_bytes.extend_from_slice(&b);
            }
            let dtype = dtype.expect("non-empty expert group");
            let (d_out, d_in) = dims.expect("2-D expert shape");
            let nm = match pc {
                0 => "ffn_gate_exps",
                1 => "ffn_up_exps",
                _ => "ffn_down_exps",
            };
            out.push(ConvertedTensor {
                gguf_name: format!("blk.{layer}.{nm}.weight"),
                shape: vec![n as u64, d_out, d_in],
                dtype,
                bytes: all_bytes,
            });
        }

        Ok(out)
    }

    /// Slim-build twin of [`mlx_moe_to_converted`]: without the `encoder`
    /// feature there are no gguf encode tables to transcode the experts,
    /// so a MoE MLX model cannot be stacked. Fail loudly.
    #[cfg(not(feature = "encoder"))]
    fn mlx_moe_to_converted(
        _mlx: &rustllama_safetensors::MlxModel,
    ) -> Result<Vec<rustllama_safetensors::ConvertedTensor>> {
        Err(CpuEngineError::Other(
            "MLX MoE load requires the `encoder` feature to transcode + stack \
             experts into a GGUF block-quant; this slim build cannot load a \
             MoE MLX model"
                .to_string(),
        ))
    }

    /// Test-only hook: install a pre-loaded vision tower and a chosen
    /// image-placeholder token id directly, bypassing the
    /// [`load_with_mmproj`] tokenizer single-token check. The check
    /// rejects strings that the GPT-2 BPE synth tokenizer can't
    /// surface as a single id (which is most strings); this hook
    /// lets VLM-prefill tests pick any vocab id without hand-rolling
    /// a tokenizer extension.
    ///
    /// Not part of the stable API — the `__test` prefix + `doc(hidden)`
    /// make that explicit. Production load path stays
    /// [`load_with_mmproj`].
    #[doc(hidden)]
    pub fn __install_vision_for_test(
        &mut self,
        vision: Arc<rustllama_models::vision_arch::VisionModel>,
        image_token_id: u32,
    ) {
        self.vision = Some(vision);
        self.image_token_id = Some(image_token_id);
        self.placeholder_mode = Some(rustllama_models::vision_arch::PlaceholderMode::OnePerImage);
    }

    /// Test-only: install vision tower with explicit placeholder
    /// mode. Used by the OnePerPatch (Qwen2-VL) parity tests.
    #[doc(hidden)]
    pub fn __install_vision_for_test_with_mode(
        &mut self,
        vision: Arc<rustllama_models::vision_arch::VisionModel>,
        image_token_id: u32,
        mode: rustllama_models::vision_arch::PlaceholderMode,
    ) {
        self.vision = Some(vision);
        self.image_token_id = Some(image_token_id);
        self.placeholder_mode = Some(mode);
    }

    /// Test-only accessor for the loaded text model. Lets V-6b-3c
    /// parity tests compare `vlm_prefill_ids` against a plain
    /// `forward_prefill` on the same model without exposing the
    /// `model` Arc on the stable surface.
    #[doc(hidden)]
    pub fn __model_for_test(&self) -> &rustllama_models::llama_arch::LlamaModel {
        &self.model
    }

    /// VLM prefill primitive (V-6b-3c): given a tokenized prompt and a
    /// list of raw image bytes, build the spliced input embedding
    /// sequence and run the transformer's prefill pass over it.
    ///
    /// Returns the final logit vector — exactly what
    /// `forward_prefill` returns for a text-only prompt.
    ///
    /// The pipeline (one call to each of V-3 / V-5 / V-6b-1 / V-6b-3a
    /// / V-6b-3b):
    ///
    /// 1. [`prepare_vlm_inputs`] locates image-placeholder positions
    ///    in `prompt_ids`, runs the vision pipeline on each attached
    ///    image, returns `[positions, features, d_text]`.
    /// 2. [`LlamaModel::embed_tokens`] looks up text-token rows from
    ///    the text decoder's embedding table.
    /// 3. [`splice_image_embeddings`] merges text rows with image
    ///    feature rows at the right positions, producing the full
    ///    `[seq_len, d_model]` input embedding buffer.
    /// 4. [`LlamaModel::forward_prefill_from_embeds`] runs the
    ///    transformer body over the spliced buffer, updating `kv`
    ///    in place.
    ///
    /// # Errors
    ///
    /// - [`CpuEngineError::Other`] if no mmproj is loaded
    ///   (`load_with_mmproj` hasn't been called).
    /// - Pipeline / splice / shape errors are propagated through the
    ///   same channel with a descriptive message.
    ///
    /// # Notes
    ///
    /// Takes `prompt_ids: &[i32]` rather than `&str` so callers that
    /// already have a tokenized prompt (e.g. the chat loop after
    /// `tokenizer.render_chat + encode`) don't re-tokenize. The
    /// chat-stream wiring on top of this primitive lands in V-6b-3d.
    pub fn vlm_prefill_ids(
        &self,
        prompt_ids: &[i32],
        image_payloads: &[&[u8]],
        start_pos: u32,
        kv: &mut rustllama_models::llama_arch::KvCache,
    ) -> Result<Vec<f32>> {
        use rustllama_models::vision_arch::{
            prepare_vlm_inputs_with_mode, splice_image_embeddings, PlaceholderMode,
        };

        let vision = self.vision.as_ref().ok_or_else(|| {
            CpuEngineError::Other(
                "vlm_prefill_ids requires a vision tower loaded via \
                 load_with_mmproj"
                    .into(),
            )
        })?;
        let image_token_id = self.image_token_id.ok_or_else(|| {
            // The two fields are populated atomically by load_with_mmproj,
            // so this branch should be unreachable — keep it for
            // defense-in-depth so a future load surface that forgets to
            // wire image_token_id surfaces here cleanly.
            CpuEngineError::Other("image_token_id missing despite vision loaded".into())
        })?;
        let placeholder_mode = self
            .placeholder_mode
            .unwrap_or(PlaceholderMode::OnePerImage);

        // The `prepare_vlm_inputs` helper wants `u32` ids (token-vocab
        // domain), while the text-decoder forward path uses `i32` (so
        // the embedding-table lookup can pass them through without
        // wraparound on legal vocab sizes). Translate at the boundary.
        let prompt_u32: Vec<u32> = prompt_ids.iter().map(|&t| t as u32).collect();
        let vlm_inputs = prepare_vlm_inputs_with_mode(
            vision.as_ref(),
            &prompt_u32,
            image_token_id,
            image_payloads,
            placeholder_mode,
        )
        .map_err(|e| CpuEngineError::Other(format!("vlm prefill: {e}")))?;

        // Embed the text tokens (placeholder positions included — the
        // splice helper drops those rows at the splice index).
        let text_embeds = self.model.embed_tokens(prompt_ids);
        // d_text == d_model is guaranteed by load_with_mmproj.
        let d_model = self.model.cfg.d_model;
        debug_assert_eq!(vlm_inputs.d_text, d_model);
        let feat_slices = vlm_inputs.feature_slices();
        let spliced =
            splice_image_embeddings(&text_embeds, &vlm_inputs.positions, &feat_slices, d_model)
                .map_err(|e| CpuEngineError::Other(format!("vlm splice: {e}")))?;
        // Final transformer prefill over the spliced embedding sequence.
        // M5-engine-wire: for a Qwen2-VL / Qwen2.5-VL model (rope.dimension_
        // sections set, 1:1 OnePerPatch splice) drive the sectioned 3-D mRoPE
        // prefill so image-then-text positions rotate correctly. Otherwise —
        // a non-sectioned model, or a layout whose placeholder runs don't
        // partition into the per-image grids — fall back to the scalar entry
        // (byte-identical). The grid rule lives in the models surface
        // (`VlmPrefillInputs::mrope_positions_for_sequence`), not here.
        let seq_len = spliced.len() / d_model;
        Ok(
            match self
                .model
                .cfg
                .rope_dimension_sections
                .as_ref()
                .and_then(|_| vlm_inputs.mrope_positions_for_sequence(seq_len, start_pos))
            {
                Some(mrope) => self
                    .model
                    .forward_prefill_from_embeds_mrope(&spliced, start_pos, &mrope, kv),
                None => self
                    .model
                    .forward_prefill_from_embeds(&spliced, start_pos, kv),
            },
        )
    }

    /// Eagerly pre-upload every packed-quant weight tensor to USM at
    /// load time. Shifts the "first chat lazy weight upload" cost
    /// (~1-2 s for a 7B Q4_K_M model on Iris Xe shared memory) to
    /// model load time, where the user already expects a wait. After
    /// this returns, the user's first chat starts decoding without
    /// burning seconds copying weights into shared GPU memory.
    ///
    /// Because the USM weight cache lives in **thread-local** storage,
    /// the warmup must run on the same kind of thread that later runs
    /// generations — tokio's blocking pool. The pool reuses recently-
    /// idle threads LIFO, so the warmed thread is overwhelmingly the
    /// one that picks up the first chat request.
    ///
    /// On platforms where USM is unavailable (no GPU,
    /// `RUSTLLAMA_SYCL_DISPATCH=0`, or no `RUSTLLAMA_USM_ATTN`) this
    /// is a fast no-op returning `0`.
    ///
    /// Returns the number of weight tensors uploaded (0 on no-op /
    /// failure; ~200 for a typical 28-layer 7B model).
    ///
    /// Idempotent — calling twice on the same engine returns the
    /// same warmed thread's cache and skips already-uploaded
    /// weights.
    pub async fn warmup_for_sycl_async(self: &Arc<Self>) -> usize {
        let start = std::time::Instant::now();
        let (done_tx, done_rx) = tokio::sync::oneshot::channel::<usize>();
        self.sycl_worker.submit_warmup(WarmupJob {
            model: Arc::clone(&self.model),
            max_ctx: self.max_ctx,
            n_gpu_layers: self.n_gpu_layers,
            done_tx,
        });
        let uploaded = done_rx.await.unwrap_or(0);
        if uploaded > 0 {
            tracing::info!(
                uploaded,
                elapsed_ms = start.elapsed().as_millis() as u64,
                "load-time USM weight pre-upload complete on dedicated worker \
                 — first chat starts warm-cached on the same thread"
            );
        }
        uploaded
    }

    /// Cap on complete tool-call bodies the streaming grammar allows
    /// per response. `0` disables the cap (no limit on recursion).
    /// Defaults to [`DEFAULT_MAX_TOOL_ITERATIONS`].
    pub fn set_max_tool_iterations(&mut self, n: u32) {
        self.max_tool_iterations = n;
    }

    pub fn max_tool_iterations(&self) -> u32 {
        self.max_tool_iterations
    }

    /// Snapshot of the most recent generation's stats. Call this after
    /// the engine's response stream has drained while still holding
    /// the per-model gate; otherwise a concurrent next request may
    /// overwrite the slot. See [`RequestStats`].
    pub fn last_request_stats(&self) -> RequestStats {
        self.last_stats.lock().expect("last_stats lock").clone()
    }

    /// Exponential moving average of recent decode-throughput samples
    /// (tok/s). `None` until the first non-trivial generation
    /// completes. Smoothed over ~4 requests' worth of samples (alpha
    /// = 0.3) so the GUI's "tok/s" indicator stays readable instead
    /// of swinging 5× between a cold first probe and a warm second.
    ///
    /// The single-request `last_tok_s` (computed from
    /// `last_request_stats()`) remains available alongside this for
    /// callers that want the instantaneous number.
    pub fn ema_tok_s(&self) -> Option<f64> {
        use std::sync::atomic::Ordering;
        let bits = self.ema_tok_s_bits.load(Ordering::Relaxed);
        let v = f64::from_bits(bits);
        if v.is_nan() {
            None
        } else {
            Some(v)
        }
    }

    /// Commit per-request stats: stores them in `last_stats` for the
    /// metrics endpoint to read, folds the decode throughput into
    /// `ema_tok_s_bits`, and folds every counter into the lifetime
    /// `cumulative_stats`. Zero-token / zero-time requests skip the
    /// EMA update so a cancelled request can't pin the EMA to zero,
    /// but they still increment the cumulative request count (a
    /// 0-token request still happened).
    fn commit_request_stats(&self, stats: RequestStats) {
        update_ema_tok_s(&self.ema_tok_s_bits, &stats);
        self.cumulative_stats.add(&stats);
        *self.last_stats.lock().expect("last_stats") = stats;
    }

    /// Cumulative-since-startup counters snapshot. Lockless atomic
    /// reads — safe to call from the metrics polling path on any
    /// thread. Exposed as both an inherent method and via the
    /// [`crate::Engine`] trait (`cumulative_stats_snapshot`).
    pub fn cumulative_request_stats(&self) -> crate::CumulativeStatsSnapshot {
        self.cumulative_stats.snapshot()
    }

    /// The on-disk GGUF path this engine was loaded from. Useful for
    /// finding sidecar files (e.g., HF model card JSON).
    pub fn source_path(&self) -> &std::path::Path {
        &self.source_path
    }

    /// Enable or disable the prefix-reuse cache. Default is enabled — a clean
    /// engine starts with `last_ids` empty so the first generation can't
    /// reuse anything regardless.
    pub fn set_prefix_cache(&mut self, enabled: bool) {
        self.prefix_cache = enabled;
    }

    pub fn prefix_cache_enabled(&self) -> bool {
        self.prefix_cache
    }

    /// Override the prefill chunk size (default: 512). Smaller values give
    /// finer partial-cache snapshots on cancellation at the cost of slightly
    /// more bookkeeping; larger values mean less bookkeeping but a longer
    /// gap between cancel-recovery checkpoints. Clamped to at least 1.
    pub fn set_prefill_chunk_size(&mut self, size: usize) {
        self.prefill_chunk_size = size.max(1);
    }

    pub fn prefill_chunk_size(&self) -> usize {
        self.prefill_chunk_size
    }

    /// Enable / disable n-gram (prompt-lookup) speculative decoding for
    /// this engine's text-generation path. `None` (default) keeps the
    /// classic single-token decode. When `Some`, `chat` / `generate`
    /// route through [`Self::speculate_ngram_stream`], which proposes
    /// tokens from history n-gram matches and verifies them in one
    /// batched target forward per round (see `verify_speculation_inner`).
    /// No second model is required.
    pub fn set_ngram_speculative(&mut self, cfg: Option<NgramDrafterConfig>) {
        self.ngram_spec = cfg;
    }

    pub fn ngram_speculative(&self) -> Option<NgramDrafterConfig> {
        self.ngram_spec
    }

    /// F10 — enable / disable lookahead (Jacobi n-gram-pool) speculative
    /// decoding for this engine's text-generation path. `None` (default)
    /// keeps the classic single-token decode (byte-identical). When `Some`,
    /// grammar-free `chat` / `generate` route through
    /// [`Self::speculate_lookahead_stream_from_ids`], which harvests an
    /// n-gram pool from the realized history and walks it for a Jacobi
    /// window of guess tokens, verified in one batched forward per round via
    /// the SAME accept/reject the n-gram path uses. No second model required.
    pub fn set_lookahead_speculative(&mut self, cfg: Option<LookaheadDrafterConfig>) {
        self.lookahead_spec = cfg;
    }

    pub fn lookahead_speculative(&self) -> Option<LookaheadDrafterConfig> {
        self.lookahead_spec
    }

    /// Enable / disable MTP / NextN self-speculative decoding for this
    /// engine's text-generation path. `false` (default) keeps the
    /// classic single-token decode. When `true` AND the loaded model is
    /// hybrid with a NextN head, grammar-free `chat` / `generate`
    /// requests route through [`Self::speculate_mtp_stream_from_ids`],
    /// which uses the model's own NextN head to draft the +2 token each
    /// round and verifies it with the next forward. On a model without a
    /// NextN head this is a silent no-op (classic decode). Takes
    /// precedence over n-gram speculation when both are enabled. Set at
    /// load time from `[inference].speculative_mtp`.
    pub fn set_mtp_speculative(&mut self, on: bool) {
        self.mtp_spec = on;
    }

    /// Whether MTP self-speculation is enabled AND the loaded model can
    /// actually use it (hybrid + NextN head present). The server routes
    /// on this so it can fall back cleanly on non-MTP models.
    pub fn mtp_speculative(&self) -> bool {
        self.mtp_spec && self.model_supports_mtp()
    }

    /// True when the loaded model is hybrid and carries a NextN head —
    /// the precondition for the MTP self-speculative driver.
    fn model_supports_mtp(&self) -> bool {
        self.model.weights.is_hybrid() && self.model.weights.nextn_head.is_some()
    }

    /// Enable (or clear) tree-structured MTP speculation (Feature 4). Set at load
    /// time from `[inference].speculative_tree` + `speculative_tree_widths`.
    pub fn set_tree_speculative(&mut self, widths: Option<Vec<u32>>) {
        self.tree_spec = widths.filter(|w| !w.is_empty());
    }

    /// Whether tree spec is enabled AND the loaded model can use it. Both
    /// architecture families are supported: a NON-hybrid model with MTP heads
    /// (the F32-contiguous batched tree verify) and a HYBRID model with a draft
    /// head (the recurrent-state tree walk — see [`tree_mtp_round_hybrid`]).
    pub fn tree_speculative(&self) -> bool {
        self.tree_spec.is_some() && self.model_supports_tree()
    }

    /// P2: true iff this engine's contiguous KV cache has any SWA ring
    /// layer. All speculative drivers (n-gram / MTP / tree / draft-model)
    /// verify via a BATCHED target forward (and tree spec compacts KV with
    /// [`KvCache::compact_tokens`]) — both assume a full `max_ctx`-strided
    /// contiguous history and would mis-address a `cap`-strided ring, so a
    /// ring model falls back to classic single-token decode (which drives
    /// the ring-aware decode arm). Cheap read-lock; the dispatch sites that
    /// call this do not hold the state lock. A poisoned lock defaults to
    /// `true` (disable spec — the safe direction).
    fn swa_ring_decode(&self) -> bool {
        match self.state.lock() {
            Ok(st) => matches!(&st.kv_backend, KvBackend::Contiguous(kv) if kv.has_ring_layers()),
            Err(_) => true,
        }
    }

    /// A model can draft+verify a tree iff it carries a DRAFT head. Non-hybrid
    /// models draft from DeepSeek-style `mtp_heads`; hybrid models draft from
    /// either `mtp_heads` OR the single qwen35moe `nextn_head` (the hybrid tree
    /// verify handles the DeltaNet/SSM recurrence under the tree). WITHOUT a
    /// draft head the round is inert — classic decode — so this gate keeps the
    /// driver honest: no draft head ⇒ no tree spec, same as the linear MTP
    /// driver's `model_supports_mtp`.
    fn model_supports_tree(&self) -> bool {
        let w = &self.model.weights;
        if w.is_hybrid() {
            w.mtp_heads.is_some() || w.nextn_head.is_some()
        } else {
            w.mtp_heads.is_some()
        }
    }

    /// P10b — attach (or clear) a loaded EAGLE draft head. `Some` routes
    /// grammar-free, non-SWA-ring decode on a supported model through
    /// [`Self::speculate_eagle_stream`]; `None` restores the pre-EAGLE path
    /// byte-identically. Forks inherit the head.
    pub fn set_eagle_head(&mut self, head: Option<Arc<rustllama_models::llama_arch::EagleHead>>) {
        self.eagle_head = head;
    }

    /// The loaded EAGLE head, if any (for the serve/HTTP load paths + tests).
    pub fn eagle_head(&self) -> Option<Arc<rustllama_models::llama_arch::EagleHead>> {
        self.eagle_head.clone()
    }

    /// P10b — load an EAGLE draft head from a companion GGUF and attach it,
    /// validated against this model's config. Mirrors P11's `--lora` attach
    /// (load-then-set). Errors (not an eagle GGUF / shape mismatch / missing
    /// fusion) propagate so a misconfigured path fails the load loudly rather
    /// than silently degrading decode.
    pub fn attach_eagle_head(&mut self, path: &std::path::Path) -> Result<()> {
        let head = rustllama_models::llama_arch::EagleHead::from_path(path, &self.model.cfg)
            .map_err(|e| CpuEngineError::Other(format!("eagle head load failed: {e}")))?;
        self.eagle_head = Some(Arc::new(head));
        Ok(())
    }

    /// Whether EAGLE speculation is enabled AND the loaded model can use it: a
    /// dense/MoE model (NOT hybrid / MLA — those take different forward bodies
    /// the EAGLE tip forward doesn't cover). The dense contiguous tree verify
    /// is the shared stack; the per-round guard also re-checks the backend.
    pub fn eagle_speculative(&self) -> bool {
        self.eagle_head.is_some() && self.model_supports_eagle()
    }

    /// A model can host an EAGLE draft head iff it runs the standard dense/MoE
    /// forward ([`LlamaModel::forward_one_with_hidden`] captures its hidden) —
    /// i.e. NOT hybrid and NOT MLA. The companion head is model-matched; this
    /// gate keeps the driver honest on an arch the tip forward can't feed.
    fn model_supports_eagle(&self) -> bool {
        let w = &self.model.weights;
        !w.is_hybrid() && !w.is_mla()
    }

    /// Set (or clear) the CacheBlend recompute ratio applied to RAG `documents`
    /// requests. The serve load path calls this with the autotuned ratio (`tune
    /// --cacheblend` winner when `[tuning].auto_apply_cacheblend`) or the
    /// `[inference].cacheblend_recompute_ratio` fallback. `None`, or a ratio ≤ 0,
    /// disables CacheBlend (requests fall back to a normal prefill). The
    /// selection layer is fixed at 1 (layer-0 K carries no cross-attention
    /// signal — see [`rustllama_models::cacheblend`]).
    pub fn set_cacheblend(&mut self, recompute_ratio: Option<f32>) {
        self.cacheblend = recompute_ratio.filter(|&r| r > 0.0).map(|r| {
            rustllama_models::cacheblend::CacheBlendConfig {
                recompute_ratio: r.min(1.0),
                selection_layer: 1,
            }
        });
    }

    /// The loaded model handle, for in-crate measurement harnesses (e.g. the
    /// CacheBlend coherence sweep). Not part of the public inference API.
    pub(crate) fn model_arc(&self) -> Arc<LlamaModel> {
        self.model.clone()
    }

    /// Pair this engine with a draft model for speculative decoding.
    /// Grammar-free `generate`/`chat` requests then route through
    /// [`Engine::speculate`] with `k` candidates per round. Takes
    /// precedence over n-gram speculation when both are set (a real
    /// draft model's acceptance rate beats prompt-lookup on free
    /// prose). The caller is responsible for tokenizer compatibility
    /// — use [`Self::draft_compatible_with`] before attaching.
    pub fn set_draft_speculative(&mut self, draft: Option<(std::sync::Arc<dyn Engine>, u32)>) {
        self.draft_spec = draft;
    }

    /// The paired draft engine + per-round candidate count, if draft-model
    /// speculation is active on this engine (set via
    /// [`Self::set_draft_speculative`], inherited by forks). The server
    /// routes `/v1/completions` through the ID-level
    /// [`Self::speculate_draft_stream_from_ids`] when this is `Some` and the
    /// request is grammar-free — mirroring how the chat path's
    /// `Engine::generate` consults `draft_spec`.
    pub fn draft_speculative(&self) -> Option<(std::sync::Arc<dyn Engine>, u32)> {
        self.draft_spec.clone()
    }

    /// Cheap tokenizer/vocab compatibility check between this engine
    /// (the target) and a prospective draft engine. Speculative
    /// accept/reject compares token IDS, so the two models must share
    /// a vocabulary exactly: same vocab size, same BOS/EOS ids.
    pub fn draft_compatible_with(&self, draft: &CpuEngine) -> Result<()> {
        let t = &self.model.cfg;
        let d = &draft.model.cfg;
        if t.vocab_size != d.vocab_size {
            return Err(CpuEngineError::Other(format!(
                "draft model vocab_size {} != target vocab_size {} — speculative \
                 decoding requires an identical tokenizer",
                d.vocab_size, t.vocab_size
            )));
        }
        if t.bos_token_id != d.bos_token_id || t.eos_token_id != d.eos_token_id {
            return Err(CpuEngineError::Other(format!(
                "draft model special tokens differ (bos {:?}/{:?}, eos {:?}/{:?}) — \
                 speculative decoding requires an identical tokenizer",
                d.bos_token_id, t.bos_token_id, d.eos_token_id, t.eos_token_id
            )));
        }
        Ok(())
    }

    /// How many distinct prompt-prefix snapshots to keep in the pool.
    /// 0 disables the multi-snapshot pool (the engine still does
    /// single-snapshot LCP against the most recent request).
    pub fn set_prefix_cache_max_snapshots(&mut self, n: usize) {
        self.prefix_cache_max_snapshots = n;
        let mut state = self.state.lock().expect("state lock");
        state.pool.set_max_entries(n);
    }

    pub fn prefix_cache_max_snapshots(&self) -> usize {
        self.prefix_cache_max_snapshots
    }

    /// Drop all snapshots from the prefix pool. Useful for tests and
    /// for explicit cache invalidation after a model reload.
    pub fn clear_prefix_cache(&self) {
        let mut state = self.state.lock().expect("state lock");
        state.pool.clear();
        state.last_ids.clear();
        state.kv_backend.reset();
    }

    /// Number of snapshots currently held in the pool. Test/diagnostic use.
    pub fn prefix_cache_snapshot_count(&self) -> usize {
        let state = self.state.lock().expect("state lock");
        state.pool.len()
    }

    pub fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    pub fn vocab_size(&self) -> usize {
        self.model.cfg.vocab_size
    }

    /// Whether the loaded model carries routed MoE experts (dense-MoE
    /// or hybrid). The MoE-placement tuner sweep refuses non-MoE
    /// models — there is nothing to place.
    pub fn is_moe_model(&self) -> bool {
        self.model.cfg.moe.is_some()
    }

    /// E4 phase 4c: does this engine's model carry Multi-Token
    /// Prediction heads suitable for same-model speculative
    /// decoding? When `true`, the streaming dispatcher can choose
    /// the MTP-driven speculation path (via
    /// [`forward_one_with_mtp_drafts_via_backend`] +
    /// [`crate::speculative::MtpDrafter`] + the existing
    /// `verify_speculation` + `accept_reject` pipeline) instead of
    /// requiring a separate draft `Engine` to be loaded.
    ///
    /// Currently false for paged-KV-backed engines even when the
    /// model declares MTP heads — the paged variant of
    /// `forward_one_with_mtp_logits` is a follow-up.
    pub fn supports_mtp_drafting(&self) -> bool {
        if self.model.cfg.n_mtp_heads == 0 {
            return false;
        }
        let state = self.state.lock().expect("state lock");
        matches!(state.kv_backend, KvBackend::Contiguous(_))
    }

    pub fn n_layers(&self) -> usize {
        self.model.cfg.n_layers
    }

    /// Live KV-cache dtype. Surfaced for `/v1/metrics` so the GUI
    /// can display "currently serving with tq4 KV" without parsing
    /// the model config directly. Reads the construction-time copy —
    /// the KV backend is built exactly once per engine, and going
    /// through the state mutex here parked the metrics poll behind
    /// any in-flight generation (which holds that mutex from prefill
    /// through commit).
    pub fn kv_dtype(&self) -> KvDtype {
        self.kv_dtype_cached
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Attach a K-cache mean-centering bias sidecar (fork `kv_bar`
    /// GGUF format — see `rustllama_models::kv_bias`). `explicit`
    /// overrides auto-discovery of `<model stem>.kvbias.gguf` beside
    /// the model file. Returns `Ok(true)` when a bias was attached.
    ///
    /// Skips quietly (`Ok(false)`) when: `RUSTLLAMA_KV_BIAS=0`, no
    /// sidecar exists (auto mode), or the live KV dtype is not Q4_0
    /// in auto mode (the bias only affects quantized-K quality; it is
    /// exactly softmax-invariant either way). An *explicit* path with
    /// a non-Q4_0 cache, a malformed file, or a calibration-basis
    /// mismatch (recorded `kv_mean_center.k_rot` vs the runtime's
    /// whitening state) is a hard error, mirroring the fork: a
    /// mismatched basis degrades quality instead of improving it.
    pub fn attach_kv_bias(&self, explicit: Option<&std::path::Path>) -> Result<bool> {
        if std::env::var("RUSTLLAMA_KV_BIAS").is_ok_and(|v| v == "0") {
            tracing::info!("kv-bias disabled via RUSTLLAMA_KV_BIAS=0");
            return Ok(false);
        }
        let dtype = self.kv_dtype();
        let path = match explicit {
            Some(p) => {
                if !p.exists() {
                    return Err(CpuEngineError::Other(format!(
                        "kv_bias_path {} does not exist",
                        p.display()
                    )));
                }
                if !matches!(dtype, KvDtype::Q4_0) {
                    return Err(CpuEngineError::Other(format!(
                        "kv_bias_path is set but the KV cache dtype is {dtype:?}; \
                         the K mean-centering bias applies to q4_0 KV only — set \
                         [inference].kv_dtype/k_dtype/v_dtype = \"q4_0\" or unset kv_bias_path"
                    )));
                }
                p.to_path_buf()
            }
            None => {
                if !matches!(dtype, KvDtype::Q4_0) {
                    return Ok(false);
                }
                let auto = self.source_path.with_extension("kvbias.gguf");
                if !auto.exists() {
                    return Ok(false);
                }
                auto
            }
        };

        let cfg = &self.model.cfg;
        let bias = rustllama_models::kv_bias::KvBiasData::load(
            &path,
            cfg.n_layers,
            cfg.head_dim,
            cfg.n_kv_heads,
        )
        .map_err(CpuEngineError::Other)?;

        let whitening = rustllama_models::llama_arch::kv_whitening_active_for(dtype, cfg.head_dim);
        match bias.k_rot {
            Some(recorded) if recorded != whitening => {
                return Err(CpuEngineError::Other(format!(
                    "kv-bias sidecar {} was calibrated with whitening {}, but whitening \
                     is {} for this run — recalibrate with matching settings (or set \
                     RUSTLLAMA_KV_WHITEN consistently in both)",
                    path.display(),
                    if recorded { "active" } else { "inactive" },
                    if whitening { "active" } else { "inactive" },
                )));
            }
            None => {
                tracing::warn!(
                    path = %path.display(),
                    whitening_active = whitening,
                    "kv-bias sidecar does not record its calibration basis \
                     (kv_mean_center.k_rot); a basis mismatch degrades quality"
                );
            }
            _ => {}
        }

        let n_centered = bias.n_centered();
        let mut st = self.state.lock().expect("state lock");
        match &mut st.kv_backend {
            KvBackend::Contiguous(kv) => {
                kv.kv_bias = Some(std::sync::Arc::new(bias));
            }
            _ => {
                return Err(CpuEngineError::Other(
                    "kv-bias requires the contiguous KV backend".to_string(),
                ));
            }
        }
        tracing::info!(
            path = %path.display(),
            layers_centered = n_centered,
            "K-cache mean-centering bias attached"
        );
        Ok(true)
    }

    pub fn config(&self) -> &rustllama_models::llama_config::LlamaConfig {
        &self.model.cfg
    }

    pub fn tokenizer(&self) -> Option<&Tokenizer> {
        self.tokenizer.as_deref()
    }

    /// Borrow the loaded model's parsed config (arch + hyperparams +
    /// optional MoE config). Surfaces details the server's
    /// `/v1/capabilities` endpoint needs without exposing the full
    /// `LlamaModel` weights handle.
    pub fn llama_config(&self) -> &rustllama_models::llama_config::LlamaConfig {
        &self.model.cfg
    }

    /// Build an engine that shares this one's model weights, tokenizer,
    /// and source GGUF mmap (via `Arc`) but carries an **independent**
    /// KV cache, prefix-cache pool, and `last_stats` slot. Two
    /// concurrent requests can each drive their own fork without
    /// touching each other's state — concurrency only costs you one
    /// extra `KvCache` allocation (sized `n_layers × max_ctx × n_kv_heads
    /// × head_dim × dtype_bytes`) per fork.
    ///
    /// Use case: multi-flight serving. The server's [`ServingModel`]
    /// keeps a pool of forks sized to the configured concurrency
    /// (`[server].concurrency`); each admitted request grabs a free
    /// fork from the pool and gives it back when the response stream
    /// finishes. The single shared weights matrix means the per-fork
    /// memory cost is just the KV cache, not the model.
    ///
    /// Configuration carried over from `self`:
    /// - `max_ctx`, `model_id`, `source_path`, `prefix_cache` flag,
    ///   `prefill_chunk_size`, `prefix_cache_max_snapshots`,
    ///   `max_tool_iterations`.
    ///
    /// Configuration that is **reset** on the fork:
    /// - KV cache (fresh allocation, `seq_len = 0`).
    /// - Prefix cache pool (empty — no cross-fork prefix reuse in v1.
    ///   Cross-fork pool sharing is a v1.x knob behind the `[server].
    ///   share_prefix_cache_across_forks` config that becomes available
    ///   once the pool is `Sync`).
    /// - `last_stats` slot (reset to default).
    pub fn fork_for_concurrent_use(&self) -> Self {
        let kv_dtype = {
            let state = self.state.lock().expect("state lock");
            state.kv_backend.kv_dtype()
        };
        // Rebuild a backend matching the parent's layout. Paged
        // forks get their own page pool sized to `max_ctx` — no
        // cross-fork page sharing in V1 (multi-slot pooling lands
        // alongside the scheduler-driven generate loop in 3.7).
        let kv_backend = KvBackend::from_inference_config_with_page_size_and_keep(
            &self.kv_layout,
            &self.model.cfg,
            self.max_ctx as u32,
            kv_dtype,
            self.kv_page_size,
            hybrid_kv_keep_mask(&self.model).as_deref(),
        )
        .expect("parent's layout already validated at engine load");
        // Infallible fork of an ALREADY-loaded model: its DeltaNet
        // geometry was validated when the parent loaded, so a rebuild
        // error here is impossible in practice. Degrade to no cache +
        // log rather than reintroduce a panic in this infallible path.
        let delta_net_cache = build_delta_net_cache(&self.model).unwrap_or_else(|e| {
            tracing::error!(
                error = %e,
                "fork_for_concurrent_use: DeltaNet cache rebuild failed for an \
                 already-validated model"
            );
            None
        });
        let kv_dtype_cached = kv_backend.kv_dtype();
        Self {
            model: self.model.clone(),
            tokenizer: self.tokenizer.clone(),
            kv_dtype_cached,
            // Forks report their own KV occupancy.
            metrics_cache: Arc::new(MetricsCacheAtomics::default()),
            state: Arc::new(Mutex::new(EngineState {
                kv_backend,
                last_ids: Vec::new(),
                pool: PrefixCachePool::new(self.prefix_cache_max_snapshots),
                delta_net_cache,
            })),
            // Forks share the same SYCL worker. Their generations
            // serialize through it, which matches our single-flight
            // server policy. Separate workers per fork would duplicate
            // the USM weight cache (one full upload per fork) without
            // any concurrency benefit on a shared-memory iGPU.
            sycl_worker: Arc::clone(&self.sycl_worker),
            max_ctx: self.max_ctx,
            model_id: self.model_id.clone(),
            source_path: self.source_path.clone(),
            prefix_cache: self.prefix_cache,
            prefill_chunk_size: self.prefill_chunk_size,
            prefix_cache_max_snapshots: self.prefix_cache_max_snapshots,
            last_stats: Arc::new(Mutex::new(RequestStats::default())),
            // Forks track their own EMA: their first generation will
            // seed the slot from scratch. Sharing the parent's EMA
            // would mix throughput from different KV-cache layouts.
            ema_tok_s_bits: Arc::new(std::sync::atomic::AtomicU64::new(f64::NAN.to_bits())),
            // Same reasoning for cumulative counters: forks track
            // their own — a fork's metrics endpoint surfaces its
            // own request history, not the parent's.
            cumulative_stats: Arc::new(crate::CumulativeStats::default()),
            max_tool_iterations: self.max_tool_iterations,
            flash_attention: self.flash_attention,
            kv_layout: self.kv_layout.clone(),
            kv_page_size: self.kv_page_size,
            n_gpu_layers: self.n_gpu_layers,
            cpu_force_patterns: self.cpu_force_patterns.clone(),
            // Forks share the parent's loaded vision tower — same
            // reasoning as the model Arc: vision inference is stateless
            // beyond its loaded tensors so concurrent forks calling
            // `forward_image_bytes` don't collide.
            vision: self.vision.clone(),
            image_token_id: self.image_token_id,
            placeholder_mode: self.placeholder_mode,
            // Forks inherit the parent's speculative setting (Copy).
            ngram_spec: self.ngram_spec,
            lookahead_spec: self.lookahead_spec,
            draft_spec: self.draft_spec.clone(),
            // MTP self-speculation is a plain Copy bool; forks inherit it.
            mtp_spec: self.mtp_spec,
            tree_spec: self.tree_spec.clone(),
            // P10b — forks inherit the parent's EAGLE head (Arc-shared).
            eagle_head: self.eagle_head.clone(),
            // F9-tail — forks inherit the parent's canonical-Mamba model (Arc-shared).
            mamba: self.mamba.clone(),
            image_wrapper: self.image_wrapper.clone(),
            vision_feature_memo: Arc::clone(&self.vision_feature_memo),
            // Forks share the parent's locked ranges: they use the same
            // `Arc<LlamaModel>`, so the pages are already pinned. Cloning
            // the Arc keeps the locks alive until the last fork drops; a
            // fork doesn't re-lock (would double-count VirtualLock).
            lock_registry: self.lock_registry.clone(),
            // One sidecar flusher per model: the loading engine owns
            // it. The accel-side counters are process-global, so a
            // fork's routing still lands in the parent's flushes.
            expert_usage: None,
            // One .rlkv writer per model, same reasoning.
            kv_persist_owner: false,
            // Forks inherit the parent's tuned CacheBlend ratio (plain Copy).
            cacheblend: self.cacheblend,
            checkpoint_clock: Mutex::new(crate::kv_persist::CheckpointClock::new(
                crate::kv_persist::kv_persist_interval(),
                std::time::Instant::now(),
            )),
            // Forks SHARE the resident LoRA registry (an adapter attached at
            // runtime is visible to every concurrent fork).
            lora: Arc::clone(&self.lora),
        }
    }

    // ----- P11 Phase 2/3: resident multi-LoRA adapters -----

    /// Attach (load, UNMERGED) a LoRA adapter GGUF into the resident registry
    /// under `name`. The adapter's low-rank factors are parsed and held as
    /// deltas; the base weights are untouched. A later request selects it via
    /// [`SamplingParams::lora_adapter`] = `Some(name)`. Re-attaching a name
    /// replaces it. Returns the number of adapted base tensors.
    ///
    /// This is the engine twin of the Phase-1 merge-at-load path
    /// ([`Self::load_with_options_layout_page_size_and_lora`]): there the
    /// adapter is folded once into the weights; here it stays separate so N
    /// adapters can be swapped per request without reloading the model.
    pub fn attach_lora_adapter(&self, name: &str, path: &std::path::Path) -> Result<usize> {
        let g = Gguf::open(path)?;
        let adapter = rustllama_models::lora::LoraAdapter::from_gguf(&g, name)
            .map_err(|e| CpuEngineError::Other(format!("LoRA adapter load failed: {e}")))?;
        let n = adapter.len();
        let mut reg = self
            .lora
            .write()
            .map_err(|_| CpuEngineError::Other("LoRA registry lock poisoned".into()))?;
        reg.insert(adapter);
        tracing::info!(
            adapter = name,
            adapted_tensors = n,
            "LoRA adapter attached (P11 resident)"
        );
        Ok(n)
    }

    /// Names of the adapters currently resident (registration order is not
    /// guaranteed — this is a set).
    pub fn lora_adapter_names(&self) -> Vec<String> {
        self.lora
            .read()
            .map(|r| r.names().map(|s| s.to_string()).collect())
            .unwrap_or_default()
    }

    /// `true` iff an adapter named `name` is resident.
    pub fn has_lora_adapter(&self, name: &str) -> bool {
        self.lora.read().map(|r| r.contains(name)).unwrap_or(false)
    }

    /// Resolve the per-request adapter selection in `sampling` against the
    /// resident registry. `None` selection ⇒ `Ok(None)` (base, byte-identical).
    /// A selected name that is NOT resident is an error — generation fails
    /// rather than silently serving the base.
    fn lora_adapter_for(
        &self,
        sampling: &SamplingParams,
    ) -> Result<Option<std::sync::Arc<rustllama_models::lora::LoraAdapter>>> {
        let Some(name) = sampling.lora_adapter.as_deref() else {
            return Ok(None);
        };
        if name.is_empty() {
            return Ok(None);
        }
        let reg = self
            .lora
            .read()
            .map_err(|_| CpuEngineError::Other("LoRA registry lock poisoned".into()))?;
        reg.get(name).map(Some).ok_or_else(|| {
            CpuEngineError::Other(format!("unknown LoRA adapter `{name}` (not resident)"))
        })
    }

    /// Toggle FlashAttention-decode for the F32 KV path. Propagates
    /// via per-thread TLS at the start of each generate call.
    /// Default `true`. Set from `[inference].flash_attention` by the
    /// CLI/server wiring; the GUI Settings page exposes the toggle.
    pub fn set_flash_attention(&mut self, enabled: bool) {
        self.flash_attention = enabled;
    }

    /// Set the hybrid-placement layer cutoff. Layers
    /// `0..n_gpu_layers` go through SYCL/USM; layers at or beyond
    /// the cutoff stay on the CPU dispatch path. Pass `u32::MAX`
    /// for "all GPU" (the default), `0` for "all CPU."
    ///
    /// Pushed into the per-thread TLS slot at the start of each
    /// generate call — values set here apply to every request the
    /// engine subsequently runs until set again.
    pub fn set_n_gpu_layers(&mut self, n: u32) {
        self.n_gpu_layers = n;
    }

    pub fn n_gpu_layers(&self) -> u32 {
        self.n_gpu_layers
    }

    /// Run the auto-placement planner against the currently loaded
    /// model + the active CPU-force patterns, and set
    /// `self.n_gpu_layers` to the planner's choice.
    ///
    /// The planner queries the SYCL device's reported global memory
    /// and counts per-layer weight bytes (excluding any tensor whose
    /// name matches a `cpu_force_patterns` entry — those stay on
    /// CPU and don't count against the budget). It picks the
    /// largest cutoff such that cumulative GPU residency fits.
    ///
    /// Returns the full decision struct for logging / diagnostics.
    /// On SYCL-unavailable hosts the decision falls back to all-CPU
    /// (`n_gpu_layers = 0`) — same conservative default as a fresh
    /// `set_n_gpu_layers(0)`.
    ///
    /// Convenient wiring:
    /// ```ignore
    /// let mut engine = CpuEngine::load_with_options(path, ctx, true, kv)?;
    /// engine.set_cpu_force_patterns(cfg.placement.cpu_overrides());
    /// let d = engine.auto_place_layers(&Default::default());
    /// tracing::info!(
    ///     n_gpu_layers = d.n_gpu_layers,
    ///     total_layers = d.total_layers,
    ///     budget_bytes = d.effective_budget_bytes,
    ///     gpu_bytes = d.gpu_resident_bytes,
    ///     reason = %d.reason,
    ///     "auto-placement",
    /// );
    /// ```
    pub fn auto_place_layers(
        &mut self,
        opts: &crate::placement_auto::AutoPlacementOpts,
    ) -> crate::placement_auto::AutoPlacementDecision {
        // Clear any stale multi-GPU heat plan: the VRAM-fit path is the
        // non-heat loader (and the tune-measurement loader), so no per-tensor
        // plan should be installed for it. Safe no-op when none is installed
        // (byte-identical to before this line existed).
        rustllama_models::accel::set_multi_gpu_plan(None);
        let decision = self.plan_placement(opts);
        self.n_gpu_layers = decision.n_gpu_layers;
        decision
    }

    /// Measured-perf HEAT placement (Phase 5). Reads the per-device decode
    /// tok/s the autotune sweep recorded (`TuningResult.per_device_perf`),
    /// ranks the ENABLED tiers by measured throughput, heat-ranks the model's
    /// weights, and installs a per-tensor device-assignment plan
    /// ([`rustllama_models::accel::MultiGpuPlan`]) that SUBSUMES the flat
    /// `n_gpu_layers` cutoff AND the MoE experts→CPU split with one heat-ranked
    /// split.
    ///
    /// Two code paths, no third heuristic:
    ///   - **measured perf present** ⇒ compute + apply the heat plan here.
    ///   - **measured perf absent** ⇒ only inside the tune's own measurement
    ///     loads (every SERVED model is tuned first); falls back to the
    ///     VRAM-fit [`Self::auto_place_layers`].
    ///
    /// On a hard device-tier violation (`cpu_enabled=false` / enforced
    /// `vram_only` with no GPU fit) it returns a decision carrying
    /// `placement_error` and installs NO plan, so the caller refuses the load.
    ///
    /// NOTE: the single-GPU + CPU per-tensor heat split flows entirely through
    /// the engine's EXISTING residency gate
    /// (`accel::tensor_forced_to_cpu`); only cross-GPU routing (>1 usable GPU)
    /// exercises Phase 4's per-device caches, which is unvalidated on a
    /// single-GPU host.
    pub fn auto_place_heat(
        &mut self,
        opts: &crate::placement_auto::AutoPlacementOpts,
    ) -> crate::placement_auto::AutoPlacementDecision {
        // Merge engine cpu_force_patterns into the opts (same as plan_placement
        // / auto_place_layers), so a user's explicit CPU pins are honored.
        let mut effective = opts.clone();
        for p in &self.cpu_force_patterns {
            if !effective.cpu_force_patterns.iter().any(|q| q == p) {
                effective.cpu_force_patterns.push(p.clone());
            }
        }
        // Load the measured per-device perf from the tuner cache.
        let per_device_perf: std::collections::HashMap<String, f32> =
            rustllama_tuner::default_cache_dir()
                .and_then(|dir| {
                    let key = rustllama_tuner::system_fingerprint();
                    rustllama_tuner::load_cache(&dir, &key).ok().flatten()
                })
                .map(|t| t.per_device_perf)
                .unwrap_or_default();

        match crate::multi_gpu::plan_heat_placement(
            &self.model.weights,
            &per_device_perf,
            &effective,
        ) {
            Some(outcome) => {
                if outcome.decision.placement_error.is_some() {
                    // Hard device-tier violation: install nothing; the caller
                    // refuses and surfaces the error. Leave n_gpu_layers as-is.
                    rustllama_models::accel::set_multi_gpu_plan(None);
                    return outcome.decision;
                }
                self.n_gpu_layers = outcome.applied_n_gpu_layers;
                let plan = std::sync::Arc::new(outcome.plan.to_multi_gpu_plan());
                rustllama_models::accel::set_multi_gpu_plan(Some(plan));
                outcome.decision
            }
            // Measured perf absent. With mandatory first-load autotune this
            // should never happen at serve — so DON'T silently degrade to an
            // all-CPU/VRAM-fit placement. Surface a hard placement_error so the
            // serve path refuses and the operator re-runs the (mandatory) tune.
            // (The tune's own measurement loads use explicit `set_n_gpu_layers`,
            // not this path, so nothing in the sweep depends on a fallback here.)
            None => {
                let mut d = crate::placement_auto::auto_n_gpu_layers(&self.model.weights, opts);
                d.placement_error = Some(
                    "no measured per-device performance in the tuner cache — this \
                     model has not been autotuned. First-load autotune is mandatory; \
                     re-run `rustllama tune` for this model (or delete the tuner cache \
                     and reload to force a fresh sweep)."
                        .to_string(),
                );
                d
            }
        }
    }

    /// Compute a placement decision WITHOUT mutating `n_gpu_layers`. Same
    /// merge-of-`cpu_force_patterns` behavior as [`Self::auto_place_layers`],
    /// but pure — used to VALIDATE device-tier constraints
    /// (`cpu_enabled = false` / `vram_only`) before committing a placement,
    /// so a guardrail check doesn't clobber a cached/static `n_gpu_layers`.
    pub fn plan_placement(
        &self,
        opts: &crate::placement_auto::AutoPlacementOpts,
    ) -> crate::placement_auto::AutoPlacementDecision {
        // Merge engine's current cpu_force_patterns into the options
        // so callers don't have to repeat them. Caller-provided
        // patterns in `opts` win on duplicates.
        let mut effective = opts.clone();
        for p in &self.cpu_force_patterns {
            if !effective.cpu_force_patterns.iter().any(|q| q == p) {
                effective.cpu_force_patterns.push(p.clone());
            }
        }
        crate::placement_auto::auto_n_gpu_layers(&self.model.weights, &effective)
    }

    /// Set the list of "force this tensor to CPU" patterns. Each
    /// pattern is a substring matched against tensor names —
    /// `"ffn"` pins every weight tensor containing "ffn" (gate,
    /// up, down) to the CPU path even when the layer would
    /// otherwise route to GPU. Empty list disables overrides.
    ///
    /// Pushed to the per-thread TLS slot at the start of each
    /// generate call. CLI populates from
    /// `[inference].placement.overrides` (entries with
    /// `device = "cpu"`).
    pub fn set_cpu_force_patterns(&mut self, patterns: Vec<String>) {
        self.cpu_force_patterns = patterns;
    }

    pub fn cpu_force_patterns(&self) -> &[String] {
        &self.cpu_force_patterns
    }

    pub fn flash_attention_enabled(&self) -> bool {
        self.flash_attention
    }

    /// Run the target's forward pass over `prompt_ids` followed by each
    /// `candidate`, and return one softmaxed probability distribution per
    /// candidate-slot plus one **bonus** distribution. The returned vector
    /// has length `candidates.len() + 1`: index `i ∈ [0, K)` is the target's
    /// distribution at the position where `candidates[i]` would commit, and
    /// index `K` is the distribution at the position one past the last
    /// candidate (the "bonus" slot consumed when [`crate::speculative::
    /// accept_reject`] accepts every draft).
    ///
    /// Probabilities are **raw-softmax** of the model's logits — no
    /// temperature, top-k, or top-p applied. The spec-decode math wants p
    /// and q on the same distribution, and the simplest way to guarantee
    /// that without coordinating sampler knobs across two engines is to
    /// keep both on raw-softmax. (A "speculation temperature" knob shared
    /// by both sides is a v1.x follow-up.)
    ///
    /// State invariant: the engine's KV cache + `last_ids` are restored to
    /// their pre-call values before this method returns. Verification is a
    /// pure read — callers can chain a subsequent `generate_token_ids` /
    /// `Engine::generate` without worrying about state corruption from the
    /// speculation forwards.
    ///
    /// The prefix-cache LCP path is respected on the way in (prompt-prefix
    /// reuse still happens) but the speculation forwards themselves are
    /// **not** committed to the cache — the next call sees the same
    /// `last_ids` it would have without speculation.
    pub fn verify_speculation(
        &self,
        prompt_ids: &[i32],
        candidates: &[u32],
    ) -> Result<Vec<Vec<f32>>> {
        verify_speculation_inner(
            self.model.clone(),
            self.state.clone(),
            self.prefix_cache,
            self.prefill_chunk_size,
            self.max_ctx,
            prompt_ids,
            candidates,
        )
    }

    /// Teacher-force each `option` as a continuation of `context_ids` and
    /// return the summed conditional logprob `log P(option | context)` per
    /// option. Pure read of the model: it snapshots and restores the KV
    /// cache + DeltaNet (hybrid) state exactly like [`verify_speculation`],
    /// so the live serving state is never mutated. Powers the
    /// typed-decision endpoints (Choice / Score / Boolean) — softmax over the
    /// returned per-option scores gives the decision probabilities; the
    /// caller normalizes by option length (mean) as it sees fit.
    pub fn score_continuations(
        &self,
        context_ids: &[i32],
        options: &[Vec<u32>],
    ) -> Result<Vec<f32>> {
        score_continuations_inner(
            self.model.clone(),
            self.state.clone(),
            self.prefill_chunk_size,
            self.max_ctx,
            context_ids,
            options,
        )
    }

    /// Teacher-force the full token sequence `ids` and return
    /// `(total_logprob, per_token_logprob)` where `per_token_logprob[j] =
    /// log P(ids[j+1] | ids[..=j])` for `j in 0..ids.len()-1`. The first
    /// token has no predecessor to condition on, so it is not scored (the
    /// returned vector has length `ids.len() - 1`); `total_logprob` is the
    /// sum. Like [`score_continuations`] this is a pure read: it snapshots
    /// and restores the KV + DeltaNet state, so the live serving state is
    /// never mutated. Powers the sequence-likelihood / perplexity endpoint
    /// (`/v1/score`) and best-of-N candidate scoring. Returns `(0.0, [])`
    /// for sequences shorter than two tokens.
    pub fn sequence_logprob(&self, ids: &[i32]) -> Result<(f32, Vec<f32>)> {
        sequence_logprob_inner(
            self.model.clone(),
            self.state.clone(),
            self.prefill_chunk_size,
            self.max_ctx,
            ids,
        )
    }

    /// Build the [`GrammarMask`] for this sampling request, if any.
    /// Returns `None` when no grammar was requested. Returns `None` and
    /// logs a warning if a grammar was requested but the engine has no
    /// tokenizer (so we can't pre-decode the vocab).
    fn build_grammar(&self, sampling: &SamplingParams) -> Option<GrammarMask> {
        let kind = sampling.grammar.as_ref()?;
        let tokenizer = match self.tokenizer.as_deref() {
            Some(t) => t,
            None => {
                tracing::warn!(
                    "grammar requested ({kind:?}) but engine has no tokenizer; ignoring"
                );
                return None;
            }
        };
        let vocab = self.model.cfg.vocab_size;
        let eos = self.model.cfg.eos_token_id;
        // Pre-decoding 150k tokens takes ~30 ms on a modern CPU. We do
        // it once per request, not per token, so the amortized cost is
        // negligible compared to the forward pass.
        let decode = |id: u32| {
            tokenizer
                .decode_single(id, true)
                .unwrap_or_default()
                .into_bytes()
        };
        match kind {
            GrammarKind::Json => Some(GrammarMask::new_json(vocab, eos, decode)),
            GrammarKind::JsonSchema { schema } => Some(GrammarMask::new_json_schema(
                schema.clone(),
                vocab,
                eos,
                decode,
            )),
            GrammarKind::ToolCallStream {
                schemas_by_name,
                min_completed,
            } => Some(GrammarMask::new_tool_call_stream(
                schemas_by_name.clone(),
                self.max_tool_iterations,
                *min_completed,
                vocab,
                eos,
                decode,
            )),
            GrammarKind::Code { language } => {
                // `language` drives the parser's comment-recognition
                // style. C-family ("rust", "javascript", "go", …) →
                // `//` + `/* … */`. Hash-style ("python", "ruby",
                // "bash", …) → `#`. Unknown → no comment recognition
                // (v1.0 behavior). Bracket-balance + string-aware
                // tracking is universal across languages.
                Some(GrammarMask::new_code_for_language(
                    language, vocab, eos, decode,
                ))
            }
            GrammarKind::Regex { pattern } => {
                // Anchored byte-DFA. A malformed pattern reaches here
                // only if the server-side validation slipped — fall
                // back to plain JSON validity rather than a hard panic
                // so the model still produces something parseable.
                // The validation at the request boundary is the
                // authoritative reject path; this is defense-in-depth.
                match GrammarMask::new_regex(pattern, vocab, eos, decode) {
                    Ok(m) => Some(m),
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            pattern,
                            "invalid regex pattern in GrammarKind::Regex — falling back to plain JSON grammar"
                        );
                        None
                    }
                }
            }
            GrammarKind::Gbnf { grammar } => {
                // Compile the GBNF text to the pushdown-automaton mask. A
                // malformed grammar should have been rejected at the request
                // boundary; degrade to "no grammar" defensively rather than
                // panic on a worker thread.
                match GrammarMask::new_gbnf(grammar, vocab, eos, decode) {
                    Ok(m) => Some(m),
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "invalid GBNF grammar in GrammarKind::Gbnf — dropping grammar"
                        );
                        None
                    }
                }
            }
        }
    }

    /// Count the tokens the chat-template + tokenizer would produce for a
    /// given message list. Used by the server to populate the OpenAI
    /// `usage.prompt_tokens` field without forcing the engine to surface
    /// the rendered prompt through its streaming API.
    ///
    /// Re-runs the same render+encode the engine will do internally, so
    /// adds ~5–10 ms per request — negligible compared to model compute.
    pub fn count_chat_prompt(&self, msgs: &[ChatMessage]) -> Result<u32> {
        let tokenizer = self
            .tokenizer
            .as_deref()
            .ok_or(CpuEngineError::NoTokenizer("count_chat_prompt"))?;
        let tok_msgs: Vec<TokChat<'_>> = msgs
            .iter()
            .map(|m| TokChat {
                role: &m.role,
                content: &m.content,
            })
            .collect();
        let prompt = tokenizer.render_chat(&tok_msgs, true)?;
        let ids = tokenizer.encode(&prompt, tokenizer.add_bos_token())?;
        Ok(ids.len() as u32)
    }

    /// Render chat messages through the model's chat template and tokenize to
    /// the exact prompt token ids `Engine::chat` would decode from — the same
    /// path as [`Self::count_chat_prompt`], but returning the ids. Used by the
    /// server's beam-search chat path, which needs token-level access the
    /// text-streaming `Engine::chat` API doesn't expose.
    pub fn chat_prompt_ids(&self, msgs: &[ChatMessage]) -> Result<Vec<i32>> {
        let tokenizer = self
            .tokenizer
            .as_deref()
            .ok_or(CpuEngineError::NoTokenizer("chat_prompt_ids"))?;
        let tok_msgs: Vec<TokChat<'_>> = msgs
            .iter()
            .map(|m| TokChat {
                role: &m.role,
                content: &m.content,
            })
            .collect();
        let prompt = tokenizer.render_chat(&tok_msgs, true)?;
        let ids = tokenizer.encode(&prompt, tokenizer.add_bos_token())?;
        Ok(ids.iter().map(|&t| t as i32).collect())
    }

    /// Tokenize a free-text prompt with BOS (matches what `Engine::generate`
    /// does internally). Cheap helper for usage accounting in the
    /// `/v1/completions` non-FIM path.
    pub fn count_text_prompt(&self, prompt: &str) -> Result<u32> {
        let tokenizer = self
            .tokenizer
            .as_deref()
            .ok_or(CpuEngineError::NoTokenizer("count_text_prompt"))?;
        let ids = tokenizer.encode(prompt, tokenizer.add_bos_token())?;
        Ok(ids.len() as u32)
    }

    /// Reset the KV cache and prefix-reuse state. Call between independent
    /// generations whose prompts intentionally do NOT share a prefix.
    pub fn reset_state(&self) {
        self.state.lock().expect("state lock").reset();
    }

    /// Run prefill on `prompt_ids` then sample `n_new` continuation tokens.
    /// Returns just the newly generated tokens (without the prompt).
    /// Helper for the synchronous `generate_token_ids*` entries.
    /// Pre-builds the per-thread USM attention context so the packed-
    /// matvec USM hook can fire. Mirrors the eager setup that
    /// `drive_generation` (the streaming path) does at line 3083.
    /// Without this, synchronous callers (the `bench` subcommand,
    /// the chat REPL, any direct test harness) install the SYCL
    /// stream but leave `USM_ATTN` un-initialized — every matvec
    /// then short-circuits with "USM_ATTN slot is None
    /// (prepare_usm_context failed)" and the model runs on CPU.
    /// `false` is benign: the matvec falls through to its CPU path.
    fn prepare_usm_for_sync_path(&self) {
        if !rustllama_models::accel::has_sycl_stream() {
            return;
        }
        let cfg = &self.model.cfg;
        let head_dim = if cfg.head_dim > 0 {
            cfg.head_dim
        } else {
            cfg.d_model / cfg.n_heads.max(1)
        };
        let _ = rustllama_models::accel::prepare_usm_context(
            cfg.n_layers as u32,
            cfg.n_heads as u32,
            cfg.n_kv_heads as u32,
            head_dim as u32,
            self.max_ctx as u32,
        );
    }

    /// F9-tail — canonical Mamba/SSM selective-scan generation. For a Mamba
    /// model the engine's `model` field is a transformer SHELL; the real
    /// forward is the tested [`rustllama_models::mamba_arch::MambaModel`]
    /// `forward_sequence` / `forward_one` selective scan over a per-request
    /// `MambaCache` (the recurrent "KV"). The shared `Sampler` + grammar glue
    /// is identical to the transformer decode loop, so greedy decode is
    /// deterministic + in-vocab. A fresh cache per call makes repeated greedy
    /// runs bit-identical (cross-request recurrent reuse / prefix cache is a
    /// follow-up — correctness first).
    fn generate_token_ids_mamba(
        &self,
        mm: &rustllama_models::mamba_arch::MambaModel,
        prompt_ids: &[i32],
        n_new: u32,
        sampling: &SamplingParams,
    ) -> Result<Vec<u32>> {
        if prompt_ids.is_empty() {
            return Ok(Vec::new());
        }
        let total_required = prompt_ids.len() + n_new as usize;
        if total_required > self.max_ctx {
            return Err(CpuEngineError::PromptTooLong {
                prompt_len: prompt_ids.len(),
                max_ctx: self.max_ctx,
            });
        }
        let vocab = mm.cfg.vocab_size;
        let mut logits = vec![0f32; vocab];
        let mut cache = rustllama_models::mamba_arch::MambaCache::new(&mm.cfg);
        let mut history: Vec<u32> = prompt_ids.iter().map(|&t| t as u32).collect();
        // Prefill: fold the whole prompt through the recurrence; the last
        // token leaves the next-token logits in `logits`.
        mm.forward_sequence(prompt_ids, &mut cache, &mut logits);

        let mut sampler = Sampler::new(sampling.clone());
        let mut grammar = self.build_grammar(sampling);
        let eos = self.model.cfg.eos_token_id;
        let mut out = Vec::with_capacity(n_new as usize);
        for _ in 0..n_new {
            let next_tok = sampler.sample_with_grammar(&mut logits, &history, grammar.as_ref());
            if let Some(g) = grammar.as_mut() {
                g.advance(next_tok);
            }
            out.push(next_tok);
            history.push(next_tok);
            if Some(next_tok) == eos {
                break;
            }
            mm.forward_one(next_tok as i32, &mut cache, &mut logits);
        }
        Ok(out)
    }

    pub fn generate_token_ids(
        &self,
        prompt_ids: &[i32],
        n_new: u32,
        sampling: &SamplingParams,
    ) -> Result<Vec<u32>> {
        // P11: scope the per-request LoRA selection for the whole call (this
        // path runs the forwards INLINE on the calling thread, so a scope here
        // covers prefill + every decode step). `None` ⇒ base, byte-identical.
        let _lora_scope =
            rustllama_models::lora::scope_active_lora(self.lora_adapter_for(sampling)?);
        // F9-tail — canonical Mamba/SSM models decode through the selective-scan
        // forward, not the transformer backend (the `model` field is a shell).
        if let Some(mm) = self.mamba.clone() {
            return self.generate_token_ids_mamba(&mm, prompt_ids, n_new, sampling);
        }
        let _sycl_guard = install_sycl_dispatch_if_requested();
        rustllama_models::accel::set_flash_attention(self.flash_attention);
        rustllama_models::accel::set_n_gpu_layers(self.n_gpu_layers);
        rustllama_models::accel::set_cpu_force_patterns(self.cpu_force_patterns.clone());
        self.prepare_usm_for_sync_path();
        let mut state = self.state.lock().expect("state lock");

        let total_required = prompt_ids.len() + n_new as usize;
        if total_required > self.max_ctx {
            return Err(CpuEngineError::PromptTooLong {
                prompt_len: prompt_ids.len(),
                max_ctx: self.max_ctx,
            });
        }

        let vocab = self.model.cfg.vocab_size;
        let mut logits = vec![0f32; vocab];
        let mut history: Vec<u32> = prompt_ids.iter().map(|&t| t as u32).collect();

        if prompt_ids.is_empty() {
            return Ok(Vec::new());
        }
        let prompt_u32: Vec<u32> = prompt_ids.iter().map(|&t| t as u32).collect();
        // Prefix-cache LCP across both the live state and the snapshot
        // pool; restores pool data into the live cache if a different
        // conversation thread shared a longer prefix than the current one.
        let prompt_max = prompt_ids.len().saturating_sub(1);
        // P11: the prompt-prefix cache is keyed on token content, NOT on the
        // adapter, so an adapted request must not reuse/populate it with KV from
        // a different selection. Disable prefix reuse when an adapter is active;
        // `None` ⇒ `&& true` ⇒ byte-identical to the base path.
        let prefix_cache = self.prefix_cache && sampling.lora_adapter.is_none();
        let effective = state.prepare_prefix_reuse(
            &prompt_u32,
            prefix_cache,
            PREFIX_REUSE_MIN_TOKENS,
            prompt_max,
        );

        let mut chrome = ChromeTracer::from_env();
        let prefill_start = std::time::Instant::now();
        let (prefill, last) = prompt_ids.split_at(prompt_ids.len() - 1);
        let _ = run_chunked_prefill(
            &self.model,
            &mut state,
            &prompt_u32,
            prefill,
            effective,
            self.prefill_chunk_size,
            &mut logits,
            &mut chrome,
            || false, // non-streaming path can't be cancelled
        );
        chrome.span("prefill_total", prefill_start);
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        let mut sampler = Sampler::new(sampling.clone());
        let mut grammar = self.build_grammar(sampling);
        let mut next_input = last[0];
        let mut next_pos = prefill.len() as u32;
        let mut out = Vec::with_capacity(n_new as usize);
        let mut decode_ms = 0.0f64;

        for step in 0..n_new {
            let t_fwd = std::time::Instant::now();
            {
                let EngineState {
                    kv_backend,
                    delta_net_cache,
                    ..
                } = &mut *state;
                forward_one_via_backend_hybrid(
                    &self.model,
                    kv_backend,
                    delta_net_cache.as_mut(),
                    next_input,
                    next_pos,
                    &mut logits,
                );
            }
            chrome.span(&format!("decode_fwd_{step}"), t_fwd);
            decode_ms += t_fwd.elapsed().as_secs_f64() * 1000.0;

            let t_sample = std::time::Instant::now();
            let next_tok = sampler.sample_with_grammar(&mut logits, &history, grammar.as_ref());
            chrome.span(&format!("sample_{step}"), t_sample);

            if let Some(g) = grammar.as_mut() {
                g.advance(next_tok);
            }
            out.push(next_tok);
            history.push(next_tok);

            if let Some(eos) = self.model.cfg.eos_token_id {
                if next_tok == eos {
                    break;
                }
            }

            next_input = next_tok as i32;
            next_pos += 1;
            if (next_pos as usize) >= self.max_ctx {
                // P1: try a context shift (generate past ctx_size) before
                // giving up. `false` ⇒ disabled/inapplicable ⇒ stop as before.
                if !try_context_shift(
                    &self.model,
                    &mut state.kv_backend,
                    self.max_ctx,
                    &mut next_pos,
                ) {
                    break;
                }
            }
        }
        chrome.flush();

        let tool_call_limit_hit = grammar
            .as_ref()
            .map(|g| g.tool_call_limit_blocked())
            .unwrap_or(false);
        self.commit_request_stats(RequestStats {
            prefill_ms,
            decode_ms,
            tokens_prefilled: (prefill.len().saturating_sub(effective)) as u32,
            cache_hit_tokens: effective as u32,
            tokens_generated: out.len() as u32,
            tool_call_limit_hit,
        });

        // Snapshot the full sequence (prompt + generated) so the next call
        // can reuse this run's K/V entries via LCP. If we hit a hard error
        // mid-stream, leave last_ids untouched — the partial cache stays
        // bound to the previous prompt and a non-matching next prompt will
        // self-invalidate via LCP=0.
        let mut snapshot = prompt_u32;
        snapshot.extend(out.iter().copied());
        state.commit_snapshot(snapshot);
        // F12: periodic warm-restart checkpoint (inert unless
        // `kv_persist_interval_secs > 0` and the clock has elapsed). The pool
        // now carries this request's full sequence, so the checkpoint preserves
        // the live session. Uses the held `state` — never re-locks.
        self.maybe_checkpoint_kv_persist(&state);

        Ok(out)
    }

    /// P12b — beam search. Maintains `num_beams` partial hypotheses, expands
    /// each by its top candidates per step, keeps the global top `num_beams`
    /// by cumulative log-prob, and returns the top `k` finished beams ranked by
    /// the length-penalty-adjusted score.
    ///
    /// Beam search is **deterministic** (no sampling): the same prompt yields
    /// the same beams. `num_beams == 1` is byte-identical to greedy
    /// [`Self::generate_token_ids`] with a temperature-0 (argmax) sampler —
    /// the server routes `num_beams == 1` through the normal path, but this
    /// driver reproduces greedy so the parity anchor can prove the machinery
    /// doesn't perturb the degenerate case.
    ///
    /// Per-beam KV is forked with the existing host-side
    /// [`KvCache::snapshot_prefix`] / [`KvCache::restore_prefix`] (the same
    /// primitive the prompt prefix cache uses): each beam carries a snapshot of
    /// its K/V, which the driver restores before extending it. This works
    /// regardless of USM backing. Beam search requires the **contiguous** KV
    /// backend on a plain transformer — a hybrid DeltaNet recurrence, an MLA
    /// latent cache, or a sliding-window ring layer aren't captured by the
    /// snapshot, so those are refused with a clear error (the paged-engine beam
    /// path is a noted follow-up).
    pub fn generate_beam_token_ids(
        &self,
        prompt_ids: &[i32],
        n_new: u32,
        beam: &crate::beam::BeamParams,
    ) -> Result<Vec<crate::beam::BeamHypothesis>> {
        use crate::beam::{build_sorted_candidates, select_step, BeamPool};

        let num_beams = beam.num_beams.max(1);
        let _sycl_guard = install_sycl_dispatch_if_requested();
        rustllama_models::accel::set_flash_attention(self.flash_attention);
        rustllama_models::accel::set_n_gpu_layers(self.n_gpu_layers);
        rustllama_models::accel::set_cpu_force_patterns(self.cpu_force_patterns.clone());
        self.prepare_usm_for_sync_path();

        if prompt_ids.is_empty() {
            return Ok(Vec::new());
        }
        let total_required = prompt_ids.len() + n_new as usize;
        if total_required > self.max_ctx {
            return Err(CpuEngineError::PromptTooLong {
                prompt_len: prompt_ids.len(),
                max_ctx: self.max_ctx,
            });
        }

        let mut state = self.state.lock().expect("state lock");

        // Beam search needs the contiguous KV backend (host-side
        // snapshot/restore forks each beam's K/V) on a plain transformer.
        {
            let kv = match &state.kv_backend {
                KvBackend::Contiguous(kv) => kv,
                _ => {
                    return Err(CpuEngineError::Other(
                        "beam search requires the contiguous KV backend \
                         (set [inference].kv_cache_layout = \"contiguous\")"
                            .to_string(),
                    ))
                }
            };
            if self.model.weights.is_hybrid() || kv.mla.is_some() || kv.has_ring_layers() {
                return Err(CpuEngineError::Other(
                    "beam search is not supported for hybrid / MLA / sliding-window \
                     models in this build"
                        .to_string(),
                ));
            }
        }

        // Fresh KV — beam search doesn't participate in the prefix cache.
        state.reset();

        let vocab = self.model.cfg.vocab_size;
        let eos = self.model.cfg.eos_token_id;
        let length_penalty = beam.length_penalty;
        // Candidate breadth per beam: 2*num_beams guarantees num_beams non-EOS
        // survivors (HuggingFace convention), capped at the vocab.
        let cand_k = (2 * num_beams).min(vocab).max(1);

        let prompt_u32: Vec<u32> = prompt_ids.iter().map(|&t| t as u32).collect();
        let mut logits = vec![0f32; vocab];
        let mut chrome = ChromeTracer::from_env();

        // Prefill exactly like `generate_token_ids`: chunk-prefill
        // prompt[..L-1], then one forward of the last prompt token to produce
        // the first-token logits — identical code path, so num_beams=1 is
        // byte-identical to greedy.
        let (prefill, last) = prompt_ids.split_at(prompt_ids.len() - 1);
        let _ = run_chunked_prefill(
            &self.model,
            &mut state,
            &prompt_u32,
            prefill,
            0,
            self.prefill_chunk_size,
            &mut logits,
            &mut chrome,
            || false,
        );
        let prompt_len = prompt_ids.len();
        {
            let EngineState {
                kv_backend,
                delta_net_cache,
                ..
            } = &mut *state;
            forward_one_via_backend_hybrid(
                &self.model,
                kv_backend,
                delta_net_cache.as_mut(),
                last[0],
                (prompt_len - 1) as u32,
                &mut logits,
            );
        }
        // KV now holds positions [0, prompt_len); seq_len == prompt_len.

        // Per-beam state. Each beam carries a host snapshot of its advanced
        // K/V plus the next-token candidate set computed from its last forward.
        struct LiveBeam {
            tokens: Vec<u32>,
            cum_logprob: f32,
            snap: rustllama_models::llama_arch::KvSnapshot,
            topk: Vec<(u32, f32)>,
        }
        let base_snap = match &state.kv_backend {
            KvBackend::Contiguous(kv) => kv.snapshot_prefix(prompt_len),
            _ => unreachable!("backend verified contiguous above"),
        };
        let mut active: Vec<LiveBeam> = vec![LiveBeam {
            tokens: Vec::new(),
            cum_logprob: 0.0,
            snap: base_snap,
            topk: top_k_by_logit(&logits, cand_k),
        }];

        let mut pool = BeamPool::new(num_beams, length_penalty);

        for _step in 0..n_new {
            // 1) Build + sort candidates across all live beams, then select the
            //    survivors. `build_sorted_candidates` returns owned candidates,
            //    so `per_beam`'s borrow of `active` ends on this line.
            let cands = {
                let per_beam: Vec<(f32, &[(u32, f32)])> = active
                    .iter()
                    .map(|b| (b.cum_logprob, b.topk.as_slice()))
                    .collect();
                build_sorted_candidates(&per_beam)
            };
            let sel = select_step(&cands, num_beams, eos);

            // 2) Retire EOS-finishers to the pool. The stored tokens include
            //    the EOS token to match greedy (which emits EOS before stopping).
            for f in &sel.finished {
                let mut toks = active[f.parent].tokens.clone();
                toks.push(f.token);
                pool.add(toks, f.cum_logprob);
            }

            if sel.continuing.is_empty() {
                break;
            }

            // 3) Materialize each surviving beam: restore its parent's K/V,
            //    forward the new token, snapshot the advanced K/V, recompute
            //    its candidate set.
            let mut next: Vec<LiveBeam> = Vec::with_capacity(sel.continuing.len());
            for c in &sel.continuing {
                let parent_len = active[c.parent].tokens.len();
                let new_pos = (prompt_len + parent_len) as u32;
                match &mut state.kv_backend {
                    KvBackend::Contiguous(kv) => kv.restore_prefix(&active[c.parent].snap),
                    _ => unreachable!("backend verified contiguous above"),
                }
                {
                    let EngineState {
                        kv_backend,
                        delta_net_cache,
                        ..
                    } = &mut *state;
                    forward_one_via_backend_hybrid(
                        &self.model,
                        kv_backend,
                        delta_net_cache.as_mut(),
                        c.token as i32,
                        new_pos,
                        &mut logits,
                    );
                }
                let snap = match &state.kv_backend {
                    KvBackend::Contiguous(kv) => kv.snapshot_prefix(prompt_len + parent_len + 1),
                    _ => unreachable!("backend verified contiguous above"),
                };
                let mut toks = active[c.parent].tokens.clone();
                toks.push(c.token);
                next.push(LiveBeam {
                    tokens: toks,
                    cum_logprob: c.cum_logprob,
                    snap,
                    topk: top_k_by_logit(&logits, cand_k),
                });
            }
            active = next;

            // 4) Context-window guard — every live beam is the same length.
            let cur_total = prompt_len + active[0].tokens.len();
            if cur_total >= self.max_ctx {
                break;
            }

            // 5) Early-stopping / unbeatable-bound done check.
            let best_running = active
                .iter()
                .map(|b| b.cum_logprob)
                .fold(f32::NEG_INFINITY, f32::max);
            let cur_len = active[0].tokens.len();
            if pool.is_done(beam.early_stopping, best_running, cur_len) {
                break;
            }
        }

        // Finalize: any still-live beams become finished hypotheses (no EOS).
        for b in active {
            pool.add(b.tokens, b.cum_logprob);
        }

        // Reset so the next request re-prefills cleanly — beam left the live KV
        // on the last materialized beam, which no longer matches `last_ids`.
        state.reset();
        chrome.flush();

        let mut ranked = pool.into_ranked();
        ranked.truncate(beam.k());
        Ok(ranked)
    }

    /// CacheBlend generation (vLLM-style non-prefix KV reuse for RAG). The
    /// prompt is `chunks` (retrieved documents, each with its KV precomputed in
    /// isolation + cached content-addressed) followed by `suffix_ids` (the
    /// query). Instead of a full prefill, the chunk KV is injected (re-roped to
    /// its offset) and only the highest cross-attention-deviation tokens are
    /// recomputed (see [`rustllama_models::cacheblend`]).
    ///
    /// Falls back to the normal [`Self::generate_token_ids`] (over the fully
    /// assembled prompt) whenever CacheBlend can't apply: no ratio is set
    /// ([`Self::set_cacheblend`] clear / 0 — the autotuned or config value),
    /// there are no chunks or no query, the model is hybrid, or the KV backend
    /// isn't contiguous F32. The fallback is byte-identical to a plain request,
    /// so passing `documents` is always safe.
    pub fn generate_token_ids_cacheblend(
        &self,
        chunks: &[Vec<i32>],
        suffix_ids: &[i32],
        n_new: u32,
        sampling: &SamplingParams,
    ) -> Result<Vec<u32>> {
        // Assemble the full prompt once — the fallback path and the decode base.
        let mut prompt_ids: Vec<i32> = Vec::new();
        for c in chunks {
            prompt_ids.extend_from_slice(c);
        }
        let suffix_start = prompt_ids.len();
        prompt_ids.extend_from_slice(suffix_ids);

        // Cheap disqualifiers → normal generation over the assembled prompt.
        // The recompute ratio is the autotuned/config value applied at load via
        // `set_cacheblend` (None ⇒ CacheBlend off for this model).
        let blend = match self.cacheblend {
            Some(b) => b,
            None => return self.generate_token_ids(&prompt_ids, n_new, sampling),
        };
        // D2b removed the SWA/softcap disqualifier: `forward_prefill_cacheblend`
        // now threads the per-layer sliding-window span + Gemma2 attn-logit
        // soft-cap through both of its attention branches, so a windowed / capped
        // model re-assembles correctly. CacheBlend stays F32-only (the
        // `kv_dtype_cached != KvDtype::F32` disqualifier below), and the SWA/softcap
        // kv_dtype coercion at load keeps such models on F32 KV — so window/softcap
        // always ride the F32 CacheBlend path here.
        if chunks.iter().all(|c| c.is_empty())
            || suffix_ids.is_empty()
            || suffix_start == 0
            || prompt_ids.len() < 2
            || self.model.weights.is_hybrid()
            || self.kv_dtype_cached != KvDtype::F32
        {
            return self.generate_token_ids(&prompt_ids, n_new, sampling);
        }

        let _sycl_guard = install_sycl_dispatch_if_requested();
        rustllama_models::accel::set_flash_attention(self.flash_attention);
        rustllama_models::accel::set_n_gpu_layers(self.n_gpu_layers);
        rustllama_models::accel::set_cpu_force_patterns(self.cpu_force_patterns.clone());
        self.prepare_usm_for_sync_path();
        let mut state = self.state.lock().expect("state lock");

        // The backend layout (contiguous vs paged) is fixed at load; CacheBlend
        // injects at absolute offsets into a contiguous F32 slab.
        //
        // P2: a SWA ring cache is `cap`-strided and ring-addressed, so
        // CacheBlend's absolute-offset KV injection + re-RoPE would corrupt it
        // (and it also builds isolated chunk snapshots via the batched prefill,
        // which the ring doesn't support). Fall back to a plain generate for
        // ring models — correct, just without the RAG KV-reuse optimization.
        let kv_is_ring = matches!(
            &state.kv_backend,
            KvBackend::Contiguous(kv) if kv.has_ring_layers()
        );
        if !matches!(state.kv_backend, KvBackend::Contiguous(_)) || kv_is_ring {
            drop(state);
            return self.generate_token_ids(&prompt_ids, n_new, sampling);
        }

        let total_required = prompt_ids.len() + n_new as usize;
        if total_required > self.max_ctx {
            return Err(CpuEngineError::PromptTooLong {
                prompt_len: prompt_ids.len(),
                max_ctx: self.max_ctx,
            });
        }

        // Precompute / fetch each chunk's isolated KV snapshot, content-addressed
        // and shared across requests + forks (so a hot retrieved document is
        // prefilled once). Offsets stack in chunk order.
        let cache = cacheblend_cache_for_model(&self.model_id);
        let mut placements: Vec<rustllama_models::cacheblend::ChunkPlacement> =
            Vec::with_capacity(chunks.len());
        let mut off = 0usize;
        for c in chunks {
            if c.is_empty() {
                continue;
            }
            let snap = cacheblend_get_or_build_snapshot(&self.model, &cache, c);
            placements.push(rustllama_models::cacheblend::ChunkPlacement {
                offset: off,
                snapshot: snap,
            });
            off += c.len();
        }

        let vocab = self.model.cfg.vocab_size;
        let mut logits = vec![0f32; vocab];
        let mut history: Vec<u32> = prompt_ids.iter().map(|&t| t as u32).collect();
        let prompt_u32: Vec<u32> = history.clone();

        // CacheBlend builds KV from chunk snapshots at absolute offsets, so it
        // needs a FRESH cache — neutralize prefix reuse (resets KV, returns 0).
        let _ = state.prepare_prefix_reuse(&prompt_u32, false, PREFIX_REUSE_MIN_TOKENS, 0);

        let mut chrome = ChromeTracer::from_env();
        let prefill_start = std::time::Instant::now();
        // Selective-recompute prefill over the WHOLE prompt; its final-position
        // logits seed decode (the query's last token is always in the suffix,
        // hence always recomputed → its distribution is correct).
        {
            let KvBackend::Contiguous(kv) = &mut state.kv_backend else {
                unreachable!("CacheBlend gate verified a contiguous backend");
            };
            let cb_logits = self.model.forward_prefill_cacheblend(
                &prompt_ids,
                &placements,
                suffix_start,
                blend,
                kv,
            );
            logits.copy_from_slice(&cb_logits);
        }
        chrome.span("prefill_total", prefill_start);
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        let mut sampler = Sampler::new(sampling.clone());
        let mut grammar = self.build_grammar(sampling);
        // The full prompt occupies positions [0, len); the first sampled token
        // lands at `len`.
        let mut next_pos = prompt_ids.len() as u32;
        let mut out = Vec::with_capacity(n_new as usize);
        let mut decode_ms = 0.0f64;

        for step in 0..n_new {
            // `logits` holds the next-token distribution (CacheBlend prefill on
            // step 0, the previous decode forward after).
            let t_sample = std::time::Instant::now();
            let next_tok = sampler.sample_with_grammar(&mut logits, &history, grammar.as_ref());
            chrome.span(&format!("sample_{step}"), t_sample);
            if let Some(g) = grammar.as_mut() {
                g.advance(next_tok);
            }
            out.push(next_tok);
            history.push(next_tok);
            if let Some(eos) = self.model.cfg.eos_token_id {
                if next_tok == eos {
                    break;
                }
            }
            if (next_pos as usize) >= self.max_ctx {
                break;
            }
            // Forward the sampled token to produce the next distribution.
            let t_fwd = std::time::Instant::now();
            {
                let EngineState {
                    kv_backend,
                    delta_net_cache,
                    ..
                } = &mut *state;
                forward_one_via_backend_hybrid(
                    &self.model,
                    kv_backend,
                    delta_net_cache.as_mut(),
                    next_tok as i32,
                    next_pos,
                    &mut logits,
                );
            }
            chrome.span(&format!("decode_fwd_{step}"), t_fwd);
            decode_ms += t_fwd.elapsed().as_secs_f64() * 1000.0;
            next_pos += 1;
        }
        chrome.flush();

        let tool_call_limit_hit = grammar
            .as_ref()
            .map(|g| g.tool_call_limit_blocked())
            .unwrap_or(false);
        // `suffix_start` chunk tokens were reused (CacheBlend's "cache hit"); the
        // rest of the prompt was recomputed.
        let reused = suffix_start.min(prompt_ids.len());
        self.commit_request_stats(RequestStats {
            prefill_ms,
            decode_ms,
            tokens_prefilled: (prompt_ids.len().saturating_sub(reused)) as u32,
            cache_hit_tokens: reused as u32,
            tokens_generated: out.len() as u32,
            tool_call_limit_hit,
        });
        // No `commit_snapshot`: the blended KV isn't a clean token prefix, so it
        // must not poison the prefix pool for later exact-prefix requests.

        Ok(out)
    }

    /// Like [`generate_token_ids`] but also collects per-token logprob info
    /// (chosen-token logprob + top-K alternatives) for callers that want to
    /// expose `logprobs` in their response (the OpenAI `/v1/completions`
    /// path). Uses the same prefix-cache + sampler pipeline; the only extra
    /// work is one `compute_logprobs` call per emitted token, which captures
    /// raw logits before the sampler mutates them.
    ///
    /// `k` is the number of top alternatives to capture (clamped to vocab).
    /// `k == 0` returns logprobs with no `top` vec.
    pub fn generate_token_ids_with_logprobs(
        &self,
        prompt_ids: &[i32],
        n_new: u32,
        sampling: &SamplingParams,
        k: usize,
    ) -> Result<Vec<(u32, crate::TokenLogprobs)>> {
        // P11: scope the per-request LoRA selection for this inline call.
        let _lora_scope =
            rustllama_models::lora::scope_active_lora(self.lora_adapter_for(sampling)?);
        let _sycl_guard = install_sycl_dispatch_if_requested();
        rustllama_models::accel::set_flash_attention(self.flash_attention);
        rustllama_models::accel::set_n_gpu_layers(self.n_gpu_layers);
        rustllama_models::accel::set_cpu_force_patterns(self.cpu_force_patterns.clone());
        self.prepare_usm_for_sync_path();
        let mut state = self.state.lock().expect("state lock");

        let total_required = prompt_ids.len() + n_new as usize;
        if total_required > self.max_ctx {
            return Err(CpuEngineError::PromptTooLong {
                prompt_len: prompt_ids.len(),
                max_ctx: self.max_ctx,
            });
        }
        if prompt_ids.is_empty() {
            return Ok(Vec::new());
        }

        let vocab = self.model.cfg.vocab_size;
        let mut logits = vec![0f32; vocab];
        let mut history: Vec<u32> = prompt_ids.iter().map(|&t| t as u32).collect();
        let prompt_u32: Vec<u32> = prompt_ids.iter().map(|&t| t as u32).collect();

        let prompt_max = prompt_ids.len().saturating_sub(1);
        // P11: the prompt-prefix cache is keyed on token content, NOT on the
        // adapter, so an adapted request must not reuse/populate it with KV from
        // a different selection. Disable prefix reuse when an adapter is active;
        // `None` ⇒ `&& true` ⇒ byte-identical to the base path.
        let prefix_cache = self.prefix_cache && sampling.lora_adapter.is_none();
        let effective = state.prepare_prefix_reuse(
            &prompt_u32,
            prefix_cache,
            PREFIX_REUSE_MIN_TOKENS,
            prompt_max,
        );

        let mut chrome = ChromeTracer::from_env();
        let prefill_start = std::time::Instant::now();
        let (prefill, last) = prompt_ids.split_at(prompt_ids.len() - 1);
        let _ = run_chunked_prefill(
            &self.model,
            &mut state,
            &prompt_u32,
            prefill,
            effective,
            self.prefill_chunk_size,
            &mut logits,
            &mut chrome,
            || false,
        );
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

        let mut sampler = Sampler::new(sampling.clone());
        let mut grammar = self.build_grammar(sampling);
        let mut next_input = last[0];
        let mut next_pos = prefill.len() as u32;
        let mut out: Vec<(u32, crate::TokenLogprobs)> = Vec::with_capacity(n_new as usize);
        let mut new_ids: Vec<u32> = Vec::with_capacity(n_new as usize);
        let mut decode_ms = 0.0f64;

        for _ in 0..n_new {
            let t_fwd = std::time::Instant::now();
            {
                let EngineState {
                    kv_backend,
                    delta_net_cache,
                    ..
                } = &mut *state;
                forward_one_via_backend_hybrid(
                    &self.model,
                    kv_backend,
                    delta_net_cache.as_mut(),
                    next_input,
                    next_pos,
                    &mut logits,
                );
            }
            decode_ms += t_fwd.elapsed().as_secs_f64() * 1000.0;

            // Snapshot raw logits before the sampler mutates them.
            let raw = logits.clone();

            let next_tok = sampler.sample_with_grammar(&mut logits, &history, grammar.as_ref());

            if let Some(g) = grammar.as_mut() {
                g.advance(next_tok);
            }
            let lp = compute_logprobs(&raw, next_tok, k);
            out.push((next_tok, lp));
            new_ids.push(next_tok);
            history.push(next_tok);

            if let Some(eos) = self.model.cfg.eos_token_id {
                if next_tok == eos {
                    break;
                }
            }

            next_input = next_tok as i32;
            next_pos += 1;
            if (next_pos as usize) >= self.max_ctx {
                break;
            }
        }

        let tool_call_limit_hit = grammar
            .as_ref()
            .map(|g| g.tool_call_limit_blocked())
            .unwrap_or(false);
        self.commit_request_stats(RequestStats {
            prefill_ms,
            decode_ms,
            tokens_prefilled: (prefill.len().saturating_sub(effective)) as u32,
            cache_hit_tokens: effective as u32,
            tokens_generated: new_ids.len() as u32,
            tool_call_limit_hit,
        });

        let mut snapshot = prompt_u32;
        snapshot.extend(new_ids);
        state.commit_snapshot(snapshot);
        // F12: periodic warm-restart checkpoint (inert unless
        // `kv_persist_interval_secs > 0` and the clock has elapsed). The pool
        // now carries this request's full sequence, so the checkpoint preserves
        // the live session. Uses the held `state` — never re-locks.
        self.maybe_checkpoint_kv_persist(&state);

        Ok(out)
    }

    /// P15 (echo + logprobs): the per-token logprob of each PROMPT token under
    /// the model — what OpenAI's `/v1/completions` returns when `echo: true`
    /// AND `logprobs` are both set (the prompt logprobs are prepended to the
    /// completion's).
    ///
    /// Returns a `Vec` the same length as `prompt_ids`. Index `0` is `None`:
    /// the first prompt token has no preceding context, so its conditional
    /// probability is undefined (OpenAI emits `null` there). Index `i > 0`
    /// holds `logP(prompt[i] | prompt[0..i])` plus the top-`k` alternatives at
    /// that position.
    ///
    /// Implemented as a fresh SERIAL forward over the prompt (one position at a
    /// time): forwarding `prompt[pos]` at position `pos` yields the logits that
    /// predict `prompt[pos+1]`, so one pass produces every conditional. This is
    /// correct for every model family (dense / MoE / hybrid) and KV dtype — the
    /// batched prefill keeps only the FINAL-position logits, which is why the
    /// scan is needed. Only invoked on the opt-in echo+logprobs path, so the
    /// extra forward work is acceptable. The engine's KV state is reset before
    /// and after so neither a prior request nor the caller's own generation is
    /// disturbed.
    pub fn prompt_token_logprobs(
        &self,
        prompt_ids: &[i32],
        k: usize,
    ) -> Result<Vec<Option<crate::TokenLogprobs>>> {
        let _sycl_guard = install_sycl_dispatch_if_requested();
        rustllama_models::accel::set_flash_attention(self.flash_attention);
        rustllama_models::accel::set_n_gpu_layers(self.n_gpu_layers);
        rustllama_models::accel::set_cpu_force_patterns(self.cpu_force_patterns.clone());
        self.prepare_usm_for_sync_path();

        let l = prompt_ids.len();
        if l == 0 {
            return Ok(Vec::new());
        }
        if l > self.max_ctx {
            return Err(CpuEngineError::PromptTooLong {
                prompt_len: l,
                max_ctx: self.max_ctx,
            });
        }
        let vocab = self.model.cfg.vocab_size;
        let mut logits = vec![0f32; vocab];
        let mut out: Vec<Option<crate::TokenLogprobs>> = Vec::with_capacity(l);
        // The first prompt token has no conditioning context → null logprob.
        out.push(None);

        let mut state = self.state.lock().expect("state lock");
        // Fresh KV: an independent scan, not a continuation of any prior request.
        state.reset();
        for pos in 0..l.saturating_sub(1) {
            {
                let EngineState {
                    kv_backend,
                    delta_net_cache,
                    ..
                } = &mut *state;
                forward_one_via_backend_hybrid(
                    &self.model,
                    kv_backend,
                    delta_net_cache.as_mut(),
                    prompt_ids[pos],
                    pos as u32,
                    &mut logits,
                );
            }
            let next_tok = prompt_ids[pos + 1] as u32;
            out.push(Some(compute_logprobs(&logits, next_tok, k)));
        }
        // Wipe the scan's KV so the caller's own generation starts from a known
        // (empty) state — its `prepare_prefix_reuse` sees `last_ids` cleared and
        // won't mistake the half-scan for a reusable live prefix.
        state.reset();
        Ok(out)
    }

    /// True-streaming variant of [`generate_token_ids`]: pushes each
    /// token to the provided `tx` as soon as it's sampled rather than
    /// collecting everything into a `Vec<u32>` and returning at the end.
    /// Drops out of the generation loop when the receiver disconnects.
    ///
    /// When `logprobs_k` is `Some(k)`, each emitted [`crate::Token`]
    /// carries the chosen token's logprob plus the top-K alternatives
    /// (raw-logit log-softmax, computed from the model's distribution
    /// before any sampler mutation).
    ///
    /// Designed for `/v1/completions` streaming: that endpoint has the
    /// prompt already tokenized (FIM/Bos special tokens injected), so a
    /// text-prompt method like [`Engine::generate`] doesn't fit. Same
    /// prefix-cache, sampler, and cancellation semantics as
    /// [`drive_generation`].
    /// Tokenize `text_chunks` and concatenate into one prompt-id Vec.
    /// Used by [`Self::generate_text_chunks_streaming`] so callers
    /// who pre-split very long prompts (whole codebases, multi-MB
    /// transcripts) at natural boundaries (file separators, message
    /// boundaries) don't have to glue the chunks back into one
    /// `String` first — RAM cost drops to one chunk at a time during
    /// tokenization.
    ///
    /// Returns an error when the tokenizer isn't bound. The
    /// individual chunk boundaries are passed straight to
    /// `tokenizer.encode_no_special` so the tokenizer can apply its
    /// own normalization without injecting BOS/EOS at every seam —
    /// only the first chunk gets the leading BOS treatment via
    /// `add_bos` on the first call.
    pub fn tokenize_chunks(
        &self,
        text_chunks: impl IntoIterator<Item = String>,
        add_bos_on_first: bool,
    ) -> Result<Vec<i32>> {
        let tokenizer = self
            .tokenizer
            .as_deref()
            .ok_or(CpuEngineError::NoTokenizer("tokenize_chunks"))?;
        let mut out: Vec<i32> = Vec::new();
        let mut first = true;
        for chunk in text_chunks {
            // First chunk keeps the caller's BOS preference; subsequent
            // chunks suppress BOS so the stream re-concatenates
            // cleanly (the engine's prompt-id contract is "exactly
            // what gets prefilled").
            let add_bos = first && add_bos_on_first;
            let ids = tokenizer.encode(&chunk, add_bos)?;
            for id in ids {
                if id > i32::MAX as u32 {
                    return Err(CpuEngineError::Other(format!(
                        "tokenizer produced id {id} outside i32 range"
                    )));
                }
                out.push(id as i32);
            }
            first = false;
        }
        Ok(out)
    }

    /// Streaming-prefill generation: accepts `text_chunks` as an
    /// owned iterator of `String`, pipelines tokenization with
    /// engine prefill via a bounded mpsc, and streams tokens out via
    /// `tx` exactly like [`Self::generate_token_ids_streaming`].
    ///
    /// The pipeline shape:
    ///   - A `spawn_blocking` task tokenizes chunks one at a time and
    ///     pushes the resulting `Vec<i32>` into a `mpsc::channel(1)`.
    ///   - The engine task drains the channel and appends each
    ///     chunk's ids to a single growing prompt-id Vec, then runs
    ///     the existing prefill + decode loop on the concatenated
    ///     result.
    ///
    /// Net win on ultra-long contexts: chunk N+1's tokenize step
    /// overlaps with chunk N's tokenize-already-in-vec wait time.
    /// For typical (≤8K-token) prompts the overlap is negligible
    /// because tokenization is microseconds.
    ///
    /// Cancellation: if `tx.is_closed()` becomes true mid-tokenize,
    /// the producer task drops the next chunk and exits. The engine
    /// task observes the closed channel + bails through the existing
    /// cancellation path in [`Self::generate_token_ids_streaming`].
    pub async fn generate_text_chunks_streaming(
        self: Arc<Self>,
        text_chunks: Vec<String>,
        sampling: SamplingParams,
        logprobs_k: Option<usize>,
        add_bos_on_first: bool,
        tx: tokio::sync::mpsc::Sender<std::result::Result<crate::Token, String>>,
    ) -> Result<()> {
        let (chunk_tx, mut chunk_rx) = tokio::sync::mpsc::channel::<Vec<i32>>(1);
        let producer = {
            let engine = self.clone();
            let tx_closed = tx.clone();
            tokio::task::spawn_blocking(move || -> Result<()> {
                let tokenizer = engine
                    .tokenizer
                    .as_deref()
                    .ok_or(CpuEngineError::NoTokenizer(
                        "generate_text_chunks_streaming",
                    ))?;
                let mut first = true;
                for chunk in text_chunks {
                    if tx_closed.is_closed() {
                        // Consumer bailed; stop tokenizing.
                        return Ok(());
                    }
                    let add_bos = first && add_bos_on_first;
                    let ids = tokenizer.encode(&chunk, add_bos)?;
                    let mut as_i32 = Vec::with_capacity(ids.len());
                    for id in ids {
                        if id > i32::MAX as u32 {
                            return Err(CpuEngineError::Other(format!(
                                "tokenizer produced id {id} outside i32 range"
                            )));
                        }
                        as_i32.push(id as i32);
                    }
                    // Blocking send into a tokio channel from a
                    // blocking thread — uses the channel's
                    // sync-friendly path.
                    if chunk_tx.blocking_send(as_i32).is_err() {
                        return Ok(());
                    }
                    first = false;
                }
                Ok(())
            })
        };

        // Consumer side: drain the channel into a single prompt-id
        // Vec. We accumulate fully before calling into the engine —
        // the prefill machinery (run_chunked_prefill) already handles
        // its own per-chunk batching, so we don't need to interleave
        // engine prefill with channel reads. The win is in the
        // tokenize/wait overlap above, not in interleaved prefill.
        let mut prompt_ids: Vec<i32> = Vec::new();
        while let Some(chunk_ids) = chunk_rx.recv().await {
            prompt_ids.extend(chunk_ids);
        }
        // Surface any tokenizer error from the producer task.
        producer
            .await
            .map_err(|e| CpuEngineError::Other(format!("tokenize task panicked: {e}")))??;

        // Hand the assembled ids to the existing streaming generate
        // path on a blocking thread so the engine's `state.lock`
        // doesn't deadlock the tokio runtime.
        let engine = self.clone();
        tokio::task::spawn_blocking(move || {
            engine.generate_token_ids_streaming(&prompt_ids, &sampling, logprobs_k, &tx)
        })
        .await
        .map_err(|e| CpuEngineError::Other(format!("generate task panicked: {e}")))?
    }

    /// F9-tail — streaming counterpart of [`Self::generate_token_ids_mamba`]
    /// so a canonical Mamba/SSM model is also SERVED (the `serve` / chat paths
    /// stream through `generate_token_ids_streaming`). Same selective-scan
    /// forward + shared sampler/grammar glue; emits each decoded token into
    /// `tx` with incremental detokenization, identical to the transformer
    /// streaming loop below.
    fn generate_token_ids_streaming_mamba(
        &self,
        mm: &rustllama_models::mamba_arch::MambaModel,
        prompt_ids: &[i32],
        sampling: &SamplingParams,
        logprobs_k: Option<usize>,
        tx: &tokio::sync::mpsc::Sender<std::result::Result<crate::Token, String>>,
    ) -> Result<()> {
        let tokenizer = self
            .tokenizer
            .as_deref()
            .ok_or(CpuEngineError::NoTokenizer("generate_token_ids_streaming"))?;
        if prompt_ids.is_empty() {
            return Ok(());
        }
        let total_required = prompt_ids.len() + sampling.max_tokens as usize;
        if total_required > self.max_ctx {
            return Err(CpuEngineError::PromptTooLong {
                prompt_len: prompt_ids.len(),
                max_ctx: self.max_ctx,
            });
        }
        let vocab = mm.cfg.vocab_size;
        let mut logits = vec![0f32; vocab];
        let mut cache = rustllama_models::mamba_arch::MambaCache::new(&mm.cfg);
        let mut history: Vec<u32> = prompt_ids.iter().map(|&t| t as u32).collect();
        mm.forward_sequence(prompt_ids, &mut cache, &mut logits);

        let mut sampler = Sampler::new(sampling.clone());
        let mut grammar = self.build_grammar(sampling);
        let eos = self.model.cfg.eos_token_id;
        for _ in 0..sampling.max_tokens {
            if tx.is_closed() {
                break;
            }
            let raw_logits: Option<Vec<f32>> = logprobs_k.map(|_| logits.clone());
            let next_tok = sampler.sample_with_grammar(&mut logits, &history, grammar.as_ref());
            if let Some(g) = grammar.as_mut() {
                g.advance(next_tok);
            }
            let token_lp = match (logprobs_k, raw_logits) {
                (Some(k), Some(raw)) => Some(compute_logprobs(&raw, next_tok, k)),
                _ => None,
            };
            history.push(next_tok);
            let text = tokenizer.decode_single(next_tok, true).unwrap_or_default();
            if tx
                .blocking_send(Ok(crate::Token {
                    id: next_tok,
                    text,
                    logprobs: token_lp,
                }))
                .is_err()
            {
                break;
            }
            if Some(next_tok) == eos {
                break;
            }
            mm.forward_one(next_tok as i32, &mut cache, &mut logits);
        }
        Ok(())
    }

    pub fn generate_token_ids_streaming(
        &self,
        prompt_ids: &[i32],
        sampling: &SamplingParams,
        logprobs_k: Option<usize>,
        tx: &tokio::sync::mpsc::Sender<std::result::Result<crate::Token, String>>,
    ) -> Result<()> {
        // P11: scope the per-request LoRA selection for this inline call.
        let _lora_scope =
            rustllama_models::lora::scope_active_lora(self.lora_adapter_for(sampling)?);
        // F9-tail — canonical Mamba/SSM models stream through the selective-scan
        // forward (the `model` field is a transformer shell, never executed).
        if let Some(mm) = self.mamba.clone() {
            return self
                .generate_token_ids_streaming_mamba(&mm, prompt_ids, sampling, logprobs_k, tx);
        }
        let _sycl_guard = install_sycl_dispatch_if_requested();
        rustllama_models::accel::set_flash_attention(self.flash_attention);
        rustllama_models::accel::set_n_gpu_layers(self.n_gpu_layers);
        rustllama_models::accel::set_cpu_force_patterns(self.cpu_force_patterns.clone());
        self.prepare_usm_for_sync_path();
        let tokenizer = self
            .tokenizer
            .as_deref()
            .ok_or(CpuEngineError::NoTokenizer("generate_token_ids_streaming"))?;
        let mut state = self.state.lock().expect("state lock");

        let total_required = prompt_ids.len() + sampling.max_tokens as usize;
        if total_required > self.max_ctx {
            return Err(CpuEngineError::PromptTooLong {
                prompt_len: prompt_ids.len(),
                max_ctx: self.max_ctx,
            });
        }
        if prompt_ids.is_empty() {
            return Ok(());
        }

        let vocab = self.model.cfg.vocab_size;
        let mut logits = vec![0f32; vocab];
        let mut history: Vec<u32> = prompt_ids.iter().map(|&t| t as u32).collect();
        let prompt_u32: Vec<u32> = prompt_ids.iter().map(|&t| t as u32).collect();

        let prompt_max = prompt_ids.len().saturating_sub(1);
        // P11: the prompt-prefix cache is keyed on token content, NOT on the
        // adapter, so an adapted request must not reuse/populate it with KV from
        // a different selection. Disable prefix reuse when an adapter is active;
        // `None` ⇒ `&& true` ⇒ byte-identical to the base path.
        let prefix_cache = self.prefix_cache && sampling.lora_adapter.is_none();
        let effective = state.prepare_prefix_reuse(
            &prompt_u32,
            prefix_cache,
            PREFIX_REUSE_MIN_TOKENS,
            prompt_max,
        );

        let mut chrome = ChromeTracer::from_env();
        let prefill_start = std::time::Instant::now();
        let (prefill, last) = prompt_ids.split_at(prompt_ids.len() - 1);
        let prefilled = run_chunked_prefill(
            &self.model,
            &mut state,
            &prompt_u32,
            prefill,
            effective,
            self.prefill_chunk_size,
            &mut logits,
            &mut chrome,
            || tx.is_closed(),
        );
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
        if prefilled < prefill.len() {
            // Cancelled mid-prefill. `state.last_ids` already reflects
            // exactly the work that completed, ready for the next request
            // to pick up via LCP. Don't proceed to decode.
            return Ok(());
        }

        let mut sampler = Sampler::new(sampling.clone());
        let mut grammar = self.build_grammar(sampling);
        let mut next_input = last[0];
        let mut next_pos = prefill.len() as u32;
        let mut generated: Vec<u32> = Vec::with_capacity(sampling.max_tokens as usize);
        let mut decode_ms = 0.0f64;

        for _ in 0..sampling.max_tokens {
            if tx.is_closed() {
                break;
            }
            let t_fwd = std::time::Instant::now();
            {
                let EngineState {
                    kv_backend,
                    delta_net_cache,
                    ..
                } = &mut *state;
                forward_one_via_backend_hybrid(
                    &self.model,
                    kv_backend,
                    delta_net_cache.as_mut(),
                    next_input,
                    next_pos,
                    &mut logits,
                );
            }
            decode_ms += t_fwd.elapsed().as_secs_f64() * 1000.0;

            // Capture raw logits before the sampler mutates them, but only
            // when the caller actually asked for logprobs (avoids the
            // 152K-element clone on the common path).
            let raw_logits: Option<Vec<f32>> = logprobs_k.map(|_| logits.clone());

            let next_tok = sampler.sample_with_grammar(&mut logits, &history, grammar.as_ref());

            if let Some(g) = grammar.as_mut() {
                g.advance(next_tok);
            }
            let token_lp = match (logprobs_k, raw_logits) {
                (Some(k), Some(raw)) => Some(compute_logprobs(&raw, next_tok, k)),
                _ => None,
            };

            generated.push(next_tok);
            history.push(next_tok);

            // Decode this token's text incrementally. Detokenizing only
            // the new id (not the full so-far sequence) matches what
            // OpenAI streaming clients expect: each delta carries the
            // exact bytes that should be appended to the running output.
            let text = tokenizer.decode_single(next_tok, true).unwrap_or_default();

            if tx
                .blocking_send(Ok(crate::Token {
                    id: next_tok,
                    text,
                    logprobs: token_lp,
                }))
                .is_err()
            {
                break;
            }

            if let Some(eos) = self.model.cfg.eos_token_id {
                if next_tok == eos {
                    break;
                }
            }

            next_input = next_tok as i32;
            next_pos += 1;
            if (next_pos as usize) >= self.max_ctx {
                break;
            }
        }

        let tool_call_limit_hit = grammar
            .as_ref()
            .map(|g| g.tool_call_limit_blocked())
            .unwrap_or(false);
        self.commit_request_stats(RequestStats {
            prefill_ms,
            decode_ms,
            tokens_prefilled: (prefill.len().saturating_sub(effective)) as u32,
            cache_hit_tokens: effective as u32,
            tokens_generated: generated.len() as u32,
            tool_call_limit_hit,
        });

        // Snapshot prompt+generated for the prefix cache, same as the
        // batch variant.
        let mut snapshot = prompt_u32;
        snapshot.extend(generated);
        state.commit_snapshot(snapshot);
        // F12: periodic warm-restart checkpoint (inert unless
        // `kv_persist_interval_secs > 0` and the clock has elapsed). The pool
        // now carries this request's full sequence, so the checkpoint preserves
        // the live session. Uses the held `state` — never re-locks.
        self.maybe_checkpoint_kv_persist(&state);

        Ok(())
    }

    /// Synchronous text-in / text-out generation. Used by the streaming
    /// `Engine::generate` and `Engine::chat` impls via `spawn_blocking`.
    pub fn generate_text(&self, prompt: &str, sampling: &SamplingParams) -> Result<String> {
        let tokenizer = self
            .tokenizer
            .as_deref()
            .ok_or(CpuEngineError::NoTokenizer("generate_text"))?;
        let add_bos = tokenizer.add_bos_token();
        let ids = tokenizer
            .encode(prompt, add_bos)?
            .into_iter()
            .map(|v| v as i32)
            .collect::<Vec<_>>();
        let new_ids = self.generate_token_ids(&ids, sampling.max_tokens, sampling)?;
        let text = tokenizer.decode(&new_ids, true)?;
        Ok(text)
    }

    /// Like [`generate_text`] but the caller has already assembled the
    /// prompt as token ids. Used by FIM completion where the prompt is a
    /// mix of model-specific special tokens (e.g. `<|fim_prefix|>`) and
    /// tokenized text segments.
    pub fn generate_text_from_ids(
        &self,
        prompt_ids: &[u32],
        sampling: &SamplingParams,
    ) -> Result<String> {
        let tokenizer = self
            .tokenizer
            .as_deref()
            .ok_or(CpuEngineError::NoTokenizer("generate_text_from_ids"))?;
        let ids: Vec<i32> = prompt_ids.iter().map(|&v| v as i32).collect();
        let new_ids = self.generate_token_ids(&ids, sampling.max_tokens, sampling)?;
        let text = tokenizer.decode(&new_ids, true)?;
        Ok(text)
    }

    /// Text-returning CacheBlend generation: `chunks` (retrieved documents) +
    /// `suffix_ids` (the query). Delegates to [`Self::generate_token_ids_cacheblend`]
    /// (which falls back to a normal prefill over the assembled prompt when
    /// CacheBlend can't apply), then detokenizes. Safe to call unconditionally.
    pub fn generate_text_cacheblend(
        &self,
        chunks: &[Vec<i32>],
        suffix_ids: &[i32],
        sampling: &SamplingParams,
    ) -> Result<String> {
        let tokenizer = self
            .tokenizer
            .as_deref()
            .ok_or(CpuEngineError::NoTokenizer("generate_text_cacheblend"))?;
        let new_ids =
            self.generate_token_ids_cacheblend(chunks, suffix_ids, sampling.max_tokens, sampling)?;
        let text = tokenizer.decode(&new_ids, true)?;
        Ok(text)
    }
}

/// Per-model content-addressed chunk-KV cache registry. Keyed by model id so a
/// chunk's isolated KV (which depends only on (model, tokens)) is shared across
/// every request and every `fork_for_concurrent_use` clone. The byte cap
/// AUTO-SIZES from free RAM (no env var) — the cache is demand-filled and LRU,
/// so the cap just bounds how many hot documents' KV it retains.
fn cacheblend_cache_for_model(
    model_id: &str,
) -> Arc<Mutex<rustllama_models::cacheblend::ChunkKvCache>> {
    static REG: std::sync::OnceLock<
        Mutex<
            std::collections::HashMap<
                String,
                Arc<Mutex<rustllama_models::cacheblend::ChunkKvCache>>,
            >,
        >,
    > = std::sync::OnceLock::new();
    let reg = REG.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    // ≤ 1/16 of free RAM, capped at 4 GiB. Demand-filled + LRU, so this only
    // bounds retention; a tiny box simply caches fewer chunks.
    let avail = rustllama_runtime::memory_info().available_bytes as usize;
    let cap_bytes = (avail / 16).min(4 * 1024 * 1024 * 1024);
    let mut map = reg.lock().expect("cacheblend registry");
    map.entry(model_id.to_string())
        .or_insert_with(|| {
            Arc::new(Mutex::new(rustllama_models::cacheblend::ChunkKvCache::new(
                cap_bytes,
            )))
        })
        .clone()
}

/// Fetch (or compute + cache) one chunk's isolated KV snapshot — a standalone
/// F32-KV prefill of the chunk at position 0. The cache lock is held only for
/// the get/insert, never across the (slow) prefill.
fn cacheblend_get_or_build_snapshot(
    model: &LlamaModel,
    cache: &Mutex<rustllama_models::cacheblend::ChunkKvCache>,
    chunk: &[i32],
) -> Arc<rustllama_models::llama_arch::KvSnapshot> {
    let hash = rustllama_models::cacheblend::chunk_content_hash(chunk);
    {
        let mut c = cache.lock().expect("chunk cache lock");
        if let Some(snap) = c.get(hash) {
            return snap;
        }
    }
    // Miss: compute the isolated snapshot (no cache lock held). A standalone
    // prefill at position 0 leaves the chunk's KV for re-roped placement.
    // P2: force a NON-ring (full-slab) cache — this chunk prefill is then
    // snapshotted at absolute offsets, which a SWA ring can't represent (the
    // full slab still masks an SWA model's window correctly).
    let mut scratch = rustllama_models::llama_arch::KvCache::new_with_dtype_no_ring(
        &model.cfg,
        chunk.len().max(1),
        rustllama_models::llama_arch::KvDtype::F32,
    );
    let _ = model.forward_prefill_batched_f32(chunk, 0, &mut scratch);
    let snap = Arc::new(scratch.snapshot_prefix(chunk.len()));
    cache
        .lock()
        .expect("chunk cache lock")
        .insert(hash, snap.clone());
    snap
}

/// Walk `prefill[effective..]` in chunks of `chunk_size` tokens, calling
/// `forward_one` per token. After each chunk completes, snapshot the
/// partial prefix into `state.last_ids` so a cancellation mid-prefill
/// leaves a usable cache for the next request. Per-token cancellation
/// polling preserves the snappy-abort property the streaming API needs.
///
/// Returns the index of the next token that would need prefilling. Equal
/// to `prefill.len()` if the loop completed; less if `should_stop` fired
/// midway. Callers should bail without proceeding to decode in that case
/// — `state.last_ids` already reflects the partial work.
fn run_chunked_prefill(
    model: &LlamaModel,
    state: &mut EngineState,
    prompt_u32: &[u32],
    prefill: &[i32],
    effective: usize,
    chunk_size: usize,
    logits: &mut [f32],
    chrome: &mut ChromeTracer,
    mut should_stop: impl FnMut() -> bool,
) -> usize {
    let chunk = chunk_size.max(1);
    let mut idx = effective;
    // Batched chunk prefill (4a): non-hybrid models on the contiguous
    // backend forward each chunk through `LlamaModel::forward_prefill`
    // — one multi-query flash-prefill attention call per layer per
    // chunk instead of a `forward_one` per token. Gated by
    // `[inference].prefill_batched` (the typed `accel::prefill_batched()`
    // load-option knob, applied from config/env by `LoadOptions`);
    // `forward_prefill` itself picks the batched kernel per KV dtype
    // and would serial-loop otherwise, so the gate doubles as the
    // dispatch condition. Cancellation granularity becomes the chunk —
    // exactly the granularity chunking exists to provide — and the
    // per-chunk `last_ids` snapshot cadence is unchanged.
    let batched = !model.weights.is_hybrid()
        && matches!(state.kv_backend, KvBackend::Contiguous(_))
        && rustllama_models::accel::prefill_batched();
    if batched {
        while idx < prefill.len() {
            if should_stop() {
                return idx;
            }
            let chunk_end = (idx + chunk).min(prefill.len());
            let t_chunk = std::time::Instant::now();
            {
                let KvBackend::Contiguous(kv) = &mut state.kv_backend else {
                    unreachable!("batched prefill gate verified a contiguous backend");
                };
                let chunk_logits = model.forward_prefill(&prefill[idx..chunk_end], idx as u32, kv);
                logits.copy_from_slice(&chunk_logits);
            }
            chrome.span(&format!("prefill_chunk_{idx}_to_{chunk_end}"), t_chunk);
            idx = chunk_end;
            // Same invariant as the serial loop below: `last_ids`
            // credits exactly the tokens whose K/V is in the cache.
            state.last_ids = prompt_u32[..idx].to_vec();
        }
        return idx;
    }
    // Hybrid chunk prefill (Phase 4): grouped-expert MoE execution +
    // layer-ahead expert-pool readahead via
    // `LlamaModel::forward_prefill_hybrid`. Requires the contiguous
    // backend (hybrid models force contiguous F32 KV anyway). Gated
    // by its own opt-in — separate from `prefill_batched` — until
    // output coherence is verified against the serial path on real
    // weights: this model family's output quality has been fragile
    // under numerically-equivalent-but-reordered math (see the
    // tighter-than-nano quant notebook), so the default stays off
    // for one release. Flip `[inference].prefill_batched_hybrid`
    // (or RUSTLLAMA_PREFILL_BATCHED_HYBRID=1) after an A/B.
    let batched_hybrid = model.weights.is_hybrid()
        && matches!(state.kv_backend, KvBackend::Contiguous(_))
        && rustllama_models::accel::prefill_batched_hybrid();
    if batched_hybrid {
        while idx < prefill.len() {
            if should_stop() {
                return idx;
            }
            let chunk_end = (idx + chunk).min(prefill.len());
            let t_chunk = std::time::Instant::now();
            {
                let EngineState {
                    kv_backend,
                    delta_net_cache,
                    ..
                } = &mut *state;
                let KvBackend::Contiguous(kv) = kv_backend else {
                    unreachable!("hybrid batched prefill gate verified a contiguous backend");
                };
                let dn = delta_net_cache
                    .as_mut()
                    .expect("engine bug: hybrid model loaded without a DeltaNetCache");
                let chunk_logits =
                    model.forward_prefill_hybrid(&prefill[idx..chunk_end], idx as u32, kv, dn);
                logits.copy_from_slice(&chunk_logits);
            }
            chrome.span(&format!("prefill_chunk_{idx}_to_{chunk_end}"), t_chunk);
            idx = chunk_end;
            state.last_ids = prompt_u32[..idx].to_vec();
        }
        return idx;
    }
    while idx < prefill.len() {
        let chunk_end = (idx + chunk).min(prefill.len());
        let t_chunk = std::time::Instant::now();
        let chunk_start_idx = idx;
        let mut stopped = false;
        for i in chunk_start_idx..chunk_end {
            if should_stop() {
                stopped = true;
                idx = i;
                break;
            }
            let t_tok = std::time::Instant::now();
            {
                let EngineState {
                    kv_backend,
                    delta_net_cache,
                    ..
                } = &mut *state;
                forward_one_via_backend_hybrid(
                    model,
                    kv_backend,
                    delta_net_cache.as_mut(),
                    prefill[i],
                    i as u32,
                    logits,
                );
            }
            chrome.span(&format!("prefill_token_{i}"), t_tok);
        }
        if !stopped {
            idx = chunk_end;
        }
        chrome.span(
            &format!("prefill_chunk_{chunk_start_idx}_to_{idx}"),
            t_chunk,
        );
        // Snapshot the partial prefix so a future LCP lookup credits us
        // for the work done here. Invariant after this write:
        // `state.last_ids[i] == prompt_u32[i]` for all i < idx, and the
        // K/V cache at positions [0..idx) holds those same tokens'
        // attention state.
        state.last_ids = prompt_u32[..idx].to_vec();
        if stopped {
            return idx;
        }
    }
    idx
}

// `is_greedy` used to short-circuit the sampler when no nondeterministic
// path could be hit. After the grammar refactor every callsite routes
// through `Sampler::sample_with_grammar`, which has its own
// temperature-zero short-circuit, so this helper is no longer needed.

/// Numerically-stable softmax. Returns a new vector; the input is unchanged.
///
/// Forwards to the shared kernel-crate SIMD implementation (Tier A
/// AVX2/AVX-512 paths). Called K+1 times per speculation verify in
/// `verify_speculation_inner`; SIMD-accelerated softmax pays back
/// at vocab=128K.
pub(crate) fn softmax_to_vec(logits: &[f32]) -> Vec<f32> {
    let mut out = logits.to_vec();
    rustllama_kernels_cpu::softmax_f32_inplace(&mut out);
    out
}

/// Free-function backing for [`CpuEngine::verify_speculation`]. Takes the
/// cloned `Arc` handles directly so the caller can move them into a
/// `spawn_blocking` task without holding `&self`.
/// Which batched-speculation path to take in `verify_speculation_inner`.
/// One arm per KV dtype that has a `forward_speculation_batched_*`
/// variant. All KV dtypes now wire through batched specs — covers the
/// full FP-quant and integer-quant range:
/// - `F32`, `Nvfp4`, and `Mxfp4/6/8` are the FP-quant variants.
/// - `Q8_0`, `Q4_0`, and `Tq` are integer-quant.
#[derive(Copy, Clone, Debug)]
enum BatchedSpec {
    F32,
    Q8_0,
    Tq,
    Nvfp4,
    Q4_0,
    Mxfp4,
    Mxfp6,
    Mxfp8,
}

/// Shared speculation forward core: LCP-credited prefill of
/// `prompt[..len-1]`, then a batched forward of
/// `[prompt[len-1], candidates..]`, returning the K+1 raw-softmax
/// distributions. Leaves the engine state ADVANCED (`seq_len` at
/// `prompt.len() + candidates.len()`, `last_ids` at the last prefill
/// chunk) — the caller decides what to keep: the pure wrapper
/// [`verify_speculation_inner`] restores everything, the committing
/// wrapper [`verify_and_commit_speculation`] retains the accepted
/// prefix.
///
/// `want_dn_replay_base`: additionally snapshot the DeltaNet state
/// right after the prompt prefill — the replay base for a mid-batch
/// rejection. Only meaningful for hybrid models with candidates.
#[allow(clippy::too_many_arguments)]
fn spec_prefill_and_forward(
    model: &LlamaModel,
    state: &mut EngineState,
    prefix_cache: bool,
    prefill_chunk_size: usize,
    prompt_ids: &[i32],
    candidates: &[u32],
    want_dn_replay_base: bool,
    // When true, the returned rows are the RAW per-position logits instead
    // of softmax probabilities. Used by decision scoring
    // (`score_continuations`), which needs `logit - log_sum_exp` per token
    // to avoid f32 underflow when summing a multi-token option's logprobs.
    raw: bool,
) -> Result<(
    Vec<Vec<f32>>,
    Option<rustllama_models::llama_arch::DeltaNetSnapshot>,
)> {
    let vocab = model.cfg.vocab_size;
    let mut logits = vec![0f32; vocab];
    let prompt_u32: Vec<u32> = prompt_ids.iter().map(|&t| t as u32).collect();

    let prompt_max = prompt_ids.len().saturating_sub(1);
    let _effective = state.prepare_prefix_reuse(
        &prompt_u32,
        prefix_cache,
        PREFIX_REUSE_MIN_TOKENS,
        prompt_max,
    );

    let mut chrome = ChromeTracer::from_env();
    let (prefill, last) = prompt_ids.split_at(prompt_ids.len() - 1);
    let _ = run_chunked_prefill(
        model,
        state,
        &prompt_u32,
        prefill,
        _effective,
        prefill_chunk_size,
        &mut logits,
        &mut chrome,
        || false,
    );

    let dn_replay_base = if want_dn_replay_base {
        state.delta_net_cache.as_ref().map(|dn| dn.snapshot())
    } else {
        None
    };

    let inputs: Vec<i32> = std::iter::once(last[0])
        .chain(candidates.iter().map(|&c| c as i32))
        .collect();
    let n_pos = inputs.len();
    let mut dists: Vec<Vec<f32>> = Vec::with_capacity(n_pos);

    // Hybrid models get their own batched path regardless of KV
    // dtype: the dense BatchedSpec forwards walk `blocks`/`moe_blocks`
    // AttnBlock views and know nothing about `hybrid_layers` or the
    // DeltaNet state, so routing a hybrid model through them would be
    // wrong. `forward_speculation_batched_hybrid` reuses the chunked
    // hybrid-prefill body with an all-positions LM head.
    if model.weights.is_hybrid() {
        let EngineState {
            kv_backend,
            delta_net_cache,
            ..
        } = state;
        match (kv_backend, delta_net_cache.as_mut()) {
            (crate::kv_backend::KvBackend::Contiguous(kv), Some(dn)) => {
                let mut all_logits = vec![0f32; n_pos * vocab];
                model.forward_speculation_batched_hybrid(
                    &inputs,
                    prefill.len() as u32,
                    kv,
                    dn,
                    &mut all_logits,
                );
                for i in 0..n_pos {
                    let row = &all_logits[i * vocab..(i + 1) * vocab];
                    dists.push(if raw {
                        row.to_vec()
                    } else {
                        softmax_to_vec(row)
                    });
                }
            }
            _ => {
                return Err(CpuEngineError::Other(
                    "hybrid speculation requires the contiguous KV backend \
                     and a DeltaNet cache"
                        .to_string(),
                ));
            }
        }
        chrome.flush();
        return Ok((dists, dn_replay_base));
    }

    // Pick the batched path when the live KV backend supports one;
    // otherwise fall through to the serial loop.
    let batched_dtype = match &state.kv_backend {
        // F4: the speculation batched path has one spec for the whole
        // cache, so it needs a UNIFORM dtype. A mixed per-layer cache
        // (`uniform_dtype() == None`) falls through to the serial loop,
        // which dispatches each layer on its own `KvLayer` variant.
        crate::kv_backend::KvBackend::Contiguous(kv) => match kv.uniform_dtype() {
            Some(rustllama_models::llama_arch::KvDtype::F32) => Some(BatchedSpec::F32),
            Some(rustllama_models::llama_arch::KvDtype::Q8_0) => Some(BatchedSpec::Q8_0),
            Some(rustllama_models::llama_arch::KvDtype::Tq(_)) => Some(BatchedSpec::Tq),
            Some(rustllama_models::llama_arch::KvDtype::Nvfp4) => Some(BatchedSpec::Nvfp4),
            Some(rustllama_models::llama_arch::KvDtype::Q4_0) => Some(BatchedSpec::Q4_0),
            // MXFP KV batched-spec fast path (Wave 2): one arm per
            // element format — `mxfp_kv_prefill` inside the batched
            // forward selects the block bytes + kernels.
            Some(rustllama_models::llama_arch::KvDtype::Mxfp4) => Some(BatchedSpec::Mxfp4),
            Some(rustllama_models::llama_arch::KvDtype::Mxfp6) => Some(BatchedSpec::Mxfp6),
            Some(rustllama_models::llama_arch::KvDtype::Mxfp8) => Some(BatchedSpec::Mxfp8),
            // M6: half KV has no speculation batched spec yet → serial.
            Some(rustllama_models::llama_arch::KvDtype::F16)
            | Some(rustllama_models::llama_arch::KvDtype::Bf16) => None,
            // Mixed per-layer dtypes (F4) → serial per-layer forward.
            None => None,
        },
        // Paged backend: serial fallback. Paged is F32-only in v1
        // (per the kv_backend gate); the F32 batched path can't be
        // used directly because the cache layout differs.
        _ => None,
    };

    if let Some(spec) = batched_dtype {
        let mut all_logits = vec![0f32; n_pos * vocab];
        if let crate::kv_backend::KvBackend::Contiguous(ref mut kv) = state.kv_backend {
            match spec {
                BatchedSpec::F32 => model.forward_speculation_batched_f32(
                    &inputs,
                    prefill.len() as u32,
                    kv,
                    &mut all_logits,
                ),
                BatchedSpec::Q8_0 => model.forward_speculation_batched_q8_0(
                    &inputs,
                    prefill.len() as u32,
                    kv,
                    &mut all_logits,
                ),
                BatchedSpec::Tq => model.forward_speculation_batched_tq(
                    &inputs,
                    prefill.len() as u32,
                    kv,
                    &mut all_logits,
                ),
                BatchedSpec::Nvfp4 => model.forward_speculation_batched_nvfp4(
                    &inputs,
                    prefill.len() as u32,
                    kv,
                    &mut all_logits,
                ),
                BatchedSpec::Q4_0 => model.forward_speculation_batched_q4_0(
                    &inputs,
                    prefill.len() as u32,
                    kv,
                    &mut all_logits,
                ),
                BatchedSpec::Mxfp4 => model.forward_speculation_batched_mxfp4(
                    &inputs,
                    prefill.len() as u32,
                    kv,
                    &mut all_logits,
                ),
                BatchedSpec::Mxfp6 => model.forward_speculation_batched_mxfp6(
                    &inputs,
                    prefill.len() as u32,
                    kv,
                    &mut all_logits,
                ),
                BatchedSpec::Mxfp8 => model.forward_speculation_batched_mxfp8(
                    &inputs,
                    prefill.len() as u32,
                    kv,
                    &mut all_logits,
                ),
            }
        }
        for i in 0..n_pos {
            let row = &all_logits[i * vocab..(i + 1) * vocab];
            dists.push(if raw {
                row.to_vec()
            } else {
                softmax_to_vec(row)
            });
        }
    } else {
        for (i, &input) in inputs.iter().enumerate() {
            let pos = prefill.len() as u32 + i as u32;
            forward_one_via_backend(model, &mut state.kv_backend, input, pos, &mut logits);
            dists.push(if raw {
                logits.clone()
            } else {
                softmax_to_vec(&logits)
            });
        }
    }
    chrome.flush();
    Ok((dists, dn_replay_base))
}

fn verify_speculation_inner(
    model: Arc<LlamaModel>,
    state: Arc<Mutex<EngineState>>,
    prefix_cache: bool,
    prefill_chunk_size: usize,
    max_ctx: usize,
    prompt_ids: &[i32],
    candidates: &[u32],
) -> Result<Vec<Vec<f32>>> {
    if prompt_ids.is_empty() {
        return Ok(Vec::new());
    }
    let k = candidates.len();
    if prompt_ids.len() + k > max_ctx {
        return Err(CpuEngineError::PromptTooLong {
            prompt_len: prompt_ids.len() + k,
            max_ctx,
        });
    }
    let mut state = state.lock().expect("state lock");
    // Snapshot to restore at the end so verification is a pure read of
    // the model — no commit, no `last_ids` bump. This is the probing
    // surface (`CpuEngine::verify_speculation`, equivalence tests).
    // The speculation STREAMS do NOT use this: full restore every
    // round meant nothing retained the committed prefix, so each round
    // re-prefilled prompt+committed from a stale LCP — quadratic in
    // generated length (observed as the "spec hang" on the 27B). They
    // call [`verify_and_commit_speculation`] instead.
    let saved_seq_len = state.kv_backend.seq_len();
    let saved_last_ids = state.last_ids.clone();
    let saved_dn = state.delta_net_cache.as_ref().map(|dn| dn.snapshot());
    let result = spec_prefill_and_forward(
        &model,
        &mut state,
        prefix_cache,
        prefill_chunk_size,
        prompt_ids,
        candidates,
        false,
        false,
    );
    state.kv_backend.set_seq_len(saved_seq_len);
    state.last_ids = saved_last_ids;
    if let (Some(dn), Some(snap)) = (state.delta_net_cache.as_mut(), saved_dn.as_ref()) {
        dn.restore(snap);
    }
    result.map(|(dists, _)| dists)
}

/// Score each option as a continuation of `context_ids` via teacher forcing.
/// Returns the summed conditional logprob per option. Snapshots + restores
/// KV + DeltaNet between options so each option is scored against the same
/// clean base state and the live serving state is left untouched (pure
/// read). `prefix_cache` is forced OFF here so options never contaminate
/// each other through the prefix pool.
fn score_continuations_inner(
    model: Arc<LlamaModel>,
    state: Arc<Mutex<EngineState>>,
    prefill_chunk_size: usize,
    max_ctx: usize,
    context_ids: &[i32],
    options: &[Vec<u32>],
) -> Result<Vec<f32>> {
    if context_ids.is_empty() {
        return Err(CpuEngineError::Other(
            "score_continuations: empty context".to_string(),
        ));
    }
    let max_opt = options.iter().map(|o| o.len()).max().unwrap_or(0);
    if context_ids.len() + max_opt > max_ctx {
        return Err(CpuEngineError::PromptTooLong {
            prompt_len: context_ids.len() + max_opt,
            max_ctx,
        });
    }
    let mut state = state.lock().expect("state lock");
    let saved_seq_len = state.kv_backend.seq_len();
    let saved_last_ids = state.last_ids.clone();
    let saved_dn = state.delta_net_cache.as_ref().map(|dn| dn.snapshot());

    let mut scores = Vec::with_capacity(options.len());
    for opt in options {
        if opt.is_empty() {
            scores.push(f32::NEG_INFINITY);
            continue;
        }
        let res = spec_prefill_and_forward(
            &model,
            &mut state,
            false, // prefix_cache off — fresh prefill per option
            prefill_chunk_size,
            context_ids,
            opt,
            false, // want_dn_replay_base
            true,  // raw logits (avoid softmax underflow when summing)
        );
        // Restore to the base snapshot before the next option (and on error).
        state.kv_backend.set_seq_len(saved_seq_len);
        state.last_ids = saved_last_ids.clone();
        if let (Some(dn), Some(snap)) = (state.delta_net_cache.as_mut(), saved_dn.as_ref()) {
            dn.restore(snap);
        }
        let (rows, _) = res?;
        // rows has 1 + opt.len() positions; row[j] is the raw logits that
        // predict opt[j]. Sum log P(opt[j]) = logit[opt[j]] - logsumexp(row).
        let mut sum = 0.0f32;
        for (j, &tok) in opt.iter().enumerate() {
            let row = &rows[j];
            let lse = rustllama_kernels_cpu::log_sum_exp_f32(row);
            sum += row[tok as usize] - lse;
        }
        scores.push(sum);
    }
    Ok(scores)
}

/// Free-function backing for [`CpuEngine::sequence_logprob`]. Scores the
/// whole sequence `ids` under teacher forcing by treating `ids[0]` as a
/// one-token context and `ids[1..]` as the continuation, reusing the same
/// `spec_prefill_and_forward` raw-logits path as
/// [`score_continuations_inner`] and restoring the KV + DeltaNet state
/// afterward. Returns `(sum_logprob, per_token_logprob)` over `ids[1..]`.
fn sequence_logprob_inner(
    model: Arc<LlamaModel>,
    state: Arc<Mutex<EngineState>>,
    prefill_chunk_size: usize,
    max_ctx: usize,
    ids: &[i32],
) -> Result<(f32, Vec<f32>)> {
    if ids.len() < 2 {
        return Ok((0.0, Vec::new()));
    }
    if ids.len() > max_ctx {
        return Err(CpuEngineError::PromptTooLong {
            prompt_len: ids.len(),
            max_ctx,
        });
    }
    let context_ids = &ids[..1];
    let option: Vec<u32> = ids[1..].iter().map(|&t| t as u32).collect();

    let mut state = state.lock().expect("state lock");
    let saved_seq_len = state.kv_backend.seq_len();
    let saved_last_ids = state.last_ids.clone();
    let saved_dn = state.delta_net_cache.as_ref().map(|dn| dn.snapshot());

    let res = spec_prefill_and_forward(
        &model,
        &mut state,
        false, // prefix_cache off — fresh prefill
        prefill_chunk_size,
        context_ids,
        &option,
        false, // want_dn_replay_base
        true,  // raw logits
    );
    // Restore to the base snapshot (also on error).
    state.kv_backend.set_seq_len(saved_seq_len);
    state.last_ids = saved_last_ids.clone();
    if let (Some(dn), Some(snap)) = (state.delta_net_cache.as_mut(), saved_dn.as_ref()) {
        dn.restore(snap);
    }
    let (rows, _) = res?;
    // rows[j] is the raw logits that predict option[j] = ids[j + 1].
    let mut per_token = Vec::with_capacity(option.len());
    let mut sum = 0.0f32;
    for (j, &tok) in option.iter().enumerate() {
        let row = &rows[j];
        let lse = rustllama_kernels_cpu::log_sum_exp_f32(row);
        let lp = row[tok as usize] - lse;
        per_token.push(lp);
        sum += lp;
    }
    Ok((sum, per_token))
}

/// One full speculation round WITH state retention: forward, run
/// accept/reject under the state lock, and commit the accepted prefix
/// so the next round's prefill is an LCP no-op plus one suffix token.
///
/// Post-commit invariants (P = `prompt_ids.len()`, a = accepted):
///   - `kv seq_len == P + a`
///   - `last_ids == prompt ++ accepted`
///   - DeltaNet state == after `prompt ++ accepted` (hybrids)
/// The `replacement`/bonus token is intentionally NOT forwarded here —
/// it joins the next round's prompt tail.
///
/// Commit strategy:
///   - all K drafts accepted → the candidate KV rows and the DeltaNet
///     advance ARE the committed sequence; keep everything (zero cost).
///   - rejection at j on a dense model → the accepted rows are already
///     valid; truncating `seq_len` is the whole commit.
///   - rejection at j on a hybrid → DeltaNet can't truncate: restore
///     the post-prefill base and replay `[last_prompt, accepted..]`
///     through the batched body (≈ one forward's weight traffic).
#[allow(clippy::too_many_arguments)]
fn verify_and_commit_speculation(
    model: Arc<LlamaModel>,
    state: Arc<Mutex<EngineState>>,
    prefix_cache: bool,
    prefill_chunk_size: usize,
    max_ctx: usize,
    prompt_ids: &[i32],
    drafts: &[DraftToken],
    greedy: bool,
    typical: Option<(f32, f32)>,
    rng: &mut Rng,
    repeat_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
) -> Result<crate::speculative::SpeculationOutcome> {
    if prompt_ids.is_empty() {
        return Err(CpuEngineError::Other(
            "speculation round requires a non-empty prompt".to_string(),
        ));
    }
    let k = drafts.len();
    if prompt_ids.len() + k > max_ctx {
        return Err(CpuEngineError::PromptTooLong {
            prompt_len: prompt_ids.len() + k,
            max_ctx,
        });
    }
    let candidates: Vec<u32> = drafts.iter().map(|d| d.id).collect();
    let mut state = state.lock().expect("state lock");
    let p_len = prompt_ids.len();
    let hybrid = model.weights.is_hybrid();
    // Whether the request carries non-default sampling penalties. When it
    // does we must apply them to the TARGET logits before the accept/reject
    // verify — exactly as the classic sampler applies them before its
    // greedy short-circuit (`Sampler::sample_with_grammar`) — or greedy
    // decode with `repeat_penalty != 1.0` would diverge from classic
    // greedy. Default penalties ⇒ this is a no-op, so we keep the original
    // softmax path with zero behavior change (the common case).
    let penalize = repeat_penalty != 1.0 || frequency_penalty != 0.0 || presence_penalty != 0.0;
    let (dists, dn_base) = spec_prefill_and_forward(
        &model,
        &mut state,
        prefix_cache,
        prefill_chunk_size,
        prompt_ids,
        &candidates,
        hybrid && k > 0,
        penalize, // raw logits when penalizing (we softmax after penalizing)
    )?;
    let dists: Vec<Vec<f32>> = if penalize {
        // Row i is the model's next-token distribution CONDITIONED on
        // `[prompt, candidates[0..i]]` (the batched forward is fed
        // `[prompt.last(), candidates..]`), so the penalty history that
        // mirrors the classic path's per-step `history` is exactly
        // `prompt ++ candidates[0..i]`. In greedy accept/reject the walk
        // only reaches row i after accepting `candidates[0..i]`, so this
        // history equals the tokens the classic sampler would have seen.
        let base: Vec<u32> = prompt_ids.iter().map(|&t| t as u32).collect();
        dists
            .into_iter()
            .enumerate()
            .map(|(i, mut row)| {
                let take = i.min(candidates.len());
                let mut recent = Vec::with_capacity(base.len() + take);
                recent.extend_from_slice(&base);
                recent.extend_from_slice(&candidates[..take]);
                crate::sampling::apply_all_penalties(
                    &mut row,
                    &recent,
                    repeat_penalty,
                    frequency_penalty,
                    presence_penalty,
                );
                softmax_to_vec(&row)
            })
            .collect()
    } else {
        dists
    };
    let refs: Vec<&[f32]> = dists.iter().map(|d| d.as_slice()).collect();
    // P10a: greedy stays the exact argmax limit; otherwise the opt-in
    // typical-acceptance mode (when the request set it) else the default
    // Chen et al. rejection sampling — `typical == None` is byte-identical
    // to the pre-P10a `accept_reject` call.
    let mode = if greedy {
        AcceptMode::Greedy
    } else if let Some((epsilon, delta)) = typical {
        AcceptMode::Typical { epsilon, delta }
    } else {
        AcceptMode::Rejection
    };
    let outcome = accept_reject_dispatch(drafts, &refs, mode, rng);
    let a = outcome.accepted.len();
    if a == k {
        // Full acceptance (always the case for K=0 rounds): state is
        // already exactly the committed sequence.
        debug_assert_eq!(state.kv_backend.seq_len(), p_len + a);
    } else if hybrid {
        state.kv_backend.set_seq_len(p_len - 1);
        if let (Some(dn), Some(snap)) = (state.delta_net_cache.as_mut(), dn_base.as_ref()) {
            dn.restore(snap);
        }
        let replay: Vec<i32> = std::iter::once(prompt_ids[p_len - 1])
            .chain(outcome.accepted.iter().map(|&c| c as i32))
            .collect();
        let vocab = model.cfg.vocab_size;
        let mut replay_logits = vec![0f32; replay.len() * vocab];
        {
            let EngineState {
                kv_backend,
                delta_net_cache,
                ..
            } = &mut *state;
            match (kv_backend, delta_net_cache.as_mut()) {
                (crate::kv_backend::KvBackend::Contiguous(kv), Some(dn)) => {
                    model.forward_speculation_batched_hybrid(
                        &replay,
                        (p_len - 1) as u32,
                        kv,
                        dn,
                        &mut replay_logits,
                    );
                }
                _ => {
                    return Err(CpuEngineError::Other(
                        "hybrid speculation requires the contiguous KV backend \
                         and a DeltaNet cache"
                            .to_string(),
                    ));
                }
            }
        }
        debug_assert_eq!(state.kv_backend.seq_len(), p_len + a);
    } else {
        state.kv_backend.set_seq_len(p_len + a);
    }
    // `last_ids` must mirror exactly what the KV/DN now cover.
    let mut new_last: Vec<u32> = prompt_ids.iter().map(|&t| t as u32).collect();
    new_last.extend(outcome.accepted.iter().copied());
    state.last_ids = new_last;
    Ok(outcome)
}

/// One-time debug log for the "MTP requested but model can't do it"
/// fall-through, so an operator who flips `speculative_mtp` on for a
/// non-NextN model sees why nothing changed without spamming the log
/// on every request.
fn mtp_fallback_log_once() {
    use std::sync::Once;
    static WARN_ONCE: Once = Once::new();
    WARN_ONCE.call_once(|| {
        tracing::debug!(
            "speculative_mtp is enabled but the loaded model has no NextN head \
             (or isn't hybrid) — falling back to classic decode"
        );
    });
}

/// Result of one [`mtp_round`]: the tokens committed this round (1 on a
/// no-draft round or a rejected verify, 2 on an accepted verify), the
/// updated decode frontier, the draft to carry into the next round, and
/// per-round speculation counters for the cumulative stats.
struct MtpRoundOutcome {
    /// Tokens to emit, in order (never empty on success).
    committed: Vec<u32>,
    /// Next token to forward (already emitted / prompt content).
    next_input: i32,
    /// Position of `next_input`; the KV/DeltaNet caches cover
    /// `[0, next_pos)` on return.
    next_pos: u32,
    /// NextN draft for `next_pos + 1`, or `None` when this round did not
    /// produce one (the has-draft rounds consume the draft and leave
    /// `None`; the no-draft round produces a fresh one).
    pending: Option<DraftToken>,
    /// Drafts proposed this round (0 or 1) — for `add_speculation`.
    drafted: usize,
    /// Drafts accepted this round (0 or 1) — for `add_speculation`.
    accepted: usize,
}

/// One round of MTP / NextN self-speculation, run under the engine
/// state lock on the SYCL worker thread. Two modes, chosen by whether a
/// draft is carried in:
///
///   - **no-draft** (`pending == None`, or no room to speculate): a
///     single [`LlamaModel::forward_one_hybrid_with_nextn_logits`] call.
///     The main head commits the next token (same argmax/multinomial
///     policy the ngram path uses via an empty-draft `accept_reject`),
///     and the NextN head drafts the token two positions ahead for the
///     NEXT round. KV/DeltaNet advance by exactly one position.
///
///   - **has-draft** (`pending == Some`, room available): a batched
///     [`LlamaModel::forward_speculation_batched_hybrid`] over
///     `[next_input, draft]` verifies the carried draft. On ACCEPT the
///     draft (position `base_pos+1`) plus the bonus (position
///     `base_pos+2`) are committed — two tokens for one batched forward,
///     the MTP win. On REJECT the draft position is rewound exactly like
///     [`verify_and_commit_speculation`]'s hybrid path (truncate KV,
///     restore the DeltaNet snapshot taken before the forward, replay
///     `next_input`) and only the true main token commits.
///
/// Position bookkeeping matches the model-side contract:
/// `forward_one_hybrid_with_nextn_logits` sets `kv.seq_len = base_pos+1`
/// and advances the DeltaNet cache by one; the batched forward sets
/// `kv.seq_len = base_pos + tokens.len()` and advances the DeltaNet
/// cache by that many. This keeps the KV + recurrent state consistent
/// with the classic path token-for-token.
#[allow(clippy::too_many_arguments)]
fn mtp_round(
    model: &LlamaModel,
    state: &Arc<Mutex<EngineState>>,
    next_input: i32,
    next_pos: u32,
    pending: Option<DraftToken>,
    allow_spec: bool,
    greedy: bool,
    rng: &mut Rng,
    vocab: usize,
    repeat: f32,
    frequency: f32,
    presence: f32,
) -> Result<MtpRoundOutcome> {
    // Non-default sampling penalties must be applied to the TARGET logits
    // before the accept/reject verify — mirroring the classic sampler,
    // which penalizes before its greedy short-circuit — so greedy decode
    // with `repeat_penalty != 1.0` stays token-for-token identical to the
    // classic path. Default penalties ⇒ this stays a no-op. The penalty
    // history for a row is the full left context of the token that row
    // predicts, which equals `state.last_ids` (the KV/DeltaNet coverage,
    // == positions `[0, base_pos)`) plus the tokens forwarded ahead of
    // that row this round (`next_input`, then the draft).
    let penalize = repeat != 1.0 || frequency != 0.0 || presence != 0.0;
    let mut state = state.lock().expect("state lock");
    let base_pos = next_pos;

    // MTP requires the contiguous KV backend + a DeltaNet cache. The
    // dispatch layer already gated on hybrid + NextN head; this guards a
    // paged-KV misconfiguration with a clean error instead of a panic
    // deep in the forward.
    let contiguous_ok =
        matches!(state.kv_backend, KvBackend::Contiguous(_)) && state.delta_net_cache.is_some();
    if !contiguous_ok {
        return Err(CpuEngineError::Other(
            "MTP self-speculation requires the contiguous KV backend and a \
             DeltaNet cache (hybrid model)"
                .to_string(),
        ));
    }

    if let (true, Some(draft)) = (allow_spec, pending) {
        // ---- has-draft round: batched verify [next_input, draft] ----
        // DeltaNet base snapshot BEFORE the forward (covers
        // `[0, base_pos)`) so a rejection can restore + replay, exactly
        // like `verify_and_commit_speculation`.
        let dn_base = state.delta_net_cache.as_ref().map(|dn| dn.snapshot());
        let mut two = vec![0f32; 2 * vocab];
        {
            let EngineState {
                kv_backend,
                delta_net_cache,
                ..
            } = &mut *state;
            match (kv_backend, delta_net_cache.as_mut()) {
                (KvBackend::Contiguous(kv), Some(dn)) => {
                    model.forward_speculation_batched_hybrid(
                        &[next_input, draft.id as i32],
                        base_pos,
                        kv,
                        dn,
                        &mut two,
                    );
                }
                _ => unreachable!("contiguous_ok checked above"),
            }
        }
        // Verify the single draft against row0 (the true next-token
        // distribution). When penalties are active they are applied to the
        // target logits before the softmax (so the greedy verify matches
        // the classic penalized sampler); otherwise this is the raw-softmax
        // path shared with the ngram verify.
        if penalize {
            // row0 predicts the token at base_pos+1: its left context is
            // last_ids (== [0, base_pos)) plus the forwarded `next_input`.
            let mut recent = state.last_ids.clone();
            recent.push(next_input as u32);
            crate::sampling::apply_all_penalties(
                &mut two[0..vocab],
                &recent,
                repeat,
                frequency,
                presence,
            );
            // row1 (bonus) predicts the token at base_pos+2: extend the
            // context by the forwarded draft.
            recent.push(draft.id);
            crate::sampling::apply_all_penalties(
                &mut two[vocab..2 * vocab],
                &recent,
                repeat,
                frequency,
                presence,
            );
        }
        let row0 = softmax_to_vec(&two[0..vocab]);
        let row1 = softmax_to_vec(&two[vocab..2 * vocab]);
        let target: [&[f32]; 2] = [row0.as_slice(), row1.as_slice()];
        let drafts = [draft];
        let outcome = if greedy {
            accept_reject_greedy(&drafts, &target)
        } else {
            accept_reject(&drafts, &target, rng)
        };
        if outcome.accepted.len() == 1 {
            // ACCEPT: next_input (base_pos) + draft (base_pos+1) are both
            // valid in KV/DeltaNet; the bonus (`replacement`) is the
            // fresh token at base_pos+2 and is NOT forwarded — it seeds
            // the next round. seq_len is already base_pos+2, no rewind.
            debug_assert_eq!(state.kv_backend.seq_len(), base_pos as usize + 2);
            // Extend `last_ids` by the two FORWARDED tokens (next_input +
            // draft) so it stays == the KV/DeltaNet coverage. The bonus
            // (`replacement`) is not forwarded, so it is not appended.
            state.last_ids.push(next_input as u32);
            state.last_ids.push(draft.id);
            debug_assert_eq!(state.last_ids.len(), state.kv_backend.seq_len());
            Ok(MtpRoundOutcome {
                committed: vec![draft.id, outcome.replacement],
                next_input: outcome.replacement as i32,
                next_pos: base_pos + 2,
                pending: None,
                drafted: 1,
                accepted: 1,
            })
        } else {
            // REJECT: the draft row (base_pos+1) is wrong. Keep
            // next_input's forward (base_pos) and drop the draft: rewind
            // seq_len to base_pos, restore the DeltaNet base, and replay
            // [next_input] so KV + DeltaNet both cover `[0, base_pos+1)`.
            // Mirrors `verify_and_commit_speculation`'s hybrid reject.
            state.kv_backend.set_seq_len(base_pos as usize);
            if let (Some(dn), Some(snap)) = (state.delta_net_cache.as_mut(), dn_base.as_ref()) {
                dn.restore(snap);
            }
            let mut one = vec![0f32; vocab];
            {
                let EngineState {
                    kv_backend,
                    delta_net_cache,
                    ..
                } = &mut *state;
                match (kv_backend, delta_net_cache.as_mut()) {
                    (KvBackend::Contiguous(kv), Some(dn)) => {
                        model.forward_speculation_batched_hybrid(
                            &[next_input],
                            base_pos,
                            kv,
                            dn,
                            &mut one,
                        );
                    }
                    _ => unreachable!("contiguous_ok checked above"),
                }
            }
            debug_assert_eq!(state.kv_backend.seq_len(), base_pos as usize + 1);
            // Only next_input was (re)forwarded; the true replacement token
            // seeds the next round unforwarded.
            state.last_ids.push(next_input as u32);
            debug_assert_eq!(state.last_ids.len(), state.kv_backend.seq_len());
            Ok(MtpRoundOutcome {
                committed: vec![outcome.replacement],
                next_input: outcome.replacement as i32,
                next_pos: base_pos + 1,
                pending: None,
                drafted: 1,
                accepted: 0,
            })
        }
    } else {
        // ---- no-draft round: one NextN forward. Commit the main token
        //      and draft the +2 token for the next round. ----
        let mut main_logits = vec![0f32; vocab];
        let mut nextn_logits = vec![0f32; vocab];
        {
            let EngineState {
                kv_backend,
                delta_net_cache,
                ..
            } = &mut *state;
            match (kv_backend, delta_net_cache.as_mut()) {
                (KvBackend::Contiguous(kv), Some(dn)) => {
                    // Conditioning caveat: the NextN head wants the TRUE
                    // token at base_pos+1 as `next_token_id`, but that is
                    // exactly what this same forward's main head predicts
                    // — unknowable pre-call with the single-shot NextN
                    // primitive. We pass `next_input` (a repeat prior).
                    // The draft is VERIFIED next round, so a poor guess
                    // only lowers acceptance, never correctness.
                    model.forward_one_hybrid_with_nextn_logits(
                        next_input,
                        next_input,
                        base_pos,
                        kv,
                        dn,
                        &mut main_logits,
                        &mut nextn_logits,
                    );
                }
                _ => unreachable!("contiguous_ok checked above"),
            }
        }
        debug_assert_eq!(state.kv_backend.seq_len(), base_pos as usize + 1);
        // Commit the main token with the SAME policy the has-draft round
        // uses: an empty-draft `accept_reject` == "sample one token from
        // this position" (argmax when greedy, else multinomial over the
        // softmax), so greedy MTP output is token-for-token identical
        // to classic greedy — including under penalties, which are applied
        // to the target logits here before the softmax when non-default.
        if penalize {
            // The main token is at base_pos+1: its left context is last_ids
            // (== [0, base_pos)) plus the forwarded `next_input`.
            let mut recent = state.last_ids.clone();
            recent.push(next_input as u32);
            crate::sampling::apply_all_penalties(
                main_logits.as_mut_slice(),
                &recent,
                repeat,
                frequency,
                presence,
            );
        }
        let row0 = softmax_to_vec(&main_logits);
        let target: [&[f32]; 1] = [row0.as_slice()];
        let no_drafts: [DraftToken; 0] = [];
        let m_out = if greedy {
            accept_reject_greedy(&no_drafts, &target)
        } else {
            accept_reject(&no_drafts, &target, rng)
        };
        let m = m_out.replacement;
        // Draft the +2 token from the NextN head (argmax + softmax q),
        // reusing the tested `MtpDrafter` argmax path. `None` when the
        // logit row is degenerate/empty.
        let pending = MtpDrafter::default()
            .propose(&[nextn_logits])
            .into_iter()
            .next();
        // Only next_input was forwarded this round; the sampled main token
        // `m` seeds the next round unforwarded.
        state.last_ids.push(next_input as u32);
        debug_assert_eq!(state.last_ids.len(), state.kv_backend.seq_len());
        Ok(MtpRoundOutcome {
            committed: vec![m],
            next_input: m as i32,
            next_pos: base_pos + 1,
            pending,
            drafted: 0,
            accepted: 0,
        })
    }
}

/// One TREE MTP self-speculation round (Feature 4), the branching generalization
/// of [`mtp_round`] for NON-hybrid models carrying MTP heads. Reuses the
/// parity-guarded tree primitives end-to-end: one tip forward produces the main
/// (target_root) distribution + per-head drafts; [`build_mtp_tree`] shapes a
/// draft tree; [`LlamaModel::forward_tree_verify_f32`] scores every node in ONE
/// forward (each node attends only its ancestors); [`tree_accept_reject`] walks
/// the best accepted path; and [`KvCache::compact_tokens`] moves the accepted
/// path's scattered K/V into contiguous committed slots — so an accepted run of
/// `a` tokens costs ONE verify forward, no replay.
///
/// Requires the contiguous F32 KV backend (the tree verify + compaction are
/// F32-contiguous); returns an error otherwise so the caller falls back. Default
/// sampling penalties are a no-op; non-default repeat/frequency/presence
/// penalties are applied BOTH to the root verified distribution AND, per node,
/// to the drafted tree (`build_mtp_tree`), so the proposed branches respect the
/// same penalties the sampler does (neutral penalties leave the tree
/// byte-identical).
#[allow(clippy::too_many_arguments)]
/// Shared DENSE tree-spec tail (Feature 4 + P10b): given an already-built draft
/// tree `nodes` + the (penalty-applied) root distribution `main_dist`, run the
/// batched ancestor-masked verify, walk the accepted path, and commit it into
/// the contiguous F32 KV via [`KvCache::compact_tokens`]. Factored so BOTH the
/// MTP self-spec round ([`tree_mtp_round`]) and the EAGLE round
/// ([`tree_eagle_round`]) reuse ONE verify/accept/commit — EAGLE only replaces
/// the DRAFTING that produces `nodes`. The caller holds the state lock and has
/// already run the tip forward (K/V at `base_pos`, `kv.seq_len == base_pos+1`).
#[allow(clippy::too_many_arguments)]
fn dense_tree_verify_commit(
    model: &LlamaModel,
    state: &mut EngineState,
    next_input: i32,
    base_pos: usize,
    main_dist: &[f32],
    nodes: &[crate::speculative::TreeNode],
    max_ctx: usize,
    greedy: bool,
    rng: &mut Rng,
    vocab: usize,
) -> MtpRoundOutcome {
    use crate::speculative::{tree_accept_reject, tree_accept_reject_greedy, tree_depths};
    let fits = !nodes.is_empty() && base_pos + 1 + nodes.len() <= max_ctx;
    if !fits {
        // No tree (no drafts / overflow) ⇒ commit one token from target_root,
        // exactly the empty-draft accept/reject (== classic sample).
        let drafts: [DraftToken; 0] = [];
        let target: [&[f32]; 1] = [main_dist];
        let out = if greedy {
            accept_reject_greedy(&drafts, &target)
        } else {
            accept_reject(&drafts, &target, rng)
        };
        state.kv_backend.set_seq_len(base_pos + 1);
        state.last_ids.push(next_input as u32);
        return MtpRoundOutcome {
            committed: vec![out.replacement],
            next_input: out.replacement as i32,
            next_pos: base_pos as u32 + 1,
            pending: None,
            drafted: 0,
            accepted: 0,
        };
    }

    // --- Tree verify forward: per-node target_next. Writes each node's K/V to
    //     slot `base_pos + 1 + node_index`, attending prefix + ancestors. ---
    let node_tokens: Vec<i32> = nodes.iter().map(|n| n.id as i32).collect();
    let depths = tree_depths(nodes);
    let parent: Vec<i32> = nodes
        .iter()
        .map(|n| n.parent.map(|p| p as i32).unwrap_or(-1))
        .collect();
    let mut tree_logits = vec![0f32; nodes.len() * vocab];
    {
        let KvBackend::Contiguous(kv) = &mut state.kv_backend else {
            unreachable!()
        };
        model.forward_tree_verify_f32(
            &node_tokens,
            base_pos as u32 + 1,
            &depths,
            &parent,
            kv,
            &mut tree_logits,
        );
    }
    let next_dists: Vec<Vec<f32>> = (0..nodes.len())
        .map(|i| softmax_to_vec(&tree_logits[i * vocab..(i + 1) * vocab]))
        .collect();
    let next_refs: Vec<&[f32]> = next_dists.iter().map(|d| d.as_slice()).collect();

    // --- Walk the best accepted path. ---
    let outcome = if greedy {
        tree_accept_reject_greedy(nodes, main_dist, &next_refs)
    } else {
        tree_accept_reject(nodes, main_dist, &next_refs, rng)
    };
    let a = outcome.accepted.len();

    // --- Commit: compact the accepted path's scattered K/V into contiguous
    //     slots, then truncate. One verify forward paid for `a` tokens. ---
    {
        let KvBackend::Contiguous(kv) = &mut state.kv_backend else {
            unreachable!()
        };
        kv.compact_tokens(base_pos + 1, &outcome.accepted_nodes);
    }
    state.kv_backend.set_seq_len(base_pos + 1 + a);
    // last_ids mirrors the committed KV coverage: context + tip + accepted path.
    state.last_ids.push(next_input as u32);
    state.last_ids.extend(outcome.accepted.iter().copied());
    debug_assert_eq!(state.last_ids.len(), state.kv_backend.seq_len());

    let mut committed = outcome.accepted;
    committed.push(outcome.replacement);
    MtpRoundOutcome {
        committed,
        next_input: outcome.replacement as i32,
        next_pos: base_pos as u32 + 1 + a as u32,
        pending: None,
        drafted: nodes.len(),
        accepted: a,
    }
}

fn tree_mtp_round(
    model: &LlamaModel,
    state: &Arc<Mutex<EngineState>>,
    next_input: i32,
    next_pos: u32,
    widths: &[u32],
    max_ctx: usize,
    greedy: bool,
    rng: &mut Rng,
    vocab: usize,
    repeat: f32,
    frequency: f32,
    presence: f32,
) -> Result<MtpRoundOutcome> {
    use crate::speculative::build_mtp_tree;
    let penalize = repeat != 1.0 || frequency != 0.0 || presence != 0.0;
    let mut state = state.lock().expect("state lock");
    let base_pos = next_pos as usize;

    if !matches!(state.kv_backend, KvBackend::Contiguous(_)) {
        return Err(CpuEngineError::Other(
            "tree speculation requires the contiguous KV backend".to_string(),
        ));
    }

    // --- Tip forward: main (target_root) + per-MTP-head drafts. Writes the tip
    //     token's K/V at `base_pos`; kv.seq_len = base_pos + 1. ---
    let n_mtp = model
        .weights
        .mtp_heads
        .as_ref()
        .map(|h| h.len())
        .unwrap_or(0);
    let mut main_logits = vec![0f32; vocab];
    let mut head_logits: Vec<Vec<f32>> = vec![vec![0f32; vocab]; n_mtp];
    {
        let KvBackend::Contiguous(kv) = &mut state.kv_backend else {
            unreachable!()
        };
        model.forward_one_with_mtp_logits(
            next_input,
            next_pos,
            kv,
            &mut main_logits,
            &mut head_logits,
        );
    }
    debug_assert_eq!(state.kv_backend.seq_len(), base_pos + 1);

    // Penalty history for the first speculative token = `last_ids` (== [0,
    // base_pos)) plus the just-forwarded tip. Built ONCE and shared by the root
    // target penalty (below) and the PER-NODE draft penalties threaded into
    // `build_mtp_tree` — so both the verified distribution AND the proposed tree
    // respect the same repeat/frequency/presence penalties the classic sampler
    // does. Default penalties ⇒ empty context + a no-op everywhere (the tree is
    // byte-identical to the pre-penalty path; the common decode stays
    // allocation-free).
    let recent: Vec<u32> = if penalize {
        let mut r = state.last_ids.clone();
        r.push(next_input as u32);
        r
    } else {
        Vec::new()
    };
    if penalize {
        crate::sampling::apply_all_penalties(
            &mut main_logits,
            &recent,
            repeat,
            frequency,
            presence,
        );
    }
    let main_dist = softmax_to_vec(&main_logits);

    // --- Build the draft tree; the shared dense tail verifies + commits it
    //     (bails to a single-token sample if empty / would overflow). Per-node
    //     penalties steer the drafted branches to match the sampler (neutral ⇒
    //     unchanged tree). ---
    let widths_usz: Vec<usize> = widths.iter().map(|&w| (w as usize).max(1)).collect();
    let nodes = build_mtp_tree(
        &head_logits,
        &widths_usz,
        &recent,
        repeat,
        frequency,
        presence,
    );
    Ok(dense_tree_verify_commit(
        model,
        &mut *state,
        next_input,
        base_pos,
        &main_dist,
        &nodes,
        max_ctx,
        greedy,
        rng,
        vocab,
    ))
}

/// P10b — one EAGLE tree self-speculation round (the dense/MoE counterpart of
/// [`tree_mtp_round`]). Identical plumbing EXCEPT the drafting: instead of the
/// base model's per-position MTP sidecars, a SINGLE trained EAGLE head
/// ([`LlamaModel::forward_eagle_head`]) is run autoregressively from the base's
/// captured top hidden + the tip token's embedding to produce the per-depth
/// draft logit rows. Those rows feed the SAME [`build_mtp_tree`] +
/// [`dense_tree_verify_commit`] (verify → accept → compact-commit) the MTP
/// round uses, so EAGLE only replaces the DRAFT source.
///
/// Requires the contiguous F32 KV backend + a dense/MoE model (the tip forward
/// [`LlamaModel::forward_one_with_hidden`] doesn't cover hybrid / MLA — the
/// dispatch gates those out); returns an error otherwise so the caller falls
/// back. Default sampling penalties are a no-op; non-default
/// repeat/frequency/presence penalties are applied BOTH to the root verified
/// distribution AND, per node, to the drafted tree (`build_mtp_tree`) — the
/// same contract [`tree_mtp_round`] documents.
#[allow(clippy::too_many_arguments)]
fn tree_eagle_round(
    model: &LlamaModel,
    state: &Arc<Mutex<EngineState>>,
    eagle: &rustllama_models::llama_arch::EagleHead,
    next_input: i32,
    next_pos: u32,
    widths: &[u32],
    max_ctx: usize,
    greedy: bool,
    rng: &mut Rng,
    vocab: usize,
    repeat: f32,
    frequency: f32,
    presence: f32,
) -> Result<MtpRoundOutcome> {
    use crate::speculative::build_mtp_tree;
    let penalize = repeat != 1.0 || frequency != 0.0 || presence != 0.0;
    let mut state = state.lock().expect("state lock");
    let base_pos = next_pos as usize;

    if !matches!(state.kv_backend, KvBackend::Contiguous(_)) {
        return Err(CpuEngineError::Other(
            "EAGLE tree speculation requires the contiguous KV backend".to_string(),
        ));
    }

    // --- Tip forward: main (target_root) distribution + the base hidden the
    //     EAGLE head fuses. Writes the tip token's K/V at `base_pos`;
    //     kv.seq_len = base_pos + 1. ---
    let mut main_logits = vec![0f32; vocab];
    let mut base_hidden: Vec<f32> = Vec::new();
    {
        let KvBackend::Contiguous(kv) = &mut state.kv_backend else {
            unreachable!()
        };
        model.forward_one_with_hidden(next_input, next_pos, kv, &mut main_logits, &mut base_hidden);
    }
    debug_assert_eq!(state.kv_backend.seq_len(), base_pos + 1);

    // Penalty history for the first speculative token = `last_ids` (== [0,
    // base_pos)) plus the just-forwarded tip. Shared by the root target penalty
    // and the per-node draft penalties (same contract as `tree_mtp_round`).
    let recent: Vec<u32> = if penalize {
        let mut r = state.last_ids.clone();
        r.push(next_input as u32);
        r
    } else {
        Vec::new()
    };
    if penalize {
        crate::sampling::apply_all_penalties(
            &mut main_logits,
            &recent,
            repeat,
            frequency,
            presence,
        );
    }
    let main_dist = softmax_to_vec(&main_logits);

    // --- Draft: run the EAGLE head autoregressively to `widths.len()` depths,
    //     producing one full-vocab logit row per depth — the drop-in for the
    //     MTP sidecar logits. Then shape the tree exactly like the MTP round. ---
    let widths_usz: Vec<usize> = widths.iter().map(|&w| (w as usize).max(1)).collect();
    let head_logits = model.forward_eagle_head(eagle, &base_hidden, next_input, widths_usz.len());
    let nodes = build_mtp_tree(
        &head_logits,
        &widths_usz,
        &recent,
        repeat,
        frequency,
        presence,
    );
    Ok(dense_tree_verify_commit(
        model,
        &mut *state,
        next_input,
        base_pos,
        &main_dist,
        &nodes,
        max_ctx,
        greedy,
        rng,
        vocab,
    ))
}

/// One TREE MTP self-speculation round for a HYBRID model (Feature 4, hybrid
/// path) — the recurrent-architecture counterpart of [`tree_mtp_round`].
///
/// A hybrid model carries DeltaNet/SSM layers whose state is SEQUENTIAL, so the
/// dense tree path's single batched ancestor-masked verify + `compact_tokens`
/// commit does not apply. This round instead:
///
///   1. **Tip forward** — [`LlamaModel::forward_one_hybrid_with_draft_heads`]
///      advances the KV + DeltaNet caches by `next_input` and returns the main
///      distribution plus one logit row per draft head (DeepSeek `mtp_heads`,
///      else the qwen35moe `nextn_head`).
///   2. **Tree build** — [`build_mtp_tree`] shapes a caterpillar from the head
///      logits (width 1 everywhere ⇒ the linear MTP chain, pinned by
///      `mtp_tree_width_one_is_a_chain`).
///   3. **Recurrent tree verify** — [`LlamaModel::forward_tree_verify_hybrid_f32`]
///      walks the tree node-by-node, snapshotting/restoring the DeltaNet state so
///      each branch evolves its own recurrence from the shared ancestor (the
///      "caterpillar" walk). Attention layers stay correct because each node is
///      stepped at `start_pos + depth` so its ancestor K/V form a contiguous
///      causal window.
///   4. **Walk + commit** — [`tree_accept_reject`] picks the accepted path, then
///      we RESTORE the tip recurrent snapshot and REPLAY the accepted tokens via
///      [`LlamaModel::forward_speculation_batched_hybrid`]. Unlike dense KV, a
///      recurrence can't be compacted in place, so the accepted path is re-walked
///      from the tip state — exactly the reject-replay pattern
///      [`verify_and_commit_speculation`] uses on hybrids. The verify forward
///      scores the whole tree in one DFS; the replay pays only for the `a`
///      ACCEPTED tokens.
///
/// Requires the contiguous F32 KV backend + a DeltaNet cache (the hybrid path
/// forces contiguous F32 KV at load); returns an error otherwise so the caller
/// falls back. Default sampling penalties are a no-op; non-default
/// repeat/frequency/presence penalties are applied BOTH to the root verified
/// distribution AND, per node, to the drafted tree (`build_mtp_tree`), so the
/// proposed branches respect the same penalties the sampler does (neutral
/// penalties leave the tree byte-identical) — the same contract the dense
/// [`tree_mtp_round`] documents.
#[allow(clippy::too_many_arguments)]
fn tree_mtp_round_hybrid(
    model: &LlamaModel,
    state: &Arc<Mutex<EngineState>>,
    next_input: i32,
    next_pos: u32,
    widths: &[u32],
    max_ctx: usize,
    greedy: bool,
    rng: &mut Rng,
    vocab: usize,
    repeat: f32,
    frequency: f32,
    presence: f32,
) -> Result<MtpRoundOutcome> {
    use crate::speculative::{
        build_mtp_tree, tree_accept_reject, tree_accept_reject_greedy, tree_depths,
    };
    let penalize = repeat != 1.0 || frequency != 0.0 || presence != 0.0;
    let mut state = state.lock().expect("state lock");
    let base_pos = next_pos as usize;

    // Tree verify + the DeltaNet replay both require the contiguous F32 KV
    // backend AND a DeltaNet cache. The dispatch already gated on hybrid + a
    // draft head; this guards a paged-KV misconfiguration with a clean error.
    let contiguous_ok =
        matches!(state.kv_backend, KvBackend::Contiguous(_)) && state.delta_net_cache.is_some();
    if !contiguous_ok {
        return Err(CpuEngineError::Other(
            "hybrid tree speculation requires the contiguous KV backend and a \
             DeltaNet cache"
                .to_string(),
        ));
    }

    // --- Tip forward: main distribution + per-head drafts. Advances KV +
    //     DeltaNet by the tip token; kv.seq_len = base_pos + 1. ---
    let mut main_logits = vec![0f32; vocab];
    let head_logits: Vec<Vec<f32>> = {
        let EngineState {
            kv_backend,
            delta_net_cache,
            ..
        } = &mut *state;
        match (kv_backend, delta_net_cache.as_mut()) {
            (KvBackend::Contiguous(kv), Some(dn)) => model.forward_one_hybrid_with_draft_heads(
                next_input,
                next_pos,
                kv,
                dn,
                &mut main_logits,
            ),
            _ => unreachable!("contiguous_ok checked above"),
        }
    };
    debug_assert_eq!(state.kv_backend.seq_len(), base_pos + 1);

    // Penalty history for the first speculative token = `last_ids` (== [0,
    // base_pos)) plus the just-forwarded tip. Built ONCE and shared by the root
    // target penalty (below) and the PER-NODE draft penalties threaded into
    // `build_mtp_tree` (same contract as the dense `tree_mtp_round`). Default
    // penalties ⇒ empty context + a no-op everywhere (byte-identical tree).
    let recent: Vec<u32> = if penalize {
        let mut r = state.last_ids.clone();
        r.push(next_input as u32);
        r
    } else {
        Vec::new()
    };
    if penalize {
        crate::sampling::apply_all_penalties(
            &mut main_logits,
            &recent,
            repeat,
            frequency,
            presence,
        );
    }
    let main_dist = softmax_to_vec(&main_logits);

    // --- Build the caterpillar; bail to a single-token sample if it is empty
    //     (no draft head / degenerate logits) or would overflow the context.
    //     Nodes occupy depth slots `base_pos + 1 + depth`, so the deepest slot
    //     touched is `base_pos + 1 + max_depth`. Per-node penalties steer the
    //     drafted branches to match the sampler (neutral ⇒ unchanged tree). ---
    let widths_usz: Vec<usize> = widths.iter().map(|&w| (w as usize).max(1)).collect();
    let nodes = build_mtp_tree(
        &head_logits,
        &widths_usz,
        &recent,
        repeat,
        frequency,
        presence,
    );
    let depths = tree_depths(&nodes);
    let max_depth = depths.iter().copied().max().unwrap_or(0) as usize;
    let fits = !nodes.is_empty() && base_pos + 1 + max_depth + 1 <= max_ctx;
    if !fits {
        // No tree ⇒ commit one token from target_root, exactly the empty-draft
        // accept/reject (== classic sample). KV/DeltaNet already cover the tip.
        let drafts: [DraftToken; 0] = [];
        let target: [&[f32]; 1] = [main_dist.as_slice()];
        let out = if greedy {
            accept_reject_greedy(&drafts, &target)
        } else {
            accept_reject(&drafts, &target, rng)
        };
        state.kv_backend.set_seq_len(base_pos + 1);
        state.last_ids.push(next_input as u32);
        debug_assert_eq!(state.last_ids.len(), state.kv_backend.seq_len());
        return Ok(MtpRoundOutcome {
            committed: vec![out.replacement],
            next_input: out.replacement as i32,
            next_pos: base_pos as u32 + 1,
            pending: None,
            drafted: 0,
            accepted: 0,
        });
    }

    // Snapshot the TIP recurrent state. It is BOTH the verify walk's branch root
    // AND the replay base — the walk overwrites shared depth slots and the live
    // DeltaNet state, so this snapshot is how the commit re-establishes a clean
    // starting point. Reset discipline: a fresh snapshot each round means a later
    // round / model reload can never reuse a stale per-node recurrent state.
    let dn_tip = state.delta_net_cache.as_ref().map(|dn| dn.snapshot());

    // --- Recurrent tree verify forward: per-node logits via the DFS walk. ---
    let node_tokens: Vec<i32> = nodes.iter().map(|n| n.id as i32).collect();
    let parent: Vec<i32> = nodes
        .iter()
        .map(|n| n.parent.map(|p| p as i32).unwrap_or(-1))
        .collect();
    let mut tree_logits = vec![0f32; nodes.len() * vocab];
    {
        let EngineState {
            kv_backend,
            delta_net_cache,
            ..
        } = &mut *state;
        match (kv_backend, delta_net_cache.as_mut()) {
            (KvBackend::Contiguous(kv), Some(dn)) => {
                model.forward_tree_verify_hybrid_f32(
                    &node_tokens,
                    next_pos + 1,
                    &depths,
                    &parent,
                    kv,
                    dn,
                    &mut tree_logits,
                );
            }
            _ => unreachable!("contiguous_ok checked above"),
        }
    }
    let next_dists: Vec<Vec<f32>> = (0..nodes.len())
        .map(|i| softmax_to_vec(&tree_logits[i * vocab..(i + 1) * vocab]))
        .collect();
    let next_refs: Vec<&[f32]> = next_dists.iter().map(|d| d.as_slice()).collect();

    // --- Walk the best accepted path. ---
    let outcome = if greedy {
        tree_accept_reject_greedy(&nodes, &main_dist, &next_refs)
    } else {
        tree_accept_reject(&nodes, &main_dist, &next_refs, rng)
    };
    let a = outcome.accepted.len();

    // --- Commit: restore the tip recurrent state and REPLAY the accepted path
    //     so KV (contiguous slots base_pos+1..) + DeltaNet both cover exactly
    //     `tip ++ accepted`. A recurrence can't be compacted like dense KV, so
    //     the accepted tokens are re-walked from the tip state (one forward's
    //     weight traffic over `a` tokens). ---
    if let (Some(dn), Some(snap)) = (state.delta_net_cache.as_mut(), dn_tip.as_ref()) {
        dn.restore(snap);
    }
    state.kv_backend.set_seq_len(base_pos + 1);
    if a > 0 {
        let replay: Vec<i32> = outcome.accepted.iter().map(|&c| c as i32).collect();
        let mut replay_logits = vec![0f32; replay.len() * vocab];
        let EngineState {
            kv_backend,
            delta_net_cache,
            ..
        } = &mut *state;
        match (kv_backend, delta_net_cache.as_mut()) {
            (KvBackend::Contiguous(kv), Some(dn)) => {
                model.forward_speculation_batched_hybrid(
                    &replay,
                    (base_pos + 1) as u32,
                    kv,
                    dn,
                    &mut replay_logits,
                );
            }
            _ => unreachable!("contiguous_ok checked above"),
        }
    }
    state.kv_backend.set_seq_len(base_pos + 1 + a);
    // last_ids mirrors the committed KV coverage: context + tip + accepted path.
    state.last_ids.push(next_input as u32);
    state.last_ids.extend(outcome.accepted.iter().copied());
    debug_assert_eq!(state.last_ids.len(), state.kv_backend.seq_len());

    let mut committed = outcome.accepted;
    committed.push(outcome.replacement);
    Ok(MtpRoundOutcome {
        committed,
        next_input: outcome.replacement as i32,
        next_pos: base_pos as u32 + 1 + a as u32,
        pending: None,
        drafted: nodes.len(),
        accepted: a,
    })
}

// ----- streaming Engine impl -----

impl Engine for CpuEngine {
    fn metrics(&self) -> Metrics {
        use std::sync::atomic::Ordering;
        // `try_lock`, never `lock`: the state mutex is held for the
        // ENTIRE duration of a generation, and this method is polled
        // ~1 Hz by the GUI through `/v1/metrics` on a tokio worker.
        // Blocking here progressively parked reactor threads for
        // minutes at a time. On contention we serve the last
        // published numbers instead — bounded staleness is exactly
        // right for a status poll.
        let (context_used, paged_total_pages, paged_free_pages) = match self.state.try_lock() {
            Ok(s) => {
                let seq_len = s.kv_backend.seq_len() as u32;
                let (total, free) = s.kv_backend.paged_pool_stats().unwrap_or((0, 0));
                self.metrics_cache.seq_len.store(seq_len, Ordering::Relaxed);
                self.metrics_cache
                    .paged_total
                    .store(total, Ordering::Relaxed);
                self.metrics_cache.paged_free.store(free, Ordering::Relaxed);
                (seq_len, total, free)
            }
            Err(_) => (
                self.metrics_cache.seq_len.load(Ordering::Relaxed),
                self.metrics_cache.paged_total.load(Ordering::Relaxed),
                self.metrics_cache.paged_free.load(Ordering::Relaxed),
            ),
        };
        // CpuEngine is single-flight at the engine layer (concurrency
        // comes from forks). One "slot" is active iff the KV is
        // non-empty.
        let paged_active_slots = if context_used > 0 && paged_total_pages > 0 {
            1
        } else {
            0
        };
        Metrics {
            tokens_per_second: 0.0,
            context_used,
            vram_estimate_mb: 0,
            ram_estimate_mb: 0,
            paged_total_pages,
            paged_free_pages,
            paged_active_slots,
        }
    }

    fn n_ctx(&self) -> u32 {
        self.max_ctx as u32
    }

    fn vocab_size(&self) -> usize {
        self.model.cfg.vocab_size
    }

    fn last_request_stats_snapshot(&self) -> Option<RequestStats> {
        // Forward to the inherent method that already holds the
        // value; `Some(...)` is meaningful even before any
        // request runs (RequestStats::default() = all zeros),
        // but distinguishing "no request yet" from "zero stats"
        // matters less than the server's contract that the
        // snapshot fields are `Option<f64>` — pre-first-request
        // we still send 0s rather than missing.
        Some(self.last_request_stats())
    }

    fn cumulative_stats_snapshot(&self) -> Option<crate::CumulativeStatsSnapshot> {
        Some(self.cumulative_request_stats())
    }

    // Tool-calling support — expose the real tokenizer + context window
    // generically so the server's tools handlers work off `&dyn Engine`
    // (not the concrete `CpuEngine`). Owned `Arc` clone is cheap.
    fn tokenizer_arc(&self) -> Option<Arc<Tokenizer>> {
        self.tokenizer.clone()
    }
    fn max_ctx(&self) -> usize {
        self.max_ctx
    }

    // P11 — resident multi-LoRA (trait surface for the server's `Arc<dyn
    // Engine>`; delegates to the inherent methods, mapping the error type).
    fn attach_lora_adapter(&self, name: &str, path: &std::path::Path) -> EngineResult<usize> {
        CpuEngine::attach_lora_adapter(self, name, path)
            .map_err(|e| crate::EngineError::Engine(e.to_string()))
    }
    fn lora_adapter_names(&self) -> Vec<String> {
        CpuEngine::lora_adapter_names(self)
    }
    fn has_lora_adapter(&self, name: &str) -> bool {
        CpuEngine::has_lora_adapter(self, name)
    }

    fn tokenize(&self, text: &str) -> EngineResult<Vec<u32>> {
        let tokenizer = self
            .tokenizer
            .as_deref()
            .ok_or_else(|| crate::EngineError::Unimplemented("tokenize requires a tokenizer"))?;
        Ok(tokenizer.encode(text, false)?)
    }

    fn supports_vision(&self) -> bool {
        // True once `load_with_mmproj` has loaded a vision tower
        // alongside the text decoder. The chat path still rejects
        // image-bearing requests via `reject_if_has_images` until
        // V-6b-3 lands the splice into the transformer body.
        self.vision.is_some()
    }

    fn chat(&self, msgs: &[ChatMessage], s: &SamplingParams) -> EngineResult<TokenStream> {
        let has_images = msgs.iter().any(|m| !m.images.is_empty());
        // V-6a / V-6b-3d routing:
        // - text-only request OR engine has no vision tower → plain text path
        // - text-only engine + image-bearing request          → 400 (gate)
        // - vision-loaded engine + image-bearing request     → VLM splice path
        if has_images && self.vision.is_none() {
            return Err(crate::EngineError::VisionNotSupported);
        }
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| crate::EngineError::Unimplemented("chat requires a tokenizer"))?
            .clone();
        // D1 placeholder injection: the tokenizer context is a flat
        // string per message, so a Qwen3-VL template's content-block
        // image macros can never fire. Instead, the ENGINE prepends
        // one wrapper (`<|vision_start|><|image_pad|><|vision_end|>`
        // for the merger family, bare placeholder otherwise) per
        // attached image to that message's content before render.
        // Idempotent: a message that already carries the wrapper
        // (client-side injection) is left untouched.
        let injected: Vec<String>;
        let prompt = {
            let contents: Vec<&str> =
                if let (true, Some(wrapper)) = (has_images, self.image_wrapper.as_deref()) {
                    injected = msgs
                        .iter()
                        .map(|m| {
                            if m.images.is_empty() || m.content.contains(wrapper) {
                                m.content.clone()
                            } else {
                                let mut c = wrapper.repeat(m.images.len());
                                c.push('\n');
                                c.push_str(&m.content);
                                c
                            }
                        })
                        .collect();
                    injected.iter().map(|s| s.as_str()).collect()
                } else {
                    msgs.iter().map(|m| m.content.as_str()).collect()
                };
            let tok_msgs: Vec<TokChat<'_>> = msgs
                .iter()
                .zip(contents.iter())
                .map(|(m, c)| TokChat {
                    role: &m.role,
                    content: c,
                })
                .collect();
            tokenizer.render_chat(&tok_msgs, true)?
        };
        if has_images {
            // Gather every attached image, in message order. The VLM
            // prefill expects one entry per `<image>` placeholder; the
            // engine's image_token_id is what
            // `tokenizer.render_chat` will have produced for the
            // placeholder positions in the prompt.
            let images: Vec<Vec<u8>> = msgs.iter().flat_map(|m| m.images.iter().cloned()).collect();
            return self.chat_vlm_stream(prompt, images, s.clone());
        }
        // N-gram speculative decoding, when enabled for this engine.
        // Skipped for grammar-constrained requests: the speculative path
        // verifies on raw-softmax target probs and cannot honor a grammar
        // mask, so those keep the classic sampler.
        // Draft-model speculation, when paired. Grammar requests keep
        // the classic sampler for the same raw-softmax reason as the
        // n-gram path.
        // P2: a contiguous SWA ring cache can't serve the batched/compacting
        // speculative verify — fall back to classic single-token decode.
        // P11: an adapted request also skips every speculative path — those run
        // through `RunOnWorker` closures that don't carry the LoRA scope, so
        // routing them to the LoRA-aware `spawn_stream` keeps the adapter
        // applied (a resident adapter beats the spec throughput win here, and no
        // request is ever silently served the base).
        let allow_spec = !self.swa_ring_decode() && s.lora_adapter.is_none();
        if let Some((draft, k)) = self.draft_spec.as_ref() {
            if s.grammar.is_none() && allow_spec {
                return self.speculate(&prompt, draft.clone(), *k, s);
            }
        }
        // P10b — EAGLE trained draft head: takes PRECEDENCE over the in-base
        // MTP/NextN + n-gram drafters (`eagle > MTP > n-gram > none`). A
        // loaded head on a supported (dense/MoE, non-SWA-ring) model drafts a
        // tree autoregressively; the base verifies via the shared tree-spec
        // stack. Grammar-free only, same raw-softmax caveat as the others.
        if let Some(head) = self.eagle_head.clone() {
            if s.grammar.is_none() && allow_spec && self.model_supports_eagle() {
                return self.speculate_eagle_stream(prompt, s.clone(), head);
            }
        }
        // MTP / NextN self-speculation: takes precedence over n-gram
        // (the model's own NextN head beats prompt-lookup on free
        // prose). Grammar-free only, same raw-softmax caveat as the
        // other speculative paths. Silent fall-through to n-gram/classic
        // on models without a NextN head.
        // Tree-structured MTP spec (Feature 4) takes precedence over linear MTP
        // / n-gram when enabled on a non-hybrid MTP model.
        if let Some(widths) = self.tree_spec.clone() {
            if s.grammar.is_none() && allow_spec && self.model_supports_tree() {
                return self.speculate_mtp_stream(prompt, s.clone(), Some(widths));
            }
        }
        if self.mtp_spec && s.grammar.is_none() && allow_spec {
            if self.model_supports_mtp() {
                return self.speculate_mtp_stream(prompt, s.clone(), None);
            }
            mtp_fallback_log_once();
        }
        // F10 — lookahead (Jacobi n-gram-pool) spec: just above plain n-gram
        // (`… > MTP > lookahead > n-gram`). Grammar-free only, same raw-softmax
        // caveat as the other spec paths.
        if let Some(cfg) = self.lookahead_spec {
            if s.grammar.is_none() && allow_spec {
                return self.speculate_lookahead_stream(prompt, cfg, s.clone());
            }
        }
        if let Some(cfg) = self.ngram_spec {
            if s.grammar.is_none() && allow_spec {
                return self.speculate_ngram_stream(prompt, cfg, s.clone());
            }
        }
        self.spawn_stream(prompt, s.clone())
    }

    fn generate(&self, prompt: &str, s: &SamplingParams) -> EngineResult<TokenStream> {
        // Without a tokenizer, fall through and return a clear error from the
        // spawn path so the call still produces a Stream rather than panic.
        // MTP / NextN self-speculation takes precedence over n-gram; a
        // no-op fall-through on non-MTP models. Grammar-free only.
        // P11: an adapted request skips spec (see `chat`) → LoRA-aware
        // `spawn_stream` instead, so the adapter is never silently dropped.
        let allow_spec = !self.swa_ring_decode() && s.lora_adapter.is_none();
        // P10b — EAGLE head first (`eagle > MTP > n-gram > none`).
        if let Some(head) = self.eagle_head.clone() {
            if s.grammar.is_none() && allow_spec && self.model_supports_eagle() {
                return self.speculate_eagle_stream(prompt.to_string(), s.clone(), head);
            }
        }
        if let Some(widths) = self.tree_spec.clone() {
            if s.grammar.is_none() && allow_spec && self.model_supports_tree() {
                return self.speculate_mtp_stream(prompt.to_string(), s.clone(), Some(widths));
            }
        }
        if self.mtp_spec && s.grammar.is_none() && allow_spec {
            if self.model_supports_mtp() {
                return self.speculate_mtp_stream(prompt.to_string(), s.clone(), None);
            }
            mtp_fallback_log_once();
        }
        // F10 — lookahead spec, just above plain n-gram (see `generate_stream`).
        if let Some(cfg) = self.lookahead_spec {
            if s.grammar.is_none() && allow_spec {
                return self.speculate_lookahead_stream(prompt.to_string(), cfg, s.clone());
            }
        }
        if let Some(cfg) = self.ngram_spec {
            if s.grammar.is_none() && allow_spec {
                return self.speculate_ngram_stream(prompt.to_string(), cfg, s.clone());
            }
        }
        self.spawn_stream(prompt.to_string(), s.clone())
    }

    /// Speculative-decoding driver — multi-round.
    ///
    /// Each round:
    ///   1. Drive the `draft` engine for K tokens with `logprobs = Some(0)`
    ///      so each emitted [`Token`] carries the chosen-id's logprob.
    ///      `q_i = exp(logprob_i)` — the draft's raw-softmax probability of
    ///      the token it sampled at position i.
    ///   2. Run [`Self::verify_speculation`] over `prompt + candidates`,
    ///      capturing the target's K+1 raw-softmax distributions.
    ///   3. Hand `(drafts, target_dists)` to
    ///      [`crate::speculative::accept_reject`]. The accepted prefix +
    ///      replacement (residual sample on reject, or bonus from the K+1th
    ///      target distribution on full-accept) become the committed
    ///      output.
    ///   4. Emit committed tokens through a `TokenStream`.
    ///   5. Stop when total committed reaches `s.max_tokens` or when a
    ///      committed token equals the model's EOS id.
    ///   6. Otherwise, append committed token ids + their decoded text
    ///      to the prompt and loop.
    ///
    /// KV-rewind on reject: `verify_speculation_inner` snapshot+restores
    /// the target's KV state around the verify-forward, so each round's
    /// verify call leaves the target's cache exactly where the
    /// committed-tokens prefix ends. No explicit rewind needed at this
    /// layer — the verify primitive provides it.
    ///
    /// Sampling: this path runs on **raw-softmax** probabilities — neither
    /// side's temperature / top-k / top-p is applied to p or q. The
    /// spec-decode math needs p and q on the same distribution, and the
    /// simplest way to guarantee that across two arbitrary `Engine` impls
    /// (without a coordinated "speculation temperature" knob) is to keep
    /// both raw. A v1.x follow-up adds a shared knob and threads it through.
    ///
    /// K=0 short-circuits to plain `Engine::generate` (no speculation).
    fn speculate(
        &self,
        prompt: &str,
        draft: std::sync::Arc<dyn Engine>,
        k: u32,
        s: &SamplingParams,
    ) -> EngineResult<TokenStream> {
        if k == 0 {
            // K=0 degrades to plain generation; reuse the existing
            // single-engine path.
            return self.generate(prompt, s);
        }
        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| crate::EngineError::Unimplemented("speculate requires a tokenizer"))?;
        let add_bos = tokenizer.add_bos_token();
        let initial_prompt_ids: Vec<i32> = tokenizer
            .encode(prompt, add_bos)?
            .into_iter()
            .map(|t| t as i32)
            .collect();
        // Hand off to the shared multi-round driver. The ID-level
        // `/v1/completions` entry (`speculate_draft_stream_from_ids`) shares it.
        self.speculate_from_parts(initial_prompt_ids, prompt.to_string(), draft, k, s.clone())
    }
}

impl CpuEngine {
    /// ID-level entry for DRAFT-MODEL speculative decoding — the P10a
    /// `/v1/completions` route. The endpoint tokenizes its own prompt (no
    /// BOS; FIM specials for `suffix`), so re-encoding text through
    /// [`Engine::speculate`] would change the prompt. The target verify is
    /// already ID-based; only the draft is driven from text, seeded by
    /// decoding `initial_prompt_ids` (a mismatched draft tokenization only
    /// lowers acceptance, never correctness — the target's ids are
    /// authoritative). Mirrors `speculate_ngram_stream_from_ids`.
    pub fn speculate_draft_stream_from_ids(
        &self,
        initial_prompt_ids: Vec<i32>,
        draft: std::sync::Arc<dyn Engine>,
        k: u32,
        s: SamplingParams,
    ) -> EngineResult<TokenStream> {
        let k = k.max(1);
        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| crate::EngineError::Unimplemented("speculate requires a tokenizer"))?;
        let ids_u32: Vec<u32> = initial_prompt_ids.iter().map(|&t| t as u32).collect();
        let initial_prompt_text = tokenizer.decode(&ids_u32, true).unwrap_or_default();
        self.speculate_from_parts(initial_prompt_ids, initial_prompt_text, draft, k, s)
    }

    /// Shared multi-round draft-model speculation driver behind both the
    /// text [`Engine::speculate`] and the ID-level
    /// [`Self::speculate_draft_stream_from_ids`]. `initial_prompt_text`
    /// drives the draft; `initial_prompt_ids` drives the authoritative
    /// target verify.
    fn speculate_from_parts(
        &self,
        initial_prompt_ids: Vec<i32>,
        initial_prompt_text: String,
        draft: std::sync::Arc<dyn Engine>,
        k: u32,
        s: SamplingParams,
    ) -> EngineResult<TokenStream> {
        if initial_prompt_ids.is_empty() {
            let stream = async_stream::stream! {
                if false { yield Ok(Token { id: 0, text: String::new(), logprobs: None }); }
            };
            return Ok(Box::pin(stream));
        }
        let tokenizer = self
            .tokenizer
            .clone()
            .ok_or_else(|| crate::EngineError::Unimplemented("speculate requires a tokenizer"))?;
        // Per-round draft sampling: K tokens with logprobs for `q`.
        let mut draft_sampling = s.clone();
        draft_sampling.max_tokens = k;
        draft_sampling.logprobs = Some(0);

        let candidate_cap = k as usize;
        let seed = s.seed;
        let max_tokens = s.max_tokens as usize;
        let model_eos = self.model.cfg.eos_token_id;
        let stop_strings = s.stop.clone();
        let greedy_spec = s.temperature <= 0.0;
        // P10a: resolve the opt-in typical-acceptance knob once; `None`
        // keeps the default Chen et al. rejection sampling (byte-identical).
        let typical = if s.spec_typical_accept {
            Some((s.spec_typical_epsilon, s.spec_typical_delta))
        } else {
            None
        };

        // Snapshot the cloned Arc handles. Moving these into the async
        // stream sidesteps `&self` borrows for the whole stream lifetime.
        let model = self.model.clone();
        let state = self.state.clone();
        let prefix_cache = self.prefix_cache;
        let prefill_chunk_size = self.prefill_chunk_size;
        let max_ctx = self.max_ctx;
        let cumulative_stats = self.cumulative_stats.clone();
        // Publish live throughput to last_stats/EMA so the GUI tok/s
        // isn't blank under speculative decode (the non-spec path does
        // this in drive_generation; the spec streams did not).
        let last_stats = self.last_stats.clone();
        let ema_tok_s_bits = Arc::clone(&self.ema_tok_s_bits);
        // Route the per-round verify-forward onto the SYCL worker thread
        // (which owns the persistent SYCL stream + warmed USM weight
        // cache) instead of a bare tokio blocking thread. Also capture
        // this request's dispatch state so speculation honors
        // `[inference].n_gpu_layers` / placement / cpu-force exactly like
        // the non-speculative streaming path.
        let worker = self.sycl_worker.submitter();
        let flash_attention = self.flash_attention;
        let n_gpu_layers = self.n_gpu_layers;
        let cpu_force_patterns = self.cpu_force_patterns.clone();

        let stream = async_stream::stream! {
            use futures::StreamExt;
            // Decode-throughput clock for the GUI tok/s (published per token below).
            let decode_start = std::time::Instant::now();
            // Moved into each round's spawn_blocking (accept/reject
            // now runs inside the state lock) and moved back out.
            let mut rng_slot = Some(Rng::from_seed(seed));
            // Per-round prompt grows by committed tokens between rounds.
            let mut current_prompt_text = initial_prompt_text;
            let mut current_prompt_ids: Vec<i32> = initial_prompt_ids;
            let mut total_emitted: usize = 0;
            // Buffered emitted text for multi-token stop-string
            // matching — same scheme as `drive_generation`.
            let mut emitted_text = String::new();
            // UTF-8-safe incremental detok for the yielded deltas (kept
            // separate from `committed_text`, which feeds prompt continuation).
            let mut utf8 = Utf8Stream::default();

            // Install this request's dispatch state on the worker thread
            // once (it persists there until the next request overwrites
            // it) and make sure the USM attention context exists, so the
            // verify-forwards below dispatch matvec + flash attention to
            // the GPU. Idempotent — warmup usually built the context
            // already; this is defensive against an absent/failed warmup.
            {
                let model_install = model.clone();
                let flash = flash_attention;
                let ngl = n_gpu_layers;
                let cpu_force = cpu_force_patterns.clone();
                let _ = worker
                    .run_blocking(move || {
                        rustllama_models::accel::set_flash_attention(flash);
                        rustllama_models::accel::set_n_gpu_layers(ngl);
                        rustllama_models::accel::set_cpu_force_patterns(cpu_force);
                        let cfg = &model_install.cfg;
                        let head_dim = if cfg.head_dim > 0 {
                            cfg.head_dim
                        } else {
                            cfg.d_model / cfg.n_heads.max(1)
                        };
                        let _ = rustllama_models::accel::prepare_usm_context(
                            cfg.n_layers as u32,
                            cfg.n_heads as u32,
                            cfg.n_kv_heads as u32,
                            head_dim as u32,
                            max_ctx as u32,
                        );
                    })
                    .await;
            }

            loop {
                if total_emitted >= max_tokens {
                    return;
                }
                // Allow the LAST round to ask the draft for fewer than K
                // candidates so total_emitted can't overshoot max_tokens
                // by more than 1 (the replacement is always emitted).
                let remaining = max_tokens.saturating_sub(total_emitted);
                let round_k = candidate_cap.min(remaining.max(1));
                let mut round_sampling = draft_sampling.clone();
                round_sampling.max_tokens = round_k as u32;

                // ---- Step 1: draft proposes K candidates ----
                let draft_stream = match draft.generate(&current_prompt_text, &round_sampling) {
                    Ok(s) => s,
                    Err(e) => { yield Err(e); return; }
                };
                let mut drafts: Vec<DraftToken> = Vec::with_capacity(round_k);
                {
                    let mut ds = draft_stream;
                    while let Some(t) = ds.next().await {
                        match t {
                            Ok(tok) => {
                                let q = match tok.logprobs.as_ref() {
                                    Some(lp) => lp.logprob.exp().clamp(1e-30, 1.0),
                                    None => 1e-3,
                                };
                                drafts.push(DraftToken { id: tok.id, q });
                                if drafts.len() == round_k { break; }
                            }
                            Err(e) => { yield Err(e); return; }
                        }
                    }
                }
                if drafts.is_empty() {
                    // Draft produced nothing (EOS at first step or
                    // engine stopped) — end speculation.
                    return;
                }
                // ---- Step 2+3: verify + accept + COMMIT. The round
                //      retains the accepted prefix in KV/DN/`last_ids`
                //      so the next round's prefill is an LCP no-op —
                //      see `verify_and_commit_speculation` (the old
                //      pure-read verify re-prefilled everything every
                //      round: quadratic, the "spec hang"). ----
                let prompt_ids_for_verify = current_prompt_ids.clone();
                let drafts_for_verify = drafts.clone();
                let model_c = model.clone();
                let state_c = state.clone();
                let mut rng_c = rng_slot.take().expect("rng slot");
                let joined = worker
                    .run_blocking(move || {
                        // Two-engine speculation stays on raw softmax by
                        // design (p and q must share a distribution across
                        // two arbitrary engines — see this method's doc), so
                        // no penalties are applied here (no-op args).
                        let out = verify_and_commit_speculation(
                            model_c, state_c, prefix_cache, prefill_chunk_size, max_ctx,
                            &prompt_ids_for_verify, &drafts_for_verify, greedy_spec,
                            typical,
                            &mut rng_c, 1.0, 0.0, 0.0,
                        );
                        (out, rng_c)
                    })
                    .await;
                let outcome = match joined {
                    Ok((Ok(o), r)) => {
                        rng_slot = Some(r);
                        o
                    }
                    Ok((Err(e), r)) => {
                        rng_slot = Some(r);
                        let _ = &rng_slot;
                        yield Err(crate::EngineError::Engine(format!(
                            "verify_speculation: {e}"
                        )));
                        return;
                    }
                    Err(recv_err) => {
                        yield Err(crate::EngineError::Engine(format!(
                            "verify worker recv: {recv_err}"
                        )));
                        return;
                    }
                };
                cumulative_stats
                    .add_speculation(drafts.len() as u64, outcome.accepted.len() as u64);
                let mut committed: Vec<u32> = outcome.accepted;
                committed.push(outcome.replacement);

                // ---- Step 4: yield committed; check stop/EOS/max ----
                let mut committed_text = String::new();
                for &id in &committed {
                    let text = tokenizer.decode_single(id, true).unwrap_or_else(|_| format!("<{id}>"));
                    committed_text.push_str(&text);
                    emitted_text.push_str(&text);
                    // Stop semantics mirror `drive_generation`: the
                    // token whose text completes the stop string is
                    // still emitted, then the stream ends.
                    let hit_stop = stop_strings
                        .iter()
                        .any(|st| !st.is_empty() && emitted_text.contains(st.as_str()));
                    // UTF-8-safe delta for the client (holds back a multi-byte
                    // char split across tokens instead of emitting `��`).
                    let out_text = utf8.next_text(id, |ids| {
                        tokenizer.decode(ids, true).unwrap_or_default()
                    });
                    yield Ok(Token { id, text: out_text, logprobs: None });
                    total_emitted += 1;
                    // Publish running decode throughput so the GUI tok/s
                    // isn't blank during speculative decode. Per-token so
                    // every stream-exit path carries the final value.
                    {
                        let elapsed_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
                        let s = RequestStats {
                            prefill_ms: 0.0,
                            decode_ms: elapsed_ms,
                            tokens_prefilled: 0,
                            cache_hit_tokens: 0,
                            tokens_generated: total_emitted as u32,
                            tool_call_limit_hit: false,
                        };
                        update_ema_tok_s(&ema_tok_s_bits, &s);
                        *last_stats.lock().expect("last_stats") = s;
                    }
                    if hit_stop {
                        return;
                    }
                    // EOS in committed tail ends speculation immediately.
                    if Some(id) == model_eos {
                        return;
                    }
                    if total_emitted >= max_tokens {
                        return;
                    }
                }
                // Extend prompt for the next round. Both the tokenized
                // form (for verify) and the text form (for draft) are
                // updated in lockstep so the two engines see the same
                // committed prefix.
                current_prompt_ids.extend(committed.iter().map(|&id| id as i32));
                current_prompt_text.push_str(&committed_text);
                // Continue the outer round loop.
            }
        };
        Ok(Box::pin(stream))
    }

    /// N-gram (prompt-lookup) speculative-decoding driver — no draft
    /// model. Each round:
    ///   1. [`NgramDrafter::propose`] scans the running token history for
    ///      a repeat of the trailing `n_match` tokens and proposes up to
    ///      `n_draft` tokens that followed the earlier occurrence.
    ///   2. [`verify_speculation_inner`] forwards the target over
    ///      `prompt[-1] + candidates`, returning K+1 raw-softmax
    ///      distributions while snapshot/restoring its own KV — that is
    ///      the rewind-on-reject piece, identical to [`Engine::speculate`].
    ///   3. [`accept_reject`] commits the accepted prefix + one
    ///      replacement (residual sample on reject) or bonus token.
    ///
    /// When the drafter finds no match it proposes nothing; the round
    /// degenerates to a single-token sample (K=0 ⇒ `accept_reject`
    /// returns the lone bonus from the one target distribution). So this
    /// path is never slower than classic decode by more than the
    /// microsecond-scale history scan, and is a clear win whenever recent
    /// context repeats (code, structured text, copy-and-edit).
    ///
    /// SAMPLING SEMANTICS — IMPORTANT: verification samples from the
    /// target's softmax with the request's `temperature` / `top_k` /
    /// `top_p` / grammar NOT applied on this path, but repeat / frequency /
    /// presence PENALTIES ARE applied to the target logits before the
    /// verify (so greedy decode with `repeat_penalty != 1.0` stays
    /// token-for-token identical to the classic path). Callers route here
    /// only for grammar-free requests (see `chat` / `generate`); the
    /// temperature caveat is the documented trade-off of the v1
    /// speculative path and a shared "speculation temperature" knob is the
    /// follow-up.
    pub fn speculate_ngram_stream(
        &self,
        prompt: String,
        cfg: NgramDrafterConfig,
        s: SamplingParams,
    ) -> EngineResult<TokenStream> {
        let tokenizer = self.tokenizer.clone().ok_or_else(|| {
            crate::EngineError::Unimplemented("speculate_ngram requires a tokenizer")
        })?;
        let add_bos = tokenizer.add_bos_token();
        let initial_prompt_ids: Vec<i32> = tokenizer
            .encode(&prompt, add_bos)?
            .into_iter()
            .map(|t| t as i32)
            .collect();
        self.speculate_ngram_stream_from_ids(initial_prompt_ids, cfg, s)
    }

    /// ID-level entry for n-gram speculation. The server's
    /// `/v1/completions` path routes here directly: that endpoint
    /// tokenizes its own prompt (no BOS, FIM specials for `suffix`
    /// requests), so re-encoding text through
    /// [`Self::speculate_ngram_stream`] would change the prompt.
    /// Everything downstream of the encode is ID-based anyway — the
    /// drafter scans ids, verify forwards ids, and only the emit
    /// loop touches the tokenizer (per-token detok).
    pub fn speculate_ngram_stream_from_ids(
        &self,
        initial_prompt_ids: Vec<i32>,
        cfg: NgramDrafterConfig,
        s: SamplingParams,
    ) -> EngineResult<TokenStream> {
        let tokenizer = self.tokenizer.clone().ok_or_else(|| {
            crate::EngineError::Unimplemented("speculate_ngram requires a tokenizer")
        })?;
        if initial_prompt_ids.is_empty() {
            let stream = async_stream::stream! {
                if false { yield Ok(Token { id: 0, text: String::new(), logprobs: None }); }
            };
            return Ok(Box::pin(stream));
        }

        let drafter = NgramDrafter::new(cfg);
        let candidate_cap = cfg.n_draft.max(1);
        let seed = s.seed;
        let max_tokens = s.max_tokens as usize;
        let model_eos = self.model.cfg.eos_token_id;
        let stop_strings = s.stop.clone();
        let greedy_spec = s.temperature <= 0.0;
        // P10a: opt-in typical acceptance; `None` keeps the default Chen et
        // al. rejection sampling (byte-identical to the pre-P10a path).
        let typical = if s.spec_typical_accept {
            Some((s.spec_typical_epsilon, s.spec_typical_delta))
        } else {
            None
        };
        // Sampling penalties applied to the target logits before each
        // round's accept/reject verify (keeps greedy-with-penalties
        // token-for-token identical to the classic path).
        let repeat_pen = s.repeat_penalty;
        let freq_pen = s.frequency_penalty;
        let pres_pen = s.presence_penalty;

        // Snapshot Arc handles so the async stream owns them without
        // borrowing `&self` for its whole lifetime (mirrors `speculate`).
        let model = self.model.clone();
        let state = self.state.clone();
        let prefix_cache = self.prefix_cache;
        let prefill_chunk_size = self.prefill_chunk_size;
        let max_ctx = self.max_ctx;
        let cumulative_stats = self.cumulative_stats.clone();
        // Publish live throughput to last_stats/EMA so the GUI tok/s
        // isn't blank under n-gram speculative decode (mirrors
        // drive_generation, which the spec streams previously skipped).
        let last_stats = self.last_stats.clone();
        let ema_tok_s_bits = Arc::clone(&self.ema_tok_s_bits);
        // Route each round's verify-forward onto the SYCL worker thread
        // (persistent SYCL stream + warmed USM cache) + capture this
        // request's dispatch state, exactly like `speculate`. This is the
        // production decode path (n-gram spec is the default): without it
        // every verify ran on a tokio blocking thread with no SYCL guard
        // installed → 100% CPU dispatch even with a GPU present.
        let worker = self.sycl_worker.submitter();
        let flash_attention = self.flash_attention;
        let n_gpu_layers = self.n_gpu_layers;
        let cpu_force_patterns = self.cpu_force_patterns.clone();

        let stream = async_stream::stream! {
            // Moved into each round's spawn_blocking (accept/reject
            // now runs inside the state lock) and moved back out.
            let mut rng_slot = Some(Rng::from_seed(seed));
            let mut current_prompt_ids: Vec<i32> = initial_prompt_ids;
            let mut total_emitted: usize = 0;
            // Buffered emitted text for multi-token stop-string
            // matching — same scheme as `drive_generation`.
            let mut emitted_text = String::new();
            // UTF-8-safe incremental detok for the yielded deltas.
            let mut utf8 = Utf8Stream::default();
            // Decode-throughput clock for the GUI tok/s (published per token below).
            let decode_start = std::time::Instant::now();

            // Install this request's dispatch state on the worker thread
            // + ensure the USM context exists (see `speculate` for the
            // rationale). Idempotent; runs once before the round loop.
            {
                let model_install = model.clone();
                let flash = flash_attention;
                let ngl = n_gpu_layers;
                let cpu_force = cpu_force_patterns.clone();
                let _ = worker
                    .run_blocking(move || {
                        rustllama_models::accel::set_flash_attention(flash);
                        rustllama_models::accel::set_n_gpu_layers(ngl);
                        rustllama_models::accel::set_cpu_force_patterns(cpu_force);
                        let cfg = &model_install.cfg;
                        let head_dim = if cfg.head_dim > 0 {
                            cfg.head_dim
                        } else {
                            cfg.d_model / cfg.n_heads.max(1)
                        };
                        let _ = rustllama_models::accel::prepare_usm_context(
                            cfg.n_layers as u32,
                            cfg.n_heads as u32,
                            cfg.n_kv_heads as u32,
                            head_dim as u32,
                            max_ctx as u32,
                        );
                    })
                    .await;
            }

            loop {
                if total_emitted >= max_tokens {
                    return;
                }
                let remaining = max_tokens.saturating_sub(total_emitted);

                // ---- Step 1: n-gram drafter proposes candidates ----
                let hist_u32: Vec<u32> =
                    current_prompt_ids.iter().map(|&t| t as u32).collect();
                let mut drafts = drafter.propose(&hist_u32);
                // Cap candidates so committed (= accepted + 1 replacement)
                // can't overshoot max_tokens. `remaining >= 1` here.
                let round_cap = candidate_cap.min(remaining.saturating_sub(1));
                if drafts.len() > round_cap {
                    drafts.truncate(round_cap);
                }
                // ---- Step 2+3: verify + accept + COMMIT (retains the
                //      accepted prefix; next round's prefill is an LCP
                //      no-op plus one suffix token — see
                //      `verify_and_commit_speculation`) ----
                let prompt_ids_for_verify = current_prompt_ids.clone();
                let drafts_for_verify = drafts.clone();
                let model_c = model.clone();
                let state_c = state.clone();
                let mut rng_c = rng_slot.take().expect("rng slot");
                let joined = worker
                    .run_blocking(move || {
                        let out = verify_and_commit_speculation(
                            model_c, state_c, prefix_cache, prefill_chunk_size, max_ctx,
                            &prompt_ids_for_verify, &drafts_for_verify, greedy_spec,
                            typical, &mut rng_c, repeat_pen, freq_pen, pres_pen,
                        );
                        (out, rng_c)
                    })
                    .await;
                let outcome = match joined {
                    Ok((Ok(o), r)) => {
                        rng_slot = Some(r);
                        o
                    }
                    Ok((Err(e), r)) => {
                        rng_slot = Some(r);
                        let _ = &rng_slot;
                        yield Err(crate::EngineError::Engine(format!(
                            "verify_speculation (ngram): {e}"
                        )));
                        return;
                    }
                    Err(recv_err) => {
                        yield Err(crate::EngineError::Engine(format!(
                            "verify worker recv: {recv_err}"
                        )));
                        return;
                    }
                };
                cumulative_stats
                    .add_speculation(drafts.len() as u64, outcome.accepted.len() as u64);
                let mut committed: Vec<u32> = outcome.accepted;
                committed.push(outcome.replacement);

                // ---- Step 4: emit committed; check stop/EOS/max ----
                for &id in &committed {
                    // UTF-8-safe delta: holds back a multi-byte char split
                    // across tokens instead of emitting `��`.
                    let text = utf8.next_text(id, |ids| {
                        tokenizer.decode(ids, true).unwrap_or_default()
                    });
                    emitted_text.push_str(&text);
                    // Stop semantics mirror `drive_generation`: the
                    // token whose text completes the stop string is
                    // still emitted, then the stream ends.
                    let hit_stop = stop_strings
                        .iter()
                        .any(|st| !st.is_empty() && emitted_text.contains(st.as_str()));
                    yield Ok(Token { id, text, logprobs: None });
                    total_emitted += 1;
                    // Publish running decode throughput so the GUI tok/s
                    // isn't blank during speculative decode. Per-token so
                    // every stream-exit path carries the final value.
                    {
                        let elapsed_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
                        let s = RequestStats {
                            prefill_ms: 0.0,
                            decode_ms: elapsed_ms,
                            tokens_prefilled: 0,
                            cache_hit_tokens: 0,
                            tokens_generated: total_emitted as u32,
                            tool_call_limit_hit: false,
                        };
                        update_ema_tok_s(&ema_tok_s_bits, &s);
                        *last_stats.lock().expect("last_stats") = s;
                    }
                    if hit_stop {
                        return;
                    }
                    if Some(id) == model_eos {
                        return;
                    }
                    if total_emitted >= max_tokens {
                        return;
                    }
                }
                // Extend history with the committed tokens so the next
                // round's drafter + verify see the full prefix.
                current_prompt_ids.extend(committed.iter().map(|&id| id as i32));
            }
        };
        Ok(Box::pin(stream))
    }

    /// F10 — text entry for lookahead (Jacobi n-gram-pool) speculation.
    /// Tokenizes `prompt` (BOS policy) and forwards to
    /// [`Self::speculate_lookahead_stream_from_ids`]. Mirrors
    /// [`Self::speculate_ngram_stream`].
    pub fn speculate_lookahead_stream(
        &self,
        prompt: String,
        cfg: LookaheadDrafterConfig,
        s: SamplingParams,
    ) -> EngineResult<TokenStream> {
        let tokenizer = self.tokenizer.clone().ok_or_else(|| {
            crate::EngineError::Unimplemented("speculate_lookahead requires a tokenizer")
        })?;
        let add_bos = tokenizer.add_bos_token();
        let initial_prompt_ids: Vec<i32> = tokenizer
            .encode(&prompt, add_bos)?
            .into_iter()
            .map(|t| t as i32)
            .collect();
        self.speculate_lookahead_stream_from_ids(initial_prompt_ids, cfg, s)
    }

    /// F10 — ID-level lookahead / Jacobi speculative-decode driver.
    ///
    /// Structurally identical to [`Self::speculate_ngram_stream_from_ids`]
    /// (same async stream, per-round verify on the persistent SYCL worker,
    /// same stop / EOS / max-token / stats handling) and REUSES the exact
    /// same verify + accept primitive ([`verify_and_commit_speculation`] →
    /// [`crate::speculative::accept_reject`]). The only difference is the
    /// drafter: a [`LookaheadDrafter`] harvests an n-gram pool from the
    /// running history and walks it for a Jacobi window of guess tokens
    /// (vs. the n-gram drafter's single-match literal run). Because the
    /// verify/accept is shared, greedy decode is token-for-token identical
    /// to classic greedy — lookahead only ever accepts more per round.
    ///
    /// Same raw-softmax sampling caveat as the n-gram path (callers route
    /// here only for grammar-free requests).
    pub fn speculate_lookahead_stream_from_ids(
        &self,
        initial_prompt_ids: Vec<i32>,
        cfg: LookaheadDrafterConfig,
        s: SamplingParams,
    ) -> EngineResult<TokenStream> {
        let tokenizer = self.tokenizer.clone().ok_or_else(|| {
            crate::EngineError::Unimplemented("speculate_lookahead requires a tokenizer")
        })?;
        if initial_prompt_ids.is_empty() {
            let stream = async_stream::stream! {
                if false { yield Ok(Token { id: 0, text: String::new(), logprobs: None }); }
            };
            return Ok(Box::pin(stream));
        }

        let drafter = LookaheadDrafter::new(cfg);
        // The Jacobi window bounds the committed-token overshoot per round.
        let candidate_cap = cfg.window.max(1);
        let seed = s.seed;
        let max_tokens = s.max_tokens as usize;
        let model_eos = self.model.cfg.eos_token_id;
        let stop_strings = s.stop.clone();
        let greedy_spec = s.temperature <= 0.0;
        let typical = if s.spec_typical_accept {
            Some((s.spec_typical_epsilon, s.spec_typical_delta))
        } else {
            None
        };
        let repeat_pen = s.repeat_penalty;
        let freq_pen = s.frequency_penalty;
        let pres_pen = s.presence_penalty;

        let model = self.model.clone();
        let state = self.state.clone();
        let prefix_cache = self.prefix_cache;
        let prefill_chunk_size = self.prefill_chunk_size;
        let max_ctx = self.max_ctx;
        let cumulative_stats = self.cumulative_stats.clone();
        let last_stats = self.last_stats.clone();
        let ema_tok_s_bits = Arc::clone(&self.ema_tok_s_bits);
        let worker = self.sycl_worker.submitter();
        let flash_attention = self.flash_attention;
        let n_gpu_layers = self.n_gpu_layers;
        let cpu_force_patterns = self.cpu_force_patterns.clone();

        let stream = async_stream::stream! {
            let mut rng_slot = Some(Rng::from_seed(seed));
            let mut current_prompt_ids: Vec<i32> = initial_prompt_ids;
            let mut total_emitted: usize = 0;
            let mut emitted_text = String::new();
            let mut utf8 = Utf8Stream::default();
            let decode_start = std::time::Instant::now();

            // Install this request's dispatch state on the worker thread +
            // ensure the USM context exists (mirrors the n-gram driver so a
            // GPU host doesn't fall to 100% CPU dispatch). Idempotent.
            {
                let model_install = model.clone();
                let flash = flash_attention;
                let ngl = n_gpu_layers;
                let cpu_force = cpu_force_patterns.clone();
                let _ = worker
                    .run_blocking(move || {
                        rustllama_models::accel::set_flash_attention(flash);
                        rustllama_models::accel::set_n_gpu_layers(ngl);
                        rustllama_models::accel::set_cpu_force_patterns(cpu_force);
                        let cfg = &model_install.cfg;
                        let head_dim = if cfg.head_dim > 0 {
                            cfg.head_dim
                        } else {
                            cfg.d_model / cfg.n_heads.max(1)
                        };
                        let _ = rustllama_models::accel::prepare_usm_context(
                            cfg.n_layers as u32,
                            cfg.n_heads as u32,
                            cfg.n_kv_heads as u32,
                            head_dim as u32,
                            max_ctx as u32,
                        );
                    })
                    .await;
            }

            loop {
                if total_emitted >= max_tokens {
                    return;
                }
                let remaining = max_tokens.saturating_sub(total_emitted);

                // ---- Step 1: lookahead drafter proposes a Jacobi guess ----
                let hist_u32: Vec<u32> =
                    current_prompt_ids.iter().map(|&t| t as u32).collect();
                let mut drafts = drafter.propose(&hist_u32);
                let round_cap = candidate_cap.min(remaining.saturating_sub(1));
                if drafts.len() > round_cap {
                    drafts.truncate(round_cap);
                }
                // ---- Step 2+3: verify + accept + COMMIT (shared primitive) ----
                let prompt_ids_for_verify = current_prompt_ids.clone();
                let drafts_for_verify = drafts.clone();
                let model_c = model.clone();
                let state_c = state.clone();
                let mut rng_c = rng_slot.take().expect("rng slot");
                let joined = worker
                    .run_blocking(move || {
                        let out = verify_and_commit_speculation(
                            model_c, state_c, prefix_cache, prefill_chunk_size, max_ctx,
                            &prompt_ids_for_verify, &drafts_for_verify, greedy_spec,
                            typical, &mut rng_c, repeat_pen, freq_pen, pres_pen,
                        );
                        (out, rng_c)
                    })
                    .await;
                let outcome = match joined {
                    Ok((Ok(o), r)) => {
                        rng_slot = Some(r);
                        o
                    }
                    Ok((Err(e), r)) => {
                        rng_slot = Some(r);
                        let _ = &rng_slot;
                        yield Err(crate::EngineError::Engine(format!(
                            "verify_speculation (lookahead): {e}"
                        )));
                        return;
                    }
                    Err(recv_err) => {
                        yield Err(crate::EngineError::Engine(format!(
                            "verify worker recv: {recv_err}"
                        )));
                        return;
                    }
                };
                cumulative_stats
                    .add_speculation(drafts.len() as u64, outcome.accepted.len() as u64);
                let mut committed: Vec<u32> = outcome.accepted;
                committed.push(outcome.replacement);

                // ---- Step 4: emit committed; check stop/EOS/max ----
                for &id in &committed {
                    let text = utf8.next_text(id, |ids| {
                        tokenizer.decode(ids, true).unwrap_or_default()
                    });
                    emitted_text.push_str(&text);
                    let hit_stop = stop_strings
                        .iter()
                        .any(|st| !st.is_empty() && emitted_text.contains(st.as_str()));
                    yield Ok(Token { id, text, logprobs: None });
                    total_emitted += 1;
                    {
                        let elapsed_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
                        let s = RequestStats {
                            prefill_ms: 0.0,
                            decode_ms: elapsed_ms,
                            tokens_prefilled: 0,
                            cache_hit_tokens: 0,
                            tokens_generated: total_emitted as u32,
                            tool_call_limit_hit: false,
                        };
                        update_ema_tok_s(&ema_tok_s_bits, &s);
                        *last_stats.lock().expect("last_stats") = s;
                    }
                    if hit_stop {
                        return;
                    }
                    if Some(id) == model_eos {
                        return;
                    }
                    if total_emitted >= max_tokens {
                        return;
                    }
                }
                current_prompt_ids.extend(committed.iter().map(|&id| id as i32));
            }
        };
        Ok(Box::pin(stream))
    }

    /// Text entry for MTP / NextN self-speculation. Tokenizes `prompt`
    /// (with the model's BOS policy) and forwards to
    /// [`Self::speculate_mtp_stream_from_ids`]. Mirrors
    /// [`Self::speculate_ngram_stream`].
    ///
    /// PRECONDITIONS (the caller — `chat` / `generate` — checks these
    /// via [`Self::model_supports_mtp`] before routing here): the loaded
    /// model is hybrid AND carries a NextN head, and the request is
    /// grammar-free. On a model without a NextN head this still runs but
    /// `mtp_round` would error on the missing head, so never route a
    /// non-MTP model here.
    pub fn speculate_mtp_stream(
        &self,
        prompt: String,
        s: SamplingParams,
        tree_widths: Option<Vec<u32>>,
    ) -> EngineResult<TokenStream> {
        let tokenizer = self.tokenizer.clone().ok_or_else(|| {
            crate::EngineError::Unimplemented("speculate_mtp requires a tokenizer")
        })?;
        let add_bos = tokenizer.add_bos_token();
        let initial_prompt_ids: Vec<i32> = tokenizer
            .encode(&prompt, add_bos)?
            .into_iter()
            .map(|t| t as i32)
            .collect();
        self.speculate_mtp_stream_from_ids(initial_prompt_ids, s, tree_widths)
    }

    /// ID-level MTP / NextN self-speculative decode driver.
    ///
    /// Structurally mirrors [`Self::speculate_ngram_stream_from_ids`]
    /// (async stream, per-round work on the persistent SYCL worker, same
    /// streaming / stop / EOS / stats handling), but the drafter is the
    /// model's own NextN head instead of an n-gram lookup: each no-draft
    /// round produces a +2-token draft, and the next round batch-verifies
    /// it (accept commits 2 tokens for one batched forward; reject commits
    /// 1 and rewinds the draft position). See [`mtp_round`] for the KV /
    /// DeltaNet position discipline.
    ///
    /// SAMPLING SEMANTICS — like the ngram path, verification samples the
    /// target's softmax with per-request temperature / top_k / top_p NOT
    /// applied, but repeat / frequency / presence PENALTIES ARE applied to
    /// the target logits before the verify (so greedy decode with
    /// `repeat_penalty != 1.0` stays token-for-token identical to the
    /// classic path). Callers route here only for grammar-free requests.
    pub fn speculate_mtp_stream_from_ids(
        &self,
        initial_prompt_ids: Vec<i32>,
        s: SamplingParams,
        tree_widths: Option<Vec<u32>>,
    ) -> EngineResult<TokenStream> {
        self.speculate_tree_stream_from_ids(initial_prompt_ids, s, tree_widths, None)
    }

    /// P10b — text entry for EAGLE tree speculation. Tokenizes `prompt` (BOS
    /// policy) and forwards to [`Self::speculate_eagle_stream_from_ids`].
    /// PRECONDITIONS (checked by `chat` / `generate` via
    /// [`Self::model_supports_eagle`] before routing here): a loaded EAGLE head,
    /// a dense/MoE model, a grammar-free request, and no SWA ring.
    pub fn speculate_eagle_stream(
        &self,
        prompt: String,
        s: SamplingParams,
        head: Arc<rustllama_models::llama_arch::EagleHead>,
    ) -> EngineResult<TokenStream> {
        let tokenizer = self.tokenizer.clone().ok_or_else(|| {
            crate::EngineError::Unimplemented("speculate_eagle requires a tokenizer")
        })?;
        let add_bos = tokenizer.add_bos_token();
        let initial_prompt_ids: Vec<i32> = tokenizer
            .encode(&prompt, add_bos)?
            .into_iter()
            .map(|t| t as i32)
            .collect();
        self.speculate_eagle_stream_from_ids(initial_prompt_ids, s, head)
    }

    /// P10b — ID-level EAGLE tree self-speculative decode driver. Shares the
    /// whole MTP tree-spec stream (prefill, per-round verify, streaming / stop /
    /// EOS / stats) via [`Self::speculate_tree_stream_from_ids`]; the only
    /// difference is the drafter ([`tree_eagle_round`] runs the trained head
    /// instead of the base MTP sidecars). Tree widths come from the configured
    /// `speculative_tree_widths` (else the `[2,2,2,2]` default) — EAGLE always
    /// drafts a tree.
    pub fn speculate_eagle_stream_from_ids(
        &self,
        initial_prompt_ids: Vec<i32>,
        s: SamplingParams,
        head: Arc<rustllama_models::llama_arch::EagleHead>,
    ) -> EngineResult<TokenStream> {
        let widths = self
            .tree_spec
            .clone()
            .filter(|w| !w.is_empty())
            .unwrap_or_else(|| vec![2, 2, 2, 2]);
        self.speculate_tree_stream_from_ids(initial_prompt_ids, s, Some(widths), Some(head))
    }

    /// Shared tree-spec stream body for the MTP (Feature 4) and EAGLE (P10b)
    /// drafters. `eagle = Some(head)` routes each round through
    /// [`tree_eagle_round`]; `None` keeps the MTP dispatch
    /// (`tree_mtp_round` / `tree_mtp_round_hybrid` / linear `mtp_round`). Every
    /// other stage — prefill, streaming, stop / EOS / stats — is identical.
    fn speculate_tree_stream_from_ids(
        &self,
        initial_prompt_ids: Vec<i32>,
        s: SamplingParams,
        tree_widths: Option<Vec<u32>>,
        eagle: Option<Arc<rustllama_models::llama_arch::EagleHead>>,
    ) -> EngineResult<TokenStream> {
        let tokenizer = self.tokenizer.clone().ok_or_else(|| {
            crate::EngineError::Unimplemented("speculate_mtp requires a tokenizer")
        })?;
        if initial_prompt_ids.is_empty() {
            let stream = async_stream::stream! {
                if false { yield Ok(Token { id: 0, text: String::new(), logprobs: None }); }
            };
            return Ok(Box::pin(stream));
        }

        let seed = s.seed;
        let max_tokens = s.max_tokens as usize;
        let model_eos = self.model.cfg.eos_token_id;
        let stop_strings = s.stop.clone();
        let greedy_spec = s.temperature <= 0.0;
        // Sampling penalties applied to the target logits before each
        // round's accept/reject verify (keeps greedy-with-penalties
        // token-for-token identical to the classic path).
        let repeat_pen = s.repeat_penalty;
        let freq_pen = s.frequency_penalty;
        let pres_pen = s.presence_penalty;
        let vocab = self.model.cfg.vocab_size;

        // Snapshot Arc handles so the async stream owns them without
        // borrowing `&self` for its lifetime (mirrors the ngram path).
        let model = self.model.clone();
        let state = self.state.clone();
        let prefix_cache = self.prefix_cache;
        let prefill_chunk_size = self.prefill_chunk_size;
        let max_ctx = self.max_ctx;
        let cumulative_stats = self.cumulative_stats.clone();
        let last_stats = self.last_stats.clone();
        let ema_tok_s_bits = Arc::clone(&self.ema_tok_s_bits);
        // Route every forward onto the SYCL worker thread (warmed USM +
        // persistent stream), exactly like the ngram path.
        let worker = self.sycl_worker.submitter();
        let flash_attention = self.flash_attention;
        let n_gpu_layers = self.n_gpu_layers;
        let cpu_force_patterns = self.cpu_force_patterns.clone();
        // P10b — the EAGLE head (if routing EAGLE spec), owned by the stream.
        let eagle = eagle;

        let stream = async_stream::stream! {
            let mut rng_slot = Some(Rng::from_seed(seed));
            let mut total_emitted: usize = 0;
            let mut emitted_text = String::new();
            let mut utf8 = Utf8Stream::default();
            let decode_start = std::time::Instant::now();

            // Install this request's dispatch state on the worker thread +
            // ensure the USM context exists (see `speculate` for the
            // rationale). Idempotent; runs once before the round loop.
            {
                let model_install = model.clone();
                let flash = flash_attention;
                let ngl = n_gpu_layers;
                let cpu_force = cpu_force_patterns.clone();
                let _ = worker
                    .run_blocking(move || {
                        rustllama_models::accel::set_flash_attention(flash);
                        rustllama_models::accel::set_n_gpu_layers(ngl);
                        rustllama_models::accel::set_cpu_force_patterns(cpu_force);
                        let cfg = &model_install.cfg;
                        let head_dim = if cfg.head_dim > 0 {
                            cfg.head_dim
                        } else {
                            cfg.d_model / cfg.n_heads.max(1)
                        };
                        let _ = rustllama_models::accel::prepare_usm_context(
                            cfg.n_layers as u32,
                            cfg.n_heads as u32,
                            cfg.n_kv_heads as u32,
                            head_dim as u32,
                            max_ctx as u32,
                        );
                    })
                    .await;
            }

            // ---- Prefill: forward all but the last prompt token, then
            //      seed the decode frontier. Runs on the worker under the
            //      state lock (mirrors `generate_token_ids_streaming`). ----
            let prefill_ids = initial_prompt_ids.clone();
            let model_p = model.clone();
            let state_p = state.clone();
            let prefill_join = worker
                .run_blocking(move || -> Result<(i32, u32)> {
                    let mut st = state_p.lock().expect("state lock");
                    let prompt_u32: Vec<u32> =
                        prefill_ids.iter().map(|&t| t as u32).collect();
                    let mut logits = vec![0f32; model_p.cfg.vocab_size];
                    let prompt_max = prefill_ids.len().saturating_sub(1);
                    let effective = st.prepare_prefix_reuse(
                        &prompt_u32,
                        prefix_cache,
                        PREFIX_REUSE_MIN_TOKENS,
                        prompt_max,
                    );
                    let mut chrome = ChromeTracer::from_env();
                    let (prefill, last) =
                        prefill_ids.split_at(prefill_ids.len() - 1);
                    let done = run_chunked_prefill(
                        &model_p,
                        &mut st,
                        &prompt_u32,
                        prefill,
                        effective,
                        prefill_chunk_size,
                        &mut logits,
                        &mut chrome,
                        || false,
                    );
                    if done < prefill.len() {
                        return Err(CpuEngineError::Other(
                            "mtp prefill did not complete".to_string(),
                        ));
                    }
                    // Bootstrap `last_ids` to exactly the forwarded prefix
                    // (positions `[0, next_pos)`), overriding whatever
                    // `prepare_prefix_reuse` left. `mtp_round` then extends
                    // it by each forwarded token so `last_ids.len()` stays
                    // == `kv.seq_len` — the invariant the next request's
                    // hybrid "continue in place" reuse relies on (a stale
                    // `last_ids` there would desync the DeltaNet state).
                    st.last_ids = prompt_u32[..prefill.len()].to_vec();
                    Ok((last[0], prefill.len() as u32))
                })
                .await;
            let (mut next_input, mut next_pos) = match prefill_join {
                Ok(Ok(v)) => v,
                Ok(Err(e)) => {
                    yield Err(crate::EngineError::Engine(format!(
                        "mtp prefill: {e}"
                    )));
                    return;
                }
                Err(recv_err) => {
                    yield Err(crate::EngineError::Engine(format!(
                        "mtp prefill worker recv: {recv_err}"
                    )));
                    return;
                }
            };

            // Carried NextN draft for `next_pos + 1` (produced by the
            // previous no-draft round). `None` on the first round.
            let mut pending: Option<DraftToken> = None;

            loop {
                if total_emitted >= max_tokens {
                    return;
                }
                // Context exhausted — the single forward writes `next_pos`.
                if next_pos as usize >= max_ctx {
                    return;
                }
                let remaining = max_tokens - total_emitted;
                // Only batch-verify when there is room for a bonus token
                // (>= 2 left) AND room in the KV for the 2-position batch.
                let allow_spec =
                    remaining >= 2 && (next_pos as usize + 2) <= max_ctx;

                let model_c = model.clone();
                let state_c = state.clone();
                let mut rng_c = rng_slot.take().expect("rng slot");
                let ni = next_input;
                let np = next_pos;
                let pend = pending;
                let vocab_c = vocab;
                // P10b — EAGLE head first: the trained draft head drafts the
                // tree, verified via the shared dense stack. Else tree spec
                // (Feature 4) when widths are set + the model supports it; else
                // the linear NextN `mtp_round`. All return `MtpRoundOutcome`, so
                // the whole stream (emit / stop / stats) is shared.
                let tw = tree_widths.clone();
                let mc = max_ctx;
                let eagle_c = eagle.clone();
                let joined = worker
                    .run_blocking(move || {
                        let out = if let Some(head) = eagle_c.as_ref() {
                            // EAGLE tree spec (dense/MoE contiguous): the head
                            // drafts autoregressively; verify/accept/commit is
                            // the shared `dense_tree_verify_commit` stack.
                            let widths = tw.unwrap_or_else(|| vec![2, 2, 2, 2]);
                            tree_eagle_round(
                                &model_c, &state_c, head, ni, np, &widths, mc,
                                greedy_spec, &mut rng_c, vocab_c,
                                repeat_pen, freq_pen, pres_pen,
                            )
                        } else if let Some(widths) = tw {
                            // Tree spec: hybrid models go through the recurrent-
                            // state tree walk; dense/MoE through the batched
                            // ancestor-masked verify. Both return MtpRoundOutcome.
                            if model_c.weights.is_hybrid() {
                                tree_mtp_round_hybrid(
                                    &model_c, &state_c, ni, np, &widths, mc,
                                    greedy_spec, &mut rng_c, vocab_c,
                                    repeat_pen, freq_pen, pres_pen,
                                )
                            } else {
                                tree_mtp_round(
                                    &model_c, &state_c, ni, np, &widths, mc,
                                    greedy_spec, &mut rng_c, vocab_c,
                                    repeat_pen, freq_pen, pres_pen,
                                )
                            }
                        } else {
                            mtp_round(
                                &model_c, &state_c, ni, np, pend, allow_spec,
                                greedy_spec, &mut rng_c, vocab_c,
                                repeat_pen, freq_pen, pres_pen,
                            )
                        };
                        (out, rng_c)
                    })
                    .await;
                let round = match joined {
                    Ok((Ok(r), rng)) => {
                        rng_slot = Some(rng);
                        r
                    }
                    Ok((Err(e), rng)) => {
                        rng_slot = Some(rng);
                        let _ = &rng_slot;
                        yield Err(crate::EngineError::Engine(format!(
                            "mtp round: {e}"
                        )));
                        return;
                    }
                    Err(recv_err) => {
                        yield Err(crate::EngineError::Engine(format!(
                            "mtp round worker recv: {recv_err}"
                        )));
                        return;
                    }
                };
                cumulative_stats
                    .add_speculation(round.drafted as u64, round.accepted as u64);

                // Emit committed tokens; check stop / EOS / max after each.
                for &id in &round.committed {
                    let text = utf8.next_text(id, |ids| {
                        tokenizer.decode(ids, true).unwrap_or_default()
                    });
                    emitted_text.push_str(&text);
                    let hit_stop = stop_strings
                        .iter()
                        .any(|st| !st.is_empty() && emitted_text.contains(st.as_str()));
                    yield Ok(Token { id, text, logprobs: None });
                    total_emitted += 1;
                    {
                        let elapsed_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
                        let s = RequestStats {
                            prefill_ms: 0.0,
                            decode_ms: elapsed_ms,
                            tokens_prefilled: 0,
                            cache_hit_tokens: 0,
                            tokens_generated: total_emitted as u32,
                            tool_call_limit_hit: false,
                        };
                        update_ema_tok_s(&ema_tok_s_bits, &s);
                        *last_stats.lock().expect("last_stats") = s;
                    }
                    if hit_stop {
                        return;
                    }
                    if Some(id) == model_eos {
                        return;
                    }
                    if total_emitted >= max_tokens {
                        return;
                    }
                }

                next_input = round.next_input;
                next_pos = round.next_pos;
                pending = round.pending;
            }
        };
        Ok(Box::pin(stream))
    }

    /// VLM streaming chat — image-bearing path.
    ///
    /// V-6b-3d wiring: when `Engine::chat` sees a request with
    /// attached image bytes and the engine has a vision tower loaded
    /// (`load_with_mmproj` succeeded), it routes here instead of the
    /// text-only `spawn_stream`. Pipeline:
    ///
    /// 1. Tokenize the rendered chat prompt.
    /// 2. Acquire the engine's KV state (must be the contiguous
    ///    backend — paged KV + VLM is a v2 enhancement).
    /// 3. Call [`vlm_prefill_ids`] to run prepare_vlm_inputs +
    ///    embed_tokens + splice + forward_prefill_from_embeds. The
    ///    KV cache is advanced by the spliced sequence length.
    /// 4. Decode loop: sample from the prefill-output logits, call
    ///    [`forward_one_via_backend`] each iteration, push tokens
    ///    through an mpsc to the returned `TokenStream`.
    ///
    /// Routed through `spawn_blocking` rather than the SYCL worker
    /// thread because the prefill stage runs the vision pipeline +
    /// the from-embeds path, neither of which has the per-thread
    /// USM warmups the text path relies on. v1 takes the slower
    /// (uncached) route; v2 can wire VLM through the SYCL worker
    /// once the from-embeds path gets its own packed-weight upload.
    fn chat_vlm_stream(
        &self,
        prompt: String,
        images: Vec<Vec<u8>>,
        sampling: SamplingParams,
    ) -> EngineResult<TokenStream> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| crate::EngineError::Unimplemented("chat requires a tokenizer"))?
            .clone();
        // Same channel cap as `spawn_stream` — small so engine stays
        // ~4 tokens ahead of the consumer.
        let (tx, rx) = tokio::sync::mpsc::channel::<EngineResult<Token>>(4);

        let model = self.model.clone();
        let state = self.state.clone();
        let vision = self.vision.clone();
        let image_token_id = self.image_token_id;
        let placeholder_mode = self.placeholder_mode;
        let max_ctx = self.max_ctx;
        let model_eos = model.cfg.eos_token_id;
        let feature_memo = Arc::clone(&self.vision_feature_memo);

        // P11 (VLM): resolve the per-request LoRA selection NOW, on the
        // submitting thread (which holds the registry), exactly like
        // `spawn_stream`. An unknown adapter fails the request here instead of
        // the VLM path silently serving the un-adapted base. `None` ⇒ base
        // (byte-identical). The resolved adapter is scoped inside
        // `drive_vlm_generation` for the whole generation.
        let lora_adapter = self
            .lora_adapter_for(&sampling)
            .map_err(|e| crate::EngineError::Engine(e.to_string()))?;

        tokio::task::spawn_blocking(move || {
            let result = drive_vlm_generation(
                model,
                state,
                tokenizer,
                vision,
                image_token_id,
                placeholder_mode,
                max_ctx,
                model_eos,
                prompt,
                images,
                sampling,
                feature_memo,
                lora_adapter,
                &tx,
            );
            if let Err(e) = result {
                let _ = tx.blocking_send(Err(e));
            }
        });

        let stream = async_stream::stream! {
            let mut rx = rx;
            while let Some(item) = rx.recv().await {
                yield item;
            }
        };
        Ok(Box::pin(stream))
    }

    /// Drive a generation on the dedicated SYCL worker thread,
    /// yielding tokens one by one through an mpsc channel that backs
    /// an async Stream. Submitting via the worker (instead of
    /// `spawn_blocking` onto tokio's blocking pool) keeps the
    /// per-thread USM weight cache + SyclStreamGuard alive across
    /// requests — the warmed thread is the SAME thread every time.
    fn spawn_stream(&self, prompt: String, sampling: SamplingParams) -> EngineResult<TokenStream> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| crate::EngineError::Unimplemented("generate/chat require a tokenizer"))?
            .clone();
        let model = self.model.clone();
        let state = self.state.clone();
        let max_ctx = self.max_ctx;
        let model_eos = model.cfg.eos_token_id;
        let prefix_cache = self.prefix_cache;
        let max_tool_iterations = self.max_tool_iterations;

        // Small buffer keeps the engine in lockstep with the SSE consumer.
        // Larger buffers let the engine race ahead of the network, which
        // means a disconnect takes longer to drain: every queued token
        // is wasted CPU + delays the engine's cancellation by one
        // iteration. With cap=4 the engine stays ≤4 tokens ahead of the
        // socket and stops within ~one forward pass of a drop.
        let (tx, rx) = tokio::sync::mpsc::channel::<EngineResult<Token>>(4);
        let last_stats = self.last_stats.clone();

        // P11: resolve the per-request LoRA selection now (on the submitting
        // thread, which holds the registry). An unknown name fails the request
        // instead of silently serving the base.
        let lora_adapter = self
            .lora_adapter_for(&sampling)
            .map_err(|e| crate::EngineError::Engine(e.to_string()))?;

        self.sycl_worker.submit_generation(DriveGenJob {
            model,
            state,
            tokenizer,
            max_ctx,
            model_eos,
            prefix_cache,
            max_tool_iterations,
            prompt,
            sampling,
            last_stats,
            ema_tok_s_bits: Arc::clone(&self.ema_tok_s_bits),
            cumulative_stats: Arc::clone(&self.cumulative_stats),
            response_tx: tx,
            flash_attention: self.flash_attention,
            n_gpu_layers: self.n_gpu_layers,
            cpu_force_patterns: self.cpu_force_patterns.clone(),
            lora_adapter,
        });

        let stream = async_stream::stream! {
            let mut rx = rx;
            while let Some(item) = rx.recv().await {
                yield item;
            }
        };
        Ok(Box::pin(stream) as Pin<Box<dyn Stream<Item = EngineResult<Token>> + Send>>)
    }
}

/// UTF-8-safe incremental detokenizer for streaming output.
///
/// Byte-level BPE (Qwen / Bonsai / Llama-BPE) can split one multi-byte
/// character — an emoji like 👋 (4 bytes) or a CJK glyph — across two
/// tokens. Decoding each token in isolation (`decode_single`) then yields a
/// `U+FFFD` replacement char per fragment, so the client renders `��`.
///
/// This buffers the running generated-id list and, each token, decodes the
/// whole list and emits only the newly-completed suffix. A trailing
/// incomplete character shows up as `U+FFFD` at the end of the full decode,
/// so we hold the delta back until the next token completes it. Byte-level
/// decode is append-only during generation, so a previously-emitted byte
/// length stays a valid char boundary of the longer decode.
#[derive(Default)]
struct Utf8Stream {
    ids: Vec<u32>,
    printed: usize,
}

impl Utf8Stream {
    /// Push a token id and return the text now safe to emit (empty while a
    /// multi-byte character is still incomplete). `decode` maps the running
    /// id list to its full surface string.
    fn next_text<F: FnOnce(&[u32]) -> String>(&mut self, id: u32, decode: F) -> String {
        self.ids.push(id);
        let full = decode(&self.ids);
        // Trailing replacement char ⇒ last character is still incomplete.
        if full.ends_with('\u{FFFD}') {
            return String::new();
        }
        if self.printed <= full.len() && full.is_char_boundary(self.printed) {
            let out = full[self.printed..].to_string();
            self.printed = full.len();
            out
        } else {
            // Non-append decode (shouldn't happen for byte-level BPE) —
            // resync without re-emitting to avoid a panic / duplicate text.
            self.printed = full.len();
            String::new()
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn drive_generation(
    model: Arc<LlamaModel>,
    state: Arc<Mutex<EngineState>>,
    tokenizer: Arc<Tokenizer>,
    max_ctx: usize,
    model_eos: Option<u32>,
    prefix_cache: bool,
    max_tool_iterations: u32,
    prompt: String,
    sampling: SamplingParams,
    last_stats: Arc<Mutex<RequestStats>>,
    ema_tok_s_bits: Arc<std::sync::atomic::AtomicU64>,
    cumulative_stats: Arc<crate::CumulativeStats>,
    tx: &tokio::sync::mpsc::Sender<EngineResult<Token>>,
    flash_attention: bool,
    n_gpu_layers: u32,
    cpu_force_patterns: Vec<String>,
    lora_adapter: Option<Arc<rustllama_models::lora::LoraAdapter>>,
) -> EngineResult<()> {
    // P11: scope the selected LoRA adapter as the thread-active adapter for the
    // whole generation on THIS worker thread. `None` restores/keeps the base
    // (byte-identical). Set before any forward and restored on drop (incl. an
    // early return / `?`), so a selected adapter is applied to prefill + every
    // decode step and the next request on this worker starts clean.
    let adapter_active = lora_adapter.is_some();
    let _lora_scope = rustllama_models::lora::scope_active_lora(lora_adapter);
    // P11: the prompt-prefix cache is content-keyed, not adapter-keyed — an
    // adapted request must not reuse/populate it. `None` ⇒ unchanged.
    let prefix_cache = prefix_cache && !adapter_active;
    // Install the per-thread SYCL stream + flash-attention flag for
    // the duration of this generation. This is the *streaming*
    // codepath used by `Engine::chat` and `Engine::generate` (the
    // shape the server + GUI hit); the synchronous
    // `generate_token_ids*` variants install their own guards.
    // Without this, a thread with no TLS stream installed would
    // return `false` from every `try_*_usm_f32` / `try_*_f32` hook
    // in `models::accel` → 100% CPU dispatch.
    //
    // When called on the dedicated `SyclWorker` thread, an outer
    // guard is already installed for the worker's lifetime. We must
    // NOT install a second one because `SyclStreamGuard::Drop`
    // clears the TLS slot unconditionally — a local install/drop
    // cycle would tear down the worker's persistent SYCL state and
    // force a stream re-creation on every request. Skip the install
    // in that case and rely on the outer guard.
    let _sycl_guard = if rustllama_models::accel::has_sycl_stream() {
        None
    } else {
        install_sycl_dispatch_if_requested()
    };
    // Per-thread dispatch state, mirroring the sync
    // `generate_token_ids*` paths: flash-attention flag, the
    // hybrid-placement layer cutoff, and the CPU-force override
    // patterns. The worker thread's TLS defaults are "flash on,
    // ALL layers GPU, no overrides", so skipping these installs
    // silently discarded `[inference].n_gpu_layers`, the
    // auto-placement planner's decision, and
    // `placement.overrides` for every streamed request — and
    // layers the warmup deliberately did NOT preload (>= cutoff)
    // still took the GPU path, paying lazy mid-decode uploads.
    rustllama_models::accel::set_flash_attention(flash_attention);
    rustllama_models::accel::set_n_gpu_layers(n_gpu_layers);
    rustllama_models::accel::set_cpu_force_patterns(cpu_force_patterns);
    // Pre-build the USM attention context with the model's dims so
    // the packed-matvec USM hook can fire during prefill as well as
    // decode. Without this, the USM context isn't created until the
    // first `try_flash_attn_decode_usm_f32` call (which only runs
    // during decode), so the entire prefill phase silently falls
    // back to CPU even with `RUSTLLAMA_USM_ATTN=1`. `false` here is
    // benign — engine continues on the CPU path.
    //
    // We use `has_sycl_stream()` (not `_sycl_guard.is_some()`) here
    // because on the dedicated worker thread the outer guard is
    // installed but `_sycl_guard` is `None` — see comment above.
    if rustllama_models::accel::has_sycl_stream() {
        let cfg = &model.cfg;
        let head_dim = if cfg.head_dim > 0 {
            cfg.head_dim
        } else {
            cfg.d_model / cfg.n_heads.max(1)
        };
        let ctx_ready = rustllama_models::accel::prepare_usm_context(
            cfg.n_layers as u32,
            cfg.n_heads as u32,
            cfg.n_kv_heads as u32,
            head_dim as u32,
            max_ctx as u32,
        );
        // (The per-generation x-upload-cache invalidation that lived
        // here is gone: the (ptr, len) dedup itself was removed on
        // 2026-09-07 — every matvec hook now uploads its activation
        // unconditionally, which closes the whole stale-content class
        // this call only partially defended against.)
        // Once-per-thread: warm the USM weight cache by uploading
        // every packed-quant weight tensor in the model. Without this
        // each weight is lazily uploaded on its first matvec call,
        // making the first chat on this thread 1-2s slower than
        // subsequent chats. We gate on a per-thread flag because the
        // server's blocking-pool can land later requests on threads
        // we've never preloaded; a process-wide OnceLock would skip
        // those.
        if ctx_ready && !rustllama_models::accel::packed_weights_preloaded() {
            let start = std::time::Instant::now();
            let (uploaded, skipped, _bytes) = model.preload_packed_weights_to_usm();
            tracing::info!(
                uploaded,
                skipped,
                elapsed_ms = start.elapsed().as_millis() as u64,
                "USM weight pre-upload complete — first chat on this thread now warm-cached",
            );
            rustllama_models::accel::mark_packed_weights_preloaded();
        }
    }
    // Reset stats up-front so a mid-request error doesn't leave stale
    // numbers from the previous request visible to the next caller.
    *last_stats.lock().expect("last_stats") = RequestStats::default();
    let add_bos = tokenizer.add_bos_token();
    // Interleaved streaming prefill: segment 0 of the prompt is tokenized
    // synchronously on this thread, the rest go to a background
    // `encode_batch` worker, and the prefill loop forwards each chunk's
    // tokens through the model as soon as they arrive. The first decode
    // forward (which drives the first sample) can start before the last
    // chunk's encode finishes — meaningful TTFT savings on long chat
    // contexts. See [`Tokenizer::encode_streaming`].
    let stream = tokenizer.encode_streaming(&prompt, add_bos)?;
    let first_chunk: Vec<u32> = match stream.next() {
        Some(Ok(c)) if !c.is_empty() => c,
        Some(Ok(_)) | None => return Ok(()),
        Some(Err(e)) => return Err(e.into()),
    };
    let mut prompt_acc: Vec<u32> = first_chunk;

    let mut state = state.lock().expect("state lock");

    // Defer the full max_ctx check to per-chunk arrival once we know
    // the running prompt length.
    if prompt_acc.len() + sampling.max_tokens as usize > max_ctx {
        return Err(crate::EngineError::Engine(format!(
            "prompt too long: {prompt_len} tokens, max_ctx={max_ctx}",
            prompt_len = prompt_acc.len()
        )));
    }

    let vocab = model.cfg.vocab_size;
    let mut logits = vec![0f32; vocab];

    let trace_perf = std::env::var("RUSTLLAMA_TRACE_PERF").is_ok();
    let mut chrome = ChromeTracer::from_env();
    let mut prefill_ms = 0.0f64;
    let mut decode_fwd_ms = 0.0f64;
    let mut sample_ms = 0.0f64;
    let mut detok_ms = 0.0f64;

    // Initial prefix-cache prep using just the first chunk. Cap at
    // `prompt_acc.len()` (no `- 1`) because more chunks may still arrive;
    // the saturated-cache edge case is handled with a one-token back-off
    // after the streaming loop ends.
    let effective = state.prepare_prefix_reuse(
        &prompt_acc,
        prefix_cache,
        PREFIX_REUSE_MIN_TOKENS,
        prompt_acc.len(),
    );
    if trace_perf && effective > 0 {
        eprintln!(
            "[prefix-cache] reused {effective} of first {} prompt tokens",
            prompt_acc.len()
        );
    }

    let prefill_start = std::time::Instant::now();
    let mut prefill_pos = effective;
    // Stats counters. `cache_hit_tokens` starts with the initial LCP from
    // `prepare_prefix_reuse` and grows by each extended-LCP match; the
    // prefill-forward count is the number of actual `forward_one` calls.
    let mut cache_hit_tokens = effective as u32;
    let mut prefill_forwards = 0u32;

    loop {
        // Forward not-yet-prefilled tokens up to (acc.len() - 1). The
        // held-back-by-one keeps the most recent received token out of
        // prefill so it can drive the first decode forward once the
        // stream closes.
        while prefill_pos + 1 < prompt_acc.len() {
            if tx.is_closed() {
                chrome.span("prefill_cancelled", prefill_start);
                chrome.flush();
                return Ok(());
            }
            let t_tok = std::time::Instant::now();
            let tok = prompt_acc[prefill_pos] as i32;
            {
                // Hybrid-aware dispatch: DeltaNet models panic in the
                // plain wrapper ("no DeltaNetCache passed") — this was
                // the GUI chat path's crash on qwen35moe models.
                let EngineState {
                    kv_backend,
                    delta_net_cache,
                    ..
                } = &mut *state;
                forward_one_via_backend_hybrid(
                    model.as_ref(),
                    kv_backend,
                    delta_net_cache.as_mut(),
                    tok,
                    prefill_pos as u32,
                    &mut logits,
                );
            }
            chrome.span(&format!("prefill_token_{prefill_pos}"), t_tok);
            prefill_pos += 1;
            prefill_forwards += 1;
        }

        // Wait for the next chunk. `None` means the worker has finished
        // and no more chunks will arrive.
        let chunk = match stream.next() {
            Some(Ok(c)) if !c.is_empty() => c,
            Some(Ok(_)) => continue,
            Some(Err(e)) => return Err(e.into()),
            None => break,
        };

        if prompt_acc.len() + chunk.len() + sampling.max_tokens as usize > max_ctx {
            return Err(crate::EngineError::Engine(format!(
                "prompt too long: {prompt_len} tokens, max_ctx={max_ctx}",
                prompt_len = prompt_acc.len() + chunk.len()
            )));
        }

        // Extended-LCP: positions [prefill_pos, state.last_ids.len())
        // in the KV still hold the restored pool entry's tokens (we
        // haven't forwarded anything past prefill_pos). If the leading
        // tokens of the new chunk match `state.last_ids[prefill_pos..]`,
        // those KV cells are already correct for the new prompt and
        // we can skip the corresponding forwards.
        let extendable = state.last_ids.len().saturating_sub(prefill_pos);
        let to_check = chunk.len().min(extendable);
        let mut matched = 0;
        while matched < to_check && chunk[matched] == state.last_ids[prefill_pos + matched] {
            matched += 1;
        }
        prefill_pos += matched;
        cache_hit_tokens += matched as u32;
        // Prefix-cache extended-LCP is contiguous-only — paged has
        // already early-returned `last_ids` empty / matched == 0, so
        // `prefill_pos` is unchanged and this set is a no-op for
        // paged (set_seq_len only shrinks paged). For contiguous it
        // promotes seq_len to the prefix-cache hit point.
        state.kv_backend.set_seq_len(prefill_pos);
        prompt_acc.extend_from_slice(&chunk);
    }

    // Saturated-cache edge: if the entire prompt happened to match the
    // pool entry exactly, `prefill_pos == prompt_acc.len()` after the
    // loop. Back off by one so the decode loop has a token to drive
    // the first sample. Costs a single redundant forward pass — fine.
    // Unreachable on paged backend (no prefix-cache hit ⇒ prefill_pos == 0).
    if prefill_pos >= prompt_acc.len() {
        if state.delta_net_cache.is_some() {
            // Hybrid: re-forwarding a position is NOT idempotent —
            // the DeltaNet recurrence would fold that token into a
            // state that already contains it. Restart this rare
            // exact-resend case from scratch instead.
            state.reset();
            prefill_pos = 0;
            cache_hit_tokens = 0;
        } else {
            prefill_pos = prompt_acc.len() - 1;
            state.kv_backend.set_seq_len(prefill_pos);
        }
    }

    chrome.span("prefill_total", prefill_start);
    if trace_perf {
        prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    }

    let prompt_ids: Vec<i32> = prompt_acc.iter().map(|&t| t as i32).collect();
    let prompt_u32: Vec<u32> = prompt_acc;
    let mut history: Vec<u32> = prompt_u32.clone();

    let mut sampler = Sampler::new(sampling.clone());
    let vocab_size = model.cfg.vocab_size;
    let decode_one = |id: u32| {
        tokenizer
            .decode_single(id, true)
            .unwrap_or_default()
            .into_bytes()
    };
    let mut grammar: Option<GrammarMask> = match sampling.grammar.as_ref() {
        Some(GrammarKind::Json) => Some(GrammarMask::new_json(vocab_size, model_eos, decode_one)),
        Some(GrammarKind::JsonSchema { schema }) => Some(GrammarMask::new_json_schema(
            schema.clone(),
            vocab_size,
            model_eos,
            decode_one,
        )),
        Some(GrammarKind::ToolCallStream {
            schemas_by_name,
            min_completed,
        }) => Some(GrammarMask::new_tool_call_stream(
            schemas_by_name.clone(),
            max_tool_iterations,
            *min_completed,
            vocab_size,
            model_eos,
            decode_one,
        )),
        Some(GrammarKind::Code { language: _ }) => {
            Some(GrammarMask::new_code(vocab_size, model_eos, decode_one))
        }
        Some(GrammarKind::Regex { pattern }) => {
            // Same defense-in-depth as the chat-stream path: a
            // malformed pattern that slipped past the request
            // boundary degrades to "no grammar" rather than
            // crashing the worker thread.
            match GrammarMask::new_regex(pattern, vocab_size, model_eos, decode_one) {
                Ok(m) => Some(m),
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        pattern,
                        "invalid regex pattern in drive_generation — falling back to no grammar"
                    );
                    None
                }
            }
        }
        Some(GrammarKind::Gbnf { grammar }) => {
            match GrammarMask::new_gbnf(grammar, vocab_size, model_eos, decode_one) {
                Ok(m) => Some(m),
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "invalid GBNF grammar in drive_generation — falling back to no grammar"
                    );
                    None
                }
            }
        }
        None => None,
    };
    let mut next_input = prompt_ids[prefill_pos];
    let mut next_pos = prefill_pos as u32;
    let stop_strings = sampling.stop.clone();
    let logprobs_k = sampling.logprobs.map(|k| k as usize);

    // Buffer the emitted-but-not-yet-checked-for-stop text so we can match
    // multi-token stop strings reliably.
    let mut emitted_text = String::new();
    // UTF-8-safe incremental detok for the streamed deltas.
    let mut utf8 = Utf8Stream::default();
    let mut n_decoded = 0u32;
    let mut generated_ids: Vec<u32> = Vec::with_capacity(sampling.max_tokens as usize);

    // Tokens the consumer hasn't accepted yet. We `try_send` instead
    // of `blocking_send` because this loop runs WITH the engine state
    // mutex held: a slow-but-alive SSE client used to back up the
    // 4-deep channel and then park this thread inside the lock,
    // freezing `/v1/metrics` and every queued request indefinitely.
    // Now a full channel spills locally (bounded by max_tokens) and
    // the tail is drained with a blocking send AFTER the state guard
    // is released. A *disconnected* client still stops generation
    // within one forward pass via the `is_closed` polls.
    let mut pending: std::collections::VecDeque<Token> = std::collections::VecDeque::new();
    // Returns `true` when the channel is closed (consumer gone).
    let flush_pending =
        |pending: &mut std::collections::VecDeque<Token>,
         tx: &tokio::sync::mpsc::Sender<EngineResult<Token>>| {
            while let Some(item) = pending.pop_front() {
                match tx.try_send(Ok(item)) {
                    Ok(()) => {}
                    Err(tokio::sync::mpsc::error::TrySendError::Full(Ok(item))) => {
                        pending.push_front(item);
                        return false;
                    }
                    Err(tokio::sync::mpsc::error::TrySendError::Full(Err(_))) => {
                        unreachable!("only Ok tokens are queued")
                    }
                    Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => return true,
                }
            }
            false
        };

    for step in 0..sampling.max_tokens {
        // Cancellation poll: a client dropping the SSE stream becomes
        // visible here one full forward pass sooner than the send-side
        // closed check at the bottom of the loop.
        if tx.is_closed() {
            break;
        }
        // Opportunistic flush of spilled tokens from earlier
        // iterations (no-op when the consumer is keeping up).
        if flush_pending(&mut pending, tx) {
            break;
        }
        let t0 = std::time::Instant::now();
        {
            // Hybrid-aware dispatch — see the prefill loop above.
            let EngineState {
                kv_backend,
                delta_net_cache,
                ..
            } = &mut *state;
            forward_one_via_backend_hybrid(
                model.as_ref(),
                kv_backend,
                delta_net_cache.as_mut(),
                next_input,
                next_pos,
                &mut logits,
            );
        }
        chrome.span(&format!("decode_fwd_{step}"), t0);
        if trace_perf {
            decode_fwd_ms += t0.elapsed().as_secs_f64() * 1000.0;
        }

        let t1 = std::time::Instant::now();
        // Capture raw logits before the sampler mutates them. We only pay
        // the alloc when logprobs are actually requested.
        let raw_logits: Option<Vec<f32>> = logprobs_k.map(|_| logits.clone());

        let next_tok = sampler.sample_with_grammar(&mut logits, &history, grammar.as_ref());
        if let Some(g) = grammar.as_mut() {
            g.advance(next_tok);
        }
        chrome.span(&format!("sample_{step}"), t1);
        if trace_perf {
            sample_ms += t1.elapsed().as_secs_f64() * 1000.0;
        }

        let token_logprobs = match (logprobs_k, raw_logits) {
            (Some(k), Some(raw)) => Some(compute_logprobs(&raw, next_tok, k)),
            _ => None,
        };

        history.push(next_tok);
        generated_ids.push(next_tok);

        let t2 = std::time::Instant::now();
        let text = tokenizer.decode_single(next_tok, true).unwrap_or_default();
        chrome.span(&format!("detok_{step}"), t2);
        if trace_perf {
            detok_ms += t2.elapsed().as_secs_f64() * 1000.0;
        }
        emitted_text.push_str(&text);
        n_decoded += 1;

        let mut hit_stop = false;
        for s in &stop_strings {
            if !s.is_empty() && emitted_text.contains(s.as_str()) {
                hit_stop = true;
                break;
            }
        }

        // UTF-8-safe delta for the client (holds back a multi-byte char
        // split across tokens instead of emitting `��`).
        let out_text = utf8.next_text(next_tok, |ids| {
            tokenizer.decode(ids, true).unwrap_or_default()
        });
        // Emit the token. Receiver-dropped → break out of the loop so
        // the engine releases the state mutex promptly. A merely-full
        // channel spills to `pending` (see above) instead of blocking
        // inside the state lock.
        pending.push_back(Token {
            id: next_tok,
            text: out_text,
            logprobs: token_logprobs,
        });
        if flush_pending(&mut pending, tx) {
            break;
        }

        if hit_stop {
            break;
        }

        if let Some(eos) = model_eos {
            if next_tok == eos {
                break;
            }
        }

        next_input = next_tok as i32;
        next_pos += 1;
        if (next_pos as usize) >= max_ctx {
            // P1: context shift (generate past ctx_size) on the streaming
            // path. `state` is the held EngineState guard. `false` ⇒
            // disabled/inapplicable ⇒ stop as before (byte-identical).
            if !try_context_shift(&model, &mut state.kv_backend, max_ctx, &mut next_pos) {
                break;
            }
        }
    }

    if trace_perf {
        let total = decode_fwd_ms + sample_ms + detok_ms;
        eprintln!(
            "[perf] prefill={prefill_ms:.0}ms ({} tokens) | decode {n_decoded} tokens: fwd={decode_fwd_ms:.0}ms sample={sample_ms:.0}ms detok={detok_ms:.0}ms total={total:.0}ms ({:.1}ms/tok)",
            prompt_ids.len(),
            total / n_decoded.max(1) as f64
        );
    }
    chrome.flush();

    // Publish per-request stats so the server's response handler can
    // pick them up via `CpuEngine::last_request_stats()` while it still
    // holds the per-model gate. Also fold this request's decode tok/s
    // into the engine-wide EMA so `/v1/metrics` returns a smoothed
    // throughput number instead of swinging by 5× between cold and
    // warm requests.
    let tool_call_limit_hit = grammar
        .as_ref()
        .map(|g| g.tool_call_limit_blocked())
        .unwrap_or(false);
    let stats = RequestStats {
        prefill_ms,
        decode_ms: decode_fwd_ms,
        tokens_prefilled: prefill_forwards,
        cache_hit_tokens,
        tokens_generated: n_decoded,
        tool_call_limit_hit,
    };
    update_ema_tok_s(&ema_tok_s_bits, &stats);
    cumulative_stats.add(&stats);
    *last_stats.lock().expect("last_stats") = stats;

    // Snapshot the full sequence (prompt + generated) for prefix-reuse on
    // the next call. Done unconditionally — even when prefix caching is off
    // the snapshot is cheap and harmless, but its only consumer is the LCP
    // check which `prefix_cache=false` skips anyway.
    let mut snapshot = prompt_u32;
    snapshot.extend(generated_ids);
    state.commit_snapshot(snapshot);

    // Deliver any tokens the consumer hasn't accepted yet — with the
    // state mutex RELEASED, so a slow reader only stalls this
    // request's thread, never the engine.
    drop(state);
    for item in pending {
        if tx.blocking_send(Ok(item)).is_err() {
            break;
        }
    }

    Ok(())
}

/// Length of the longest common prefix of two slices.
fn compute_lcp(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// M5-engine-wire: build the per-token 3-D multimodal-RoPE `[t, h, w]`
/// positions for a SPLICED VLM sequence, or `None` when the model isn't an
/// mRoPE VLM (`rope.dimension_sections` unset) or the placeholder runs don't
/// partition into the per-image grids — in which case the caller falls back to
/// the scalar entry (byte-identical 1-D RoPE).
///
/// `placeholder_positions` are the image-placeholder indices in the SPLICED
/// token space. The sectioned-mRoPE family (Qwen2-VL / Qwen2.5-VL) uses the
/// 1:1 `OnePerPatch` splice, so these equal the prompt placeholder positions
/// and `seq_len == prompt_tokens.len()`. `per_image_features` are the vision
/// pipeline's per-image `[num_patches, d_text]` buffers; their patch counts
/// give each image's square post-merge grid `[1, g, g]` (`g = √num_patches`) —
/// the SAME derivation
/// [`rustllama_models::vision_arch::prepare_vlm_inputs_with_mode`] uses, mirrored
/// here because `drive_vlm_generation` builds its splice inline (overlapping the
/// ViT with tokenization) rather than through `prepare_vlm_inputs`. The actual
/// position math is delegated to the models surface
/// ([`rustllama_models::vision_arch::VlmPrefillInputs::mrope_positions_for_sequence`])
/// so the engine never re-implements the grid rule.
fn vlm_mrope_positions(
    rope_dimension_sections: Option<&[u32]>,
    placeholder_positions: &[usize],
    per_image_features: &[Vec<f32>],
    d_text: usize,
    seq_len: usize,
) -> Option<Vec<[u32; 3]>> {
    // Non-mRoPE model → scalar path (the models forward is byte-identical
    // anyway, but gate here so the engine doesn't pay the position build).
    if rope_dimension_sections.is_none() {
        return None;
    }
    let grid_thw: Vec<[u32; 3]> = per_image_features
        .iter()
        .map(|feat| {
            let n = if d_text > 0 { feat.len() / d_text } else { 0 };
            let g = (n as f64).sqrt() as usize;
            if g > 0 && g * g == n {
                [1, g as u32, g as u32]
            } else {
                [1, 1, n as u32]
            }
        })
        .collect();
    let vlm = rustllama_models::vision_arch::VlmPrefillInputs {
        positions: placeholder_positions.to_vec(),
        features: Vec::new(),
        d_text,
        grid_thw,
    };
    vlm.mrope_positions_for_sequence(seq_len, 0)
}

/// Dispatch a single-token forward through whichever KV backend the
/// engine currently holds. Same shape as `model.forward_one(...)`
/// but routes through `forward_one_paged_f32` when the backend is
/// paged.
///
/// Paged callers MUST have a cache with capacity for `pos + 1`
/// tokens. The contiguous path pre-allocates the full max_ctx
/// slab at engine load so this is a non-issue; the paged path
/// grows on demand. We do the grow here (rather than at every
/// per-token call site) so the engine's dispatch surface stays
/// V-6b-3d streaming generation for image-bearing requests. Mirrors
/// [`drive_generation`] but uses [`CpuEngine::vlm_prefill_ids`] for
/// prefill and skips the chunked-prefill / tool-grammar / logprobs
/// machinery (image prefill is a single shot per request).
///
/// **P-1 (VLM prefix cache):** consults the cross-request prefix
/// pool with an `image_hash` filter (FNV-1a 64 over concatenated
/// image bytes). On a full-prefix match the cached KV state is
/// restored and only the *suffix* tokens (the new turn's portion
/// of the prompt) are re-prefilled. Skips the vision pipeline + the
/// from-embeds prefill of the cached prefix entirely — typical
/// multi-turn-same-image chat now reuses the previous turn's KV
/// for everything up to the new user message.
///
/// **P-2 (streaming prefill — VLM analog):** on a cache miss, runs
/// the vision pipeline (per-image ViT forward + projector) on a
/// background thread concurrent with tokenization + text embedding
/// on this thread. Vision is the dominant cost of a fresh VLM
/// prefill (each image runs all ViT layers); overlapping it with
/// the much-cheaper token side saves ~100ms per fresh request.
///
/// v1 constraints:
/// - Contiguous KV backend only (paged + VLM is a v2 enhancement).
/// - Cache reuse requires the new prompt's prefix to be an exact
///   superset of the cached entry (full-prefix match). Partial-trim
///   restoration would require a token→KV-position map per entry;
///   shipping it later if multi-turn workloads start producing
///   diverging branches.
/// - No tool-call grammar / no logprobs surface — coding-VLM use
///   case in v1 is "describe / explain this image"; tool-calling
///   on image requests is a follow-up.
#[allow(clippy::too_many_arguments)]
fn drive_vlm_generation(
    model: Arc<LlamaModel>,
    state: Arc<Mutex<EngineState>>,
    tokenizer: Arc<Tokenizer>,
    vision: Option<Arc<rustllama_models::vision_arch::VisionModel>>,
    image_token_id: Option<u32>,
    placeholder_mode: Option<rustllama_models::vision_arch::PlaceholderMode>,
    max_ctx: usize,
    model_eos: Option<u32>,
    prompt: String,
    images: Vec<Vec<u8>>,
    sampling: SamplingParams,
    feature_memo: Arc<Mutex<Option<(u64, Vec<Vec<f32>>)>>>,
    lora_adapter: Option<Arc<rustllama_models::lora::LoraAdapter>>,
    tx: &tokio::sync::mpsc::Sender<EngineResult<Token>>,
) -> EngineResult<()> {
    use rustllama_models::vision_arch::PlaceholderMode;
    // These should be Some — Engine::chat only routes here when
    // vision is loaded, which atomically sets image_token_id and
    // placeholder_mode. Defense in depth: if any is None, surface
    // a clear error.
    let vision = vision
        .ok_or_else(|| crate::EngineError::Engine("vlm chat: vision tower not loaded".into()))?;
    let image_token_id = image_token_id
        .ok_or_else(|| crate::EngineError::Engine("vlm chat: image_token_id not set".into()))?;
    let placeholder_mode = placeholder_mode.unwrap_or(PlaceholderMode::OnePerImage);

    // P11 (VLM): scope the selected LoRA adapter as the thread-active adapter
    // for the WHOLE VLM generation (prefill-from-embeds + every decode step) on
    // this spawn_blocking thread. The VLM path uses the contiguous forward, so
    // the per-request thread-local `ACTIVE_LORA` applies cleanly — mirroring how
    // the text `DriveGenJob` scopes it. `None` restores/keeps the base
    // (byte-identical). Restored on drop (incl. an early `?`), so the next task
    // on this pool thread starts clean. The guard must outlive every forward,
    // so it is bound here at the top and held for the function body.
    let adapter_active = lora_adapter.is_some();
    let _lora_scope = rustllama_models::lora::scope_active_lora(lora_adapter);

    rustllama_models::accel::set_flash_attention(true);

    // Tokenize the rendered chat-template prompt.
    let add_bos = tokenizer.add_bos_token();
    let prompt_ids: Vec<i32> = tokenizer
        .encode(&prompt, add_bos)
        .map_err(|e| crate::EngineError::Engine(format!("vlm tokenize: {e}")))?
        .into_iter()
        .map(|t| t as i32)
        .collect();
    if prompt_ids.is_empty() {
        return Ok(());
    }

    // Image-bearing prefill runs against the contiguous KV cache.
    // Paged KV + the from-embeds forward path is a v2 surgery.
    let prompt_u32: Vec<u32> = prompt_ids.iter().map(|&t| t as u32).collect();
    // P-1: image_hash discriminates VLM cache entries from text-only
    // entries and from each other across distinct images.
    let image_refs: Vec<&[u8]> = images.iter().map(|v| v.as_slice()).collect();
    let image_hash = Some(crate::prefix_cache::fnv1a64_bytes_chained(&image_refs));

    let mut state_guard = state.lock().expect("state lock");
    // Destructure the lock guard into separate field references so
    // pool + kv_backend can be borrowed mutably in the same scope
    // (Rust's field-level split-borrow rule).
    let EngineState {
        kv_backend,
        last_ids,
        pool,
        delta_net_cache,
    } = &mut *state_guard;

    // Hybrid models: the VLM prefix pool stores KV-only snapshots for
    // image-hash entries (no DeltaNet state), so restoring one would
    // silently drop the recurrent state. v1: skip BOTH the lookup and
    // the post-generation insertion when hybrid — fresh prefill every
    // image turn (the V-7 vision-feature memo softens the cost).
    let is_hybrid = model.weights.is_hybrid();
    // M5-engine-wire: a Qwen2-VL / Qwen2.5-VL (sectioned mRoPE) model
    // COMPRESSES rotation positions wherever images appear, so the KV-cache
    // length diverges from the running mRoPE position. The prefix snapshot
    // records only the KV length (not the compressed mRoPE position), so a
    // suffix-only re-prefill can't continue the sectioned rotation correctly.
    // Treat mRoPE like hybrid: skip BOTH the prefix lookup and the post-gen
    // insertion → always a fresh, position-correct prefill (the V-7
    // vision-feature memo still skips the ViT, so only the cheap text-model
    // prefill repeats).
    let is_mrope = model.cfg.rope_dimension_sections.is_some();
    // P11: the VLM prefix pool is content-keyed (image_hash + prompt ids), NOT
    // adapter-keyed, so an adapted request must neither reuse nor populate it —
    // skip BOTH the lookup and the post-gen insertion when a LoRA adapter is
    // active (mirrors the text path's `prefix_cache && !adapter_active`). `None`
    // adapter ⇒ unchanged / byte-identical.
    let vlm_prefix_cache_ok = !is_hybrid && !is_mrope && !adapter_active;

    // P-1 lookup: try a full-prefix match in the pool for this
    // image_hash. On hit we restore the cached KV state and skip
    // the vision + cached-prefix prefill entirely. The "full
    // prefix" constraint guarantees the cached ids contain all of
    // the image placeholder(s); the suffix (new turn portion) is
    // guaranteed text-only since the user hasn't added a new image.
    let cache_hit = match &kv_backend {
        KvBackend::Contiguous(_) if vlm_prefix_cache_ok => {
            pool.find_full_match_with_image_hash(&prompt_u32, image_hash)
        }
        _ => None,
    };
    // H5: telemetry — record VLM cache lookup outcome.
    pool.record_lookup(cache_hit.is_some());

    let kv = match kv_backend {
        KvBackend::Contiguous(kv) => kv,
        KvBackend::Paged { .. }
        | KvBackend::PagedQ8_0 { .. }
        | KvBackend::PagedTQ { .. }
        | KvBackend::PagedNvfp4 { .. }
        | KvBackend::PagedMxfp4 { .. }
        | KvBackend::PagedMxfp6 { .. }
        | KvBackend::PagedMxfp8 { .. }
        | KvBackend::PagedF16 { .. } => {
            return Err(crate::EngineError::Engine(
                "VLM chat requires the contiguous KV backend in v1 — \
                 set [inference].kv_cache_layout = \"contiguous\""
                    .into(),
            ));
        }
    };

    let (mut logits, next_pos, prefix_token_count, prefix_kv_len, mrope_next) = if let Some(idx) =
        cache_hit
    {
        // Cache hit — restore KV state from the snapshot and
        // re-prefill only the suffix tokens.
        let snap = pool.touch(idx).clone();
        kv.restore_prefix(&snap.kv);
        let prefix_token_count = snap.ids.len();
        let prefix_kv_len = snap.kv_len;
        tracing::debug!(
            cached_tokens = prefix_token_count,
            cached_kv_positions = prefix_kv_len,
            suffix_tokens = prompt_ids.len() - prefix_token_count,
            "vlm prefix cache hit — re-prefilling suffix only"
        );

        // Suffix = the new tokens this request adds beyond the
        // cached prefix. Guaranteed image-free (full-prefix
        // match means the placeholder is already covered).
        let suffix = &prompt_ids[prefix_token_count..];
        if suffix.is_empty() {
            // No new tokens — generate from the cached state.
            // The cached snapshot doesn't carry the post-prefix
            // logits, so we run one forward_one at the boundary
            // to recover them. The "boundary" token is the last
            // cached id; we forward it again at position
            // prefix_kv_len - 1 to read the logits. But the KV
            // for that position is already populated, so do a
            // dummy single-token re-run is unnecessary — instead
            // just synthesize an empty stream. Real-world hit
            // requires at least one new user token after the
            // assistant reply, so suffix should be non-empty in
            // practice.
            let logits = vec![0.0f32; model.cfg.vocab_size];
            (
                logits,
                prefix_kv_len as u32,
                prefix_token_count,
                prefix_kv_len,
                // mRoPE disables the prefix cache (vlm_prefix_cache_ok), so a
                // cache hit only ever happens on a non-mRoPE model → scalar.
                None,
            )
        } else {
            // (hybrid never reaches here — vlm_prefix_cache_ok
            // gated the lookup — so the dense from-embeds forward
            // is always the right one. Primal rows == raw rows on
            // non-Hadamard dense models.)
            let suffix_embeds = model.embed_tokens_primal(suffix);
            let logits =
                model.forward_prefill_from_embeds(&suffix_embeds, prefix_kv_len as u32, kv);
            let next_pos = (prefix_kv_len + suffix.len()) as u32;
            (
                logits,
                next_pos,
                prefix_token_count,
                prefix_kv_len + suffix.len(),
                None,
            )
        }
    } else {
        // Miss: full fresh prefill. P-2 streaming-prefill (VLM
        // analog): run the vision pipeline (per-image ViT
        // forward + projector) on a background thread concurrent
        // with tokenize-side work (placeholder-position scan +
        // text embedding lookup) on this thread. Vision is the
        // dominant cost of a fresh VLM prefill; overlapping it
        // with the much-cheaper token side saves ~100ms+ of
        // wall-clock latency per fresh request.
        last_ids.clear();
        kv.reset();
        if let Some(dn) = delta_net_cache.as_mut() {
            dn.reset();
        }

        // V-7 feature memo: a repeated image set (same chained
        // hash) reuses the projected features and skips the ViT
        // entirely. `image_hash` is Some(..) on this path.
        let memo_hit: Option<Vec<Vec<f32>>> = image_hash.and_then(|h| {
            feature_memo
                .lock()
                .ok()
                .and_then(|g| g.as_ref().filter(|(k, _)| *k == h).map(|(_, f)| f.clone()))
        });
        if memo_hit.is_some() {
            tracing::debug!("vlm prefill: vision-feature memo hit — skipping ViT");
        }
        let vision_for_thread = Arc::clone(&vision);
        let images_for_thread: Vec<Vec<u8>> = images.clone();
        let memo_for_thread = memo_hit;
        let vision_handle =
            std::thread::spawn(move || -> std::result::Result<Vec<Vec<f32>>, String> {
                if let Some(cached) = memo_for_thread {
                    return Ok(cached);
                }
                let mut features = Vec::with_capacity(images_for_thread.len());
                for img_bytes in &images_for_thread {
                    let feat = vision_for_thread
                        .forward_image_bytes(img_bytes)
                        .map_err(|e| e.to_string())?;
                    features.push(feat);
                }
                Ok(features)
            });

        // ----- Concurrent work on the main thread -----
        // (1) Placeholder-position scan over the tokenized prompt.
        //     The validation differs between modes:
        //     - OnePerImage: positions.len() == images.len()
        //     - OnePerPatch: runs of consecutive placeholders
        //       must match per-image patch counts
        let positions: Vec<usize> = prompt_u32
            .iter()
            .enumerate()
            .filter_map(|(i, &t)| if t == image_token_id { Some(i) } else { None })
            .collect();
        // OnePerImage mode validates here so the vision-thread
        // resources free cleanly on a count mismatch. OnePerPatch
        // mode needs the per-image patch counts from the vision
        // pipeline to validate, so its check runs after the join.
        if matches!(placeholder_mode, PlaceholderMode::OnePerImage)
            && positions.len() != images.len()
        {
            let _ = vision_handle.join();
            return Err(crate::EngineError::Engine(format!(
                "vlm prefill: prompt contains {} image-placeholder tokens \
                     (id={image_token_id}) but the request attached {} image \
                     payloads",
                positions.len(),
                images.len(),
            )));
        }
        // (2) Embed text tokens. CPU work, runs concurrent with
        //     the spawned vision pipeline.
        let text_embeds = model.embed_tokens_primal(&prompt_ids);
        let d_text = vision.projector.d_text();

        // ----- Join the vision thread -----
        let features = vision_handle
            .join()
            .map_err(|_| crate::EngineError::Engine("vlm prefill: vision thread panicked".into()))?
            .map_err(|e| crate::EngineError::Engine(format!("vlm prefill: {e}")))?;

        // V-7: memoize the (possibly fresh) features for the next
        // turn with the same image set.
        if let (Some(h), Ok(mut g)) = (image_hash, feature_memo.lock()) {
            *g = Some((h, features.clone()));
        }

        // OnePerPatch validation: count consecutive placeholder
        // runs in the prompt and check they match per-image patch
        // counts from the vision pipeline.
        let per_patch_features: Vec<Vec<f32>>;
        let final_positions: Vec<usize>;
        let final_feature_slabs: Vec<&[f32]>;
        match placeholder_mode {
            PlaceholderMode::OnePerImage => {
                final_positions = positions;
                per_patch_features = Vec::new();
                final_feature_slabs = features.iter().map(|f| f.as_slice()).collect();
                let _ = &per_patch_features;
            }
            PlaceholderMode::OnePerPatch => {
                // Compute consecutive-run lengths in `positions`.
                let mut run_lengths: Vec<usize> = Vec::new();
                let mut current_run = 0usize;
                let mut prev_pos: Option<usize> = None;
                for &p in &positions {
                    match prev_pos {
                        Some(prev) if p == prev + 1 => current_run += 1,
                        _ => {
                            if current_run > 0 {
                                run_lengths.push(current_run);
                            }
                            current_run = 1;
                        }
                    }
                    prev_pos = Some(p);
                }
                if current_run > 0 {
                    run_lengths.push(current_run);
                }
                let patches_per_image: Vec<usize> =
                    features.iter().map(|f| f.len() / d_text).collect();
                if run_lengths != patches_per_image {
                    return Err(crate::EngineError::Engine(format!(
                        "vlm prefill (OnePerPatch): placeholder runs \
                             {run_lengths:?} don't match per-image patch counts \
                             {patches_per_image:?}"
                    )));
                }
                // Split each image's [num_patches, d_text] buffer
                // into per-patch [d_text] rows aligned with
                // positions (1:1 placeholder→patch substitution).
                let total_patches: usize = patches_per_image.iter().sum();
                let mut split: Vec<Vec<f32>> = Vec::with_capacity(total_patches);
                for feat in &features {
                    let n_patches = feat.len() / d_text;
                    for p in 0..n_patches {
                        let row = feat[p * d_text..(p + 1) * d_text].to_vec();
                        split.push(row);
                    }
                }
                per_patch_features = split;
                final_positions = positions;
                final_feature_slabs = per_patch_features.iter().map(|f| f.as_slice()).collect();
            }
        }

        let total_image_rows: usize = match placeholder_mode {
            PlaceholderMode::OnePerImage => features.iter().map(|f| f.len() / d_text).sum(),
            // OnePerPatch: total_image_rows == positions.len()
            // because each placeholder maps 1:1 to one patch row,
            // and the splice doesn't change the total length.
            PlaceholderMode::OnePerPatch => final_positions.len(),
        };
        let spliced_len = prompt_ids.len() - final_positions.len() + total_image_rows;
        if spliced_len + (sampling.max_tokens as usize) > max_ctx {
            return Err(crate::EngineError::Engine(format!(
                "vlm prefill too long: spliced_len={spliced_len} + \
                     max_tokens={} > max_ctx={max_ctx}",
                sampling.max_tokens
            )));
        }
        let feat_slices: Vec<&[f32]> = final_feature_slabs;
        let positions = final_positions;
        let spliced = rustllama_models::vision_arch::splice_image_embeddings(
            &text_embeds,
            &positions,
            &feat_slices,
            d_text,
        )
        .map_err(|e| crate::EngineError::Engine(format!("vlm splice: {e}")))?;
        let mut mrope_next: Option<u32> = None;
        let logits = if is_hybrid {
            let dn = delta_net_cache.as_mut().ok_or_else(|| {
                crate::EngineError::Engine(
                    "vlm prefill: hybrid model without a DeltaNet cache".into(),
                )
            })?;
            model.forward_prefill_hybrid_from_embeds(&spliced, 0, kv, dn)
        } else if let Some(mrope) = vlm_mrope_positions(
            model.cfg.rope_dimension_sections.as_deref(),
            &positions,
            &features,
            d_text,
            spliced_len,
        ) {
            // M5-engine-wire: Qwen2-VL / Qwen2.5-VL multimodal sectioned RoPE.
            // Each spliced token carries its 3-D (t,h,w) position (image tokens
            // = patch-grid coords, text = (p,p,p)); the KV slot stays the
            // physical position. Decode then continues at (max 3-D position
            // used) + 1, advancing uniformly per step.
            mrope_next = Some(
                mrope
                    .iter()
                    .map(|p| p[0].max(p[1]).max(p[2]))
                    .max()
                    .unwrap_or(0)
                    + 1,
            );
            model.forward_prefill_from_embeds_mrope(&spliced, 0, &mrope, kv)
        } else {
            model.forward_prefill_from_embeds(&spliced, 0, kv)
        };
        (
            logits,
            spliced_len as u32,
            prompt_ids.len(),
            spliced_len,
            mrope_next,
        )
    };

    let _ = prefix_token_count; // structured here for readability + future logging

    // Decode loop. After prefill, the KV cache holds the spliced
    // length of positions; the next forward_one runs at the live
    // seq_len position.
    let mut sampler = Sampler::new(sampling.clone());
    let mut history: Vec<u32> = prompt_u32;
    let mut next_pos = next_pos;
    // M5-engine-wire: the running mRoPE rotation position for decode
    // (`Some` only for a sectioned mRoPE prefill that produced 3-D
    // positions). It advances uniformly and INDEPENDENTLY of `next_pos`
    // (the physical KV slot), because mRoPE compresses rotation positions
    // where images appeared.
    let mut mrope_pos = mrope_next;
    let mut spliced_len_for_snapshot = prefix_kv_len;
    for _ in 0..sampling.max_tokens {
        if tx.is_closed() {
            break;
        }
        let next_tok = sampler.sample_with_grammar(&mut logits, &history, None);
        // EOS gate: stop without emitting the EOS token itself —
        // matches the text-only chat path's behavior.
        if let Some(eos) = model_eos {
            if next_tok == eos {
                break;
            }
        }
        history.push(next_tok);
        let text = tokenizer
            .decode_single(next_tok, true)
            .unwrap_or_else(|_| format!("<{next_tok}>"));
        // Best-effort send — if the consumer dropped we stop the loop
        // on the next is_closed() check.
        if tx
            .blocking_send(Ok(Token {
                id: next_tok,
                text,
                logprobs: None,
            }))
            .is_err()
        {
            break;
        }
        // Advance position + forward_one for the next iteration.
        // `kv` is &mut KvCache; the forward_one_via_backend helper
        // wants `&mut KvBackend`, so call the per-family forward
        // directly on the contiguous cache to avoid re-borrowing
        // kv_backend.
        if is_hybrid {
            let dn = delta_net_cache.as_mut().ok_or_else(|| {
                crate::EngineError::Engine(
                    "vlm decode: hybrid model without a DeltaNet cache".into(),
                )
            })?;
            model.forward_one_hybrid(next_tok as i32, next_pos, kv, dn, &mut logits);
        } else if let Some(m) = mrope_pos {
            // M5-engine-wire: sectioned-mRoPE decode. The rotation position
            // advances uniformly (text after an image continues at
            // max(grid)+1, +1 per step); the KV slot / causal pos stays the
            // physical `next_pos`. `forward_one` has no 3-D entry, so embed
            // the token and route through `forward_one_from_embed_mrope`
            // (primal == raw rows on these dense non-Hadamard VLM decoders).
            let embed = model.embed_tokens_primal(&[next_tok as i32]);
            model.forward_one_from_embed_mrope(&embed, next_pos, Some([m, m, m]), kv, &mut logits);
            mrope_pos = Some(m + 1);
        } else {
            model.forward_one(next_tok as i32, next_pos, kv, &mut logits);
        }
        next_pos += 1;
        spliced_len_for_snapshot += 1;
        if (next_pos as usize) >= max_ctx {
            break;
        }
    }

    // P-1: snapshot the post-generation state into the prefix pool
    // with the request's image_hash. The next turn with the same
    // image + an extending prompt will find this as a full-prefix
    // match and skip the vision pipeline + cached-prefix prefill.
    //
    // `kv` is still the contiguous backend we've been writing to;
    // `pool` is the destructured prefix pool from the top. The
    // split-borrow above lets us call snapshot_and_insert here
    // without a re-borrow of state_guard.
    if vlm_prefix_cache_ok
        && spliced_len_for_snapshot >= PREFIX_REUSE_MIN_TOKENS
        && history.len() >= PREFIX_REUSE_MIN_TOKENS
    {
        pool.snapshot_and_insert_with_image_hash(history.clone(), kv, image_hash);
    }
    *last_ids = history;
    Ok(())
}

/// P1: on hitting `max_ctx` mid-generation, try a llama.cpp-style context
/// shift so generation CONTINUES instead of hard-stopping. Keeps the first
/// `keep` positions (the BOS/system prefix), drops the next chunk, and
/// re-RoPE's the retained suffix down to stay contiguous (see
/// [`KvCache::context_shift`]). Returns `true` when the shift applied —
/// `*next_pos` is set to the new (lower) write position and the caller keeps
/// going — or `false` when context-shift is disabled or inapplicable, in
/// which case the caller stops exactly as it did before (byte-identical).
///
/// Scope: contiguous **F32** KV + un-scaled NeoX RoPE only. Scaled-RoPE
/// models (the K rows carry a scaled inv-freq the base-theta shift kernel
/// doesn't reproduce), quant/half KV, and the paged backends all return
/// `false` (keep the hard-stop) — fail-safe, never a silent-corruption shift.
fn try_context_shift(
    model: &LlamaModel,
    backend: &mut KvBackend,
    max_ctx: usize,
    next_pos: &mut u32,
) -> bool {
    if !rustllama_models::accel::context_shift_enabled() {
        return false;
    }
    if model.cfg.rope_scaling.is_some() {
        return false;
    }
    let KvBackend::Contiguous(kv) = backend else {
        return false;
    };
    if !kv.is_uniform_f32() {
        return false;
    }
    // Keep the configured prefix (clamped so there is always room to discard).
    let keep = (rustllama_models::accel::context_shift_keep() as usize).min(max_ctx / 2);
    // Discard half of the non-kept span — llama.cpp's `(n_ctx - n_keep) / 2`.
    let discard = max_ctx.saturating_sub(keep) / 2;
    if discard == 0 {
        return false;
    }
    if !kv.context_shift(keep, discard, model.cfg.rope_theta) {
        return false;
    }
    *next_pos = kv.seq_len as u32;
    true
}

/// Sort `(token_id, logit)` pairs by logit DESCENDING, ties broken by the
/// LOWER token id — so index 0 is the argmax with the same lowest-index
/// tie-break as [`crate::sampling::argmax`].
fn sort_desc_logit_then_idx(v: &mut [(u32, f32)]) {
    v.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
}

/// P12b beam-search helper: select the top-`k` tokens by raw logit (lowest-id
/// tie-break, matching the greedy argmax) and return each as
/// `(token, logprob)` where `logprob = logit - log_sum_exp(logits)`.
/// Deterministic. The top element is always the argmax token, so a single
/// beam's top candidate equals greedy's pick.
pub(crate) fn top_k_by_logit(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let k = k.min(logits.len()).max(1);
    let lse = rustllama_kernels_cpu::log_sum_exp_f32(logits);
    // Single-pass top-k by logit: maintain a length-k running buffer sorted
    // descending; a strict `>` comparison keeps the lower index on ties.
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(k);
    for (i, &v) in logits.iter().enumerate() {
        if top.len() < k {
            top.push((i as u32, v));
            if top.len() == k {
                sort_desc_logit_then_idx(&mut top);
            }
            continue;
        }
        if v > top[k - 1].1 {
            top[k - 1] = (i as u32, v);
            let mut j = k - 1;
            while j > 0 && top[j].1 > top[j - 1].1 {
                top.swap(j, j - 1);
                j -= 1;
            }
        }
    }
    // When the vocab is smaller than k the buffer was never sorted above.
    if top.len() < k {
        sort_desc_logit_then_idx(&mut top);
    }
    top.into_iter()
        .map(|(id, logit)| (id, logit - lse))
        .collect()
}

/// one-line and the grow logic is owned by exactly one function.
/// Panics on grow failure — the engine's max_ctx clamp guarantees
/// the page pool was sized to hold one full request, so a failure
/// here is a bug, not a runtime condition.
#[inline]
fn forward_one_via_backend(
    model: &LlamaModel,
    backend: &mut KvBackend,
    token_id: i32,
    pos: u32,
    logits_out: &mut [f32],
) {
    forward_one_via_backend_hybrid(model, backend, None, token_id, pos, logits_out)
}

/// Variant that takes an optional DeltaNet cache; when the loaded
/// model is hybrid the caller threads its cache here. Standard
/// transformer models route through the existing dense / MoE forward
/// paths unchanged.
#[inline]
fn forward_one_via_backend_hybrid(
    model: &LlamaModel,
    backend: &mut KvBackend,
    delta_net_cache: Option<&mut rustllama_models::llama_arch::DeltaNetCache>,
    token_id: i32,
    pos: u32,
    logits_out: &mut [f32],
) {
    if model.weights.is_hybrid() {
        let dn = delta_net_cache
            .expect("engine bug: hybrid model loaded but no DeltaNetCache passed to forward_one");
        match backend {
            KvBackend::Contiguous(kv) => {
                model.forward_one_hybrid(token_id, pos, kv, dn, logits_out)
            }
            KvBackend::Paged { .. }
            | KvBackend::PagedQ8_0 { .. }
            | KvBackend::PagedTQ { .. }
            | KvBackend::PagedNvfp4 { .. }
            | KvBackend::PagedMxfp4 { .. }
            | KvBackend::PagedMxfp6 { .. }
            | KvBackend::PagedMxfp8 { .. }
            | KvBackend::PagedF16 { .. } => {
                // Hybrid models don't route through the paged backend —
                // the hybrid forward only supports contiguous KV.
                panic!(
                    "hybrid model + paged KV backend not supported — \
                     use [inference].kv_cache_layout = \"contiguous\" for \
                     hybrid models"
                );
            }
        }
        return;
    }
    match backend {
        KvBackend::Contiguous(kv) => model.forward_one(token_id, pos, kv, logits_out),
        KvBackend::Paged {
            cache,
            store,
            table,
        } => {
            cache
                .ensure_capacity(table, pos + 1)
                .expect("paged cache grow failed — pool was sized to max_ctx at engine load");
            model.forward_one_paged_f32(token_id, pos, cache, store, logits_out);
        }
        KvBackend::PagedQ8_0 {
            cache,
            store,
            table,
        } => {
            // H9a: Q8_0 paged forward dispatch — routes through the
            // shared `forward_one_paged_*_inner` via the
            // `PagedKvStoreOps` trait object.
            cache
                .ensure_capacity(table, pos + 1)
                .expect("paged cache grow failed — pool was sized to max_ctx at engine load");
            model.forward_one_paged_q8_0(token_id, pos, cache, store, logits_out);
        }
        KvBackend::PagedTQ {
            cache,
            store,
            table,
        } => {
            // H9b: TurboQuant paged forward dispatch.
            cache
                .ensure_capacity(table, pos + 1)
                .expect("paged cache grow failed — pool was sized to max_ctx at engine load");
            model.forward_one_paged_tq(token_id, pos, cache, store, logits_out);
        }
        KvBackend::PagedNvfp4 {
            cache,
            store,
            table,
        } => {
            // H9b: NVFP4 paged forward dispatch.
            cache
                .ensure_capacity(table, pos + 1)
                .expect("paged cache grow failed — pool was sized to max_ctx at engine load");
            model.forward_one_paged_nvfp4(token_id, pos, cache, store, logits_out);
        }
        KvBackend::PagedMxfp4 {
            cache,
            store,
            table,
        } => {
            // Wave 2: MXFP4 paged forward dispatch.
            cache
                .ensure_capacity(table, pos + 1)
                .expect("paged cache grow failed — pool was sized to max_ctx at engine load");
            model.forward_one_paged_mxfp4(token_id, pos, cache, store, logits_out);
        }
        KvBackend::PagedMxfp6 {
            cache,
            store,
            table,
        } => {
            // Wave 2: MXFP6 paged forward dispatch.
            cache
                .ensure_capacity(table, pos + 1)
                .expect("paged cache grow failed — pool was sized to max_ctx at engine load");
            model.forward_one_paged_mxfp6(token_id, pos, cache, store, logits_out);
        }
        KvBackend::PagedMxfp8 {
            cache,
            store,
            table,
        } => {
            // Wave 2: MXFP8 paged forward dispatch.
            cache
                .ensure_capacity(table, pos + 1)
                .expect("paged cache grow failed — pool was sized to max_ctx at engine load");
            model.forward_one_paged_mxfp8(token_id, pos, cache, store, logits_out);
        }
        KvBackend::PagedF16 {
            cache,
            store,
            table,
        } => {
            // M6: half-precision (f16 / bf16) paged forward dispatch.
            cache
                .ensure_capacity(table, pos + 1)
                .expect("paged cache grow failed — pool was sized to max_ctx at engine load");
            model.forward_one_paged_f16(token_id, pos, cache, store, logits_out);
        }
    }
}

/// E4 phase 4c: single-call forward + MTP-driven draft proposal.
///
/// Runs one main forward to produce `main_logits` for `token_id`
/// at `pos`, then for each MTP head consumes the head's logit row
/// via [`crate::speculative::MtpDrafter::propose`] to produce a
/// `Vec<DraftToken>` (length == `cfg.n_mtp_heads`). The drafts are
/// what the speculation verifier consumes; the main logits are
/// the next-token distribution from which the caller commits the
/// "anchor" token before verification proceeds.
///
/// Returns `(main_logits, Vec::new())` when the model has no MTP
/// heads — callers should then fall through to the existing
/// non-speculative or n-gram-drafter path.
///
/// Contiguous-only for now: paged KV doesn't yet have a paged
/// counterpart of `LlamaModel::forward_one_with_mtp_logits`. The
/// dispatcher fast-fails on paged with `EngineError::Unimplemented`
/// rather than silently falling back, so the engine surface stays
/// honest about which backend supports MTP.
#[allow(dead_code)] // Reserved for the streaming MTP dispatch path.
fn forward_one_with_mtp_drafts_via_backend(
    model: &LlamaModel,
    backend: &mut KvBackend,
    token_id: i32,
    pos: u32,
    drafter: crate::speculative::MtpDrafter,
) -> std::result::Result<(Vec<f32>, Vec<crate::speculative::DraftToken>), crate::EngineError> {
    let cfg = &model.cfg;
    let mut main_logits = vec![0f32; cfg.vocab_size];
    let n_heads = cfg.n_mtp_heads as usize;
    let mut head_logits: Vec<Vec<f32>> = (0..n_heads).map(|_| vec![0f32; cfg.vocab_size]).collect();

    match backend {
        KvBackend::Contiguous(kv) => {
            model.forward_one_with_mtp_logits(
                token_id,
                pos,
                kv,
                &mut main_logits,
                &mut head_logits,
            );
        }
        KvBackend::Paged { .. }
        | KvBackend::PagedQ8_0 { .. }
        | KvBackend::PagedTQ { .. }
        | KvBackend::PagedNvfp4 { .. }
        | KvBackend::PagedMxfp4 { .. }
        | KvBackend::PagedMxfp6 { .. }
        | KvBackend::PagedMxfp8 { .. }
        | KvBackend::PagedF16 { .. } => {
            return Err(crate::EngineError::Unimplemented(
                "MTP drafting on paged KV backend (forward_one_with_mtp_logits_paged is a \
                 follow-up; use the contiguous backend for MTP-driven speculation)",
            ));
        }
    }

    let drafts = drafter.propose(&head_logits);
    Ok((main_logits, drafts))
}

/// MoE tiered-expert VRAM-budget projection (P20) — the pure, unit-tested core of
/// the `auto` device-tier promotion. `tier_avail` is the device's VRAM proxy:
/// `Some(free_vram)` on CUDA, `Some(total_vram)` on a dedicated SYCL GPU
/// (Arc/PVC), or `None` off the tier — an integrated iGPU (Iris Xe), Apple MLX
/// unified memory, or no usable GPU, where there is no dedicated VRAM to promote
/// into. Returns the bytes of experts to promote: `0` unless this is a MoE model
/// on a tier-capable device, else `avail − non_expert_weights − margin`, saturating
/// to 0 and capped at `expert_bytes`. The `None`/non-MoE/over-margin cases all
/// yield 0 → no promotion (inert, so an over-estimate or an unsupported device
/// degrades to "run on CPU/host", never OOM or wrong output).
fn project_moe_tier_budget_bytes(
    tier_avail: Option<u64>,
    is_moe: bool,
    expert_bytes: u64,
    non_expert_bytes: u64,
    margin: u64,
) -> u64 {
    match tier_avail {
        Some(avail) if is_moe => avail
            .saturating_sub(non_expert_bytes)
            .saturating_sub(margin)
            .min(expert_bytes),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        compute_lcp, project_moe_tier_budget_bytes, update_ema_tok_s, vlm_mrope_positions,
        RequestStats, TOK_S_EMA_ALPHA,
    };
    use std::sync::atomic::{AtomicU64, Ordering};

    fn make_stats(decode_ms: f64, n: u32) -> RequestStats {
        RequestStats {
            prefill_ms: 0.0,
            decode_ms,
            tokens_prefilled: 0,
            cache_hit_tokens: 0,
            tokens_generated: n,
            tool_call_limit_hit: false,
        }
    }

    fn ema_value(slot: &AtomicU64) -> Option<f64> {
        let v = f64::from_bits(slot.load(Ordering::Relaxed));
        if v.is_nan() {
            None
        } else {
            Some(v)
        }
    }

    // ---- M5-engine-wire: the engine's VLM mRoPE position builder ----

    /// Non-mRoPE model (`rope.dimension_sections` unset) ⇒ the engine's
    /// position builder returns `None`, so `drive_vlm_generation` /
    /// `vlm_prefill_ids` fall back to the scalar forward (byte-identical
    /// 1-D RoPE). This is the default-preservation half of Tail 1.
    #[test]
    fn vlm_mrope_positions_none_without_sections() {
        // d_text = 2, one image of 4 patches (→ square 1x2x2 grid).
        let feats = vec![vec![0.0f32; 4 * 2]];
        assert!(
            vlm_mrope_positions(None, &[1, 2, 3, 4], &feats, 2, 7).is_none(),
            "a model without rope.dimension_sections must take the scalar path"
        );
    }

    /// mRoPE model ⇒ the engine hands the forward per-token 3-D positions
    /// that follow the HF `get_rope_index` grid rule: text tokens get
    /// (p,p,p); an image's patches get their (t,h,w) grid coords based at
    /// the image-start position; and the text AFTER an image resumes at
    /// `max(image 3-D positions) + 1` (NOT the flat patch count). This is
    /// the position-correctness half of Tail 1 — the engine supplies the
    /// grid-ruled positions the sectioned kernel consumes.
    #[test]
    fn vlm_mrope_positions_follows_grid_rule_image_then_text() {
        // Prompt layout: 1 text token, then a 4-patch image at placeholder
        // positions [1,2,3,4] (OnePerPatch is 1:1), then 2 trailing text
        // tokens → spliced seq_len = 7. d_text = 2, so the one image's
        // 4 patches give a square grid [1, 2, 2].
        let d_text = 2;
        let feats = vec![vec![0.0f32; 4 * d_text]];
        let positions = [1usize, 2, 3, 4];
        let mrope = vlm_mrope_positions(Some(&[1, 1, 1]), &positions, &feats, d_text, 7)
            .expect("an mRoPE model with a partitioning layout must build positions");

        // Expected per-token triples (segments: Text(1), Image{1,2,2}, Text(2)
        // seeded at 0): text before, the frame-major/row-major/col image grid
        // based at the image-start position 1, then text resuming at 3.
        let expected: Vec<[u32; 3]> = vec![
            [0, 0, 0], // text token 0
            [1, 1, 1], // image patch (hh=0, ww=0)
            [1, 1, 2], // image patch (hh=0, ww=1)
            [1, 2, 1], // image patch (hh=1, ww=0)
            [1, 2, 2], // image patch (hh=1, ww=1)
            [3, 3, 3], // text resumes at max(image)+1 == 3 (NOT 1 + 4 patches)
            [4, 4, 4],
        ];
        assert_eq!(mrope, expected, "mRoPE positions must follow the grid rule");

        // The decode seed `drive_vlm_generation` computes = max component + 1.
        let decode_next = mrope
            .iter()
            .map(|p| p[0].max(p[1]).max(p[2]))
            .max()
            .unwrap()
            + 1;
        assert_eq!(
            decode_next, 5,
            "decode must continue uniformly at max(3-D position)+1"
        );
    }

    /// A placeholder layout that doesn't partition into the per-image grids
    /// (here: 4 placeholders but a 9-patch image) ⇒ `None`, so the engine
    /// degrades to scalar rather than mis-rotating.
    #[test]
    fn vlm_mrope_positions_none_on_grid_mismatch() {
        let d_text = 2;
        let feats = vec![vec![0.0f32; 9 * d_text]]; // 9 patches (3x3)
        let positions = [1usize, 2, 3, 4]; // only 4 placeholder slots
        assert!(
            vlm_mrope_positions(Some(&[1, 1, 1]), &positions, &feats, d_text, 7).is_none(),
            "a grid/placeholder-run mismatch must fall back to scalar (None)"
        );
    }

    // ---- P20: MoE tiered-expert VRAM-budget projection ----
    const GIB: u64 = 1024 * 1024 * 1024;

    /// CUDA-like: plenty of free VRAM, a MoE model → promote `avail − non_expert −
    /// margin`, capped at the expert bytes. 16 GiB free, 2 GiB non-expert, 1.5 GiB
    /// margin, 20 GiB of experts → 12.5 GiB (budget < expert total, not capped).
    #[test]
    fn moe_tier_budget_cuda_like_positive() {
        let avail = 16 * GIB;
        let got = project_moe_tier_budget_bytes(
            Some(avail),
            true,
            20 * GIB,    // expert_bytes
            2 * GIB,     // non_expert_bytes
            3 * GIB / 2, // margin
        );
        assert_eq!(got, 16 * GIB - 2 * GIB - 3 * GIB / 2);
        assert!(got > 0);
    }

    /// Dedicated-SYCL-like: same positive projection path (the fn is backend
    /// agnostic — `Some(total_vram)` behaves exactly like CUDA's `Some(free)`).
    /// Here the expert total is the binding cap (budget would exceed it).
    #[test]
    fn moe_tier_budget_dedicated_sycl_capped_at_experts() {
        let avail = 24 * GIB;
        let got = project_moe_tier_budget_bytes(
            Some(avail),
            true,
            6 * GIB, // expert_bytes — the cap
            2 * GIB,
            1 * GIB,
        );
        assert_eq!(got, 6 * GIB, "budget must cap at total expert bytes");
        assert!(got > 0);
    }

    /// Unified / MLX / integrated / no-GPU: `tier_avail = None` → budget 0, no
    /// promotion (the tier is inert by construction, not forced off).
    #[test]
    fn moe_tier_budget_unified_or_mlx_is_zero() {
        assert_eq!(
            project_moe_tier_budget_bytes(None, true, 20 * GIB, 2 * GIB, GIB),
            0
        );
    }

    /// Non-MoE model on a tier-capable device → budget 0 (nothing to promote).
    #[test]
    fn moe_tier_budget_non_moe_is_zero() {
        assert_eq!(
            project_moe_tier_budget_bytes(Some(16 * GIB), false, 0, 2 * GIB, GIB),
            0
        );
    }

    /// Over-reserved margin (non_expert + margin exceed avail) saturates to 0 —
    /// no promotion rather than an underflow/huge budget.
    #[test]
    fn moe_tier_budget_over_margin_saturates_zero() {
        assert_eq!(
            project_moe_tier_budget_bytes(Some(2 * GIB), true, 20 * GIB, 3 * GIB, GIB),
            0
        );
    }

    #[test]
    fn ema_first_sample_seeds_unsmoothed() {
        // First sample should land verbatim — no smoothing baseline
        // to mix against. Avoids the chicken-and-egg of "what's the
        // smoothed value before any samples".
        let slot = AtomicU64::new(f64::NAN.to_bits());
        // 50 tokens in 1000ms = 50 tok/s.
        update_ema_tok_s(&slot, &make_stats(1000.0, 50));
        let v = ema_value(&slot).expect("seeded");
        assert!((v - 50.0).abs() < 1e-6, "first sample: got {v}");
    }

    #[test]
    fn ema_skips_zero_token_requests() {
        // A cancelled / failed request has tokens_generated = 0 OR
        // decode_ms = 0. We must NOT fold those into the EMA (would
        // either divide-by-zero or pin EMA to 0). Slot stays unset.
        let slot = AtomicU64::new(f64::NAN.to_bits());
        update_ema_tok_s(&slot, &make_stats(1000.0, 0));
        assert!(
            ema_value(&slot).is_none(),
            "zero-token request stayed unfolded"
        );
        update_ema_tok_s(&slot, &make_stats(0.0, 50));
        assert!(
            ema_value(&slot).is_none(),
            "zero-time request stayed unfolded"
        );
        // First valid sample after the skipped ones still seeds.
        update_ema_tok_s(&slot, &make_stats(1000.0, 30));
        assert!((ema_value(&slot).unwrap() - 30.0).abs() < 1e-6);
    }

    #[test]
    fn ema_smooths_subsequent_samples() {
        // After the seed sample, each new sample contributes alpha
        // and the previous EMA contributes (1 - alpha). Verify the
        // formula on a hand-checked sequence.
        let slot = AtomicU64::new(f64::NAN.to_bits());
        update_ema_tok_s(&slot, &make_stats(1000.0, 100)); // seed 100
        update_ema_tok_s(&slot, &make_stats(1000.0, 50)); // smooth toward 50
        let expected = TOK_S_EMA_ALPHA * 50.0 + (1.0 - TOK_S_EMA_ALPHA) * 100.0;
        assert!((ema_value(&slot).unwrap() - expected).abs() < 1e-6);
    }

    #[test]
    fn ema_cold_start_outlier_does_not_dominate() {
        // Failure mode the EMA exists to fix: first request is a
        // cold-start outlier (3 tok/s vs steady-state 60). After
        // several warm requests, the displayed value should be
        // closer to 60 than to 3 — proving the cold seed is
        // recoverable, not a permanent ceiling.
        let slot = AtomicU64::new(f64::NAN.to_bits());
        update_ema_tok_s(&slot, &make_stats(1000.0, 3)); // cold
        for _ in 0..5 {
            update_ema_tok_s(&slot, &make_stats(1000.0, 60)); // warm
        }
        let v = ema_value(&slot).unwrap();
        let dist_to_cold = (v - 3.0).abs();
        let dist_to_warm = (v - 60.0).abs();
        assert!(
            dist_to_warm < dist_to_cold,
            "EMA migrated toward warm samples: v={v} (warm=60, cold=3)"
        );
    }

    #[test]
    fn lcp_empty_slices() {
        assert_eq!(compute_lcp(&[], &[]), 0);
        assert_eq!(compute_lcp(&[1, 2, 3], &[]), 0);
        assert_eq!(compute_lcp(&[], &[4, 5, 6]), 0);
    }

    #[test]
    fn lcp_identical_slices() {
        assert_eq!(compute_lcp(&[1, 2, 3], &[1, 2, 3]), 3);
    }

    #[test]
    fn lcp_one_is_prefix_of_other() {
        assert_eq!(compute_lcp(&[1, 2, 3, 4, 5], &[1, 2, 3]), 3);
        assert_eq!(compute_lcp(&[1, 2], &[1, 2, 9, 9]), 2);
    }

    #[test]
    fn lcp_diverge_in_middle() {
        assert_eq!(compute_lcp(&[1, 2, 3, 4], &[1, 2, 99, 4]), 2);
    }

    #[test]
    fn lcp_no_common_prefix() {
        assert_eq!(compute_lcp(&[1, 2, 3], &[9, 8, 7]), 0);
    }
}
