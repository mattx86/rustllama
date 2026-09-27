//! Generation engine.
//!
//! Production inference is served by [`CpuEngine`] (and the SYCL/CUDA
//! accelerated paths layered on top). [`MockEngine`] is a lightweight
//! test-only fixture — a canned token stream that lets `rustllama-server`
//! exercise its HTTP/API layer without loading a real model; it is never
//! used to serve real requests. The [`Engine`] trait shape here is what
//! `rustllama-server` and the CLI's `chat` REPL depend on, so keep it
//! stable.

use std::pin::Pin;

use futures::Stream;
use serde::{Deserialize, Serialize};

pub mod batch_scheduler;
pub mod cpu;
pub mod expert_cache;
pub mod grammar;
pub mod kv_backend;
pub mod kv_persist;
pub mod measurement;
pub mod memory_budget;
pub mod multi_gpu;
pub mod pagelock;
pub mod paged_batch;
pub mod prefix_cache;
pub mod placement_auto;
pub mod sampler_gpu;
pub mod sampling;
pub mod speculative;
pub mod sycl_accel;
pub mod sycl_resources;
pub mod tool_grammar;

pub use cpu::{CpuEngine, CpuEngineError};
pub use rustllama_models::imatrix_collect;
pub use rustllama_models::kv_bias;
pub use rustllama_models::llama_arch::kv_whitening_active_for;
pub use rustllama_models::vision_arch::PlaceholderMode;
pub use rustllama_models::llama_arch::KvDtype;
pub use sycl_accel::{SyclAccelerator, SyclAccelError};
pub use sycl_resources::{SyclEngineResources, SyclResourcesError, UsmRawBuffer};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Image payloads attached to this message, in the order the wire
    /// protocol presented them. Each `Vec<u8>` is raw image bytes
    /// (PNG or JPEG) — the decoder layer (`rustllama-server::image_url`)
    /// produced these from `data:image/...;base64,...` URIs.
    ///
    /// Empty for the common text-only path. The server attaches images
    /// only when the wire content was a multimodal block array
    /// containing one or more `image_url` blocks; the corresponding
    /// `content` string still holds the placeholder markers so a
    /// vision-aware engine knows where to splice the projected
    /// features in. Text-only engines reject any message where this
    /// is non-empty via [`EngineError::VisionNotSupported`].
    ///
    /// Default empty so serde-deserialized messages from older
    /// clients (no `images` field) still parse cleanly.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<Vec<u8>>,
}

impl ChatMessage {
    /// Construct a plain text-only message. Convenience for the
    /// many call sites (tests, CLI, anthropic adapter, etc.) that
    /// pre-date the multimodal `images` field.
    pub fn text(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            images: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    /// "Locally Typical" sampling (Meister et al. 2022): keep the
    /// smallest set of tokens whose log-prob is closest to the
    /// distribution's entropy and whose cumulative probability ≥
    /// `typical_p`. `1.0` (default) disables — full vocab kept;
    /// `0.0` disables as well (sampler gate; not "keep one token",
    /// for consistency with [`top_p`]).
    ///
    /// Often used in combination with high temperature: typical-p
    /// truncates by "naturalness" rather than "popularity", which
    /// keeps coherence while letting creative completions surface.
    #[serde(default = "default_typical_p")]
    pub typical_p: f32,
    /// Multiplicative repetition penalty (llama.cpp / vLLM convention).
    /// `1.0` = disabled; typical 1.1.
    pub repeat_penalty: f32,
    /// OpenAI-style additive frequency penalty: each token's logit is
    /// reduced by `frequency_penalty × count(token in history)`.
    /// `0.0` = disabled.
    pub frequency_penalty: f32,
    /// OpenAI-style additive presence penalty: each token that appears in
    /// the history has its logit reduced by `presence_penalty` (once,
    /// regardless of count). `0.0` = disabled.
    pub presence_penalty: f32,
    pub max_tokens: u32,
    pub stop: Vec<String>,
    pub seed: u64,
    /// Number of top alternative logprobs to capture per generated token.
    /// `None` = skip logprob computation entirely (fast path).
    /// `Some(0)` = only the chosen token's logprob (no alternatives).
    /// `Some(N)` = chosen + top-N alternatives.
    pub logprobs: Option<u32>,
    /// Grammar that constrains the output. `None` = unconstrained
    /// (default). `Some(GrammarKind::Json)` = the sampler masks out any
    /// token that would break JSON validity at the current parser
    /// state. Wires up to the OpenAI `response_format: json_object`
    /// request field.
    #[serde(default)]
    pub grammar: Option<GrammarKind>,
    /// Mirostat adaptive-temperature sampler. `0` = disabled (the
    /// classic temperature + top-k + top-p stack runs). `1` = v1
    /// (Zipfian-exponent-driven dynamic top-k), `2` = v2 (surprise-
    /// threshold pruning — what Ollama / llama.cpp default to).
    /// Mirostat aims to keep generation perplexity stable across long
    /// outputs by adjusting its own truncation per token; when set,
    /// `top_k` and `top_p` are bypassed since Mirostat IS the
    /// truncation strategy.
    #[serde(default)]
    pub mirostat: u32,
    /// Mirostat target surprise (cross-entropy in nats per token, for
    /// both v1 and v2). Higher values allow more entropy; lower
    /// values produce more conservative output. Typical: 5.0.
    /// Ignored when `mirostat == 0`.
    #[serde(default = "default_mirostat_tau")]
    pub mirostat_tau: f32,
    /// Mirostat learning rate. Larger values let `mu` track changes
    /// in observed surprise more aggressively; smaller values smooth
    /// the trajectory. Typical: 0.1. Ignored when `mirostat == 0`.
    #[serde(default = "default_mirostat_eta")]
    pub mirostat_eta: f32,
}

fn default_mirostat_tau() -> f32 {
    5.0
}

fn default_mirostat_eta() -> f32 {
    0.1
}

fn default_typical_p() -> f32 {
    1.0
}

/// Grammar kinds the sampler understands. Add new variants here as
/// new constraints land (regex, code-syntax, …).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum GrammarKind {
    /// JSON top-level value (object, array, string, number, bool, null).
    Json,
    /// JSON value that also conforms to a JSON Schema. The schema is
    /// not validated at construction time — unsupported keywords are
    /// silently ignored (over-permissive). See [`grammar::Schema`].
    JsonSchema { schema: grammar::Schema },
    /// Free text + recognized `<tool_call>...</tool_call>` blocks. The
    /// inner body is constrained to
    /// `{"name": "<known>", "arguments": <schema-for-name>}`. The
    /// argument schema is dispatched by the resolved name from the
    /// supplied map.
    ToolCallStream {
        schemas_by_name: std::collections::BTreeMap<String, grammar::Schema>,
        /// Lower bound on the number of complete tool-call bodies before
        /// EOS is allowed. `0` (the default) imposes no floor; the
        /// server sets `1` for OpenAI `tool_choice: "required"` and for
        /// a forced specific function. Defaulted for backward-compatible
        /// deserialization of configs written before this field existed.
        #[serde(default)]
        min_completed: u32,
    },
    /// Code-syntax constraint. v1 enforces bracket balance with
    /// quote-aware string tracking; `language` is accepted from the
    /// wire but not yet used (per-language parsers slot in as
    /// follow-up variants without breaking the existing dispatch).
    /// Wires up to the OpenAI `response_format: {"type": "code",
    /// "language": "python"}` extension.
    Code {
        #[serde(default)]
        language: String,
    },
    /// Regex-constrained output. Anchored at start of generation;
    /// the model must emit bytes that, taken from position 0,
    /// match `pattern`. Wires up to a `response_format: {"type":
    /// "regex", "pattern": "..."}` extension. See
    /// [`grammar::RegexGrammarParser`] for the byte-DFA stepping.
    Regex { pattern: String },
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.95,
            top_k: 40,
            typical_p: default_typical_p(),
            repeat_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            max_tokens: 512,
            stop: Vec::new(),
            seed: 0,
            logprobs: None,
            grammar: None,
            mirostat: 0,
            mirostat_tau: default_mirostat_tau(),
            mirostat_eta: default_mirostat_eta(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Metrics {
    pub tokens_per_second: f32,
    pub context_used: u32,
    pub vram_estimate_mb: u64,
    pub ram_estimate_mb: u64,
    /// Paged-KV pool: total pages allocated to the engine at load.
    /// `0` when the engine uses the contiguous KV layout (the
    /// default) — the GUI Status page hides the paged-pool panel
    /// in that case.
    #[serde(default)]
    pub paged_total_pages: u32,
    /// Paged-KV pool: pages currently on the free list. Drops as
    /// slots prefill and grows back as slots complete and call
    /// `release_shared`. `paged_total_pages - paged_free_pages`
    /// is the live in-use count.
    #[serde(default)]
    pub paged_free_pages: u32,
    /// Active slots in the engine's fused-decode loop. `0` on
    /// `CpuEngine` (which is single-flight at the engine layer —
    /// concurrency comes from forks); meaningful on
    /// `PagedBatchEngine` where the driver thread tracks
    /// in-flight slots.
    #[serde(default)]
    pub paged_active_slots: u32,
}

/// Per-request performance + cache-hit stats, captured at the end of
/// each generation. Read via [`crate::cpu::CpuEngine::last_request_stats`]
/// while the request's server-gate permit is still held — the
/// serial-by-default Semaphore(1) per model means no other request
/// can clobber the snapshot between handler reading it and writing it
/// to the OpenAI `usage` block.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RequestStats {
    /// Wall-clock time spent running prefill forward passes (ms).
    pub prefill_ms: f64,
    /// Wall-clock time spent running decode forward passes (ms).
    pub decode_ms: f64,
    /// Number of prompt-position forwards that actually ran during
    /// prefill (after subtracting any positions skipped via the prefix
    /// cache). Plus [`Self::cache_hit_tokens`], this is the prompt
    /// length minus one (the last prompt token drives the first decode
    /// forward, not a prefill forward).
    pub tokens_prefilled: u32,
    /// Number of prompt positions whose K/V state was reused from the
    /// prefix cache (live `last_ids` LCP + multi-snapshot pool restore
    /// + extended-LCP after restore). Higher is better — it's the part
    /// of the prefill the engine got to skip.
    pub cache_hit_tokens: u32,
    /// Number of decode forwards (== completion tokens emitted, before
    /// any stop-string truncation).
    pub tokens_generated: u32,
    /// `true` if the tool-call streaming grammar's `max_tool_iterations`
    /// cap was hit mid-generation (i.e., the model tried to open
    /// another `<tool_call>` after reaching the cap, and the grammar
    /// rejected the opener byte). When `true`, the server surfaces
    /// `finish_reason: "tool_call_iteration_limit"` on the response.
    pub tool_call_limit_hit: bool,
}

/// Cumulative request stats across the engine's lifetime. Read via
/// [`Engine::cumulative_stats_snapshot`] and surfaced on the metrics
/// endpoint so operators can answer "is the prefix cache paying off?"
/// without scraping per-request logs.
///
/// The atomic counters live on the engine so a single
/// [`RequestStats`] commit at the end of each generation is enough
/// to keep them in sync — no separate accounting bookkeeping is
/// required at the call sites.
#[derive(Debug, Default)]
pub struct CumulativeStats {
    /// Number of successfully-committed requests (each
    /// `commit_request_stats` call increments this once).
    pub total_requests: std::sync::atomic::AtomicU64,
    /// Sum of `tokens_prefilled` across every request — i.e., the
    /// total prompt-token forwards that actually ran (cache misses).
    pub total_tokens_prefilled: std::sync::atomic::AtomicU64,
    /// Sum of `cache_hit_tokens` across every request — i.e., the
    /// total prompt tokens skipped via the prefix cache.
    pub total_cache_hit_tokens: std::sync::atomic::AtomicU64,
    /// Sum of `tokens_generated` across every request.
    pub total_tokens_generated: std::sync::atomic::AtomicU64,
    /// Sum of `prefill_ms` across every request, stored in
    /// microseconds so the counter stays a clean integer
    /// (`AtomicF64` doesn't exist on stable). Read back as ms by
    /// dividing by 1000.0.
    pub total_prefill_us: std::sync::atomic::AtomicU64,
    /// Sum of `decode_ms` across every request, also in
    /// microseconds.
    pub total_decode_us: std::sync::atomic::AtomicU64,
    /// Number of speculation verify rounds that proposed at least
    /// one draft token (rounds with an empty draft carry no
    /// acceptance information and are not counted).
    pub total_spec_rounds: std::sync::atomic::AtomicU64,
    /// Total draft tokens proposed across all speculation rounds
    /// (n-gram and draft-model paths both feed this).
    pub total_spec_drafted: std::sync::atomic::AtomicU64,
    /// Total draft tokens accepted by the target across all rounds.
    /// `total_spec_accepted / total_spec_drafted` is the aggregate
    /// acceptance rate.
    pub total_spec_accepted: std::sync::atomic::AtomicU64,
}

impl CumulativeStats {
    /// Bump every counter by the contents of one finished request.
    /// `Ordering::Relaxed` is fine — the only invariant is
    /// monotonicity, which `fetch_add` provides regardless of
    /// ordering. The snapshot read is a separate concern.
    pub fn add(&self, stats: &RequestStats) {
        use std::sync::atomic::Ordering;
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_tokens_prefilled
            .fetch_add(stats.tokens_prefilled as u64, Ordering::Relaxed);
        self.total_cache_hit_tokens
            .fetch_add(stats.cache_hit_tokens as u64, Ordering::Relaxed);
        self.total_tokens_generated
            .fetch_add(stats.tokens_generated as u64, Ordering::Relaxed);
        // Convert ms (f64) → us (u64). Round to nearest; clamp
        // negatives to 0 so a clock skew bug doesn't poison the
        // counter.
        self.total_prefill_us.fetch_add(
            (stats.prefill_ms * 1000.0).max(0.0).round() as u64,
            Ordering::Relaxed,
        );
        self.total_decode_us.fetch_add(
            (stats.decode_ms * 1000.0).max(0.0).round() as u64,
            Ordering::Relaxed,
        );
    }

    /// Record one speculation verify round: `drafted` tokens were
    /// proposed, `accepted` of them survived `accept_reject`. Called
    /// per round from the n-gram and draft-model speculation loops
    /// (not from `commit_request_stats` — rounds resolve mid-stream).
    /// A round with `drafted == 0` is a no-op: it carries no
    /// acceptance signal and would dilute the per-round average.
    pub fn add_speculation(&self, drafted: u64, accepted: u64) {
        use std::sync::atomic::Ordering;
        if drafted == 0 {
            return;
        }
        self.total_spec_rounds.fetch_add(1, Ordering::Relaxed);
        self.total_spec_drafted.fetch_add(drafted, Ordering::Relaxed);
        self.total_spec_accepted
            .fetch_add(accepted.min(drafted), Ordering::Relaxed);
    }

    /// Atomic-snapshot read suitable for serializing to the metrics
    /// endpoint. Each counter is loaded independently (no global
    /// barrier) — adjacent reads can race against a concurrent
    /// `add`, but the worst case is a sub-microsecond inconsistency
    /// in derived ratios. Acceptable for a polling-interval
    /// snapshot.
    pub fn snapshot(&self) -> CumulativeStatsSnapshot {
        use std::sync::atomic::Ordering;
        let total_requests = self.total_requests.load(Ordering::Relaxed);
        let total_tokens_prefilled = self.total_tokens_prefilled.load(Ordering::Relaxed);
        let total_cache_hit_tokens = self.total_cache_hit_tokens.load(Ordering::Relaxed);
        let total_tokens_generated = self.total_tokens_generated.load(Ordering::Relaxed);
        let total_prefill_ms = self.total_prefill_us.load(Ordering::Relaxed) as f64 / 1000.0;
        let total_decode_ms = self.total_decode_us.load(Ordering::Relaxed) as f64 / 1000.0;
        // Prefix-cache hit rate = hits / (hits + misses). Computed
        // here so every downstream consumer sees the same number,
        // and 0/0 returns 0.0 instead of NaN.
        let total_prompt_tokens = total_cache_hit_tokens + total_tokens_prefilled;
        let prefix_cache_hit_rate = if total_prompt_tokens == 0 {
            0.0
        } else {
            total_cache_hit_tokens as f64 / total_prompt_tokens as f64
        };
        let total_spec_rounds = self.total_spec_rounds.load(Ordering::Relaxed);
        let total_spec_drafted = self.total_spec_drafted.load(Ordering::Relaxed);
        let total_spec_accepted = self.total_spec_accepted.load(Ordering::Relaxed);
        let spec_acceptance_rate = if total_spec_drafted == 0 {
            0.0
        } else {
            total_spec_accepted as f64 / total_spec_drafted as f64
        };
        CumulativeStatsSnapshot {
            total_requests,
            total_tokens_prefilled,
            total_cache_hit_tokens,
            total_tokens_generated,
            total_prefill_ms,
            total_decode_ms,
            prefix_cache_hit_rate,
            total_spec_rounds,
            total_spec_drafted,
            total_spec_accepted,
            spec_acceptance_rate,
        }
    }
}

/// JSON-serializable point-in-time read of [`CumulativeStats`].
/// Exposed via [`Engine::cumulative_stats_snapshot`] and surfaced on
/// the `/metrics` endpoint.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CumulativeStatsSnapshot {
    pub total_requests: u64,
    pub total_tokens_prefilled: u64,
    pub total_cache_hit_tokens: u64,
    pub total_tokens_generated: u64,
    pub total_prefill_ms: f64,
    pub total_decode_ms: f64,
    /// Ratio of prompt tokens served from the prefix cache
    /// (`total_cache_hit_tokens / (total_cache_hit_tokens +
    /// total_tokens_prefilled)`). `0.0` when no requests have
    /// completed yet. Closer to 1.0 = the cache is doing more work.
    pub prefix_cache_hit_rate: f64,
    /// Speculation verify rounds that proposed ≥1 draft token.
    /// `#[serde(default)]` keeps snapshots from older builds
    /// deserializable.
    #[serde(default)]
    pub total_spec_rounds: u64,
    /// Total draft tokens proposed (n-gram + draft-model paths).
    #[serde(default)]
    pub total_spec_drafted: u64,
    /// Total draft tokens the target accepted.
    #[serde(default)]
    pub total_spec_accepted: u64,
    /// `total_spec_accepted / total_spec_drafted`; `0.0` before any
    /// speculation has run. Operators read this as "how much of the
    /// speculative work is paying off" — at temp 0 with the n-gram
    /// drafter on repetitive text it should sit well above 0.5.
    #[serde(default)]
    pub spec_acceptance_rate: f64,
}

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("engine not yet implemented: {0}")]
    Unimplemented(&'static str),
    #[error("engine: {0}")]
    Engine(String),
    #[error("gguf: {0}")]
    Gguf(#[from] rustllama_gguf::GgufError),
    #[error("tokenizer: {0}")]
    Tokenizer(#[from] rustllama_tokenizer::TokenizerError),
    #[error("model: {0}")]
    Model(#[from] rustllama_models::ModelError),
    #[error(
        "speculative decoding not supported on this Engine impl — \
         use CpuEngine (or wait for a SYCL build with target-side \
         distribution extraction)"
    )]
    SpeculationUnsupported,
    /// A chat request carried `image_url` blocks (decoded to bytes
    /// and attached to [`ChatMessage::images`]), but the active
    /// engine is text-only. Maps to HTTP 400 in the server with a
    /// clear "this model doesn't support vision" message rather than
    /// silently dropping the image and giving the user a confusing
    /// text-only response.
    ///
    /// A future `VlmEngine` (loads an mmproj alongside the text
    /// decoder) overrides [`Engine::supports_vision`] to return
    /// `true` and consumes the bytes through the V-3 / V-5 pipeline.
    #[error(
        "this model does not support image inputs — the active engine \
         is text-only; load a model with a paired mmproj GGUF to enable \
         vision"
    )]
    VisionNotSupported,
}

pub type Result<T> = std::result::Result<T, EngineError>;

/// If any message in `msgs` carries attached images, return
/// [`EngineError::VisionNotSupported`]; otherwise `Ok(())`. The
/// canonical gate for text-only [`Engine::chat`] impls — call this
/// before tokenizing so the user gets a clean 400 instead of a
/// silent "the model just ignored my image" outcome.
///
/// Lives free-standing rather than on the trait so a future
/// `VlmEngine` can call its own per-message validation (e.g.
/// "max 4 images per message") without going through this gate.
pub fn reject_if_has_images(msgs: &[ChatMessage]) -> Result<()> {
    if msgs.iter().any(|m| !m.images.is_empty()) {
        return Err(EngineError::VisionNotSupported);
    }
    Ok(())
}

/// One emitted token from a generation stream.
#[derive(Debug, Clone)]
pub struct Token {
    pub id: u32,
    pub text: String,
    /// Per-token logprob info, if [`SamplingParams::logprobs`] was set.
    /// `None` means logprobs were not requested.
    pub logprobs: Option<TokenLogprobs>,
}

/// Logprob of the chosen token plus optional top-K alternatives. Token
/// strings are not included here — callers can detokenize ids themselves
/// (the engine doesn't have the tokenizer at this layer).
#[derive(Debug, Clone)]
pub struct TokenLogprobs {
    /// log P(chosen token | context). `0.0` => certain; `-inf` => never.
    pub logprob: f32,
    /// Top-K alternatives (sorted descending by logprob). The chosen token
    /// is also included here if it ranked within the top K.
    pub top: Vec<TopLogprob>,
}

#[derive(Debug, Clone)]
pub struct TopLogprob {
    pub id: u32,
    pub logprob: f32,
}

pub type TokenStream = Pin<Box<dyn Stream<Item = Result<Token>> + Send>>;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model_path: std::path::PathBuf,
    pub n_gpu_layers: u32,
    pub ctx_size: u32,
    pub batch_size: u32,
    pub threads: u32,
    pub device: rustllama_tensor::Device,
}

/// The engine surface that callers (server, CLI, GUI) talk to.
/// Production implementor: `CpuEngine` (Rust scalar / SIMD, with the
/// SYCL + CUDA accelerated kernel paths layered on). [`MockEngine`] is a
/// test-only fixture (see its docs).
pub trait Engine: Send + Sync {
    fn metrics(&self) -> Metrics;
    fn n_ctx(&self) -> u32;
    fn vocab_size(&self) -> usize;
    fn tokenize(&self, text: &str) -> Result<Vec<u32>>;
    fn chat(&self, msgs: &[ChatMessage], s: &SamplingParams) -> Result<TokenStream>;
    fn generate(&self, prompt: &str, s: &SamplingParams) -> Result<TokenStream>;

    /// Whether this engine consumes image bytes attached to a
    /// [`ChatMessage::images`]. Default `false`: text-only engines
    /// (the v1 default) reject image-bearing requests with
    /// [`EngineError::VisionNotSupported`] via [`reject_if_has_images`]
    /// so the server can return a clean HTTP 400. A future
    /// `VlmEngine` overrides this to `true` and routes images
    /// through the V-3 vision pipeline + V-5 splice helper.
    fn supports_vision(&self) -> bool {
        false
    }

    /// Speculative-decoding driver. `self` is the **target** model;
    /// `draft` is a smaller / faster engine that proposes `k` tokens
    /// per round. The accept/reject loop from
    /// [`crate::speculative::accept_reject`] decides which prefix of
    /// the proposed tokens to commit.
    ///
    /// Default impl returns
    /// [`EngineError::SpeculationUnsupported`] — the trait method is
    /// here so [`CpuEngine`] (and future GPU engines) can override it
    /// without callers having to feature-detect.
    ///
    /// Returns a [`TokenStream`] that yields the committed tokens.
    /// The stream finishes after one round of speculation in this
    /// scaffold; multi-round speculation needs KV-rewind on reject,
    /// which lands alongside the paged-KV continuous-batching path.
    fn speculate(
        &self,
        _prompt: &str,
        _draft: std::sync::Arc<dyn Engine>,
        _k: u32,
        _s: &SamplingParams,
    ) -> Result<TokenStream> {
        Err(EngineError::SpeculationUnsupported)
    }

    /// Last-completed-request timing + token-count stats, or
    /// `None` if no request has yet completed (or this engine
    /// doesn't track per-request stats — e.g. `MockEngine`).
    ///
    /// Distinct name from `CpuEngine::last_request_stats` (which
    /// is an inherent method returning `RequestStats` directly)
    /// to avoid the inherent-vs-trait method-resolution clash —
    /// callers with a typed `CpuEngine` use the inherent; callers
    /// with `&dyn Engine` (the server's metrics handler) use this.
    ///
    /// Surfaces the `Last request: prefill X ms, decode Y ms,
    /// cache hits Z, generated N` row on the GUI Status page.
    /// Default returns `None` so engine impls without tracking
    /// don't need to opt in.
    fn last_request_stats_snapshot(&self) -> Option<RequestStats> {
        None
    }

    /// Cumulative-since-startup counters: total requests, total
    /// prefill / cache-hit / generated tokens, total prefill +
    /// decode time, derived prefix-cache hit rate. Surfaced on
    /// `/metrics` so operators can see at a glance whether the
    /// prefix cache is paying off. Default returns `None` for
    /// engines that don't track aggregate counters (e.g.
    /// `MockEngine`); `CpuEngine` overrides to return its live
    /// [`CumulativeStats`] snapshot.
    fn cumulative_stats_snapshot(&self) -> Option<CumulativeStatsSnapshot> {
        None
    }
}

/// Test-only fixture engine. Returns the prompt back tokenized as ASCII
/// bytes / a canned stream, so `rustllama-server`'s integration tests can
/// verify HTTP/API wiring without loading a real model. Never serves real
/// requests in production (that's [`CpuEngine`] and the GPU-accelerated
/// paths).
pub struct MockEngine;

impl Engine for MockEngine {
    fn metrics(&self) -> Metrics {
        Metrics {
            tokens_per_second: 0.0,
            context_used: 0,
            vram_estimate_mb: 0,
            ram_estimate_mb: 0,
            paged_total_pages: 0,
            paged_free_pages: 0,
            paged_active_slots: 0,
        }
    }
    fn n_ctx(&self) -> u32 {
        2048
    }
    fn vocab_size(&self) -> usize {
        256
    }
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        Ok(text.bytes().map(|b| b as u32).collect())
    }
    fn chat(&self, msgs: &[ChatMessage], _s: &SamplingParams) -> Result<TokenStream> {
        reject_if_has_images(msgs)?;
        let payload = msgs
            .iter()
            .map(|m| format!("[{}] {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join(" / ");
        mock_stream(format!("(mock) you said: {payload}"))
    }
    fn generate(&self, prompt: &str, _s: &SamplingParams) -> Result<TokenStream> {
        mock_stream(format!("(mock) {prompt}"))
    }
}

fn mock_stream(s: String) -> Result<TokenStream> {
    let words: Vec<String> = s.split_whitespace().map(String::from).collect();
    let stream = async_stream::stream! {
        for (i, w) in words.into_iter().enumerate() {
            yield Ok(Token { id: i as u32, text: format!("{w} "), logprobs: None });
        }
    };
    Ok(Box::pin(stream))
}

#[cfg(test)]
mod vision_tests {
    use super::*;

    fn user(text: &str) -> ChatMessage {
        ChatMessage::text("user", text)
    }

    fn user_with_image(text: &str, image: Vec<u8>) -> ChatMessage {
        ChatMessage {
            role: "user".into(),
            content: text.into(),
            images: vec![image],
        }
    }

    /// Text-only message — gate passes.
    #[test]
    fn reject_if_has_images_passes_text_only() {
        let msgs = [user("hello"), user("world")];
        assert!(reject_if_has_images(&msgs).is_ok());
    }

    /// Empty `images` vec on every message — gate still passes
    /// (the gate looks at `is_empty`, not the presence of the field).
    #[test]
    fn reject_if_has_images_passes_when_images_vec_is_empty() {
        let msgs = [
            ChatMessage {
                role: "user".into(),
                content: "hi".into(),
                images: Vec::new(),
            },
        ];
        assert!(reject_if_has_images(&msgs).is_ok());
    }

    /// Any one message with attached images trips the gate.
    #[test]
    fn reject_if_has_images_errors_when_one_message_has_images() {
        let msgs = [
            user("describe: "),
            user_with_image("[image: data:...]", vec![0x89, 0x50, 0x4e, 0x47]),
        ];
        match reject_if_has_images(&msgs) {
            Err(EngineError::VisionNotSupported) => {}
            other => panic!("expected VisionNotSupported, got {other:?}"),
        }
    }

    /// `MockEngine::chat` rejects via the same gate. End-to-end pin.
    #[tokio::test]
    async fn mock_engine_chat_rejects_image_bearing_messages() {
        let engine = MockEngine;
        let sampling = SamplingParams::default();
        let msgs = vec![user_with_image("[image: ...]", vec![0xff, 0xd8, 0xff])];
        match engine.chat(&msgs, &sampling) {
            Err(EngineError::VisionNotSupported) => {}
            Ok(_) => panic!("MockEngine should reject image-bearing messages"),
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    /// `supports_vision` default is false; impls must opt in.
    #[test]
    fn supports_vision_default_is_false() {
        let engine = MockEngine;
        assert!(!engine.supports_vision());
    }

    /// `ChatMessage::text` builds a message with empty images.
    #[test]
    fn chat_message_text_constructor_leaves_images_empty() {
        let m = ChatMessage::text("user", "hello");
        assert_eq!(m.role, "user");
        assert_eq!(m.content, "hello");
        assert!(m.images.is_empty());
    }

    /// Serde round-trip: a JSON ChatMessage without `images` field
    /// deserializes cleanly with an empty `images` (so older callers
    /// and stored conversations still load).
    #[test]
    fn chat_message_deserializes_without_images_field() {
        let json = r#"{"role":"user","content":"hi"}"#;
        let m: ChatMessage = serde_json::from_str(json).unwrap();
        assert!(m.images.is_empty());
    }

    /// Serialization omits `images` when empty (so the on-wire shape
    /// stays compact for the text-only common case).
    #[test]
    fn chat_message_omits_empty_images_in_json() {
        let m = ChatMessage::text("user", "hi");
        let json = serde_json::to_string(&m).unwrap();
        assert!(!json.contains("images"), "json: {json}");
    }

    /// And conversely: when `images` is non-empty it serializes.
    #[test]
    fn chat_message_serializes_non_empty_images() {
        let m = ChatMessage {
            role: "user".into(),
            content: "hi".into(),
            images: vec![vec![1, 2, 3]],
        };
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains("images"), "json: {json}");
    }
}

#[cfg(test)]
mod cumulative_stats_tests {
    use super::*;

    fn req(prefill_ms: f64, decode_ms: f64, prefilled: u32, hits: u32, generated: u32) -> RequestStats {
        RequestStats {
            prefill_ms,
            decode_ms,
            tokens_prefilled: prefilled,
            cache_hit_tokens: hits,
            tokens_generated: generated,
            tool_call_limit_hit: false,
        }
    }

    #[test]
    fn snapshot_of_fresh_counters_is_zero_with_zero_hit_rate() {
        let c = CumulativeStats::default();
        let s = c.snapshot();
        assert_eq!(s.total_requests, 0);
        assert_eq!(s.total_tokens_prefilled, 0);
        assert_eq!(s.total_cache_hit_tokens, 0);
        assert_eq!(s.total_tokens_generated, 0);
        assert_eq!(s.total_prefill_ms, 0.0);
        assert_eq!(s.total_decode_ms, 0.0);
        assert_eq!(
            s.prefix_cache_hit_rate, 0.0,
            "zero requests must report 0.0, not NaN"
        );
    }

    #[test]
    fn add_speculation_accumulates_and_derives_acceptance_rate() {
        let c = CumulativeStats::default();
        // Fresh: no speculation → rate 0.0, not NaN.
        assert_eq!(c.snapshot().spec_acceptance_rate, 0.0);
        c.add_speculation(8, 6);
        c.add_speculation(8, 2);
        // Empty round: no acceptance signal, must not count.
        c.add_speculation(0, 0);
        // Defensive clamp: accepted can never exceed drafted.
        c.add_speculation(4, 9);
        let s = c.snapshot();
        assert_eq!(s.total_spec_rounds, 3, "0-draft round not counted");
        assert_eq!(s.total_spec_drafted, 20, "8 + 8 + 4");
        assert_eq!(s.total_spec_accepted, 12, "6 + 2 + 4 (clamped)");
        assert!(
            (s.spec_acceptance_rate - 0.6).abs() < 1e-9,
            "12 / 20; got {}",
            s.spec_acceptance_rate
        );
    }

    #[test]
    fn add_sums_counters_and_computes_hit_rate() {
        let c = CumulativeStats::default();
        c.add(&req(10.0, 100.0, 80, 20, 50));
        c.add(&req(5.0, 200.0, 40, 60, 100));
        let s = c.snapshot();
        assert_eq!(s.total_requests, 2);
        assert_eq!(s.total_tokens_prefilled, 120, "80 + 40");
        assert_eq!(s.total_cache_hit_tokens, 80, "20 + 60");
        assert_eq!(s.total_tokens_generated, 150, "50 + 100");
        assert!((s.total_prefill_ms - 15.0).abs() < 1e-6);
        assert!((s.total_decode_ms - 300.0).abs() < 1e-6);
        // Hit rate = 80 / (80 + 120) = 0.4
        assert!(
            (s.prefix_cache_hit_rate - 0.4).abs() < 1e-9,
            "got {}",
            s.prefix_cache_hit_rate
        );
    }

    #[test]
    fn add_zero_prompt_request_does_not_break_hit_rate() {
        // A "decode-only" request (e.g. a generation that hit
        // the cache 100% and had no prefill forwards) must not
        // produce NaN.
        let c = CumulativeStats::default();
        c.add(&req(0.0, 50.0, 0, 0, 20));
        let s = c.snapshot();
        assert_eq!(s.prefix_cache_hit_rate, 0.0);
    }

    #[test]
    fn add_full_hit_request_reports_hit_rate_one() {
        let c = CumulativeStats::default();
        c.add(&req(0.0, 50.0, 0, 100, 20));
        let s = c.snapshot();
        assert!(
            (s.prefix_cache_hit_rate - 1.0).abs() < 1e-9,
            "100% cache hit must be 1.0; got {}",
            s.prefix_cache_hit_rate
        );
    }

    #[test]
    fn add_negative_ms_floors_to_zero() {
        // Clock-skew defense: a negative `prefill_ms` (shouldn't
        // happen, but Instant::elapsed has been seen to return
        // 0 across thread migrations) must not wrap the u64
        // counter or appear as a huge positive value.
        let c = CumulativeStats::default();
        c.add(&req(-1.0, -1.0, 10, 0, 10));
        let s = c.snapshot();
        assert_eq!(s.total_prefill_ms, 0.0);
        assert_eq!(s.total_decode_ms, 0.0);
    }
}
