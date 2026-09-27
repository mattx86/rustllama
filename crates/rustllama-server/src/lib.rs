//! Axum OpenAI-compatible HTTP server.
//!
//! Phase 0 implements `/healthz` and `/v1/models` against a mock engine.
//! `/v1/chat/completions` (stream + non-stream) lands in phase 6 alongside
//! the real engine.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use rustllama_engine::batch_scheduler::{
    MultiFlightScheduler, RequestId, SharedScheduler, SingleFlightScheduler, Slot, SlotState,
};
use rustllama_engine::Engine;
use serde::Serialize;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

mod anthropic;
mod autotune;
// Shared first-load auto-tune gate, called by both the `/v1/models/load`
// handler below and the CLI `serve` startup load so both paths auto-tune a
// never-seen model exactly once before loading it.
pub use autotune::maybe_first_load_autotune;
mod chat;
mod completions;
mod decide;
mod embeddings;
pub mod env_hint;
#[cfg(feature = "history")]
pub mod history;
// V-5 wires this into chat.rs; until then the decoder lives standalone
// with full unit-test coverage (`image_url::tests`). Suppress the
// dead-code warning for the V-4 commit — V-5 adds the use site.
#[allow(dead_code)]
mod image_url;
mod ollama;
mod rag;
mod rerank;

/// OpenAI `usage` field — emitted on every non-streaming response and in
/// the final streaming chunk (when `stream_options.include_usage = true`).
///
/// The three baseline fields (`prompt_tokens`, `completion_tokens`,
/// `total_tokens`) are OpenAI-standard. The remaining `*_ms` and
/// `*_tokens` fields are rustllama extensions that surface per-request
/// performance + cache-hit data; clients that don't know about them
/// can safely ignore the extra JSON keys.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    /// Wall-clock time in the prefill phase (ms). Extension.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_ms: Option<f64>,
    /// Wall-clock time in decode forward passes (ms). Extension.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_ms: Option<f64>,
    /// Number of prompt-position forwards that actually ran during
    /// prefill (after subtracting `cache_hit_tokens`). Extension.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_prefilled: Option<u32>,
    /// Number of prompt tokens whose K/V state was reused from the
    /// prefix cache (live LCP + multi-snapshot pool + extended-LCP).
    /// Higher is better. Extension.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_hit_tokens: Option<u32>,
}

impl Usage {
    pub fn new(prompt: u32, completion: u32) -> Self {
        Self {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: prompt + completion,
            prefill_ms: None,
            decode_ms: None,
            tokens_prefilled: None,
            cache_hit_tokens: None,
        }
    }

    /// Attach engine-side per-request stats. `tokens_generated` from
    /// the stats overrides the `completion_tokens` baseline when the
    /// caller couldn't re-tokenize the response (rare).
    pub fn with_stats(mut self, s: &rustllama_engine::RequestStats) -> Self {
        self.prefill_ms = Some(s.prefill_ms);
        self.decode_ms = Some(s.decode_ms);
        self.tokens_prefilled = Some(s.tokens_prefilled);
        self.cache_hit_tokens = Some(s.cache_hit_tokens);
        self
    }
}

/// OpenAI `stream_options` — controls per-stream behavior. We honor
/// `include_usage` (emit a final usage chunk before `[DONE]`).
#[derive(Debug, serde::Deserialize, Default)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

/// OpenAI-compat `system_fingerprint` for chat/completions responses.
/// Stable string identifier for the backend config (server version +
/// model id + KV dtype) — same inputs always give the same output
/// within a process. Editor clients (Aider, Continue) compare it
/// across consecutive responses to detect "the server silently
/// swapped models on me." Format: `fp_<12 hex chars>` matches the
/// OpenAI shape so clients that regex-validate it accept ours.
pub fn system_fingerprint(version: &str, model_id: &str, kv_dtype: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    version.hash(&mut h);
    model_id.hash(&mut h);
    kv_dtype.hash(&mut h);
    let v = h.finish();
    // Truncate to 48 bits so the hex output is exactly 12 chars,
    // matching OpenAI's "fp_44709d6fcb" style. `DefaultHasher` is
    // process-randomized so absolute values differ across server
    // restarts; clients only compare consecutive same-process
    // responses so that's fine.
    format!("fp_{:012x}", v & 0xFFFF_FFFF_FFFF)
}

/// Snapshot of a loaded model and its engines. Handed out by
/// [`AppState::resolve`] / [`AppState::current`] as a cheap-clone bundle
/// of `Arc`s so handlers can hold a stable view across an entire request
/// even if a load/unload lands mid-request. In-flight generations keep
/// their snapshot's `Arc<CpuEngine>` alive past an unload; the eviction
/// only affects future requests.
///
/// Each model owns its own single-flight gate. With multi-model loaded,
/// requests to *different* models can run concurrently (subject to host
/// CPU/GPU contention); requests to the *same* model still serialize. The
/// gate is replaced by per-slot scheduling in v1.1's continuous batching.
#[derive(Clone)]
pub struct ServingModel {
    pub engine: Arc<dyn Engine>,
    pub cpu_engine: Option<Arc<rustllama_engine::CpuEngine>>,
    pub model_id: String,
    pub gate: Arc<Semaphore>,
    /// Per-model scheduler. Today this is a [`SingleFlightScheduler`]
    /// that matches the `Semaphore(1)` gate's behavior exactly — admit
    /// at most one slot at a time, queue the rest. When continuous
    /// batching lands, this swaps to a `ContinuousBatchingScheduler`
    /// and the `Semaphore` gate goes away; the wire-protocol surface
    /// (`try_acquire`/`PermitGuard`) stays unchanged so handlers don't
    /// need updates.
    pub scheduler: SharedScheduler,
    /// Live count of "in-flight + waiting" requests against this model.
    /// Incremented in `try_acquire` before the gate await, decremented
    /// when the returned [`PermitGuard`] drops. Used purely for the
    /// `503 / Retry-After` backpressure decision.
    pub pending: Arc<AtomicUsize>,
    /// Hard cap on `pending`. Beyond this, `try_acquire` returns
    /// `BackpressureError` and the handler emits a `503` with
    /// `Retry-After`. Configurable via `[server].max_pending_per_model`.
    pub max_pending: usize,
    /// Monotonic stamp updated each time [`AppState::resolve`] hands this
    /// model out. The warm-pool LRU eviction uses it to pick the least-
    /// recently-resolved entry once `max_loaded_models` is exceeded.
    pub last_used: Arc<AtomicU64>,
    /// Multi-flight serving pool. `None` for single-flight (the default
    /// — every test fixture and the default CLI `serve`). `Some(pool)`
    /// when the operator configured `[server].concurrency > 1`; in that
    /// case the gate is a `Semaphore::new(concurrency)` and `try_acquire`
    /// hands out an [`EngineHandle`] whose `engine` field points to a
    /// per-request engine from the pool. Each pool entry is a
    /// [`rustllama_engine::CpuEngine::fork_for_concurrent_use`] fork —
    /// shared model weights via `Arc`, independent KV cache.
    #[doc(hidden)]
    pub multi: Option<MultiFlightPool>,
}

/// Internal: pool of forked engines for multi-flight serving. Exposed
/// so test fixtures can construct `ServingModel` literals with
/// `multi: None`, but the contained fields are not part of the stable
/// surface — use [`ServingModel::with_concurrency`] to populate it.
#[doc(hidden)]
#[derive(Clone)]
pub struct MultiFlightPool {
    pub engines: Vec<Arc<dyn Engine>>,
    pub cpu_engines: Vec<Arc<rustllama_engine::CpuEngine>>,
    /// Shared round-robin counter. `Arc`-cloned across `ServingModel`
    /// clones so concurrent `resolve()` calls see the same monotonic
    /// pick sequence.
    pub next: Arc<AtomicUsize>,
}

/// Default queue depth per model: one request executing + three waiting.
/// At our v1 single-flight throughput (~6 tok/s on the 0.5B Q4_K_M, so
/// ~5 s per ~30-token completion), this caps worst-case latency for an
/// accepted request at ~20 s. Configurable via
/// `[server].max_pending_per_model`.
pub const DEFAULT_MAX_PENDING_PER_MODEL: usize = 4;

impl ServingModel {
    /// Build a fresh single-flight gate. v1 contract: **one in-flight
    /// generation per model**. Concurrent requests against the same
    /// model serialize on the gate; concurrent requests against
    /// *different* loaded models (multi-model deployments) run in
    /// parallel subject to host CPU / GPU contention.
    ///
    /// Single-flight is hardcoded for v1 because:
    ///   - The `KvCache` is per-engine and not safe for concurrent
    ///     mutation. Two in-flight forward passes would race on the
    ///     same allocation and corrupt state.
    ///   - The prefix-cache pool and `last_ids` snapshot assume a
    ///     serialized request stream; a parallel scheduler would need
    ///     per-request cache slots (paged KV) which is on the v2
    ///     continuous-batching roadmap.
    ///   - The CPU forward pass is already throughput-bound on the
    ///     host's vector units, so adding parallelism doesn't reduce
    ///     latency for a single request and would only increase
    ///     contention.
    ///
    /// When continuous batching lands (v2), this is the boundary that
    /// changes — `Semaphore::new(1)` becomes `Semaphore::new(n_slots)`.
    pub fn new_gate() -> Arc<Semaphore> {
        Arc::new(Semaphore::new(1))
    }

    /// Build a fresh per-model scheduler. Today this is a
    /// [`SingleFlightScheduler`] whose `admit/complete` track the
    /// slot lifecycle in parallel with the `Semaphore(1)` gate. When
    /// continuous batching lands, swap the constructed type and the
    /// rest of the server keeps working.
    pub fn new_scheduler() -> SharedScheduler {
        Arc::new(SingleFlightScheduler::new())
    }

    /// Build a gate sized to a specific concurrency level. Use
    /// `concurrency == 1` for single-flight or `> 1` for the
    /// multi-flight serving path (the workspace default is 4).
    pub fn new_gate_for_concurrency(concurrency: usize) -> Arc<Semaphore> {
        Arc::new(Semaphore::new(concurrency.max(1)))
    }

    /// Build a scheduler sized to a specific concurrency level. Picks
    /// [`SingleFlightScheduler`] for `concurrency == 1` (exact prior
    /// behavior) and [`MultiFlightScheduler`] otherwise.
    pub fn new_scheduler_for_concurrency(concurrency: usize) -> SharedScheduler {
        if concurrency <= 1 {
            Arc::new(SingleFlightScheduler::new())
        } else {
            Arc::new(MultiFlightScheduler::new(concurrency))
        }
    }

    /// Promote a single-flight serving entry to multi-flight by
    /// allocating `concurrency` engine forks (shared model weights,
    /// independent KV caches) and resizing the gate + scheduler. Must
    /// be called before the entry is inserted into the [`AppState`]
    /// registry; mutating concurrency after admission would race with
    /// in-flight requests.
    ///
    /// Requires `self.cpu_engine` to be populated — only the CPU
    /// engine can produce forks. For mock engines (test fixtures) the
    /// method is a no-op that returns `self` unchanged.
    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        if concurrency <= 1 {
            return self;
        }
        let Some(cpu) = self.cpu_engine.as_ref() else {
            // Mock-engine path: leave single-flight. Multi-flight is
            // only meaningful for the real CPU engine.
            return self;
        };
        let mut engines: Vec<Arc<dyn Engine>> = Vec::with_capacity(concurrency);
        let mut cpu_engines: Vec<Arc<rustllama_engine::CpuEngine>> = Vec::with_capacity(concurrency);
        // The first pool entry IS the original engine; subsequent
        // entries are fresh forks. This preserves prefix-cache state
        // for the very first request after a model load.
        engines.push(self.engine.clone());
        cpu_engines.push(cpu.clone());
        for _ in 1..concurrency {
            let fork = cpu.fork_for_concurrent_use();
            let fork_arc: Arc<rustllama_engine::CpuEngine> = Arc::new(fork);
            cpu_engines.push(fork_arc.clone());
            engines.push(fork_arc as Arc<dyn Engine>);
        }
        self.multi = Some(MultiFlightPool {
            engines,
            cpu_engines,
            next: Arc::new(AtomicUsize::new(0)),
        });
        self.gate = Self::new_gate_for_concurrency(concurrency);
        self.scheduler = Self::new_scheduler_for_concurrency(concurrency);
        self
    }

    /// Number of permits this model's gate exposes — `1` for
    /// single-flight, `> 1` for multi-flight. Surfaced for the
    /// `/v1/metrics` snapshot.
    pub fn concurrency(&self) -> usize {
        match &self.multi {
            Some(p) => p.engines.len(),
            None => 1,
        }
    }

    /// Build a fused-decode [`ServingModel`] backed by a single
    /// [`rustllama_engine::paged_batch::PagedBatchEngine`] sized
    /// to `concurrency` concurrent slots. Replaces the per-fork
    /// [`MultiFlightPool`] with one shared engine + driver thread
    /// that batches up to M decode steps into a single forward
    /// pass per tick.
    ///
    /// Selected by the loader when `[server].fused_decode = true`
    /// + `[inference].kv_cache_layout = "paged"`. The
    /// admission gate (`Semaphore::new(concurrency)`) gives
    /// request-level backpressure semantics consistent with the
    /// per-fork path — once `concurrency` requests hold permits,
    /// the next admission waits or fails fast at
    /// `max_pending_per_model`.
    ///
    /// `cpu_engine` is set to `None` — `PagedBatchEngine` is not
    /// a `CpuEngine`. Features that reach into `cpu_engine`
    /// directly (USM warmup, prefix-cache config, raw fork) are
    /// unavailable on this path; the V1 trade-off is documented
    /// in the `fused_decode` config docstring.
    pub fn new_fused_decode_paged(
        paged: Arc<rustllama_engine::paged_batch::PagedBatchEngine>,
        model_id: String,
        max_pending: usize,
        concurrency: usize,
    ) -> Self {
        let concurrency = concurrency.max(1);
        Self {
            engine: paged as Arc<dyn Engine>,
            cpu_engine: None,
            model_id,
            gate: Self::new_gate_for_concurrency(concurrency),
            scheduler: Self::new_scheduler_for_concurrency(concurrency),
            pending: Arc::new(AtomicUsize::new(0)),
            max_pending,
            last_used: Arc::new(AtomicU64::new(0)),
            multi: None,
        }
    }

    /// Try to enter the per-model serving queue. Increments
    /// `pending`; if that would exceed `max_pending`, decrements and
    /// returns [`BackpressureError`]. Otherwise awaits the gate and
    /// returns an [`EngineHandle`] that:
    ///   - Exposes a per-request `engine` field (round-robin from the
    ///     multi-flight pool, or the single-flight engine otherwise).
    ///   - Decrements `pending`, releases the permit, and tells the
    ///     scheduler the slot is complete when dropped.
    ///
    /// Convenience wrapper: `try_admit()? + permit.acquire_gate().await`
    /// in one call. New code that wants to do pre-engine work
    /// (tokenization, prefix-cache lookup) while waiting on the gate
    /// should use the split form so the work overlaps the wait.
    pub async fn try_acquire(&self) -> Result<EngineHandle, BackpressureError> {
        self.try_admit()?.acquire_gate().await
    }

    /// Backpressure-only admission: increment `pending`, check the
    /// queue cap, register a scheduler slot. Returns immediately —
    /// the actual gate wait happens in
    /// [`RequestPermit::acquire_gate`]. The caller can do
    /// tokenization, chat-template rendering, and prefix-cache
    /// lookups between admit and acquire_gate so that work overlaps
    /// the gate wait — meaningful TTFT reduction on queued requests
    /// because the next request's tokenization happens while the
    /// current request is still decoding.
    ///
    /// Drop semantics: if the returned [`RequestPermit`] is dropped
    /// without calling `acquire_gate`, the pending counter is
    /// decremented and the scheduler slot is completed (same RAII
    /// guarantees as a fully-acquired permit).
    pub fn try_admit(&self) -> Result<RequestPermit, BackpressureError> {
        self.try_admit_with_priority(
            rustllama_engine::batch_scheduler::Priority::DEFAULT,
        )
    }

    /// E3.2: try_admit with an explicit priority. HTTP handlers
    /// that read the `X-RustLlama-Priority` header from the
    /// incoming request call this; everyone else uses
    /// [`Self::try_admit`] which defaults to `Priority::DEFAULT`.
    /// The priority lands on the slot's `priority` field and
    /// drives ordering when the scheduler's waiting queue is
    /// non-empty.
    pub fn try_admit_with_priority(
        &self,
        priority: rustllama_engine::batch_scheduler::Priority,
    ) -> Result<RequestPermit, BackpressureError> {
        let prev = self.pending.fetch_add(1, Ordering::AcqRel);
        if prev >= self.max_pending {
            self.pending.fetch_sub(1, Ordering::AcqRel);
            return Err(BackpressureError {
                model_id: self.model_id.clone(),
                pending: prev,
                max: self.max_pending,
            });
        }
        let slot = Slot {
            request_id: RequestId(0),
            state: SlotState::Pending,
            next_pos: 0,
            max_new_tokens: 0,
            n_emitted: 0,
            pages: Vec::new(),
            priority,
            admit_seq: 0,
        };
        let request_id = self.scheduler.admit(slot);
        Ok(RequestPermit {
            model_id: self.model_id.clone(),
            gate: self.gate.clone(),
            pending: self.pending.clone(),
            scheduler: self.scheduler.clone(),
            engine: self.engine.clone(),
            cpu_engine: self.cpu_engine.clone(),
            multi: self.multi.clone(),
            request_id,
            consumed: false,
        })
    }
}

/// Pre-gate permit returned by [`ServingModel::try_admit`]. Holds the
/// pending-counter slot + scheduler entry but has NOT yet awaited the
/// per-model semaphore gate. Call [`Self::acquire_gate`] when the
/// caller is ready to wait on the gate (typically after
/// tokenization / chat-template rendering).
///
/// Drop without `acquire_gate`: decrements pending + completes the
/// scheduler slot. Used for the rare case where the caller decides
/// not to proceed after admission (e.g. tokenization failed and the
/// caller wants to release the slot before the gate even contends).
pub struct RequestPermit {
    model_id: String,
    gate: Arc<Semaphore>,
    pending: Arc<AtomicUsize>,
    scheduler: SharedScheduler,
    engine: Arc<dyn Engine>,
    cpu_engine: Option<Arc<rustllama_engine::CpuEngine>>,
    multi: Option<MultiFlightPool>,
    request_id: RequestId,
    consumed: bool,
}

impl RequestPermit {
    /// Borrow the shared engine for read-only pre-gate work
    /// (tokenizer access, prompt rendering). Returns the
    /// `serving.cpu_engine` directly — pre-gate work is identical
    /// across all forks in the multi-flight pool because the
    /// tokenizer is `Arc`-shared.
    pub fn shared_cpu_engine(&self) -> Option<&Arc<rustllama_engine::CpuEngine>> {
        self.cpu_engine.as_ref()
    }

    /// Wait on the per-model gate, then materialize an
    /// [`EngineHandle`] with the per-request engine fork selected.
    /// Consumes the permit; subsequent drop is a no-op (the gate
    /// permit + scheduler slot now live on the returned handle).
    pub async fn acquire_gate(mut self) -> Result<EngineHandle, BackpressureError> {
        self.consumed = true;
        let permit = match self.gate.clone().acquire_owned().await {
            Ok(p) => p,
            Err(_) => {
                // Semaphore closed — runtime is tearing down.
                self.scheduler.complete(self.request_id);
                self.pending.fetch_sub(1, Ordering::AcqRel);
                return Err(BackpressureError {
                    model_id: format!("{} (gate closed)", self.model_id),
                    pending: 0,
                    max: 0,
                });
            }
        };
        let (engine, cpu_engine, idx) = match &self.multi {
            Some(pool) => {
                let n = pool.engines.len();
                let i = pool.next.fetch_add(1, Ordering::Relaxed) % n;
                let e = pool.engines[i].clone();
                let cpu = pool
                    .cpu_engines
                    .get(i)
                    .cloned()
                    .or_else(|| self.cpu_engine.clone());
                (e, cpu, i)
            }
            None => (self.engine.clone(), self.cpu_engine.clone(), 0),
        };
        Ok(EngineHandle {
            engine,
            cpu_engine,
            engine_idx: idx,
            _permit: Some(permit),
            pending: self.pending.clone(),
            scheduler: Some(self.scheduler.clone()),
            request_id: Some(self.request_id),
        })
    }
}

impl Drop for RequestPermit {
    fn drop(&mut self) {
        if !self.consumed {
            self.scheduler.complete(self.request_id);
            self.pending.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

/// Returned by [`ServingModel::try_acquire`] when the per-model queue
/// is full. Handlers convert this into a `503` with `Retry-After`.
#[derive(Debug)]
pub struct BackpressureError {
    pub model_id: String,
    pub pending: usize,
    pub max: usize,
}

impl IntoResponse for BackpressureError {
    fn into_response(self) -> Response {
        let message = format!(
            "model `{}` is busy ({} pending, max {}); retry shortly",
            self.model_id, self.pending, self.max
        );
        // OpenAI SDKs treat 503 as retryable and read `error.type` /
        // `error.code` to classify it; `rate_limit_exceeded` is the code
        // their retry logic keys on for queue-saturation backoff.
        let mut resp = openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            message,
            "server_error",
            Some("rate_limit_exceeded"),
        );
        // Suggested retry interval. With ~5 s per generation a 2s retry
        // delay is roughly P25 wait time on a saturated queue.
        resp.headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from_static("2"));
        resp
    }
}

/// RAII handle returned by [`ServingModel::try_acquire`]. Holds the
/// per-request engine (chosen round-robin from the multi-flight pool,
/// or the canonical engine in single-flight) alongside the slot-
/// lifecycle bookkeeping. On drop:
/// (1) releases the semaphore permit, (2) tells the scheduler the
/// slot is complete (so a waiting request can promote in), and
/// (3) decrements the `pending` backpressure counter.
///
/// Holders must keep this alive for the lifetime of the request.
/// Use `handle.engine.{chat, generate, speculate}(...)` for state-
/// mutating generation — those calls are guaranteed to land on the
/// engine assigned to this request. Non-mutating queries (`tokenize`,
/// `n_ctx`, `metrics`) are safe on either `handle.engine` or
/// `serving.engine`, since they don't touch the KV cache.
pub struct EngineHandle {
    /// Per-request engine. In single-flight this is the same `Arc` as
    /// `ServingModel::engine`. In multi-flight this is one fork of N,
    /// with an independent KV cache.
    pub engine: Arc<dyn Engine>,
    /// Same shape, typed for the CPU engine. `None` when the model
    /// was loaded as a mock engine (test fixtures).
    pub cpu_engine: Option<Arc<rustllama_engine::CpuEngine>>,
    /// Index of the chosen engine within the multi-flight pool, `0`
    /// in single-flight. Surfaced for diagnostics / tests.
    pub engine_idx: usize,
    _permit: Option<OwnedSemaphorePermit>,
    pending: Arc<AtomicUsize>,
    /// Scheduler the slot was admitted into. `None` only on the
    /// "drop without ever owning a slot" path (currently unused —
    /// every guard built by `try_acquire` populates this).
    scheduler: Option<SharedScheduler>,
    request_id: Option<RequestId>,
}

/// Legacy alias retained so test fixtures that named `PermitGuard`
/// explicitly continue to compile. New code should write
/// [`EngineHandle`] directly.
pub type PermitGuard = EngineHandle;

impl Drop for EngineHandle {
    fn drop(&mut self) {
        // Order: semaphore permit → scheduler complete → pending--.
        // The permit release is the gate that lets the next admitted
        // slot proceed; calling `scheduler.complete` afterward avoids
        // a race where a waiter notices the freed slot before the
        // scheduler's bookkeeping reflects the completion.
        self._permit.take();
        if let (Some(s), Some(id)) =
            (self.scheduler.take(), self.request_id.take())
        {
            s.complete(id);
        }
        self.pending.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Owns the map of loaded models. Tracked behind a single `RwLock` so
/// resolve / upsert / unload / set_default are all cheap and coherent.
///
/// `max_loaded` bounds the warm pool size. When `upsert` would push the
/// registry over the cap, the entry with the smallest `last_used` stamp
/// is evicted — except the current default, which is never evicted by
/// the LRU. An in-flight request holds its own `Arc<CpuEngine>` clone,
/// so an evicted model still finishes serving requests that already
/// picked it up before being dropped from the registry.
struct Registry {
    models: HashMap<String, ServingModel>,
    default_id: String,
    /// Maximum concurrently-loaded models. `0` disables the cap.
    max_loaded: usize,
    /// Monotonic counter used to stamp `ServingModel::last_used` on
    /// each resolve. Always incremented; never reset.
    use_counter: u64,
}

/// Default cap on simultaneously warm-loaded models. Picks a value
/// small enough to keep RAM bounded on typical hosts but big enough
/// that "active model + 2-3 swap-ins" stays warm.
pub const DEFAULT_MAX_LOADED_MODELS: usize = 4;

#[derive(Clone)]
pub struct AppState {
    inner: Arc<tokio::sync::RwLock<Registry>>,
    /// In-flight request cancellation registry. Each streaming handler
    /// inserts `(request_id, AtomicBool::new(false))` on start and an
    /// RAII [`CancelGuard`] removes the entry on drop. `POST /v1/cancel`
    /// flips the bool — the engine polls it between forward passes and
    /// exits gracefully.
    cancellations: Arc<std::sync::Mutex<HashMap<String, Arc<std::sync::atomic::AtomicBool>>>>,
    /// Flips to true when the server receives SIGINT/SIGTERM/Ctrl-C.
    /// Middleware checks this before dispatch — new requests get 503 +
    /// Retry-After while existing ones finish. `axum::serve(...)
    /// .with_graceful_shutdown(...)` already keeps the listener up
    /// until in-flight tasks complete; this flag turns away NEW work
    /// during the drain window.
    pub shutdown: Arc<std::sync::atomic::AtomicBool>,
    pub version: String,
    /// Path to the `config.toml` that backs `GET/PUT /v1/config`. When
    /// `None`, the config endpoints return `409` so a misconfigured
    /// embedded server (e.g. a test fixture) doesn't accidentally
    /// write to a default-resolved system path. `cli::serve` always
    /// populates this from the resolved `--config` flag.
    pub config_path: Option<Arc<std::path::PathBuf>>,
    /// Optional sqlite-backed conversation history. Only populated
    /// when the binary is built with `--features history` and a
    /// store was opened at startup. Handlers degrade to `501 Not
    /// Implemented` when this is `None`.
    #[cfg(feature = "history")]
    pub history: Option<Arc<crate::history::HistoryStore>>,
    /// Per-request audit log sink. When `Some`, the router attaches
    /// a middleware that appends a JSONL line per request. When
    /// `None` (default for tests and audit_log=false configs), the
    /// middleware isn't attached — zero overhead.
    pub audit: Option<AuditSink>,
    /// Lazy-loaded embedding model + tokenizer. First call to
    /// `/v1/embeddings` (or `/api/embeddings`) triggers the load from
    /// `[embeddings]` config and caches the result — Ok with the
    /// bundled [`LoadedEmbeddingModel`], or Err with a load-error
    /// string the handler surfaces as a 5xx. `None` outer slot when
    /// `[embeddings]` isn't configured at startup (set in
    /// `cli::serve`); inner OnceLock is the "load-once, hand out the
    /// same Arc" cache. The tokenizer lives next to the model so a
    /// single GGUF open populates both (text input → tokenize →
    /// forward; pre-tokenized input → forward).
    pub embedding_model:
        Option<Arc<std::sync::OnceLock<std::result::Result<LoadedEmbeddingModel, String>>>>,
    /// Lazy-loaded reranker model + tokenizer. Same shape /
    /// lifecycle as [`AppState::embedding_model`]; backs `POST
    /// /v1/rerank`. `None` outer slot when `[reranker]` isn't
    /// configured.
    pub reranker_model:
        Option<Arc<std::sync::OnceLock<std::result::Result<LoadedRerankerModel, String>>>>,
    /// Workspace RAG index slot. `None` until the first
    /// `POST /v1/rag/index` succeeds; a full reindex replaces the
    /// inner `RagIndex` atomically. Wrapped in `Arc<RwLock<…>>` so
    /// concurrent `/v1/rag/query` callers share the reader lock
    /// while a reindex holds the writer for the swap only.
    pub rag_index: Arc<
        tokio::sync::RwLock<Option<Arc<tokio::sync::RwLock<rustllama_rag::RagIndex>>>>,
    >,
    /// Bearer-token auth state. When `Some`, the router attaches
    /// a middleware that requires `Authorization: Bearer <key>` on
    /// every request except `/healthz` and same-host loopback
    /// requests. When `None` (empty `[server].api_key` or test
    /// fixtures), the middleware isn't attached and every endpoint
    /// is open — matching the historical "trust any local client"
    /// default.
    pub auth: Option<AuthState>,
    /// Monotonic clock value (seconds since process start) of the
    /// last user request. Updated by the audit middleware on every
    /// non-`/healthz` request. The opportunistic-refine background
    /// task reads this to decide "idle long enough to spend cycles
    /// on tuner work without stealing time from a user request."
    /// `0` means "never had a request"; the task waits for at least
    /// one before considering refine.
    pub last_request_secs: Arc<std::sync::atomic::AtomicU64>,
}

/// Bundled embedding-model + tokenizer cached in [`AppState::embedding_model`].
/// One GGUF open populates both; the handler dispatches on the input
/// shape (text → tokenize then forward; pre-tokenized → forward directly).
///
/// `Clone` is cheap — both fields are `Arc`'d under the hood — so the
/// handler can clone the bundle out of the `OnceLock` and run the
/// forward pass on a `spawn_blocking` worker without holding any
/// state-wide lock.
#[derive(Clone)]
pub struct LoadedEmbeddingModel {
    pub model: Arc<rustllama_models::bert_arch::BertModel>,
    pub tokenizer: Arc<rustllama_tokenizer::Tokenizer>,
}

/// Bundled reranker-model + tokenizer cached in
/// [`AppState::reranker_model`]. Same shape as
/// [`LoadedEmbeddingModel`] but the underlying `BertModel` is
/// guaranteed to carry a classifier head (loader rejects
/// embedding-only GGUFs with a clear error).
#[derive(Clone)]
pub struct LoadedRerankerModel {
    pub model: Arc<rustllama_models::bert_arch::BertModel>,
    pub tokenizer: Arc<rustllama_tokenizer::Tokenizer>,
}

/// Configured bearer-token authentication. Set when `[server].api_key`
/// is non-empty; cloned freely (cheap — internal storage is `Arc<[u8]>`).
/// Empty / unset → auth middleware no-ops, matching the historical
/// "any localhost client trusted" default.
#[derive(Clone)]
pub struct AuthState {
    /// Configured api_key bytes. Compared in constant time against
    /// the request's `Authorization: Bearer <token>` value.
    key: Arc<[u8]>,
    /// Per-key token bucket. v1 has a single global key so one bucket
    /// suffices; when multi-key auth lands, this becomes
    /// `HashMap<key_hash, RateLimitBucket>` keyed by SHA-256(key).
    /// `None` when `[server].rate_limit_per_minute` is unset or 0.
    rate_limit: Option<Arc<RateLimitBucket>>,
}

/// Token-bucket rate limiter. `tokens_per_minute` refills at a
/// constant rate; each request consumes 1 token. Bucket is shared
/// across all requests presenting the same key — so a single client
/// hammering the API gets throttled while other (unauthenticated
/// loopback or future-multi-key) traffic doesn't.
///
/// Implemented with a Mutex around `(tokens_left, last_refill)`
/// rather than an atomic — the refill calculation reads-modify-writes
/// two fields together, which is a natural Mutex shape. Contention is
/// per-key per-request, microseconds at most.
pub struct RateLimitBucket {
    /// Refill rate in tokens per second (config field is per-minute;
    /// stored as per-second for cheaper math).
    refill_rate: f64,
    /// Max bucket size — same as the per-minute limit so a single
    /// burst can use a minute's worth of capacity at once.
    capacity: f64,
    /// (tokens_remaining, last_refill_instant).
    state: std::sync::Mutex<(f64, std::time::Instant)>,
}

impl RateLimitBucket {
    fn new(per_minute: u32) -> Self {
        let capacity = per_minute as f64;
        Self {
            refill_rate: capacity / 60.0,
            capacity,
            state: std::sync::Mutex::new((capacity, std::time::Instant::now())),
        }
    }

    /// Try to admit one request. Returns `Ok` on success (1 token
    /// consumed), or `Err(retry_after_secs)` when the bucket is empty.
    /// `retry_after_secs` is `ceil(1.0 / refill_rate)` so the client
    /// gets a sensible `Retry-After` header value.
    pub fn try_admit(&self) -> std::result::Result<(), u64> {
        let mut g = self.state.lock().expect("rate-limit mutex poisoned");
        let (ref mut tokens, ref mut last) = *g;
        // Refill since last touch. Cap at capacity so a long idle
        // doesn't accumulate infinite credit.
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(*last).as_secs_f64();
        *tokens = (*tokens + elapsed * self.refill_rate).min(self.capacity);
        *last = now;
        if *tokens >= 1.0 {
            *tokens -= 1.0;
            Ok(())
        } else {
            let retry_after = (1.0 / self.refill_rate).ceil().max(1.0) as u64;
            Err(retry_after)
        }
    }
}

impl AuthState {
    pub fn new(api_key: &str) -> Option<Self> {
        Self::new_with_rate_limit(api_key, 0)
    }

    /// Construct with optional per-key rate limit. `per_minute == 0`
    /// disables the bucket; any non-zero value caps requests-per-
    /// minute for the configured key.
    pub fn new_with_rate_limit(api_key: &str, per_minute: u32) -> Option<Self> {
        if api_key.is_empty() {
            return None;
        }
        let rate_limit = if per_minute > 0 {
            Some(Arc::new(RateLimitBucket::new(per_minute)))
        } else {
            None
        };
        Some(Self {
            key: Arc::from(api_key.as_bytes().to_vec().into_boxed_slice()),
            rate_limit,
        })
    }

    /// Try to consume one request from the per-key rate-limit bucket.
    /// Returns `Ok` when the bucket is unset (no limit configured) or
    /// admission succeeds; `Err(retry_after_secs)` when the bucket is
    /// empty.
    pub fn try_admit_rate_limited(&self) -> std::result::Result<(), u64> {
        match self.rate_limit.as_ref() {
            None => Ok(()),
            Some(b) => b.try_admit(),
        }
    }
}

/// JSONL audit-log writer. One open file handle behind a mutex,
/// shared via `Arc` so cloning the surrounding state is cheap.
/// Records `{ts_ms, method, path, query, status, latency_ms}` per
/// request. Never writes bodies or header values — see
/// `audit_middleware` for the exact shape.
#[derive(Clone)]
pub struct AuditSink {
    writer: Arc<std::sync::Mutex<std::io::BufWriter<std::fs::File>>>,
    path: Arc<std::path::PathBuf>,
}

impl AuditSink {
    /// Open (or create + append) the file at `path`. Creates parent
    /// directories if needed. Returns an io error when the path is
    /// unwritable — caller decides whether to abort startup or just
    /// log the failure and continue without an audit log.
    pub fn try_open(path: std::path::PathBuf) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            writer: Arc::new(std::sync::Mutex::new(std::io::BufWriter::new(file))),
            path: Arc::new(path),
        })
    }

    /// Path the sink writes to. Surfaced via `/v1/lan_info`-shaped
    /// helpers so the GUI can show where the log lives.
    pub fn path(&self) -> &std::path::Path {
        &self.path
    }

    fn write_entry(&self, line: serde_json::Value) {
        use std::io::Write;
        if let Ok(mut w) = self.writer.lock() {
            // Best-effort: a write error is logged via `tracing` but
            // never bubbled — losing one audit line shouldn't break
            // the user's request.
            if let Err(e) = writeln!(w, "{line}") {
                tracing::warn!(error = %e, "audit log write failed");
                return;
            }
            // Flush eagerly so a `tail -f` viewer (or the GUI) sees
            // lines without waiting for the BufWriter to fill.
            let _ = w.flush();
        }
    }
}

/// RAII handle that removes a request's cancellation entry on drop.
/// Streaming handlers hold one for the request lifetime. The dereferenced
/// `Arc<AtomicBool>` is the actual signal: pass it to the engine, and
/// `POST /v1/cancel` flips it.
pub struct CancelGuard {
    pub id: String,
    pub flag: Arc<std::sync::atomic::AtomicBool>,
    pub(crate) cancellations:
        Arc<std::sync::Mutex<HashMap<String, Arc<std::sync::atomic::AtomicBool>>>>,
}

impl Drop for CancelGuard {
    fn drop(&mut self) {
        if let Ok(mut m) = self.cancellations.lock() {
            m.remove(&self.id);
        }
    }
}

/// Standard 404 response when a request references a model not loaded in
/// the registry. Helper so each handler doesn't reinvent the message.
/// Build an OpenAI-shaped error response:
/// `{"error":{"message":…,"type":…,"param":null,"code":…}}` with the
/// given HTTP status. OpenAI client SDKs (openai-python, openai-node,
/// the LangChain/LlamaIndex wrappers, editor plugins) parse
/// `error.message` / `error.type` / `error.code` out of this envelope
/// and raise a typed exception; a bare-string body makes them fail with
/// an opaque JSON-decode error instead. Every `/v1/*` failure path that
/// clients surface to users should route through here. Ollama-compat
/// (`/api/*`) keeps its own flat `{"error":"…"}` shape — see `ollama.rs`.
pub fn openai_error(
    status: StatusCode,
    message: impl Into<String>,
    err_type: &str,
    code: Option<&str>,
) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message.into(),
            "type": err_type,
            "param": serde_json::Value::Null,
            "code": code,
        }
    });
    (status, Json(body)).into_response()
}

pub fn model_not_found(requested: Option<&str>) -> Response {
    let message = match requested {
        Some(s) if !s.is_empty() => {
            format!("model '{s}' is not loaded. Load it via POST /v1/models/load.")
        }
        _ => "no default model loaded".to_string(),
    };
    openai_error(
        StatusCode::NOT_FOUND,
        message,
        "invalid_request_error",
        Some("model_not_found"),
    )
}

impl AppState {
    /// Elastic expert-cache budget (roadmap Phase 3): hot-apply a new
    /// `moe_expert_cache_mb` to every loaded CPU-engine model at a
    /// request-safe point. For each model, all gate permits are held
    /// while the cache is rebuilt — no generation is in flight — then
    /// released; queued requests proceed against the revised cache.
    /// Returns `true` when at least one model applied it. Mock-engine
    /// entries (no `cpu_engine`) are skipped.
    pub async fn apply_expert_budget_live(&self, budget_mb: u64) -> bool {
        let reg = self.inner.read().await;
        let mut any = false;
        for (id, m) in reg.models.iter() {
            let Some(cpu) = m.cpu_engine.as_ref() else {
                continue;
            };
            let permits = m.multi.as_ref().map(|p| p.engines.len()).unwrap_or(1) as u32;
            // Safe point: waiting our turn behind in-flight + queued
            // requests, then holding every permit.
            let Ok(_all) = m.gate.acquire_many(permits).await else {
                continue; // semaphore closed (shutdown) — skip
            };
            let (pinned, bytes) = cpu.apply_expert_cache_budget(budget_mb);
            tracing::info!(
                model_id = %id,
                budget_mb,
                prepinned_experts = pinned,
                prepinned_mb = bytes / (1024 * 1024),
                "expert-cache budget hot-applied without reload"
            );
            any = true;
        }
        any
    }

    /// Register a new in-flight request that the `/v1/cancel` endpoint
    /// can target. Returns a guard that automatically un-registers when
    /// the request finishes (success or error). The handler passes the
    /// guard's `flag` to the engine.
    pub fn register_cancel(&self, request_id: &str) -> CancelGuard {
        let flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
        if let Ok(mut m) = self.cancellations.lock() {
            m.insert(request_id.to_string(), flag.clone());
        }
        CancelGuard {
            id: request_id.to_string(),
            flag,
            cancellations: self.cancellations.clone(),
        }
    }

    /// Look up an in-flight request by id and flip its cancel flag.
    /// Returns true if the id was registered (false → 404).
    pub fn fire_cancel(&self, request_id: &str) -> bool {
        let m = match self.cancellations.lock() {
            Ok(m) => m,
            Err(_) => return false,
        };
        match m.get(request_id) {
            Some(flag) => {
                flag.store(true, std::sync::atomic::Ordering::Release);
                true
            }
            None => false,
        }
    }

    pub fn new(serving: ServingModel, version: String) -> Self {
        Self::with_max_loaded(serving, version, DEFAULT_MAX_LOADED_MODELS)
    }

    /// Construct an `AppState` with an explicit cap on the warm pool's
    /// size. Use `0` to disable the LRU eviction (the registry grows
    /// unbounded). See [`Registry::max_loaded`].
    pub fn with_max_loaded(serving: ServingModel, version: String, max_loaded: usize) -> Self {
        let default_id = serving.model_id.clone();
        let mut models = HashMap::new();
        models.insert(default_id.clone(), serving);
        Self {
            inner: Arc::new(tokio::sync::RwLock::new(Registry {
                models,
                default_id,
                max_loaded,
                use_counter: 0,
            })),
            cancellations: Arc::new(std::sync::Mutex::new(HashMap::new())),
            shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            version,
            config_path: None,
            #[cfg(feature = "history")]
            history: None,
            audit: None,
            auth: None,
            embedding_model: None,
            reranker_model: None,
            rag_index: Arc::new(tokio::sync::RwLock::new(None)),
            last_request_secs: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Construct an `AppState` with **no** loaded models. Used at GUI
    /// startup when the user hasn't configured `[model].path` yet —
    /// the server still runs (so `/healthz`, `/v1/models`, and the
    /// `POST /v1/models/load` endpoint all work), and the user loads
    /// a model via the Models page or the CLI.
    ///
    /// `max_pending_hint` is the default per-model `max_pending` cap
    /// to apply to any model that later gets loaded via `upsert` —
    /// it's stashed inside `AppState` for that future use.
    pub fn empty(version: String, max_loaded: usize, _max_pending_hint: usize) -> Self {
        Self {
            inner: Arc::new(tokio::sync::RwLock::new(Registry {
                models: HashMap::new(),
                default_id: String::new(),
                max_loaded,
                use_counter: 0,
            })),
            cancellations: Arc::new(std::sync::Mutex::new(HashMap::new())),
            shutdown: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            version,
            config_path: None,
            #[cfg(feature = "history")]
            history: None,
            audit: None,
            auth: None,
            embedding_model: None,
            reranker_model: None,
            rag_index: Arc::new(tokio::sync::RwLock::new(None)),
            last_request_secs: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }
    }

    /// Attach the `config.toml` path so `GET/PUT /v1/config` can read
    /// and persist the live configuration. `cli::serve` calls this
    /// with the resolved `--config` flag; tests leave it unset.
    pub fn with_config_path(mut self, path: std::path::PathBuf) -> Self {
        self.config_path = Some(Arc::new(path));
        self
    }

    /// True when no models have been loaded yet. Handlers that need a
    /// default model (e.g. `chat` without a `model` field) should
    /// emit `model_not_found(None)` when this is `true`.
    pub async fn is_empty(&self) -> bool {
        self.inner.read().await.models.is_empty()
    }

    /// Attach a conversation-history store. The GUI's Chat-page
    /// sidebar uses it; CLI / server-only deployments leave this
    /// `None` and the `/api/conversations/*` routes return 501.
    #[cfg(feature = "history")]
    pub fn with_history(mut self, store: Arc<crate::history::HistoryStore>) -> Self {
        self.history = Some(store);
        self
    }

    /// Attach an audit-log sink. When set, `router_with_cors`
    /// installs middleware that writes a JSONL line per request to
    /// the sink's file. Off by default (audit_log is opt-in).
    pub fn with_audit_sink(mut self, sink: AuditSink) -> Self {
        self.audit = Some(sink);
        self
    }

    /// Attach bearer-token auth state. When set, `router_with_cors`
    /// installs middleware that rejects requests missing or
    /// presenting a wrong `Authorization: Bearer <key>` header
    /// (except `/healthz` and loopback connections). Off by
    /// default — `AuthState::new("")` returns `None` so an empty
    /// `[server].api_key` keeps the legacy open-access behavior.
    pub fn with_auth(mut self, auth: AuthState) -> Self {
        self.auth = Some(auth);
        self
    }

    /// Enable the lazy embedding-model slot. Called by `cli::serve`
    /// when `[embeddings].path` or `.hub` is configured; the slot
    /// is otherwise `None` and `/v1/embeddings` falls into the
    /// "no model configured" 501 branch. The OnceLock itself is
    /// empty until the first embedding request triggers the load.
    pub fn with_embedding_slot(mut self) -> Self {
        self.embedding_model = Some(Arc::new(std::sync::OnceLock::new()));
        self
    }

    /// Enable the lazy reranker-model slot. Called by `cli::serve`
    /// when `[reranker].path` or `.hub` is configured. Mirrors
    /// `with_embedding_slot` — slot is otherwise `None` and
    /// `/v1/rerank` returns the same "no model configured" 501.
    pub fn with_reranker_slot(mut self) -> Self {
        self.reranker_model = Some(Arc::new(std::sync::OnceLock::new()));
        self
    }

    /// Adjust the warm-pool cap. Triggers eviction immediately if the
    /// new cap is smaller than the current registry size.
    pub async fn set_max_loaded(&self, n: usize) {
        let mut g = self.inner.write().await;
        g.max_loaded = n;
        if n > 0 {
            evict_to_fit(&mut g, n);
        }
    }

    pub async fn max_loaded(&self) -> usize {
        self.inner.read().await.max_loaded
    }

    /// Mark the server as draining. Subsequent new requests get 503 +
    /// Retry-After; in-flight requests run to completion. Idempotent:
    /// calling twice has the same effect as calling once.
    pub fn begin_shutdown(&self) {
        self.shutdown
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// True after [`begin_shutdown`] fires. Middleware reads this.
    pub fn is_shutting_down(&self) -> bool {
        self.shutdown.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Resolve a request's `model` field to a loaded `ServingModel`. Pass
    /// `None` (or an empty string) to get the current default. Returns
    /// `None` if the requested id is not in the registry.
    ///
    /// Bumps the model's `last_used` stamp so the LRU eviction policy
    /// keeps frequently-resolved models warm.
    pub async fn resolve(&self, model_id: Option<&str>) -> Option<ServingModel> {
        let mut g = self.inner.write().await;
        let key = match model_id {
            None => g.default_id.clone(),
            Some(s) if s.is_empty() => g.default_id.clone(),
            Some(s) => s.to_string(),
        };
        let model = g.models.get(&key).cloned()?;
        g.use_counter = g.use_counter.wrapping_add(1);
        model.last_used.store(g.use_counter, Ordering::Release);
        Some(model)
    }

    /// Convenience: snapshot of the current default model. Kept for
    /// handlers that don't accept a `model` field (healthz, /api/tags).
    /// Bumps the model's `last_used` stamp like [`Self::resolve`].
    ///
    /// Panics if the registry is empty (i.e., the server was started
    /// via [`Self::empty`] and no model has been loaded yet). Use
    /// [`Self::try_current`] when the empty case is possible.
    pub async fn current(&self) -> ServingModel {
        self.try_current()
            .await
            .expect("AppState::current called on empty registry; use try_current")
    }

    /// Fallible variant of [`Self::current`] — returns `None` when no
    /// default model is loaded. Healthz, metrics, and `/v1/models/load`
    /// (which inherits config from the prior default) use this so the
    /// server can start with an empty registry and accept a load
    /// request as the first user interaction.
    pub async fn try_current(&self) -> Option<ServingModel> {
        let mut g = self.inner.write().await;
        if g.default_id.is_empty() {
            return None;
        }
        let model = g.models.get(&g.default_id).cloned()?;
        g.use_counter = g.use_counter.wrapping_add(1);
        model.last_used.store(g.use_counter, Ordering::Release);
        Some(model)
    }

    /// Insert or replace `model.model_id` in the registry. Returns the
    /// previous entry if one existed. The newly-added model becomes the
    /// default *only if* the registry was empty before (which today it
    /// can't be — there's always a default — so this is a no-op there).
    ///
    /// If inserting would push the registry past `max_loaded`, the
    /// least-recently-used non-default entry is evicted first.
    pub async fn upsert(&self, model: ServingModel) -> Option<ServingModel> {
        let mut g = self.inner.write().await;
        let id = model.model_id.clone();
        let was_empty = g.models.is_empty();
        // Stamp the new model with a fresh `last_used` so it doesn't
        // get evicted before it's had a chance to handle any request.
        g.use_counter = g.use_counter.wrapping_add(1);
        model.last_used.store(g.use_counter, Ordering::Release);
        let prev = g.models.insert(id.clone(), model);
        if was_empty {
            g.default_id = id;
        }
        let cap = g.max_loaded;
        if cap > 0 {
            evict_to_fit(&mut g, cap);
        }
        prev
    }

    /// Remove a model from the registry. Returns the removed entry, or
    /// `None` if the id wasn't registered. Refuses to remove the last
    /// remaining model (server would have nothing to serve).
    pub async fn unload(&self, model_id: &str) -> Result<Option<ServingModel>, &'static str> {
        let mut g = self.inner.write().await;
        let removed = g.models.remove(model_id);
        // If the default was removed, promote the next remaining id — or
        // clear it when the registry is now empty. "Eject" intentionally
        // frees the last model too (LM Studio semantics): the server
        // tolerates an empty registry (every request path resolves via
        // `resolve`/`try_current`, which return None → "model not loaded"),
        // and dropping the last `ServingModel` releases its weights/RAM.
        // Clearing `default_id` keeps `try_current().is_empty()` accurate.
        if g.default_id == model_id {
            g.default_id = g.models.keys().next().cloned().unwrap_or_default();
        }
        Ok(removed)
    }

    /// All currently-loaded model ids alongside whether they're the
    /// current default. Order is implementation-defined.
    pub async fn list(&self) -> Vec<(String, bool)> {
        let g = self.inner.read().await;
        g.models
            .keys()
            .map(|id| (id.clone(), *id == g.default_id))
            .collect()
    }

    /// Promote an already-loaded model to the default. Errors if the
    /// requested id isn't in the registry.
    pub async fn set_default(&self, model_id: &str) -> Result<(), &'static str> {
        let mut g = self.inner.write().await;
        if !g.models.contains_key(model_id) {
            return Err("model not loaded");
        }
        g.default_id = model_id.to_string();
        Ok(())
    }

    /// Persist `model_id` as the on-disk default by writing its GGUF
    /// path to `[model].path` in config.toml, so the choice survives a
    /// restart. Best-effort: returns `false` (and logs) when there's
    /// no config path, the model isn't a file-backed CpuEngine, or the
    /// write fails — the in-memory default (set via [`Self::set_default`])
    /// is independent and already took. This closes the gap where
    /// `POST /v1/models/default` only changed the live default and
    /// reverted to the configured model on the next launch.
    pub async fn persist_default_model(&self, model_id: &str) -> bool {
        let Some(cfg_path) = self.config_path.as_ref() else {
            return false;
        };
        let src = {
            let g = self.inner.read().await;
            g.models
                .get(model_id)
                .and_then(|m| m.cpu_engine.as_ref())
                .map(|c| c.source_path().to_path_buf())
        };
        let Some(src) = src else {
            tracing::warn!(model_id, "persist default: no file-backed source path");
            return false;
        };
        let mut cfg = rustllama_config::load(cfg_path.as_ref()).unwrap_or_default();
        cfg.model.path = Some(src);
        cfg.model.hub = None; // path + hub are mutually exclusive
        match rustllama_config::save(cfg_path.as_ref(), &cfg) {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!(model_id, error = %e, "persist default: config write failed");
                false
            }
        }
    }
}

/// Evict the least-recently-used non-default models from the registry
/// until `models.len() <= cap`. Always preserves the current default,
/// even if it's the LRU — promoting eviction over the default would
/// strand the server with no default model.
///
/// In-flight requests hold their own `ServingModel` clone (taken via
/// `resolve()` / `current()` before any potential eviction), so the
/// engine keeps serving requests that already started against an
/// evicted entry; only future resolves miss.
fn evict_to_fit(g: &mut Registry, cap: usize) {
    while g.models.len() > cap {
        // Find the LRU non-default entry. The default is pinned
        // because promoting the next-LRU to default mid-eviction would
        // be confusing and isn't what users typically want.
        let mut victim: Option<(String, u64)> = None;
        for (id, m) in &g.models {
            if id == &g.default_id {
                continue;
            }
            let used = m.last_used.load(Ordering::Acquire);
            match &victim {
                Some((_, prev)) if *prev <= used => {}
                _ => victim = Some((id.clone(), used)),
            }
        }
        let Some((victim_id, _)) = victim else {
            // Only the default model remains — nothing more to evict.
            break;
        };
        g.models.remove(&victim_id);
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ServerError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("addr parse: {0}")]
    Addr(#[from] std::net::AddrParseError),
}

pub fn router(state: AppState) -> Router {
    router_with_cors(state, &[])
}

/// Same as [`router`] but additionally attaches a CORS layer that
/// allows the supplied origins. Use this when the server is bound on
/// LAN (`0.0.0.0`) so editors / web UIs running on a separate host
/// can `fetch()` the API — without an explicit `Access-Control-Allow-
/// Origin`, browsers reject the preflight.
///
/// Pass an empty slice to skip CORS entirely (the legacy default).
/// Pass `["*"]` to allow any origin, or a list of exact origins for
/// principle-of-least-privilege deployments. Localhost-only servers
/// don't need this — same-origin requests bypass CORS.
pub fn router_with_cors(state: AppState, cors_origins: &[String]) -> Router {
    use axum::middleware;
    use tower_http::cors::{Any, CorsLayer};
    use axum::http::HeaderValue;
    let router = Router::new()
        .route("/healthz", get(healthz))
        .route("/v1/metrics", get(metrics))
        .route("/v1/tokenize", post(tokenize_handler))
        .route("/v1/detokenize", post(detokenize_handler))
        .route("/v1/models", get(list_models))
        .route("/v1/models/load", post(load_model))
        .route("/v1/models/unload", post(unload_model))
        .route("/v1/models/:id", get(get_model))
        // Read-only metadata probe for a cached GGUF (loaded or not).
        // The GUI Models page hits this when the user clicks "Inspect"
        // on a cached row to see arch / size / dtype histogram without
        // paying the load cost.
        .route("/api/gguf/inspect", post(inspect_gguf))
        .route("/v1/models/default", post(set_default_model))
        .route("/v1/cancel", post(cancel_request))
        .route("/v1/config", get(get_config).put(put_config))
        // Quick-switch between named `[[profiles]]` blocks in config.toml.
        // Server loads the on-disk config, applies the named profile's
        // sparse overrides, saves the merged result, and returns the
        // same diff/reload-flag shape as PUT /v1/config. The on-disk
        // watcher picks up the write and live-applies what it can.
        .route("/v1/config/profile/apply", post(apply_config_profile))
        // Render a Jinja chat template against a sample conversation.
        // The Settings page's chat_template editor uses this for a
        // live preview — the user can see exactly what the prompt
        // string looks like before saving the template to config.
        .route("/v1/chat/template/preview", post(preview_chat_template))
        // Tuner-cache visibility for the GUI Status page: surfaces
        // device fingerprint, last-tuned timestamp, kernel-LWS entry
        // counts, placement winners, batch_size winner, and the
        // auto_apply flags so the user can see what `rustllama tune`
        // has captured.
        .route("/v1/tuning_summary", get(tuning_summary))
        // Pull-shaped TuningRecommended event surface. Returns the
        // (kernel, M, K) shapes the engine has dispatched without a
        // cached LWS entry. GUI Status page polls this on a slow tick
        // to show "N kernels untuned — Tune now." Cleared after a
        // successful `rustllama tune` via `POST` to the same path.
        .route(
            "/v1/tuning/recommendations",
            get(tuning_recommendations).post(tuning_recommendations_clear),
        )
        // Run-tune endpoints — GUI buttons + scripted-tune flows
        // hit these to drive measurement loops without dropping
        // to a terminal. Synchronous (blocks for the whole sweep,
        // ~30s-5min depending on model size + candidate count) and
        // persists the winner to the per-device cache so the next
        // model load consumes it automatically.
        .route("/v1/tune/placement", post(tune_placement))
        .route("/v1/tune/batch_size", post(tune_batch_size))
        // Full-sweep auto-tune: `POST /v1/tune/model` force-retunes a
        // model (drives `tune --all` as a subprocess); `GET
        // /v1/tune/progress` streams the live stage/percent/log the GUI
        // renders in its progress window. First-load auto-tune runs
        // inline in `load_model` (see `autotune::is_untuned`).
        .route("/v1/tune/model", post(autotune::retune_handler))
        .route("/v1/tune/progress", get(autotune::progress_handler))
        // Audit log tail: surface the last N JSONL entries so the
        // GUI Settings page can show recent requests without
        // shelling out to `tail -f`. Audit log writing itself is
        // gated by `[server].audit_log = true`; the tail endpoint
        // works regardless of that flag — it just reads the
        // configured file off disk.
        .route("/v1/audit_log/tail", get(audit_log_tail))
        // LAN discovery for the GUI: returns the host's primary LAN
        // IP + the URL another device should hit + an SVG QR for
        // that URL. Useful for "scan with phone, chat from there"
        // when the server binds to 0.0.0.0.
        .route("/v1/lan_info", get(lan_info))
        // Single-shot feature-detection. Editor / GUI clients hit
        // this once at startup to learn what's wired without
        // probing each endpoint.
        .route("/v1/capabilities", get(capabilities))
        // Crash log management: list + view + delete files dropped
        // by the runtime panic hook. The GUI Settings page renders
        // a viewer panel so users don't have to dig into the
        // user-data dir to triage a panic.
        .route("/v1/crash_logs", get(list_crash_logs))
        .route(
            "/v1/crash_logs/:name",
            get(read_crash_log).delete(delete_crash_log),
        )
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/completions", post(completions::completions))
        // Typed-decision endpoints (Choice / Score / Boolean): return
        // a typed value + probabilities by scoring candidate options against
        // the loaded model instead of generating prose.
        .route("/v1/decide/choice", post(decide::choice))
        .route("/v1/decide/score", post(decide::score))
        .route("/v1/decide/boolean", post(decide::boolean))
        // Extended typed-decision surface: full ranking, per-label yes/no,
        // best-of-N candidate selection, tool selection, and raw sequence
        // likelihood / perplexity. All reuse the same scoring primitives.
        .route("/v1/decide/rank", post(decide::rank))
        .route("/v1/decide/labels", post(decide::labels))
        .route("/v1/decide/best_of", post(decide::best_of))
        .route("/v1/decide/tool", post(decide::tool))
        .route("/v1/score", post(decide::sequence_score))
        // Anthropic Messages API — Claude Code, anthropic-sdk-* clients.
        .route("/v1/messages", post(anthropic::messages))
        // Ollama-compatibility surface. Editors that target localhost:11434
        // (Continue.dev, ZED, llama-vscode, Cline, ...) speak this shape.
        .route("/api/tags", get(ollama::tags))
        .route("/api/show", post(ollama::show))
        .route("/api/chat", post(ollama::chat))
        .route("/api/generate", post(ollama::generate))
        .route("/api/version", get(ollama::version))
        .route("/api/pull", post(ollama::pull))
        // Realtime HuggingFace discovery for the GUI Models tab (proxied
        // server-side to avoid the webview's CORS restrictions).
        .route("/api/hf/search", get(ollama::hf_search))
        .route("/api/hf/files", get(ollama::hf_files))
        .route("/api/delete", axum::routing::delete(ollama::delete_model))
        .route("/api/ps", get(ollama::ps))
        // Embeddings endpoints. All three share the lazy-loaded
        // BERT bundle from `[embeddings]` config — when unconfigured
        // they 501 with the same diagnostic. The OpenAI-shape
        // `/v1/embeddings` accepts both text and pre-tokenized int
        // input; `/api/embeddings` is the legacy Ollama shape
        // (single `prompt` → `{embedding}`); `/api/embed` is the
        // modern Ollama shape (string | array → `{embeddings, …}`).
        .route("/api/embeddings", post(ollama::embeddings_legacy))
        .route("/api/embed", post(ollama::embed))
        .route("/v1/embeddings", post(embeddings::embeddings))
        // Reranker (Cohere/Jina shape). Backed by the same
        // BERT-loader path as embeddings but the GGUF must carry
        // a `cls.weight` classifier head. Off by default — 501
        // when `[reranker]` is unconfigured.
        .route("/v1/rerank", post(rerank::rerank))
        // RAG endpoints. Both require `[embeddings]` to be configured;
        // the inner store is in-memory and ephemeral in v1 (sqlite +
        // file-watching are deferred). `/index` walks + chunks +
        // embeds a workspace root; `/query` runs cosine top-K.
        .route("/v1/rag/index", post(rag::index))
        .route("/v1/rag/query", post(rag::query))
        // RAG persistence: serialize the in-memory store to disk
        // (custom binary format, `RLLMRAG\0` magic) and reload it
        // later. Pairs with `/v1/rag/index` — index once at startup
        // / on demand, save, then reload across server restarts
        // without re-walking the workspace + re-embedding.
        .route("/v1/rag/save", post(rag::save))
        .route("/v1/rag/load", post(rag::load))
        // Incremental refresh: re-walk the listed paths, drop their
        // existing chunks, embed + re-add. The file-watcher case
        // wires `/v1/rag/update` from a background tokio task — for
        // v1 we expose only the manual HTTP form so the lifecycle of
        // the watcher itself stays opt-in / out-of-band.
        .route("/v1/rag/update", post(rag::update));

    // Conversation history routes (GUI sidebar). Only registered when
    // the server is built with `--features history`; non-feature
    // clients see 404 rather than 501 since the routes don't exist.
    #[cfg(feature = "history")]
    let router = router
        .route(
            "/api/conversations",
            get(history::list_handler).post(history::create_handler),
        )
        // FTS5 search across all stored message contents. Registered
        // before `/:id` so axum's path matcher doesn't try to parse
        // "search" as an i64.
        .route(
            "/api/conversations/search",
            get(history::search_handler),
        )
        .route(
            "/api/conversations/:id",
            get(history::get_handler)
                .delete(history::delete_handler)
                .patch(history::rename_handler),
        )
        .route(
            "/api/conversations/:id/messages",
            post(history::append_handler),
        )
        .route(
            "/api/conversations/:id/export",
            get(history::export_handler),
        );

    // CORS layer: opt-in via `[server].cors_origins`. Empty slice
    // means "no CORS headers" (same-origin / curl / native clients).
    // Wildcard `"*"` forwards `Access-Control-Allow-Origin: *`; named
    // origins build an exact-match allow list.
    let router = if cors_origins.is_empty() {
        router
    } else if cors_origins.iter().any(|o| o == "*") {
        // Wildcard: allow any origin, any method, any header.
        // Browsers reject `*` together with credentials, which is
        // fine because we don't issue cookies. Editor / web-UI use
        // cases work.
        router.layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        )
    } else {
        // Build an exact-match allow list. Skip any entry that
        // doesn't parse as a HeaderValue rather than failing the
        // whole startup — a typo in one origin shouldn't take down
        // the server.
        let origins: Vec<HeaderValue> = cors_origins
            .iter()
            .filter_map(|o| HeaderValue::from_str(o).ok())
            .collect();
        router.layer(
            CorsLayer::new()
                .allow_origin(origins)
                .allow_methods(Any)
                .allow_headers(Any),
        )
    };

    // Audit middleware attaches only when a sink is configured.
    // Off-by-default — zero overhead for users / tests that haven't
    // opted in. When on, the middleware runs `next` first and
    // appends the JSONL line after, so the recorded `latency_ms`
    // matches what the client actually saw.
    let router = if state.audit.is_some() {
        router.layer(middleware::from_fn_with_state(
            state.clone(),
            audit_middleware,
        ))
    } else {
        router
    };

    // Bearer-token auth attaches only when `[server].api_key` is
    // non-empty. Layers run outermost-last in axum, so auth runs
    // BEFORE audit (good — failed-auth requests still get logged,
    // since audit wraps the whole stack including auth's 401s).
    let router = if state.auth.is_some() {
        router.layer(middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
    } else {
        router
    };

    router
        .layer(middleware::from_fn_with_state(
            state.clone(),
            shutdown_guard_middleware,
        ))
        .with_state(state)
}

/// Per-request audit-log middleware. Runs the request through, then
/// appends a one-line JSON record to the audit sink. Never logs
/// bodies or header values — only method, path, query (with `api_key`
/// / `token` / `key` / `auth` params redacted), status, and
/// wall-clock latency.
async fn audit_middleware(
    State(state): State<AppState>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let start = std::time::Instant::now();
    let method = req.method().to_string();
    let path = req.uri().path().to_string();
    let query = req
        .uri()
        .query()
        .map(redact_query_string)
        .unwrap_or_default();
    let resp = next.run(req).await;
    if let Some(sink) = state.audit.as_ref() {
        let ts_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let line = serde_json::json!({
            "ts_ms": ts_ms,
            "method": method,
            "path": path,
            "query": query,
            "status": resp.status().as_u16(),
            "latency_ms": start.elapsed().as_millis() as u64,
        });
        sink.write_entry(line);
    }
    resp
}

// ----- /v1/audit_log/tail --------------------------------------------------
//
// Reads the configured audit log file off disk + returns the last
// N JSONL entries parsed as structured rows. The audit middleware
// writes entries by appending; this handler reads-only. Works
// independently of the `[server].audit_log` flag — if the file
// exists, the handler tails it regardless of whether new entries
// are still being written.
//
// Resolves the file path the same way `cli::serve` does: explicit
// `[server].audit_log_path` if set, otherwise
// `<crash_log_dir>/audit.log.jsonl`. Returns null `path` +
// empty `entries` when the file doesn't exist (fresh install
// before any tune / no audit-log run yet).

#[derive(serde::Deserialize)]
struct AuditTailQuery {
    /// Max entries to return. Default 50, hard-capped at 1000.
    #[serde(default = "default_audit_tail_n")]
    n: usize,
}

fn default_audit_tail_n() -> usize {
    50
}

#[derive(Serialize)]
struct AuditTailResponse {
    /// Resolved on-disk path, or null when no file exists yet.
    path: Option<String>,
    /// Total entry count across the whole file. Larger than
    /// `entries.len()` when the file holds more than `n` lines.
    total_entries: usize,
    /// Most-recent entries first (newest at index 0). Each is the
    /// `{ts_ms, method, path, query, status, latency_ms}` shape
    /// the audit middleware writes. Returned as raw JSON values so
    /// future audit-format additions show through without a
    /// breaking schema bump on this endpoint.
    entries: Vec<serde_json::Value>,
}

async fn audit_log_tail(
    State(state): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<AuditTailQuery>,
) -> Response {
    let n = q.n.min(1000);
    let cfg = state
        .config_path
        .as_ref()
        .and_then(|p| rustllama_config::load(p.as_ref()).ok())
        .unwrap_or_default();
    let path = if cfg.server.audit_log_path.is_empty() {
        rustllama_runtime::paths()
            .crash_log_dir
            .join("audit.log.jsonl")
    } else {
        std::path::PathBuf::from(&cfg.server.audit_log_path)
    };
    if !path.exists() {
        return Json(AuditTailResponse {
            path: None,
            total_entries: 0,
            entries: Vec::new(),
        })
        .into_response();
    }
    // Stream the file through a bounded ring buffer of the last N
    // parsed entries. Memory stays O(N) regardless of file size —
    // LAN deployments at sustained 10 req/sec grow ~170MB/day, so
    // a `read_to_string` of the full file got pricey fast. Two-pass
    // would let us tally `total_entries` first; we fold it into the
    // same pass instead by counting valid lines as we stream.
    let file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("open audit log {}: {e}", path.display()),
            )
                .into_response();
        }
    };
    use std::io::BufRead;
    let reader = std::io::BufReader::new(file);
    let mut total: usize = 0;
    let mut ring: std::collections::VecDeque<serde_json::Value> =
        std::collections::VecDeque::with_capacity(n.min(1024));
    for line in reader.lines() {
        let line = match line {
            Ok(s) => s,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("read audit log {}: {e}", path.display()),
                )
                    .into_response();
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else {
            continue;
        };
        total += 1;
        if n == 0 {
            continue;
        }
        if ring.len() == n {
            ring.pop_front();
        }
        ring.push_back(v);
    }
    // Reverse to newest-first so the GUI table renders the most-
    // recent request at the top without an extra reverse step.
    let tail: Vec<serde_json::Value> = ring.into_iter().rev().collect();
    Json(AuditTailResponse {
        path: Some(path.display().to_string()),
        total_entries: total,
        entries: tail,
    })
    .into_response()
}

/// Mask sensitive query-string parameters before logging. Param
/// values for the listed keys are replaced with `***`. Comparison
/// is ASCII case-insensitive so `API_KEY=…` redacts too.
/// Everything else passes through verbatim — the audit consumer
/// needs the original keys for filtering.
///
/// The list covers the common spellings editor clients and SDKs
/// use when smuggling secrets through query strings (some pre-OAuth
/// clients do `?api_key=...` instead of bearer headers). Bodies +
/// headers are never logged regardless — they're filtered at the
/// middleware layer.
fn redact_query_string(q: &str) -> String {
    const SENSITIVE: &[&str] = &[
        // Bearer / OAuth shapes
        "api_key",
        "apikey",
        "access_token",
        "refresh_token",
        "bearer",
        "authorization",
        // Generic names
        "key",
        "token",
        "auth",
        "secret",
        // Password shapes (HTTP-basic-in-URL legacy clients)
        "password",
        "passwd",
        "pwd",
        // PIN / OTP / single-use codes
        "pin",
        "passcode",
        "otp",
        // Signed-URL bits (S3-style presigned-URL pre-flight probes)
        "signature",
        "x-amz-signature",
    ];
    q.split('&')
        .map(|pair| {
            let (k, _v) = match pair.split_once('=') {
                Some((k, v)) => (k, v),
                None => return pair.to_string(),
            };
            let lk = k.to_ascii_lowercase();
            if SENSITIVE.iter().any(|s| *s == lk) {
                format!("{k}=***")
            } else {
                pair.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("&")
}

/// Bearer-token auth middleware. Reads `Authorization: Bearer <key>`
/// from the request, constant-time compares it against the configured
/// api_key, returns 401 with `WWW-Authenticate: Bearer` on failure.
///
/// Bypasses (in priority order):
///   - `/healthz` always — monitoring probes shouldn't need creds
///   - Same-host loopback requests when `ConnectInfo` resolves to
///     a loopback peer — the GUI's own fetches and local CLI clients
///     bypass auth, but a sibling on the LAN must authenticate
///
/// The parsed bearer token lives in a `zeroize::Zeroizing<Vec<u8>>`
/// so it gets wiped from memory when the request completes, win or
/// lose. Constant-time compare via `subtle::ConstantTimeEq` defends
/// against timing side-channels.
async fn auth_middleware(
    State(state): State<AppState>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    use subtle::ConstantTimeEq;

    let Some(auth) = state.auth.as_ref() else {
        return next.run(req).await;
    };

    // /healthz: open. Monitoring probes shouldn't need creds.
    if req.uri().path() == "/healthz" {
        return next.run(req).await;
    }

    // Loopback bypass: ConnectInfo is set by `axum::serve(…
    // .into_make_service_with_connect_info::<SocketAddr>())`. When
    // the GUI / CLI / curl hit 127.0.0.1, we treat them as trusted.
    // When ConnectInfo isn't available (test harnesses use
    // `app.oneshot()` which doesn't set it), the bypass doesn't
    // fire and the request must present the token like any other
    // remote — tests opt in explicitly.
    if let Some(connect_info) = req
        .extensions()
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
    {
        if connect_info.0.ip().is_loopback() {
            return next.run(req).await;
        }
    }

    // Pull the Authorization header. Missing → 401.
    let raw = match req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
    {
        Some(s) => s,
        None => return auth_401("missing Authorization header"),
    };

    // Accept both `Bearer <key>` (case-insensitive scheme) and a
    // bare `<key>` for clients that don't add the scheme. The Bearer
    // prefix is the documented form; bare tokens are a courtesy.
    let token_str = match raw.split_once(' ') {
        Some((scheme, rest)) if scheme.eq_ignore_ascii_case("Bearer") => rest.trim(),
        _ => raw.trim(),
    };

    // Copy into a Zeroizing wrapper so the parsed bytes get wiped
    // when this scope ends — successful auth, failed auth, or panic.
    let presented: zeroize::Zeroizing<Vec<u8>> =
        zeroize::Zeroizing::new(token_str.as_bytes().to_vec());

    // subtle's ct_eq returns Choice (1/0); call .into() to bool.
    // Comparing slices of different lengths is fine — ct_eq XORs
    // up to the shorter length and folds in a length mismatch, so
    // a 5-byte presented token vs 10-byte configured key fails
    // without leaking the key length via early-return timing.
    let matched: bool = presented.as_slice().ct_eq(&auth.key).into();
    if !matched {
        return auth_401("invalid api key");
    }

    // Token is correct. `presented` drops here (zeroed) — `next`
    // doesn't see it.
    drop(presented);

    // Per-key rate limit. v1 has a single configured key so the
    // bucket is per-process, not per-key — when multi-key auth lands
    // the bucket lookup becomes `HashMap<key_hash, Bucket>`. Skipped
    // when `[server].rate_limit_per_minute` is 0 / unset.
    if let Err(retry_after) = auth.try_admit_rate_limited() {
        return (
            StatusCode::TOO_MANY_REQUESTS,
            [(axum::http::header::RETRY_AFTER, retry_after.to_string())],
            "rate limit exceeded for this api key",
        )
            .into_response();
    }

    next.run(req).await
}

fn auth_401(msg: &'static str) -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [(axum::http::header::WWW_AUTHENTICATE, "Bearer")],
        msg,
    )
        .into_response()
}

/// Middleware that rejects new requests with `503 Service Unavailable +
/// Retry-After: 5` once the server has entered its drain window. The
/// `/healthz` path stays open so external monitors can observe the
/// drain state. In-flight requests aren't affected — they continue to
/// completion under axum's `with_graceful_shutdown`.
/// Background task: when `[tuning].opportunistic_refine = true`, wake
/// every minute and check if the server's been idle for `idle_threshold_secs`
/// and the untuned-shape registry has entries. If both, log a
/// recommendation event so operators / the GUI can act on it (run
/// `rustllama tune` against the same model).
///
/// v1 of this task only **observes** + emits the recommendation; the
/// actual kernel-LWS sweep on an idle engine is gated on a shared
/// `Mutex<SyclStream>` that the engine doesn't expose yet. The
/// scaffold is in place so that lifecycle hook lands cleanly in a
/// follow-up turn.
///
/// Spawned by `cli::serve` when the config flag is set; otherwise the
/// task never starts and the cost is zero. Exits when the receiver
/// half of a shutdown signal closes.
pub fn spawn_opportunistic_refine_task(
    state: AppState,
    idle_threshold_secs: u64,
) -> tokio::task::JoinHandle<()> {
    tokio::task::spawn(async move {
        // Poll once per minute. Cheap: the task does one atomic load
        // + a Vec snapshot of the untuned-shape registry. No
        // contention with request handlers.
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(60));
        loop {
            interval.tick().await;
            if state.is_shutting_down() {
                tracing::debug!("opportunistic-refine task exiting on shutdown");
                return;
            }
            let last = state
                .last_request_secs
                .load(std::sync::atomic::Ordering::Relaxed);
            if last == 0 {
                // Never saw a request yet — wait for one before
                // considering refine.
                continue;
            }
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let idle = now.saturating_sub(last);
            if idle < idle_threshold_secs {
                continue;
            }
            // Snapshot the untuned-shape registry. Skip when empty.
            let shapes = rustllama_models::accel::snapshot_untuned_shapes();
            if shapes.is_empty() {
                continue;
            }
            tracing::info!(
                idle_secs = idle,
                untuned_count = shapes.len(),
                "opportunistic-refine: idle long enough + untuned shapes exist — \
                 run `rustllama tune` to populate the cache (auto-sweep on an \
                 idle engine ships in a follow-up turn)"
            );
            // Future enhancement: actually kick off a sweep here.
            // Requires the engine to expose a `try_borrow_sycl_stream`
            // shape that returns `None` when an inference is in
            // flight, so we don't compete with a user request that
            // arrived between our idle check and the sweep start.
        }
    })
}

async fn shutdown_guard_middleware(
    State(state): State<AppState>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    if state.is_shutting_down() {
        let path = req.uri().path();
        // Let monitoring + cancel endpoints through so an operator can
        // observe the drain and abort in-flight requests if needed.
        if path != "/healthz" && path != "/v1/cancel" {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                [("retry-after", "5")],
                "server is draining; retry shortly",
            )
                .into_response();
        }
    }
    // Idle tracking: bump `last_request_secs` on every non-`/healthz`
    // request. Polling probes shouldn't reset the idle clock —
    // otherwise the opportunistic-refine task never sees real idle
    // time on a server with active liveness monitoring.
    if req.uri().path() != "/healthz" {
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        state
            .last_request_secs
            .store(secs, std::sync::atomic::Ordering::Relaxed);
    }
    next.run(req).await
}

#[derive(serde::Deserialize)]
struct LoadModelRequest {
    /// Absolute path to a `.gguf` file. Mutually exclusive with `hub`
    /// and `name`.
    path: Option<std::path::PathBuf>,
    /// HuggingFace reference of the form `owner/repo:filename`. Resolves
    /// against the local cache; does NOT trigger a download (use the CLI
    /// `pull` command or `/api/pull` for that).
    hub: Option<String>,
    /// Short name — the GGUF's file stem as surfaced by `GET /api/tags`
    /// (e.g. `"DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32"`). Resolved by
    /// walking the cache and matching `path.file_stem()`. This is the
    /// shape the GUI Models page sends when the user clicks "Load" on
    /// a cached row that doesn't look like a HuggingFace ref.
    name: Option<String>,
    /// Optional context size override. Defaults to 8192.
    ctx_size: Option<usize>,
    /// Optional prefill chunk size (tokens). Drives partial-prefix-cache
    /// snapshot granularity and the prefill trace-span size. Defaults to
    /// 512 if unset.
    batch_size: Option<u32>,
    /// Optional KV-cache dtype: `"f32"` (default), `"q8_0"`,
    /// `"q4_0"`, `"tq1"`/`"tq2"`/`"tq4"`/`"tq8"`, or `"nvfp4"`. See the
    /// `[inference].kv_dtype` docs in `rustllama-config` for the
    /// memory-vs-accuracy tradeoffs.
    kv_dtype: Option<String>,
}

#[derive(Serialize)]
struct LoadModelResponse {
    model_id: String,
    /// `None` if this is the first load (registry was empty / new id).
    /// Otherwise the id of the previously-occupying entry for this id.
    previous_model_id: Option<String>,
    is_default: bool,
    loaded_at: u64,
}

async fn load_model(
    State(state): State<AppState>,
    Json(req): Json<LoadModelRequest>,
) -> Response {
    let provided = (
        req.path.is_some(),
        req.hub.is_some(),
        req.name.is_some(),
    );
    let path: std::path::PathBuf = match provided {
        (true, false, false) => req.path.unwrap(),
        (false, true, false) => {
            let hub_ref = req.hub.unwrap();
            let Ok(href) = rustllama_hub::HubRef::parse(&hub_ref) else {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("invalid hub ref `{hub_ref}`: expected `owner/repo:filename`"),
                )
                    .into_response();
            };
            let Some(cache) = rustllama_hub::default_cache_dir() else {
                return (StatusCode::INTERNAL_SERVER_ERROR, "no cache dir").into_response();
            };
            let p = href.local_path(&cache);
            if !p.exists() {
                return (
                    StatusCode::NOT_FOUND,
                    format!(
                        "model not in cache: {}. Run `rustllama pull {hub_ref}` first.",
                        p.display()
                    ),
                )
                    .into_response();
            }
            p
        }
        (false, false, true) => {
            // Short-name resolution: walk the cache, match file stems.
            // This is the path the GUI Models page hits when the user
            // clicks "Load" on a row from `/api/tags` (which surfaces
            // file stems, not full paths).
            let name = req.name.unwrap();
            let Some(cache) = rustllama_hub::default_cache_dir() else {
                return (StatusCode::INTERNAL_SERVER_ERROR, "no cache dir").into_response();
            };
            let paths = match rustllama_hub::list_cached(&cache) {
                Ok(p) => p,
                Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("failed to enumerate cache: {e}"),
                    )
                        .into_response();
                }
            };
            let matched = paths.into_iter().find(|p| {
                p.file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s == name)
                    .unwrap_or(false)
            });
            match matched {
                Some(p) => p,
                None => {
                    return (
                        StatusCode::NOT_FOUND,
                        format!(
                            "no cached GGUF with file stem `{name}` under {}",
                            cache.display()
                        ),
                    )
                        .into_response();
                }
            }
        }
        (false, false, false) => {
            return (
                StatusCode::BAD_REQUEST,
                "specify one of `path`, `hub` (owner/repo:filename), or `name` (file stem)",
            )
                .into_response();
        }
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                "specify exactly one of `path`, `hub`, or `name`",
            )
                .into_response();
        }
    };
    // Model-card-driven ctx_size: when the caller didn't pin a
    // value, respect what the model was trained for, capped at the
    // historical 8192 default so a 1M-ctx model doesn't allocate
    // 1M of KV cache by surprise. A 4096-ctx model gets 4096
    // (saves memory); a 32K-ctx model still caps at 8192 (user
    // opts in to higher by passing ctx_size explicitly).
    const DEFAULT_CTX_CAP: usize = 8192;
    let max_ctx = match req.ctx_size {
        Some(v) => v,
        None => {
            let inferred = peek_gguf_ctx_train(&path)
                .map(|n| n as usize)
                .unwrap_or(DEFAULT_CTX_CAP);
            let resolved = inferred.min(DEFAULT_CTX_CAP);
            tracing::info!(
                model_path = %path.display(),
                inferred_ctx_train = inferred,
                resolved_ctx_size = resolved,
                "ctx_size unset — derived from model metadata"
            );
            resolved
        }
    };

    // Pre-flight: refuse loads that would push the host past its
    // OS commit budget — physical RAM + page-file / swap space the
    // kernel can reserve. The on-disk GGUF size is a near-lower-
    // bound on commit usage (raw quant tensors get copied 1:1;
    // sub-L3 tensors may grow ~4× when dequantized to F16). We use
    // the file size + a 1.5× safety factor and compare against
    // commit-available, not physical-available — a host with a
    // generous page file can load a model that won't fit in
    // physical RAM (cold pages spill to disk; GPU-resident weights
    // stay hot in USM). The check skips when we can't stat the
    // file.
    //
    // If physical-available is *also* below the need, we log a
    // warning so the user knows their load will pressure the
    // working set even though it's allowed. Better that than a
    // false-positive refusal when the OS is happy to back the
    // commit with swap.
    if let Ok(meta) = std::fs::metadata(&path) {
        let file_bytes = meta.len();
        let mem = rustllama_runtime::memory_info();
        let need = file_bytes.saturating_mul(3) / 2;
        if need > mem.commit_available_bytes {
            return (
                StatusCode::INSUFFICIENT_STORAGE,
                format!(
                    "model would not fit in available commit (RAM + page file): \
                     file is {} MiB, estimated peak need ~{} MiB (1.5×), only \
                     {} MiB commit-available of {} MiB total commit (physical \
                     RAM: {} MiB avail / {} MiB total). Pick a smaller quant \
                     (Q4_K_M / IQ3_S) or a smaller parameter count, set \
                     `[inference].keep_quant_raw = true` to skip F16 dequant, \
                     or increase the system page-file size.",
                    file_bytes / 1_048_576,
                    need / 1_048_576,
                    mem.commit_available_bytes / 1_048_576,
                    mem.commit_total_bytes / 1_048_576,
                    mem.available_bytes / 1_048_576,
                    mem.total_bytes / 1_048_576,
                ),
            )
                .into_response();
        }
        if need > mem.available_bytes {
            tracing::warn!(
                file_mib = file_bytes / 1_048_576,
                need_mib = need / 1_048_576,
                phys_avail_mib = mem.available_bytes / 1_048_576,
                commit_avail_mib = mem.commit_available_bytes / 1_048_576,
                "model load exceeds physical RAM available; OS will back the \
                 overflow with the page file. Expect slower load + cold-page \
                 reads during inference. Close other apps if performance \
                 matters."
            );
        }
    }

    // First-load auto-tune (Milestone 1d). Autotune is MANDATORY before a
    // model is served: when this model has never been tuned on this device,
    // run the full sweep BEFORE loading so the kv_dtype / placement /
    // dispatch / per-device-perf winners below are read from the freshly-
    // populated cache. Blocking by design — the GUI opens a progress window
    // and polls `GET /v1/tune/progress`. A failed sweep is non-fatal: the
    // load just falls back to config + coherence-guardrail defaults. An
    // already-cached model is an instant no-op. The exact same call runs on
    // the CLI `serve` startup load (see the shared `maybe_first_load_autotune`),
    // with `console: true` there.
    {
        let tune_path = path.clone();
        let _ = tokio::task::spawn_blocking(move || {
            autotune::maybe_first_load_autotune(&tune_path, false)
        })
        .await;
    }

    // Load happens on a blocking pool — mapping a multi-GB GGUF and
    // memcpy-ing weights into per-layer tensors easily blows past the
    // tokio worker timing budget.
    let kv_dtype = match req.kv_dtype.as_deref() {
        None | Some("") => rustllama_engine::KvDtype::F32,
        Some(s) => {
            // Coherence guardrail (1d): downgrade an aggressive quant
            // KV to f32 when this model isn't validated for it.
            let (safe, warn) = rustllama_config::coherence_safe_kv_dtype(
                s,
                &path,
                rustllama_config::force_quant_kv_from_env(),
            );
            if let Some(w) = warn {
                tracing::warn!("{w}");
            }
            match rustllama_engine::KvDtype::parse(&safe) {
                Some(d) => d,
                None => {
                    return (
                        StatusCode::BAD_REQUEST,
                        format!(
                            "unknown kv_dtype `{s}` (expected: f32, q8_0, q4_0, \
                             tq1/tq2/tq4/tq8, or nvfp4)"
                        ),
                    )
                        .into_response();
                }
            }
        }
    };
    let load_path = path.clone();
    let load_result = tokio::task::spawn_blocking(move || {
        rustllama_engine::CpuEngine::load_with_options(&load_path, max_ctx, true, kv_dtype)
    })
    .await;
    let mut cpu = match load_result {
        Ok(Ok(c)) => c,
        Ok(Err(e)) => {
            // Phase 2-C removed the load-time MoE rejection (the
            // engine now supports MoE inference end-to-end). The
            // previous special-case 400 with `error_type:
            // "moe_not_supported"` is gone — any remaining load
            // error is a real "couldn't load this file" failure
            // and surfaces as 500.
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("load model {}: {e}", path.display()),
            )
                .into_response();
        }
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };
    // Carry the runtime inference knobs over to the newly-loaded model so
    // hot-loaded models behave like the startup-loaded one.
    //
    // Precedence (highest first):
    //   1. Request field (`req.batch_size`) — explicit override
    //   2. Tuner cache entry for this device + model — when
    //      `[tuning].auto_apply_*` is true and the cache has a
    //      relevant slot populated by `rustllama tune`
    //   3. Config default (`[inference].batch_size` /
    //      `[inference].n_gpu_layers`) — legacy fallback
    //
    // The cache reads happen here (not inside the blocking load
    // task) so the auto_apply flags can be flipped via PUT
    // /v1/config without a restart.
    let cfg_for_tuning = state
        .config_path
        .as_ref()
        .and_then(|p| rustllama_config::load(p.as_ref()).ok())
        .unwrap_or_default();
    let model_key = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown-model")
        .to_string();
    let resolved_batch_size = match req.batch_size {
        Some(size) => Some(size),
        None => {
            if cfg_for_tuning.tuning.auto_apply_batch_size {
                tuner_cached_batch_size().map(|n| {
                    tracing::info!(
                        model = %model_key,
                        applied = n,
                        "load_model: applied batch_size from tuner cache"
                    );
                    n
                })
            } else {
                None
            }
        }
    };
    if let Some(size) = resolved_batch_size {
        cpu.set_prefill_chunk_size(size as usize);
    }
    // Placement precedence (mirrors the CLI `serve` load path):
    //   1. an explicit `[inference].n_gpu_layers` override (any value other
    //      than the AUTO sentinel), honored only when the GPU tier is enabled;
    //   2. else a flat placement winner cached for this (device, model) from
    //      `rustllama tune --placement` (gated by `auto_apply_placement`);
    //   3. else AUTO: measured-perf heat placement, which subsumes the flat
    //      layer cutoff + the MoE experts→CPU split and installs a per-tensor
    //      device plan. The mandatory first-load tune above has already
    //      populated `per_device_perf`, so this produces a real heat plan;
    //      pre-tune it falls back to the VRAM-fit planner. Cheap (queries VRAM
    //      + counts weight bytes, no generation) so it runs inline.
    {
        let cfg_inf = &cfg_for_tuning.inference;
        let override_n = if cfg_inf.gpu_enabled {
            cfg_inf.n_gpu_layers_override()
        } else {
            None
        };
        let cached_n = cfg_for_tuning
            .tuning
            .auto_apply_placement
            .then(|| tuner_cached_placement(&model_key))
            .flatten();
        if let Some(n_gpu) = override_n {
            tracing::info!(
                model = %model_key,
                applied = n_gpu,
                "load_model: explicit [inference].n_gpu_layers override"
            );
            cpu.set_n_gpu_layers(n_gpu);
        } else if let Some(n_gpu) = cached_n {
            tracing::info!(
                model = %model_key,
                applied = n_gpu,
                "load_model: applied n_gpu_layers from tuner cache"
            );
            cpu.set_n_gpu_layers(n_gpu);
        } else {
            let opts = rustllama_engine::placement_auto::AutoPlacementOpts {
                cpu_enabled: cfg_inf.cpu_enabled,
                gpu_enabled: cfg_inf.gpu_enabled,
                vram_only: cfg_inf.vram_only,
                ..Default::default()
            };
            let decision = cpu.auto_place_heat(&opts);
            if let Some(err) = &decision.placement_error {
                // Device-tier constraint (cpu_enabled=false / vram_only) not
                // satisfied. The engine keeps its prior cutoff; surface the
                // reason (the HTTP load path does not hard-fail here).
                tracing::warn!(
                    model = %model_key,
                    error = %err,
                    "load_model: device-tier placement constraint not satisfied"
                );
            }
            tracing::info!(
                model = %model_key,
                n_gpu_layers = decision.n_gpu_layers,
                total_layers = decision.total_layers,
                reason = %decision.reason,
                "load_model: auto placement (measured-perf heat plan, VRAM-fit fallback)"
            );
        }
    }
    let model_id = cpu.model_id().to_string();
    let cpu = Arc::new(cpu);
    // Pre-upload packed-quant weights to USM on a blocking-pool
    // thread so the first chat doesn't pay the lazy weight-cache
    // miss cost. No-op when SYCL is disabled or no GPU is visible.
    cpu.warmup_for_sycl_async().await;
    // Inherit the max_pending + concurrency from the current default
    // model. When the registry is empty (first load after starting
    // with no [model].path), fall back to sensible defaults: the
    // single-flight gate (concurrency=1) and the standard
    // backpressure cap.
    let (inherit_max_pending, inherit_concurrency) = match state.try_current().await {
        Some(s) => (s.max_pending, s.concurrency()),
        None => (DEFAULT_MAX_PENDING_PER_MODEL, 1usize),
    };
    let new_serving = ServingModel {
        engine: cpu.clone() as Arc<dyn Engine>,
        cpu_engine: Some(cpu),
        model_id: model_id.clone(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(AtomicUsize::new(0)),
        max_pending: inherit_max_pending,
        last_used: Arc::new(AtomicU64::new(0)),
        multi: None,
    }
    .with_concurrency(inherit_concurrency);

    // Upsert into the registry. In-flight generations hold their own
    // `ServingModel` clone (taken via `AppState::resolve` at request
    // start) so the displaced entry's `Arc`s stay alive until those
    // requests drain.
    let prev = state.upsert(new_serving).await;
    // After upsert, current always has SOMETHING — even if the
    // registry was empty before, `upsert` promotes the new model to
    // default automatically.
    let default_id = state
        .try_current()
        .await
        .map(|s| s.model_id)
        .unwrap_or_default();
    let is_default = default_id == model_id;
    let loaded_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let previous_model_id = prev.as_ref().map(|p| p.model_id.clone());
    if let Some(p) = &prev {
        tracing::info!(model = %model_id, replaced = %p.model_id, "model upserted");
    } else {
        tracing::info!(model = %model_id, "model loaded");
    }
    Json(LoadModelResponse {
        model_id,
        previous_model_id,
        is_default,
        loaded_at,
    })
    .into_response()
}

// ----- POST /api/gguf/inspect -----------------------------------------------
//
// Read-only metadata probe for a cached GGUF, loaded or not. Returns the
// architecture, transformer dimensions, tensor stats, and a dtype histogram
// — everything the CLI `models inspect` subcommand surfaces, in JSON shape.
// No engine is created; the file is mmap-opened, the header + tensor table
// parsed, and the mmap dropped. Safe to call against multi-GB GGUFs.
//
// Body shape mirrors the load endpoint: exactly one of `path`, `hub`,
// `name`. The GUI Models page passes `name` (the file-stem surface from
// `/api/tags`); editors poking at the API directly use `path`.

#[derive(serde::Deserialize)]
struct InspectGgufRequest {
    /// Absolute path to a `.gguf` file. Mutually exclusive with `hub`
    /// and `name`.
    path: Option<std::path::PathBuf>,
    /// HuggingFace ref `owner/repo:filename`. Resolved against the
    /// local cache; doesn't trigger a download.
    hub: Option<String>,
    /// Short name — the GGUF file stem as surfaced by `/api/tags`.
    name: Option<String>,
    /// When `true`, include the full per-tensor list (name, dtype,
    /// shape, byte size, offset). Off by default since a 7B model has
    /// ~300 tensors and the GUI doesn't render all of them — the
    /// dtype histogram is the at-a-glance view.
    #[serde(default)]
    include_tensors: bool,
}

#[derive(Serialize)]
struct InspectDtypeBucket {
    dtype: &'static str,
    tensor_count: u64,
    bytes: u64,
}

#[derive(Serialize)]
struct InspectTensorEntry {
    name: String,
    dtype: &'static str,
    shape: Vec<u64>,
    elements: u64,
    bytes: u64,
}

#[derive(Serialize)]
struct InspectGgufResponse {
    path: String,
    architecture: String,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    size_label: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    file_type: Option<u32>,
    context_length: Option<u32>,
    block_count: Option<u32>,
    embedding_length: Option<u32>,
    head_count: Option<u32>,
    head_count_kv: Option<u32>,
    head_dim: Option<u32>,
    vocab_size: Option<u32>,
    /// Mixture-of-experts routed expert count. Present only when
    /// the GGUF carries `{arch}.expert_count` (Mixtral / Qwen3-MoE /
    /// DeepSeek-V3); absent on dense GGUFs.
    #[serde(skip_serializing_if = "Option::is_none")]
    n_experts: Option<u32>,
    /// Top-K routed experts per token. Present alongside
    /// `n_experts`. Mixtral=2, Qwen3-MoE=8, DeepSeek-V3=8.
    #[serde(skip_serializing_if = "Option::is_none")]
    n_experts_used: Option<u32>,
    /// Always-active shared experts (DeepSeek-V3 only — 1).
    /// Present alongside `n_experts` even when 0 so clients can
    /// distinguish "shared not reported" from "0 shared".
    #[serde(skip_serializing_if = "Option::is_none")]
    n_experts_shared: Option<u32>,
    tensor_count: u64,
    total_params: u64,
    total_tensor_bytes: u64,
    file_bytes: u64,
    dtypes: Vec<InspectDtypeBucket>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tensors: Option<Vec<InspectTensorEntry>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_card: Option<rustllama_hub::ModelCard>,
}

async fn inspect_gguf(Json(req): Json<InspectGgufRequest>) -> Response {
    // Path resolution: same precedence as `/v1/models/load` but
    // without the RAM preflight (we're not loading) or kv_dtype
    // (we're not running). Refactoring the load_model resolver into a
    // shared helper would mean threading state, errors, and the cache
    // dir through it — the duplication is contained and clearer
    // inline.
    let provided = (
        req.path.is_some(),
        req.hub.is_some(),
        req.name.is_some(),
    );
    let path: std::path::PathBuf = match provided {
        (true, false, false) => req.path.unwrap(),
        (false, true, false) => {
            let hub_ref = req.hub.unwrap();
            let Ok(href) = rustllama_hub::HubRef::parse(&hub_ref) else {
                return (
                    StatusCode::BAD_REQUEST,
                    format!("invalid hub ref `{hub_ref}`: expected `owner/repo:filename`"),
                )
                    .into_response();
            };
            let Some(cache) = rustllama_hub::default_cache_dir() else {
                return (StatusCode::INTERNAL_SERVER_ERROR, "no cache dir").into_response();
            };
            let p = href.local_path(&cache);
            if !p.exists() {
                return (
                    StatusCode::NOT_FOUND,
                    format!("model not in cache: {}", p.display()),
                )
                    .into_response();
            }
            p
        }
        (false, false, true) => {
            let name = req.name.unwrap();
            let Some(cache) = rustllama_hub::default_cache_dir() else {
                return (StatusCode::INTERNAL_SERVER_ERROR, "no cache dir").into_response();
            };
            let paths = match rustllama_hub::list_cached(&cache) {
                Ok(p) => p,
                Err(e) => {
                    return (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        format!("failed to enumerate cache: {e}"),
                    )
                        .into_response();
                }
            };
            let matched = paths.into_iter().find(|p| {
                p.file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s == name)
                    .unwrap_or(false)
            });
            match matched {
                Some(p) => p,
                None => {
                    return (
                        StatusCode::NOT_FOUND,
                        format!(
                            "no cached GGUF with file stem `{name}` under {}",
                            cache.display()
                        ),
                    )
                        .into_response();
                }
            }
        }
        (false, false, false) => {
            return (
                StatusCode::BAD_REQUEST,
                "specify one of `path`, `hub`, or `name`",
            )
                .into_response();
        }
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                "specify exactly one of `path`, `hub`, or `name`",
            )
                .into_response();
        }
    };

    // Open + parse on a blocking pool: opening mmaps the file and the
    // tensor-info table can be hundreds of KB to walk on a 70B model.
    let include_tensors = req.include_tensors;
    let join = tokio::task::spawn_blocking(move || inspect_gguf_blocking(&path, include_tensors))
        .await;
    match join {
        Ok(Ok(resp)) => Json(resp).into_response(),
        Ok(Err((status, msg))) => (status, msg).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

/// `GET /v1/tuning_summary` — visibility surface for the GUI
/// Status page. Reports what `rustllama tune` has captured in the
/// per-device cache + the live `[tuning].auto_apply_*` flags so
/// the user can see whether the engine is consuming cached values
/// or running on config defaults.
///
/// Mock-mode hosts (no SYCL device visible) return null fingerprint
/// + empty cache fields rather than 404-ing — the panel renders a
/// "no SYCL device" state in that case.

#[derive(Serialize)]
struct TuningSummaryResponse {
    /// The single SYCL device shown for the cache fingerprint; `null`
    /// when no SYCL device is detected (NVIDIA-only / CPU-only host, or
    /// no oneAPI runtime). See `gpus` for the full inventory.
    device: Option<TuningDeviceFp>,
    /// Full compute-GPU inventory (SYCL + CUDA), so the tuning panel
    /// shows every GPU — not only the single SYCL `device` above. The
    /// tuner cache is keyed by the whole-system fingerprint, so tuning
    /// already spans all of these; this is the display list.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    gpus: Vec<GpuMetric>,
    /// The host CPU compute tier — the fallback backend — shown in the
    /// inventory alongside `gpus` so the tuning panel reflects every tier
    /// placement can target, not just the GPUs.
    cpu: CpuMetric,
    /// Where the cache file lives. `null` when no cache dir
    /// resolvable (rare — happens only when both portable detection
    /// + standard dirs fail).
    cache_path: Option<String>,
    /// `true` when a cache file existed AND parsed cleanly. `false`
    /// for fresh installs (no cache yet) or a parse error.
    cache_present: bool,
    /// ISO-8601-ish epoch-seconds string from the last tune run.
    /// `null` when no tune has ever completed for this device.
    last_tuned: Option<String>,
    /// Total tuned kernel-LWS entries (sum across all kernels +
    /// shapes). Surfaces "have I run `rustllama tune` at all?" at
    /// a glance.
    kernel_entry_count: usize,
    /// Full kernel-LWS table — one row per (kernel, shape-bucket)
    /// the user has tuned. `key` is the cache's HashMap key
    /// (kernel + bucketed shape, e.g. `q4k_packed_usm:M=4096,K=4096`);
    /// `kernel` is the canonical kernel name; `params` is the
    /// tuner-specific JSON blob (`{"lws": 128}` for the packed
    /// matvec kernels). Surfaced for power users debugging "why
    /// isn't my tune speeding things up" — collapsed by default in
    /// the GUI panel since 30+ entries get long.
    kernels: Vec<TuningKernelEntry>,
    /// Per-model placement winners.
    placement: Vec<TuningPlacementEntry>,
    /// Per-device prefill chunk-size winner.
    batch_size: Option<u32>,
    /// Per-device kv_dtype winner from `rustllama tune --kv-dtype`
    /// or `--all`. `None` when no kv_dtype sweep has run.
    kv_dtype: Option<String>,
    /// Per-device flash_attention winner.
    flash_attention: Option<bool>,
    /// Per-device kv_cache_layout winner.
    kv_cache_layout: Option<String>,
    /// Live auto_apply flags so the GUI can label the panel
    /// "ACTIVE" vs "stored but not applied".
    auto_apply_placement: bool,
    auto_apply_batch_size: bool,
    auto_apply_kv_dtype: bool,
    auto_apply_flash_attention: bool,
    auto_apply_kv_cache_layout: bool,
}

#[derive(Serialize)]
struct TuningDeviceFp {
    pci_id: u32,
    name: String,
    driver_ver: String,
    vram_mb: u64,
    /// Filesystem-safe slug used as the cache filename.
    slug: String,
}

#[derive(Serialize)]
struct TuningPlacementEntry {
    model_key: String,
    n_gpu_layers: u32,
}

#[derive(Serialize)]
struct TuningKernelEntry {
    /// HashMap key from `TuningResult.kernels` — typically
    /// `<kernel_name>:<shape_bucket>`.
    key: String,
    /// Canonical kernel name (e.g. `q4k_packed_usm`).
    kernel: String,
    /// Tuner-specific JSON blob. For packed-matvec kernels this is
    /// `{"lws": <u32>}`; future kernel families may carry richer
    /// param shapes.
    params: serde_json::Value,
}

async fn tuning_summary(State(state): State<AppState>) -> Response {
    let cfg = state
        .config_path
        .as_ref()
        .and_then(|p| rustllama_config::load(p.as_ref()).ok())
        .unwrap_or_default();

    let (
        device,
        cache_path,
        cache_present,
        last_tuned,
        kernel_entry_count,
        kernels,
        placement,
        batch_size,
        kv_dtype,
        flash_attention,
        kv_cache_layout,
    ) = match rustllama_tuner::default_cache_dir() {
        Some(cache_dir) => {
            // Cache key = whole-system fingerprint (resolves on SYCL/CUDA/CPU).
            // The SYCL device, if any, is surfaced only for display.
            let key = rustllama_tuner::system_fingerprint();
            let path = rustllama_tuner::cache_path_for(&cache_dir, &key);
            let path_str = Some(path.display().to_string());
            let device_summary =
                rustllama_tuner::fingerprint_active_device().map(|fp| TuningDeviceFp {
                    pci_id: fp.pci_id,
                    name: fp.name.clone(),
                    driver_ver: fp.driver_ver.clone(),
                    vram_mb: fp.vram_mb,
                    slug: fp.slug(),
                });
            match rustllama_tuner::load_cache(&cache_dir, &key) {
                Ok(Some(tuning)) => {
                    let placement_entries: Vec<TuningPlacementEntry> = tuning
                        .placement
                        .iter()
                        .map(|(k, v)| TuningPlacementEntry {
                            model_key: k.clone(),
                            n_gpu_layers: v.n_gpu_layers,
                        })
                        .collect();
                    // Sort kernel rows by key so the GUI table
                    // ordering is stable across reloads — HashMap
                    // iteration order would otherwise reshuffle on
                    // every fetch and the panel would scroll-jitter.
                    let mut kernel_entries: Vec<TuningKernelEntry> = tuning
                        .kernels
                        .iter()
                        .map(|(k, v)| TuningKernelEntry {
                            key: k.clone(),
                            kernel: v.kernel.clone(),
                            params: v.params.clone(),
                        })
                        .collect();
                    kernel_entries.sort_by(|a, b| a.key.cmp(&b.key));
                    (
                        device_summary,
                        path_str,
                        true,
                        tuning.last_tuned.clone(),
                        tuning.kernels.len(),
                        kernel_entries,
                        placement_entries,
                        tuning.batch_size,
                        tuning.kv_dtype.clone(),
                        tuning.flash_attention,
                        tuning.kv_cache_layout.clone(),
                    )
                }
                _ => (
                    device_summary,
                    path_str,
                    false,
                    None,
                    0,
                    Vec::new(),
                    Vec::new(),
                    None,
                    None,
                    None,
                    None,
                ),
            }
        }
        None => (
            None,
            None,
            false,
            None,
            0,
            Vec::new(),
            Vec::new(),
            None,
            None,
            None,
            None,
        ),
    };

    Json(TuningSummaryResponse {
        device,
        gpus: gpus_probe(),
        cpu: cpus_probe(),
        cache_path,
        cache_present,
        last_tuned,
        kernel_entry_count,
        kernels,
        placement,
        batch_size,
        kv_dtype,
        flash_attention,
        kv_cache_layout,
        auto_apply_placement: cfg.tuning.auto_apply_placement,
        auto_apply_batch_size: cfg.tuning.auto_apply_batch_size,
        auto_apply_kv_dtype: cfg.tuning.auto_apply_kv_dtype,
        auto_apply_flash_attention: cfg.tuning.auto_apply_flash_attention,
        auto_apply_kv_cache_layout: cfg.tuning.auto_apply_kv_cache_layout,
    })
    .into_response()
}

// ----- /v1/tune/placement + /v1/tune/batch_size ----------------------------
//
// Synchronous run-tune endpoints. Drive the engine's measurement
// loops via `rustllama_engine::measurement::*`, persist the winner
// to the per-device tuner cache, return the full structured report
// so the GUI can render a results card.
//
// Blocking: a 7B Q4_K_M placement sweep with 32 candidates ×
// 3 repeats × ~5s per run is on the order of 5-10 minutes. The
// fetch() call on the GUI side handles this fine; we just don't
// stream progress (yet). Single-flight: no concurrency guard —
// running tune while a chat is in flight contends for the engine
// and produces noisier measurements. Document, don't enforce.

#[derive(serde::Deserialize)]
struct TunePlacementRequest {
    /// Absolute path to a `.gguf` file. Mutually exclusive with
    /// `model_name`. One of the two must be present.
    #[serde(default)]
    model_path: Option<String>,
    /// Short name — the GGUF file stem as surfaced by `GET
    /// /api/tags`. Resolved by walking the local cache for a
    /// matching stem (same logic as `/v1/models/load` with
    /// `name`). The GUI passes this so users don't have to know
    /// absolute paths.
    #[serde(default)]
    model_name: Option<String>,
    /// Context size for the measurement load + candidate KV sizing.
    /// Defaults to 2048 — big enough for prefill cost to matter,
    /// small enough that KV cache fits on Iris Xe-class budgets.
    #[serde(default = "default_ctx_size_for_tune")]
    ctx_size: u32,
    /// Synthetic prompt length per measurement run. Default 64.
    #[serde(default = "default_tune_prompt_tokens")]
    prompt_tokens: u32,
    /// Decode tokens per measurement run. Default 32.
    #[serde(default = "default_tune_decode_tokens")]
    decode_tokens: u32,
    /// Repeats per candidate; median tok/s is the score. Default 3.
    #[serde(default = "default_tune_repeats")]
    repeats: u32,
    /// VRAM budget in MiB for the candidate-generation step.
    /// Defaults to 4096 — sensible Iris Xe-class start.
    #[serde(default = "default_vram_mb")]
    vram_mb: u64,
    /// Headroom in MiB kept free after candidates fit. Default 256.
    #[serde(default = "default_vram_headroom_mb")]
    vram_headroom_mb: u64,
}

fn default_ctx_size_for_tune() -> u32 {
    2048
}
fn default_tune_prompt_tokens() -> u32 {
    64
}
fn default_tune_decode_tokens() -> u32 {
    32
}
fn default_tune_repeats() -> u32 {
    3
}
fn default_vram_mb() -> u64 {
    4096
}
fn default_vram_headroom_mb() -> u64 {
    256
}

#[derive(Serialize)]
struct TunePlacementResponse {
    winner: Option<u32>,
    winner_tps: f64,
    load_ms: f64,
    candidates: Vec<TunePlacementCandidate>,
    /// Path the winner was persisted to, or `null` when no SYCL
    /// device was detected (mock-mode build) — the measurement
    /// itself still runs in that case (CPU fallback), but the
    /// result can't be written to a device-keyed cache file.
    cache_path: Option<String>,
}

#[derive(Serialize)]
struct TunePlacementCandidate {
    n_gpu_layers: u32,
    warmup_ms: f64,
    median_tps: Option<f64>,
    max_tps: Option<f64>,
    error: Option<String>,
}

#[derive(Serialize)]
struct TuningRecommendationsResponse {
    /// Total distinct `(kernel, M, K)` shapes the engine has seen
    /// without a cached LWS. `0` means everything's tuned (or the
    /// engine hasn't dispatched any USM matvec yet on this run).
    untuned_count: u32,
    /// Per-shape detail, sorted lexicographically. Bounded in practice
    /// (the engine sees ~30 distinct shapes per model loaded), so the
    /// wire payload stays small.
    shapes: Vec<TuningRecommendationShape>,
}

#[derive(Serialize)]
struct TuningRecommendationShape {
    kernel: &'static str,
    m: u32,
    k: u32,
}

/// `GET /v1/tuning/recommendations` — list of matvec shapes the engine
/// has dispatched on the current run that the autotuner cache had no
/// entry for. The GUI Status page polls this on a slow tick to render
/// a "N kernels untuned — Tune now" banner with a link to the tune
/// buttons. Returns `{ untuned_count: 0, shapes: [] }` when everything
/// is tuned, when the engine hasn't seen any USM matvec yet, or on
/// non-SYCL (mock-mode) builds where the registry stays empty.
async fn tuning_recommendations(State(_state): State<AppState>) -> Response {
    let shapes = rustllama_models::accel::snapshot_untuned_shapes();
    let resp = TuningRecommendationsResponse {
        untuned_count: shapes.len() as u32,
        shapes: shapes
            .into_iter()
            .map(|(kernel, m, k)| TuningRecommendationShape {
                kernel,
                m: m as u32,
                k: k as u32,
            })
            .collect(),
    };
    Json(resp).into_response()
}

/// `POST /v1/tuning/recommendations` — reset the untuned-shape
/// registry. Called by the GUI's "Tune now" flow after a successful
/// run, and by the CLI `rustllama tune` exit path. Idempotent — a
/// post on an empty registry is a no-op. Returns the new (empty)
/// snapshot so the caller can refresh its UI in one round-trip.
async fn tuning_recommendations_clear(State(_state): State<AppState>) -> Response {
    rustllama_models::accel::clear_untuned_shapes();
    Json(TuningRecommendationsResponse {
        untuned_count: 0,
        shapes: Vec::new(),
    })
    .into_response()
}

async fn tune_placement(
    State(state): State<AppState>,
    Json(req): Json<TunePlacementRequest>,
) -> Response {
    let cfg = state
        .config_path
        .as_ref()
        .and_then(|p| rustllama_config::load(p.as_ref()).ok())
        .unwrap_or_default();
    let model_path = match resolve_tune_model_path(
        req.model_path.as_deref(),
        req.model_name.as_deref(),
    ) {
        Ok(p) => p,
        Err((status, msg)) => return (status, msg).into_response(),
    };

    // Run the whole sweep on a blocking-pool thread — measurement
    // is heavy and pure CPU/GPU work, no point holding a tokio
    // worker for minutes.
    let join = tokio::task::spawn_blocking(move || {
        run_placement_sweep_blocking(&model_path, &req, &cfg)
    })
    .await;
    match join {
        Ok(Ok(resp)) => Json(resp).into_response(),
        Ok(Err((status, msg))) => (status, msg).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

fn run_placement_sweep_blocking(
    model_path: &std::path::Path,
    req: &TunePlacementRequest,
    cfg: &rustllama_config::Config,
) -> std::result::Result<TunePlacementResponse, (StatusCode, String)> {
    use rustllama_engine::measurement::{
        measure_placement_candidates, MeasurementConfig,
    };
    use rustllama_tuner::placement::{candidate_placements, DEFAULT_SWEEP_STEP};

    // Step 1: derive ModelDims + quant from the GGUF — mirrors the
    // CLI's `read_dims_and_quant_from_gguf` but inlined here so the
    // server doesn't take rustllama-cli as a dep.
    let (dims, quant) = read_dims_and_quant_from_gguf_for_tune(model_path)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("GGUF read failed: {e}")))?;

    // Step 2: generate VRAM-budget-fitting candidates.
    let candidates = candidate_placements(
        dims,
        quant,
        req.vram_mb * 1024 * 1024,
        req.vram_headroom_mb * 1024 * 1024,
        req.ctx_size,
        DEFAULT_SWEEP_STEP,
    );
    if candidates.is_empty() {
        return Err((
            StatusCode::UNPROCESSABLE_ENTITY,
            "no candidate fits the VRAM budget — raise --vram-mb or lower --placement-ctx"
                .to_string(),
        ));
    }

    // Step 3: resolve the K dtype for measurement. K and V can be
    // configured independently; today's coupled `KvLayer` storage
    // means the engine actually uses K's dtype for both sides, so
    // measurement does the same. A V override is logged so the user
    // sees what landed.
    let (k, v) = cfg.inference.resolved_kv_dtypes();
    if k != v {
        tracing::warn!(
            k = k,
            v = v,
            "split K/V dtypes configured; measurement uses K's dtype \
             (engine storage couples K/V today — V dtype takes effect once \
             per-side KV storage lands)"
        );
    }
    let kv_dtype = rustllama_engine::KvDtype::parse(&k.to_string()).ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("unrecognized kv_dtype: {k:?}"),
        )
    })?;
    let m_cfg = MeasurementConfig {
        kv_dtype,
        kv_cache_layout: cfg.inference.kv_cache_layout.clone(),
        flash_attention: cfg.inference.flash_attention,
        n_gpu_layers: cfg.inference.n_gpu_layers,
    };

    // Step 4: run the engine's measurement loop.
    let report = measure_placement_candidates(
        model_path,
        &candidates,
        req.ctx_size as usize,
        req.prompt_tokens,
        req.decode_tokens,
        req.repeats,
        &m_cfg,
    )
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("measurement failed: {e}"),
        )
    })?;

    // Step 5: persist winner to the tuner cache (if SYCL device
    // visible). Best-effort — a persistence failure shouldn't lose
    // the measurement result.
    let cache_path = report.winner.and_then(|n_gpu| {
        let cache_dir = rustllama_tuner::default_cache_dir()?;
        let (key, device) = rustllama_tuner::cache_context();
        let model_key = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown-model")
            .to_string();
        persist_placement_winner_inline(&cache_dir, &key, &device, &model_key, n_gpu).ok()
    });

    Ok(TunePlacementResponse {
        winner: report.winner,
        winner_tps: report.winner_tps,
        load_ms: report.load_ms,
        candidates: report
            .candidates
            .iter()
            .map(|c| TunePlacementCandidate {
                n_gpu_layers: c.n_gpu_layers,
                warmup_ms: c.warmup_ms,
                median_tps: c.median_tps,
                max_tps: c.max_tps,
                error: c.error.clone(),
            })
            .collect(),
        cache_path: cache_path.map(|p| p.display().to_string()),
    })
}

#[derive(serde::Deserialize)]
struct TuneBatchSizeRequest {
    #[serde(default)]
    model_path: Option<String>,
    #[serde(default)]
    model_name: Option<String>,
    /// Candidate batch sizes. Defaults to the plan-prescribed grid.
    #[serde(default = "default_batch_candidates")]
    candidates: Vec<u32>,
    #[serde(default = "default_batch_prompt_tokens")]
    prompt_tokens: u32,
    #[serde(default = "default_tune_repeats")]
    repeats: u32,
}

fn default_batch_candidates() -> Vec<u32> {
    vec![128, 256, 512, 1024, 2048]
}
fn default_batch_prompt_tokens() -> u32 {
    2048
}

#[derive(Serialize)]
struct TuneBatchSizeResponse {
    winner: Option<u32>,
    winner_tps: f64,
    load_ms: f64,
    candidates: Vec<TuneBatchSizeCandidate>,
    cache_path: Option<String>,
}

#[derive(Serialize)]
struct TuneBatchSizeCandidate {
    batch_size: u32,
    warmup_ms: f64,
    median_tps: Option<f64>,
    max_tps: Option<f64>,
    error: Option<String>,
}

async fn tune_batch_size(
    State(state): State<AppState>,
    Json(req): Json<TuneBatchSizeRequest>,
) -> Response {
    let cfg = state
        .config_path
        .as_ref()
        .and_then(|p| rustllama_config::load(p.as_ref()).ok())
        .unwrap_or_default();
    let model_path = match resolve_tune_model_path(
        req.model_path.as_deref(),
        req.model_name.as_deref(),
    ) {
        Ok(p) => p,
        Err((status, msg)) => return (status, msg).into_response(),
    };
    if req.candidates.is_empty() {
        return (StatusCode::BAD_REQUEST, "candidates must be non-empty").into_response();
    }

    let join = tokio::task::spawn_blocking(move || {
        run_batch_size_sweep_blocking(&model_path, &req, &cfg)
    })
    .await;
    match join {
        Ok(Ok(resp)) => Json(resp).into_response(),
        Ok(Err((status, msg))) => (status, msg).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

fn run_batch_size_sweep_blocking(
    model_path: &std::path::Path,
    req: &TuneBatchSizeRequest,
    cfg: &rustllama_config::Config,
) -> std::result::Result<TuneBatchSizeResponse, (StatusCode, String)> {
    use rustllama_engine::measurement::{
        measure_batch_size_candidates, MeasurementConfig,
    };

    let (k, v) = cfg.inference.resolved_kv_dtypes();
    if k != v {
        tracing::warn!(
            k = k,
            v = v,
            "split K/V dtypes configured; measurement uses K's dtype \
             (engine storage couples K/V today — V dtype takes effect once \
             per-side KV storage lands)"
        );
    }
    let kv_dtype = rustllama_engine::KvDtype::parse(&k.to_string()).ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("unrecognized kv_dtype: {k:?}"),
        )
    })?;
    let m_cfg = MeasurementConfig {
        kv_dtype,
        kv_cache_layout: cfg.inference.kv_cache_layout.clone(),
        flash_attention: cfg.inference.flash_attention,
        n_gpu_layers: cfg.inference.n_gpu_layers,
    };
    let candidates: Vec<usize> = req.candidates.iter().map(|n| *n as usize).collect();
    let report = measure_batch_size_candidates(
        model_path,
        &candidates,
        req.prompt_tokens,
        req.repeats,
        &m_cfg,
    )
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("measurement failed: {e}"),
        )
    })?;

    let cache_path = report.winner.and_then(|b| {
        let cache_dir = rustllama_tuner::default_cache_dir()?;
        let (key, device) = rustllama_tuner::cache_context();
        persist_batch_size_winner_inline(&cache_dir, &key, &device, b as u32).ok()
    });

    Ok(TuneBatchSizeResponse {
        winner: report.winner.map(|b| b as u32),
        winner_tps: report.winner_tps,
        load_ms: report.load_ms,
        candidates: report
            .candidates
            .iter()
            .map(|c| TuneBatchSizeCandidate {
                batch_size: c.batch_size as u32,
                warmup_ms: c.warmup_ms,
                median_tps: c.median_tps,
                max_tps: c.max_tps,
                error: c.error.clone(),
            })
            .collect(),
        cache_path: cache_path.map(|p| p.display().to_string()),
    })
}

/// Resolve a tune endpoint's `model_path` / `model_name` body
/// pair to an absolute, existing path. Mirrors the same precedence
/// + cache-stem lookup as `/v1/models/load`'s name-resolution arm
/// so users get one consistent shape across endpoints.
///
/// Returns `(StatusCode, message)` on failure for the handler to
/// `.into_response()`.
fn resolve_tune_model_path(
    model_path: Option<&str>,
    model_name: Option<&str>,
) -> std::result::Result<std::path::PathBuf, (StatusCode, String)> {
    match (model_path, model_name) {
        (Some(p), None) => {
            let path = std::path::PathBuf::from(p);
            if !path.exists() {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!("model_path does not exist: {}", path.display()),
                ));
            }
            Ok(path)
        }
        (None, Some(name)) => {
            let Some(cache) = rustllama_hub::default_cache_dir() else {
                return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "no cache dir".to_string(),
                ));
            };
            let paths = rustllama_hub::list_cached(&cache).map_err(|e| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("failed to enumerate cache: {e}"),
                )
            })?;
            paths
                .into_iter()
                .find(|p| {
                    p.file_stem()
                        .and_then(|s| s.to_str())
                        .map(|s| s == name)
                        .unwrap_or(false)
                })
                .ok_or_else(|| {
                    (
                        StatusCode::NOT_FOUND,
                        format!(
                            "no cached GGUF with file stem `{name}` under {}",
                            cache.display()
                        ),
                    )
                })
        }
        (None, None) => Err((
            StatusCode::BAD_REQUEST,
            "specify one of `model_path` (absolute) or `model_name` (file stem)".to_string(),
        )),
        (Some(_), Some(_)) => Err((
            StatusCode::BAD_REQUEST,
            "specify exactly one of `model_path` or `model_name`, not both".to_string(),
        )),
    }
}

/// Inline placement persister — mirrors the helper in
/// rustllama-cli. Re-implemented here (rather than depending on
/// cli) to avoid the cycle.
fn persist_placement_winner_inline(
    cache_dir: &std::path::Path,
    key: &str,
    device: &rustllama_tuner::DeviceFingerprint,
    model_key: &str,
    n_gpu_layers: u32,
) -> std::io::Result<std::path::PathBuf> {
    use rustllama_tuner::{cache_path_for, load_cache, save_cache, PlacementPlan, TuningResult};
    std::fs::create_dir_all(cache_dir)?;
    let mut tuning = load_cache(cache_dir, key)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
        .unwrap_or_else(|| TuningResult::empty(key.to_string(), device.clone()));
    tuning.placement.insert(
        model_key.to_string(),
        PlacementPlan {
            n_gpu_layers,
            overrides: Vec::new(),
        },
    );
    tuning.last_tuned = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| format!("{}", d.as_secs()));
    save_cache(cache_dir, &tuning)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    Ok(cache_path_for(cache_dir, key))
}

fn persist_batch_size_winner_inline(
    cache_dir: &std::path::Path,
    key: &str,
    device: &rustllama_tuner::DeviceFingerprint,
    batch_size: u32,
) -> std::io::Result<std::path::PathBuf> {
    use rustllama_tuner::{cache_path_for, load_cache, save_cache, TuningResult};
    std::fs::create_dir_all(cache_dir)?;
    let mut tuning = load_cache(cache_dir, key)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
        .unwrap_or_else(|| TuningResult::empty(key.to_string(), device.clone()));
    tuning.batch_size = Some(batch_size);
    tuning.last_tuned = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| format!("{}", d.as_secs()));
    save_cache(cache_dir, &tuning)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
    Ok(cache_path_for(cache_dir, key))
}

/// Read GGUF metadata for placement-candidate generation. Inlined
/// to avoid a cli dep (which would cycle). Mirrors the cli's
/// `read_dims_and_quant_from_gguf` shape. Returns a flat error
/// string since rustllama-server doesn't pull `anyhow`.
fn read_dims_and_quant_from_gguf_for_tune(
    path: &std::path::Path,
) -> std::result::Result<
    (
        rustllama_tuner::placement::ModelDims,
        rustllama_tuner::placement::WeightQuant,
    ),
    String,
> {
    use rustllama_gguf::{GgmlType, Gguf, MetadataValue};
    use rustllama_tuner::placement::{ModelDims, WeightQuant};

    let gguf = Gguf::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let arch = gguf
        .architecture()
        .ok_or_else(|| "GGUF has no general.architecture metadata".to_string())?
        .to_string();
    let key = |k: &str| -> Option<u32> {
        let full = format!("{arch}.{k}");
        gguf.metadata_get(&full)
            .or_else(|| gguf.metadata_get(k))
            .and_then(MetadataValue::as_u32)
    };
    let n_layers = key("block_count")
        .ok_or_else(|| format!("GGUF missing `{arch}.block_count`"))?;
    let d_model = key("embedding_length")
        .ok_or_else(|| format!("GGUF missing `{arch}.embedding_length`"))?;
    let d_ff = key("feed_forward_length")
        .ok_or_else(|| format!("GGUF missing `{arch}.feed_forward_length`"))?;
    let n_heads = key("attention.head_count")
        .ok_or_else(|| format!("GGUF missing `{arch}.attention.head_count`"))?;
    let n_kv_heads = key("attention.head_count_kv").unwrap_or(n_heads);
    let head_dim = key("attention.key_length")
        .or_else(|| key("rope.dimension_count"))
        .unwrap_or(d_model / n_heads.max(1));
    let vocab_size = gguf
        .metadata_get("tokenizer.ggml.tokens")
        .and_then(|v| match v {
            MetadataValue::Array(a) => Some(a.len() as u32),
            _ => None,
        })
        .ok_or_else(|| "GGUF missing tokenizer.ggml.tokens".to_string())?;

    // MoE: zero when the GGUF doesn't carry expert metadata (dense
    // models). `expert_count` is the GGUF key Mixtral / Qwen3-MoE /
    // DeepSeek-V3 set; `expert_used_count` and `expert_shared_count`
    // describe top-K routing and DeepSeek-style always-active shared
    // experts respectively.
    let n_experts = key("expert_count").unwrap_or(0);
    let n_experts_used = if n_experts >= 2 { key("expert_used_count").unwrap_or(1) } else { 0 };
    let n_experts_shared =
        if n_experts >= 2 { key("expert_shared_count").unwrap_or(0) } else { 0 };

    let mut bytes_by: Vec<(GgmlType, u64)> = Vec::new();
    for t in gguf.tensors() {
        match bytes_by.iter_mut().find(|(d, _)| *d == t.dtype) {
            Some(entry) => entry.1 += t.byte_size,
            None => bytes_by.push((t.dtype, t.byte_size)),
        }
    }
    let dominant = bytes_by
        .iter()
        .max_by_key(|(_, b)| *b)
        .map(|(t, _)| *t)
        .unwrap_or(GgmlType::F16);
    let quant = match dominant {
        GgmlType::Q4_K => WeightQuant::Q4_K_M,
        GgmlType::Q5_K => WeightQuant::Q5_K_M,
        GgmlType::Q8_0 => WeightQuant::Q8_0,
        GgmlType::F16 | GgmlType::Bf16 => WeightQuant::F16,
        _ => WeightQuant::Q4_K_M,
    };
    Ok((
        ModelDims {
            n_layers,
            d_model,
            d_ff,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            n_experts,
            n_experts_used,
            n_experts_shared,
        },
        quant,
    ))
}

/// Look up the cached `n_gpu_layers` winner for this system (keyed by the
/// whole-system fingerprint) + the supplied model key. Returns `None` on
/// cache miss (fresh install, model never tuned) or an unresolvable cache
/// dir — both cases let the caller fall back to the config default unchanged.
fn tuner_cached_placement(model_key: &str) -> Option<u32> {
    let cache_dir = rustllama_tuner::default_cache_dir()?;
    let key = rustllama_tuner::system_fingerprint();
    let tuning = rustllama_tuner::load_cache(&cache_dir, &key).ok()??;
    tuning.placement.get(model_key).map(|p| p.n_gpu_layers)
}

/// Look up the cached `batch_size` winner for the active SYCL
/// device. Per-device (not per-model) because the optimal prefill
/// chunk size is a function of the device's compute / memory ratio,
/// not the specific model's shape.
fn tuner_cached_batch_size() -> Option<u32> {
    let cache_dir = rustllama_tuner::default_cache_dir()?;
    let key = rustllama_tuner::system_fingerprint();
    let tuning = rustllama_tuner::load_cache(&cache_dir, &key).ok()??;
    tuning.batch_size
}

/// Peek at a GGUF file's `<arch>.context_length` metadata without
/// loading the model. Used by `load_model` to pick a sensible
/// default `ctx_size` from what the model was actually trained for.
/// Returns `None` on file open / parse failures so the caller falls
/// back to the legacy 8192 default.
fn peek_gguf_ctx_train(path: &std::path::Path) -> Option<u32> {
    use rustllama_gguf::{Gguf, MetadataValue};
    let gguf = Gguf::open(path).ok()?;
    let arch = gguf.architecture()?.to_string();
    // Try `<arch>.context_length` first (canonical), fall back to
    // the bare `context_length` for older converters that didn't
    // namespace metadata keys.
    let full = format!("{arch}.context_length");
    gguf.metadata_get(&full)
        .or_else(|| gguf.metadata_get("context_length"))
        .and_then(MetadataValue::as_u32)
}

fn inspect_gguf_blocking(
    path: &std::path::Path,
    include_tensors: bool,
) -> std::result::Result<InspectGgufResponse, (StatusCode, String)> {
    use rustllama_gguf::{Gguf, MetadataValue};
    let gguf = Gguf::open(path).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            format!("open {}: {e}", path.display()),
        )
    })?;

    let arch = gguf.architecture().unwrap_or("?").to_string();
    let name = gguf
        .metadata_get("general.name")
        .and_then(MetadataValue::as_string)
        .unwrap_or("(unnamed)")
        .to_string();
    let size_label = gguf
        .metadata_get("general.size_label")
        .and_then(MetadataValue::as_string)
        .map(str::to_string);
    let file_type = gguf
        .metadata_get("general.file_type")
        .and_then(MetadataValue::as_u32);

    let key = |k: &str| -> Option<u32> {
        let full = format!("{arch}.{k}");
        gguf.metadata_get(&full)
            .or_else(|| gguf.metadata_get(k))
            .and_then(MetadataValue::as_u32)
    };
    let context_length = key("context_length");
    let block_count = key("block_count");
    let embedding_length = key("embedding_length");
    let head_count = key("attention.head_count");
    let head_count_kv = key("attention.head_count_kv");
    let head_dim = key("attention.key_length").or_else(|| key("rope.dimension_count"));
    let vocab_size = gguf
        .metadata_get("tokenizer.ggml.tokens")
        .and_then(|v| match v {
            MetadataValue::Array(a) => Some(a.len() as u32),
            _ => None,
        });

    // MoE expert counts. Absent on dense GGUFs (Option::None →
    // skip_serializing_if drops the field from the JSON output).
    // When `expert_count` is present, surface used + shared too —
    // shared defaults to 0 since most MoE families don't use the
    // DeepSeek-V3 shared-expert design.
    let n_experts = key("expert_count").filter(|&n| n >= 2);
    let n_experts_used = n_experts.and_then(|_| key("expert_used_count").or(Some(1)));
    let n_experts_shared = n_experts.and_then(|_| key("expert_shared_count").or(Some(0)));

    let tensors = gguf.tensors();
    let tensor_count = tensors.len() as u64;
    let total_tensor_bytes: u64 = tensors.iter().map(|t| t.byte_size).sum();
    let total_params: u64 = tensors.iter().map(|t| t.element_count()).sum();
    let file_bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    // Dtype histogram, alphabetized so the GUI doesn't have to sort.
    let mut buckets: std::collections::BTreeMap<&'static str, (u64, u64)> =
        std::collections::BTreeMap::new();
    for t in tensors {
        let n = t.dtype.as_str();
        let e = buckets.entry(n).or_insert((0, 0));
        e.0 += 1;
        e.1 += t.byte_size;
    }
    let dtypes: Vec<InspectDtypeBucket> = buckets
        .into_iter()
        .map(|(dtype, (tensor_count, bytes))| InspectDtypeBucket {
            dtype,
            tensor_count,
            bytes,
        })
        .collect();

    let tensors_out = if include_tensors {
        Some(
            tensors
                .iter()
                .map(|t| InspectTensorEntry {
                    name: t.name.clone(),
                    dtype: t.dtype.as_str(),
                    shape: t.dims.clone(),
                    elements: t.element_count(),
                    bytes: t.byte_size,
                })
                .collect(),
        )
    } else {
        None
    };

    let model_card = rustllama_hub::model_card::load_for_gguf(path);

    Ok(InspectGgufResponse {
        path: path.display().to_string(),
        architecture: arch,
        name,
        size_label,
        file_type,
        context_length,
        block_count,
        embedding_length,
        head_count,
        head_count_kv,
        head_dim,
        vocab_size,
        n_experts,
        n_experts_used,
        n_experts_shared,
        tensor_count,
        total_params,
        total_tensor_bytes,
        file_bytes,
        dtypes,
        tensors: tensors_out,
        model_card,
    })
}

#[derive(serde::Deserialize)]
struct UnloadRequest {
    // The GUI (and older clients) post `{"model_id": "..."}`; the
    // OpenAI-shaped canonical field is `model`. Accept both so an
    // Eject never 422s on a field-name mismatch.
    #[serde(alias = "model_id")]
    model: String,
}

#[derive(Serialize)]
struct UnloadResponse {
    unloaded: String,
}

async fn unload_model(
    State(state): State<AppState>,
    Json(req): Json<UnloadRequest>,
) -> Response {
    match state.unload(&req.model).await {
        Ok(Some(removed)) => {
            // The model is already out of the registry; freeing the engine
            // (unmap ~5.6 GB of weights, release ~70 VirtualLock regions,
            // free ~1 GB of USM KV cache) takes several seconds and used to
            // run inline here — stalling the HTTP 200 (and the GUI's
            // "Ejecting…" spinner) for the whole teardown. Drop it on the
            // blocking pool so Eject returns in milliseconds. Only the last
            // Arc holder pays the teardown; in-flight requests keep their
            // clone alive until they finish, then drop it off this path too.
            tokio::task::spawn_blocking(move || drop(removed));
            Json(UnloadResponse {
                unloaded: req.model,
            })
            .into_response()
        }
        Ok(None) => (
            StatusCode::NOT_FOUND,
            format!("model not loaded: {}", req.model),
        )
            .into_response(),
        Err(e) => (StatusCode::BAD_REQUEST, e).into_response(),
    }
}

#[derive(serde::Deserialize)]
struct SetDefaultRequest {
    // Same client contract as UnloadRequest: accept `model_id` (GUI)
    // and `model` (canonical) so Set-default doesn't silently 422.
    #[serde(alias = "model_id")]
    model: String,
}

async fn set_default_model(
    State(state): State<AppState>,
    Json(req): Json<SetDefaultRequest>,
) -> Response {
    match state.set_default(&req.model).await {
        Ok(()) => {
            // Persist to config.toml so the default survives restart
            // (best-effort; the live default is already set).
            let persisted = state.persist_default_model(&req.model).await;
            Json(serde_json::json!({"default": req.model, "persisted": persisted}))
                .into_response()
        }
        Err(e) => (StatusCode::NOT_FOUND, e).into_response(),
    }
}

#[derive(serde::Deserialize)]
struct CancelRequest {
    /// The `id` field of an in-flight chat/completions response — i.e.
    /// the `chatcmpl-…` or `cmpl-…` value the server returned in the
    /// initial chunk. Cancellation only applies to streaming requests
    /// (non-stream completes too quickly to be useful).
    id: String,
}

async fn cancel_request(
    State(state): State<AppState>,
    Json(req): Json<CancelRequest>,
) -> Response {
    if state.fire_cancel(&req.id) {
        Json(serde_json::json!({ "cancelled": req.id })).into_response()
    } else {
        (
            StatusCode::NOT_FOUND,
            format!("no in-flight request with id `{}`", req.id),
        )
            .into_response()
    }
}

#[derive(Serialize)]
struct Health {
    status: &'static str,
    model_id: String,
    version: String,
    uptime_s: u64,
    /// IPv4 URLs that should be reachable from other hosts on the LAN.
    /// Populated when the server is bound to `0.0.0.0` (`serve --lan`);
    /// empty otherwise. Useful for editor configs and the future GUI's
    /// "share this server" button.
    lan_urls: Vec<String>,
    /// True after SIGINT/SIGTERM/Ctrl-C — the server is draining
    /// in-flight requests and rejecting new ones with 503. Monitoring
    /// tools can poll `/healthz` and see `draining: true` to know not
    /// to send fresh work.
    draining: bool,
    /// True when the process is running in portable mode (paths root
    /// under the exe dir, not %APPDATA%). Lets clients tell whether a
    /// model cache lives next to the binary or in the OS config dir.
    portable: bool,
}

async fn healthz(State(state): State<AppState>) -> Json<Health> {
    // Empty registry is OK: server started without a model
    // configured. `/healthz` still reports "ok" so the GUI knows the
    // backend is up; `model_id` is the empty string and the Models
    // page is where the user populates the registry from.
    let model_id = state
        .try_current()
        .await
        .map(|s| s.model_id)
        .unwrap_or_default();
    Json(Health {
        status: if state.is_shutting_down() { "draining" } else { "ok" },
        model_id,
        version: state.version.clone(),
        uptime_s: 0,
        lan_urls: enumerate_lan_urls(&state),
        draining: state.is_shutting_down(),
        portable: rustllama_runtime::paths().portable,
    })
}

/// Live snapshot of per-model serving metrics. Returned by
/// `GET /v1/metrics`. Polled by the GUI Status page (~1 Hz) to render
/// tok/s + KV/cache sparklines without each chart needing its own
/// streaming endpoint. The fields are all numbers so the page can
/// scope-update a fixed-shape store without parsing nested objects.
#[derive(Serialize)]
struct MetricsSnapshot {
    model_id: String,
    /// Engine's reported context window (`max_ctx`).
    ctx_size: u32,
    /// Current KV `seq_len` for this serving model — how much of the
    /// context window is occupied by the previous request's prompt +
    /// generation. Drives the "ctx used" bar in the GUI.
    ctx_used: u32,
    /// Outstanding requests against this model: active + waiting in
    /// the scheduler. Matches `pending` used by the backpressure
    /// decision so the GUI can warn before requests get 503'd.
    pending: u32,
    /// Hard cap on `pending` — beyond this `try_acquire` returns 503.
    /// Surfaced so the GUI can render "3 / 4 in flight".
    max_pending: u32,
    /// Most recent generation's tokens / second (computed from
    /// `decode_ms` + `tokens_generated`). `None` when no decode has
    /// run yet, or when `decode_ms == 0` (which happens for cancelled
    /// requests with 0 generated tokens).
    last_tok_s: Option<f64>,
    /// Exponential moving average (alpha = 0.3) of recent decode
    /// tok/s samples. `None` until the first non-trivial generation
    /// completes. Smoothed over ~4 requests, so a cold-start outlier
    /// doesn't dominate the displayed throughput. Prefer this over
    /// `last_tok_s` for any "this is how fast my model is running"
    /// UI surface — the per-request number swings 5× between cold
    /// and warm and is not a useful "is my GPU happy" signal.
    ema_tok_s: Option<f64>,
    /// Per-request stats from the last drained generation. Set to
    /// `None` when no request has yet completed since startup.
    last_prefill_ms: Option<f64>,
    last_decode_ms: Option<f64>,
    last_tokens_prefilled: Option<u32>,
    last_tokens_generated: Option<u32>,
    last_cache_hit_tokens: Option<u32>,
    /// Per-process uptime in seconds — handy for "this server has
    /// been up for 47 minutes" UI affordances.
    uptime_s: u64,
    /// Per-model concurrency. `1` for single-flight, `>1` when the
    /// operator set `[server].concurrency > 1` and the model is a
    /// real CpuEngine (mocks stay single-flight regardless).
    concurrency: u32,
    /// KV-cache dtype as a config string: `"f32"`, `"q8_0"`, `"tq1"`,
    /// `"tq2"`, `"tq4"`, `"tq8"`, `"q4_0"`, or `"nvfp4"`. `null` for mock
    /// engines (which don't carry a KV dtype). Lets the GUI surface
    /// "currently serving with tq4 KV" without parsing the model
    /// config separately.
    kv_dtype: Option<String>,
    /// Host total physical RAM in bytes. Sampled live on each call
    /// via sysinfo. The GUI uses this + `ram_available_bytes` to
    /// surface a "you have X GB free" indicator on the Models page
    /// and to warn before pulling models that won't fit.
    ram_total_bytes: u64,
    /// Host available (currently-free) RAM in bytes.
    ram_available_bytes: u64,
    /// OS commit limit in bytes — physical RAM + page-file (Windows)
    /// or RAM + swap (Linux). The load pre-flight allows models up
    /// to `commit_available_bytes`, not just physical-available, so
    /// a generous page file lets you load models bigger than RAM
    /// (cold pages spill to disk; GPU-resident weights stay hot).
    #[serde(default)]
    commit_total_bytes: u64,
    /// OS commit-available in bytes (commit total minus what's
    /// already reserved by other processes). This is what the
    /// load pre-flight actually compares against.
    #[serde(default)]
    commit_available_bytes: u64,
    /// Number of SYCL devices visible to the engine. Reports 0 when
    /// no SYCL device is present at runtime (no oneAPI runtime on
    /// the host, or no Intel GPU); reports the real count from
    /// `sycl::device::get_devices(gpu)` when a SYCL runtime is
    /// installed and enumerates a device. Surfaced on the
    /// Settings page so users can confirm their Intel GPU is visible.
    sycl_device_count: u32,
    /// Paged-KV pool total page count. `0` on engines using the
    /// contiguous KV layout (the default) — the GUI Status page
    /// hides its "Paged KV pool" panel in that case. Non-zero only
    /// when `[inference].kv_cache_layout = "paged"` (CpuEngine
    /// paged backend) or when the engine is a `PagedBatchEngine`
    /// (the fused-decode CB path).
    #[serde(default)]
    paged_total_pages: u32,
    /// Paged-KV pool free pages. `paged_total_pages - paged_free_pages`
    /// is the live in-use count. Drops as slots prefill and grows
    /// back on slot completion.
    #[serde(default)]
    paged_free_pages: u32,
    /// Active slots in a fused-decode engine. `0` on `CpuEngine`
    /// (single-flight per engine; concurrency via forks); usable
    /// only on `PagedBatchEngine`. V1 placeholder always reports
    /// `0` — driver-thread shared-counter wire-up is a follow-up.
    #[serde(default)]
    paged_active_slots: u32,
    /// Cumulative-since-startup counters from the engine. `None`
    /// for engines that don't track aggregate stats (e.g. mock).
    /// Surfaces `prefix_cache_hit_rate` so operators can see at a
    /// glance whether the prefix cache is paying off — closer to
    /// 1.0 = the cache is doing more work.
    #[serde(skip_serializing_if = "Option::is_none")]
    cumulative: Option<rustllama_engine::CumulativeStatsSnapshot>,
    /// Process resident memory (working-set on Windows, RSS on
    /// Linux). Sampled live via `sysinfo`. The user's
    /// "is rustllama still bloated after model unload?" question
    /// gets a direct answer from this number.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    process_rss_bytes: Option<u64>,
    /// System-wide CPU utilization % (0..=100) for the GUI's CPU-over-RAM
    /// paired status-bar row. Delta since the previous metrics poll.
    cpu_utilization_pct: f32,
    /// CPU brand string (e.g. "Intel Core i7-…") for the CPU-row tooltip.
    #[serde(default)]
    cpu_brand: String,
    /// Page/Swap (system paging) disk I/O, bytes/sec, split read/write.
    /// Read = hard-fault page-ins from disk; write = page-outs to the
    /// pagefile/swap. `None` off Windows/Linux.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    page_io_read_bytes_per_sec: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    page_io_write_bytes_per_sec: Option<u64>,
    /// Model I/O — this (server) process's disk I/O, bytes/sec, read/write.
    /// Dominated by GGUF weight reads on load / cold faults. `None` off
    /// Windows/Linux.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_io_read_bytes_per_sec: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    model_io_write_bytes_per_sec: Option<u64>,
    /// Intel GPU Sysman snapshot. `None` on non-Intel hosts,
    /// mock-mode builds, or when the driver doesn't expose Sysman.
    /// See [`GpuSysmanSnapshot`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    gpu_sysman: Option<GpuSysmanSnapshot>,
    /// Per-physical-GPU VRAM (one entry per GPU detected). The GUI
    /// renders a VRAM usage bar per entry. Empty when no GPU is visible.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    gpus: Vec<GpuMetric>,
}

/// Per-physical-GPU VRAM snapshot for the GUI status bar. Intel GPUs
/// come from SYCL `device_info` (deduped by name — the same physical GPU
/// enumerates once per SYCL backend) with VRAM-free from the L0 Sysman
/// probe; NVIDIA GPUs come from the runtime driver probe (`gpu_detect`):
/// total from enumeration, free from `cuMemGetInfo`
/// (`nvidia_free_vram_all`, dlopen'd CUDA driver — no toolkit).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GpuMetric {
    /// Stable RustLlama enumeration index (Intel/SYCL devices first, then
    /// NVIDIA), fixed regardless of which GPUs are disabled — so ignoring
    /// GPU 0 leaves GPU 1 still labeled "GPU 1". This is the index the
    /// `[inference].disabled_gpus` / `RUSTLLAMA_DISABLED_GPUS` list refers
    /// to. The GUI labels the row by this, not by array position.
    pub index: u32,
    /// "intel" | "nvidia" | "amd" | "gpu".
    pub vendor: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_total_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_free_bytes: Option<u64>,
    /// GPU engine (compute/render) utilization %, 0..=100. `None` when
    /// the backend doesn't expose it yet (Intel engine-activity + NVIDIA
    /// NVML land in a follow-up); the GUI shows "n/a" for the util bar.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub utilization_pct: Option<f32>,
}

/// The host CPU as a first-class compute-tier inventory entry, symmetric
/// with [`GpuMetric`]. CPU is the fallback backend when no usable GPU is
/// present, and placement sizes CPU-resident weights against its RAM — so
/// the inventory surfaces its capacity (logical cores + total RAM) and
/// kernel capability (SIMD), not just the GPUs.
#[derive(Debug, Clone, serde::Serialize)]
pub struct CpuMetric {
    /// CPU brand string, e.g. "Intel Core i9-13900H".
    pub brand: String,
    /// Logical CPU count (`available_parallelism`) — the compute width.
    pub logical_cores: u32,
    /// Host SIMD features the kernels dispatch on (`avx2`/`avx512f`/`fma`/
    /// `f16c`). Empty on non-x86 (kernels run scalar / NEON).
    pub simd_features: Vec<&'static str>,
    /// Host total physical RAM in bytes — the CPU tier's residency budget.
    pub ram_total_bytes: u64,
    /// System-wide CPU utilization %, 0..=100 (delta-sampled).
    pub utilization_pct: f32,
    /// Logical processors actually usable for compute = `logical_cores`
    /// minus [`Self::disabled_cpus`]. This is the CPU tier's effective
    /// compute width (the rayon pool is sized to it + pinned onto it).
    pub enabled_cores: u32,
    /// Logical-processor indices excluded from the CPU pool via
    /// `[inference].disabled_cpus` / `RUSTLLAMA_DISABLED_CPUS` (the CPU
    /// analog of the GPU disable-list). Empty = use every core.
    pub disabled_cpus: Vec<u32>,
    /// Whether the CPU is an enabled compute tier (`[inference].cpu_enabled`
    /// / `RUSTLLAMA_CPU_ENABLED`). `false` = GPU-only placement.
    pub cpu_enabled: bool,
    /// Whether the GPU is an enabled compute tier (`[inference].gpu_enabled`
    /// / `RUSTLLAMA_GPU_ENABLED`). `false` = CPU-only placement.
    pub gpu_enabled: bool,
    /// Whether VRAM-only weight residency was requested
    /// (`[inference].vram_only` / `RUSTLLAMA_VRAM_ONLY`).
    pub vram_only: bool,
}

/// GPU sensor snapshot via Level Zero Sysman. Each field is `Option`
/// because availability depends on driver + GPU class (integrated
/// graphics expose VRAM but typically not temp/power).
#[derive(Debug, Clone, Copy, Default, serde::Serialize, serde::Deserialize)]
pub struct GpuSysmanSnapshot {
    /// Total VRAM (or shared-memory equivalent on Iris Xe) in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_total_bytes: Option<u64>,
    /// Currently-free VRAM in bytes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vram_free_bytes: Option<u64>,
    /// Max temperature across all sensors (°C). `None` on Iris Xe
    /// integrated graphics — Sysman temp probes are usually only
    /// exposed on discrete Arc parts.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_temp_c: Option<f64>,
    /// Cumulative energy counter (µJ). Caller derives instantaneous
    /// power as `(e2 - e1) / (t2 - t1) µW` from two samples.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub energy_uj: Option<u64>,
    /// Energy-counter sample timestamp (µs from driver epoch).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub energy_timestamp_us: Option<u64>,
    /// GPU clock (MHz) on the first frequency domain. Iris Xe idles
    /// around 300 MHz; Arc parts run higher and expose additional
    /// domains (memory/media) that v1 doesn't surface.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_freq_mhz: Option<f64>,
}

/// Process-start instant, captured at the time the metrics module is
/// first touched. `OnceLock` not `LazyLock` to keep the MSRV bar at
/// 1.83 (LazyLock is 1.80, but OnceLock matches the rest of this file).
static SERVER_START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

async fn metrics(State(state): State<AppState>) -> Json<MetricsSnapshot> {
    let start = *SERVER_START.get_or_init(std::time::Instant::now);
    // Empty registry → most fields are zero/None, but uptime + the
    // shape stay valid so the GUI's polling doesn't error-loop.
    let serving = match state.try_current().await {
        Some(s) => s,
        None => {
            let mem = rustllama_runtime::memory_info();
            let cpu = rustllama_runtime::cpu_info();
            let page_io = rustllama_runtime::paging_io_bytes_per_sec();
            let model_io = rustllama_runtime::process_io_bytes_per_sec();
            return Json(MetricsSnapshot {
                model_id: String::new(),
                ctx_size: 0,
                ctx_used: 0,
                pending: 0,
                max_pending: DEFAULT_MAX_PENDING_PER_MODEL as u32,
                last_tok_s: None,
                ema_tok_s: None,
                last_prefill_ms: None,
                last_decode_ms: None,
                last_tokens_prefilled: None,
                last_tokens_generated: None,
                last_cache_hit_tokens: None,
                uptime_s: start.elapsed().as_secs(),
                concurrency: 1,
                kv_dtype: None,
                ram_total_bytes: mem.total_bytes,
                ram_available_bytes: mem.available_bytes,
                commit_total_bytes: mem.commit_total_bytes,
                commit_available_bytes: mem.commit_available_bytes,
                sycl_device_count: sycl_device_count(),
                paged_total_pages: 0,
                paged_free_pages: 0,
                paged_active_slots: 0,
                cumulative: None,
                process_rss_bytes: rustllama_runtime::process_rss_bytes(),
                cpu_utilization_pct: cpu.utilization_pct,
                cpu_brand: cpu.brand,
                page_io_read_bytes_per_sec: page_io.map(|(r, _)| r),
                page_io_write_bytes_per_sec: page_io.map(|(_, w)| w),
                // Model I/O read = system hard-fault (page-in) rate — the
                // memory-mapped GGUF's disk reads register as page faults,
                // not as process ReadFile bytes. Write = the process's own
                // file writes (KV/conversation DB; the model is read-only).
                model_io_read_bytes_per_sec: page_io.map(|(r, _)| r),
                model_io_write_bytes_per_sec: model_io.map(|(_, w)| w),
                gpu_sysman: gpu_sysman_probe(),
                gpus: gpus_probe(),
            });
        }
    };
    let engine_metrics = serving.engine.metrics();
    let pending = serving.pending.load(Ordering::Acquire) as u32;
    // Per-request stats from the trait method — same shape on
    // every backend: `CpuEngine` returns its inherent value;
    // `PagedBatchEngine` reads its driver-thread mutex; mock
    // engines return `None`. The contiguous-vs-paged distinction
    // is invisible at this layer.
    let (last_prefill_ms, last_decode_ms, last_prefilled, last_generated, last_cache_hit) =
        match serving.engine.last_request_stats_snapshot() {
            Some(s) => (
                Some(s.prefill_ms),
                Some(s.decode_ms),
                Some(s.tokens_prefilled),
                Some(s.tokens_generated),
                Some(s.cache_hit_tokens),
            ),
            None => (None, None, None, None, None),
        };
    let last_tok_s = match (last_decode_ms, last_generated) {
        (Some(ms), Some(n)) if ms > 0.0 && n > 0 => Some(n as f64 / (ms / 1000.0)),
        _ => None,
    };
    let ema_tok_s = serving.cpu_engine.as_ref().and_then(|cpu| cpu.ema_tok_s());
    // Pull the live KV dtype from the engine's cache. Pretty-printed
    // so the GUI doesn't have to know about the `Tq(N)` debug form.
    let kv_dtype: Option<String> = serving.cpu_engine.as_ref().map(|cpu| {
        match cpu.kv_dtype() {
            rustllama_engine::KvDtype::F32 => "f32".to_string(),
            rustllama_engine::KvDtype::Q8_0 => "q8_0".to_string(),
            rustllama_engine::KvDtype::Tq(bits) => format!("tq{bits}"),
            rustllama_engine::KvDtype::Nvfp4 => "nvfp4".to_string(),
            rustllama_engine::KvDtype::Q4_0 => "q4_0".to_string(),
        }
    });
    let mem = rustllama_runtime::memory_info();
    let cpu = rustllama_runtime::cpu_info();
    let page_io = rustllama_runtime::paging_io_bytes_per_sec();
    let model_io = rustllama_runtime::process_io_bytes_per_sec();
    Json(MetricsSnapshot {
        model_id: serving.model_id.clone(),
        ctx_size: serving.engine.n_ctx(),
        ctx_used: engine_metrics.context_used,
        pending,
        max_pending: serving.max_pending as u32,
        last_tok_s,
        ema_tok_s,
        last_prefill_ms,
        last_decode_ms,
        last_tokens_prefilled: last_prefilled,
        last_tokens_generated: last_generated,
        last_cache_hit_tokens: last_cache_hit,
        uptime_s: start.elapsed().as_secs(),
        concurrency: serving.concurrency() as u32,
        kv_dtype,
        ram_total_bytes: mem.total_bytes,
        ram_available_bytes: mem.available_bytes,
        commit_total_bytes: mem.commit_total_bytes,
        commit_available_bytes: mem.commit_available_bytes,
        sycl_device_count: sycl_device_count(),
        paged_total_pages: engine_metrics.paged_total_pages,
        paged_free_pages: engine_metrics.paged_free_pages,
        paged_active_slots: engine_metrics.paged_active_slots,
        // Cumulative-since-startup counters. `None` for engines
        // that don't track them (mock); real engines provide a
        // snapshot read with `prefix_cache_hit_rate` derived for
        // the operator.
        cumulative: serving.engine.cumulative_stats_snapshot(),
        process_rss_bytes: rustllama_runtime::process_rss_bytes(),
        cpu_utilization_pct: cpu.utilization_pct,
        cpu_brand: cpu.brand,
        page_io_read_bytes_per_sec: page_io.map(|(r, _)| r),
        page_io_write_bytes_per_sec: page_io.map(|(_, w)| w),
        // Model I/O read = system page-in rate (the mmap'd GGUF's disk reads
        // are page faults, not process ReadFile bytes); write = process writes.
        model_io_read_bytes_per_sec: page_io.map(|(r, _)| r),
        model_io_write_bytes_per_sec: model_io.map(|(_, w)| w),
        gpu_sysman: gpu_sysman_probe(),
        gpus: gpus_probe(),
    })
}

/// SYCL device count snapshot. Always callable — returns 0 when
/// no SYCL device is present at runtime; otherwise it
/// returns `sycl::device::get_devices(gpu).len()` via the engine's
/// kernel-sycl crate. Wrapped to keep the call-site noise low.
fn sycl_device_count() -> u32 {
    rustllama_kernels_sycl::device_count().unwrap_or(0)
}

/// Probe Intel GPU sensors via Level Zero Sysman. Returns `None` on
/// non-Intel hosts, when the loader DLL is absent, when `zesInit`
/// returns UNINITIALIZED (driver doesn't expose Sysman), or when no
/// devices enumerate. All failure modes degrade gracefully — the
/// GUI Status page just hides the GPU sensor row in those cases.
///
/// On Iris Xe / Arc with a recent Intel driver this returns VRAM
/// total + free at minimum. Temperature is typically only exposed
/// on discrete Arc; power varies by driver version.
///
/// Best-effort cached behind a per-process `OnceLock` inside
/// `rustllama-l0-sys` — the loader runs once; subsequent calls just
/// re-probe the live counters.
fn gpu_sysman_probe() -> Option<GpuSysmanSnapshot> {
    use rustllama_l0_sys::{LevelZero, Sysman};
    // Sysman load is gated on the env var being set before zeInit.
    // Set it here defensively — if the user launched the server
    // without ZES_ENABLE_SYSMAN=1, zesInit returns UNINITIALIZED
    // and this probe gives up cleanly.
    let sysman = Sysman::load().ok()?;
    let l0 = LevelZero::load().ok()?;
    let drivers = l0.drivers().ok()?;
    if drivers.is_empty() {
        return None;
    }
    let devices = l0.devices(drivers[0]).ok()?;
    if devices.is_empty() {
        return None;
    }
    let reading = sysman.probe(devices[0]);
    // Only return Some when at least one field was populated —
    // otherwise the wire shape carries an empty snapshot that
    // clients have to special-case.
    if reading.vram_total_bytes.is_none()
        && reading.vram_free_bytes.is_none()
        && reading.max_temp_c.is_none()
        && reading.energy_counter.is_none()
        && reading.gpu_freq_mhz.is_none()
    {
        return None;
    }
    Some(GpuSysmanSnapshot {
        vram_total_bytes: reading.vram_total_bytes,
        vram_free_bytes: reading.vram_free_bytes,
        max_temp_c: reading.max_temp_c,
        energy_uj: reading.energy_counter.map(|c| c.energy_uj),
        energy_timestamp_us: reading.energy_counter.map(|c| c.timestamp_us),
        gpu_freq_mhz: reading.gpu_freq_mhz,
    })
}

/// Host SIMD features the CPU kernels dispatch on. Subset of
/// `["avx", "avx2", "fma", "f16c", "avx512f"]`; empty on non-x86 targets
/// (the kernels still run, just scalar / NEON). Shared by `cpus_probe` +
/// the `/v1/capabilities` CPU backend so both report the same set.
fn host_simd_features() -> Vec<&'static str> {
    #[allow(unused_mut)]
    let mut v: Vec<&'static str> = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx") { v.push("avx"); }
        if std::arch::is_x86_feature_detected!("avx2") { v.push("avx2"); }
        if std::arch::is_x86_feature_detected!("fma") { v.push("fma"); }
        if std::arch::is_x86_feature_detected!("f16c") { v.push("f16c"); }
        if std::arch::is_x86_feature_detected!("avx512f") { v.push("avx512f"); }
    }
    v
}

/// The host CPU as a first-class compute-tier inventory entry, symmetric
/// with [`gpus_probe`]. CPU is the fallback backend; placement sizes its
/// weight residency against RAM. Reuses the runtime's CPU/mem probes.
fn cpus_probe() -> CpuMetric {
    let cpu = rustllama_runtime::cpu_info();
    let mem = rustllama_runtime::memory_info();
    let logical_cores = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(0);
    let (enabled_cores, disabled_cpus) = cpu_enabled_cores(logical_cores);
    CpuMetric {
        brand: cpu.brand,
        logical_cores,
        simd_features: host_simd_features(),
        ram_total_bytes: mem.total_bytes,
        utilization_pct: cpu.utilization_pct,
        enabled_cores,
        disabled_cpus,
        cpu_enabled: rustllama_runtime::cpu_tier_enabled(),
        gpu_enabled: rustllama_runtime::gpu_tier_enabled(),
        vram_only: rustllama_runtime::vram_only_requested(),
    }
}

/// Shared CPU-tier inventory helper: the count of logical processors left
/// enabled after `disabled_cpus`, plus the sorted disabled list (both the
/// config-promoted env `RUSTLLAMA_DISABLED_CPUS` and any direct env). Only
/// indices within `[0, logical_cores)` count against the enabled total.
fn cpu_enabled_cores(logical_cores: u32) -> (u32, Vec<u32>) {
    let disabled = rustllama_runtime::disabled_cpu_indices();
    let mut list: Vec<u32> = disabled.iter().copied().collect();
    list.sort_unstable();
    let in_range = disabled.iter().filter(|&&i| i < logical_cores).count() as u32;
    (logical_cores.saturating_sub(in_range), list)
}

/// Per-physical-GPU VRAM for the GUI status bar. See [`GpuMetric`].
///
/// Enumeration order is stable: Intel/SYCL devices first (in SYCL
/// enumeration order, deduped by name), then NVIDIA — so on a system
/// whose GPUs don't change, "GPU 0" is the same card every run. Each GPU
/// gets a fixed stable `index`; GPUs in the disable-list
/// (`RUSTLLAMA_DISABLED_GPUS` / `[inference].disabled_gpus`) are omitted
/// from the list but the survivors keep their original indices.
fn gpus_probe() -> Vec<GpuMetric> {
    let disabled = rustllama_runtime::disabled_gpu_indices();
    let mut out = Vec::new();
    let mut idx: u32 = 0; // unified stable index (also the phys index for Intel)
    let free_by_phys = sysman_vram_free_per_device();
    let busy_by_phys = sysman_busy_pct_per_device();
    if let Ok(n) = rustllama_kernels_sycl::device_count() {
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for i in 0..n {
            if let Ok(info) = rustllama_kernels_sycl::device_info(i) {
                if seen.insert(info.name.clone()) {
                    let this = idx;
                    idx += 1;
                    if disabled.contains(&this) {
                        continue;
                    }
                    let vendor = match info.vendor_id {
                        0x8086 => "intel",
                        0x10de => "nvidia",
                        0x1002 => "amd",
                        _ => "gpu",
                    };
                    out.push(GpuMetric {
                        index: this,
                        vendor: vendor.to_string(),
                        name: info.name,
                        vram_total_bytes: Some(info.vram_bytes),
                        // `this` doubles as the physical index for the
                        // positional Sysman VRAM-free pairing.
                        vram_free_bytes: free_by_phys.get(this as usize).copied().flatten(),
                        // GPU engine-activity util via L0 Sysman (delta-
                        // sampled). `None` if the iGPU exposes no engine
                        // groups or on the first poll (no prior sample).
                        utilization_pct: busy_by_phys.get(this as usize).copied().flatten(),
                    });
                }
            }
        }
    }
    // NVIDIA GPUs via the runtime driver probe (dlopen). Total VRAM comes
    // from the enumeration probe; free VRAM from `cuMemGetInfo` in one
    // driver session (dlopen + cuInit once), keyed by CUDA-driver device
    // index (`NvidiaGpu::index`). Empty map on a non-NVIDIA host.
    if let Some(nv) = rustllama_runtime::gpu_detect::detect_nvidia() {
        let free_by_driver_idx: std::collections::HashMap<u32, u64> =
            rustllama_runtime::gpu_detect::nvidia_free_vram_all()
                .into_iter()
                .collect();
        for g in nv.gpus {
            let this = idx;
            idx += 1;
            if disabled.contains(&this) {
                continue;
            }
            out.push(GpuMetric {
                index: this,
                vendor: "nvidia".to_string(),
                // Map free VRAM back by the GPU's CUDA-driver index, not by
                // the unified stable index (`this`), since the free probe
                // enumerates in driver order.
                vram_free_bytes: free_by_driver_idx.get(&g.index).copied(),
                vram_total_bytes: Some(g.total_mem_bytes),
                name: g.name,
                // NVIDIA NVML util lands in A3b.
                utilization_pct: None,
            });
        }
    }
    // Windows PDH fallback for GPU utilization when the driver's engine-
    // activity probe (L0 Sysman) exposed none — the integrated Iris Xe
    // has no Sysman engine groups but its GPU-Engine perf counters work.
    // PDH reports a system-wide GPU-engine total, so only apply it when
    // exactly one GPU is present (unambiguous attribution).
    if out.len() == 1 && out[0].utilization_pct.is_none() {
        if let Some(pct) = rustllama_runtime::gpu_busy_pct() {
            out[0].utilization_pct = Some(pct);
        }
    }
    // Windows PDH fallback for VRAM-in-use when L0 Sysman exposed no memory
    // module (the integrated Iris Xe: `vram_free_bytes` stays None, so the
    // GUI can only draw the total). PDH's "GPU Adapter Memory" gives the
    // system-wide used bytes; derive free = total - used. Same single-GPU
    // gate as the utilization fallback (unambiguous attribution).
    if out.len() == 1 && out[0].vram_free_bytes.is_none() {
        if let (Some(total), Some(used)) =
            (out[0].vram_total_bytes, rustllama_runtime::gpu_mem_used_bytes())
        {
            out[0].vram_free_bytes = Some(total.saturating_sub(used));
        }
    }
    out
}

/// L0 Sysman VRAM-free (bytes) per physical device, in enumeration
/// order. Empty when Sysman / Level Zero isn't available.
fn sysman_vram_free_per_device() -> Vec<Option<u64>> {
    use rustllama_l0_sys::{LevelZero, Sysman};
    let mut out = Vec::new();
    let (Ok(sysman), Ok(l0)) = (Sysman::load(), LevelZero::load()) else {
        return out;
    };
    let Ok(drivers) = l0.drivers() else {
        return out;
    };
    for driver in drivers {
        if let Ok(devices) = l0.devices(driver) {
            for dev in devices {
                out.push(sysman.probe(dev).vram_free_bytes);
            }
        }
    }
    out
}

/// L0 Sysman GPU engine-utilization % per physical device, in
/// enumeration order (paired positionally with the VRAM list). Delta-
/// sampled against the previous call — the per-device `(active,
/// timestamp)` samples are kept process-globally, so the first call
/// after startup returns `None` and later calls return utilization over
/// the poll interval. `None` for a device whose driver exposes no
/// engine groups (typical on integrated Iris Xe). Empty when Sysman /
/// Level Zero isn't available.
fn sysman_busy_pct_per_device() -> Vec<Option<f32>> {
    use rustllama_l0_sys::{LevelZero, Sysman};
    use std::sync::{Mutex, OnceLock};
    static LAST: OnceLock<Mutex<Vec<Option<(u64, u64)>>>> = OnceLock::new();
    let mut out = Vec::new();
    let (Ok(sysman), Ok(l0)) = (Sysman::load(), LevelZero::load()) else {
        return out;
    };
    let Ok(drivers) = l0.drivers() else {
        return out;
    };
    // Current (active_time, timestamp) sample per device, in order.
    let mut samples: Vec<Option<(u64, u64)>> = Vec::new();
    for driver in drivers {
        if let Ok(devices) = l0.devices(driver) {
            for dev in devices {
                samples.push(
                    sysman
                        .probe(dev)
                        .engine_stats
                        .map(|s| (s.active_time, s.timestamp)),
                );
            }
        }
    }
    let last_cell = LAST.get_or_init(|| Mutex::new(Vec::new()));
    let mut last = last_cell.lock().unwrap_or_else(|p| p.into_inner());
    for (i, cur) in samples.iter().enumerate() {
        let prev = last.get(i).and_then(|o| *o);
        let util = match (cur, prev) {
            (Some((active, ts)), Some((la, lt))) if *ts > lt && *active >= la => Some(
                (((*active - la) as f64 / (*ts - lt) as f64) * 100.0).clamp(0.0, 100.0) as f32,
            ),
            _ => None,
        };
        out.push(util);
    }
    *last = samples;
    out
}

/// `GET /v1/config` — returns the current on-disk config plus the
/// file path and per-section hot-apply semantics. The GUI Settings
/// page uses `hot_apply` to label which fields take effect immediately
/// vs require a model reload or server restart.
///
/// Sections covered:
///   - `ui` / `hub` / `tuning`: hot-applied by the watcher
///   - `server`: requires a server restart for `port` / `bind_addr` /
///     `concurrency`; `cors_origins` / `api_key` / `max_pending_per_model`
///     could hot-apply later but today require restart
///   - `model` / `inference`: require a model reload
#[derive(serde::Serialize)]
struct ConfigEnvelope {
    config: rustllama_config::Config,
    config_path: Option<String>,
    /// `true` if a change to the section is picked up without a
    /// reload, `false` if it needs a reload, `"restart"` if it needs
    /// a server restart. Encoded as a free-form string so the GUI
    /// can render label text without parsing booleans.
    hot_apply: ConfigHotApply,
}

#[derive(serde::Serialize)]
struct ConfigHotApply {
    model: &'static str,
    inference: &'static str,
    server: &'static str,
    ui: &'static str,
    hub: &'static str,
    tuning: &'static str,
}

const HOT_APPLY_INFO: ConfigHotApply = ConfigHotApply {
    model: "reload",
    inference: "reload",
    server: "restart",
    ui: "live",
    hub: "live",
    tuning: "live",
};

async fn get_config(State(state): State<AppState>) -> Response {
    let Some(path) = state.config_path.as_ref() else {
        return (
            StatusCode::CONFLICT,
            "config.toml path not set on this server (likely an embedded test fixture)",
        )
            .into_response();
    };
    let cfg = match rustllama_config::load(path) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("config read failed: {e}"),
            )
                .into_response();
        }
    };
    Json(ConfigEnvelope {
        config: cfg,
        config_path: Some(path.display().to_string()),
        hot_apply: HOT_APPLY_INFO,
    })
    .into_response()
}

/// `PUT /v1/config` — replaces the entire on-disk config with the
/// posted body. The body must be a valid full `Config` JSON
/// (deserialised by `serde(default)`, so missing keys revert to
/// schema defaults — not to the previous on-disk value). Use this
/// from the GUI after rendering the existing config into the form so
/// the round-trip is non-destructive.
///
/// Returns a [`ConfigPutResponse`] describing what changed and
/// whether a model reload or server restart is needed. The on-disk
/// watcher will pick up the write and broadcast the same delta to
/// the running engine — `[ui]` / `[hub]` / `[tuning]` apply live,
/// `[model]` / `[inference]` / `[server]` need the user to follow up.
#[derive(serde::Serialize)]
struct ConfigPutResponse {
    changes: ConfigChanges,
    requires_model_reload: bool,
    requires_server_restart: bool,
    /// Elastic budget (Phase 3): `true` when the change was a
    /// budget-only `[inference]` edit (`moe_expert_cache_mb`) that
    /// was hot-applied to at least one loaded model — no reload
    /// needed, the expert cache was rebuilt at a request-safe point.
    applied_live: bool,
}

#[derive(serde::Serialize)]
struct ConfigChanges {
    model: bool,
    inference: bool,
    server: bool,
    ui: bool,
    tuning: bool,
}

async fn put_config(
    State(state): State<AppState>,
    Json(new_cfg): Json<rustllama_config::Config>,
) -> Response {
    let Some(path) = state.config_path.as_ref() else {
        return (
            StatusCode::CONFLICT,
            "config.toml path not set on this server (likely an embedded test fixture)",
        )
            .into_response();
    };
    if let Err(e) = rustllama_config::validate(&new_cfg) {
        return (StatusCode::UNPROCESSABLE_ENTITY, format!("invalid config: {e}"))
            .into_response();
    }
    let old_cfg = rustllama_config::load(path).unwrap_or_default();
    let changes = rustllama_config::ChangeSet::diff(&old_cfg, &new_cfg);
    if let Err(e) = rustllama_config::save(path, &new_cfg) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("config write failed: {e}"),
        )
            .into_response();
    }
    // Elastic budget (Phase 3): a change touching ONLY
    // [inference].moe_expert_cache_mb hot-applies to loaded models at
    // a request-safe point instead of demanding a reload.
    let applied_live = if changes.inference_budget_only {
        state
            .apply_expert_budget_live(new_cfg.inference.moe_expert_cache_mb)
            .await
    } else {
        false
    };
    Json(ConfigPutResponse {
        changes: ConfigChanges {
            model: changes.model,
            inference: changes.inference,
            server: changes.server,
            ui: changes.ui,
            tuning: changes.tuning,
        },
        requires_model_reload: changes.requires_model_reload(),
        requires_server_restart: changes.requires_server_restart(),
        applied_live,
    })
    .into_response()
}

/// `POST /v1/config/profile/apply` — switches the on-disk config to a
/// named profile from `[[profiles]]` and returns the same diff shape
/// as `PUT /v1/config`. Body: `{ "name": "<profile-name>" }`. An
/// unknown profile name is rejected with 404 rather than the warn-
/// and-no-op behavior `load_with_profile` uses at startup — at
/// startup a stale CLI flag shouldn't crash, but here the GUI's
/// dropdown is the source of truth, so a name mismatch is a real
/// error worth surfacing.
#[derive(serde::Deserialize)]
struct ApplyProfileRequest {
    name: String,
}

async fn apply_config_profile(
    State(state): State<AppState>,
    Json(req): Json<ApplyProfileRequest>,
) -> Response {
    let Some(path) = state.config_path.as_ref() else {
        return (
            StatusCode::CONFLICT,
            "config.toml path not set on this server (likely an embedded test fixture)",
        )
            .into_response();
    };
    let old_cfg = match rustllama_config::load(path) {
        Ok(c) => c,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("config read failed: {e}"),
            )
                .into_response();
        }
    };
    let mut new_cfg = old_cfg.clone();
    if !new_cfg.apply_profile(&req.name) {
        let available: Vec<String> = old_cfg.profile_names().iter().map(|s| s.to_string()).collect();
        return (
            StatusCode::NOT_FOUND,
            format!(
                "unknown profile `{}` (available: {})",
                req.name,
                if available.is_empty() {
                    "<none defined>".to_string()
                } else {
                    available.join(", ")
                }
            ),
        )
            .into_response();
    }
    // Apply-profile only mutates sections; the validate pass catches
    // any combo a profile could produce that the schema rejects (e.g.
    // setting both [model].path and [model].hub).
    if let Err(e) = rustllama_config::validate(&new_cfg) {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            format!("profile `{}` produces invalid config: {e}", req.name),
        )
            .into_response();
    }
    let changes = rustllama_config::ChangeSet::diff(&old_cfg, &new_cfg);
    if let Err(e) = rustllama_config::save(path, &new_cfg) {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("config write failed: {e}"),
        )
            .into_response();
    }
    tracing::info!(profile = %req.name, ?changes, "profile applied");
    // Same elastic-budget hot-apply as PUT /v1/config: a profile that
    // only moves the expert-cache budget applies without a reload.
    let applied_live = if changes.inference_budget_only {
        state
            .apply_expert_budget_live(new_cfg.inference.moe_expert_cache_mb)
            .await
    } else {
        false
    };
    Json(ConfigPutResponse {
        changes: ConfigChanges {
            model: changes.model,
            inference: changes.inference,
            server: changes.server,
            ui: changes.ui,
            tuning: changes.tuning,
        },
        requires_model_reload: changes.requires_model_reload(),
        requires_server_restart: changes.requires_server_restart(),
        applied_live,
    })
    .into_response()
}

// ----- POST /v1/chat/template/preview ---------------------------------------
//
// Pure render of a Jinja chat template against a supplied sample of
// chat messages. No tokenization, no model load — just runs the
// template through the same minijinja env the engine uses, with the
// active model's BOS/EOS tokens forwarded when a model is loaded
// (otherwise `None`, which `render_chat_template_with_specials`
// coerces to empty under lenient-undefined behavior). The Settings
// page's chat_template editor uses this for a live preview as the
// user types.

#[derive(serde::Deserialize)]
struct ChatTemplatePreviewMessage {
    role: String,
    content: String,
}

#[derive(serde::Deserialize)]
struct ChatTemplatePreviewRequest {
    /// Jinja template to render. Use the exact string that would go
    /// into `[model].chat_template` in `config.toml`.
    template: String,
    /// Sample conversation to render against. At least one entry
    /// recommended — an empty array is legal and renders whatever
    /// the template's pre/post chrome produces standalone.
    #[serde(default)]
    messages: Vec<ChatTemplatePreviewMessage>,
    /// Whether to append the assistant-turn primer at the end. Most
    /// chat completion flows pass `true`; FIM / completion flows pass
    /// `false`. Defaults to `true` to match what the chat path does.
    #[serde(default = "default_true")]
    add_generation_prompt: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Serialize)]
struct ChatTemplatePreviewResponse {
    /// Rendered prompt string. Whatever Jinja produced — leading /
    /// trailing whitespace preserved, special tokens (`<|im_start|>`
    /// etc.) inline.
    rendered: String,
    /// Whether the active model's tokenizer contributed BOS/EOS
    /// strings to the render. `false` when no model is loaded or the
    /// model has no BOS/EOS set; the user can read this as "the
    /// preview substitutes nothing for `{{ bos_token }}`".
    used_engine_specials: bool,
}

async fn preview_chat_template(
    State(state): State<AppState>,
    Json(req): Json<ChatTemplatePreviewRequest>,
) -> Response {
    use rustllama_tokenizer::{render_chat_template_with_specials, ChatMessage as TokChat};

    // Best-effort: if a model is loaded, route the preview through
    // its tokenizer's BOS/EOS so the rendered output matches what
    // the live engine would produce. Otherwise None/None; the lenient
    // minijinja env treats those as empty.
    let (bos, eos, used_engine_specials) = match state.try_current().await {
        Some(serving) => match serving.cpu_engine.as_ref().and_then(|c| c.tokenizer()) {
            Some(tok) => {
                let bos = tok.bos_token_str();
                let eos = tok.eos_token_str();
                let used = bos.is_some() || eos.is_some();
                (bos, eos, used)
            }
            None => (None, None, false),
        },
        None => (None, None, false),
    };

    let msgs: Vec<TokChat<'_>> = req
        .messages
        .iter()
        .map(|m| TokChat {
            role: m.role.as_str(),
            content: m.content.as_str(),
        })
        .collect();
    match render_chat_template_with_specials(
        &req.template,
        &msgs,
        req.add_generation_prompt,
        None,
        bos.as_deref(),
        eos.as_deref(),
    ) {
        Ok(rendered) => Json(ChatTemplatePreviewResponse {
            rendered,
            used_engine_specials,
        })
        .into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            format!("template render failed: {e}"),
        )
            .into_response(),
    }
}

// ----- GET /v1/capabilities ------------------------------------------------
//
// Single-shot feature-detection endpoint. Editor / GUI clients hit
// this once at startup to learn which optional features the server
// has wired (embeddings, rerank, history, audit, auth, etc.) and
// which `response_format` kinds the chat endpoint will honor. Lets
// the GUI conditionally render UI for unconfigured features and
// lets editor plugins skip probing endpoints that don't exist.

#[derive(Serialize)]
struct CapabilitiesResponse {
    /// Server version string (matches `MetricsSnapshot.uptime_s`'s
    /// implicit version axis).
    server_version: String,
    /// `response_format.type` values the chat endpoint accepts.
    /// Hard-coded list — adding a new grammar variant means
    /// updating this in lockstep with the dispatch in `chat.rs`
    /// so feature-detection stays accurate.
    response_formats: Vec<&'static str>,
    /// Major HTTP routes the server exposes. Surfaces the OpenAI
    /// + Ollama + rustllama-extension paths in one place so clients
    /// can find the endpoint they need without docs.
    endpoints: Vec<&'static str>,
    /// Embedding-model capability. `configured` is the lazy-slot
    /// gate; `dimensions` is `null` pre-first-load (same shape as
    /// the entry advertised on `/v1/models`).
    embeddings: EmbeddingCapability,
    /// Reranker-model capability. Same gating shape as
    /// [`Self::embeddings`].
    rerank: RerankCapability,
    /// Tool / function-calling capability. `supported: true` since
    /// the chat endpoint accepts `tools` + emits `tool_calls` /
    /// streams `<tool_call>` blocks; `max_iterations` is the
    /// hard cap on tool-call rounds per response.
    tools: ToolCapability,
    /// Conversation-history persistence. `enabled` is true when
    /// the server was built with `--features history` AND a store
    /// was successfully opened at startup.
    history: HistoryCapability,
    /// Per-request audit log. `enabled` is true when `[server].audit_log`
    /// resolves to an open file handle.
    audit_log: AuditCapability,
    /// Bearer-token auth. `required` is true when `[server].api_key`
    /// is non-empty (the middleware is attached).
    auth: AuthCapability,
    /// Fill-in-the-middle capability. `available` is true when the
    /// currently-loaded default chat model's tokenizer carries one
    /// of the known FIM marker sets (Qwen2.5-Coder
    /// `<|fim_prefix|>`/`<|fim_suffix|>`/`<|fim_middle|>`,
    /// DeepSeek-Coder, StarCoder/CodeLlama). Editor plugins
    /// (Continue, Cursor, etc.) use this to decide between
    /// `/v1/completions` with `suffix` (FIM) and plain completion.
    /// `null`-shaped (always-present field) — even servers with no
    /// loaded model report `available: false` so clients have a
    /// stable boolean to branch on.
    fim: FimCapability,
    /// MoE (mixture-of-experts) capability. `supported` is always
    /// true on phase 2-C servers (the inference engine handles
    /// Qwen3-MoE / Mixtral / DeepSeek-V3 GGUFs end-to-end). The
    /// `active_*` fields describe the currently-loaded default
    /// chat model: `n_experts` + `n_experts_used` populate when
    /// a MoE model is loaded, `null` otherwise. Editor / GUI
    /// clients use this to display a "this model has 8 experts,
    /// 2 routed per token" badge in the model card.
    moe: MoeCapability,
    /// Active compute backends. `cpu.available` is always true;
    /// `sycl.available` is true when at least one Intel GPU is
    /// currently visible (the SYCL kernels are always compiled in).
    backends: BackendsCapability,
}

/// Aggregate of every compute backend the server can dispatch to.
/// The GUI surfaces this in a "Backends" panel; clients use it to
/// branch on "is GPU available" without polling `/v1/metrics` and
/// inferring from `sycl_device_count`.
#[derive(Serialize)]
struct BackendsCapability {
    /// CPU dispatch — always available, with SIMD specializations
    /// per host (`avx2`/`avx512f`/`fma`/`f16c`) detected at runtime.
    /// `simd_features` enumerates what the runtime sees.
    cpu: CpuBackend,
    /// SYCL GPU dispatch — the always-compiled L0/OpenCL kernel
    /// layer. `available` gates on the runtime device count.
    sycl: SyclBackend,
    /// CUDA GPU dispatch — the always-compiled native NVIDIA kernel
    /// layer, a first-class peer of `sycl`. `available` gates on the
    /// CUDA driver enumeration; `compute_ready` on the runtime's
    /// ability to launch kernels. All-zero / false on non-NVIDIA hosts.
    cuda: CudaBackend,
}

#[derive(Serialize)]
struct CpuBackend {
    available: bool,
    /// Logical CPU count the runtime sees (`available_parallelism`).
    /// The CPU tier's capacity, reported symmetrically with each GPU
    /// backend's `device_count` so placement/UX can size the fallback.
    logical_cores: u32,
    /// Host SIMD features the runtime detected. Subset of
    /// `["avx", "avx2", "fma", "f16c", "avx512f"]`. Empty on non-x86
    /// targets (the kernels still work, just scalar).
    simd_features: Vec<&'static str>,
    /// Whether the rayon parallel-matvec path is active. Always
    /// true today (rayon's global pool is initialized at engine
    /// load); the gate flips parallel vs serial at runtime.
    parallel_matvec: bool,
    /// Logical processors usable for compute = `logical_cores` minus
    /// `disabled_cpus` — the CPU tier's effective width.
    enabled_cores: u32,
    /// Logical-processor indices excluded from the CPU pool
    /// (`[inference].disabled_cpus` / `RUSTLLAMA_DISABLED_CPUS`), the CPU
    /// analog of the GPU disable-list. Empty = every core is used.
    disabled_cpus: Vec<u32>,
    /// Whether the CPU is an enabled compute tier. `false` = GPU-only
    /// placement (`[inference].cpu_enabled` / `serve --no-cpu`).
    cpu_enabled: bool,
    /// Whether the GPU is an enabled compute tier. `false` = CPU-only
    /// placement (`[inference].gpu_enabled` / `serve --no-gpu`).
    gpu_enabled: bool,
    /// Whether VRAM-only weight residency was requested
    /// (`[inference].vram_only` / `serve --vram-only`).
    vram_only: bool,
}

#[derive(Serialize)]
struct CudaBackend {
    /// True when at least one NVIDIA GPU is visible to the CUDA driver
    /// (runtime dlopen — no toolkit needed to detect).
    available: bool,
    /// Number of NVIDIA GPUs the CUDA driver enumerates.
    device_count: u32,
    /// How many of those the native compute crate can actually launch
    /// kernels on. `compute_ready == 0` with `device_count > 0` means a
    /// driver/runtime mismatch (GPU visible, kernels can't run).
    compute_ready: u32,
    /// CUDA driver version string, when a GPU is present; `null` otherwise.
    driver: Option<String>,
}

#[derive(Serialize)]
struct SyclBackend {
    /// True when at least one SYCL device enumerates at runtime.
    /// False when no SYCL device is present — no oneAPI runtime, or
    /// a host without an Intel GPU + driver.
    available: bool,
    /// Number of GPUs the SYCL runtime sees.
    device_count: u32,
    /// Backend the runtime picked: `"level_zero"`, `"opencl"`, or
    /// the fallback labels. `null` when `available` is false.
    backend: Option<&'static str>,
    /// User preference from `RUSTLLAMA_SYCL_BACKEND`. Default
    /// `"level_zero"`; `"opencl"` / `"any"` are the other valid
    /// values. Surface this so the GUI shows "level_zero requested,
    /// opencl actual" when a fallback engaged.
    preference: String,
    /// True when the L0 USM import fast-path is eligible — i.e.
    /// backend == "level_zero". Editor / GUI uses this to decide
    /// whether to recommend `rustllama tune` (much higher gain on
    /// L0 than on OpenCL).
    l0_import_eligible: bool,
}

#[derive(Serialize)]
struct EmbeddingCapability {
    configured: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

#[derive(Serialize)]
struct RerankCapability {
    configured: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    n_labels: Option<u32>,
}

#[derive(Serialize)]
struct ToolCapability {
    supported: bool,
    max_iterations: u32,
}

#[derive(Serialize)]
struct HistoryCapability {
    enabled: bool,
}

#[derive(Serialize)]
struct AuditCapability {
    enabled: bool,
}

#[derive(Serialize)]
struct AuthCapability {
    required: bool,
}

#[derive(Serialize)]
struct MoeCapability {
    /// Engine-side MoE support. Always `true` post-phase-2-C —
    /// surfaced so clients can pattern-match on the field rather
    /// than inferring from server version.
    supported: bool,
    /// When the active chat model is MoE: the GGUF's expert count.
    /// `null` for dense models or when no model is loaded.
    #[serde(skip_serializing_if = "Option::is_none")]
    active_n_experts: Option<u32>,
    /// When the active chat model is MoE: the top-K routed per
    /// token (Mixtral: 2; Qwen3-MoE: 8; DeepSeek-V3: 8).
    #[serde(skip_serializing_if = "Option::is_none")]
    active_n_experts_used: Option<u32>,
    /// When the active model is DeepSeek-V3-style MoE with always-
    /// active shared experts: the shared count. `null` for
    /// Mixtral / Qwen3-MoE (which have 0) and for dense models.
    #[serde(skip_serializing_if = "Option::is_none")]
    active_n_experts_shared: Option<u32>,
}

#[derive(Serialize)]
struct FimCapability {
    /// `true` when the active chat model's tokenizer exposes a
    /// recognized FIM marker triplet. Always `false` when no chat
    /// model is loaded (e.g. mock engine in tests, empty registry
    /// at startup before the first `/v1/models/load`).
    available: bool,
    /// FIM prefix-marker token id, when [`Self::available`] is
    /// `true`. Surfaced so power-user clients can construct FIM
    /// prompts manually via `/v1/completions` with pre-tokenized
    /// input — most editors will just use the `suffix` field and
    /// let the server insert the markers, but the ids are handy
    /// for debugging.
    #[serde(skip_serializing_if = "Option::is_none")]
    prefix_id: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    suffix_id: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    middle_id: Option<u32>,
}

async fn capabilities(State(state): State<AppState>) -> Json<CapabilitiesResponse> {
    let embeddings = match state.embedding_model.as_ref() {
        Some(slot) => {
            let dimensions = slot
                .get()
                .and_then(|r| r.as_ref().ok())
                .map(|b| b.model.cfg.d_model as u32);
            EmbeddingCapability {
                configured: true,
                dimensions,
            }
        }
        None => EmbeddingCapability {
            configured: false,
            dimensions: None,
        },
    };
    let rerank = match state.reranker_model.as_ref() {
        Some(slot) => {
            let n_labels = slot
                .get()
                .and_then(|r| r.as_ref().ok())
                .and_then(|b| b.model.classifier_head.as_ref().map(|h| h.n_labels as u32));
            RerankCapability {
                configured: true,
                n_labels,
            }
        }
        None => RerankCapability {
            configured: false,
            n_labels: None,
        },
    };

    // History capability: only true when the feature was compiled in
    // AND a store was attached at startup. Compile-time check via
    // cfg + the runtime field guards both paths.
    #[cfg(feature = "history")]
    let history = HistoryCapability {
        enabled: state.history.is_some(),
    };
    #[cfg(not(feature = "history"))]
    let history = HistoryCapability { enabled: false };

    // FIM capability: probe the active chat model's tokenizer for
    // a recognized FIM marker triplet. Mock engines + empty
    // registries report `available: false` cleanly. Held inside an
    // async block so we don't await across the registry-read lock.
    let fim = {
        let probe = state.try_current().await.and_then(|serving| {
            let cpu = serving.cpu_engine.as_ref()?;
            cpu.tokenizer().and_then(|t| t.fim_tokens())
        });
        match probe {
            Some(toks) => FimCapability {
                available: true,
                prefix_id: Some(toks.prefix),
                suffix_id: Some(toks.suffix),
                middle_id: Some(toks.middle),
            },
            None => FimCapability {
                available: false,
                prefix_id: None,
                suffix_id: None,
                middle_id: None,
            },
        }
    };

    // MoE capability: engine supports MoE inference. If the active
    // chat model is MoE, surface its expert counts so clients can
    // render a "8 experts, top-2" badge. Probes the CpuEngine's
    // underlying LlamaModel config.
    let moe = {
        let probe = state.try_current().await.and_then(|serving| {
            let cpu = serving.cpu_engine.as_ref()?;
            cpu.llama_config().moe.clone()
        });
        match probe {
            Some(m) => MoeCapability {
                supported: true,
                active_n_experts: Some(m.n_experts),
                active_n_experts_used: Some(m.n_experts_used),
                active_n_experts_shared: if m.n_experts_shared > 0 {
                    Some(m.n_experts_shared)
                } else {
                    None
                },
            },
            None => MoeCapability {
                supported: true,
                active_n_experts: None,
                active_n_experts_used: None,
                active_n_experts_shared: None,
            },
        }
    };

    Json(CapabilitiesResponse {
        server_version: state.version.clone(),
        response_formats: vec!["json_object", "json_schema", "code", "regex", "diff"],
        endpoints: vec![
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/embeddings",
            "/v1/rerank",
            "/v1/models",
            "/v1/messages",
            "/v1/metrics",
            "/v1/lan_info",
            "/v1/capabilities",
            "/api/chat",
            "/api/generate",
            "/api/embeddings",
            "/api/embed",
            "/api/tags",
            "/api/show",
            "/api/version",
            "/healthz",
        ],
        embeddings,
        rerank,
        tools: ToolCapability {
            supported: true,
            // Mirrors the engine's DEFAULT_MAX_TOOL_ITERATIONS without
            // forcing a public re-export. Surfaced so clients can cap
            // their own retry loops to a sensible number.
            max_iterations: 8,
        },
        history,
        audit_log: AuditCapability {
            enabled: state.audit.is_some(),
        },
        auth: AuthCapability {
            required: state.auth.is_some(),
        },
        fim,
        moe,
        backends: probe_backends(),
    })
}

/// Build the [`BackendsCapability`] payload by combining compile-
/// time feature flags with runtime probes (SYCL device count +
/// backend, host SIMD detection). Cheap enough to call per request
/// — the SYCL backend probe is the only non-trivial cost (~ms,
/// constructs an ephemeral stream); cached on a `OnceLock` so only
/// the first request pays it.
fn probe_backends() -> BackendsCapability {
    // CPU SIMD probe. Runtime feature detection mirrors the kernels
    // crate's `is_x86_feature_detected!` dispatch.
    let simd_features = host_simd_features();
    // SYCL probe. The backend-name lookup spins up an ephemeral
    // stream the first time it's called; cache the result so the
    // GUI's slow-tick polling doesn't repeatedly pay the
    // construction cost.
    static SYCL_BACKEND: std::sync::OnceLock<Option<&'static str>> = std::sync::OnceLock::new();
    // `device_count()` counts SYCL device *enumerations*, not physical
    // GPUs — one Iris Xe appears twice (Level Zero + OpenCL). Report the
    // physical-GPU count instead (deduped by name, matching
    // `gpus_probe`), so the GUI's Backends panel shows GPUs, not backend
    // views.
    let raw_count = rustllama_kernels_sycl::device_count().unwrap_or(0);
    let device_count = if raw_count > 0 {
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for i in 0..raw_count {
            if let Ok(info) = rustllama_kernels_sycl::device_info(i) {
                seen.insert(info.name);
            }
        }
        seen.len().max(1) as u32
    } else {
        0
    };
    let backend = if raw_count > 0 {
        // Report the backend dispatch actually uses: Level Zero if any
        // device exposes it, else OpenCL, else device 0's backend
        // (matches the "prefer Level Zero, collapse" policy).
        *SYCL_BACKEND.get_or_init(|| {
            let mut opencl = None;
            for i in 0..raw_count {
                match rustllama_kernels_sycl::current_backend_name(i) {
                    Some("level_zero") => return Some("level_zero"),
                    Some("opencl") if opencl.is_none() => opencl = Some("opencl"),
                    _ => {}
                }
            }
            opencl.or_else(|| rustllama_kernels_sycl::current_backend_name(0))
        })
    } else {
        None
    };
    let preference = std::env::var("RUSTLLAMA_SYCL_BACKEND")
        .unwrap_or_else(|_| "level_zero".to_string());
    let l0_import_eligible = backend == Some("level_zero");
    let sycl = SyclBackend {
        available: device_count > 0,
        device_count,
        backend,
        preference,
        l0_import_eligible,
    };
    // CUDA probe — first-class peer of SYCL. The CUDA *driver* (dlopen, no
    // toolkit) enumerates NVIDIA GPUs; the native compute crate reports how
    // many the *runtime* can launch kernels on. Inert (all-zero) on non-NVIDIA.
    let nvidia = rustllama_runtime::gpu_detect::detect_nvidia();
    let cuda_device_count = nvidia.as_ref().map(|i| i.gpus.len() as u32).unwrap_or(0);
    let cuda = CudaBackend {
        available: cuda_device_count > 0,
        device_count: cuda_device_count,
        compute_ready: rustllama_kernels_cuda::device_count() as u32,
        driver: nvidia
            .as_ref()
            .filter(|i| !i.gpus.is_empty())
            .map(|i| i.driver_version_str()),
    };
    let logical_cores = std::thread::available_parallelism()
        .map(|n| n.get() as u32)
        .unwrap_or(0);
    let (enabled_cores, disabled_cpus) = cpu_enabled_cores(logical_cores);
    BackendsCapability {
        cpu: CpuBackend {
            available: true,
            logical_cores,
            simd_features,
            parallel_matvec: true,
            enabled_cores,
            disabled_cpus,
            cpu_enabled: rustllama_runtime::cpu_tier_enabled(),
            gpu_enabled: rustllama_runtime::gpu_tier_enabled(),
            vram_only: rustllama_runtime::vram_only_requested(),
        },
        sycl,
        cuda,
    }
}

// ----- GET /v1/lan_info -----------------------------------------------------
//
// Surfaces the host's LAN IP + the URL a sibling device should
// connect to + an SVG QR encoding of that URL. The GUI Settings
// page renders this panel so the user can scan with a phone /
// tablet on the same network and hit the local server.
//
// LAN-IP discovery: the OS already picks the routable interface
// when we connect to a "remote" address; we exploit that with a
// no-op UDP connect to a routable IPv4. No packet is actually sent
// — `connect()` on a UDP socket just sets the default destination
// and lets us read back `local_addr()`. This works on Windows
// without enumerating NICs, doesn't require a new dep, and returns
// the IP a phone scanning the QR would actually hit. Multi-NIC
// hosts get the OS's default-route interface, which is what they'd
// expect.

#[derive(Serialize)]
struct LanInfoResponse {
    /// Configured `bind_addr` from `config.toml`. Echoed back so
    /// the GUI can label the panel correctly: `127.0.0.1` means
    /// LAN access is disabled; `0.0.0.0` means open on every
    /// interface.
    bind_addr: String,
    /// Configured `port` (from `[server].port`).
    port: u16,
    /// The primary LAN IPv4 if discovery succeeded. `None` on
    /// hosts with no routable interface (rare — even an isolated
    /// laptop usually has a `link-local` address).
    primary_lan_ip: Option<String>,
    /// `true` if `[server].api_key` is non-empty. Sensitive —
    /// don't echo the key itself; the GUI uses this to decide
    /// whether to render an "auth required" hint next to the URL.
    api_key_set: bool,
    /// The URL another device should connect to. `null` when LAN
    /// access is disabled (bind_addr is loopback) or no IP could
    /// be discovered.
    url: Option<String>,
    /// SVG markup for a QR encoding `url`. `null` when `url` is
    /// null. The GUI embeds this inline via `dangerouslySetInnerHTML`
    /// — qrcode's SVG output is deterministic, doesn't contain
    /// script tags, and renders at any size.
    qr_svg: Option<String>,
}

fn discover_primary_lan_ip() -> Option<String> {
    // No packet is sent — UDP `connect()` just sets the default
    // peer. The local_addr the OS hands back is the IP it would
    // route through. Works regardless of whether 8.8.8.8 is
    // actually reachable.
    let sock = std::net::UdpSocket::bind("0.0.0.0:0").ok()?;
    sock.connect("8.8.8.8:80").ok()?;
    let ip = sock.local_addr().ok()?.ip();
    // Suppress the loopback fallback some OSes return when there's
    // no routable interface; the caller should see "no LAN" instead
    // of a misleading 127.0.0.1.
    if ip.is_loopback() || ip.is_unspecified() {
        return None;
    }
    Some(ip.to_string())
}

async fn lan_info(State(state): State<AppState>) -> Response {
    // Pull the live config off disk so the panel reflects the most
    // recent saved bind_addr / port / api_key. Falls back to a
    // sensible default when no config_path is attached (test
    // fixtures) so the endpoint never 5xxs on a quirky state.
    let (bind_addr, port, api_key_set) = match state.config_path.as_ref() {
        Some(path) => match rustllama_config::load(path) {
            Ok(cfg) => (
                cfg.server.bind_addr.clone(),
                cfg.server.port,
                !cfg.server.api_key.is_empty(),
            ),
            Err(_) => ("127.0.0.1".to_string(), 11434, false),
        },
        None => ("127.0.0.1".to_string(), 11434, false),
    };

    // Decide whether to produce a URL + QR. Loopback bind means
    // LAN access is disabled by config; the GUI should prompt the
    // user to flip bind_addr to 0.0.0.0 first.
    let bind_is_loopback = bind_addr.starts_with("127.") || bind_addr == "::1";
    let (url, qr_svg) = if bind_is_loopback {
        (None, None)
    } else {
        match discover_primary_lan_ip() {
            Some(ip) => {
                let u = format!("http://{ip}:{port}");
                let svg = match qrcode::QrCode::new(u.as_bytes()) {
                    Ok(code) => Some(
                        code.render::<qrcode::render::svg::Color>()
                            .min_dimensions(180, 180)
                            .build(),
                    ),
                    Err(e) => {
                        tracing::warn!(error = %e, "qr encode failed");
                        None
                    }
                };
                (Some(u), svg)
            }
            None => (None, None),
        }
    };

    let primary_lan_ip = discover_primary_lan_ip();
    Json(LanInfoResponse {
        bind_addr,
        port,
        primary_lan_ip,
        api_key_set,
        url,
        qr_svg,
    })
    .into_response()
}

// ----- /v1/crash_logs ------------------------------------------------------
//
// Surfaces files written by `rustllama_runtime::crash::install_panic_hook`
// — one per panic, in `<paths.crash_log_dir>/crash-<secs>-<pid>.log`.
// The GUI Settings page renders a viewer panel against these endpoints
// so users can triage a panic without digging through the user-data dir.

#[derive(Serialize)]
struct CrashLogEntry {
    /// File stem with extension, e.g. `crash-1234567890-9876.log`.
    /// This is the value the GUI passes back to `:name` for read /
    /// delete — never an absolute path, so a malicious client can't
    /// reach outside the crash log dir.
    name: String,
    /// Absolute path on disk. Surfaced for display purposes only;
    /// the GUI's "view" / "delete" actions go through `:name`.
    path: String,
    size_bytes: u64,
    /// Unix epoch seconds extracted from the `crash-<secs>-<pid>.log`
    /// filename. Faster + more reliable than stat'ing each file for
    /// `mtime`, and matches what the panic hook stamps anyway.
    epoch_secs: u64,
}

#[derive(Serialize)]
struct ListCrashLogsResponse {
    dir: String,
    entries: Vec<CrashLogEntry>,
}

/// Validate a `:name` path parameter against the strict crash-log
/// shape: `crash-<digits>-<digits>.log`. Rejects everything else —
/// any `/`, `..`, or unusual character. The GUI only ever passes
/// values returned from `list_crash_logs`, but the validator is the
/// real defence against a hand-crafted client.
fn is_valid_crash_log_name(name: &str) -> bool {
    if !name.starts_with("crash-") || !name.ends_with(".log") {
        return false;
    }
    let middle = &name["crash-".len()..name.len() - ".log".len()];
    let Some((secs, pid)) = middle.split_once('-') else {
        return false;
    };
    !secs.is_empty()
        && !pid.is_empty()
        && secs.bytes().all(|b| b.is_ascii_digit())
        && pid.bytes().all(|b| b.is_ascii_digit())
}

async fn list_crash_logs() -> Response {
    let dir = rustllama_runtime::paths().crash_log_dir.clone();
    if !dir.exists() {
        // No crash dir yet → empty list, not an error. The dir
        // only materializes after the first panic; users browsing
        // before that get a clean "nothing here" state.
        return Json(ListCrashLogsResponse {
            dir: dir.display().to_string(),
            entries: Vec::new(),
        })
        .into_response();
    }
    let read = match std::fs::read_dir(&dir) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("read crash log dir {}: {e}", dir.display()),
            )
                .into_response();
        }
    };
    let mut entries: Vec<CrashLogEntry> = Vec::new();
    for ent in read.flatten() {
        let name = ent.file_name().to_string_lossy().to_string();
        if !is_valid_crash_log_name(&name) {
            continue;
        }
        let meta = match ent.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };
        if !meta.is_file() {
            continue;
        }
        // Extract the epoch_secs from the filename. Validated above
        // so the unwrap path is dead in practice; defensive parse
        // here lets the iterator stay infallible.
        let middle = &name["crash-".len()..name.len() - ".log".len()];
        let epoch_secs = middle
            .split_once('-')
            .and_then(|(s, _)| s.parse::<u64>().ok())
            .unwrap_or(0);
        entries.push(CrashLogEntry {
            name,
            path: ent.path().display().to_string(),
            size_bytes: meta.len(),
            epoch_secs,
        });
    }
    // Newest first — operator wants the most recent panic at the top.
    entries.sort_by(|a, b| b.epoch_secs.cmp(&a.epoch_secs));
    // Cap at 50 so the GUI doesn't render a huge list when a flaky
    // dev build has dropped hundreds of crashes. Older files stay
    // on disk; the user can sweep them via the file system.
    entries.truncate(50);
    Json(ListCrashLogsResponse {
        dir: dir.display().to_string(),
        entries,
    })
    .into_response()
}

async fn read_crash_log(
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Response {
    if !is_valid_crash_log_name(&name) {
        return (StatusCode::BAD_REQUEST, "invalid crash log name").into_response();
    }
    let path = rustllama_runtime::paths().crash_log_dir.join(&name);
    // Defence-in-depth: even after the name-shape check, refuse if
    // the resolved path escapes the crash log dir. Catches symlink
    // tricks if a name somehow survives the regex but resolves
    // outside the dir.
    let canon_dir = std::fs::canonicalize(&rustllama_runtime::paths().crash_log_dir).ok();
    let canon_path = std::fs::canonicalize(&path).ok();
    if let (Some(d), Some(p)) = (canon_dir.as_ref(), canon_path.as_ref()) {
        if !p.starts_with(d) {
            return (StatusCode::BAD_REQUEST, "path escapes crash log dir").into_response();
        }
    }
    match std::fs::read_to_string(&path) {
        Ok(body) => (
            StatusCode::OK,
            [(axum::http::header::CONTENT_TYPE, "text/plain; charset=utf-8")],
            body,
        )
            .into_response(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            (StatusCode::NOT_FOUND, format!("crash log not found: {name}")).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("read crash log {name}: {e}"),
        )
            .into_response(),
    }
}

async fn delete_crash_log(
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Response {
    if !is_valid_crash_log_name(&name) {
        return (StatusCode::BAD_REQUEST, "invalid crash log name").into_response();
    }
    let path = rustllama_runtime::paths().crash_log_dir.join(&name);
    match std::fs::remove_file(&path) {
        Ok(()) => (StatusCode::OK, format!("deleted {name}")).into_response(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (
            StatusCode::NOT_FOUND,
            format!("crash log not found: {name}"),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("delete crash log {name}: {e}"),
        )
            .into_response(),
    }
}

#[derive(serde::Deserialize)]
struct TokenizeRequest {
    /// Text to tokenize. Must be non-empty (empty returns `{tokens:[]}`).
    #[serde(default)]
    content: String,
    /// Optional model override. Falls back to the current default.
    #[serde(default)]
    model: Option<String>,
    /// Whether to prepend the tokenizer's BOS token. Defaults to the
    /// tokenizer's own `add_bos_token` config, which matches what
    /// `Engine::generate` does internally — so the count surfaced
    /// here equals what the generation path actually sees.
    #[serde(default)]
    add_bos: Option<bool>,
}

#[derive(Serialize)]
struct TokenizeResponse {
    tokens: Vec<u32>,
    /// Echoed back so clients can pin the count against a specific
    /// model in multi-model deployments (the tokenizer differs per
    /// model — same text, different counts).
    model_id: String,
    count: u32,
}

async fn tokenize_handler(
    State(state): State<AppState>,
    Json(req): Json<TokenizeRequest>,
) -> Response {
    let Some(serving) = state.resolve(req.model.as_deref()).await else {
        return model_not_found(req.model.as_deref());
    };
    let Some(cpu) = serving.cpu_engine.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "tokenize requires a real CpuEngine (not the MockEngine)",
        )
            .into_response();
    };
    let Some(tokenizer) = cpu.tokenizer() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "model has no tokenizer attached",
        )
            .into_response();
    };
    let add_bos = req.add_bos.unwrap_or_else(|| tokenizer.add_bos_token());
    let tokens = match tokenizer.encode(&req.content, add_bos) {
        Ok(t) => t,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("encode: {e}"),
            )
                .into_response()
        }
    };
    let count = tokens.len() as u32;
    Json(TokenizeResponse {
        tokens,
        model_id: serving.model_id.clone(),
        count,
    })
    .into_response()
}

#[derive(serde::Deserialize)]
struct DetokenizeRequest {
    tokens: Vec<u32>,
    #[serde(default)]
    model: Option<String>,
    /// Strip the tokenizer's reserved/special-token markers from the
    /// rendered text. Mirrors the `skip_special_tokens` flag every
    /// real HF tokenizer exposes. Defaults to `true` — the common
    /// case for "show this to a user".
    #[serde(default = "default_skip_special")]
    skip_special_tokens: bool,
}

fn default_skip_special() -> bool {
    true
}

#[derive(Serialize)]
struct DetokenizeResponse {
    content: String,
    model_id: String,
}

async fn detokenize_handler(
    State(state): State<AppState>,
    Json(req): Json<DetokenizeRequest>,
) -> Response {
    let Some(serving) = state.resolve(req.model.as_deref()).await else {
        return model_not_found(req.model.as_deref());
    };
    let Some(cpu) = serving.cpu_engine.as_ref() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "detokenize requires a real CpuEngine",
        )
            .into_response();
    };
    let Some(tokenizer) = cpu.tokenizer() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            "model has no tokenizer attached",
        )
            .into_response();
    };
    // Walk one id at a time so we tolerate inputs that include
    // tokenizer-special ids (BOS/EOS) without the batch path
    // erroring out on the special-token check.
    let mut out = String::new();
    for id in &req.tokens {
        match tokenizer.decode_single(*id, req.skip_special_tokens) {
            Ok(s) => out.push_str(&s),
            Err(_) => {
                // Out-of-vocab → render as `<N>` placeholder rather
                // than failing the whole request.
                out.push_str(&format!("<{id}>"));
            }
        }
    }
    Json(DetokenizeResponse {
        content: out,
        model_id: serving.model_id.clone(),
    })
    .into_response()
}

/// Cache of the LAN URLs computed at server-startup time. We don't
/// recompute on every healthz call — interface enumeration on Windows
/// can hit slow paths (e.g., RDP/VPN adapter probes). The CLI's `serve`
/// command stamps this once after binding.
static LAN_URLS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();

fn enumerate_lan_urls(_state: &AppState) -> Vec<String> {
    LAN_URLS.get().cloned().unwrap_or_default()
}

/// Stash the LAN URL list so subsequent `/healthz` queries can surface
/// it. Called once by the CLI's `serve` after binding. Idempotent: only
/// the first call sticks (this is a `OnceLock`).
pub fn set_lan_urls(urls: Vec<String>) {
    let _ = LAN_URLS.set(urls);
}

/// Unix-seconds `created` timestamp for a `/v1/models` entry. OpenAI's
/// model schema types this field as an integer and several client
/// validators reject `null`, so every `ModelObject` carries one. We use
/// the backing GGUF's mtime — a stable, meaningful "when this model
/// appeared" value that survives restarts — and fall back to the server
/// process start time for the virtual embedding/reranker slots that have
/// no single backing file. The fallback is memoized so repeated listings
/// report a consistent timestamp within a run.
fn model_created_ts(source: Option<&std::path::Path>) -> u64 {
    fn server_start() -> u64 {
        static START: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        // Not self-referential: the initializer reads the clock, it does
        // not call `server_start` (standing rule: no self-recursive
        // OnceLock init).
        *START.get_or_init(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        })
    }
    source
        .and_then(|p| std::fs::metadata(p).ok())
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or_else(server_start)
}

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
    /// Unix-seconds creation timestamp (OpenAI schema requires an int).
    created: u64,
    owned_by: &'static str,
    is_default: bool,
    /// HuggingFace model-card metadata, when a sidecar `_card.json`
    /// exists next to the loaded GGUF. Surfaces license, tags,
    /// base_model, etc. to clients that want to render rich UI.
    #[serde(skip_serializing_if = "Option::is_none")]
    model_card: Option<rustllama_hub::ModelCard>,
    /// `"chat"` for chat/completion models (the default, omitted on
    /// the wire for backwards-compat) or `"embedding"` for the
    /// configured `[embeddings]` model when present. Lets clients
    /// distinguish without guessing from the id string.
    #[serde(skip_serializing_if = "Option::is_none")]
    purpose: Option<&'static str>,
    /// For purpose="embedding" entries: vector dimension. Only set
    /// after the model has been lazy-loaded (first `/v1/embeddings`
    /// request triggers the load). Discovery callers see `null`
    /// until then, which is enough to know the endpoint is wired.
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
    /// MoE (mixture-of-experts) info, present only when the loaded
    /// chat model is MoE. Absent on dense models and on the embedding
    /// / reranker entries. Lets clients render a "8 routed, top-2"
    /// badge in the model picker without a separate
    /// `/v1/capabilities` round-trip.
    #[serde(skip_serializing_if = "Option::is_none")]
    moe: Option<ModelObjectMoe>,
}

#[derive(Serialize)]
struct ModelObjectMoe {
    n_experts: u32,
    n_experts_used: u32,
    /// `0` for Mixtral / Qwen3-MoE; positive for DeepSeek-V3-style
    /// always-active shared experts. Always serialized so clients
    /// don't have to differentiate "field absent" from "0 shared".
    n_experts_shared: u32,
}

#[derive(Serialize)]
struct ModelListResp {
    object: &'static str,
    data: Vec<ModelObject>,
}

/// `GET /v1/models/{id}` — OpenAI single-model retrieval. Returns the
/// same `ModelObject` shape `/v1/models` produces, just for one entry.
/// 404 when the id isn't loaded (chat models) and isn't one of the
/// special slot ids `rustllama-embeddings` / `rustllama-reranker`.
///
/// Editor clients use this to verify "is THIS model loaded yet?"
/// without parsing the full list — handy for warm-up polling and
/// for resolving aliases the user typed in their config.
async fn get_model(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> Response {
    // Check the special embedding / reranker slots first — they
    // never appear in `state.list()` (which only walks chat models).
    if id == "rustllama-embeddings" {
        let Some(slot) = state.embedding_model.as_ref() else {
            return model_not_found(Some(&id));
        };
        let dimensions = slot
            .get()
            .and_then(|r| r.as_ref().ok())
            .map(|b| b.model.cfg.d_model as u32);
        return Json(ModelObject {
            id,
            object: "model",
            created: model_created_ts(None),
            owned_by: "rustllama",
            is_default: false,
            model_card: None,
            purpose: Some("embedding"),
            dimensions,
            moe: None,
        })
        .into_response();
    }
    if id == "rustllama-reranker" {
        let Some(slot) = state.reranker_model.as_ref() else {
            return model_not_found(Some(&id));
        };
        let dimensions = slot
            .get()
            .and_then(|r| r.as_ref().ok())
            .and_then(|b| b.model.classifier_head.as_ref().map(|h| h.n_labels as u32));
        return Json(ModelObject {
            id,
            object: "model",
            created: model_created_ts(None),
            owned_by: "rustllama",
            is_default: false,
            model_card: None,
            purpose: Some("rerank"),
            dimensions,
            moe: None,
        })
        .into_response();
    }
    // Chat-model lookup. `state.resolve` returns the snapshot for the
    // exact id only — clients that want the default should hit
    // `/v1/models` and read `is_default`.
    let Some(serving) = state.resolve(Some(&id)).await else {
        return model_not_found(Some(&id));
    };
    // Default-detection mirrors `list_models`: walk `state.list()`
    // since it carries the (id, is_default) pair we need. Cheap —
    // the registry is small.
    let is_default = state
        .list()
        .await
        .into_iter()
        .find(|(eid, _)| eid == &id)
        .map(|(_, def)| def)
        .unwrap_or(false);
    let model_card = serving
        .cpu_engine
        .as_ref()
        .and_then(|cpu| rustllama_hub::model_card::load_for_gguf(cpu.source_path()));
    let moe = serving
        .cpu_engine
        .as_ref()
        .and_then(|cpu| cpu.llama_config().moe.as_ref())
        .map(|m| ModelObjectMoe {
            n_experts: m.n_experts,
            n_experts_used: m.n_experts_used,
            n_experts_shared: m.n_experts_shared,
        });
    Json(ModelObject {
        id,
        object: "model",
        created: model_created_ts(serving.cpu_engine.as_ref().map(|c| c.source_path())),
        owned_by: "rustllama",
        is_default,
        model_card,
        purpose: None,
        dimensions: None,
        moe,
    })
    .into_response()
}

async fn list_models(State(state): State<AppState>) -> Json<ModelListResp> {
    let entries = state.list().await;
    let mut data = Vec::with_capacity(entries.len() + 1);
    for (id, is_default) in entries {
        // Look up the model_card sidecar via the serving model's source
        // path. Falls back to None for models without a cached card
        // (e.g., loaded by raw path). Same lookup also surfaces MoE
        // expert counts when the loaded model is mixture-of-experts.
        let serving = state.resolve(Some(&id)).await;
        let model_card = serving.as_ref().and_then(|s| {
            s.cpu_engine
                .as_ref()
                .and_then(|cpu| rustllama_hub::model_card::load_for_gguf(cpu.source_path()))
        });
        let moe = serving.as_ref().and_then(|s| {
            s.cpu_engine
                .as_ref()
                .and_then(|cpu| cpu.llama_config().moe.as_ref())
                .map(|m| ModelObjectMoe {
                    n_experts: m.n_experts,
                    n_experts_used: m.n_experts_used,
                    n_experts_shared: m.n_experts_shared,
                })
        });
        data.push(ModelObject {
            id,
            object: "model",
            created: model_created_ts(
                serving
                    .as_ref()
                    .and_then(|s| s.cpu_engine.as_ref().map(|c| c.source_path())),
            ),
            owned_by: "rustllama",
            is_default,
            model_card,
            purpose: None,
            dimensions: None,
            moe,
        });
    }

    // Advertise the configured embedding model when the slot is
    // enabled. `dimensions` is only known after lazy-load; discovery
    // before the first /v1/embeddings request sees `null` and that's
    // enough to know the endpoint is wired. Once loaded, the d_model
    // is surfaced so RAG clients can validate their vector store
    // dimension upfront.
    if let Some(slot) = state.embedding_model.as_ref() {
        let dimensions = slot
            .get()
            .and_then(|r| r.as_ref().ok())
            .map(|b| b.model.cfg.d_model as u32);
        data.push(ModelObject {
            id: "rustllama-embeddings".to_string(),
            object: "model",
            created: model_created_ts(None),
            owned_by: "rustllama",
            is_default: false,
            model_card: None,
            purpose: Some("embedding"),
            dimensions,
            moe: None,
        });
    }

    // Reranker capability. Same lazy-load shape; `dimensions`
    // re-purposed for "number of classifier outputs" so a multi-
    // label reranker is visible to clients that care.
    if let Some(slot) = state.reranker_model.as_ref() {
        let dimensions = slot
            .get()
            .and_then(|r| r.as_ref().ok())
            .and_then(|b| b.model.classifier_head.as_ref().map(|h| h.n_labels as u32));
        data.push(ModelObject {
            id: "rustllama-reranker".to_string(),
            object: "model",
            created: model_created_ts(None),
            owned_by: "rustllama",
            is_default: false,
            model_card: None,
            purpose: Some("rerank"),
            dimensions,
            moe: None,
        });
    }

    Json(ModelListResp {
        object: "list",
        data,
    })
}

pub async fn run(state: AppState, bind: SocketAddr) -> Result<(), ServerError> {
    run_with_cors(state, bind, &[]).await
}

/// Same as [`run`] but additionally configures CORS from the supplied
/// origin list. Pass `cfg.server.cors_origins.as_slice()` from the
/// CLI to thread the config knob through.
pub async fn run_with_cors(
    state: AppState,
    bind: SocketAddr,
    cors_origins: &[String],
) -> Result<(), ServerError> {
    let app = router_with_cors(state.clone(), cors_origins);
    let listener = tokio::net::TcpListener::bind(bind).await?;
    tracing::info!(%bind, "rustllama-server listening");
    // `into_make_service_with_connect_info::<SocketAddr>()` wires
    // per-connection peer info through to handler extensions so the
    // auth middleware can recognize loopback connections and bypass
    // the bearer check for them. Required for the LAN-bind use case
    // (auth enforced for remote callers, GUI's own fetches stay free).
    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>())
        .with_graceful_shutdown(shutdown_signal(state))
        .await?;
    Ok(())
}

/// Future that resolves on the first SIGINT (Ctrl-C) or SIGTERM, after
/// marking `state` as draining. axum keeps the listener up until in-
/// flight tasks complete; the middleware turns away new requests during
/// that window. SIGTERM is Unix-only; on Windows we listen for Ctrl-C
/// only (which is what `taskkill /F /PID <pid>` does NOT send — that
/// hard-aborts, but `taskkill <pid>` without /F does send Ctrl-C).
async fn shutdown_signal(state: AppState) {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl-C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};
        signal(SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("received Ctrl-C; draining in-flight requests");
        }
        _ = terminate => {
            tracing::info!("received SIGTERM; draining in-flight requests");
        }
    }
    state.begin_shutdown();
}

#[cfg(test)]
mod tests {
    use super::{redact_query_string, AuthState, RateLimitBucket};

    #[test]
    fn rate_limit_bucket_admits_within_capacity() {
        // 60/min = 1/sec. Bucket starts at 60; can admit 60 in
        // immediate succession.
        let b = RateLimitBucket::new(60);
        for i in 0..60 {
            b.try_admit().unwrap_or_else(|r| panic!("admit {i} failed, retry_after={r}"));
        }
    }

    #[test]
    fn rate_limit_bucket_rejects_when_empty() {
        // Tiny bucket — 6/min refills 0.1/sec. After 6 admissions
        // it must reject with a retry_after value.
        let b = RateLimitBucket::new(6);
        for _ in 0..6 {
            b.try_admit().unwrap();
        }
        let err = b.try_admit().expect_err("expected rate-limit rejection");
        assert!(err >= 1, "Retry-After must be >= 1 second");
    }

    #[test]
    fn rate_limit_disabled_by_default() {
        // `AuthState::new` uses the 0-rate-limit constructor; admission
        // always succeeds.
        let auth = AuthState::new("secret").unwrap();
        for _ in 0..1000 {
            auth.try_admit_rate_limited().unwrap();
        }
    }

    #[test]
    fn rate_limit_attached_via_new_with_rate_limit() {
        let auth = AuthState::new_with_rate_limit("secret", 3).unwrap();
        // Three admissions succeed.
        for _ in 0..3 {
            auth.try_admit_rate_limited().unwrap();
        }
        // Fourth fails.
        assert!(auth.try_admit_rate_limited().is_err());
    }

    #[test]
    fn rate_limit_zero_per_minute_disables_bucket() {
        // Explicit zero → same as None.
        let auth = AuthState::new_with_rate_limit("secret", 0).unwrap();
        for _ in 0..1000 {
            auth.try_admit_rate_limited().unwrap();
        }
    }


    /// Each name in the SENSITIVE list masks the value, regardless
    /// of case. The audit log relies on this — adding a name to
    /// the list without a test risks a typo slipping through.
    #[test]
    fn redact_query_string_masks_all_known_sensitive_param_names() {
        // Cover every name the function lists. Each name appears in
        // lowercase, uppercase, and a mixed-case variant — proves
        // the ASCII case-insensitive compare.
        let cases = &[
            ("api_key=secret", "api_key=***"),
            ("APIKEY=hex123", "APIKEY=***"),
            ("Access_Token=eyJ", "Access_Token=***"),
            ("refresh_token=r1", "refresh_token=***"),
            ("Bearer=tok", "Bearer=***"),
            ("Authorization=basic", "Authorization=***"),
            ("KEY=42", "KEY=***"),
            ("token=tok", "token=***"),
            ("auth=basic", "auth=***"),
            ("secret=hush", "secret=***"),
            ("password=hunter2", "password=***"),
            ("passwd=hunter2", "passwd=***"),
            ("PWD=hunter2", "PWD=***"),
            ("pin=4321", "pin=***"),
            ("passcode=4321", "passcode=***"),
            ("OTP=123456", "OTP=***"),
            ("signature=AKIA", "signature=***"),
            ("X-Amz-Signature=AKIA", "X-Amz-Signature=***"),
        ];
        for (input, expected) in cases {
            assert_eq!(redact_query_string(input), *expected, "input={input:?}");
        }
    }

    /// Non-sensitive params pass through verbatim. The audit
    /// consumer needs the original key=value pairs for filtering
    /// (which model was loaded, which conv was queried, etc.).
    #[test]
    fn redact_query_string_passes_through_non_sensitive_params() {
        let cases = &[
            ("model=qwen2.5-coder-7b", "model=qwen2.5-coder-7b"),
            ("limit=50&offset=0", "limit=50&offset=0"),
            ("conv_id=abc123", "conv_id=abc123"),
            ("include_tensors=true", "include_tensors=true"),
        ];
        for (input, expected) in cases {
            assert_eq!(redact_query_string(input), *expected, "input={input:?}");
        }
    }

    /// Mixed query strings: sensitive params get masked while
    /// neighbors pass through. Order is preserved.
    #[test]
    fn redact_query_string_only_masks_sensitive_params_in_mixed_strings() {
        let q = "model=qwen&api_key=hush&limit=10&token=abc&n=5";
        let got = redact_query_string(q);
        assert_eq!(got, "model=qwen&api_key=***&limit=10&token=***&n=5");
    }

    /// Bare keys (no `=value`) pass through unchanged — they can't
    /// leak a secret since there's no value attached.
    #[test]
    fn redact_query_string_tolerates_bare_keys_without_panicking() {
        assert_eq!(redact_query_string("debug&verbose"), "debug&verbose");
        assert_eq!(redact_query_string("api_key"), "api_key");
    }

    /// Empty / single-param edge cases.
    #[test]
    fn redact_query_string_handles_empty_and_single_param() {
        assert_eq!(redact_query_string(""), "");
        assert_eq!(redact_query_string("model=x"), "model=x");
        assert_eq!(redact_query_string("api_key=x"), "api_key=***");
    }
}
