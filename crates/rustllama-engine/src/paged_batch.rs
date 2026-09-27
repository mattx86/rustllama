//! Continuous-batching engine over a shared paged KV pool.
//!
//! [`crate::cpu::CpuEngine`] is the single-request workhorse:
//! one model + one KV backend serving one generation at a time.
//! The server's [`crate::cpu::CpuEngine::fork_for_concurrent_use`]
//! gives M concurrent requests via M independent forks — real
//! parallelism on the orchestration side but each fork issues its
//! own kernel launches (M × launches per decode step).
//!
//! [`PagedBatchEngine`] is the fused-decode alternative: one
//! model + one shared paged KV pool serving M concurrent requests
//! in a single forward pass per decode tick. Same model weights;
//! same per-slot semantics (bit-identical logits asserted by
//! `forward_decode_paged_batched_matches_serial_forward_one_paged`);
//! the win is one batched kernel launch per matmul instead of M
//! independent launches, which on integrated GPUs is a sizeable
//! fraction of decode time.
//!
//! What this module ships today (3.7d):
//!   - [`PagedBatchEngine::load`] — builds the model + a shared
//!     paged store sized to hold `max_slots` concurrent max-context
//!     requests.
//!   - [`PagedBatchEngine::generate_batched_token_ids`] — synchronous
//!     "drive M requests to completion" entry point. Each request
//!     prefills serially (via single-owner paged prefill under
//!     the store lock), then the engine runs a fused-decode loop
//!     across all admitted slots until each reaches its `max_tokens`
//!     or an EOS token.
//!
//! What lands in 3.7e:
//!   - Async streaming entry (`generate_chat_streaming` / SSE) so
//!     the server can route admitted requests into this engine
//!     instead of into per-fork [`crate::cpu::CpuEngine`]s when
//!     `[server].fused_decode = true`.
//!   - Scheduler integration: per-slot lifecycle events
//!     (`Scheduler::admit` on request start, `complete` on EOS /
//!     max_tokens / cancel) so the existing
//!     [`ContinuousBatchingScheduler`] bookkeeping reflects real
//!     in-flight slots.

use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use rustllama_gguf::Gguf;
use rustllama_models::llama_arch::{DecodeSlot, LlamaModel};
use rustllama_models::page_table::PageTable;
use rustllama_models::paged_kv_cache::PagedKvCache;
use rustllama_models::paged_kv_store::PagedKvStore;
use rustllama_models::shared_paged_kv::SharedPagedKv;
use rustllama_tokenizer::{ChatMessage as TokChat, Tokenizer};

use crate::cpu::CpuEngineError;
use crate::sampling::Sampler;
use crate::{
    ChatMessage, Engine, EngineError, Metrics, RequestStats, Result as EngineResult,
    SamplingParams, Token, TokenStream,
};

/// V1 page size for the shared paged pool. Matches
/// [`crate::kv_backend::DEFAULT_PAGE_SIZE`] so the two paged
/// engines (single-slot via `KvBackend::Paged` + multi-slot via
/// `PagedBatchEngine`) agree on the page geometry.
pub const DEFAULT_PAGE_SIZE: u32 = 16;

/// Per-slot in-flight state: which prompt + sampling params it's
/// running, its KV cache, sampler RNG / Mirostat state, and the
/// growing output token list. The engine owns a `Vec<SlotState>`
/// indexed by slot position in the active set.
struct SlotState {
    sampling: SamplingParams,
    eos: Option<i32>,
    cache: PagedKvCache,
    sampler: Sampler,
    /// Growing emitted-token list — used both as the per-slot
    /// "next input" source and as the repetition-penalty history.
    history: Vec<u32>,
    /// Position of the next token to *write* into the cache (== the
    /// kv_len after the next decode step's write). After prefill of
    /// a P-token prompt this is `P`; bumps by 1 per fused decode
    /// tick.
    next_pos: u32,
    /// Token id to feed into the next fused decode step as this
    /// slot's input. Initialized to the last prompt token after
    /// prefill; updated to the sampled token after each decode step.
    next_input: i32,
    /// Tokens emitted so far for this slot. Capped at
    /// `sampling.max_tokens`.
    n_emitted: u32,
    /// True once the slot has hit max_tokens / EOS — the fused
    /// decode loop skips done slots and releases their pages.
    done: bool,
    /// Wall-clock time spent in this slot's prefill forward
    /// pass (ms). Set once during admission. Reported in the
    /// `RequestStats` snapshot on slot completion.
    prefill_ms: f64,
    /// Cumulative wall-clock time spent in fused decode ticks
    /// attributable to this slot (ms). The driver shares each
    /// tick's wall time across the slots it advanced, so this
    /// is `sum(tick_ms / active_slots_in_tick)`. Reported on
    /// slot completion.
    decode_ms: f64,
    /// Length of the original prompt (== `tokens_prefilled` in
    /// `RequestStats`). Captured at admission time so the
    /// commit-on-completion path doesn't need to re-walk
    /// `history`.
    tokens_prefilled: u32,
}

/// Incoming-request message sent from `submit_streaming` callers
/// to the long-running driver thread.
struct NewRequest {
    prompt: Vec<i32>,
    sampling: SamplingParams,
    response_tx: tokio::sync::mpsc::Sender<EngineResult<Token>>,
}

/// Per-slot driver state — `SlotState` plus the channel back to
/// the submitting caller's stream.
struct DriverSlot {
    state: SlotState,
    response_tx: tokio::sync::mpsc::Sender<EngineResult<Token>>,
}

pub struct PagedBatchEngine {
    model: Arc<LlamaModel>,
    tokenizer: Option<Arc<Tokenizer>>,
    shared: SharedPagedKv,
    max_ctx: usize,
    max_slots: u32,
    page_size: u32,
    /// Stem of the GGUF file the engine was loaded from. Mirrors
    /// `CpuEngine::model_id` so the server's `ServingModel` can
    /// build its `model_id` field uniformly across backends.
    model_id: String,
    /// Channel to the driver thread. `None` if the engine was
    /// loaded without a driver (the synchronous
    /// `generate_batched_token_ids` path stays callable without it
    /// — useful for tests and CLI batch jobs that don't need the
    /// concurrent-streaming surface). Today every `load()` spawns
    /// the driver; this field is `Option` only to allow `take()`
    /// during `Drop` so the channel closes before the join.
    request_tx: Option<tokio::sync::mpsc::UnboundedSender<NewRequest>>,
    /// Driver thread join handle. Held in an `Option` so `Drop`
    /// can `take` and `join` it after closing the request channel
    /// (closing → driver loop sees `recv() → None` and returns).
    driver_handle: Option<std::thread::JoinHandle<()>>,
    /// Atomic count of slots the driver thread currently holds in
    /// its `active` Vec. Bumped on admission, decremented on
    /// `retain` removal at slot completion. Surfaced via
    /// `metrics().paged_active_slots` so the GUI Status page can
    /// show concurrent CB occupancy in real time.
    ///
    /// `AtomicUsize` rather than wrapping the driver's `active`
    /// Vec in a mutex: the driver is the sole writer; readers
    /// (metrics endpoint, GUI) just want a snapshot per poll.
    active_slot_count: Arc<std::sync::atomic::AtomicUsize>,
    /// Stats from the most-recently-completed slot, committed by
    /// the driver on the same retain pass that releases the
    /// slot's pages. `None` until the first request completes.
    /// Surfaced via `Engine::last_request_stats_snapshot()` for
    /// the GUI Status page's "Last request: prefill X ms,
    /// decode Y ms, …" row.
    last_stats: Arc<std::sync::Mutex<Option<RequestStats>>>,
}

impl PagedBatchEngine {
    /// Load a model + initialize a shared paged pool sized to hold
    /// up to `max_slots` concurrent requests each at `max_ctx`
    /// tokens. Page pool size = `ceil(max_ctx / page_size) *
    /// max_slots` pages.
    ///
    /// `max_slots` is a hard cap — `generate_batched_token_ids`
    /// rejects request batches larger than this. The server's
    /// scheduler will gate admission at this cap once 3.7e lands.
    pub fn load(
        path: &Path,
        max_ctx: usize,
        max_slots: u32,
    ) -> Result<Self, CpuEngineError> {
        Self::load_inner(path, max_ctx, max_slots, true)
    }

    /// Same as [`Self::load`] but without a tokenizer — for callers
    /// driving token-ID inputs directly (tests, benchmarks).
    pub fn load_token_id_only(
        path: &Path,
        max_ctx: usize,
        max_slots: u32,
    ) -> Result<Self, CpuEngineError> {
        Self::load_inner(path, max_ctx, max_slots, false)
    }

    fn load_inner(
        path: &Path,
        max_ctx: usize,
        max_slots: u32,
        with_tokenizer: bool,
    ) -> Result<Self, CpuEngineError> {
        let max_slots = max_slots.max(1);
        let gguf = Gguf::open(path)?;
        let model = LlamaModel::load(&gguf)?;
        let ctx = max_ctx.min(model.cfg.ctx_train.max(max_ctx));
        let tokenizer = if with_tokenizer {
            Some(Arc::new(Tokenizer::from_gguf(&gguf)?))
        } else {
            None
        };
        let page_size = DEFAULT_PAGE_SIZE;
        let pages_per_slot = (ctx as u32).div_ceil(page_size);
        let total_pages = pages_per_slot
            .checked_mul(max_slots)
            .ok_or_else(|| CpuEngineError::Other(format!(
                "paged batch pool overflow: pages_per_slot={pages_per_slot} * max_slots={max_slots}"
            )))?;
        let store = PagedKvStore::new(
            total_pages,
            model.cfg.n_layers as u32,
            model.cfg.n_kv_heads as u32,
            page_size,
            model.cfg.head_dim as u32,
        )
        .ok_or_else(|| CpuEngineError::Other(
            "paged store init: geometry has a zero dim".into()
        ))?;
        let table = PageTable::new(total_pages, page_size);
        let shared = SharedPagedKv::new(store, table);
        let model = Arc::new(model);
        let model_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "rustllama-paged-batch".into());

        // Spawn the driver thread. Owns its own clones of the
        // model + shared store + tokenizer; receives requests via
        // `request_rx`; exits cleanly when the channel closes
        // (Drop on PagedBatchEngine drops the matching tx).
        let (request_tx, request_rx) = tokio::sync::mpsc::unbounded_channel::<NewRequest>();
        let active_slot_count =
            Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let last_stats: Arc<std::sync::Mutex<Option<RequestStats>>> =
            Arc::new(std::sync::Mutex::new(None));
        let driver_ctx = DriverContext {
            model: Arc::clone(&model),
            tokenizer: tokenizer.clone(),
            shared: shared.clone(),
            max_ctx: ctx,
            max_slots,
            page_size,
            active_slot_count: Arc::clone(&active_slot_count),
            last_stats: Arc::clone(&last_stats),
        };
        let driver_handle = std::thread::Builder::new()
            .name("rustllama-paged-batch-driver".to_string())
            .spawn(move || driver_loop(driver_ctx, request_rx))
            .map_err(|e| CpuEngineError::Other(format!("driver thread spawn: {e}")))?;

        // Post-load summary — mirrors `CpuEngine::load_inner`'s log
        // line shape and adds the paged-specific knobs (max_slots,
        // page_size, total_pages) so ops triaging continuous-
        // batching deployments see the slot-pool geometry alongside
        // the model dims.
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
            engine = "paged_batch",
            model_id = %model_id,
            arch = %model.cfg.arch,
            n_layers = model.cfg.n_layers,
            d_model = model.cfg.d_model,
            n_kv_heads = model.cfg.n_kv_heads,
            vocab_size = model.cfg.vocab_size,
            ctx_train = model.cfg.ctx_train,
            ctx = ctx,
            max_slots,
            page_size,
            total_pages,
            pages_per_slot,
            moe = moe_desc.as_deref().unwrap_or("none"),
            "model loaded — paged-batch driver thread spawned"
        );

        Ok(Self {
            model,
            tokenizer,
            shared,
            max_ctx: ctx,
            max_slots,
            page_size,
            model_id,
            request_tx: Some(request_tx),
            driver_handle: Some(driver_handle),
            active_slot_count,
            last_stats,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Direct handle to the driver's `active_slot_count` atomic
    /// for tests that need to poll the counter without taking the
    /// shared store/table mutexes (`metrics()` reads
    /// `shared.free_pages()` which competes with the driver for
    /// the table mutex; a tight watch loop using `metrics()` can
    /// race and starve the driver of mutex acquires).
    ///
    /// **Test-only.** Production callers should use `metrics()`.
    #[doc(hidden)]
    pub fn active_slot_count_for_tests(&self) -> Arc<std::sync::atomic::AtomicUsize> {
        Arc::clone(&self.active_slot_count)
    }

    pub fn n_ctx(&self) -> u32 {
        self.max_ctx as u32
    }
    pub fn vocab_size(&self) -> usize {
        self.model.cfg.vocab_size
    }
    pub fn max_slots(&self) -> u32 {
        self.max_slots
    }
    pub fn tokenizer(&self) -> Option<&Arc<Tokenizer>> {
        self.tokenizer.as_ref()
    }
    /// Snapshot of free pages in the shared pool. Exposed for the
    /// scheduler-driven admission decision in 3.7e and for the
    /// GUI status page.
    pub fn free_pages(&self) -> usize {
        self.shared.free_pages()
    }

    /// Drive M requests to completion using fused decode. Each
    /// request is a `(prompt_ids, sampling_params)` pair; the
    /// returned outer Vec is the same length as `requests`, and
    /// `out[i]` is the emitted token-id sequence for `requests[i]`.
    ///
    /// V1 semantics:
    ///   - Per-request prefills run serially under the shared
    ///     store lock (no inter-request batching for prefill yet).
    ///   - Decode steps after all prefills complete run as **one
    ///     fused call per tick** covering every still-active slot.
    ///   - Stop conditions: per-request `max_tokens` or EOS token
    ///     from the model config. String stop sequences and grammar
    ///     constraints are not yet plumbed through — defer to
    ///     [`crate::cpu::CpuEngine`] for those today.
    ///   - Request count must be `<= max_slots`. Exceeding it
    ///     returns an error rather than queuing (queueing is the
    ///     scheduler's job in 3.7e).
    pub fn generate_batched_token_ids(
        &self,
        requests: Vec<(Vec<i32>, SamplingParams)>,
    ) -> Result<Vec<Vec<i32>>, CpuEngineError> {
        let m = requests.len();
        if m == 0 {
            return Ok(Vec::new());
        }
        if m as u32 > self.max_slots {
            return Err(CpuEngineError::Other(format!(
                "request batch size {m} > max_slots {}",
                self.max_slots
            )));
        }

        // Build per-slot state: prefill each slot serially (under
        // the shared store lock) so the decode loop starts with
        // every slot already at `pos = prompt.len()`.
        let mut slots: Vec<SlotState> = Vec::with_capacity(m);
        let eos = self.model.cfg.eos_token_id.map(|e| e as i32);
        for (prompt, sampling) in requests.into_iter() {
            let prompt: Vec<i32> = prompt;
            let sampling: SamplingParams = sampling;
            if prompt.is_empty() {
                return Err(CpuEngineError::Other(
                    "empty prompt in batch (every slot needs at least one prompt token)"
                        .into(),
                ));
            }
            let total_capacity = (prompt.len() as u32) + sampling.max_tokens;
            if total_capacity as usize > self.max_ctx {
                return Err(CpuEngineError::PromptTooLong {
                    prompt_len: prompt.len(),
                    max_ctx: self.max_ctx,
                });
            }
            // PagedKvCache::new_for only reads geometry. Build a
            // throwaway store for it — the shared one would
            // require a lock.
            let geom_store = PagedKvStore::new(
                1,
                self.model.cfg.n_layers as u32,
                self.model.cfg.n_kv_heads as u32,
                self.page_size,
                self.model.cfg.head_dim as u32,
            )
            .expect("geom store");
            let mut cache = PagedKvCache::new_for(&geom_store);

            // E3.2: prefix-share against already-admitted slots in
            // this same batch. The synchronous batch admits slots
            // serially; later slots can share whole pages with
            // earlier ones when their prompts overlap.
            let page_size_usz = self.page_size as usize;
            let (shared_pages, shared_lcp) =
                pick_shared_prefix_from_slots(&prompt, &slots, page_size_usz);
            if !shared_pages.is_empty() {
                self.shared
                    .with_table_mut(|t| t.add_ref(&shared_pages));
                cache.pin_shared_prefix(shared_pages, shared_lcp as u32);
            }

            cache
                .ensure_capacity_shared(&self.shared, total_capacity)
                .map_err(|short| CpuEngineError::Other(format!(
                    "paged batch: pool short by {short} pages on slot prefill"
                )))?;
            // Prefill via single-owner forward_prefill_paged_f32 by
            // bridging through the shared store lock. Skip when the
            // entire prompt lives in shared pages.
            let prefill_start = std::time::Instant::now();
            if shared_lcp < prompt.len() {
                let suffix: &[i32] = &prompt[shared_lcp..];
                self.shared.with_store_mut(|store| {
                    let _ = self
                        .model
                        .forward_prefill_paged_f32(suffix, shared_lcp as u32, &mut cache, store);
                });
            }
            let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
            let last = *prompt.last().expect("non-empty prompt");
            let tokens_prefilled = prompt.len() as u32;
            let next_pos = tokens_prefilled;
            let sampler = Sampler::new(sampling.clone());
            let history: Vec<u32> = prompt.iter().map(|&t| t as u32).collect();
            slots.push(SlotState {
                sampling,
                eos,
                cache,
                sampler,
                history,
                next_pos,
                next_input: last,
                n_emitted: 0,
                done: false,
                prefill_ms,
                decode_ms: 0.0,
                tokens_prefilled,
            });
        }

        let vocab = self.model.cfg.vocab_size;
        let mut outputs: Vec<Vec<i32>> = (0..m).map(|_| Vec::new()).collect();

        // Fused decode loop. One iteration = one fused forward
        // call advancing every still-active slot by one token.
        loop {
            // Filter to still-active slots.
            let active_indices: Vec<usize> =
                (0..m).filter(|&i| !slots[i].done).collect();
            if active_indices.is_empty() {
                break;
            }
            // Allocate per-slot logits buffers for this step. Size
            // == m (one slot per logit row, padded with zeros for
            // done slots; we just don't borrow them).
            let mut step_logits: Vec<Vec<f32>> =
                (0..m).map(|_| vec![0f32; vocab]).collect();

            // Snapshot per-slot scalars (token + pos) BEFORE we
            // start the mutable-borrow phase — otherwise the
            // mutable borrow on `slots.iter_mut()` blocks the
            // shared reads below.
            let next_inputs: Vec<i32> =
                active_indices.iter().map(|&i| slots[i].next_input).collect();
            let next_positions: Vec<u32> =
                active_indices.iter().map(|&i| slots[i].next_pos).collect();
            // Build DecodeSlot vec with disjoint borrows on the
            // active slots' caches + logits. The `Option::take`
            // trick gives us by-index disjoint borrows safely
            // (each index is taken at most once).
            let mut cache_refs: Vec<Option<&mut PagedKvCache>> =
                slots.iter_mut().map(|s| Some(&mut s.cache)).collect();
            let mut logits_refs: Vec<Option<&mut Vec<f32>>> =
                step_logits.iter_mut().map(Some).collect();

            let mut decode_slots: Vec<DecodeSlot> =
                Vec::with_capacity(active_indices.len());
            for (k, &i) in active_indices.iter().enumerate() {
                let cache = cache_refs[i].take().unwrap();
                let logits = logits_refs[i].take().unwrap();
                decode_slots.push(DecodeSlot {
                    token_id: next_inputs[k],
                    pos: next_positions[k],
                    cache,
                    logits_out: logits.as_mut_slice(),
                });
            }

            self.model
                .forward_decode_paged_batched_f32(&mut decode_slots, &self.shared);
            // Release the borrows before sampling — we need
            // `&mut slots` again.
            drop(decode_slots);
            drop(cache_refs);
            drop(logits_refs);

            // Sample per active slot, update state, check stop.
            // Field destructuring on `&mut slots[i]` gives disjoint
            // borrows on `sampler` (mut) and `history` (shared) at
            // the same time, which `slots[i].sampler.sample(..,
            // &slots[i].history)` would forbid.
            for &i in &active_indices {
                let logits = &mut step_logits[i];
                let slot = &mut slots[i];
                let SlotState {
                    sampler,
                    history,
                    ..
                } = slot;
                let next_tok = sampler.sample(logits, history);
                outputs[i].push(next_tok as i32);
                slots[i].history.push(next_tok);
                slots[i].next_input = next_tok as i32;
                slots[i].next_pos += 1;
                slots[i].n_emitted += 1;
                // Stop conditions: max_tokens or EOS.
                let hit_max = slots[i].n_emitted >= slots[i].sampling.max_tokens;
                let hit_eos = slots[i]
                    .eos
                    .map(|eos| (next_tok as i32) == eos)
                    .unwrap_or(false);
                // Also stop if pos has reached the cache's capacity
                // (can't write any more tokens). Should not happen
                // when total_capacity was sized to prompt + max_tokens,
                // but guards against config drift.
                let hit_cap = slots[i].next_pos >= slots[i].cache.capacity_tokens();
                if hit_max || hit_eos || hit_cap {
                    slots[i].done = true;
                    slots[i].cache.release_shared(&self.shared);
                }
            }
        }

        Ok(outputs)
    }

    /// Submit one request to the driver thread and get back a
    /// streaming token feed. Tokens arrive as the driver completes
    /// each fused decode tick; back-pressure is via the mpsc
    /// channel's small capacity (4 tokens).
    ///
    /// Multiple concurrent `submit_streaming` calls land in the
    /// driver thread's request queue, prefilll on admission, and
    /// then run together through the fused decode loop. That's
    /// the throughput win: M concurrent submits → one decode
    /// kernel-launch sequence per tick instead of M independent
    /// ones.
    ///
    /// V1 constraint: the engine doesn't queue past `max_slots`.
    /// If `max_slots` slots are already in flight, the new request
    /// queues in the request channel and admits as soon as a slot
    /// frees up. Future scheduler integration (the `Scheduler`
    /// trait + `[server].fused_decode = true`) will let the
    /// server's admission layer make the queue-vs-503 decision
    /// before sending into this engine; today the queue is just
    /// internal back-pressure.
    pub fn submit_streaming(
        &self,
        prompt: Vec<i32>,
        sampling: SamplingParams,
    ) -> EngineResult<TokenStream> {
        let request_tx = self
            .request_tx
            .as_ref()
            .ok_or_else(|| EngineError::Engine("paged-batch engine has no driver (post-Drop)".into()))?;
        // Channel capacity 4 matches `CpuEngine::spawn_stream` —
        // keeps the engine ~one decode tick ahead of the SSE
        // consumer so a client disconnect drains fast.
        let (response_tx, mut response_rx) =
            tokio::sync::mpsc::channel::<EngineResult<Token>>(4);
        request_tx
            .send(NewRequest {
                prompt,
                sampling,
                response_tx,
            })
            .map_err(|_| EngineError::Engine("paged-batch driver thread has exited".into()))?;
        let stream = async_stream::stream! {
            while let Some(item) = response_rx.recv().await {
                yield item;
            }
        };
        Ok(Box::pin(stream) as Pin<Box<dyn Stream<Item = EngineResult<Token>> + Send>>)
    }

    /// Convenience wrapper: tokenize `prompt` via the engine's
    /// tokenizer, then submit through [`Self::submit_streaming`].
    /// Errors if the engine was loaded without a tokenizer.
    pub fn submit_text_streaming(
        &self,
        prompt: &str,
        sampling: SamplingParams,
    ) -> EngineResult<TokenStream> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| EngineError::Unimplemented("submit_text_streaming requires a tokenizer"))?;
        // BOS handling defers to the tokenizer's per-model defaults
        // (same as `CpuEngine::spawn_stream`'s tokenize call inside
        // `drive_generation`).
        let token_ids_u32 = tokenizer.encode(prompt, tokenizer.add_bos_token())?;
        let token_ids: Vec<i32> = token_ids_u32.into_iter().map(|t| t as i32).collect();
        if token_ids.is_empty() {
            return Err(EngineError::Engine(
                "prompt tokenized to zero tokens (after BOS handling)".into(),
            ));
        }
        self.submit_streaming(token_ids, sampling)
    }
}

impl Engine for PagedBatchEngine {
    fn last_request_stats_snapshot(&self) -> Option<RequestStats> {
        self.last_stats
            .lock()
            .ok()
            .and_then(|g| g.clone())
    }

    fn metrics(&self) -> Metrics {
        // Paged-pool state from SharedPagedKv; active-slot count
        // from the driver thread's shared atomic.
        let (paged_total_pages, paged_free_pages) = {
            let (total, _, _, _, _) = self.shared.geometry();
            (total, self.shared.free_pages() as u32)
        };
        let paged_active_slots = self
            .active_slot_count
            .load(std::sync::atomic::Ordering::Acquire) as u32;
        Metrics {
            tokens_per_second: 0.0,
            context_used: 0,
            vram_estimate_mb: 0,
            ram_estimate_mb: 0,
            paged_total_pages,
            paged_free_pages,
            paged_active_slots,
        }
    }

    fn n_ctx(&self) -> u32 {
        self.n_ctx()
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size()
    }

    fn tokenize(&self, text: &str) -> EngineResult<Vec<u32>> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| EngineError::Unimplemented("tokenize requires a tokenizer"))?;
        Ok(tokenizer.encode(text, tokenizer.add_bos_token())?)
    }

    fn chat(
        &self,
        msgs: &[ChatMessage],
        s: &SamplingParams,
    ) -> EngineResult<TokenStream> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| EngineError::Unimplemented("chat requires a tokenizer"))?;
        let prompt = {
            let tok_msgs: Vec<TokChat<'_>> = msgs
                .iter()
                .map(|m| TokChat {
                    role: &m.role,
                    content: &m.content,
                })
                .collect();
            tokenizer.render_chat(&tok_msgs, true)?
        };
        self.submit_text_streaming(&prompt, s.clone())
    }

    fn generate(
        &self,
        prompt: &str,
        s: &SamplingParams,
    ) -> EngineResult<TokenStream> {
        self.submit_text_streaming(prompt, s.clone())
    }
}

impl Drop for PagedBatchEngine {
    fn drop(&mut self) {
        // Close the request channel: the driver's `recv()` returns
        // `None` after this and the loop exits cleanly. The match
        // explicit drop ordering matters — if we joined before
        // dropping the tx, the driver would block forever on recv.
        self.request_tx.take();
        if let Some(handle) = self.driver_handle.take() {
            let _ = handle.join();
        }
    }
}

// ============================================================
// Driver thread
// ============================================================
//
// The driver owns all in-flight slot state. Callers submit
// requests via `request_tx`; the driver thread admits them
// (running prefill on admission), drives them through fused
// decode, sends tokens back through per-slot `response_tx`
// channels, and releases slot resources at request completion.
//
// Why a dedicated thread (vs a tokio task)?
//   - The fused forward pass is CPU-heavy (it's the actual model
//     compute on a synthetic GGUF in tests, real SYCL dispatches
//     in production). Running it on a tokio worker would block
//     the runtime's executor for tens of milliseconds per tick.
//   - The driver's slot state is non-`Sync` (raw `PagedKvCache`
//     ownership; `Sampler` RNG state). A dedicated thread owns
//     it without needing any extra synchronization beyond the
//     incoming request channel.

/// Inputs the driver thread captures by value when spawning.
struct DriverContext {
    model: Arc<LlamaModel>,
    #[allow(dead_code)] // Reserved for chat templating in the paged driver.
    tokenizer: Option<Arc<Tokenizer>>,
    shared: SharedPagedKv,
    max_ctx: usize,
    max_slots: u32,
    page_size: u32,
    /// Shared atomic the driver bumps on admission + decrements on
    /// slot completion. Engine-side `metrics()` reads it for the
    /// `paged_active_slots` field surfaced on the GUI Status page.
    active_slot_count: Arc<std::sync::atomic::AtomicUsize>,
    /// Shared slot for the most-recently-completed slot's stats.
    /// The driver writes it on retain-removal; the engine's
    /// `last_request_stats_snapshot` impl reads it.
    last_stats: Arc<std::sync::Mutex<Option<RequestStats>>>,
}

fn driver_loop(
    ctx: DriverContext,
    mut request_rx: tokio::sync::mpsc::UnboundedReceiver<NewRequest>,
) {
    let mut active: Vec<DriverSlot> = Vec::new();
    let vocab = ctx.model.cfg.vocab_size;
    let eos = ctx.model.cfg.eos_token_id.map(|e| e as i32);

    loop {
        // ---- Admit new requests (non-blocking drain) ----
        // Each tick we pull as many newly-submitted requests as
        // the channel has buffered. The hard cap is `max_slots`
        // active at once; requests beyond that stay in the
        // channel and get admitted as slots free up on completion.
        while (active.len() as u32) < ctx.max_slots {
            let msg = match request_rx.try_recv() {
                Ok(m) => m,
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                    // All senders gone and no active work — clean exit.
                    if active.is_empty() {
                        return;
                    }
                    break;
                }
            };
            // Admit. If admission fails (prompt too long, pool
            // exhausted, etc.), surface the error on the request's
            // channel and skip — the response stream sees one Err
            // and closes.
            match admit_request(&ctx, msg.prompt, msg.sampling, eos, &active) {
                Ok(state) => {
                    active.push(DriverSlot {
                        state,
                        response_tx: msg.response_tx,
                    });
                    ctx.active_slot_count
                        .store(active.len(), std::sync::atomic::Ordering::Release);
                }
                Err(e) => {
                    // Best-effort: caller may have dropped the rx already.
                    let _ = msg.response_tx.blocking_send(Err(e));
                }
            }
        }

        // ---- Idle case: no active slots, wait for the next request ----
        if active.is_empty() {
            // Blocking recv — driver thread is idle, no CPU spin.
            let msg = match request_rx.blocking_recv() {
                Some(m) => m,
                None => return, // all senders dropped, shutdown
            };
            match admit_request(&ctx, msg.prompt, msg.sampling, eos, &active) {
                Ok(state) => {
                    active.push(DriverSlot {
                        state,
                        response_tx: msg.response_tx,
                    });
                    ctx.active_slot_count
                        .store(active.len(), std::sync::atomic::Ordering::Release);
                }
                Err(e) => {
                    let _ = msg.response_tx.blocking_send(Err(e));
                }
            }
            continue;
        }

        // ---- One fused decode tick ----
        run_decode_tick(&ctx, &mut active, vocab);

        // ---- Drop fully-done slots, releasing their pages ----
        // Each removed slot publishes a final `RequestStats`
        // snapshot to `ctx.last_stats` so the metrics endpoint /
        // GUI Status page show the most-recently-completed
        // request's prefill + decode timing. If multiple slots
        // complete in the same retain pass, the last one wins
        // (matches single-flight `CpuEngine` semantics where
        // each completed request overwrites the slot).
        active.retain_mut(|slot| {
            if slot.state.done {
                let stats = RequestStats {
                    prefill_ms: slot.state.prefill_ms,
                    decode_ms: slot.state.decode_ms,
                    tokens_prefilled: slot.state.tokens_prefilled,
                    cache_hit_tokens: 0, // paged path has no prefix cache yet
                    tokens_generated: slot.state.n_emitted,
                    tool_call_limit_hit: false,
                };
                if let Ok(mut g) = ctx.last_stats.lock() {
                    *g = Some(stats);
                }
                slot.state.cache.release_shared(&ctx.shared);
                false
            } else {
                true
            }
        });
        // Sync the snapshot AFTER retain — retain_mut runs the
        // predicate for every slot; the post-retain `active.len()`
        // is the live count.
        ctx.active_slot_count
            .store(active.len(), std::sync::atomic::Ordering::Release);
    }
}

/// Prefill + slot-state-build for one admitted request. Returns
/// the slot's initial decoder state (positioned at `prompt.len()`,
/// ready for the first fused decode tick to produce token #0 of
/// the response).
fn admit_request(
    ctx: &DriverContext,
    prompt: Vec<i32>,
    sampling: SamplingParams,
    eos: Option<i32>,
    active: &[DriverSlot],
) -> EngineResult<SlotState> {
    if prompt.is_empty() {
        return Err(EngineError::Engine(
            "empty prompt (every request needs at least one prompt token)".into(),
        ));
    }
    let total_capacity = (prompt.len() as u32) + sampling.max_tokens;
    if total_capacity as usize > ctx.max_ctx {
        return Err(EngineError::Engine(format!(
            "prompt too long: {} tokens + max_tokens {} > max_ctx {}",
            prompt.len(),
            sampling.max_tokens,
            ctx.max_ctx,
        )));
    }
    let geom_store = PagedKvStore::new(
        1,
        ctx.model.cfg.n_layers as u32,
        ctx.model.cfg.n_kv_heads as u32,
        ctx.page_size,
        ctx.model.cfg.head_dim as u32,
    )
    .expect("geom store");
    let mut cache = PagedKvCache::new_for(&geom_store);

    // E3.2: cross-request prefix sharing. Find the active slot
    // whose `history` (prompt tokens + emitted tokens, all with
    // valid K/V) shares the longest prefix with this admission's
    // prompt. Page-aligned LCP becomes shared pages; the prefill
    // computes only the suffix.
    //
    // Why `history` not just the prompt: by the time another
    // request arrives, the donor slot may have decoded N tokens
    // past its prompt. The K/V at those positions is in the same
    // pages as the prompt prefix, so they're shareable too —
    // common in interactive chat where two clients pick up the
    // same system prompt + assistant turn.
    //
    // Page-aligned: the page table allocates whole pages, and
    // [`PagedKvCache::pin_shared_prefix`] requires `seq_len ==
    // pages.len() * page_size`. Round LCP down so partial-page
    // sharing isn't needed (it would force a copy of one page).
    let page_size = ctx.page_size as usize;
    let (shared_pages, shared_lcp) = pick_shared_prefix(&prompt, active, page_size);
    if !shared_pages.is_empty() {
        ctx.shared.with_table_mut(|t| t.add_ref(&shared_pages));
        cache.pin_shared_prefix(shared_pages, shared_lcp as u32);
    }

    cache
        .ensure_capacity_shared(&ctx.shared, total_capacity)
        .map_err(|short| {
            EngineError::Engine(format!(
                "paged batch admit: pool short by {short} pages \
                 (max_slots={} concurrent at max_ctx={})",
                ctx.max_slots, ctx.max_ctx,
            ))
        })?;

    // Prefill only the suffix [shared_lcp, prompt.len()). When the
    // full prompt lives in shared pages (rare: exact-match re-ask
    // of an active slot's history) the prefill is skipped entirely —
    // the next decode tick picks up at pos = prompt.len() from the
    // shared K/V.
    let prefill_start = std::time::Instant::now();
    if shared_lcp < prompt.len() {
        let suffix: &[i32] = &prompt[shared_lcp..];
        ctx.shared.with_store_mut(|store| {
            let _ = ctx
                .model
                .forward_prefill_paged_f32(suffix, shared_lcp as u32, &mut cache, store);
        });
    }
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    let last = *prompt.last().expect("non-empty prompt");
    let tokens_prefilled = prompt.len() as u32;
    let next_pos = tokens_prefilled;
    let sampler = Sampler::new(sampling.clone());
    let history: Vec<u32> = prompt.iter().map(|&t| t as u32).collect();
    Ok(SlotState {
        sampling,
        eos,
        cache,
        sampler,
        history,
        next_pos,
        next_input: last,
        n_emitted: 0,
        done: false,
        prefill_ms,
        decode_ms: 0.0,
        tokens_prefilled,
    })
}

/// E3.2 helper: scan the currently-active slots and return the
/// (pages, lcp_tokens) of the best page-aligned shared prefix
/// for `prompt`. `pages` is empty + lcp=0 when no slot has a
/// matching prefix of at least one whole page (i.e. < `page_size`
/// tokens).
fn pick_shared_prefix(
    prompt: &[i32],
    active: &[DriverSlot],
    page_size: usize,
) -> (Vec<rustllama_models::page_table::PageId>, usize) {
    let mut best_lcp: usize = 0;
    let mut best_pages: Vec<rustllama_models::page_table::PageId> = Vec::new();
    for slot in active {
        // Skip done slots — their pages are about to be released
        // and the slot's history may already be invalid in the
        // refcount sense (refcount == 1 from the donor; if we
        // share and the donor's free races, we'd be relying on
        // the refcount bump landing first). The retain pass that
        // releases done slots runs AFTER admission on the same
        // driver thread, so a done slot's pages are technically
        // still alive — but skipping is the cleaner invariant.
        if slot.state.done {
            continue;
        }
        let lcp = compute_lcp_i32_u32(prompt, &slot.state.history);
        // Round down to whole-page boundary; `pin_shared_prefix`
        // requires that.
        let lcp_aligned = (lcp / page_size) * page_size;
        if lcp_aligned > best_lcp && lcp_aligned >= page_size {
            best_lcp = lcp_aligned;
            let n_pages = lcp_aligned / page_size;
            best_pages = slot.state.cache.pages()[..n_pages].to_vec();
        }
    }
    (best_pages, best_lcp)
}

fn compute_lcp_i32_u32(a: &[i32], b: &[u32]) -> usize {
    let mut n = 0;
    let lim = a.len().min(b.len());
    while n < lim && a[n] as u32 == b[n] {
        n += 1;
    }
    n
}

/// Synchronous-batch variant of `pick_shared_prefix` operating on
/// `&[SlotState]` directly (the sync `generate_batched_token_ids`
/// holds slots as plain `Vec<SlotState>`, not `DriverSlot`).
fn pick_shared_prefix_from_slots(
    prompt: &[i32],
    slots: &[SlotState],
    page_size: usize,
) -> (Vec<rustllama_models::page_table::PageId>, usize) {
    let mut best_lcp: usize = 0;
    let mut best_pages: Vec<rustllama_models::page_table::PageId> = Vec::new();
    for s in slots {
        if s.done {
            continue;
        }
        let lcp = compute_lcp_i32_u32(prompt, &s.history);
        let lcp_aligned = (lcp / page_size) * page_size;
        if lcp_aligned > best_lcp && lcp_aligned >= page_size {
            best_lcp = lcp_aligned;
            let n_pages = lcp_aligned / page_size;
            best_pages = s.cache.pages()[..n_pages].to_vec();
        }
    }
    (best_pages, best_lcp)
}

/// One fused decode tick across every still-active slot in
/// `active`. Same shape as the inner loop of
/// `generate_batched_token_ids` but feeds tokens back into each
/// slot's `response_tx` instead of accumulating into a Vec.
fn run_decode_tick(ctx: &DriverContext, active: &mut [DriverSlot], vocab: usize) {
    let m = active.len();
    // E3.1: prompt cancellation propagation. Before paying the cost
    // of a fused forward + per-slot sampling tick, mark any slot
    // whose response channel has been closed (client disconnect,
    // dropped stream, timeout) as done. The retain pass after this
    // tick releases those slots' pages — freeing GPU memory and
    // attention compute for the queued requests that still have
    // live receivers.
    //
    // Was: cancellation was only observed AFTER the forward pass
    // when `response_tx.blocking_send(Ok(token))` returned `Err`,
    // costing one full decode tick of compute per cancelled slot.
    for i in 0..m {
        if !active[i].state.done && active[i].response_tx.is_closed() {
            active[i].state.done = true;
        }
    }
    let active_indices: Vec<usize> = (0..m).filter(|&i| !active[i].state.done).collect();
    if active_indices.is_empty() {
        return;
    }

    let mut step_logits: Vec<Vec<f32>> = (0..m).map(|_| vec![0f32; vocab]).collect();
    let next_inputs: Vec<i32> = active_indices.iter().map(|&i| active[i].state.next_input).collect();
    let next_positions: Vec<u32> = active_indices.iter().map(|&i| active[i].state.next_pos).collect();

    let mut cache_refs: Vec<Option<&mut PagedKvCache>> =
        active.iter_mut().map(|s| Some(&mut s.state.cache)).collect();
    let mut logits_refs: Vec<Option<&mut Vec<f32>>> = step_logits.iter_mut().map(Some).collect();

    let mut decode_slots: Vec<DecodeSlot> = Vec::with_capacity(active_indices.len());
    for (k, &i) in active_indices.iter().enumerate() {
        let cache = cache_refs[i].take().unwrap();
        let logits = logits_refs[i].take().unwrap();
        decode_slots.push(DecodeSlot {
            token_id: next_inputs[k],
            pos: next_positions[k],
            cache,
            logits_out: logits.as_mut_slice(),
        });
    }
    let tick_start = std::time::Instant::now();
    ctx.model
        .forward_decode_paged_batched_f32(&mut decode_slots, &ctx.shared);
    let tick_ms = tick_start.elapsed().as_secs_f64() * 1000.0;
    drop(decode_slots);
    drop(cache_refs);
    drop(logits_refs);

    // Amortize the tick's wall time across the slots it advanced
    // — every slot in `active_indices` got one token out of this
    // fused forward call. The per-slot decode_ms is what the
    // RequestStats snapshot reports, mirroring `CpuEngine`'s
    // single-slot decode timing.
    let per_slot_ms = if active_indices.is_empty() {
        0.0
    } else {
        tick_ms / active_indices.len() as f64
    };
    for &i in &active_indices {
        active[i].state.decode_ms += per_slot_ms;
    }

    // Sample per active slot, route token to its response_tx.
    for &i in &active_indices {
        let logits = &mut step_logits[i];
        let slot = &mut active[i];
        let SlotState {
            sampler, history, ..
        } = &mut slot.state;
        let next_tok = sampler.sample(logits, history);
        slot.state.history.push(next_tok);
        slot.state.next_input = next_tok as i32;
        slot.state.next_pos += 1;
        slot.state.n_emitted += 1;

        // Detokenize for the stream Token. Without a tokenizer
        // the Token's text is empty — clients can detokenize
        // themselves from the id.
        // Per-token decode: `Tokenizer::decode` operates on
        // id slices, so we hand it the single new token. Some
        // tokenizers emit empty strings for control IDs — that's
        // fine, the client gets `id` either way and can
        // detokenize on its own if it cares about the boundary.
        let text = match &ctx.tokenizer {
            Some(tok) => tok.decode(&[next_tok], false).unwrap_or_default(),
            None => String::new(),
        };
        let token = Token {
            id: next_tok,
            text,
            logprobs: None,
        };
        // Best-effort send: if the receiver dropped (client
        // disconnect), mark the slot done so its pages release
        // on the next retain pass.
        if slot.response_tx.blocking_send(Ok(token)).is_err() {
            slot.state.done = true;
            continue;
        }

        // Stop conditions: max_tokens, EOS, or cache exhaustion.
        let hit_max = slot.state.n_emitted >= slot.state.sampling.max_tokens;
        let hit_eos = slot
            .state
            .eos
            .map(|e| (next_tok as i32) == e)
            .unwrap_or(false);
        let hit_cap = slot.state.next_pos >= slot.state.cache.capacity_tokens();
        if hit_max || hit_eos || hit_cap {
            slot.state.done = true;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// E3.2: the LCP helper is the core of cross-request prefix
    /// sharing. Bit-exact on a few hand-built cases pins the
    /// algorithm without needing a full synthetic model.
    #[test]
    fn lcp_handles_empty_inputs() {
        assert_eq!(compute_lcp_i32_u32(&[], &[]), 0);
        assert_eq!(compute_lcp_i32_u32(&[1, 2, 3], &[]), 0);
        assert_eq!(compute_lcp_i32_u32(&[], &[1, 2, 3]), 0);
    }

    #[test]
    fn lcp_returns_full_match_length() {
        let a: &[i32] = &[1, 2, 3, 4];
        let b: &[u32] = &[1, 2, 3, 4];
        assert_eq!(compute_lcp_i32_u32(a, b), 4);
    }

    #[test]
    fn lcp_stops_at_first_mismatch() {
        let a: &[i32] = &[1, 2, 9, 4];
        let b: &[u32] = &[1, 2, 3, 4];
        assert_eq!(compute_lcp_i32_u32(a, b), 2);
    }

    #[test]
    fn lcp_clamps_to_shorter_input() {
        let a: &[i32] = &[1, 2, 3, 4, 5];
        let b: &[u32] = &[1, 2, 3];
        assert_eq!(compute_lcp_i32_u32(a, b), 3);
    }

    /// Defensive: a negative i32 token (shouldn't happen in
    /// practice — tokens are u32-wide) still compares correctly
    /// against the unsigned history side. `as u32` wraps; the
    /// reinterpretation makes any negative value unequal to any
    /// real token id, so the LCP stops cleanly.
    #[test]
    fn lcp_handles_negative_i32_defensively() {
        let a: &[i32] = &[1, 2, -1, 4];
        let b: &[u32] = &[1, 2, 3, 4];
        assert_eq!(compute_lcp_i32_u32(a, b), 2);
    }
}
