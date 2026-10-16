//! Continuous-batching engine over a shared paged KV pool.
//!
//! [`crate::inference::CpuEngine`] is the single-request workhorse:
//! one model + one KV backend serving one generation at a time.
//! The server's [`crate::inference::CpuEngine::fork_for_concurrent_use`]
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
//! Async + scheduling surface:
//!   - Async streaming entry (`submit_streaming` / SSE) so the server
//!     routes admitted requests into this engine instead of per-fork
//!     [`crate::inference::CpuEngine`]s when `[server].fused_decode = true`.
//!   - P14: the driver thread admits from a priority-ordered heap
//!     (`X-RustLlama-Priority` → [`crate::SubmitOpts::priority`]) rather than
//!     FIFO, enforces per-request deadlines (flipping the request's cancel
//!     flag), and does F5 recompute preemption of the lowest-priority running
//!     slot on page exhaustion. See [`PagedBatchEngine::submit_streaming_with_opts`].

use std::cmp::Ordering as CmpOrdering;
use std::collections::BinaryHeap;
use std::path::Path;
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

use futures::Stream;
use rustllama_gguf::Gguf;
use rustllama_models::llama_arch::{DecodeSlot, DeltaNetCache, KvCache, KvDtype, LlamaModel};
use rustllama_models::page_table::PageTable;
use rustllama_models::paged_kv_cache::PagedKvCache;
use rustllama_models::paged_kv_store::PagedKvStore;
use rustllama_models::shared_paged_kv::SharedPagedKv;
use rustllama_tokenizer::{ChatMessage as TokChat, Tokenizer};

use crate::batch_scheduler::Priority;
use crate::grammar::GrammarMask;
use crate::inference::CpuEngineError;
use crate::sampling::Sampler;
use crate::{
    ChatMessage, Engine, EngineError, GrammarKind, Metrics, RequestStats, Result as EngineResult,
    SamplingParams, SubmitOpts, Token, TokenStream,
};

/// P9 feature parity: tool-call grammar recursion cap for the paged driver.
/// Mirrors `CpuEngine`'s `DEFAULT_MAX_TOOL_ITERATIONS` (8) — the paged engine
/// has no per-request tool-iteration knob, so it uses the same default the
/// contiguous engine starts from.
const PAGED_MAX_TOOL_ITERATIONS: u32 = 8;

/// P9 feature parity: build the per-request grammar mask for the paged driver,
/// mirroring `CpuEngine::build_grammar`. Returns `None` when no grammar was
/// requested, or (fail-soft, like the contiguous path) when the engine has no
/// tokenizer to pre-decode the vocab. Built once per request at admission —
/// each slot owns its own advancing `GrammarMask`.
fn build_paged_grammar(
    model: &LlamaModel,
    tokenizer: Option<&Arc<Tokenizer>>,
    sampling: &SamplingParams,
) -> Option<GrammarMask> {
    let kind = sampling.grammar.as_ref()?;
    let tokenizer = match tokenizer {
        Some(t) => t,
        None => {
            tracing::warn!(
                "grammar requested ({kind:?}) but paged engine has no tokenizer; ignoring"
            );
            return None;
        }
    };
    let vocab = model.cfg.vocab_size;
    let eos = model.cfg.eos_token_id;
    let decode = |id: u32| tokenizer.decode_single(id, true).unwrap_or_default().into_bytes();
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
            PAGED_MAX_TOOL_ITERATIONS,
            *min_completed,
            vocab,
            eos,
            decode,
        )),
        GrammarKind::Code { language } => {
            Some(GrammarMask::new_code_for_language(language, vocab, eos, decode))
        }
        GrammarKind::Regex { pattern } => match GrammarMask::new_regex(pattern, vocab, eos, decode) {
            Ok(m) => Some(m),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    pattern,
                    "invalid regex pattern in GrammarKind::Regex — ignoring grammar on paged engine"
                );
                None
            }
        },
    }
}

/// V1 page size for the shared paged pool. Matches
/// [`crate::kv_backend::DEFAULT_PAGE_SIZE`] so the two paged
/// engines (single-slot via `KvBackend::Paged` + multi-slot via
/// `PagedBatchEngine`) agree on the page geometry.
pub const DEFAULT_PAGE_SIZE: u32 = 16;

/// Best-effort detection of a vision / multimodal (VLM) GGUF from what a text
/// decoder's own file reveals. Used by the F6-A fail-closed paged VLM guard.
///
/// The common VLM shape is TWO files — a text-decoder GGUF (which carries no
/// vision tensors) plus a separate `mmproj-*.gguf` holding the vision tower
/// under `v.*` / `clip.*`. So this check does NOT fire on an ordinary two-file
/// VLM's text half; the PRIMARY guard for that case is the serve load path,
/// which forces the contiguous `CpuEngine` whenever a `[model].mmproj` companion
/// is configured. This helper is the defense-in-depth half: it catches a
/// COMBINED single-file VLM GGUF — any `clip.*` / `*.vision.*` metadata key, a
/// `clip` / `siglip` / vision architecture, or a vision-namespaced tensor
/// (`v.` prefix, `mmproj`, or `.vision.`) — that would otherwise slip past the
/// mmproj check and be served text-only.
fn gguf_has_vision_indicator(gguf: &Gguf) -> bool {
    if let Some(arch) = gguf.architecture() {
        let a = arch.to_ascii_lowercase();
        if a == "clip" || a == "siglip" || a.contains("vision") {
            return true;
        }
    }
    if gguf.metadata().iter().any(|(k, _)| {
        k.starts_with("clip.") || k.contains(".vision.") || k.starts_with("vision.")
    }) {
        return true;
    }
    gguf.tensors()
        .iter()
        .any(|t| t.name.starts_with("v.") || t.name.contains("mmproj") || t.name.contains(".vision."))
}

/// F8: per-slot recurrent + attention state for a HYBRID (DeltaNet/SSM)
/// model. The paged KV store has no per-position addressing for the
/// recurrent half (a DeltaNet snapshot is valid only at the exact token
/// position it was captured — see [`DeltaNetCache`]), so a hybrid slot
/// does NOT use the shared paged pool. Instead it carries its OWN
/// contiguous F32 KV cache (for the full-attention layers) + its OWN
/// `DeltaNetCache` (for the SSM/recurrent layers) and is driven through the
/// SAME `forward_prefill_hybrid` / `forward_one_hybrid` the contiguous
/// `CpuEngine` uses — so a paged hybrid slot's output is BIT-IDENTICAL to
/// the contiguous engine on the same prompt + sampling. Multiple hybrid
/// slots interleave at the driver's tick granularity, each advancing its
/// own `kv` + `dn` independently, so there is no cross-slot contamination
/// of the recurrent state (the F8 anchor).
struct HybridSlotState {
    /// Contiguous F32 KV cache for the full-attention layers (SSM layers'
    /// slabs are allocated-but-unused — the hybrid forward never writes KV
    /// on an SSM layer). Sized to `max_ctx`.
    kv: KvCache,
    /// Per-slot DeltaNet recurrent state (conv + recurrent per SSM layer).
    dn: DeltaNetCache,
    /// `kv.max_ctx` as u32 — the slot's write capacity, used for the
    /// `hit_cap` stop check in place of the paged `cache.capacity_tokens()`.
    capacity: u32,
}

/// Per-slot in-flight state: which prompt + sampling params it's
/// running, its KV cache, sampler RNG / Mirostat state, and the
/// growing output token list. The engine owns a `Vec<SlotState>`
/// indexed by slot position in the active set.
struct SlotState {
    sampling: SamplingParams,
    eos: Option<i32>,
    cache: PagedKvCache,
    /// F8: `Some` for a HYBRID model's slot (per-slot contiguous KV +
    /// DeltaNet recurrent state; the paged `cache` above stays an empty
    /// placeholder and its page ops are no-ops). `None` for every
    /// dense / MoE model — the paged `cache` is the real KV store and
    /// the slot machinery is byte-identical to the pre-F8 path.
    hybrid: Option<HybridSlotState>,
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
    /// Prompt tokens served from the APC / shared-prefix cache (the
    /// page-aligned shared prefix length) — those positions skipped
    /// prefill. Reported as `cache_hit_tokens` on completion.
    cache_hit_tokens: u32,
    /// Chunked-prefill progress. `None` = this slot's prompt is fully
    /// prefilled (or was wholly served from shared pages) and it is
    /// decode-ready. `Some(job)` = the ragged driver is still forwarding the
    /// prompt in chunks; the slot contributes prefill rows (not a decode row)
    /// to the ragged tick until it drains.
    prefill_job: Option<PrefillJob>,
    /// P9 feature parity: the per-slot grammar / structured-output mask
    /// (`response_format`, tool grammar), built from `sampling.grammar` at
    /// admission. `None` = unconstrained. The decode path samples via
    /// `Sampler::sample_with_grammar` + advances it per token, exactly like the
    /// contiguous `CpuEngine` — so a grammar request behaves identically on the
    /// batching engine.
    grammar: Option<GrammarMask>,
    /// P9 feature parity: client-supplied string stop sequences
    /// (`sampling.stop`). Empty = none. Matched against [`Self::stop_buf`].
    stop_strings: Vec<String>,
    /// P9 feature parity: the running detokenized output buffer used for
    /// string-stop matching. Accumulated via `decode_single(id, true)`
    /// (matching `CpuEngine::drive_generation`), so a multi-token stop string
    /// spanning several tokens is caught. Only grown when `stop_strings` is
    /// non-empty (no detok cost otherwise).
    stop_buf: String,
    /// F5 recompute-preemption replay counter. `0` for a normal slot. When a
    /// slot is preempted (its pages freed), it re-admits as a FRESH prefill of
    /// its ORIGINAL prompt with `skip_emit` set to the number of tokens it had
    /// already streamed; the deterministic greedy/seeded forward regenerates
    /// those tokens bit-for-bit, and the decode path SUPPRESSES streaming them
    /// (decrementing `skip_emit`) so the client's stream is byte-identical to
    /// an uninterrupted run — the already-sent tokens are not re-sent. Once it
    /// reaches `0`, streaming resumes normally. This replay reuses the exact
    /// prefill+decode code path, so transparency holds regardless of the
    /// paged position convention.
    skip_emit: u32,
    /// F5: total tokens actually STREAMED to the client over this slot's whole
    /// (possibly multi-resume) life. PRESERVED across a preemption (unlike
    /// `n_emitted`, which resets to 0 each resume and counts regenerations).
    /// On preemption the resume's `skip_emit` is set to this, so a slot
    /// preempted AGAIN mid-replay still skips exactly the already-seen tokens
    /// and never re-streams one.
    emitted_to_client: u32,
    /// F7: the seeded RNG for this slot's MTP/tree speculative rounds
    /// (non-greedy accept/reject). Lazily created on the first spec round from
    /// `sampling.seed`, then reused across rounds so by-seed determinism holds —
    /// mirroring the contiguous `speculate_mtp_stream`'s one-RNG-per-stream. Stays
    /// `None` for a slot whose model carries no draft head (spec never engages)
    /// and for greedy spec (deterministic, no draws).
    spec_rng: Option<crate::sampling::Rng>,
}

impl SlotState {
    /// Token write-capacity for the `hit_cap` stop check. A HYBRID slot
    /// (F8) stores KV in its own contiguous cache, so its capacity is that
    /// cache's `max_ctx`; a dense/MoE slot uses the paged `cache`'s
    /// capacity. Keeps the shared decode-commit path dtype-agnostic.
    fn effective_capacity_tokens(&self) -> u32 {
        match &self.hybrid {
            Some(h) => h.capacity,
            None => self.cache.capacity_tokens(),
        }
    }
}

/// In-flight chunked prefill for one slot. `admit_request` stages the slot
/// (pages, shared-prefix, APC acquire, capacity) then, instead of forwarding
/// the whole suffix inline, hands the driver this job; each `run_ragged_tick`
/// forwards a budget-limited chunk (the whole suffix at once when the budget is
/// unbounded) until the suffix drains, at which point the prompt's blocks are
/// registered in the APC index and the slot becomes decode-ready.
struct PrefillJob {
    /// The prompt suffix still to forward (`prompt[shared_lcp..]`).
    suffix: Vec<i32>,
    /// How many suffix tokens have been forwarded so far.
    cursor: usize,
    /// Cache position the suffix starts at (the shared-prefix length).
    base: usize,
    /// APC block-hash chain for the FULL prompt, registered once on completion
    /// so a later request reuses this KV (the inline path registers at admit).
    chain: Vec<rustllama_models::apc::BlockHash>,
    /// Prompt length + wall-clock start, for the completion `RequestStats`.
    prompt_len: u32,
    cache_hit_tokens: u32,
    started: std::time::Instant,
}

/// Incoming-request message sent from `submit_streaming` callers
/// to the long-running driver thread.
struct NewRequest {
    prompt: Vec<i32>,
    sampling: SamplingParams,
    response_tx: tokio::sync::mpsc::Sender<EngineResult<Token>>,
    /// P14: admission priority (higher admits sooner from the driver's heap;
    /// FIFO within equal priority via the driver's arrival counter).
    priority: Priority,
    /// P14: optional absolute per-request deadline; the driver cancels the
    /// slot when it elapses.
    deadline: Option<Instant>,
    /// P14: optional cancel flag the driver flips on deadline expiry (the
    /// server's `CancelGuard.flag`).
    cancel_flag: Option<Arc<AtomicBool>>,
}

/// Per-slot driver state — `SlotState` plus the channel back to
/// the submitting caller's stream and its P14 scheduling metadata.
struct DriverSlot {
    state: SlotState,
    response_tx: tokio::sync::mpsc::Sender<EngineResult<Token>>,
    /// P14 scheduling: admission priority. Drives both which waiter admits
    /// first AND which running slot F5 preempts (the lowest-priority one).
    priority: Priority,
    /// P14: absolute deadline (if any). Checked each tick + at admission.
    deadline: Option<Instant>,
    /// P14: cancel flag flipped on deadline expiry.
    cancel_flag: Option<Arc<AtomicBool>>,
    /// Monotonic arrival order (FIFO tiebreaker within a priority level).
    /// Assigned by the driver when the request first enters the waiting heap;
    /// a preempted-and-requeued slot keeps its ORIGINAL value so it does not
    /// lose its place to newer same-priority arrivals.
    arrival_seq: u64,
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
    /// F5: cumulative count of preemptions the driver has performed. Shared
    /// with the driver thread; read by tests to confirm the recompute-
    /// preemption path actually fired under page exhaustion.
    preemptions: Arc<std::sync::atomic::AtomicUsize>,
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
        // Config budget 0 ⇒ the ragged mixed path is OFF unless the
        // `RUSTLLAMA_MAX_BATCHED_TOKENS` env var opts in (env still honored for
        // back-compat; see `resolve_max_batched_tokens`).
        Self::load_inner(path, max_ctx, max_slots, true, 0, None)
    }

    /// Same as [`Self::load`] but with the `[inference].max_num_batched_tokens`
    /// config value threaded in. Feature 3b: a positive budget turns on the
    /// ragged mixed prefill+decode driver path (one fused forward per tick). The
    /// `RUSTLLAMA_MAX_BATCHED_TOKENS` env var still overrides it (env-over-config,
    /// matching the repo's tiering) — see [`resolve_max_batched_tokens`]. The
    /// resolved value is stamped into [`DriverContext::max_batched_tokens`] so
    /// the driver loop never reads the env itself.
    pub fn load_with_batched_tokens(
        path: &Path,
        max_ctx: usize,
        max_slots: u32,
        max_num_batched_tokens: usize,
    ) -> Result<Self, CpuEngineError> {
        Self::load_inner(path, max_ctx, max_slots, true, max_num_batched_tokens, None)
    }

    /// Same as [`Self::load`] but without a tokenizer — for callers
    /// driving token-ID inputs directly (tests, benchmarks).
    pub fn load_token_id_only(
        path: &Path,
        max_ctx: usize,
        max_slots: u32,
    ) -> Result<Self, CpuEngineError> {
        Self::load_inner(path, max_ctx, max_slots, false, 0, None)
    }

    /// Test-only: like [`Self::load_token_id_only`] but with an explicit page
    /// pool size (pages), SMALLER than the `max_slots × pages_per_slot` the
    /// production path reserves. A production pool is sized so `max_slots`
    /// concurrent requests each at `max_ctx` always fit — so page exhaustion
    /// (and thus F5 preemption) can never trigger from admission. An
    /// undersized pool is the realistic memory-constrained config vLLM targets
    /// with preemption; this constructor lets the engine tests exercise the
    /// F5 recompute-preemption path deterministically.
    #[doc(hidden)]
    pub fn load_token_id_only_with_pool_pages(
        path: &Path,
        max_ctx: usize,
        max_slots: u32,
        total_pages: u32,
    ) -> Result<Self, CpuEngineError> {
        Self::load_inner(path, max_ctx, max_slots, false, 0, Some(total_pages))
    }

    fn load_inner(
        path: &Path,
        max_ctx: usize,
        max_slots: u32,
        with_tokenizer: bool,
        config_max_batched_tokens: usize,
        pool_pages_override: Option<u32>,
    ) -> Result<Self, CpuEngineError> {
        let max_slots = max_slots.max(1);
        let gguf = Gguf::open(path)?;
        let model = LlamaModel::load(&gguf)?;

        // D1a (relaxed by D1b, then F8): fail-closed model gate.
        // `PagedBatchEngine` is selected by `[server].fused_decode` and
        // previously did NO gating — so a model whose paged forward path is
        // wrong or missing was served silently.
        //   * HYBRID (DeltaNet/SSM): F8 lifted the blanket refusal. A hybrid
        //     slot does not use the shared paged pool at all — the recurrent
        //     half has no per-position addressing, so paged prefix reuse can't
        //     apply to it. Instead each hybrid slot carries its OWN contiguous
        //     F32 KV + its OWN `DeltaNetCache` and is driven through the SAME
        //     `forward_prefill_hybrid` / `forward_one_hybrid` the contiguous
        //     `CpuEngine` uses (see [`HybridSlotState`], [`build_hybrid_slot_state`],
        //     [`run_hybrid_tick`]) — BIT-IDENTICAL to contiguous, so it is
        //     correct to admit. The one hybrid sub-case still refused:
        //     SWA / attention-logit soft-cap, because the hybrid forward
        //     hardcodes window=0 / softcap=0.0 for its attention layers (it has
        //     no windowed/capped hybrid kernel), so a hybrid model that ALSO
        //     declares sliding-window or a soft-cap can't be served correctly on
        //     EITHER engine here — fail-closed with a pointed message rather than
        //     serve it wrong. (No real hybrid arch declares these today.)
        //   * SWA / Gemma2-softcap (non-hybrid): D1b threaded the per-layer
        //     sliding-window span + attention-logit soft-cap through the F32 paged
        //     forwards, so these models are served correctly here. The paged KV
        //     store is F32-ONLY, so there is no non-F32 paged path to fence off.
        //
        // MoE is NOT refused here. The old refusal fenced off the budget-0 INLINE
        // prefill (`forward_prefill_paged_f32`), which loops only `weights.blocks`
        // (empty for MoE). After T1 the driver has a single path — the ragged
        // forward (`forward_ragged_paged_f32`), which handles `moe_blocks` — and a
        // zero budget means UNBOUNDED ragged prefill, not the inline path. So MoE
        // is served correctly on the paged engine at any budget; the gate is gone.
        if model.weights.is_hybrid()
            && (model.cfg.sliding_window.is_some()
                || model.cfg.attn_logit_softcapping.is_some())
        {
            return Err(CpuEngineError::Other(
                "continuous batching (fused_decode) does not support a HYBRID \
                 (DeltaNet/SSM) model that also uses sliding-window attention or \
                 attention-logit soft-capping (the hybrid forward has no \
                 windowed/capped attention kernel); disable fused_decode to serve \
                 this model on the default engine"
                    .into(),
            ));
        }
        // F6-A: fail-closed VLM guard (P9 default-on prereq). A vision/mmproj model
        // has no image-embed row source in the ragged paged forward yet (F6-B adds
        // it), so serving one here would silently drop every image → text-only,
        // mis-served output. VLM-ness in rustllama is normally signalled by a
        // SEPARATE mmproj companion GGUF (the text decoder carries no vision
        // tensors), which this load site does not see — so the PRIMARY guard lives
        // where the mmproj IS known: the serve load path forces the contiguous
        // `CpuEngine` (which `attach_mmproj`s) whenever `[model].mmproj` is set
        // (cli `serve`). This block is the defense-in-depth half: it catches a
        // COMBINED single-file VLM GGUF (vision tensors / `clip.*` metadata in the
        // same file) that would otherwise slip past the mmproj check.
        if gguf_has_vision_indicator(&gguf) {
            return Err(CpuEngineError::Other(
                "continuous batching (fused_decode) does not support vision / \
                 multimodal (VLM) models yet; use the default engine for this model"
                    .into(),
            ));
        }

        let ctx = max_ctx.min(model.cfg.ctx_train.max(max_ctx));
        let tokenizer = if with_tokenizer {
            Some(Arc::new(Tokenizer::from_gguf(&gguf)?))
        } else {
            None
        };
        let page_size = DEFAULT_PAGE_SIZE;
        let pages_per_slot = (ctx as u32).div_ceil(page_size);
        // Production sizing: enough pages for `max_slots` requests each at the
        // full context. A test may pass a SMALLER `pool_pages_override` to
        // force page exhaustion and exercise F5 recompute preemption.
        let total_pages = match pool_pages_override {
            Some(p) => p.max(1),
            None => pages_per_slot
                .checked_mul(max_slots)
                .ok_or_else(|| CpuEngineError::Other(format!(
                    "paged batch pool overflow: pages_per_slot={pages_per_slot} * max_slots={max_slots}"
                )))?,
        };
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
        // LMCache: auto-size the CPU cold tier from free RAM (no env var). The
        // tier is DEMAND-FILLED — the budget is a CAP, not a preallocation, so a
        // generous fraction is safe: it only holds KV blocks that were actually
        // evicted and might be reused. Evicted blocks go to RAM instead of being
        // lost, and promote back on a later content-hash hit. A disk tier under
        // it (ephemeral per-process file) absorbs RAM-tier overflow at the same
        // auto cap. Both off only when free RAM can't be read / is ~0.
        let avail = rustllama_runtime::memory_info().available_bytes as usize;
        // ≤ 1/8 of free RAM, capped at 8 GiB — plenty for KV-block reuse without
        // crowding the model's working set.
        let cold_cap = (avail / 8).min(8 * 1024 * 1024 * 1024);
        if cold_cap >= 64 * 1024 * 1024 {
            shared.set_cold_cap_bytes(cold_cap);
            tracing::info!(
                cap_mb = cold_cap / (1024 * 1024),
                "LMCache CPU cold tier auto-enabled (paged serving engine)"
            );
            let path = std::env::temp_dir()
                .join(format!("rustllama-lmcache-{}.bin", std::process::id()));
            if shared.enable_disk_tier(&path, cold_cap) {
                tracing::info!(cap_mb = cold_cap / (1024 * 1024), ?path, "LMCache disk tier auto-enabled");
            } else {
                tracing::warn!(?path, "LMCache disk tier could not be opened — RAM tier only");
            }
        }
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
        let preemptions = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let driver_ctx = DriverContext {
            model: Arc::clone(&model),
            tokenizer: tokenizer.clone(),
            shared: shared.clone(),
            max_ctx: ctx,
            max_slots,
            page_size,
            total_pages,
            active_slot_count: Arc::clone(&active_slot_count),
            last_stats: Arc::clone(&last_stats),
            preemptions: Arc::clone(&preemptions),
            // Per-tick prompt-prefill token budget for the single ragged
            // driver path (T1). Every tick routes through the RAGGED mixed
            // forward (`forward_ragged_paged_f32`): one fused pass over the
            // budgeted prefill chunks + the decode rows of every active slot,
            // so a long prompt stops monopolizing the driver (TTFT fairness)
            // and prefill + decode share a single kernel-launch sequence.
            //
            // A resolved `0` (unset / zero) maps to `usize::MAX` = UNBOUNDED:
            // the whole prompt is prefilled in one ragged tick (token-identical
            // to the old budget-0 inline path). A positive value chunks the
            // prefill across ticks.
            //
            // Precedence (env-over-config, matching the repo's flag>env>config
            // tiering): a user-set `RUSTLLAMA_MAX_BATCHED_TOKENS` wins; otherwise
            // the `[inference].max_num_batched_tokens` config value threaded in
            // via `load_with_batched_tokens`. Resolved HERE so the driver loop
            // reads only `ctx.max_batched_tokens`, never the env.
            max_batched_tokens: resolve_max_batched_tokens(config_max_batched_tokens),
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
            preemptions,
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// F5 (test-only): number of recompute preemptions the driver has performed
    /// since load. Lets the preemption anchor confirm the path actually fired
    /// under forced page exhaustion.
    #[doc(hidden)]
    pub fn preemptions_for_tests(&self) -> usize {
        self.preemptions.load(std::sync::atomic::Ordering::Acquire)
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
    ///   - Stop conditions: per-request `max_tokens`, the model's EOS
    ///     token, cache capacity, OR a client string stop sequence
    ///     (`sampling.stop`). P9 brought the paged engine to feature
    ///     parity with [`crate::inference::CpuEngine`]: `sampling.grammar`
    ///     (structured output / tool grammar) is applied per slot via
    ///     [`Sampler::sample_with_grammar`], and string stops are matched
    ///     against the slot's running detok buffer — same semantics as the
    ///     contiguous path (the stop-completing token is emitted, then the
    ///     slot ends).
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
            // E2: client-controlled `max_tokens` can make `prompt_len +
            // max_tokens` wrap u32 in release (no overflow checks) → the
            // `> max_ctx` guard below is bypassed and generation silently
            // stops at `hit_cap`. Compute in usize with a checked add.
            let total_capacity = prompt
                .len()
                .checked_add(sampling.max_tokens as usize)
                .unwrap_or(usize::MAX);
            if total_capacity > self.max_ctx {
                return Err(CpuEngineError::PromptTooLong {
                    prompt_len: prompt.len(),
                    max_ctx: self.max_ctx,
                });
            }
            // F8: a HYBRID model's slot carries its own contiguous KV +
            // DeltaNet state (no shared pool, no cross-request prefix share),
            // prefilled through the contiguous hybrid forward — bit-identical
            // to `CpuEngine`. Build it and move to the next request.
            if self.model.weights.is_hybrid() {
                let state = build_hybrid_slot_state(
                    &self.model,
                    self.tokenizer.as_ref(),
                    self.max_ctx,
                    &prompt,
                    &sampling,
                    eos,
                    0,
                )
                .map_err(|e| CpuEngineError::Other(format!("hybrid slot build: {e}")))?;
                slots.push(state);
                continue;
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
            // Live-slot LCP (this batch's earlier slots) + vLLM APC
            // (content-addressed, across all prior requests). Keep the longer.
            let (live_pages, live_lcp) =
                pick_shared_prefix_from_slots(&prompt, &slots, page_size_usz);
            let prompt_u32: Vec<u32> = prompt.iter().map(|&t| t as u32).collect();
            let chain = rustllama_models::apc::block_hash_chain(&prompt_u32, page_size_usz);
            let apc_pages = self.shared.apc_acquire_prefix(&chain);
            let apc_lcp = apc_pages.len() * page_size_usz;
            let (shared_pages, shared_lcp) = if apc_lcp > 0 && apc_lcp >= live_lcp {
                (apc_pages, apc_lcp)
            } else {
                if !apc_pages.is_empty() {
                    self.shared.with_table_mut(|t| t.free(&apc_pages));
                }
                if !live_pages.is_empty() {
                    self.shared.with_table_mut(|t| t.add_ref(&live_pages));
                }
                (live_pages, live_lcp)
            };
            if !shared_pages.is_empty() {
                cache.pin_shared_prefix(shared_pages, shared_lcp as u32);
            }

            cache
                // E2: `total_capacity` is a usize now (checked add); it is
                // `<= self.max_ctx` here, which fits u32, so the cast can't
                // truncate.
                .ensure_capacity_shared(&self.shared, total_capacity as u32)
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
            // Register the prompt's full blocks (APC) — reusable by later slots/
            // requests by content hash. Block `i` lives in page `i`.
            if !chain.is_empty() {
                let pages = cache.pages();
                let n = chain.len().min(pages.len());
                let entries: Vec<_> = (0..n).map(|i| (chain[i], pages[i])).collect();
                self.shared.apc_register(&entries);
            }
            let last = *prompt.last().expect("non-empty prompt");
            let tokens_prefilled = prompt.len() as u32;
            let next_pos = tokens_prefilled;
            let sampler = Sampler::new(sampling.clone());
            // P9 feature parity: grammar + string stops per slot.
            let grammar = build_paged_grammar(&self.model, self.tokenizer.as_ref(), &sampling);
            let stop_strings = sampling.stop.clone();
            let history: Vec<u32> = prompt_u32;
            slots.push(SlotState {
                sampling,
                eos,
                cache,
                // Dense/MoE sync-batch slot: the paged `cache` is the real KV
                // store (hybrid gets its own path below).
                hybrid: None,
                sampler,
                history,
                next_pos,
                next_input: last,
                n_emitted: 0,
                done: false,
                prefill_ms,
                decode_ms: 0.0,
                tokens_prefilled,
                cache_hit_tokens: shared_lcp as u32,
                // The synchronous batched path always prefills inline.
                prefill_job: None,
                grammar,
                stop_strings,
                stop_buf: String::new(),
                skip_emit: 0,
                emitted_to_client: 0,
                spec_rng: None,
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

            // F8: a HYBRID model advances each active slot with its OWN
            // `forward_one_hybrid` against its OWN contiguous KV + DeltaNet
            // state (no shared paged forward). Independent per slot, so a
            // multi-slot batch keeps each slot's recurrent state isolated.
            if self.model.weights.is_hybrid() {
                for &i in &active_indices {
                    let next_input = slots[i].next_input;
                    let next_pos = slots[i].next_pos;
                    let hyb = slots[i]
                        .hybrid
                        .as_mut()
                        .expect("hybrid sync slot without hybrid state");
                    self.model.forward_one_hybrid(
                        next_input,
                        next_pos,
                        &mut hyb.kv,
                        &mut hyb.dn,
                        step_logits[i].as_mut_slice(),
                    );
                }
            } else {
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
            }

            // Sample per active slot, update state, check stop.
            // Field destructuring on `&mut slots[i]` gives disjoint
            // borrows on `sampler` (mut) and `history` (shared) at
            // the same time, which `slots[i].sampler.sample(..,
            // &slots[i].history)` would forbid.
            for &i in &active_indices {
                let logits = &mut step_logits[i];
                let slot = &mut slots[i];
                // P9: grammar-masked sample + grammar advance (feature parity
                // with `CpuEngine::generate_token_ids`).
                let next_tok = {
                    let SlotState {
                        sampler,
                        history,
                        grammar,
                        ..
                    } = slot;
                    let tok = sampler.sample_with_grammar(logits, history, grammar.as_ref());
                    if let Some(g) = grammar.as_mut() {
                        g.advance(tok);
                    }
                    tok
                };
                outputs[i].push(next_tok as i32);
                slots[i].history.push(next_tok);
                slots[i].next_input = next_tok as i32;
                slots[i].next_pos += 1;
                slots[i].n_emitted += 1;
                // P9 string-stop detection: accumulate the detok buffer exactly
                // like the contiguous CpuEngine (`decode_single(id, true)`) and
                // match. The stop-completing token is still emitted (it's in
                // `outputs[i]`), then the slot stops — mirroring
                // `drive_generation`.
                let hit_stop = if slots[i].stop_strings.is_empty() {
                    false
                } else {
                    let piece = self
                        .tokenizer
                        .as_ref()
                        .and_then(|t| t.decode_single(next_tok, true).ok())
                        .unwrap_or_default();
                    let slot = &mut slots[i];
                    slot.stop_buf.push_str(&piece);
                    let buf = &slot.stop_buf;
                    slot.stop_strings
                        .iter()
                        .any(|s| !s.is_empty() && buf.contains(s.as_str()))
                };
                // Stop conditions: max_tokens, EOS, capacity, or a string stop.
                let hit_max = slots[i].n_emitted >= slots[i].sampling.max_tokens;
                let hit_eos = slots[i]
                    .eos
                    .map(|eos| (next_tok as i32) == eos)
                    .unwrap_or(false);
                // Also stop if pos has reached the cache's capacity
                // (can't write any more tokens). Should not happen
                // when total_capacity was sized to prompt + max_tokens,
                // but guards against config drift.
                // F8: hybrid slots store KV in their own contiguous cache, so
                // use the effective capacity (the paged `cache` is empty there).
                let hit_cap = slots[i].next_pos >= slots[i].effective_capacity_tokens();
                if hit_max || hit_eos || hit_cap || hit_stop {
                    slots[i].done = true;
                    // No-op for a hybrid slot (empty placeholder paged cache).
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
        self.submit_streaming_with_opts(prompt, sampling, SubmitOpts::default())
    }

    /// P14: like [`Self::submit_streaming`] but with per-request scheduling
    /// options — admission `priority` (drives the driver's priority-ordered
    /// admission heap) and an optional `deadline` + `cancel_flag` (the driver
    /// flips the flag and ends the slot with a [`EngineError::DeadlineExceeded`]
    /// terminal once the deadline elapses). [`SubmitOpts::default`] reproduces
    /// [`Self::submit_streaming`] exactly (DEFAULT priority, no deadline).
    pub fn submit_streaming_with_opts(
        &self,
        prompt: Vec<i32>,
        sampling: SamplingParams,
        opts: SubmitOpts,
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
                priority: opts.priority,
                deadline: opts.deadline,
                cancel_flag: opts.cancel_flag,
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
        self.submit_text_streaming_with_opts(prompt, sampling, SubmitOpts::default())
    }

    /// P14: opts-aware [`Self::submit_text_streaming`] — tokenizes then submits
    /// with the given scheduling options. Used by the `Engine::chat_scheduled`
    /// / `generate_scheduled` overrides so the server's parsed
    /// priority/deadline reach the driver.
    pub fn submit_text_streaming_with_opts(
        &self,
        prompt: &str,
        sampling: SamplingParams,
        opts: SubmitOpts,
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
        self.submit_streaming_with_opts(token_ids, sampling, opts)
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

/// Resolve the effective per-tick batched-token budget with the repo's
/// env-over-config precedence. The `RUSTLLAMA_MAX_BATCHED_TOKENS` env var wins
/// when set to a parseable `usize`; otherwise the
/// `[inference].max_num_batched_tokens` config value (`config_val`) applies.
///
/// T1: the paged driver now has a SINGLE execution path — the ragged mixed
/// prefill+decode tick ([`run_ragged_tick`]). A resolved value of `0` (unset /
/// zero from both tiers) therefore means **unbounded, one-shot prefill**, NOT
/// "disabled": it maps to `usize::MAX`, so the whole prompt is prefilled in one
/// ragged tick. That is byte-identical (in generated token sequence) to the old
/// budget-0 inline prefill-then-decode pair — pinned by the ragged≡inline and
/// unbounded≡finite parity tests. A POSITIVE value bounds the prompt-prefill
/// tokens forwarded per tick (TTFT fairness: a long prompt is chunked across
/// ticks instead of monopolizing the driver). Resolving here keeps the
/// resolution in ONE place and lets the driver loop treat
/// `ctx.max_batched_tokens` as the single source of truth.
pub(crate) fn resolve_max_batched_tokens(config_val: usize) -> usize {
    let resolved = match std::env::var("RUSTLLAMA_MAX_BATCHED_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(env_val) => env_val, // env wins when present + parseable
        None => config_val,       // else the config-threaded value
    };
    // 0 = unbounded: one ragged tick prefills the whole prompt (≡ the former
    // budget-0 inline path's token output). A positive value caps the
    // prompt-prefill tokens per tick.
    if resolved == 0 {
        usize::MAX
    } else {
        resolved
    }
}

/// Inputs the driver thread captures by value when spawning.
struct DriverContext {
    model: Arc<LlamaModel>,
    #[allow(dead_code)] // Reserved for chat templating in the paged driver.
    tokenizer: Option<Arc<Tokenizer>>,
    shared: SharedPagedKv,
    max_ctx: usize,
    max_slots: u32,
    page_size: u32,
    /// Total pages in the shared pool. P14/F5: a request whose full capacity
    /// needs MORE than this can never be scheduled even by preempting every
    /// other slot, so the admission path rejects it up front (fatal) instead
    /// of spinning. With production sizing this is `max_slots * pages_per_slot`
    /// so every request fits; an undersized (test) pool makes F5 preemption
    /// reachable.
    total_pages: u32,
    /// Shared atomic the driver bumps on admission + decrements on
    /// slot completion. Engine-side `metrics()` reads it for the
    /// `paged_active_slots` field surfaced on the GUI Status page.
    active_slot_count: Arc<std::sync::atomic::AtomicUsize>,
    /// Shared slot for the most-recently-completed slot's stats.
    /// The driver writes it on retain-removal; the engine's
    /// `last_request_stats_snapshot` impl reads it.
    last_stats: Arc<std::sync::Mutex<Option<RequestStats>>>,
    /// F5: count of slots the driver has preempted (for observability + tests).
    /// Bumped in [`preempt_slot`].
    preemptions: Arc<std::sync::atomic::AtomicUsize>,
    /// Per-tick prompt-prefill token budget for the single ragged driver path
    /// (`RUSTLLAMA_MAX_BATCHED_TOKENS` / `[inference].max_num_batched_tokens`).
    /// Already resolved: `0` was mapped to `usize::MAX` (unbounded — whole
    /// prompt prefilled in one ragged tick). Never `0` here. See the
    /// construction and [`resolve_max_batched_tokens`].
    max_batched_tokens: usize,
}

/// P14/F5: one unit of pending admission work in the driver's priority heap.
/// A FRESH submit carries `skip_emit == 0`; an F5-preempted slot is requeued as
/// the SAME request (its ORIGINAL prompt + response channel + scheduling
/// metadata) with `skip_emit` = the tokens it had already streamed, so the
/// deterministic replay regenerates-but-does-not-restream them.
struct PendingReq {
    prompt: Vec<i32>,
    sampling: SamplingParams,
    response_tx: tokio::sync::mpsc::Sender<EngineResult<Token>>,
    priority: Priority,
    deadline: Option<Instant>,
    cancel_flag: Option<Arc<AtomicBool>>,
    /// Monotonic arrival order (FIFO tiebreaker within a priority level).
    /// Preserved across a preemption so a resumed slot keeps its place.
    arrival_seq: u64,
    /// F5 replay counter — 0 for a fresh request.
    skip_emit: u32,
}

/// Max-heap entry ordering [`PendingReq`] by (priority DESC, arrival_seq ASC).
struct PendingEntry(PendingReq);
impl PartialEq for PendingEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0.priority == other.0.priority && self.0.arrival_seq == other.0.arrival_seq
    }
}
impl Eq for PendingEntry {}
impl Ord for PendingEntry {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Higher priority pops first; within equal priority the OLDER
        // (smaller arrival_seq) pops first — invert the seq comparison so the
        // max-heap surfaces the oldest.
        self.0
            .priority
            .cmp(&other.0.priority)
            .then_with(|| other.0.arrival_seq.cmp(&self.0.arrival_seq))
    }
}
impl PartialOrd for PendingEntry {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

/// True when `deadline` is set and has elapsed.
#[inline]
fn deadline_expired(deadline: Option<Instant>) -> bool {
    deadline.is_some_and(|d| Instant::now() >= d)
}

/// P14: end a slot/request cleanly on deadline expiry — flip the request's
/// cancel flag (reusing the server's `CancelGuard.flag` mechanism so its
/// stream-watch loop observes the cancellation) and emit a
/// [`EngineError::DeadlineExceeded`] terminal so the consumer can report a
/// `"timeout"` finish reason distinct from a user-initiated cancel.
fn fire_deadline_timeout(
    response_tx: &tokio::sync::mpsc::Sender<EngineResult<Token>>,
    cancel_flag: &Option<Arc<AtomicBool>>,
) {
    if let Some(flag) = cancel_flag {
        flag.store(true, std::sync::atomic::Ordering::Release);
    }
    let _ = response_tx.blocking_send(Err(EngineError::DeadlineExceeded));
}

/// F5: pick the lowest-priority RUNNING slot strictly below `incoming`
/// priority (ties broken by largest `arrival_seq` — newest preempted first).
/// Returns its index in `active`, or `None` if no running slot ranks below the
/// incoming request (so preemption would not help).
fn pick_preemption_victim(active: &[DriverSlot], incoming: Priority) -> Option<usize> {
    let mut best: Option<usize> = None;
    for (i, s) in active.iter().enumerate() {
        if s.state.done || s.priority >= incoming {
            continue;
        }
        match best {
            None => best = Some(i),
            Some(b) => {
                let bslot = &active[b];
                if s.priority < bslot.priority
                    || (s.priority == bslot.priority && s.arrival_seq > bslot.arrival_seq)
                {
                    best = Some(i);
                }
            }
        }
    }
    best
}

/// F5: free a running slot's pages and requeue it for a later recompute resume.
/// The slot's SlotState is discarded (the resume rebuilds it from the ORIGINAL
/// prompt via the normal prefill path); only its response channel, scheduling
/// metadata, and already-streamed token count (`skip_emit`) are preserved.
fn preempt_slot(ctx: &DriverContext, mut slot: DriverSlot, pending: &mut BinaryHeap<PendingEntry>) {
    ctx.preemptions
        .fetch_add(1, std::sync::atomic::Ordering::Release);
    slot.state.cache.release_shared(&ctx.shared);
    // Original prompt = the first `tokens_prefilled` tokens of `history`
    // (history = prompt ++ generated). Rebuild it as the resume's prompt.
    let p = slot.state.tokens_prefilled as usize;
    let prompt: Vec<i32> = slot.state.history[..p.min(slot.state.history.len())]
        .iter()
        .map(|&t| t as i32)
        .collect();
    pending.push(PendingEntry(PendingReq {
        prompt,
        sampling: slot.state.sampling.clone(),
        response_tx: slot.response_tx,
        priority: slot.priority,
        deadline: slot.deadline,
        cancel_flag: slot.cancel_flag,
        arrival_seq: slot.arrival_seq,
        // Already-streamed tokens: regenerate silently on resume. Use the
        // persistent client-seen count (NOT `n_emitted`, which resets each
        // resume) so a slot preempted mid-replay still skips exactly the
        // tokens the client already received.
        skip_emit: slot.state.emitted_to_client,
    }));
}

/// P14/F5: make room for a request needing `needed` pages + one free slot,
/// preempting the lowest-priority running slots strictly below `incoming`
/// until there is a free slot AND enough free pages. Returns `true` on success,
/// `false` if no further (lower-priority) slot can be preempted.
fn make_room(
    ctx: &DriverContext,
    active: &mut Vec<DriverSlot>,
    pending: &mut BinaryHeap<PendingEntry>,
    needed: usize,
    incoming: Priority,
) -> bool {
    loop {
        if (active.len() as u32) < ctx.max_slots && ctx.shared.free_pages() >= needed {
            return true;
        }
        match pick_preemption_victim(active, incoming) {
            Some(idx) => {
                let victim = active.remove(idx);
                preempt_slot(ctx, victim, pending);
            }
            None => return false,
        }
    }
}

fn driver_loop(
    ctx: DriverContext,
    mut request_rx: tokio::sync::mpsc::UnboundedReceiver<NewRequest>,
) {
    let mut active: Vec<DriverSlot> = Vec::new();
    // P14: waiting work ordered by priority (then FIFO by arrival). Fresh
    // submits drain off the channel into here; F5-preempted slots requeue here.
    let mut pending: BinaryHeap<PendingEntry> = BinaryHeap::new();
    let mut next_seq: u64 = 0;
    let vocab = ctx.model.cfg.vocab_size;
    let eos = ctx.model.cfg.eos_token_id.map(|e| e as i32);
    let page_size = ctx.page_size as usize;

    loop {
        // ---- Drain newly-submitted requests into the priority heap ----
        let mut disconnected = false;
        loop {
            match request_rx.try_recv() {
                Ok(m) => {
                    pending.push(PendingEntry(PendingReq {
                        prompt: m.prompt,
                        sampling: m.sampling,
                        response_tx: m.response_tx,
                        priority: m.priority,
                        deadline: m.deadline,
                        cancel_flag: m.cancel_flag,
                        arrival_seq: next_seq,
                        skip_emit: 0,
                    }));
                    next_seq += 1;
                }
                Err(tokio::sync::mpsc::error::TryRecvError::Empty) => break,
                Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }

        // ---- Admit from the heap (priority order) with F5 preemption ----
        // Pop the highest-priority waiter; drop expired/invalid ones; preempt
        // lower-priority running slots to free capacity when exhausted. Stop as
        // soon as the top waiter cannot be admitted even with preemption (so a
        // waiter that can't fit never starves the heap in a busy loop).
        while let Some(PendingEntry(req)) = pending.pop() {
            // Client gone before admission — drop silently.
            if req.response_tx.is_closed() {
                continue;
            }
            // Deadline already elapsed in the queue — cancel with a timeout.
            if deadline_expired(req.deadline) {
                fire_deadline_timeout(&req.response_tx, &req.cancel_flag);
                continue;
            }
            // Validate shape (empty / too long / larger than the whole pool) —
            // these are fatal to the request, surfaced as an Err terminal.
            if req.prompt.is_empty() {
                let _ = req.response_tx.blocking_send(Err(EngineError::Engine(
                    "empty prompt (every request needs at least one prompt token)".into(),
                )));
                continue;
            }
            let total_capacity = req
                .prompt
                .len()
                .checked_add(req.sampling.max_tokens as usize)
                .unwrap_or(usize::MAX);
            if total_capacity > ctx.max_ctx {
                let _ = req.response_tx.blocking_send(Err(EngineError::Engine(format!(
                    "prompt too long: {} tokens + max_tokens {} > max_ctx {}",
                    req.prompt.len(),
                    req.sampling.max_tokens,
                    ctx.max_ctx,
                ))));
                continue;
            }
            // F8: a HYBRID slot never touches the shared page pool (its KV +
            // recurrent state are per-slot contiguous), so it needs 0 pages —
            // `make_room` then gates it on a free SLOT alone, not free pages.
            let needed = if ctx.model.weights.is_hybrid() {
                0
            } else {
                total_capacity.div_ceil(page_size.max(1))
            };
            if needed > ctx.total_pages as usize {
                let _ = req.response_tx.blocking_send(Err(EngineError::Engine(format!(
                    "request needs {needed} pages but the paged pool holds only {} \
                     (shorten the prompt or lower max_tokens)",
                    ctx.total_pages,
                ))));
                continue;
            }
            // Make room (free slot + pages), preempting lower-priority slots.
            if !make_room(&ctx, &mut active, &mut pending, needed, req.priority) {
                // Can't schedule the top waiter yet — requeue it and stop; it
                // retries next tick once a running slot frees pages.
                pending.push(PendingEntry(req));
                break;
            }
            match admit_request(&ctx, &req.prompt, &req.sampling, eos, &active, req.skip_emit) {
                Ok(state) => {
                    active.push(DriverSlot {
                        state,
                        response_tx: req.response_tx,
                        priority: req.priority,
                        deadline: req.deadline,
                        cancel_flag: req.cancel_flag,
                        arrival_seq: req.arrival_seq,
                    });
                    ctx.active_slot_count
                        .store(active.len(), std::sync::atomic::Ordering::Release);
                }
                Err(e) => {
                    // Shouldn't happen after `make_room` guaranteed capacity;
                    // surface defensively rather than panic.
                    let _ = req.response_tx.blocking_send(Err(e));
                }
            }
        }

        // ---- Shutdown / idle ----
        if active.is_empty() {
            // Nothing running and nothing admittable. If the senders are all
            // gone, exit; otherwise block for the next submit (no CPU spin).
            if disconnected && pending.is_empty() {
                return;
            }
            if pending.is_empty() {
                let msg = match request_rx.blocking_recv() {
                    Some(m) => m,
                    None => return, // all senders dropped
                };
                pending.push(PendingEntry(PendingReq {
                    prompt: msg.prompt,
                    sampling: msg.sampling,
                    response_tx: msg.response_tx,
                    priority: msg.priority,
                    deadline: msg.deadline,
                    cancel_flag: msg.cancel_flag,
                    arrival_seq: next_seq,
                    skip_emit: 0,
                }));
                next_seq += 1;
            }
            // Loop back to admit the (now non-empty) heap. `pending` non-empty
            // with `active` empty means every waiter was just requeued as
            // un-admittable — which can only happen when `needed > total_pages`
            // (already rejected above), so this does not spin.
            continue;
        }

        // ---- Deadline sweep over running slots (mid-generation timeout) ----
        for slot in active.iter_mut() {
            if !slot.state.done && deadline_expired(slot.deadline) {
                fire_deadline_timeout(&slot.response_tx, &slot.cancel_flag);
                slot.state.done = true;
            }
        }

        // ---- Advance all slots one tick (the single ragged path). ----
        // T1: the driver has ONE execution path. ONE fused forward per tick
        // mixes budgeted prompt-prefill chunks from the still-prefilling slots
        // with every decode-ready slot's 1-query row
        // ([`run_ragged_tick`] → [`LlamaModel::forward_ragged_paged_f32`]). An
        // unset/zero batched-token budget resolves to `usize::MAX`
        // ([`resolve_max_batched_tokens`]), so the whole prompt is prefilled in
        // ONE ragged tick — byte-identical in generated tokens to the former
        // budget-0 inline prefill-then-decode pair (pinned by the parity
        // tests). A positive budget just chunks the prefill across ticks.
        run_ragged_tick(&ctx, &mut active, vocab);

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
                    cache_hit_tokens: slot.state.cache_hit_tokens,
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

/// F8: build the per-slot state for one HYBRID (DeltaNet/SSM) request.
///
/// Unlike the dense/MoE path, a hybrid slot does NOT touch the shared paged
/// pool (the `cache` field is an empty placeholder whose page ops no-op).
/// It carries its own contiguous F32 `KvCache` + its own `DeltaNetCache`,
/// and is prefilled + decoded through the SAME `forward_one_hybrid` the
/// contiguous `CpuEngine` uses — so the resulting token stream is
/// BIT-IDENTICAL to the contiguous engine on the same prompt + sampling.
///
/// Handoff convention mirrors the contiguous engine EXACTLY (so the streams
/// match): prefill forwards `prompt[..n-1]` (positions `[0, n-1)`), leaving
/// the slot positioned at `next_pos = n-1` with `next_input = prompt[n-1]`;
/// the first decode tick forwards that last prompt token to produce the
/// first output token (`drive_generation`'s `split_at(len-1)` pattern).
/// Prefill is driven TOKEN-BY-TOKEN via `forward_one_hybrid`, matching the
/// contiguous default (`prefill_batched_hybrid` off) so the KV + recurrent
/// state are built identically.
///
/// `skip_emit` is `0` for a fresh request and the already-streamed count for
/// an F5 preempted resume (regenerated deterministically, not re-streamed).
fn build_hybrid_slot_state(
    model: &LlamaModel,
    tokenizer: Option<&Arc<Tokenizer>>,
    max_ctx: usize,
    prompt: &[i32],
    sampling: &SamplingParams,
    eos: Option<i32>,
    skip_emit: u32,
) -> EngineResult<SlotState> {
    debug_assert!(!prompt.is_empty(), "build_hybrid_slot_state: empty prompt");
    let hybrid_layers = model
        .weights
        .hybrid_layers
        .as_ref()
        .expect("build_hybrid_slot_state called on a non-hybrid model — dispatch bug");
    let mut kv = KvCache::new_with_dtype(&model.cfg, max_ctx, KvDtype::F32);
    let mut dn = DeltaNetCache::new_for_hybrid(&model.cfg, hybrid_layers)
        .map_err(|e| EngineError::Engine(format!("hybrid DeltaNet cache init: {e}")))?;
    let capacity = kv.max_ctx as u32;
    let vocab = model.cfg.vocab_size;

    // Prefill `prompt[..n-1]` token-by-token (contiguous-engine default path),
    // advancing BOTH the contiguous KV and the per-slot DeltaNet state. The
    // last prompt token is held back as `next_input` for the first decode.
    let prefill_start = std::time::Instant::now();
    let n = prompt.len();
    if n > 1 {
        let mut scratch = vec![0f32; vocab];
        for (pos, &tok) in prompt[..n - 1].iter().enumerate() {
            model.forward_one_hybrid(tok, pos as u32, &mut kv, &mut dn, &mut scratch);
        }
    }
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    let last = *prompt.last().expect("non-empty prompt");
    let tokens_prefilled = n as u32;
    let next_pos = (n - 1) as u32;
    let sampler = Sampler::new(sampling.clone());
    let grammar = build_paged_grammar(model, tokenizer, sampling);
    let stop_strings = sampling.stop.clone();
    let history: Vec<u32> = prompt.iter().map(|&t| t as u32).collect();

    // An empty placeholder paged cache: hybrid KV lives in `hybrid.kv`, so the
    // paged `cache` never acquires pages and its `release_shared` is a no-op.
    let geom_store = PagedKvStore::new(
        1,
        model.cfg.n_layers as u32,
        model.cfg.n_kv_heads as u32,
        DEFAULT_PAGE_SIZE,
        model.cfg.head_dim as u32,
    )
    .ok_or_else(|| EngineError::Engine("hybrid slot geom store: zero dim".into()))?;
    let cache = PagedKvCache::new_for(&geom_store);

    Ok(SlotState {
        sampling: sampling.clone(),
        eos,
        cache,
        hybrid: Some(HybridSlotState { kv, dn, capacity }),
        sampler,
        history,
        next_pos,
        next_input: last,
        n_emitted: 0,
        done: false,
        prefill_ms,
        decode_ms: 0.0,
        tokens_prefilled,
        // No paged shared-prefix for hybrid — the recurrent state can't be
        // prefix-shared (no per-position addressing), so nothing is reused.
        cache_hit_tokens: 0,
        prefill_job: None,
        grammar,
        stop_strings,
        stop_buf: String::new(),
        skip_emit,
        emitted_to_client: skip_emit,
        spec_rng: None,
    })
}

/// Prefill + slot-state-build for one admitted request. Returns
/// the slot's initial decoder state (positioned at `prompt.len()`,
/// ready for the first fused decode tick to produce token #0 of
/// the response).
///
/// `skip_emit` is `0` for a fresh request and `n_emitted` for an F5 preempted
/// resume (the number of already-streamed tokens the deterministic replay
/// regenerates but must NOT re-stream). Borrows `prompt`/`sampling` so the
/// driver keeps ownership for a requeue when admission has to wait.
fn admit_request(
    ctx: &DriverContext,
    prompt: &[i32],
    sampling: &SamplingParams,
    eos: Option<i32>,
    active: &[DriverSlot],
    skip_emit: u32,
) -> EngineResult<SlotState> {
    if prompt.is_empty() {
        return Err(EngineError::Engine(
            "empty prompt (every request needs at least one prompt token)".into(),
        ));
    }
    // E2: checked add in usize (see the sibling site) — a client `max_tokens`
    // near u32::MAX would otherwise wrap past this guard in release.
    let total_capacity = prompt
        .len()
        .checked_add(sampling.max_tokens as usize)
        .unwrap_or(usize::MAX);
    if total_capacity > ctx.max_ctx {
        return Err(EngineError::Engine(format!(
            "prompt too long: {} tokens + max_tokens {} > max_ctx {}",
            prompt.len(),
            sampling.max_tokens,
            ctx.max_ctx,
        )));
    }

    // F8: HYBRID (DeltaNet/SSM) models bypass the paged pool entirely —
    // per-slot contiguous KV + DeltaNet state, prefilled + decoded through the
    // contiguous hybrid forward. No shared-prefix / APC / page allocation.
    if ctx.model.weights.is_hybrid() {
        return build_hybrid_slot_state(
            &ctx.model,
            ctx.tokenizer.as_ref(),
            ctx.max_ctx,
            prompt,
            sampling,
            eos,
            skip_emit,
        );
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
    // Two sources of shared prefix, both page-aligned:
    //  (1) live-slot LCP — share with a currently-active donor's pages.
    //  (2) vLLM APC — content-addressed reuse across ALL prior requests,
    //      INCLUDING completed ones whose pages were freed but not yet
    //      physically reused (`PageTable::reacquire` revives their retained KV).
    // The APC lookup acquires ref-held pages up front; we keep whichever source
    // reuses more, releasing the other's refs.
    let (live_pages, live_lcp) = pick_shared_prefix(prompt, active, page_size);
    let prompt_u32: Vec<u32> = prompt.iter().map(|&t| t as u32).collect();
    let chain = rustllama_models::apc::block_hash_chain(&prompt_u32, page_size);
    let apc_pages = ctx.shared.apc_acquire_prefix(&chain);
    let apc_lcp = apc_pages.len() * page_size;

    let (shared_pages, shared_lcp) = if apc_lcp > 0 && apc_lcp >= live_lcp {
        // APC wins — its pages are already ref-held by apc_acquire_prefix.
        (apc_pages, apc_lcp)
    } else {
        // Live-slot wins (or no APC hit): drop the speculative APC refs and
        // add_ref the live donor's pages (the historical path's semantics).
        if !apc_pages.is_empty() {
            ctx.shared.with_table_mut(|t| t.free(&apc_pages));
        }
        if !live_pages.is_empty() {
            ctx.shared.with_table_mut(|t| t.add_ref(&live_pages));
        }
        (live_pages, live_lcp)
    };
    if !shared_pages.is_empty() {
        cache.pin_shared_prefix(shared_pages, shared_lcp as u32);
    }

    cache
        // E2: `total_capacity` is a usize now (checked add); it is
        // `<= ctx.max_ctx` here, which fits u32, so the cast can't truncate.
        .ensure_capacity_shared(&ctx.shared, total_capacity as u32)
        .map_err(|short| {
            EngineError::Engine(format!(
                "paged batch admit: pool short by {short} pages \
                 (max_slots={} concurrent at max_ctx={})",
                ctx.max_slots, ctx.max_ctx,
            ))
        })?;

    // T1: a non-empty suffix ALWAYS defers prefill to the driver's ragged tick.
    // The slot goes active in a PREFILLING state; `run_ragged_tick` drains the
    // suffix a budgeted chunk at a time (whole suffix in one tick when the
    // budget is unbounded), registering the prompt's APC blocks + finalizing
    // timing on completion. There is no longer an inline-prefill path: the
    // budget-0 branch (which prefilled the whole prompt here before the slot
    // went active) is gone, so `forward_prefill_paged_f32` is no longer called
    // from `admit_request`.
    if shared_lcp < prompt.len() {
        let suffix: Vec<i32> = prompt[shared_lcp..].to_vec();
        let last = *prompt.last().expect("non-empty prompt");
        let tokens_prefilled = prompt.len() as u32;
        let sampler = Sampler::new(sampling.clone());
        // P9 feature parity: grammar + string stops per slot.
        let grammar = build_paged_grammar(&ctx.model, ctx.tokenizer.as_ref(), sampling);
        let stop_strings = sampling.stop.clone();
        return Ok(SlotState {
            sampling: sampling.clone(),
            eos,
            cache,
            hybrid: None,
            sampler,
            history: prompt_u32,
            next_pos: tokens_prefilled, // finalized (unchanged) when prefill drains
            next_input: last,
            n_emitted: 0,
            done: false,
            prefill_ms: 0.0, // set from `started` on completion
            decode_ms: 0.0,
            tokens_prefilled,
            cache_hit_tokens: shared_lcp as u32,
            prefill_job: Some(PrefillJob {
                suffix,
                cursor: 0,
                base: shared_lcp,
                chain,
                prompt_len: tokens_prefilled,
                cache_hit_tokens: shared_lcp as u32,
                started: std::time::Instant::now(),
            }),
            grammar,
            stop_strings,
            stop_buf: String::new(),
            skip_emit,
            emitted_to_client: skip_emit,
            spec_rng: None,
        });
    }

    // Whole prompt satisfied from shared pages (`shared_lcp == prompt.len()` —
    // the early return above handled every non-empty suffix): no prefill work,
    // so the slot is decode-ready immediately and picks up at pos = prompt.len()
    // from the shared K/V on its first ragged decode row.
    debug_assert_eq!(
        shared_lcp,
        prompt.len(),
        "admit_request fell through with an un-prefilled suffix"
    );
    // Register the prompt's full blocks in the APC index — the positions are all
    // resident in the shared pages, block `i` in page `i` — so a later request
    // with the same prefix content reacquires this KV instead of recomputing it,
    // even after this request completes and frees its pages.
    if !chain.is_empty() {
        let pages = cache.pages();
        let n = chain.len().min(pages.len());
        let entries: Vec<_> = (0..n).map(|i| (chain[i], pages[i])).collect();
        ctx.shared.apc_register(&entries);
    }
    let last = *prompt.last().expect("non-empty prompt");
    let tokens_prefilled = prompt.len() as u32;
    let next_pos = tokens_prefilled;
    let sampler = Sampler::new(sampling.clone());
    // P9 feature parity: grammar + string stops per slot.
    let grammar = build_paged_grammar(&ctx.model, ctx.tokenizer.as_ref(), sampling);
    let stop_strings = sampling.stop.clone();
    let history: Vec<u32> = prompt_u32;
    Ok(SlotState {
        sampling: sampling.clone(),
        eos,
        cache,
        hybrid: None,
        sampler,
        history,
        next_pos,
        next_input: last,
        n_emitted: 0,
        done: false,
        prefill_ms: 0.0, // whole prompt was cached — no prefill compute
        decode_ms: 0.0,
        tokens_prefilled,
        cache_hit_tokens: shared_lcp as u32,
        prefill_job: None,
        grammar,
        stop_strings,
        stop_buf: String::new(),
        skip_emit,
        emitted_to_client: skip_emit,
        spec_rng: None,
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

/// Which way a slot participates in a ragged tick (Feature 3b).
#[derive(Clone, Copy)]
enum RaggedKind {
    /// Still-prefilling slot contributing `chunk` prompt tokens this tick.
    Prefill { chunk: usize },
    /// Decode-ready slot contributing its single next-query row.
    Decode,
}

/// The SINGLE driver tick (T1): ONE ragged mixed prefill+decode forward per
/// tick. It plans a single batch that mixes budgeted prompt-prefill CHUNKS from
/// the still-prefilling slots with the 1-query decode rows of every
/// decode-ready slot, issues ONE [`LlamaModel::forward_ragged_paged_f32`], then
/// does all the per-slot bookkeeping: prefill-chunk cursor advance + on-drain
/// APC registration/timing, and decode sampling + token streaming + stop
/// conditions. (This replaced the old budget-gated prefill-tick + decode-tick
/// pair, which the budget-0 inline path used before T1.)
///
/// Scheduling note: a slot that DRAINS its last prefill chunk this tick
/// produces no token this tick — its final prompt KV is only written by THIS
/// forward, so it becomes decode-ready and emits its first token on the NEXT
/// tick's decode row. That is a one-tick latency shift, not a token change: the
/// generated SEQUENCE is identical regardless of the prefill budget (an
/// unbounded budget prefills the whole prompt in one tick) — pinned by the
/// ragged≡inline and unbounded≡finite identity tests. The token budget bounds
/// only the (expensive) prefill work; decode rows are always included so every
/// running slot keeps advancing.
fn run_ragged_tick(ctx: &DriverContext, active: &mut [DriverSlot], vocab: usize) {
    use rustllama_models::llama_arch::RaggedSlot;
    // `PagedKvCache` is already imported at module scope.

    // F8: HYBRID (DeltaNet/SSM) models use a dedicated per-slot tick — their
    // recurrent state can't be cross-slot-fused into the ragged paged forward,
    // and each slot carries its own contiguous KV + DeltaNet state. The dense /
    // MoE path below is untouched (byte-identical) for non-hybrid models.
    if ctx.model.weights.is_hybrid() {
        run_hybrid_tick(ctx, active, vocab);
        return;
    }

    let m = active.len();

    // Prompt-cancellation propagation: a closed response channel ⇒ mark the
    // slot done before paying for the forward, so the retain pass frees its
    // pages for queued requests.
    for i in 0..m {
        if !active[i].state.done && active[i].response_tx.is_closed() {
            active[i].state.done = true;
        }
    }

    // F7: MTP/tree speculation is a MODEL-LEVEL property — a dense (non-hybrid)
    // model carrying an MTP draft head (`mtp_heads`). When set, a decode-ready,
    // grammar-free slot runs a per-slot speculative round INSTEAD of a plain
    // decode row (handled after the ragged forward). A grammar slot keeps the
    // classic decode row (spec is disabled under a grammar, matching the
    // contiguous path). SWA / softcap models are excluded: the paged tree
    // verify (like the contiguous `forward_tree_verify_f32`) uses the plain
    // ancestor-masked attention without window/softcap, and MTP models are
    // neither — so fail-closed to classic decode if they ever co-occur. When
    // `spec_enabled` is false (every current model, incl. all test GGUFs), the
    // planning + forward below are byte-identical to the pre-F7 path.
    let spec_enabled = !ctx.model.weights.is_hybrid()
        && ctx.model.weights.mtp_heads.is_some()
        && ctx.model.cfg.sliding_window.is_none()
        && ctx.model.cfg.attn_logit_softcapping.is_none();

    // --- Plan the ragged batch (slot order == concatenated token order) ---
    // Each non-done slot contributes EITHER a budgeted prefill chunk (if still
    // prefilling) OR one decode row (if decode-ready). A still-prefilling slot
    // that gets 0 budget this tick contributes nothing and stays prefilling.
    // Decode rows do not consume the prefill budget — the budget caps only the
    // prompt-prefill tokens.
    let mut budget = ctx.max_batched_tokens;
    let mut tokens: Vec<i32> = Vec::new();
    // plan entry: (active slot index, kind, n_query, start_pos).
    let mut plan: Vec<(usize, RaggedKind, usize, u32)> = Vec::new();
    // F7: decode-ready, spec-eligible slots handled by a per-slot speculative
    // round after the ragged forward (not added to the batched decode plan).
    let mut spec_slots: Vec<usize> = Vec::new();
    for i in 0..m {
        if active[i].state.done {
            continue;
        }
        if let Some(job) = active[i].state.prefill_job.as_ref() {
            let remaining = job.suffix.len() - job.cursor;
            let chunk = budget.min(remaining);
            if chunk == 0 {
                continue; // out of budget this tick; slot stays prefilling
            }
            let start_pos = (job.base + job.cursor) as u32;
            tokens.extend_from_slice(&job.suffix[job.cursor..job.cursor + chunk]);
            plan.push((i, RaggedKind::Prefill { chunk }, chunk, start_pos));
            budget -= chunk;
        } else if spec_enabled && active[i].state.grammar.is_none() {
            // F7: decode-ready spec slot — run its MTP/tree round separately.
            spec_slots.push(i);
        } else {
            tokens.push(active[i].state.next_input);
            plan.push((i, RaggedKind::Decode, 1, active[i].state.next_pos));
        }
    }
    if plan.is_empty() && spec_slots.is_empty() {
        // Nothing to advance (e.g. every prefilling slot was starved of budget
        // — can't happen while budget > 0 and any slot is prefilling, but guard
        // anyway so the driver loop makes forward progress). F7: when only
        // spec slots are pending, `plan` is empty but `spec_slots` is not, so
        // we fall through to the spec pass below (the ragged forward is a no-op
        // on the empty plan).
        return;
    }

    // Per-decode-row logits buffers (prefill chunks discard logits → empty
    // slice, which the forward reads as "skip the LM head for this slot").
    let mut logits_store: Vec<Vec<f32>> = plan
        .iter()
        .map(|(_, kind, _, _)| match kind {
            RaggedKind::Decode => vec![0f32; vocab],
            RaggedKind::Prefill { .. } => Vec::new(),
        })
        .collect();

    // Build the ragged descriptor slice with by-index disjoint borrows: each
    // slot participates at most once, so the `Option::take` trick hands out one
    // `&mut PagedKvCache` per slot; `logits_store.iter_mut()` gives each row its
    // own `&mut [f32]`.
    let mut cache_refs: Vec<Option<&mut PagedKvCache>> =
        active.iter_mut().map(|s| Some(&mut s.state.cache)).collect();
    let mut ragged: Vec<RaggedSlot> = Vec::with_capacity(plan.len());
    for ((i, _kind, n_query, start_pos), logits_slot) in plan.iter().zip(logits_store.iter_mut()) {
        let cache = cache_refs[*i]
            .take()
            .expect("each slot participates in the ragged batch at most once");
        ragged.push(RaggedSlot {
            n_query: *n_query,
            start_pos: *start_pos,
            cache,
            logits_out: logits_slot.as_mut_slice(),
        });
    }

    let tick_start = std::time::Instant::now();
    ctx.model
        .forward_ragged_paged_f32(&tokens, &mut ragged, &ctx.shared, None);
    let tick_ms = tick_start.elapsed().as_secs_f64() * 1000.0;
    // Release the borrows held through the forward before touching `active`.
    drop(ragged);
    drop(cache_refs);

    // Amortize the forward's wall time across the decode rows it produced
    // (per-slot decode_ms accounting; prefill timing is set from the job's
    // `started` on drain, below).
    let n_decode = plan
        .iter()
        .filter(|e| matches!(e.1, RaggedKind::Decode))
        .count();
    let per_slot_ms = if n_decode > 0 {
        tick_ms / n_decode as f64
    } else {
        0.0
    };

    // --- Per-slot post-processing ---
    for (p, &(i, kind, _n_query, _start_pos)) in plan.iter().enumerate() {
        match kind {
            RaggedKind::Prefill { chunk } => {
                // Advance the cursor; on drain, register the prompt's APC blocks
                // (deferred from admit) + finalize prefill timing + clear the
                // job so the next tick decodes the slot.
                let slot = &mut active[i];
                let drained = {
                    let job = slot
                        .state
                        .prefill_job
                        .as_mut()
                        .expect("prefill plan entry without a prefill job");
                    job.cursor += chunk;
                    job.cursor >= job.suffix.len()
                };
                if drained {
                    let pages = slot.state.cache.pages();
                    let job = slot
                        .state
                        .prefill_job
                        .take()
                        .expect("prefill job present on drain");
                    if !job.chain.is_empty() {
                        let k = job.chain.len().min(pages.len());
                        let entries: Vec<_> =
                            (0..k).map(|idx| (job.chain[idx], pages[idx])).collect();
                        ctx.shared.apc_register(&entries);
                    }
                    slot.state.prefill_ms = job.started.elapsed().as_secs_f64() * 1000.0;
                    let _ = (job.prompt_len, job.cache_hit_tokens); // already on SlotState
                }
            }
            RaggedKind::Decode => {
                active[i].state.decode_ms += per_slot_ms;
                let logits = logits_store[p].as_mut_slice();
                // Shared with the F8 hybrid tick: sample + grammar advance +
                // string stops + F5 skip-emit streaming + stop conditions.
                commit_decode_token(ctx, &mut active[i], logits);
            }
        }
    }

    // F7: per-slot MTP/tree speculative rounds for the decode-ready,
    // grammar-free slots of a draft-head model. Serial per slot (the contiguous
    // engine also speculates one request at a time); the batched prefill/decode
    // fusion above still covers every non-spec slot. Empty unless `spec_enabled`.
    for &i in &spec_slots {
        if active[i].state.done {
            continue;
        }
        run_spec_round_for_slot(ctx, &mut active[i], vocab);
    }
}

/// Sample one decode token from `logits` into `slot` and apply all the
/// per-token bookkeeping: P9 grammar-masked sample + grammar advance, history
/// push, position/emit counters, P9 string-stop detection, F5 skip-emit
/// streaming suppression, and the stop conditions (max_tokens / EOS /
/// capacity / string-stop → `done`). Shared VERBATIM between the dense ragged
/// decode row and the F8 hybrid tick so both behave identically — the only
/// difference between callers is how `logits` was produced (batched paged
/// forward vs per-slot `forward_one_hybrid`). `hit_cap` uses
/// [`SlotState::effective_capacity_tokens`] so it is correct for both the
/// paged `cache` and a hybrid slot's contiguous KV.
fn commit_decode_token(ctx: &DriverContext, slot: &mut DriverSlot, logits: &mut [f32]) {
    // P9 feature parity: grammar-masked sample + grammar advance, identical to
    // `CpuEngine::drive_generation`.
    let next_tok = {
        let SlotState {
            sampler,
            history,
            grammar,
            ..
        } = &mut slot.state;
        let tok = sampler.sample_with_grammar(logits, history, grammar.as_ref());
        if let Some(g) = grammar.as_mut() {
            g.advance(tok);
        }
        tok
    };
    slot.state.history.push(next_tok);
    slot.state.next_input = next_tok as i32;
    slot.state.next_pos += 1;
    slot.state.n_emitted += 1;

    // P9 string-stop detection: accumulate the detok buffer like the contiguous
    // CpuEngine (`decode_single(id, true)`) and match, so a stop string
    // spanning several tokens is caught. The stop-completing token is still
    // streamed (below), then the slot ends — mirroring `drive_generation`'s
    // "emit then break".
    let hit_stop = if slot.state.stop_strings.is_empty() {
        false
    } else {
        let piece = ctx
            .tokenizer
            .as_ref()
            .and_then(|t| t.decode_single(next_tok, true).ok())
            .unwrap_or_default();
        slot.state.stop_buf.push_str(&piece);
        let buf = &slot.state.stop_buf;
        slot.state
            .stop_strings
            .iter()
            .any(|s| !s.is_empty() && buf.contains(s.as_str()))
    };

    // F5 recompute-preemption replay: a resumed slot regenerates its already-
    // streamed tokens bit-for-bit (deterministic greedy / seeded forward) but
    // must NOT re-stream them — suppress the send while `skip_emit` counts down.
    if slot.state.skip_emit > 0 {
        slot.state.skip_emit -= 1;
    } else {
        let text = match &ctx.tokenizer {
            Some(tok) => tok.decode(&[next_tok], false).unwrap_or_default(),
            None => String::new(),
        };
        let token = Token {
            id: next_tok,
            text,
            logprobs: None,
        };
        if slot.response_tx.blocking_send(Ok(token)).is_err() {
            slot.state.done = true;
            return;
        }
        slot.state.emitted_to_client += 1;
    }
    let hit_max = slot.state.n_emitted >= slot.state.sampling.max_tokens;
    let hit_eos = slot
        .state
        .eos
        .map(|e| (next_tok as i32) == e)
        .unwrap_or(false);
    let hit_cap = slot.state.next_pos >= slot.state.effective_capacity_tokens();
    if hit_max || hit_eos || hit_cap || hit_stop {
        slot.state.done = true;
    }
}

/// F8 — the single driver tick for a HYBRID (DeltaNet/SSM) model. Each
/// decode-ready slot is advanced ONE token by its OWN `forward_one_hybrid`
/// against its OWN contiguous KV + `DeltaNetCache` ([`HybridSlotState`]), then
/// committed through the shared [`commit_decode_token`] (so sampling / grammar
/// / stops / F5 / streaming behave exactly like the dense path). Slots are
/// INDEPENDENT by construction — slot A's forward reads and writes only slot
/// A's `kv`/`dn`, so interleaving multiple hybrid requests in one tick can
/// never contaminate another slot's recurrent state (the F8 2-slot anchor).
///
/// This is the "ragged tick" for hybrid: the DeltaNet recurrence cannot be
/// cross-slot-fused into one kernel launch the way dense attention can (the
/// recurrent state is per-sequence and has no per-position addressing), so the
/// hybrid form of continuous batching is a per-slot sequential advance under
/// the one shared driver — concurrent requests, per-slot state, full
/// scheduling / priority / deadline / streaming / grammar / stop machinery.
/// There is no prefill-job phase (hybrid prefill runs inline at admission in
/// [`build_hybrid_slot_state`]), so every active slot contributes a decode row.
fn run_hybrid_tick(ctx: &DriverContext, active: &mut [DriverSlot], vocab: usize) {
    let m = active.len();
    // Prompt-cancellation propagation (mirrors the dense tick): a closed
    // response channel ⇒ mark done before paying for the forward.
    for slot in active.iter_mut().take(m) {
        if !slot.state.done && slot.response_tx.is_closed() {
            slot.state.done = true;
        }
    }

    let mut logits = vec![0f32; vocab];
    for slot in active.iter_mut() {
        if slot.state.done {
            continue;
        }
        let tick_start = std::time::Instant::now();
        let next_input = slot.state.next_input;
        let next_pos = slot.state.next_pos;
        {
            let hyb = slot
                .state
                .hybrid
                .as_mut()
                .expect("run_hybrid_tick slot without hybrid state — dispatch bug");
            // Independent per-slot forward: writes KV at `next_pos` into this
            // slot's own contiguous cache and advances this slot's own DeltaNet
            // recurrent state — never touches any other slot.
            ctx.model.forward_one_hybrid(
                next_input,
                next_pos,
                &mut hyb.kv,
                &mut hyb.dn,
                &mut logits,
            );
        }
        slot.state.decode_ms += tick_start.elapsed().as_secs_f64() * 1000.0;
        commit_decode_token(ctx, slot, &mut logits);
    }
}

/// F7 — ONE MTP/tree speculative round for a decode-ready paged slot, the paged
/// analog of the contiguous dense `tree_mtp_round`:
///
///   1. **Tip forward** — [`LlamaModel::forward_one_with_mtp_logits_paged`]
///      writes `next_input`'s K/V at `base_pos` and returns the MAIN
///      distribution + one draft-logit row per MTP head.
///   2. **Tree build** — [`build_mtp_tree`] shapes a draft tree (chain by
///      default → the pinned linear reduction). Bails to a single-token sample
///      (empty-draft accept/reject) when the tree is empty or would overflow.
///   3. **Tree verify** — [`LlamaModel::forward_tree_verify_paged_f32`] scores
///      every node in ONE forward (each node attends prefix + ancestors) over
///      the slot's paged KV, writing node `i` at `base_pos + 1 + i`.
///   4. **Walk + commit** — [`tree_accept_reject`] picks the accepted path;
///      [`PagedKvStore::compact_positions`] moves the accepted path's scattered
///      K/V into contiguous committed slots and `set_seq_len_for_rollback`
///      reclaims the rejected drafts' positions (the paged "free rejected
///      pages"). For a CHAIN tree the accepted path is already contiguous, so
///      the compaction is a no-op — which is why chain ≡ linear decode.
///
/// Committed tokens stream with the same skip_emit / stop / EOS / max-tokens
/// bookkeeping as the classic decode row. Reuses `build_mtp_tree` /
/// `tree_accept_reject` / `tree_depths` / `accept_reject` (no duplicated spec
/// algorithm). Write-blind: no public dense-MTP GGUF exists on-box, so this
/// runs only against real MTP weights on HW; the forward reductions are pinned
/// at the model layer by `forward_tree_verify_paged_f32_chain_matches_linear`.
fn run_spec_round_for_slot(ctx: &DriverContext, slot: &mut DriverSlot, vocab: usize) {
    use crate::inference::softmax_to_vec;
    use crate::sampling::{apply_all_penalties, Rng};
    use crate::speculative::{
        accept_reject, accept_reject_greedy, build_mtp_tree, tree_accept_reject,
        tree_accept_reject_greedy, tree_depths, DraftToken,
    };

    let n_mtp = ctx
        .model
        .weights
        .mtp_heads
        .as_ref()
        .map(|h| h.len())
        .unwrap_or(0);
    debug_assert!(n_mtp > 0, "spec round entered without MTP heads");

    // Snapshot the scalars we need before taking the mutable cache borrow.
    let base_pos = slot.state.next_pos;
    let next_input = slot.state.next_input;
    let greedy = slot.state.sampling.temperature <= 0.0;
    let repeat = slot.state.sampling.repeat_penalty;
    let frequency = slot.state.sampling.frequency_penalty;
    let presence = slot.state.sampling.presence_penalty;
    let seed = slot.state.sampling.seed;
    let max_tokens = slot.state.sampling.max_tokens;
    let penalize = repeat != 1.0 || frequency != 0.0 || presence != 0.0;
    let max_ctx = ctx.max_ctx;

    // --- Tip forward: main distribution + per-head drafts. ---
    let mut main_logits = vec![0f32; vocab];
    let mut head_logits: Vec<Vec<f32>> = vec![vec![0f32; vocab]; n_mtp];
    ctx.model.forward_one_with_mtp_logits_paged(
        next_input,
        base_pos,
        &mut slot.state.cache,
        &ctx.shared,
        &mut main_logits,
        &mut head_logits,
    );

    // Penalty history for the first speculative token. In the paged slot
    // `history` already ends with `next_input` (the tip), so it IS the recent
    // context the contiguous `tree_mtp_round` builds as `last_ids + next_input`.
    let recent: Vec<u32> = if penalize {
        slot.state.history.clone()
    } else {
        Vec::new()
    };
    if penalize {
        apply_all_penalties(&mut main_logits, &recent, repeat, frequency, presence);
    }
    let main_dist = softmax_to_vec(&main_logits);

    // --- Build the draft tree. Default widths = a chain of length n_mtp (the
    //     pinned linear-MTP shape; a wider tree is a config-driven follow-up). ---
    let widths: Vec<usize> = vec![1usize; n_mtp];
    let nodes = build_mtp_tree(&head_logits, &widths, &recent, repeat, frequency, presence);
    let base = base_pos as usize;
    let fits = !nodes.is_empty()
        && base + 1 + nodes.len() <= max_ctx
        && (base + 1 + nodes.len()) as u32 <= slot.state.cache.capacity_tokens();

    // Committed tokens this round + the slot's new (next_input, next_pos).
    let committed: Vec<u32>;
    if !fits {
        // No tree (no draft / overflow) ⇒ commit one token from the tip's main
        // distribution via the empty-draft accept/reject — identical to the
        // classic single-token sample the contiguous path falls back to. The
        // tip's K/V already covers `base_pos`; next write is `base_pos + 1`.
        let no_drafts: [DraftToken; 0] = [];
        let target: [&[f32]; 1] = [main_dist.as_slice()];
        let out = if greedy {
            accept_reject_greedy(&no_drafts, &target)
        } else {
            let rng = slot.state.spec_rng.get_or_insert_with(|| Rng::from_seed(seed));
            accept_reject(&no_drafts, &target, rng)
        };
        committed = vec![out.replacement];
        slot.state.next_input = out.replacement as i32;
        slot.state.next_pos = base_pos + 1;
    } else {
        // --- Tree verify + walk the accepted path. ---
        let node_tokens: Vec<i32> = nodes.iter().map(|n| n.id as i32).collect();
        let depths = tree_depths(&nodes);
        let parent: Vec<i32> = nodes
            .iter()
            .map(|n| n.parent.map(|p| p as i32).unwrap_or(-1))
            .collect();
        let mut tree_logits = vec![0f32; nodes.len() * vocab];
        ctx.model.forward_tree_verify_paged_f32(
            &node_tokens,
            base_pos + 1,
            &depths,
            &parent,
            &mut slot.state.cache,
            &ctx.shared,
            &mut tree_logits,
        );
        let next_dists: Vec<Vec<f32>> = (0..nodes.len())
            .map(|i| softmax_to_vec(&tree_logits[i * vocab..(i + 1) * vocab]))
            .collect();
        let next_refs: Vec<&[f32]> = next_dists.iter().map(|d| d.as_slice()).collect();
        let outcome = if greedy {
            tree_accept_reject_greedy(&nodes, &main_dist, &next_refs)
        } else {
            let rng = slot.state.spec_rng.get_or_insert_with(|| Rng::from_seed(seed));
            tree_accept_reject(&nodes, &main_dist, &next_refs, rng)
        };
        let a = outcome.accepted.len();
        // Commit: compact the accepted path's scattered K/V into the contiguous
        // committed slots `[base_pos + 1, base_pos + 1 + a)`, then shrink the
        // watermark so the rejected drafts' positions are reclaimed.
        let pages: Vec<_> = slot.state.cache.pages().to_vec();
        ctx.shared.with_store_mut(|store| {
            store.compact_positions(&pages, base + 1, &outcome.accepted_nodes);
        });
        slot
            .state
            .cache
            .set_seq_len_for_rollback(base_pos + 1 + a as u32);
        let mut c = outcome.accepted;
        c.push(outcome.replacement);
        committed = c;
        slot.state.next_input = outcome.replacement as i32;
        slot.state.next_pos = base_pos + 1 + a as u32;
    }

    // --- Emit committed tokens with the classic decode bookkeeping. ---
    for &id in &committed {
        slot.state.history.push(id);
        slot.state.n_emitted += 1;

        // String-stop detection (same running-detok-buffer match as the decode
        // row): the stop-completing token is still streamed, then the slot ends.
        let hit_stop = if slot.state.stop_strings.is_empty() {
            false
        } else {
            let piece = ctx
                .tokenizer
                .as_ref()
                .and_then(|t| t.decode_single(id, true).ok())
                .unwrap_or_default();
            slot.state.stop_buf.push_str(&piece);
            let buf = &slot.state.stop_buf;
            slot.state
                .stop_strings
                .iter()
                .any(|s| !s.is_empty() && buf.contains(s.as_str()))
        };

        // F5 replay: suppress streaming regenerated tokens while skip_emit > 0.
        if slot.state.skip_emit > 0 {
            slot.state.skip_emit -= 1;
        } else {
            let text = match &ctx.tokenizer {
                Some(tok) => tok.decode(&[id], false).unwrap_or_default(),
                None => String::new(),
            };
            let token = Token {
                id,
                text,
                logprobs: None,
            };
            if slot.response_tx.blocking_send(Ok(token)).is_err() {
                slot.state.done = true;
                return;
            }
            slot.state.emitted_to_client += 1;
        }

        let hit_max = slot.state.n_emitted >= max_tokens;
        let hit_eos = slot.state.eos.map(|e| (id as i32) == e).unwrap_or(false);
        let hit_cap = slot.state.next_pos >= slot.state.cache.capacity_tokens();
        if hit_max || hit_eos || hit_cap || hit_stop {
            slot.state.done = true;
            return;
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

    /// F6-A: the paged VLM guard's combined-file detector must NOT fire on an
    /// ordinary text GGUF (a false positive would wrongly refuse every normal
    /// fused-decode model) and MUST fire on a vision GGUF (clip `general.
    /// architecture`, `clip.*` metadata, `v.*` tensors). The common two-file VLM
    /// is caught earlier by the serve-time `[model].mmproj` check; this detector
    /// is the combined single-file defense-in-depth.
    #[test]
    fn vision_indicator_detects_clip_not_text() {
        use rustllama_gguf::synth::{
            write_synthetic_clip_mmproj_gguf, write_synthetic_llama_gguf, SynthLlama,
        };
        let text = std::env::temp_dir().join("rustllama-f6a-text-guard.gguf");
        write_synthetic_llama_gguf(&text, &SynthLlama::default());
        let text_gguf = Gguf::open(&text).expect("open synth text gguf");
        assert!(
            !gguf_has_vision_indicator(&text_gguf),
            "ordinary text GGUF must not trip the F6-A VLM guard"
        );

        let clip = std::env::temp_dir().join("rustllama-f6a-clip-guard.gguf");
        write_synthetic_clip_mmproj_gguf(&clip);
        let clip_gguf = Gguf::open(&clip).expect("open synth clip gguf");
        assert!(
            gguf_has_vision_indicator(&clip_gguf),
            "clip mmproj GGUF must trip the F6-A VLM guard"
        );
    }

    // ============================================================
    // P14: priority-ordered admission + deadline enforcement
    // ============================================================

    /// Build a throwaway `PendingReq` with a given priority + arrival order.
    /// The response channel is a live (unused) mpsc sender.
    fn pending(priority: Priority, arrival_seq: u64) -> PendingEntry {
        let (tx, _rx) = tokio::sync::mpsc::channel::<EngineResult<Token>>(4);
        PendingEntry(PendingReq {
            prompt: vec![1, 2, 3],
            sampling: SamplingParams::default(),
            response_tx: tx,
            priority,
            deadline: None,
            cancel_flag: None,
            arrival_seq,
            skip_emit: 0,
        })
    }

    /// P14 admit order: the driver pops the HIGHEST-priority waiter first,
    /// regardless of arrival order — so a high-priority request admits before a
    /// low-priority one that was queued earlier. This is the core of P14: the
    /// `X-RustLlama-Priority` value drives real admission order, not FIFO.
    #[test]
    fn pending_heap_pops_highest_priority_first() {
        let mut heap: BinaryHeap<PendingEntry> = BinaryHeap::new();
        // Queue LOW first (arrival 0), then HIGH (arrival 1), then DEFAULT (2).
        heap.push(pending(Priority::LOW, 0));
        heap.push(pending(Priority::HIGH, 1));
        heap.push(pending(Priority::DEFAULT, 2));
        // Even though LOW arrived first, HIGH pops first, then DEFAULT, then LOW.
        assert_eq!(heap.pop().unwrap().0.priority, Priority::HIGH);
        assert_eq!(heap.pop().unwrap().0.priority, Priority::DEFAULT);
        assert_eq!(heap.pop().unwrap().0.priority, Priority::LOW);
        assert!(heap.pop().is_none());
    }

    /// P14 FIFO tiebreak: within one priority level, the OLDER arrival_seq pops
    /// first — equal-priority requests keep strict arrival order.
    #[test]
    fn pending_heap_is_fifo_within_equal_priority() {
        let mut heap: BinaryHeap<PendingEntry> = BinaryHeap::new();
        // Push newest-first to prove ordering isn't just insertion order.
        heap.push(pending(Priority::DEFAULT, 2));
        heap.push(pending(Priority::DEFAULT, 0));
        heap.push(pending(Priority::DEFAULT, 1));
        assert_eq!(heap.pop().unwrap().0.arrival_seq, 0);
        assert_eq!(heap.pop().unwrap().0.arrival_seq, 1);
        assert_eq!(heap.pop().unwrap().0.arrival_seq, 2);
    }

    /// P14 mixed: higher priority always wins; FIFO only breaks ties.
    #[test]
    fn pending_heap_priority_then_fifo() {
        let mut heap: BinaryHeap<PendingEntry> = BinaryHeap::new();
        heap.push(pending(Priority::DEFAULT, 0)); // older default
        heap.push(pending(Priority::DEFAULT, 1)); // newer default
        heap.push(pending(Priority::HIGH, 2)); // newest but highest
        let a = heap.pop().unwrap().0;
        assert_eq!(a.priority, Priority::HIGH, "highest priority first");
        assert_eq!(heap.pop().unwrap().0.arrival_seq, 0, "then oldest default");
        assert_eq!(heap.pop().unwrap().0.arrival_seq, 1, "then newer default");
    }

    /// P14 deadline: `fire_deadline_timeout` flips the request's cancel flag
    /// (the `CancelGuard.flag` mechanism) AND emits a `DeadlineExceeded`
    /// terminal so the consumer can report a `"timeout"` finish reason. This is
    /// exactly what the driver does when a request passes its deadline. Run as
    /// a plain `#[test]` (NOT `#[tokio::test]`) because `fire_deadline_timeout`
    /// uses `blocking_send` — valid on the driver's std thread, but a panic if
    /// called from inside a tokio runtime. The end-to-end driver path is
    /// covered by `paged_batch_deadline_cancels_with_timeout_and_flips_flag`.
    #[test]
    fn deadline_timeout_flips_flag_and_emits_terminal() {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<EngineResult<Token>>(4);
        let flag = Arc::new(AtomicBool::new(false));
        fire_deadline_timeout(&tx, &Some(flag.clone()));
        assert!(
            flag.load(std::sync::atomic::Ordering::Acquire),
            "deadline must flip the cancel flag"
        );
        let item = rx.blocking_recv().expect("terminal item");
        assert!(
            matches!(item, Err(EngineError::DeadlineExceeded)),
            "deadline terminal must be EngineError::DeadlineExceeded (the timeout reason)"
        );
    }

    /// `deadline_expired` is the per-request timeout predicate: past deadlines
    /// fire, future deadlines + `None` do not.
    #[test]
    fn deadline_expired_predicate() {
        assert!(!deadline_expired(None));
        assert!(deadline_expired(Some(
            Instant::now() - std::time::Duration::from_millis(1)
        )));
        assert!(!deadline_expired(Some(
            Instant::now() + std::time::Duration::from_secs(60)
        )));
    }
}
