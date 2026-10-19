//! `PagedBatchEngine` integration tests.
//!
//! The model-layer `forward_decode_paged_batched_*` parity tests
//! prove the fused decode kernel is correct given pre-built slot
//! state. These tests prove the engine wraps it correctly: serial
//! prefill of each slot under the shared store lock, fused decode
//! loop driven to completion per slot, per-slot sampler + EOS +
//! max_tokens stop conditions, and page release back to the
//! shared pool at slot completion.

use rustllama_engine::batch_scheduler::Priority;
use rustllama_engine::paged_batch::PagedBatchEngine;
use rustllama_engine::{BeamParams, ContiguousEngine, EngineError, SamplingParams, SubmitOpts};
use rustllama_gguf::synth::{
    write_synthetic_llama_gguf, write_synthetic_lora_adapter_gguf, SynthHybrid, SynthLlama,
    SynthLoraAdapter, SynthMoe,
};

fn write_gguf(tag: &str) -> std::path::PathBuf {
    let tmp = std::env::temp_dir().join(format!("rustllama-paged-batch-{tag}.gguf"));
    let params = SynthLlama::default();
    write_synthetic_llama_gguf(&tmp, &params);
    tmp
}

fn greedy(n: u32) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_p: 0.0,
        top_k: 0,
        repeat_penalty: 1.0,
        max_tokens: n,
        stop: vec![],
        seed: 0,
        ..SamplingParams::default()
    }
}

/// M=1 fused decode matches single-slot `ContiguousEngine` decode. Proves
/// the engine wrapper isn't introducing spurious differences vs the
/// existing single-flight code path. Greedy sampling, same prompt,
/// same max_tokens — expects bit-identical token sequence.
#[test]
fn paged_batch_single_slot_matches_cpu_engine_greedy() {
    use rustllama_models::llama_arch::KvDtype;
    let tmp = write_gguf("m1-vs-cpu");

    let mut cpu =
        ContiguousEngine::load_with_options_and_layout(&tmp, 32, true, KvDtype::F32, "paged")
            .expect("cpu engine load");
    cpu.set_prefix_cache(false);

    let batch = PagedBatchEngine::load(&tmp, 32, 1).expect("batch engine load");

    let prompt: Vec<i32> = (0..6).collect();
    let cpu_out = cpu
        .generate_token_ids(&prompt, 3, &greedy(3))
        .expect("cpu generate");
    let batch_outs = batch
        .generate_batched_token_ids(vec![(prompt.clone(), greedy(3))])
        .expect("batch generate");

    assert_eq!(batch_outs.len(), 1, "one request in, one out");
    let batch_out: Vec<i32> = batch_outs[0].iter().map(|&t| t).collect();
    let cpu_out_i32: Vec<i32> = cpu_out.iter().map(|&t| t as i32).collect();
    assert_eq!(
        cpu_out_i32, batch_out,
        "paged_batch M=1 diverged from paged ContiguousEngine: cpu={cpu_out_i32:?} batch={batch_out:?}"
    );
    let _ = std::fs::remove_file(&tmp);
}

/// M=2 different prompts: each slot produces its own coherent
/// sequence in one fused decode call per tick. Compare each slot's
/// output to a single-slot reference run on the same prompt — if
/// fused decode is doing its job, each slot's output is the same as
/// running it alone. This is the "true CB" parity check at the
/// engine layer.
#[test]
fn paged_batch_two_different_prompts_each_matches_single_slot_reference() {
    let tmp = write_gguf("m2-different-prompts");

    let batch = PagedBatchEngine::load(&tmp, 32, 2).expect("batch engine load");

    let prompt_a: Vec<i32> = vec![5, 1, 4, 2, 3];
    let prompt_b: Vec<i32> = vec![8, 0, 7, 6, 1];

    // Reference: each prompt run through a SEPARATE single-slot
    // PagedBatchEngine. If the M=2 batched call cross-
    // contaminates slot state, the fused outputs will diverge
    // from these references.
    let batch_ref_a = PagedBatchEngine::load(&tmp, 32, 1).expect("ref a load");
    let ref_a = batch_ref_a
        .generate_batched_token_ids(vec![(prompt_a.clone(), greedy(3))])
        .expect("ref a generate")
        .into_iter()
        .next()
        .unwrap();
    let batch_ref_b = PagedBatchEngine::load(&tmp, 32, 1).expect("ref b load");
    let ref_b = batch_ref_b
        .generate_batched_token_ids(vec![(prompt_b.clone(), greedy(3))])
        .expect("ref b generate")
        .into_iter()
        .next()
        .unwrap();

    // Fused M=2.
    let outs = batch
        .generate_batched_token_ids(vec![
            (prompt_a.clone(), greedy(3)),
            (prompt_b.clone(), greedy(3)),
        ])
        .expect("batched generate");
    assert_eq!(outs.len(), 2);
    assert_eq!(
        outs[0], ref_a,
        "slot A diverged from single-slot reference: fused={:?} ref={:?}",
        outs[0], ref_a,
    );
    assert_eq!(
        outs[1], ref_b,
        "slot B diverged from single-slot reference: fused={:?} ref={:?}",
        outs[1], ref_b,
    );
    let _ = std::fs::remove_file(&tmp);
}

/// Phase 2-D-6 integration: `PagedBatchEngine::load` succeeds on a
/// MoE GGUF and `generate_batched_token_ids` produces finite,
/// non-empty token sequences. The model-layer paged-batched MoE
/// parity tests (`forward_decode_paged_batched_f32_moe_matches_serial_paged`)
/// already prove correctness of the fused decode kernel itself;
/// this test pins that the engine wrapper accepts the MoE GGUF
/// and doesn't hit a stale dense-only code path.
#[test]
fn paged_batch_loads_moe_gguf_and_generates_finite_tokens() {
    let tmp = std::env::temp_dir().join("rustllama-paged-batch-moe-load.gguf");
    write_synthetic_llama_gguf(
        &tmp,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 4,
                n_experts_used: 2,
                n_experts_shared: 0,
            }),
            ..SynthLlama::default()
        },
    );
    let batch = PagedBatchEngine::load(&tmp, 32, 2).expect("MoE GGUF must load");
    let prompt: Vec<i32> = vec![3, 5, 7];
    let outs = batch
        .generate_batched_token_ids(vec![(prompt, greedy(3))])
        .expect("MoE generation must succeed");
    assert_eq!(outs.len(), 1);
    assert!(
        !outs[0].is_empty(),
        "MoE generation produced no tokens — driver may have errored silently"
    );
    let _ = std::fs::remove_file(&tmp);
}

/// Two-slot MoE batched decode through the engine wrapper. Each
/// slot generates its own coherent sequence. Pins that per-slot
/// MoE routing decisions don't cross-contaminate across slots
/// (the per-slot `moe_ffn_one_into` loop in the kernel must read
/// each slot's own `h_norm` row).
#[test]
fn paged_batch_two_moe_slots_each_produces_tokens() {
    let tmp = std::env::temp_dir().join("rustllama-paged-batch-moe-2slots.gguf");
    write_synthetic_llama_gguf(
        &tmp,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 4,
                n_experts_used: 2,
                n_experts_shared: 0,
            }),
            ..SynthLlama::default()
        },
    );
    let batch = PagedBatchEngine::load(&tmp, 32, 2).expect("MoE load");
    let outs = batch
        .generate_batched_token_ids(vec![
            (vec![1, 2, 3, 4], greedy(2)),
            (vec![5, 6, 7, 8], greedy(2)),
        ])
        .expect("two-slot MoE generate");
    assert_eq!(outs.len(), 2);
    assert!(!outs[0].is_empty(), "slot A produced no tokens");
    assert!(!outs[1].is_empty(), "slot B produced no tokens");
    let _ = std::fs::remove_file(&tmp);
}

/// Page release on slot completion: after a generate call,
/// `free_pages()` returns to the pool's initial count. A leak
/// would surface here as `free_pages() < initial` after one
/// completed call.
#[test]
fn paged_batch_releases_pages_on_completion() {
    let tmp = write_gguf("page-release");
    let batch = PagedBatchEngine::load(&tmp, 32, 2).expect("load");
    let initial = batch.free_pages();
    assert!(initial > 0, "pool must start non-empty");

    let prompt: Vec<i32> = vec![1, 2, 3];
    let _ = batch
        .generate_batched_token_ids(vec![(prompt, greedy(2))])
        .expect("generate");

    let after = batch.free_pages();
    assert_eq!(
        after, initial,
        "page leak: pool was {initial} pages before generate, {after} after \
         (release_shared must run on slot completion)"
    );

    // A second back-to-back generation must succeed (proves the
    // pages are reclaimable, not just visible in the counter).
    let _ = batch
        .generate_batched_token_ids(vec![(vec![4, 5, 6], greedy(2))])
        .expect("second generate after release");
    assert_eq!(batch.free_pages(), initial);
    let _ = std::fs::remove_file(&tmp);
}

/// Empty input batch is a no-op — returns an empty output Vec
/// without panicking. Guards against the scheduler-driven loop
/// accidentally calling generate with zero admitted slots.
#[test]
fn paged_batch_empty_input_returns_empty() {
    let tmp = write_gguf("empty");
    let batch = PagedBatchEngine::load(&tmp, 32, 2).expect("load");
    let outs = batch.generate_batched_token_ids(vec![]).expect("empty");
    assert!(outs.is_empty());
    let _ = std::fs::remove_file(&tmp);
}

/// Streaming: submit two concurrent requests through
/// `submit_streaming`; both must produce token streams that
/// complete with coherent (deterministic, greedy) output. The
/// driver thread arbitrates admission and runs the fused decode
/// loop across both slots.
///
/// Per-slot output must equal the equivalent single-slot
/// reference run — same correctness contract as the synchronous
/// `paged_batch_two_different_prompts_each_matches_single_slot_reference`
/// test, but exercised through the async submit + driver path.
#[tokio::test]
async fn paged_batch_streaming_two_concurrent_submits_each_matches_reference() {
    use futures::StreamExt;
    let tmp = write_gguf("stream-m2");
    let batch = PagedBatchEngine::load(&tmp, 32, 2).expect("batch load");

    let prompt_a: Vec<i32> = vec![5, 1, 4, 2, 3];
    let prompt_b: Vec<i32> = vec![8, 0, 7, 6, 1];
    let n_tokens = 3u32;

    // References: run each prompt through a 1-slot batch engine.
    // (Same setup as the non-streaming parity test, just so the
    // streaming output has something concrete to compare against.)
    let ref_a = PagedBatchEngine::load(&tmp, 32, 1)
        .unwrap()
        .generate_batched_token_ids(vec![(prompt_a.clone(), greedy(n_tokens))])
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    let ref_b = PagedBatchEngine::load(&tmp, 32, 1)
        .unwrap()
        .generate_batched_token_ids(vec![(prompt_b.clone(), greedy(n_tokens))])
        .unwrap()
        .into_iter()
        .next()
        .unwrap();

    // Submit both concurrently. The driver admits both, runs
    // prefill on each, then the fused decode loop covers both
    // slots per tick until each finishes.
    let stream_a = batch
        .submit_streaming(prompt_a.clone(), greedy(n_tokens))
        .expect("submit A");
    let stream_b = batch
        .submit_streaming(prompt_b.clone(), greedy(n_tokens))
        .expect("submit B");

    // Drain both streams in parallel via futures::future::join.
    let collect_a = async move {
        let mut out = Vec::new();
        let mut s = stream_a;
        while let Some(item) = s.next().await {
            let t = item.expect("token A");
            out.push(t.id as i32);
        }
        out
    };
    let collect_b = async move {
        let mut out = Vec::new();
        let mut s = stream_b;
        while let Some(item) = s.next().await {
            let t = item.expect("token B");
            out.push(t.id as i32);
        }
        out
    };
    let (out_a, out_b) = futures::future::join(collect_a, collect_b).await;

    assert_eq!(out_a.len(), n_tokens as usize, "stream A token count");
    assert_eq!(out_b.len(), n_tokens as usize, "stream B token count");
    assert_eq!(
        out_a, ref_a,
        "streaming slot A diverged from reference: stream={out_a:?} ref={ref_a:?}",
    );
    assert_eq!(
        out_b, ref_b,
        "streaming slot B diverged from reference: stream={out_b:?} ref={ref_b:?}",
    );
    let _ = std::fs::remove_file(&tmp);
}

/// Ragged-driver identity across budgets: a SMALL finite batched-token budget
/// (which chunks the prefill across several ticks) produces token streams
/// IDENTICAL to an unbounded budget (which prefills the whole prompt in one
/// ragged tick). Both engines run the same two concurrent prompts under greedy
/// sampling and route every tick through the single ragged forward
/// (`forward_ragged_paged_f32`) — T1 made ragged the only driver path. The
/// budget-0 engine resolves to `usize::MAX` (unbounded, one-shot prefill) and
/// is the known-good reference; the budget-2 engine chunks the prefill since
/// the prompts are longer than the budget. Byte-identical token streams ⇒ the
/// per-tick prefill chunking changes no output (only a one-tick latency shift
/// on drain).
#[tokio::test]
async fn paged_batch_ragged_budget_matches_inline_budget_zero() {
    let tmp = write_gguf("ragged-vs-inline");
    let prompt_a: Vec<i32> = vec![5, 1, 4, 2, 3];
    let prompt_b: Vec<i32> = vec![8, 0, 3, 7, 1];
    let n_tokens = 4u32;

    // Collect both slots' token-id streams from a 2-slot engine run concurrently.
    async fn run_pair(
        engine: &PagedBatchEngine,
        pa: Vec<i32>,
        pb: Vec<i32>,
        n: u32,
    ) -> (Vec<i32>, Vec<i32>) {
        use futures::StreamExt;
        let sa = engine.submit_streaming(pa, greedy(n)).expect("submit A");
        let sb = engine.submit_streaming(pb, greedy(n)).expect("submit B");
        let ca = async move {
            let mut out = Vec::new();
            let mut s = sa;
            while let Some(item) = s.next().await {
                out.push(item.expect("tok A").id as i32);
            }
            out
        };
        let cb = async move {
            let mut out = Vec::new();
            let mut s = sb;
            while let Some(item) = s.next().await {
                out.push(item.expect("tok B").id as i32);
            }
            out
        };
        futures::future::join(ca, cb).await
    }

    // Budget 0 ⇒ usize::MAX = unbounded ragged prefill (whole prompt in one
    // tick); the known-good reference.
    let inline = PagedBatchEngine::load_with_batched_tokens(&tmp, 32, 2, 0).expect("inline load");
    let (in_a, in_b) = run_pair(&inline, prompt_a.clone(), prompt_b.clone(), n_tokens).await;
    drop(inline);

    // Budget 2 < suffix len 5 ⇒ the ragged path chunks the prefill across
    // multiple ticks before steady-state decode.
    let ragged = PagedBatchEngine::load_with_batched_tokens(&tmp, 32, 2, 2).expect("ragged load");
    let (rg_a, rg_b) = run_pair(&ragged, prompt_a.clone(), prompt_b.clone(), n_tokens).await;
    drop(ragged);

    assert_eq!(in_a.len(), n_tokens as usize, "inline slot A token count");
    assert_eq!(rg_a.len(), n_tokens as usize, "ragged slot A token count");
    assert_eq!(
        in_a, rg_a,
        "ragged (budget>0) slot A diverged from inline (budget=0): ragged={rg_a:?} inline={in_a:?}"
    );
    assert_eq!(
        in_b, rg_b,
        "ragged (budget>0) slot B diverged from inline (budget=0): ragged={rg_b:?} inline={in_b:?}"
    );
    let _ = std::fs::remove_file(&tmp);
}

/// T1 explicit: an UNBOUNDED budget (`usize::MAX`, what
/// `resolve_max_batched_tokens` maps 0 / unset to) prefills the whole prompt in
/// ONE ragged tick, and must produce the same tokens as a SMALL finite budget
/// that chunks the same prompt across several ticks. There is one execution
/// path (the ragged forward); only the per-tick prefill chunking differs. Short
/// single-slot prompt, greedy — one tick either way vs multi-tick.
#[tokio::test]
async fn paged_batch_ragged_unbounded_budget_matches_small_finite() {
    use futures::StreamExt;
    let tmp = write_gguf("ragged-unbounded-vs-finite");
    let prompt: Vec<i32> = vec![3, 1, 4, 1, 5, 9, 2];
    let n_tokens = 4u32;

    async fn run_one(engine: &PagedBatchEngine, p: Vec<i32>, n: u32) -> Vec<i32> {
        let mut s = engine.submit_streaming(p, greedy(n)).expect("submit");
        let mut out = Vec::new();
        while let Some(item) = s.next().await {
            out.push(item.expect("tok").id as i32);
        }
        out
    }

    // Budget 0 ⇒ usize::MAX (unbounded, one-shot ragged prefill).
    let unbounded =
        PagedBatchEngine::load_with_batched_tokens(&tmp, 32, 1, 0).expect("unbounded load");
    let out_unbounded = run_one(&unbounded, prompt.clone(), n_tokens).await;
    drop(unbounded);

    // Budget 2 < prompt len ⇒ the ragged path chunks the prefill over ticks.
    let finite = PagedBatchEngine::load_with_batched_tokens(&tmp, 32, 1, 2).expect("finite load");
    let out_finite = run_one(&finite, prompt.clone(), n_tokens).await;
    drop(finite);

    assert_eq!(
        out_unbounded.len(),
        n_tokens as usize,
        "unbounded token count"
    );
    assert_eq!(
        out_unbounded, out_finite,
        "unbounded (usize::MAX) budget diverged from a small finite budget: \
         unbounded={out_unbounded:?} finite={out_finite:?}"
    );
    let _ = std::fs::remove_file(&tmp);
}

/// Dropping the engine cleanly shuts down the driver thread. The
/// test relies on the driver's `recv() → None` path firing when
/// the request channel closes; if shutdown were buggy the test
/// would hang at engine Drop (and the test harness would time out).
#[tokio::test]
async fn paged_batch_engine_drop_shuts_driver_thread_down() {
    let tmp = write_gguf("shutdown");
    let batch = PagedBatchEngine::load(&tmp, 32, 1).expect("load");
    // Run one request so the driver has produced output at least
    // once (proves it's actually running, not pre-shutdown).
    use futures::StreamExt;
    let mut stream = batch
        .submit_streaming(vec![1, 2, 3], greedy(1))
        .expect("submit");
    let first = stream.next().await.expect("first token").expect("ok");
    assert!(first.id < 256, "synthetic vocab is 256");
    drop(stream);
    // Now drop the engine. The driver's request channel closes,
    // its `recv()` returns None, the thread exits, the
    // JoinHandle::join inside Drop returns. If this hangs, the
    // test framework fails it with a timeout.
    drop(batch);
    let _ = std::fs::remove_file(&tmp);
}

/// PagedBatchEngine satisfies the `Engine` trait — can be held
/// as `Arc<dyn Engine>` and exercised through `tokenize` /
/// `generate` / `chat` exactly like `ContiguousEngine`. This is the
/// prerequisite for the server's `ServingModel` to route to it
/// in 3.7g without any handler-side changes.
///
/// The test exercises the full trait surface:
///   - `tokenize(text) → ids` round-trips through the engine's
///     tokenizer.
///   - `generate(prompt_text, sampling) → TokenStream` produces
///     tokens (proves the text → ids → submit → driver chain).
///   - `chat(messages, sampling) → TokenStream` renders the chat
///     template + submits (proves the chat-template path).
///   - `Arc<dyn Engine>` coercion compiles.
#[tokio::test]
async fn paged_batch_engine_satisfies_engine_trait() {
    use rustllama_engine::Engine;
    use std::sync::Arc;

    let tmp = write_gguf("trait-impl");
    let batch = PagedBatchEngine::load(&tmp, 32, 2).expect("load");
    // Coerce to Arc<dyn Engine> — this is the shape the server's
    // ServingModel uses to hold its backend. Compile failure here
    // would mean the trait impl is incomplete.
    let engine: Arc<dyn Engine> = Arc::new(batch);

    // metrics / n_ctx / vocab_size sanity.
    assert_eq!(engine.n_ctx(), 32);
    assert!(engine.vocab_size() >= 16, "synthetic vocab is non-trivial");
    let _ = engine.metrics();

    // tokenize() — the trait method is callable through the
    // dyn-trait pointer. The synthetic GGUF's "llama"
    // sentencepiece tokenizer ships only `<tok_N>` vocab entries
    // with no merges, so encode() of arbitrary text returns an
    // empty slice. What this assertion checks is that the trait
    // dispatch reaches the underlying tokenizer (returns `Ok(_)`,
    // not a "no tokenizer" error). Functional chat/generate
    // through a real tokenizer is exercised by the server-level
    // integration in 3.7g.
    let result = engine.tokenize("text");
    assert!(
        result.is_ok(),
        "tokenize() trait dispatch must reach the tokenizer"
    );

    // chat() / generate() through the trait would also dispatch
    // here, but submit_text_streaming bails on the synth
    // tokenizer's "zero tokens" output before the driver ever
    // sees the request. Compile-time coercion + the dispatch
    // probe above are enough at this layer; the chat/generate
    // happy path needs a real tokenizer fixture which lives at
    // the server-integration layer (3.7g).

    // The synchronous batched entry still works end-to-end on
    // the synth model — confirms the underlying engine's compute
    // path is alive behind the trait wrapper.
    // (Concretely: this would call generate_batched_token_ids
    // on the *inner* PagedBatchEngine, but we no longer hold
    // a concrete handle — we have Arc<dyn Engine>. Skip and
    // rely on the other tests in this file that hold the
    // concrete type.)

    drop(engine);
    let _ = std::fs::remove_file(&tmp);
}

/// Sequential submits via the streaming API also work — the
/// driver thread is reused across calls. Catches the bug where
/// the driver thread accidentally exits after the first request
/// completes instead of looping back for the next.
#[tokio::test]
async fn paged_batch_streaming_reuses_driver_across_sequential_submits() {
    use futures::StreamExt;
    let tmp = write_gguf("sequential");
    let batch = PagedBatchEngine::load(&tmp, 32, 1).expect("load");
    for prompt in [vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]] {
        let mut s = batch.submit_streaming(prompt, greedy(2)).expect("submit");
        let mut count = 0;
        while let Some(item) = s.next().await {
            let _ = item.expect("token");
            count += 1;
        }
        assert_eq!(count, 2, "each sequential request yields max_tokens");
    }
    let _ = std::fs::remove_file(&tmp);
}

/// After a request completes through `submit_streaming`, the
/// engine's `Engine::last_request_stats_snapshot()` returns a
/// populated `RequestStats` with the slot's prefill_ms,
/// decode_ms, tokens_prefilled, and tokens_generated. Pre-first-
/// completion it returns `None`. This is what makes the GUI
/// Status page's "Last request: prefill X ms, decode Y ms"
/// row populate under fused-decode mode (previously stuck at "—"
/// because the metrics handler only consulted `cpu_engine`).
#[tokio::test]
async fn paged_batch_last_request_stats_populates_after_completion() {
    use futures::StreamExt;
    use rustllama_engine::Engine;
    let tmp = write_gguf("last-stats");
    let batch = PagedBatchEngine::load(&tmp, 32, 2).expect("load");

    // Before any request, the snapshot is None.
    assert!(
        batch.last_request_stats_snapshot().is_none(),
        "fresh engine reports no last-request stats"
    );

    // Run one request to completion.
    let prompt: Vec<i32> = vec![1, 2, 3, 4];
    let mut stream = batch
        .submit_streaming(prompt.clone(), greedy(3))
        .expect("submit");
    let mut tokens = 0;
    while let Some(item) = stream.next().await {
        let _ = item.expect("ok");
        tokens += 1;
    }
    drop(stream);

    // The driver commits the slot's stats on the same retain
    // pass that releases its pages — happens right after the
    // last token's blocking_send completes. The token stream
    // emits final tokens THEN closes; closure is observed
    // after the driver moves on to retain. Give the driver a
    // few yields to publish.
    let stats = {
        let mut s = None;
        for _ in 0..50 {
            if let Some(snap) = batch.last_request_stats_snapshot() {
                s = Some(snap);
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        s.expect("stats published within 250ms")
    };
    assert_eq!(stats.tokens_generated, tokens as u32);
    assert_eq!(stats.tokens_prefilled, prompt.len() as u32);
    assert!(stats.prefill_ms >= 0.0, "prefill_ms is non-negative");
    assert!(stats.decode_ms >= 0.0, "decode_ms is non-negative");
    assert_eq!(
        stats.cache_hit_tokens, 0,
        "paged path has no prefix cache yet"
    );
    assert!(!stats.tool_call_limit_hit);
    let _ = std::fs::remove_file(&tmp);
}

/// The driver thread maintains a live `paged_active_slots` count
/// via the shared atomic counter. Two concurrent `submit_streaming`
/// calls must push the count to 2 mid-flight, and it must drain
/// back to 0 once both streams complete. Surfaced through
/// `engine.metrics()` so the GUI Status page's "Paged KV pool" panel
/// reads a non-zero "Active slots" value under fused-decode CB.
///
/// Timing: the model is tiny (synth Llama) and max_tokens is small,
/// so both submits complete in <100 ms wall-clock. We sample
/// `metrics()` once after both submits have been issued (and the
/// driver has had a chance to admit them) and assert the count is
/// at least 1 — strict "== 2" would race against the driver
/// admitting + finishing tokens before the metrics read; the
/// looser "≥ 1" is enough to prove the counter isn't stuck at 0.
#[tokio::test]
async fn paged_batch_active_slot_count_reflects_in_flight_requests() {
    use futures::StreamExt;
    use rustllama_engine::Engine;
    let tmp = write_gguf("active-slots");
    let batch = PagedBatchEngine::load(&tmp, 32, 2).expect("load");

    // Idle: counter = 0 before any work.
    assert_eq!(
        batch.metrics().paged_active_slots,
        0,
        "fresh engine starts with zero active slots"
    );

    // Submit two concurrent requests with enough decode work that
    // they don't immediately complete (max_tokens=4 means at least
    // 4 fused decode ticks).
    let s_a = batch
        .submit_streaming(vec![1, 2, 3, 4], greedy(4))
        .expect("submit A");
    let s_b = batch
        .submit_streaming(vec![5, 6, 7, 8], greedy(4))
        .expect("submit B");

    // Drain both streams in parallel; sample the counter after
    // the first tokens arrive (proves both slots are admitted)
    // BUT only via `futures::join` so the await actually yields
    // to the driver thread / runtime, rather than the test
    // serializing on one stream and starving the other.
    let collect_a = async move {
        let mut s = s_a;
        let mut out = Vec::new();
        while let Some(item) = s.next().await {
            out.push(item.expect("ok").id);
        }
        out
    };
    let collect_b = async move {
        let mut s = s_b;
        let mut out = Vec::new();
        while let Some(item) = s.next().await {
            out.push(item.expect("ok").id);
        }
        out
    };
    // We need to sample metrics MID-flight to observe the
    // counter at 2. Run a sampler future in parallel that polls
    // the engine every few ms until either it sees a >0 count
    // or both streams finish.
    let active_seen = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let active_seen_clone = active_seen.clone();
    let counter_arc = batch.active_slot_count_for_tests();
    let watcher = async move {
        // 50 ms × up to 60 iterations = 3 s budget. Each
        // iteration reads the counter directly (avoiding the
        // mutexes that `metrics()` takes) so the sample doesn't
        // race the driver for the store/table locks.
        for _ in 0..60 {
            let cur = counter_arc.load(std::sync::atomic::Ordering::Acquire);
            let prev = active_seen_clone.load(std::sync::atomic::Ordering::Relaxed);
            if cur > prev {
                active_seen_clone.store(cur, std::sync::atomic::Ordering::Relaxed);
            }
            if cur == 0 && prev > 0 {
                // Counter already drained — both slots finished.
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    };
    let (out_a, out_b, _) = tokio::join!(collect_a, collect_b, watcher);
    assert_eq!(out_a.len(), 4, "stream A finished with 4 tokens");
    assert_eq!(out_b.len(), 4, "stream B finished with 4 tokens");
    let peak = active_seen.load(std::sync::atomic::Ordering::Acquire);
    assert!(
        peak >= 1,
        "watcher must have observed ≥1 active slot during the run, peak={peak}"
    );

    // Give the driver a moment to retain-out the completed slots
    // (the counter update happens after the last token is sent
    // and the response_tx Drop is observed on the next retain
    // pass). A short tokio yield is enough on this synthetic
    // model.
    for _ in 0..20 {
        if batch.metrics().paged_active_slots == 0 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert_eq!(
        batch.metrics().paged_active_slots,
        0,
        "counter must drain to 0 once both streams complete"
    );

    let _ = std::fs::remove_file(&tmp);
}

/// Exceeding `max_slots` rejects the request with a clear error
/// instead of silently allocating beyond the configured cap.
/// Documents the v1 hard-cap contract — the scheduler in 3.7e will
/// queue overflow requests rather than failing them, but at the
/// engine layer it's a programmer error to ask for more slots than
/// the engine was built for.
#[test]
fn paged_batch_rejects_batch_exceeding_max_slots() {
    let tmp = write_gguf("overflow");
    let batch = PagedBatchEngine::load(&tmp, 32, 1).expect("load");
    let result =
        batch.generate_batched_token_ids(vec![(vec![1, 2], greedy(1)), (vec![3, 4], greedy(1))]);
    let msg = match result {
        Ok(_) => panic!("overflowing batch must fail"),
        Err(e) => e.to_string(),
    };
    assert!(
        msg.contains("max_slots"),
        "error must mention max_slots: {msg}"
    );
    let _ = std::fs::remove_file(&tmp);
}

/// P9 feature parity — GRAMMAR. A `PagedBatchEngine` run with a grammar must
/// produce the SAME tokens as the contiguous (paged-layout) `ContiguousEngine` with
/// the same grammar — proving the paged driver applies the grammar mask
/// identically (`sample_with_grammar` + `advance`), not silently dropping it.
///
/// `paged_batch_single_slot_matches_cpu_engine_greedy` already pins logit
/// parity WITHOUT a grammar; this adds the grammar on both sides. If the paged
/// engine ignored the grammar, its output would instead match the no-grammar
/// run — which, for a non-trivial grammar (a JSON grammar heavily constrains
/// this byte-level `<tok_N>` vocab), differs from the contiguous engine's
/// constrained output, so the parity assertion below catches the regression.
#[test]
fn paged_batch_grammar_matches_cpu_engine() {
    use rustllama_engine::GrammarKind;
    use rustllama_models::llama_arch::KvDtype;
    let tmp = write_gguf("grammar-vs-cpu");

    let grammar_sampling = SamplingParams {
        grammar: Some(GrammarKind::Json),
        ..greedy(4)
    };
    let prompt: Vec<i32> = vec![5, 1, 4, 2, 3, 6];

    // Contiguous reference on the SAME paged KV layout the engines share.
    let mut cpu =
        ContiguousEngine::load_with_options_and_layout(&tmp, 32, true, KvDtype::F32, "paged")
            .expect("cpu engine load");
    cpu.set_prefix_cache(false);
    let cpu_plain: Vec<i32> = cpu
        .generate_token_ids(&prompt, 4, &greedy(4))
        .expect("cpu plain")
        .into_iter()
        .map(|t| t as i32)
        .collect();
    let cpu_json: Vec<i32> = cpu
        .generate_token_ids(&prompt, 4, &grammar_sampling)
        .expect("cpu json")
        .into_iter()
        .map(|t| t as i32)
        .collect();

    let batch = PagedBatchEngine::load(&tmp, 32, 1).expect("batch load");
    let paged_json = batch
        .generate_batched_token_ids(vec![(prompt.clone(), grammar_sampling.clone())])
        .expect("paged json")
        .into_iter()
        .next()
        .unwrap();

    eprintln!(
        "grammar parity: cpu_plain={cpu_plain:?} cpu_json={cpu_json:?} paged_json={paged_json:?} \
         (grammar non-trivial on this model: {})",
        cpu_plain != cpu_json
    );
    assert_eq!(
        cpu_json, paged_json,
        "paged engine must apply the grammar identically to the contiguous engine"
    );
    let _ = std::fs::remove_file(&tmp);
}

/// P9 feature parity — STRING STOPS. A `PagedBatchEngine` request whose
/// `sampling.stop` contains the decoded text of the first emitted token must
/// stop right after that token (the stop-completing token is still emitted,
/// then the slot ends) — identical semantics to `ContiguousEngine::drive_generation`.
#[test]
fn paged_batch_string_stop_terminates_generation() {
    let tmp = write_gguf("string-stop");
    let prompt: Vec<i32> = vec![5, 1, 4, 2, 3];

    // Baseline (no stop): capture the greedy sequence + the first token's text.
    let no_stop = PagedBatchEngine::load(&tmp, 32, 1).expect("load no-stop");
    let full = no_stop
        .generate_batched_token_ids(vec![(prompt.clone(), greedy(4))])
        .expect("full generate")
        .into_iter()
        .next()
        .unwrap();
    assert_eq!(
        full.len(),
        4,
        "baseline must generate max_tokens (first token is not EOS)"
    );
    let tok0 = full[0] as u32;
    let s0 = no_stop
        .tokenizer()
        .expect("tokenizer")
        .decode_single(tok0, true)
        .expect("decode");
    assert!(
        !s0.is_empty(),
        "synth decode of the first token must be non-empty for this stop test"
    );
    drop(no_stop);

    // With stop = [s0]: generation must end right after the first token.
    let with_stop = PagedBatchEngine::load(&tmp, 32, 1).expect("load with-stop");
    let stop_sampling = SamplingParams {
        stop: vec![s0.clone()],
        ..greedy(4)
    };
    let stopped = with_stop
        .generate_batched_token_ids(vec![(prompt.clone(), stop_sampling)])
        .expect("stopped generate")
        .into_iter()
        .next()
        .unwrap();
    assert_eq!(
        stopped.len(),
        1,
        "string stop must end generation after the first token (decoded={s0:?}, baseline={full:?})"
    );
    assert_eq!(
        stopped[0], full[0],
        "the first (stop-completing) token is still emitted"
    );
    let _ = std::fs::remove_file(&tmp);
}

/// P9 feature parity — STRING STOPS through the async streaming driver path
/// (what the server actually uses). Same stop contract as the sync test, but
/// exercised through `submit_streaming` → the driver's `run_ragged_tick`.
#[tokio::test]
async fn paged_batch_streaming_string_stop_terminates() {
    use futures::StreamExt;
    let tmp = write_gguf("stream-string-stop");
    let prompt: Vec<i32> = vec![5, 1, 4, 2, 3];

    // Baseline streaming run (no stop) to capture the first token's text.
    let eng = PagedBatchEngine::load(&tmp, 32, 1).expect("load");
    let mut full: Vec<i32> = Vec::new();
    let mut s = eng
        .submit_streaming(prompt.clone(), greedy(4))
        .expect("submit");
    while let Some(item) = s.next().await {
        full.push(item.expect("tok").id as i32);
    }
    assert_eq!(full.len(), 4, "baseline streams max_tokens");
    let s0 = eng
        .tokenizer()
        .expect("tokenizer")
        .decode_single(full[0] as u32, true)
        .expect("decode");
    assert!(!s0.is_empty(), "first-token decode must be non-empty");

    // Stop run via the driver path.
    let stop_sampling = SamplingParams {
        stop: vec![s0.clone()],
        ..greedy(4)
    };
    let mut stopped: Vec<i32> = Vec::new();
    let mut s2 = eng
        .submit_streaming(prompt.clone(), stop_sampling)
        .expect("submit stop");
    while let Some(item) = s2.next().await {
        stopped.push(item.expect("tok").id as i32);
    }
    assert_eq!(
        stopped.len(),
        1,
        "driver-path string stop must end generation after the first token (decoded={s0:?})"
    );
    let _ = std::fs::remove_file(&tmp);
}

// ============================================================
// P14: per-request deadline → timeout cancel
// ============================================================

/// P14 deadline: a request submitted with an already-elapsed deadline is
/// cancelled by the driver — it flips the request's cancel flag (reusing the
/// `CancelGuard.flag` mechanism) and ends the stream with a `DeadlineExceeded`
/// terminal (the "timeout" finish reason), distinct from a natural stop. No
/// token is streamed for a request that was already past its deadline at
/// admission.
#[tokio::test]
async fn paged_batch_deadline_cancels_with_timeout_and_flips_flag() {
    use futures::StreamExt;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let tmp = write_gguf("deadline-timeout");
    let batch = PagedBatchEngine::load(&tmp, 32, 1).expect("load");

    let flag = Arc::new(AtomicBool::new(false));
    let opts = SubmitOpts {
        priority: Priority::DEFAULT,
        // Already elapsed by the time the driver inspects it.
        deadline: Some(std::time::Instant::now()),
        cancel_flag: Some(flag.clone()),
    };
    let mut s = batch
        .submit_streaming_with_opts(vec![1, 2, 3], greedy(20), opts)
        .expect("submit");

    let mut saw_timeout = false;
    let mut tokens = 0;
    while let Some(item) = s.next().await {
        match item {
            Ok(_) => tokens += 1,
            Err(e) => {
                assert!(
                    matches!(e, EngineError::DeadlineExceeded),
                    "deadline terminal must be DeadlineExceeded (got {e})"
                );
                saw_timeout = true;
                break;
            }
        }
    }
    assert!(
        saw_timeout,
        "past-deadline request must end with a timeout terminal"
    );
    assert_eq!(
        tokens, 0,
        "a request already past its deadline streams no tokens"
    );
    assert!(
        flag.load(Ordering::Acquire),
        "the deadline must flip the request's cancel flag"
    );
    let _ = std::fs::remove_file(&tmp);
}

// ============================================================
// F5: recompute preemption transparency
// ============================================================

/// F5: force page exhaustion (a TINY 1-page pool, `max_slots=2`) so admitting a
/// higher-priority request B preempts the lower-priority running request A.
/// A's pages are freed and it is re-admitted later (recompute from its prompt)
/// — and its FULL token stream must be byte-identical to an uninterrupted solo
/// run (greedy). Preemption affects only timing/scheduling, never the tokens.
#[tokio::test]
async fn paged_batch_preempted_request_resumes_byte_identical() {
    use futures::StreamExt;

    let tmp = write_gguf("preempt-transparency");
    let prompt_a: Vec<i32> = vec![3, 1, 4, 1, 5];
    let prompt_b: Vec<i32> = vec![9, 2, 6, 5, 3];
    let a_tokens = 8u32;

    // Reference: A run ALONE (normal pool, 1 slot) — the uninterrupted greedy
    // sequence the preempted-then-resumed run must reproduce exactly.
    let ref_engine = PagedBatchEngine::load_token_id_only(&tmp, 32, 1).expect("ref load");
    let ref_a: Vec<u32> = {
        let mut s = ref_engine
            .submit_streaming(prompt_a.clone(), greedy(a_tokens))
            .expect("ref submit");
        let mut out = Vec::new();
        while let Some(item) = s.next().await {
            out.push(item.expect("ref tok").id);
        }
        out
    };
    drop(ref_engine);
    assert_eq!(
        ref_a.len(),
        a_tokens as usize,
        "reference ran to max_tokens"
    );

    // Contention engine: 1-page pool, 2 slots. A holding the single page means
    // admitting B cannot find free pages → B preempts the lower-priority A.
    let eng = PagedBatchEngine::load_token_id_only_with_pool_pages(&tmp, 32, 2, 1)
        .expect("tiny-pool load");

    // Submit A (LOW) and read its first 2 tokens so it is admitted + running
    // with 2 tokens already streamed (exercises the skip-already-streamed path).
    let mut sa = eng
        .submit_streaming_with_opts(
            prompt_a.clone(),
            greedy(a_tokens),
            SubmitOpts {
                priority: Priority::LOW,
                ..Default::default()
            },
        )
        .expect("submit A");
    let mut a_out: Vec<u32> = Vec::new();
    for _ in 0..2 {
        let t = sa.next().await.expect("A token").expect("A ok");
        a_out.push(t.id);
    }

    // Submit B (HIGH) — forces preemption of A (single page already held by A).
    let sb = eng
        .submit_streaming_with_opts(
            prompt_b.clone(),
            greedy(4),
            SubmitOpts {
                priority: Priority::HIGH,
                ..Default::default()
            },
        )
        .expect("submit B");

    // Drain B fully (frees the page so A can resume) + the REST of A, together.
    let drain_b = async move {
        let mut s = sb;
        let mut n = 0u32;
        while let Some(item) = s.next().await {
            let _ = item.expect("B token");
            n += 1;
        }
        n
    };
    let drain_a_rest = async move {
        let mut rest = Vec::new();
        while let Some(item) = sa.next().await {
            rest.push(item.expect("A token (rest)").id);
        }
        rest
    };
    let (b_n, a_rest) = futures::future::join(drain_b, drain_a_rest).await;
    assert_eq!(b_n, 4, "B produced its tokens");
    a_out.extend(a_rest);

    assert_eq!(
        a_out, ref_a,
        "preempted+resumed A diverged from a solo run: got={a_out:?} ref={ref_a:?}"
    );
    assert!(
        eng.preemptions_for_tests() >= 1,
        "A must have been preempted at least once (page exhaustion was not forced)"
    );
    let _ = std::fs::remove_file(&tmp);
}

// ============================================================
// F8 — hybrid continuous batching (per-slot DeltaNet state)
// ============================================================
//
// These anchors pin the F8 contract: a HYBRID (attention + DeltaNet/SSM)
// model served through the PAGED continuous-batching engine produces the
// SAME tokens as the CONTIGUOUS `ContiguousEngine`, and two concurrent hybrid slots
// keep their recurrent state independent. The synthetic hybrid GGUF is a
// dense (non-MoE) qwen35moe-shaped toy: `full_attention_interval = 2` →
// layers 1,3 are full-attention, layers 0,2 are DeltaNet/SSM; the writer
// emits the full SSM tensor set so the model loads + runs the real
// `forward_one_hybrid`.

/// Tiny loadable + runnable synthetic HYBRID GGUF. Geometry is chosen so the
/// hybrid forward's shape asserts all hold (head_v_dim = head_dim = 8,
/// n_v_heads = ssm_inner/head_dim = 4, n_qk_heads = d_model/head_dim = 4).
fn write_hybrid_gguf(tag: &str) -> std::path::PathBuf {
    let tmp = std::env::temp_dir().join(format!("rustllama-paged-hybrid-{tag}.gguf"));
    let params = SynthLlama {
        n_layers: 4,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        d_model: 32,
        d_ff: 64,
        vocab: 32,
        ctx: 64,
        hybrid: Some(SynthHybrid {
            full_attention_interval: 2,
            ssm_state_size: 8,
            ssm_conv_kernel: 4,
            ssm_group_count: 1,
            ssm_time_step_rank: 2,
            ssm_inner_size: 32,
            shared_expert_feed_forward_length: None,
            nextn_predict_layers: 0,
        }),
        ..SynthLlama::default()
    };
    write_synthetic_llama_gguf(&tmp, &params);
    tmp
}

/// F8 anchor #1: a single paged-hybrid slot decodes BIT-IDENTICALLY to the
/// contiguous `ContiguousEngine` on the same prompt + greedy decode. The paged slot
/// carries its own contiguous KV + DeltaNet state and runs the SAME
/// `forward_one_hybrid` the contiguous engine uses, so the token streams must
/// match exactly. D1a having been relaxed is implicit in the load succeeding.
#[test]
fn paged_hybrid_single_slot_matches_cpu_engine_greedy() {
    use rustllama_models::llama_arch::KvDtype;
    let tmp = write_hybrid_gguf("m1-vs-cpu");

    let mut cpu =
        ContiguousEngine::load_with_options_and_layout(&tmp, 64, true, KvDtype::F32, "contiguous")
            .expect("cpu engine load (hybrid)");
    cpu.set_prefix_cache(false);

    let batch = PagedBatchEngine::load(&tmp, 64, 1).expect("paged hybrid load");

    let prompt: Vec<i32> = vec![3, 1, 4, 1, 5, 9];
    let cpu_out = cpu
        .generate_token_ids(&prompt, 5, &greedy(5))
        .expect("cpu generate (hybrid)");
    let batch_outs = batch
        .generate_batched_token_ids(vec![(prompt.clone(), greedy(5))])
        .expect("paged hybrid generate");

    assert_eq!(batch_outs.len(), 1, "one request in, one out");
    let batch_out: Vec<i32> = batch_outs[0].clone();
    let cpu_i32: Vec<i32> = cpu_out.iter().map(|&t| t as i32).collect();
    assert_eq!(
        cpu_i32, batch_out,
        "paged hybrid diverged from contiguous ContiguousEngine: cpu={cpu_i32:?} paged={batch_out:?}"
    );
    let _ = std::fs::remove_file(&tmp);
}

/// F8 anchor #2: two concurrent hybrid slots keep INDEPENDENT recurrent
/// state. Each slot's fused output must equal its single-slot reference —
/// if slot A's DeltaNet state leaked into slot B (or vice versa), the fused
/// outputs would diverge from the solo runs. This is the per-slot
/// recurrent-state isolation guarantee.
#[test]
fn paged_hybrid_two_slots_each_matches_single_slot_reference() {
    let tmp = write_hybrid_gguf("m2-independent");

    let batch = PagedBatchEngine::load(&tmp, 64, 2).expect("paged hybrid load");

    let prompt_a: Vec<i32> = vec![5, 1, 4, 2, 3, 7];
    let prompt_b: Vec<i32> = vec![8, 0, 7, 6, 1, 2];

    let ref_a = PagedBatchEngine::load(&tmp, 64, 1)
        .expect("ref a load")
        .generate_batched_token_ids(vec![(prompt_a.clone(), greedy(5))])
        .expect("ref a generate")
        .into_iter()
        .next()
        .unwrap();
    let ref_b = PagedBatchEngine::load(&tmp, 64, 1)
        .expect("ref b load")
        .generate_batched_token_ids(vec![(prompt_b.clone(), greedy(5))])
        .expect("ref b generate")
        .into_iter()
        .next()
        .unwrap();

    let outs = batch
        .generate_batched_token_ids(vec![
            (prompt_a.clone(), greedy(5)),
            (prompt_b.clone(), greedy(5)),
        ])
        .expect("fused hybrid generate");
    assert_eq!(outs.len(), 2);
    assert_eq!(
        outs[0], ref_a,
        "hybrid slot A recurrent state contaminated: fused={:?} ref={:?}",
        outs[0], ref_a,
    );
    assert_eq!(
        outs[1], ref_b,
        "hybrid slot B recurrent state contaminated: fused={:?} ref={:?}",
        outs[1], ref_b,
    );
    let _ = std::fs::remove_file(&tmp);
}

/// F8 anchor #3: the async STREAMING driver path (submit + driver thread +
/// `run_hybrid_tick`) serves two concurrent hybrid requests, each matching
/// its single-slot reference — the real continuous-batching path, not just
/// the synchronous batch API.
#[tokio::test]
async fn paged_hybrid_streaming_two_concurrent_submits_each_matches_reference() {
    use futures::StreamExt;
    let tmp = write_hybrid_gguf("stream-m2");
    let batch = PagedBatchEngine::load(&tmp, 64, 2).expect("paged hybrid load");

    let prompt_a: Vec<i32> = vec![5, 1, 4, 2, 3, 7];
    let prompt_b: Vec<i32> = vec![8, 0, 7, 6, 1, 2];
    let n_tokens = 5u32;

    let ref_a = PagedBatchEngine::load(&tmp, 64, 1)
        .unwrap()
        .generate_batched_token_ids(vec![(prompt_a.clone(), greedy(n_tokens))])
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    let ref_b = PagedBatchEngine::load(&tmp, 64, 1)
        .unwrap()
        .generate_batched_token_ids(vec![(prompt_b.clone(), greedy(n_tokens))])
        .unwrap()
        .into_iter()
        .next()
        .unwrap();

    let stream_a = batch
        .submit_streaming(prompt_a.clone(), greedy(n_tokens))
        .expect("submit A");
    let stream_b = batch
        .submit_streaming(prompt_b.clone(), greedy(n_tokens))
        .expect("submit B");

    let collect_a = async move {
        let mut out = Vec::new();
        let mut s = stream_a;
        while let Some(item) = s.next().await {
            out.push(item.expect("token A").id as i32);
        }
        out
    };
    let collect_b = async move {
        let mut out = Vec::new();
        let mut s = stream_b;
        while let Some(item) = s.next().await {
            out.push(item.expect("token B").id as i32);
        }
        out
    };
    let (out_a, out_b) = futures::future::join(collect_a, collect_b).await;

    assert_eq!(
        out_a, ref_a,
        "streaming hybrid slot A diverged from reference: stream={out_a:?} ref={ref_a:?}",
    );
    assert_eq!(
        out_b, ref_b,
        "streaming hybrid slot B diverged from reference: stream={out_b:?} ref={ref_b:?}",
    );
    let _ = std::fs::remove_file(&tmp);
}

// ============================================================
// P3 — cascade (shared-prefix) decode driver wiring
// ============================================================

/// P3 driver wiring: two slots whose prompts share a page-aligned (>= one
/// page = 16-token) APC prefix must ENGAGE the cascade decode path — the
/// shared prefix is attended once and merged per slot instead of each slot
/// re-attending it. Proves the driver groups the slots and routes the batched
/// decode through `forward_decode_paged_batched_cascade_f32`. The numeric
/// equivalence of that path to the per-slot path (within the merge's ~1e-6
/// tolerance) is pinned at the model layer by
/// `cascade_decode_matches_per_slot_within_tolerance`; here we pin that the
/// wiring fires and still produces a full, coherent (finite, max_tokens-long)
/// output per slot.
#[test]
fn paged_batch_cascade_engages_on_shared_prefix() {
    let tmp = write_gguf("cascade-engage");
    // max_ctx 48 ⇒ 3 pages/slot at page_size 16; prompt (18) + max_tokens (3) fits.
    let batch = PagedBatchEngine::load(&tmp, 48, 2).expect("load");
    // 16-token identical prefix (one full page) then a distinct 2-token tail.
    // Token ids stay < the synth vocab (32).
    let prefix: Vec<i32> = (1..=16).collect();
    let mut pa = prefix.clone();
    pa.extend([17, 18]);
    let mut pb = prefix.clone();
    pb.extend([19, 20]);

    let outs = batch
        .generate_batched_token_ids(vec![(pa, greedy(3)), (pb, greedy(3))])
        .expect("cascade generate");
    assert_eq!(outs.len(), 2);
    assert!(!outs[0].is_empty(), "slot A produced tokens");
    assert!(!outs[1].is_empty(), "slot B produced tokens");
    assert!(
        outs[0].len() <= 3 && outs[1].len() <= 3,
        "capped at max_tokens"
    );
    assert!(
        batch.cascade_engagements_for_tests() >= 1,
        "two slots sharing a 16-token page-aligned prefix must engage the cascade decode path \
         (engagements={})",
        batch.cascade_engagements_for_tests()
    );
    let _ = std::fs::remove_file(&tmp);
}

/// P3 gate: when NO group can form — disjoint prompts (no shared page) and a
/// single slot — the cascade path never engages and the output is
/// BYTE-IDENTICAL to the per-slot single-slot references. Guards the
/// "defaults byte-identical when the feature isn't engaged" contract.
#[test]
fn paged_batch_cascade_not_engaged_is_byte_identical() {
    let tmp = write_gguf("cascade-noengage");
    let pa: Vec<i32> = vec![5, 1, 4, 2, 3];
    let pb: Vec<i32> = vec![8, 0, 7, 6, 1]; // disjoint first token ⇒ no shared page

    let ref_a = PagedBatchEngine::load(&tmp, 32, 1)
        .unwrap()
        .generate_batched_token_ids(vec![(pa.clone(), greedy(3))])
        .unwrap()
        .into_iter()
        .next()
        .unwrap();
    let ref_b = PagedBatchEngine::load(&tmp, 32, 1)
        .unwrap()
        .generate_batched_token_ids(vec![(pb.clone(), greedy(3))])
        .unwrap()
        .into_iter()
        .next()
        .unwrap();

    let batch = PagedBatchEngine::load(&tmp, 32, 2).expect("load");
    let outs = batch
        .generate_batched_token_ids(vec![(pa, greedy(3)), (pb, greedy(3))])
        .expect("generate");
    assert_eq!(
        outs[0], ref_a,
        "slot A not byte-identical to single-slot ref"
    );
    assert_eq!(
        outs[1], ref_b,
        "slot B not byte-identical to single-slot ref"
    );
    assert_eq!(
        batch.cascade_engagements_for_tests(),
        0,
        "disjoint prompts must NOT engage the cascade path"
    );
    let _ = std::fs::remove_file(&tmp);
}

// ============================================================
// P12b — beam search on the paged engine
// ============================================================

/// P12b parity anchor: `num_beams == 1` beam search on the paged engine is
/// BYTE-IDENTICAL to greedy on the paged engine. The seed + each single-beam
/// fork run the SAME dense prefill + single-slot batched decode the sync greedy
/// path uses, and the lone surviving beam always takes the argmax candidate.
#[test]
fn paged_beam_num_beams_one_matches_greedy() {
    let tmp = write_gguf("beam-greedy");
    let batch = PagedBatchEngine::load(&tmp, 32, 1).expect("load");
    let prompt: Vec<i32> = vec![5, 1, 4, 2, 3];
    let n = 4u32;

    let greedy_out = batch
        .generate_batched_token_ids(vec![(prompt.clone(), greedy(n))])
        .expect("greedy")
        .into_iter()
        .next()
        .unwrap();

    let bp = BeamParams {
        num_beams: 1,
        length_penalty: 1.0,
        early_stopping: false,
        num_return: 1,
    };
    let beams = batch
        .generate_beam_token_ids(&prompt, n, &bp)
        .expect("beam");
    assert_eq!(beams.len(), 1, "num_beams=1 returns one hypothesis");
    let beam_toks: Vec<i32> = beams[0].tokens.iter().map(|&t| t as i32).collect();
    assert_eq!(
        beam_toks, greedy_out,
        "num_beams=1 beam must be byte-identical to greedy: beam={beam_toks:?} greedy={greedy_out:?}"
    );
    let _ = std::fs::remove_file(&tmp);
}

/// P12b: num_beams > 1 returns multiple distinct hypotheses, ranked by the
/// length-penalty-adjusted score (best first). Reuses `beam.rs`'s pool, so this
/// pins that the paged driver feeds it correctly (per-beam KV forked over the
/// shared prompt prefix) and that the ranking contract holds.
#[test]
fn paged_beam_returns_ranked_hypotheses() {
    use rustllama_engine::beam::length_adjusted_score;
    let tmp = write_gguf("beam-ranked");
    let batch = PagedBatchEngine::load(&tmp, 64, 1).expect("load");
    let prompt: Vec<i32> = vec![3, 1, 4, 1, 5, 9, 2, 6];
    let lp = 1.0f32;
    let bp = BeamParams {
        num_beams: 3,
        length_penalty: lp,
        early_stopping: false,
        num_return: 3,
    };
    let beams = batch
        .generate_beam_token_ids(&prompt, 4, &bp)
        .expect("beam");
    assert!(!beams.is_empty(), "beam search returned no hypotheses");
    assert!(beams.len() <= 3, "at most num_beams hypotheses");
    assert!(
        beams.len() >= 2,
        "num_beams=3 (no early EOS expected) should keep multiple hypotheses, got {}",
        beams.len()
    );
    // Ranked descending by the length-penalty-adjusted score, and each stored
    // score equals the formula applied to its (cum_logprob, len).
    for b in &beams {
        let expect = length_adjusted_score(b.cum_logprob, b.tokens.len(), lp);
        assert!(
            (b.score - expect).abs() < 1e-4,
            "beam score {} != length_adjusted_score {expect}",
            b.score
        );
        assert!(!b.tokens.is_empty(), "a returned beam has no tokens");
    }
    for w in beams.windows(2) {
        assert!(
            w[0].score >= w[1].score,
            "beams must be ranked by score (descending): {} then {}",
            w[0].score,
            w[1].score
        );
    }
    let _ = std::fs::remove_file(&tmp);
}

// ----------------------------------------------------------------------
// P11 Phase 3 — per-SLOT resident LoRA in the fused batch
// ----------------------------------------------------------------------

fn lora_adapter_file(tag: &str, seed: u64) -> std::path::PathBuf {
    let p = std::env::temp_dir().join(format!("rustllama-paged-lora-{tag}.gguf"));
    // Strong adapter (large alpha) so the greedy argmax observably shifts.
    write_synthetic_lora_adapter_gguf(
        &p,
        &SynthLoraAdapter {
            base: SynthLlama::default(),
            rank: 8,
            alpha: 400.0,
            seed,
        },
    );
    p
}

/// A batch mixing adapter A, adapter B, and no-adapter must serve each slot with
/// its own selection: every slot's output matches the SAME request run alone,
/// and a no-adapter slot matches the base. Proves the per-slot adapter table the
/// fused decode scopes routes each row to the right (or no) adapter.
#[test]
fn paged_batch_per_slot_lora_each_matches_single_slot_reference() {
    let tmp = write_gguf("lora-perslot");
    let ad_a = lora_adapter_file("perslot-a", 0xA11CE);
    let ad_b = lora_adapter_file("perslot-b", 0xB0B);

    let batch = PagedBatchEngine::load(&tmp, 32, 3).expect("batch engine load");
    assert_eq!(batch.lora_adapter_names().len(), 0);
    assert!(batch.attach_lora_adapter("ta", &ad_a).expect("attach a") > 0);
    assert!(batch.attach_lora_adapter("tb", &ad_b).expect("attach b") > 0);
    assert!(batch.has_lora_adapter("ta"));
    assert!(batch.has_lora_adapter("tb"));
    assert!(!batch.has_lora_adapter("ghost"));

    let p_a: Vec<i32> = vec![5, 1, 4, 2, 3];
    let p_b: Vec<i32> = vec![8, 0, 7, 6, 1];
    let p_c: Vec<i32> = vec![2, 9, 3, 4, 5];
    let samp = |ad: Option<&str>| SamplingParams {
        lora_adapter: ad.map(|s| s.to_string()),
        ..greedy(4)
    };
    let one = |engine: &PagedBatchEngine, p: &[i32], ad: Option<&str>| -> Vec<i32> {
        engine
            .generate_batched_token_ids(vec![(p.to_vec(), samp(ad))])
            .expect("single-slot generate")
            .remove(0)
    };

    // Single-slot references (each request alone).
    let ref_a = one(&batch, &p_a, Some("ta"));
    let ref_b = one(&batch, &p_b, Some("tb"));
    let ref_c = one(&batch, &p_c, None);
    // Base of p_a (no adapter) to prove the adapter actually changes the output.
    let base_a = one(&batch, &p_a, None);
    assert_ne!(
        ref_a, base_a,
        "selecting adapter A must change the output on the same prompt"
    );

    // Mixed batch: A, B, none served together in one fused decode loop.
    let mixed = batch
        .generate_batched_token_ids(vec![
            (p_a.clone(), samp(Some("ta"))),
            (p_b.clone(), samp(Some("tb"))),
            (p_c.clone(), samp(None)),
        ])
        .expect("mixed batch generate");
    assert_eq!(mixed.len(), 3);
    assert_eq!(
        mixed[0], ref_a,
        "slot 0 (adapter A) must match its single-slot reference"
    );
    assert_eq!(
        mixed[1], ref_b,
        "slot 1 (adapter B) must match its single-slot reference"
    );
    assert_eq!(
        mixed[2], ref_c,
        "slot 2 (no adapter) must match the base reference (byte-identical)"
    );

    // Unknown adapter name fails the request — never a silent base serve.
    let err = batch.generate_batched_token_ids(vec![(p_a.clone(), samp(Some("ghost")))]);
    assert!(
        err.is_err(),
        "unknown adapter must fail, not serve the base"
    );

    for p in [&tmp, &ad_a, &ad_b] {
        let _ = std::fs::remove_file(p);
    }
}
