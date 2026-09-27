//! Fused-decode `ServingModel` integration tests.
//!
//! Item 3.7g landed `ServingModel::new_fused_decode_paged`, a
//! constructor that holds a single `PagedBatchEngine` as the
//! `Arc<dyn Engine>` instead of a per-fork `MultiFlightPool`.
//! These tests verify the shape (gate sized to concurrency,
//! `multi` is `None`, scheduler is multi-flight) and that
//! `try_acquire` hands out admission permits correctly under
//! concurrent load.
//!
//! End-to-end HTTP round-trips against the fused-decode engine
//! require a tokenizer fixture that can produce non-empty ids
//! for chat-template-rendered text — the synthetic GGUF used
//! here doesn't (its vocab is `<tok_N>` with no merges). Those
//! tests come with the CLI/server-init wiring in 3.7h, where
//! the real GGUF in `target/test-models/` is available.

use std::sync::Arc;

use rustllama_engine::paged_batch::PagedBatchEngine;
use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};
use rustllama_server::{ServingModel, DEFAULT_MAX_PENDING_PER_MODEL};

fn write_gguf(tag: &str) -> std::path::PathBuf {
    let tmp = std::env::temp_dir().join(format!("rustllama-fused-serving-{tag}.gguf"));
    write_synthetic_llama_gguf(&tmp, &SynthLlama::default());
    tmp
}

/// Shape probe: `new_fused_decode_paged` produces a `ServingModel`
/// with the right gate size, scheduler kind, and no fork pool.
#[test]
fn fused_decode_serving_model_shape() {
    let tmp = write_gguf("shape");
    let paged = Arc::new(PagedBatchEngine::load(&tmp, 32, 2).expect("load paged"));
    let serving = ServingModel::new_fused_decode_paged(
        paged,
        "fused-test".into(),
        DEFAULT_MAX_PENDING_PER_MODEL,
        2,
    );

    assert_eq!(serving.model_id, "fused-test");
    assert_eq!(
        serving.concurrency(),
        1,
        "concurrency() reports 1 when multi is None — fused-decode \
         shares one engine across the gate's permits"
    );
    assert!(serving.multi.is_none(), "fused_decode must not build a fork pool");
    assert!(
        serving.cpu_engine.is_none(),
        "PagedBatchEngine is not a CpuEngine — cpu_engine stays None"
    );
    // The Arc<dyn Engine> field points at the PagedBatchEngine.
    // Smoke-test via the trait method (already covered by the
    // engine-layer test; here we just confirm the dispatch reaches
    // the trait impl).
    assert_eq!(serving.engine.n_ctx(), 32);
    // The semaphore exposes `concurrency` permits — verifiable by
    // trying to acquire two without blocking.
    assert_eq!(
        serving.gate.available_permits(),
        2,
        "gate must expose `concurrency` permits"
    );
    let _ = std::fs::remove_file(&tmp);
}

/// Two concurrent `try_acquire` calls both succeed when
/// `concurrency = 2` — the fused-decode gate is sized correctly
/// for parallel admission. A third concurrent acquire would
/// block on the gate (this test stops short of that case;
/// `concurrency_one_serializes` in `multi_flight.rs` already
/// covers the back-pressure side).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fused_decode_serving_model_admits_two_concurrent_handles() {
    let tmp = write_gguf("concurrent-admit");
    let paged = Arc::new(PagedBatchEngine::load(&tmp, 32, 2).expect("load paged"));
    let serving = ServingModel::new_fused_decode_paged(
        paged,
        "fused-concurrent".into(),
        DEFAULT_MAX_PENDING_PER_MODEL,
        2,
    );
    let serving = Arc::new(serving);

    // Acquire two handles in parallel — both must succeed without
    // waiting on the gate.
    let s1 = Arc::clone(&serving);
    let s2 = Arc::clone(&serving);
    let (h1, h2) = tokio::join!(
        async move { s1.try_acquire().await },
        async move { s2.try_acquire().await },
    );
    let h1 = h1.expect("first acquire");
    let h2 = h2.expect("second acquire");
    // Both handles point at the SAME underlying engine Arc — that's
    // the defining shape of fused decode (one shared engine for all
    // slots, with the engine's driver thread arbitrating fused
    // forward passes internally).
    assert!(
        Arc::ptr_eq(&h1.engine, &h2.engine),
        "fused_decode hands out the same engine Arc to every admit"
    );
    // Engine indices are both 0 — there's no pool to round-robin.
    assert_eq!(h1.engine_idx, 0);
    assert_eq!(h2.engine_idx, 0);
    // Gate is exhausted while both handles live.
    assert_eq!(serving.gate.available_permits(), 0);
    drop(h1);
    drop(h2);
    // Permits return on guard drop.
    assert_eq!(serving.gate.available_permits(), 2);
    let _ = std::fs::remove_file(&tmp);
}
