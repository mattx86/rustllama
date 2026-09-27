//! Regression test for the v1 single-flight contract: concurrent
//! requests against the same model serialize on the per-model
//! `Semaphore::new(1)` gate. If somebody bumps the permit count past
//! 1 without first fixing the KV-cache race, this test fails by
//! finishing too quickly (both requests overlap).
//!
//! Uses the cancel-test harness's `SlowMockEngine` (200 tokens at
//! ~10ms each) so the test has measurable wall-clock to compare
//! serial vs. parallel timing.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::{
    ChatMessage, Engine, Metrics, SamplingParams, Token, TokenStream,
};
use rustllama_server::{router, AppState, ServingModel};
use tower::ServiceExt;

/// Emits 8 tokens with a 5 ms sleep each → ~40 ms per request. Enough
/// to make serial-vs-parallel timing distinguishable while keeping
/// the test fast.
struct SlowEngine;

impl Engine for SlowEngine {
    fn metrics(&self) -> Metrics {
        Metrics {
            tokens_per_second: 200.0,
            context_used: 0,
            vram_estimate_mb: 0,
            ram_estimate_mb: 0,
            ..Default::default()
        }
    }
    fn n_ctx(&self) -> u32 {
        2048
    }
    fn vocab_size(&self) -> usize {
        256
    }
    fn tokenize(&self, text: &str) -> rustllama_engine::Result<Vec<u32>> {
        Ok(text.bytes().map(|b| b as u32).collect())
    }
    fn chat(
        &self,
        _msgs: &[ChatMessage],
        _s: &SamplingParams,
    ) -> rustllama_engine::Result<TokenStream> {
        Ok(slow_stream())
    }
    fn generate(
        &self,
        _prompt: &str,
        _s: &SamplingParams,
    ) -> rustllama_engine::Result<TokenStream> {
        Ok(slow_stream())
    }
}

fn slow_stream() -> TokenStream {
    let s = async_stream::stream! {
        for i in 0..8u32 {
            tokio::time::sleep(Duration::from_millis(5)).await;
            yield Ok(Token {
                id: i,
                text: format!("t{i} "),
                logprobs: None,
            });
        }
    };
    Box::pin(s)
}

fn build_state() -> AppState {
    let serving = ServingModel {
        engine: Arc::new(SlowEngine),
        cpu_engine: None,
        model_id: "rustllama-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    AppState::new(serving, "0.0.0-test".into())
}

async fn one_chat_request(app: axum::Router) -> StatusCode {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::json!({
                "model": "rustllama-mock",
                "messages": [{"role":"user","content":"hi"}],
                "stream": false,
                "max_tokens": 8,
            })
            .to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    // Drain the body so the response future fully completes before
    // we time-end.
    let _ = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    status
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn scheduler_pending_count_settles_to_zero_after_requests() {
    // The `ServingModel.scheduler` is in the critical path: every
    // `try_acquire` admits a slot and the `PermitGuard` drop calls
    // `complete`. Pin that the scheduler's `pending` counter returns
    // to 0 after a burst of requests — same property the older
    // `pending` AtomicUsize test checks for the backpressure counter,
    // but verifying the parallel-tracked scheduler state stays
    // consistent under load.
    let state = build_state();
    let serving = state.current().await;
    let app = router(state);

    let (_, _, _) = tokio::join!(
        one_chat_request(app.clone()),
        one_chat_request(app.clone()),
        one_chat_request(app.clone()),
    );

    // Give the runtime a beat to flush PermitGuard drops.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    assert_eq!(
        serving.scheduler.pending(),
        0,
        "scheduler `pending` must return to 0 after all requests drain; \
         a non-zero value means a PermitGuard drop forgot to call \
         scheduler.complete"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_concurrent_requests_against_same_model_serialize() {
    let state = build_state();
    let app = router(state);

    // Baseline: one request.
    let one_start = Instant::now();
    let s1 = one_chat_request(app.clone()).await;
    assert_eq!(s1, StatusCode::OK);
    let one_elapsed = one_start.elapsed();

    // Two concurrent requests. If single-flight is honored they must
    // serialize (total elapsed ≈ 2 × one_elapsed). If somebody removed
    // the gate, the two would overlap and total elapsed ≈ one_elapsed.
    let two_start = Instant::now();
    let (a, b) = tokio::join!(
        one_chat_request(app.clone()),
        one_chat_request(app.clone()),
    );
    assert_eq!(a, StatusCode::OK);
    assert_eq!(b, StatusCode::OK);
    let two_elapsed = two_start.elapsed();

    // Require the concurrent run to be at least 1.5× the single-request
    // time. Generous lower bound so the test doesn't flake under
    // scheduler jitter, but tight enough to catch a regression where
    // the gate is removed or relaxed.
    let min_expected = one_elapsed.mul_f64(1.5);
    assert!(
        two_elapsed >= min_expected,
        "two concurrent requests finished in {two_elapsed:?} (baseline {one_elapsed:?}); \
         the single-flight gate appears to be honoring at least 1.5× — but observed less, \
         suggesting the gate is no longer serializing same-model requests"
    );
}
