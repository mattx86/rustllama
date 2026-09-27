//! Backpressure regression: when more requests pile up against a model
//! than `max_pending_per_model` allows, the next request is rejected
//! with `503 Service Unavailable + Retry-After: 2` and a body that
//! names the busy model. Well-behaved clients use Retry-After to back
//! off rather than hammering the server.

use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::{
    ChatMessage, Engine, Metrics, SamplingParams, Token, TokenStream,
};
use rustllama_server::{router, AppState, ServingModel};
use tower::ServiceExt;

/// Emits 8 tokens at 8ms each → ~64ms per request. Slow enough that the
/// first request still holds its permit when the second hits the gate.
struct SlowEngine;

impl Engine for SlowEngine {
    fn metrics(&self) -> Metrics {
        Metrics {
            tokens_per_second: 125.0,
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
            tokio::time::sleep(Duration::from_millis(8)).await;
            yield Ok(Token {
                id: i,
                text: format!("t{i} "),
                logprobs: None,
            });
        }
    };
    Box::pin(s)
}

fn build_state_with_max_pending(max_pending: usize) -> AppState {
    let serving = ServingModel {
        engine: Arc::new(SlowEngine),
        cpu_engine: None,
        model_id: "rustllama-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    AppState::new(serving, "0.0.0-test".into())
}

async fn chat_request(app: axum::Router) -> (StatusCode, axum::http::HeaderMap, String) {
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
    let headers = resp.headers().clone();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    (status, headers, String::from_utf8_lossy(&bytes).to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn exceeding_max_pending_returns_503_with_retry_after() {
    // max_pending=1 means: one in-flight, no queue. The first request
    // takes the permit; the second sees pending=1 (which is NOT < 1)
    // and gets rejected with 503.
    let state = build_state_with_max_pending(1);
    let app = router(state);

    // Fire two concurrent requests; the second must be rejected.
    let (r1, r2) = tokio::join!(chat_request(app.clone()), chat_request(app.clone()));

    // Exactly one of the two should be 503 (depending on which hit the
    // gate first); the other should be 200.
    let statuses = [r1.0, r2.0];
    let ok_count = statuses.iter().filter(|s| **s == StatusCode::OK).count();
    let busy_count = statuses
        .iter()
        .filter(|s| **s == StatusCode::SERVICE_UNAVAILABLE)
        .count();
    assert_eq!(ok_count, 1, "expected exactly 1 OK, got statuses {statuses:?}");
    assert_eq!(busy_count, 1, "expected exactly 1 503, got statuses {statuses:?}");

    // Find the 503 response and inspect its shape.
    let (_, headers, body) = if r1.0 == StatusCode::SERVICE_UNAVAILABLE {
        r1
    } else {
        r2
    };
    let retry_after = headers
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(retry_after, "2", "Retry-After header missing or wrong");
    assert!(
        body.contains("rustllama-mock") && body.contains("busy"),
        "503 body should name the busy model and say busy: {body}"
    );
}

#[tokio::test]
async fn pending_counter_returns_to_zero_after_backpressure_503() {
    // The 503 path decrements `pending` on the way out via the early
    // return in `try_acquire`. Verify there's no leak: after the
    // burst settles, pending is 0.
    let state = build_state_with_max_pending(1);
    let serving = state.current().await;
    let app = router(state);

    let (_, _) = tokio::join!(chat_request(app.clone()), chat_request(app.clone()));

    // Give a beat for any background drops to settle.
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(
        serving
            .pending
            .load(std::sync::atomic::Ordering::Acquire),
        0,
        "pending counter leaked after backpressure burst"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn higher_max_pending_lets_more_requests_queue() {
    // With max_pending=4 (the default), 4 concurrent requests should
    // all succeed. They serialize through the single-flight gate but
    // none get a 503.
    let state = build_state_with_max_pending(4);
    let app = router(state);

    let (a, b, c, d) = tokio::join!(
        chat_request(app.clone()),
        chat_request(app.clone()),
        chat_request(app.clone()),
        chat_request(app.clone()),
    );
    for (label, (status, _, _)) in [("a", &a), ("b", &b), ("c", &c), ("d", &d)] {
        assert_eq!(
            *status,
            StatusCode::OK,
            "request {label} should have succeeded with max_pending=4"
        );
    }
}
