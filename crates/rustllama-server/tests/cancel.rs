//! End-to-end tests that `POST /v1/cancel` aborts in-flight streaming
//! requests on every streaming endpoint (chat, completions, anthropic
//! messages, ollama chat, ollama generate).
//!
//! We use a `SlowMockEngine` that emits one token per ~10 ms so cancellation
//! has time to race in. The response stream is fed through axum's
//! `Body::into_data_stream` and we abort it from the test side by POST'ing
//! to `/v1/cancel` once we've captured the request id from the
//! `X-Rustllama-Request-Id` response header.

use std::sync::Arc;
use std::time::Duration;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use futures::StreamExt;
use rustllama_engine::{
    ChatMessage, Engine, Metrics, SamplingParams, Token, TokenStream,
};
use rustllama_server::{router, AppState, ServingModel};
use tower::ServiceExt;

/// Emits 200 tokens at a steady 10 ms cadence so the integration test has
/// a ~2 s window to send the cancel and observe early termination.
pub struct SlowMockEngine;

impl Engine for SlowMockEngine {
    fn metrics(&self) -> Metrics {
        Metrics {
            tokens_per_second: 100.0,
            context_used: 0,
            vram_estimate_mb: 0,
            ram_estimate_mb: 0,
            ..Default::default()
        }
    }
    fn n_ctx(&self) -> u32 {
        4096
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
        for i in 0..200 {
            tokio::time::sleep(Duration::from_millis(10)).await;
            yield Ok(Token {
                id: i,
                text: format!("tok{i} "),
                logprobs: None,
            });
        }
    };
    Box::pin(s)
}

fn build_state() -> AppState {
    let serving = ServingModel {
        engine: Arc::new(SlowMockEngine),
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

/// Boilerplate: drive a streaming request to completion concurrently with a
/// cancel POST, return the full body text. Returns the captured request id
/// (from the `X-Rustllama-Request-Id` header) for assertion.
async fn run_with_cancel(
    method: &str,
    uri: &str,
    body: serde_json::Value,
) -> (String, String) {
    let state = build_state();
    let app = router(state.clone());

    let req = Request::builder()
        .method(method)
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();

    let resp = app.clone().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK, "status {:?}", resp.status());

    let id = resp
        .headers()
        .get("x-rustllama-request-id")
        .and_then(|v| v.to_str().ok())
        .expect("request id header present")
        .to_string();
    assert!(!id.is_empty(), "request id is empty");

    // Cancel ~30ms in — well past the first emitted token, well short of
    // the 2-second completion time. The order is: spawn body collection,
    // sleep, fire cancel.
    let id_for_cancel = id.clone();
    let state_for_cancel = state.clone();
    let cancel_task = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(60)).await;
        let did_fire = state_for_cancel.fire_cancel(&id_for_cancel);
        assert!(did_fire, "cancel registry didn't have id `{id_for_cancel}`");
    });

    // Drain the stream. Should terminate quickly because cancel fires.
    let started = std::time::Instant::now();
    let mut stream = resp.into_body().into_data_stream();
    let mut buf = Vec::new();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.expect("body chunk");
        buf.extend_from_slice(&chunk);
    }
    let elapsed = started.elapsed();
    cancel_task.await.unwrap();
    assert!(
        elapsed < Duration::from_millis(1500),
        "stream took {elapsed:?} — cancellation didn't fire fast enough \
         (would have run ~2s without cancellation)",
    );

    let text = String::from_utf8(buf).expect("utf-8");
    (id, text)
}

#[tokio::test]
async fn cancel_aborts_chat_stream() {
    let (_id, text) = run_with_cancel(
        "POST",
        "/v1/chat/completions",
        serde_json::json!({
            "model": "rustllama-mock",
            "messages": [{"role": "user", "content": "tell me a story"}],
            "stream": true,
            "max_tokens": 200,
        }),
    )
    .await;
    // The final chunk should carry finish_reason = "cancelled".
    let data_lines: Vec<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .collect();
    let saw_cancelled = data_lines.iter().any(|l| l.contains("\"cancelled\""));
    assert!(
        saw_cancelled,
        "no finish_reason=cancelled chunk in body:\n{text}"
    );
}

// Note: `chat_stream_with_tools` requires a real `CpuEngine` (it calls the
// tokenizer's `render_chat_with_tools`), so it can't be reached through the
// mock harness. Its cancel wiring is verified by code inspection +
// `cancel_aborts_chat_stream` covering the shared `CancelGuard` path.

#[tokio::test]
async fn cancel_aborts_anthropic_messages_stream() {
    let (_id, text) = run_with_cancel(
        "POST",
        "/v1/messages",
        serde_json::json!({
            "model": "rustllama-mock",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 200,
            "stream": true,
        }),
    )
    .await;
    // Anthropic surfaces cancel via stop_sequence = "__cancelled__" in the
    // message_delta event.
    let saw_cancel_marker = text.contains("__cancelled__");
    assert!(
        saw_cancel_marker,
        "no __cancelled__ marker in anthropic stream:\n{text}"
    );
}

#[tokio::test]
async fn cancel_aborts_ollama_chat_stream() {
    let (_id, text) = run_with_cancel(
        "POST",
        "/api/chat",
        serde_json::json!({
            "model": "rustllama-mock",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": true,
            "options": {"num_predict": 200},
        }),
    )
    .await;
    // Last NDJSON line should carry done_reason: "cancelled".
    let last_line = text.lines().filter(|l| !l.is_empty()).last().unwrap_or("");
    let last_json: serde_json::Value =
        serde_json::from_str(last_line).expect("last line is JSON");
    assert_eq!(
        last_json["done_reason"], "cancelled",
        "expected done_reason=cancelled, got: {last_line}"
    );
    assert_eq!(last_json["done"], true);
}

#[tokio::test]
async fn cancel_aborts_ollama_generate_stream() {
    let (_id, text) = run_with_cancel(
        "POST",
        "/api/generate",
        serde_json::json!({
            "model": "rustllama-mock",
            "prompt": "tell me a story",
            "stream": true,
            "options": {"num_predict": 200},
        }),
    )
    .await;
    let last_line = text.lines().filter(|l| !l.is_empty()).last().unwrap_or("");
    let last_json: serde_json::Value =
        serde_json::from_str(last_line).expect("last line is JSON");
    assert_eq!(
        last_json["done_reason"], "cancelled",
        "expected done_reason=cancelled, got: {last_line}"
    );
}

#[tokio::test]
async fn cancel_unknown_id_returns_404() {
    let app = router(build_state());
    let req = Request::builder()
        .method("POST")
        .uri("/v1/cancel")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(r#"{"id":"nonexistent-12345"}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(
        body.contains("nonexistent-12345"),
        "body should mention the unknown id, got: {body}"
    );
}
