//! Graceful shutdown behavior:
//!   1. `AppState::begin_shutdown` flips the drain flag.
//!   2. New requests get `503 Service Unavailable` + `Retry-After: 5`.
//!   3. `/healthz` keeps working and reports `draining: true`.
//!   4. `/v1/cancel` keeps working so operators can abort in-flight work.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::MockEngine;
use rustllama_server::{router, AppState, ServingModel};
use tower::ServiceExt;

fn build_state() -> AppState {
    let serving = ServingModel {
        engine: Arc::new(MockEngine),
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

#[tokio::test]
async fn new_chat_request_during_drain_returns_503() {
    let state = build_state();
    state.begin_shutdown();

    let app = router(state);

    let body = serde_json::json!({
        "model": "rustllama-mock",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": false,
        "max_tokens": 4
    })
    .to_string();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    // Retry-After must be present so well-behaved clients back off.
    let retry_after = resp
        .headers()
        .get("retry-after")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert_eq!(retry_after, "5");
    let bytes = to_bytes(resp.into_body(), 1024).await.unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(
        body.contains("draining"),
        "body should mention draining: got {body}"
    );
}

#[tokio::test]
async fn healthz_stays_open_during_drain_and_reports_state() {
    let state = build_state();

    // Before drain: status=ok, draining=false.
    let resp1 = router(state.clone())
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp1.status(), StatusCode::OK);
    let body1: serde_json::Value =
        serde_json::from_slice(&to_bytes(resp1.into_body(), 1024).await.unwrap()).unwrap();
    assert_eq!(body1["status"], "ok");
    assert_eq!(body1["draining"], false);

    // After drain: still 200, draining=true, status=draining.
    state.begin_shutdown();
    let resp2 = router(state)
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/healthz")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp2.status(), StatusCode::OK);
    let body2: serde_json::Value =
        serde_json::from_slice(&to_bytes(resp2.into_body(), 1024).await.unwrap()).unwrap();
    assert_eq!(body2["status"], "draining");
    assert_eq!(body2["draining"], true);
}

#[tokio::test]
async fn cancel_endpoint_works_during_drain() {
    // Operators must be able to abort stuck in-flight requests even
    // after SIGTERM. `/v1/cancel` is excluded from the drain block.
    let state = build_state();
    state.begin_shutdown();

    let app = router(state);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/cancel")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(r#"{"id":"nope"}"#))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    // Cancel for an unknown id returns 404 (verified in tests/cancel.rs).
    // The point here: we get 404, not 503 — the drain middleware let
    // the request through to the handler.
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn models_list_blocked_by_drain() {
    let state = build_state();
    state.begin_shutdown();

    let app = router(state);
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn ollama_chat_blocked_by_drain() {
    let state = build_state();
    state.begin_shutdown();

    let app = router(state);
    let body = serde_json::json!({
        "model": "rustllama-mock",
        "messages": [{"role":"user","content":"hi"}],
        "stream": false,
    })
    .to_string();
    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn anthropic_messages_blocked_by_drain() {
    let state = build_state();
    state.begin_shutdown();
    let app = router(state);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::json!({
                "model": "rustllama-mock",
                "messages": [{"role":"user","content":"hi"}],
                "max_tokens": 8,
            })
            .to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn ollama_generate_blocked_by_drain() {
    let state = build_state();
    state.begin_shutdown();
    let app = router(state);
    let req = Request::builder()
        .method("POST")
        .uri("/api/generate")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::json!({
                "model": "rustllama-mock",
                "prompt": "hi",
                "stream": false,
            })
            .to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn pending_counter_returns_to_zero_after_request_completes() {
    // After a normal (non-shutdown) request completes, the per-model
    // `pending` atomic should drop back to 0. A leak here would cause
    // backpressure to fire after enough successful requests.
    let state = build_state();
    let serving = state.current().await;
    let initial = serving
        .pending
        .load(std::sync::atomic::Ordering::Acquire);
    assert_eq!(initial, 0, "pending must start at 0");

    let app = router(state.clone());
    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::json!({
                "model": "rustllama-mock",
                "messages": [{"role":"user","content":"hi"}],
                "stream": false,
            })
            .to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    // PermitGuard drops at the end of the handler, which decrements the
    // pending counter.
    let serving = state.current().await;
    assert_eq!(
        serving
            .pending
            .load(std::sync::atomic::Ordering::Acquire),
        0,
        "pending counter leaked after successful request"
    );
}

#[tokio::test]
async fn pending_counter_returns_to_zero_after_503_drain() {
    // The drain middleware returns 503 BEFORE invoking the handler, so
    // no PermitGuard is constructed and the counter is never touched.
    // Verifies the drain path doesn't leak — pending stays at 0
    // throughout the rejected request lifecycle.
    let state = build_state();
    state.begin_shutdown();
    let serving = state.current().await;
    let app = router(state.clone());
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::json!({
                "model": "rustllama-mock",
                "messages": [{"role":"user","content":"hi"}],
                "stream": false,
                "max_tokens": 4,
            })
            .to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        serving
            .pending
            .load(std::sync::atomic::Ordering::Acquire),
        0,
        "drain 503 must not increment pending"
    );
}
