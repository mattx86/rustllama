//! P15 integration tests for GBNF grammar request-boundary validation on
//! `/v1/chat/completions`. Covers both the `response_format: {"type":"gbnf",
//! "grammar":"..."}` shape and the llama.cpp-server-style top-level `grammar`
//! field:
//!   - missing/empty grammar body → 400
//!   - malformed grammar (no `root`, undefined reference) → 400 with detail
//!   - valid grammar → 200 (reaches the engine)
//!
//! Uses `MockEngine` so no model load is needed — the validation runs at the
//! request boundary, before the engine is invoked.

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
        model_id: "gbnf-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    AppState::new(serving, "0.0.0-test".into())
}

async fn post(app: axum::Router, body: serde_json::Value) -> (StatusCode, Vec<u8>) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    (status, bytes.to_vec())
}

#[tokio::test]
async fn gbnf_response_format_without_grammar_returns_400() {
    let app = router(build_state());
    let (status, body) = post(
        app,
        serde_json::json!({
            "model": "gbnf-mock",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": { "type": "gbnf" },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("non-empty grammar"),
        "body must name the missing grammar field: {body_str}"
    );
}

#[tokio::test]
async fn gbnf_response_format_with_invalid_grammar_returns_400_with_detail() {
    let app = router(build_state());
    let (status, body) = post(
        app,
        serde_json::json!({
            "model": "gbnf-mock",
            "messages": [{"role": "user", "content": "hi"}],
            // No `root` rule → compile error.
            "response_format": { "type": "gbnf", "grammar": "foo ::= \"x\"" },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("invalid GBNF grammar"),
        "body must name the GBNF failure: {body_str}"
    );
}

#[tokio::test]
async fn gbnf_response_format_with_valid_grammar_reaches_engine() {
    let app = router(build_state());
    let (status, _) = post(
        app,
        serde_json::json!({
            "model": "gbnf-mock",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": { "type": "gbnf", "grammar": "root ::= \"yes\" | \"no\"" },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn gbnf_top_level_grammar_invalid_returns_400() {
    let app = router(build_state());
    let (status, body) = post(
        app,
        serde_json::json!({
            "model": "gbnf-mock",
            "messages": [{"role": "user", "content": "hi"}],
            // Undefined rule reference → compile error.
            "grammar": "root ::= undefined_rule",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("invalid GBNF grammar"),
        "body must name the GBNF failure: {body_str}"
    );
}

#[tokio::test]
async fn gbnf_top_level_grammar_valid_reaches_engine() {
    let app = router(build_state());
    let (status, _) = post(
        app,
        serde_json::json!({
            "model": "gbnf-mock",
            "messages": [{"role": "user", "content": "hi"}],
            "grammar": "root ::= \"(\" root \")\" | \"\"",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn no_grammar_field_still_works() {
    let app = router(build_state());
    let (status, _) = post(
        app,
        serde_json::json!({
            "model": "gbnf-mock",
            "messages": [{"role": "user", "content": "hi"}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}
