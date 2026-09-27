//! Integration tests for `response_format: {type: "regex", pattern: ...}`
//! request-boundary validation. Covers:
//!   - empty/missing pattern → 400
//!   - invalid regex syntax → 400 with the underlying error
//!   - valid pattern → 200 (request reaches the engine)
//!   - other response_format kinds pass through unchanged

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
        model_id: "regex-mock".into(),
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
async fn regex_response_format_without_pattern_returns_400() {
    let app = router(build_state());
    let (status, body) = post(
        app,
        serde_json::json!({
            "model": "regex-mock",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": { "type": "regex" },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("non-empty pattern"),
        "body must name the missing field: {body_str}"
    );
}

#[tokio::test]
async fn regex_response_format_with_empty_pattern_returns_400() {
    let app = router(build_state());
    let (status, body) = post(
        app,
        serde_json::json!({
            "model": "regex-mock",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": { "type": "regex", "pattern": "" },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body_str = String::from_utf8_lossy(&body);
    assert!(body_str.contains("non-empty pattern"), "{body_str}");
}

#[tokio::test]
async fn regex_response_format_with_invalid_pattern_returns_400_with_detail() {
    let app = router(build_state());
    let (status, body) = post(
        app,
        serde_json::json!({
            "model": "regex-mock",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": { "type": "regex", "pattern": "[unclosed" },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("invalid regex pattern"),
        "body must name the failure: {body_str}"
    );
}

#[tokio::test]
async fn regex_response_format_with_valid_pattern_reaches_engine() {
    // Valid pattern → request passes the boundary validation and
    // reaches the mock engine, which returns a stub stream.
    // Status 200 confirms the regex validation didn't reject.
    let app = router(build_state());
    let (status, _) = post(
        app,
        serde_json::json!({
            "model": "regex-mock",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": { "type": "regex", "pattern": r"^\d+$" },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn json_object_response_format_still_passes_through() {
    // The new regex validation must not break the existing
    // `json_object` path (no `pattern` field, no failure expected).
    let app = router(build_state());
    let (status, _) = post(
        app,
        serde_json::json!({
            "model": "regex-mock",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": { "type": "json_object" },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn no_response_format_field_still_works() {
    // Sanity: bare requests without response_format must not be
    // affected by the new validation hook.
    let app = router(build_state());
    let (status, _) = post(
        app,
        serde_json::json!({
            "model": "regex-mock",
            "messages": [{"role": "user", "content": "hi"}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}
