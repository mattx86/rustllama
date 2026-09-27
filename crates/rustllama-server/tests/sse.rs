//! Integration test for the streaming `/v1/chat/completions` SSE response.
//!
//! Uses `MockEngine` so the test does not depend on a real model or tokenizer.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::MockEngine;
use rustllama_server::{router, AppState, ServingModel};
use tower::ServiceExt; // for `oneshot`

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
async fn streaming_chat_completion_emits_sse_chunks() {
    let app = router(build_state());

    let body = serde_json::json!({
        "model": "rustllama-mock",
        "messages": [{"role": "user", "content": "hello world"}],
        "stream": true,
        "max_tokens": 8
    })
    .to_string();

    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let content_type = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        content_type.starts_with("text/event-stream"),
        "expected SSE content-type, got {content_type:?}"
    );

    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).expect("utf-8");

    // Inspect SSE event lines (`data: ...`) — each followed by blank line.
    let data_lines: Vec<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .collect();

    assert!(
        data_lines.len() >= 3,
        "expected at least initial + delta + done, got {}: {:?}",
        data_lines.len(),
        data_lines
    );

    // First chunk should carry the assistant role.
    let first: serde_json::Value =
        serde_json::from_str(data_lines[0]).expect("first chunk is JSON");
    assert_eq!(first["object"], "chat.completion.chunk");
    assert_eq!(first["choices"][0]["delta"]["role"], "assistant");

    // Final non-DONE chunk should have finish_reason.
    let done_idx = data_lines
        .iter()
        .position(|s| *s == "[DONE]")
        .expect("[DONE] marker present");
    let final_chunk: serde_json::Value =
        serde_json::from_str(data_lines[done_idx - 1]).expect("final chunk is JSON");
    assert!(final_chunk["choices"][0]["finish_reason"].is_string());

    // At least one content delta should appear in between.
    let saw_content = data_lines[1..done_idx - 1]
        .iter()
        .filter_map(|s| serde_json::from_str::<serde_json::Value>(s).ok())
        .any(|v| v["choices"][0]["delta"]["content"].is_string());
    assert!(saw_content, "no content delta found: {:?}", data_lines);
}

/// `system_fingerprint` appears on every streaming chunk and matches
/// the OpenAI shape (`fp_<12 hex chars>`). Editor clients (Aider,
/// Continue) read this to detect "the server I'm streaming from
/// silently changed config mid-conversation" and warn the user.
#[tokio::test]
async fn streaming_chunks_carry_system_fingerprint() {
    let app = router(build_state());
    let body = serde_json::json!({
        "model": "rustllama-mock",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
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
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).expect("utf-8");
    let data_lines: Vec<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .collect();
    // Inspect every non-[DONE] chunk — all should carry a stable fingerprint.
    let mut seen_fps = std::collections::HashSet::new();
    let mut any_chunk = false;
    for line in &data_lines {
        if *line == "[DONE]" {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let fp = v["system_fingerprint"].as_str();
        assert!(
            fp.is_some(),
            "every streaming chunk must carry system_fingerprint: {line}"
        );
        let s = fp.unwrap();
        assert!(
            s.starts_with("fp_") && s.len() == 15,
            "fingerprint must match OpenAI shape `fp_<12hex>`, got {s:?}"
        );
        seen_fps.insert(s.to_string());
        any_chunk = true;
    }
    assert!(any_chunk, "expected at least one streaming chunk");
    // Every chunk in one stream must share the same fingerprint —
    // it's a stable per-request identifier, not a per-chunk one.
    assert_eq!(
        seen_fps.len(),
        1,
        "fingerprint must be identical across all chunks of one stream: {seen_fps:?}"
    );
}

/// `system_fingerprint` field on the non-streaming response uses the
/// same OpenAI shape. The handler computes it from the server
/// version + model id + KV dtype — same inputs always give the
/// same output within a process so consecutive non-stream requests
/// against the same model see the same fingerprint.
#[tokio::test]
async fn non_streaming_chat_response_carries_system_fingerprint() {
    let app = router(build_state());
    let body = serde_json::json!({
        "model": "rustllama-mock",
        "messages": [{"role": "user", "content": "hi"}],
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
    let body: serde_json::Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), 1 << 20).await.unwrap()).unwrap();
    let fp = body["system_fingerprint"]
        .as_str()
        .expect("system_fingerprint string");
    assert!(
        fp.starts_with("fp_") && fp.len() == 15,
        "fp shape must match OpenAI: got {fp:?}"
    );
}

#[tokio::test]
async fn non_streaming_chat_completion_rejects_remote_image_url_with_clear_400() {
    // v1 image policy (see `src/image_url.rs`): only
    // `data:image/...;base64` URIs are decoded — the shape vision
    // clients (Cursor, Continue, the OpenAI SDK helper) actually
    // emit. Remote http(s) URLs are refused with a typed 400 whose
    // message points at the data:-URI shape, because a server-side
    // fetch would be an SSRF vector. (This test previously pinned
    // the pre-VLM placeholder-render contract, which the typed
    // rejection deliberately superseded.)
    let app = router(build_state());
    let body = serde_json::json!({
        "model": "rustllama-mock",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe: "},
                {"type": "image_url", "image_url": {"url": "https://e.com/x.png"}}
            ]
        }],
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
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let text = String::from_utf8_lossy(&bytes);
    assert!(
        text.contains("data: URI"),
        "rejection must point the client at the supported shape: {text}"
    );
}

#[tokio::test]
async fn streaming_chat_completion_rejects_remote_image_url_with_clear_400() {
    // Same policy on the streaming path: the rejection happens at
    // request validation, before any SSE stream is opened.
    let app = router(build_state());
    let body = serde_json::json!({
        "model": "rustllama-mock",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "https://e.com/x.png"}}
            ]
        }],
        "stream": true,
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
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn non_streaming_chat_completion_returns_json() {
    let app = router(build_state());

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
    assert_eq!(resp.status(), StatusCode::OK);
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(ct.starts_with("application/json"), "got {ct:?}");

    let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert!(v["choices"][0]["message"]["content"].is_string());
}
