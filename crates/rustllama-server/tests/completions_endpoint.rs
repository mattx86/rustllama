//! Integration tests for `/v1/completions` — both the plain
//! completion path and the Fill-In-Middle (FIM) extension.
//!
//! The FIM path is what editor inline-completion clients
//! (Continue.dev, Tabby, llama-vscode, …) hit. These tests pin
//! the response shape + the FIM-token wrapping behavior so an
//! accidental refactor of the FIM helper or the tokenizer's
//! `fim_tokens()` doesn't silently break editor integrations.
//!
//! Uses the synthetic Llama GGUF for the model fixture — the
//! `include_fim_tokens` flag puts `<|fim_prefix|>`,
//! `<|fim_suffix|>`, `<|fim_middle|>` in the vocab so the
//! tokenizer's FIM detection fires.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::CpuEngine;
use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};
use rustllama_server::{router, AppState, ServingModel};
use tower::ServiceExt;

fn build_state(tag: &str, include_fim: bool) -> (AppState, std::path::PathBuf) {
    let tmp = std::env::temp_dir().join(format!("rustllama-completions-{tag}.gguf"));
    let params = SynthLlama {
        include_fim_tokens: include_fim,
        ..SynthLlama::default()
    };
    write_synthetic_llama_gguf(&tmp, &params);
    let cpu = CpuEngine::load_with_tokenizer(&tmp, 64).expect("load engine");
    let cpu = Arc::new(cpu);
    let serving = ServingModel {
        engine: cpu.clone() as Arc<dyn rustllama_engine::Engine>,
        cpu_engine: Some(cpu),
        model_id: "completions-test".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    (AppState::new(serving, "0.0.0-test".into()), tmp)
}

async fn post_json(
    app: axum::Router,
    path: &str,
    body: serde_json::Value,
) -> (StatusCode, Vec<u8>) {
    let req = Request::builder()
        .method("POST")
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    (status, bytes.to_vec())
}

/// Plain (non-FIM) completion returns 200 with the expected
/// OpenAI shape: id, object="text_completion", choices[0].text,
/// finish_reason, usage. The exact text content isn't asserted
/// (the synth model emits whatever the random weights pick) —
/// what we pin is the JSON shape clients depend on.
#[tokio::test]
async fn completions_plain_returns_text_completion_shape() {
    let (state, tmp) = build_state("plain", false);
    let app = router(state);

    let (status, body) = post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "model": "completions-test",
            "prompt": "<|tok_5|>",
            "max_tokens": 2,
            "temperature": 0.0,
            "stream": false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body={}", String::from_utf8_lossy(&body));
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(v["object"], "text_completion");
    assert!(v["id"].is_string());
    assert!(v["created"].is_u64());
    let choices = v["choices"].as_array().expect("choices array");
    assert_eq!(choices.len(), 1, "one choice for non-streaming completion");
    assert!(choices[0]["text"].is_string());
    assert!(choices[0]["finish_reason"].is_string());
    assert_eq!(choices[0]["index"], 0);
    // Usage block is populated.
    assert!(v["usage"]["prompt_tokens"].as_u64().is_some());
    assert!(v["usage"]["completion_tokens"].as_u64().is_some());
    // OpenAI-compat: system_fingerprint matches the `fp_<12hex>`
    // shape and is the same as /v1/chat/completions emits.
    let fp = v["system_fingerprint"].as_str().expect("system_fingerprint string");
    assert!(
        fp.starts_with("fp_") && fp.len() == 15,
        "fingerprint must be fp_<12hex>: got {fp:?}"
    );
    let _ = std::fs::remove_file(&tmp);
}

/// FIM completion with a synth model that ships FIM tokens
/// succeeds — proves the `build_prompt_ids` helper assembled the
/// `<fim_prefix> prefix <fim_suffix> suffix <fim_middle>` token
/// sequence without panicking and the engine ran it to completion.
#[tokio::test]
async fn completions_fim_succeeds_when_model_has_fim_tokens() {
    let (state, tmp) = build_state("fim-ok", true);
    let app = router(state);

    let (status, body) = post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "model": "completions-test",
            "prompt": "<|tok_3|>",
            "suffix": "<|tok_7|>",
            "max_tokens": 2,
            "temperature": 0.0,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body={}", String::from_utf8_lossy(&body));
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(v["object"], "text_completion");
    let choices = v["choices"].as_array().expect("choices array");
    assert_eq!(choices.len(), 1);
    // Prompt token count must reflect the FIM wrapping — base
    // tokenization plus the 3 FIM special tokens. We can't pin
    // an exact number (depends on tokenization of the synth
    // tokens) but it must be > 0.
    let prompt_tokens = v["usage"]["prompt_tokens"].as_u64().expect("prompt_tokens");
    assert!(
        prompt_tokens >= 3,
        "FIM wrap adds 3 special tokens; expected ≥3, got {prompt_tokens}"
    );
    let _ = std::fs::remove_file(&tmp);
}

/// FIM request against a model WITHOUT FIM tokens returns 400
/// with a message that names the constraint. This is the
/// load-bearing UX for editor clients — they should see a clear
/// error mentioning the model needs FIM support, not a generic
/// 500 or a silent fallback.
#[tokio::test]
async fn completions_fim_returns_400_when_model_lacks_fim_tokens() {
    // include_fim = false → tokenizer.fim_tokens() returns None
    let (state, tmp) = build_state("fim-missing", false);
    let app = router(state);

    let (status, body) = post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "model": "completions-test",
            "prompt": "<|tok_3|>",
            "suffix": "<|tok_7|>",
            "max_tokens": 2,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.to_lowercase().contains("fim")
            || body_str.to_lowercase().contains("fill-in-middle"),
        "400 body must mention FIM / fill-in-middle so editors can route the error: {body_str}"
    );
    let _ = std::fs::remove_file(&tmp);
}

/// Streaming completion returns SSE chunks. We don't drive a
/// full SSE parser here — just verify the response is 200, the
/// content-type is `text/event-stream`, and the body contains
/// `data:` prefixes + a `[DONE]` sentinel.
#[tokio::test]
async fn completions_streaming_returns_sse_chunks_with_done_sentinel() {
    let (state, tmp) = build_state("stream", false);
    let app = router(state);

    let (status, body) = post_json(
        app,
        "/v1/completions",
        serde_json::json!({
            "model": "completions-test",
            "prompt": "<|tok_5|>",
            "max_tokens": 2,
            "stream": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("data:"),
        "streaming body must contain SSE `data:` events: {body_str}"
    );
    assert!(
        body_str.contains("[DONE]"),
        "streaming body must terminate with `[DONE]`: {body_str}"
    );
    // OpenAI-compat: every streaming chunk carries system_fingerprint
    // at the envelope level. Verify at least one chunk parses and
    // contains the field — clients reading the SSE stream rely on
    // this being present consistently.
    let mut seen_fp = false;
    for line in body_str.lines().filter_map(|l| l.strip_prefix("data: ")) {
        if line == "[DONE]" {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(fp) = v["system_fingerprint"].as_str() {
            assert!(
                fp.starts_with("fp_") && fp.len() == 15,
                "streaming fp must be fp_<12hex>: got {fp:?}"
            );
            seen_fp = true;
        }
    }
    assert!(
        seen_fp,
        "at least one /v1/completions streaming chunk must carry system_fingerprint: {body_str}"
    );
    let _ = std::fs::remove_file(&tmp);
}
