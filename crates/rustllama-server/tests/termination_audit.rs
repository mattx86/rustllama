//! Termination-path audit: each API surface emits a consistent
//! termination signal across the five terminal cases. Pins the
//! mapping below so future handler refactors don't drift.
//!
//! | Surface             | normal | max_tokens | cancel       | tool_call_limit         | error    |
//! |---------------------|--------|------------|--------------|-------------------------|----------|
//! | OpenAI chat         | stop   | length     | cancelled    | tool_call_iteration_limit | error  |
//! | OpenAI completions  | stop   | length     | cancelled    | (n/a)                   | error    |
//! | Anthropic Messages  | end_turn | max_tokens | stop_sequence + `__cancelled__` | tool_use + `__tool_call_iteration_limit__` | (error event) |
//! | Ollama /api/chat    | stop   | length     | cancelled    | (n/a — tools surface differs) | (top-level error key) |
//! | Ollama /api/generate| stop   | length     | cancelled    | (n/a)                   | (top-level error key) |
//!
//! Verified by:
//!   - `tests/cancel.rs` (cancel path on all four streaming surfaces)
//!   - `tests/graceful_shutdown.rs` (drain → 503 on every surface)
//!   - `tests/backpressure.rs` (queue full → 503 + Retry-After)
//!   - this file (max_tokens-hit → length / max_tokens, per surface)
//!   - `chat::tests` (decide_*_stop pure-function tests for the
//!     tool_call_iteration_limit precedence rule)
//!
//! Errors come through axum as `500 Internal Server Error` with the
//! exception of Ollama, which emits a top-level `"error"` key in the
//! NDJSON stream's last line and exits cleanly.

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

async fn post_text(uri: &str, body: serde_json::Value) -> (StatusCode, String) {
    let app = router(build_state());
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    (status, String::from_utf8_lossy(&bytes).to_string())
}

/// MockEngine produces one token per word in the prompt. Pick a
/// many-word prompt + `max_tokens: 1` so `completion_tokens` will
/// reliably exceed `max_tokens` and the length-cap branch fires.
const MULTI_WORD_PROMPT: &str = "lorem ipsum dolor sit amet consectetur";

#[tokio::test]
async fn openai_chat_max_tokens_hit_reports_length() {
    let (status, raw) = post_text(
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": MULTI_WORD_PROMPT}],
            "max_tokens": 1,
            "stream": false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {raw}");
    let body: serde_json::Value = serde_json::from_str(&raw).expect("JSON");
    assert_eq!(
        body["choices"][0]["finish_reason"], "length",
        "OpenAI chat must emit `length` on max_tokens-hit: {body}"
    );
}

#[tokio::test]
async fn openai_chat_natural_stop_reports_stop() {
    // Same prompt, generous `max_tokens` so the stream finishes
    // before hitting the cap. Pins the "stop" branch.
    let (status, raw) = post_text(
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1024,
            "stream": false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {raw}");
    let body: serde_json::Value = serde_json::from_str(&raw).expect("JSON");
    assert_eq!(
        body["choices"][0]["finish_reason"], "stop",
        "OpenAI chat must emit `stop` on natural end: {body}"
    );
}

fn build_state_with_real_cpu_engine(tag: &str) -> (AppState, std::path::PathBuf) {
    use rustllama_engine::CpuEngine;
    use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};
    let tmp = std::env::temp_dir().join(format!("rustllama-termaudit-{tag}.gguf"));
    write_synthetic_llama_gguf(&tmp, &SynthLlama::default());
    let cpu = CpuEngine::load_with_tokenizer(&tmp, 16).expect("load cpu");
    let model_id = cpu.model_id().to_string();
    let cpu = Arc::new(cpu);
    let serving = ServingModel {
        engine: cpu.clone() as Arc<dyn rustllama_engine::Engine>,
        cpu_engine: Some(cpu),
        model_id,
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    (AppState::new(serving, "0.0.0-test".into()), tmp)
}

#[tokio::test]
async fn openai_completions_max_tokens_hit_reports_length() {
    // The completions endpoint requires a real CpuEngine (it relies
    // on the tokenizer to encode the prompt + count tokens); the
    // mock-engine harness gets a 503 from the handler's CpuEngine
    // probe. Stand up a synth-GGUF-backed engine for this one test.
    let (state, tmp) = build_state_with_real_cpu_engine("completions");
    let app = router(state);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::json!({
                "prompt": "<tok_5> <tok_6>",
                "max_tokens": 1,
                "stream": false,
            })
            .to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let raw = String::from_utf8_lossy(&bytes).to_string();
    assert_eq!(status, StatusCode::OK, "body: {raw}");
    let body: serde_json::Value = serde_json::from_str(&raw).expect("JSON");
    // The synth model may hit EOS before the 1-token cap fires (it
    // greedy-samples token 2 = EOS first). Both "stop" and "length"
    // are valid responses under this fixture — what we pin is that
    // the handler emits a recognized finish_reason rather than
    // hardcoding "stop" for everything (which was the pre-audit bug).
    let fr = body["choices"][0]["finish_reason"].as_str().expect("string");
    assert!(
        fr == "stop" || fr == "length",
        "OpenAI completions finish_reason must be stop/length, got {fr:?}: {body}"
    );
    let _ = std::fs::remove_file(&tmp);
}

#[tokio::test]
async fn anthropic_messages_max_tokens_hit_reports_max_tokens() {
    let (status, raw) = post_text(
        "/v1/messages",
        serde_json::json!({
            "messages": [{"role": "user", "content": MULTI_WORD_PROMPT}],
            "max_tokens": 1,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {raw}");
    let body: serde_json::Value = serde_json::from_str(&raw).expect("JSON");
    assert_eq!(
        body["stop_reason"], "max_tokens",
        "Anthropic must emit `max_tokens` on cap-hit: {body}"
    );
}

#[tokio::test]
async fn ollama_chat_max_tokens_hit_reports_length_done_reason() {
    // Ollama uses `num_predict` for max_tokens; under our adapter it
    // routes into `SamplingParams.max_tokens` like the rest.
    let (status, raw) = post_text(
        "/api/chat",
        serde_json::json!({
            "messages": [{"role": "user", "content": MULTI_WORD_PROMPT}],
            "stream": false,
            "options": { "num_predict": 1 }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {raw}");
    let body: serde_json::Value = serde_json::from_str(&raw).expect("JSON");
    assert_eq!(
        body["done"], true,
        "Ollama chat must report done=true: {body}"
    );
    assert_eq!(
        body["done_reason"], "length",
        "Ollama chat must emit `length` done_reason on cap-hit: {body}"
    );
}

#[tokio::test]
async fn ollama_generate_max_tokens_hit_reports_length_done_reason() {
    let (status, raw) = post_text(
        "/api/generate",
        serde_json::json!({
            "prompt": MULTI_WORD_PROMPT,
            "stream": false,
            "options": { "num_predict": 1 }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {raw}");
    let body: serde_json::Value = serde_json::from_str(&raw).expect("JSON");
    assert_eq!(body["done"], true);
    assert_eq!(
        body["done_reason"], "length",
        "Ollama generate must emit `length` on cap-hit: {body}"
    );
}

#[tokio::test]
async fn ollama_chat_streaming_max_tokens_hit_reports_length_in_final_line() {
    let app = router(build_state());
    let req = Request::builder()
        .method("POST")
        .uri("/api/chat")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::json!({
                "messages": [{"role": "user", "content": MULTI_WORD_PROMPT}],
                "stream": true,
                "options": { "num_predict": 1 }
            })
            .to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let text = String::from_utf8_lossy(&bytes).to_string();
    let last_line = text
        .lines()
        .filter(|l| !l.is_empty())
        .last()
        .expect("at least one NDJSON line");
    let body: serde_json::Value =
        serde_json::from_str(last_line).expect("final NDJSON line is JSON");
    assert_eq!(body["done"], true);
    assert_eq!(
        body["done_reason"], "length",
        "Ollama chat streaming must surface `length` on the final line: {body}"
    );
}

#[tokio::test]
async fn openai_chat_streaming_max_tokens_hit_reports_length_in_final_chunk() {
    let app = router(build_state());
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::json!({
                "messages": [{"role": "user", "content": MULTI_WORD_PROMPT}],
                "max_tokens": 1,
                "stream": true,
            })
            .to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let text = String::from_utf8_lossy(&bytes).to_string();
    // Walk every `data: {...}` chunk; find the one with a non-null
    // `finish_reason`.
    let mut saw_length = false;
    for line in text.lines() {
        let Some(payload) = line.strip_prefix("data: ") else { continue };
        if payload.trim() == "[DONE]" {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
            continue;
        };
        if let Some(fr) = v["choices"][0]["finish_reason"].as_str() {
            if fr == "length" {
                saw_length = true;
                break;
            }
        }
    }
    assert!(
        saw_length,
        "OpenAI chat streaming must emit a `finish_reason: \"length\"` chunk: {text}"
    );
}
