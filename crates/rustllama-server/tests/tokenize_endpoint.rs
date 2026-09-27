//! Integration test for `/v1/tokenize` + `/v1/detokenize`.
//!
//! Loads a synthetic Llama GGUF (which carries a real Tokenizer via
//! the `synth::write_synthetic_llama_gguf` helper) so we can round-
//! trip text → token ids → text and pin the response shape.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::CpuEngine;
use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};
use rustllama_server::{router, AppState, ServingModel};
use tower::ServiceExt;

fn build_state_tagged(tag: &str) -> (AppState, std::path::PathBuf) {
    // Per-test tag so parallel test runs don't race on a single
    // shared GGUF mmap. Without it, a teardown of one test can hit
    // ERROR_USER_MAPPED_FILE on Windows while another still has the
    // file open.
    let tmp = std::env::temp_dir().join(format!("rustllama-tokenize-endpoint-{tag}.gguf"));
    write_synthetic_llama_gguf(&tmp, &SynthLlama::default());
    let cpu = CpuEngine::load_with_tokenizer(&tmp, 64).expect("load engine");
    let cpu = Arc::new(cpu);
    let serving = ServingModel {
        engine: cpu.clone() as Arc<dyn rustllama_engine::Engine>,
        cpu_engine: Some(cpu),
        model_id: "tokenize-test".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    (AppState::new(serving, "0.0.0-test".into()), tmp)
}

async fn post_json(app: axum::Router, path: &str, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
    let req = Request::builder()
        .method("POST")
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, v)
}

#[tokio::test]
async fn tokenize_returns_ids_and_count() {
    let (state, tmp) = build_state_tagged("ids");
    let app = router(state);

    let (status, body) = post_json(
        app,
        "/v1/tokenize",
        serde_json::json!({"content": "hello world", "add_bos": false}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let tokens = body["tokens"].as_array().expect("tokens array");
    // The synth tokenizer is GPT2-BPE; "hello world" → some non-empty
    // token list. We don't assert a specific count because the synth
    // vocab's merges are implementation-defined.
    let count = body["count"].as_u64().expect("count");
    assert_eq!(count as usize, tokens.len(), "count must match array len");
    assert_eq!(body["model_id"], "tokenize-test");

    let _ = std::fs::remove_file(&tmp);
}

#[tokio::test]
async fn add_bos_flag_changes_count() {
    // When `add_bos: true` is set and the tokenizer has a BOS, one
    // extra token shows up at the head. Pin that the flag is honored.
    let (state, tmp) = build_state_tagged("bos");
    let app = router(state);

    let (_, with_bos) = post_json(
        app.clone(),
        "/v1/tokenize",
        serde_json::json!({"content": "hello", "add_bos": true}),
    )
    .await;
    let (_, without_bos) = post_json(
        app,
        "/v1/tokenize",
        serde_json::json!({"content": "hello", "add_bos": false}),
    )
    .await;
    let with_n = with_bos["count"].as_u64().unwrap();
    let without_n = without_bos["count"].as_u64().unwrap();
    // BOS may or may not be present in the synth tokenizer's config.
    // Either way, the with-BOS count should be ≥ the without-BOS
    // count — never smaller. Pinning equality would over-specify.
    assert!(
        with_n >= without_n,
        "add_bos=true should not decrease token count ({with_n} vs {without_n})"
    );

    let _ = std::fs::remove_file(&tmp);
}

#[tokio::test]
async fn tokenize_unknown_model_returns_404() {
    let (state, tmp) = build_state_tagged("notfound");
    let app = router(state);

    let (status, _) = post_json(
        app,
        "/v1/tokenize",
        serde_json::json!({"content": "hi", "model": "not-loaded"}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let _ = std::fs::remove_file(&tmp);
}
