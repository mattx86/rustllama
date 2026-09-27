//! End-to-end HTTP integration for the fused-decode path.
//!
//! Loads a real GGUF (one with a working tokenizer + sensible
//! vocab) as a `PagedBatchEngine`, builds a `ServingModel` via
//! `new_fused_decode_paged`, hands it to the router, and fires
//! two concurrent `/v1/chat/completions` requests. Both must
//! complete successfully — proves the full pipeline:
//!
//!   HTTP request → chat handler → try_acquire (fused-decode gate)
//!   → Arc<dyn Engine> dispatch → PagedBatchEngine::chat
//!   → render_chat → submit_text_streaming → driver thread
//!   → forward_decode_paged_batched_f32 → sampled tokens
//!   → mpsc::Receiver → async_stream → SSE / JSON response.
//!
//! Gated on `RUSTLLAMA_RUN_GPU_TESTS=1` + the presence of the
//! test model under `target/test-models/`. Mirrors the
//! `real_model.rs` gating pattern so contributors who haven't
//! run `cargo xtask fetch-test-model` see a silent skip rather
//! than a failure.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::paged_batch::PagedBatchEngine;
use rustllama_server::{router, AppState, ServingModel, DEFAULT_MAX_PENDING_PER_MODEL};
use tower::ServiceExt;

fn enabled() -> bool {
    std::env::var("RUSTLLAMA_RUN_GPU_TESTS").is_ok()
}

fn model_path(name: &str) -> Option<PathBuf> {
    // CARGO_MANIFEST_DIR is `crates/rustllama-server`. Walk up to
    // the workspace root and into `target/test-models/`.
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest.parent()?.parent()?;
    let p = workspace.join("target").join("test-models").join(name);
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

/// One non-streaming chat request — returns the HTTP status and
/// response body. Body content is bounded to 1 MiB which is well
/// above what `max_tokens = 8` could possibly produce.
async fn chat_once(app: axum::Router, model_id: &str, prompt: &str) -> (StatusCode, String) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::json!({
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "stream": false,
                "max_tokens": 8,
                "temperature": 0.0,
            })
            .to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let body = String::from_utf8_lossy(&bytes).to_string();
    (status, body)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fused_decode_http_two_concurrent_chat_requests_both_succeed() {
    if !enabled() {
        eprintln!("skipped (set RUSTLLAMA_RUN_GPU_TESTS=1 to enable)");
        return;
    }
    let Some(path) = model_path("qwen2.5-coder-0.5b-instruct-q4_k_m.gguf") else {
        eprintln!(
            "skipped (run `cargo xtask fetch-test-model qwen2.5-coder-0.5b-q4_k_m` first)"
        );
        return;
    };

    // Load the model as a PagedBatchEngine with max_slots=2 so
    // the driver thread can hold both concurrent requests at
    // once. ctx_size=256 is plenty for a short chat completion;
    // matches the real_model.rs configuration.
    let paged = PagedBatchEngine::load(&path, 256, 2).expect("load PagedBatchEngine");
    let model_id = paged.model_id().to_string();
    let paged = Arc::new(paged);
    let serving = ServingModel::new_fused_decode_paged(
        paged,
        model_id.clone(),
        DEFAULT_MAX_PENDING_PER_MODEL,
        2,
    );
    let state = AppState::new(serving, env!("CARGO_PKG_VERSION").to_string());
    let app = router(state);

    // Two concurrent requests through the router. With
    // concurrency=2 they should be admitted simultaneously and
    // share the engine's driver thread for fused decode.
    let start = Instant::now();
    let (a, b) = tokio::join!(
        chat_once(app.clone(), &model_id, "What is 2+2?"),
        chat_once(app.clone(), &model_id, "Name one prime number."),
    );
    let elapsed = start.elapsed();
    eprintln!(
        "fused-decode http parallel: {:?} (status_a={}, status_b={})",
        elapsed, a.0, b.0
    );

    assert_eq!(a.0, StatusCode::OK, "request A failed: body={}", a.1);
    assert_eq!(b.0, StatusCode::OK, "request B failed: body={}", b.1);
    // Loose wall-clock sanity: anything faster than 60 s is
    // acceptable for a 0.5B model doing 8 decode steps × 2
    // requests; we just need a regression backstop in case the
    // gate misconfigures and serializes.
    assert!(
        elapsed < Duration::from_secs(60),
        "fused-decode chat took {elapsed:?} — looks broken (>60s)",
    );

    // Sanity-check the response bodies look like real OpenAI
    // chat-completion shapes — both should contain `"choices"`
    // and one of the standard finish_reason values.
    for (label, (_, body)) in [("A", &a), ("B", &b)] {
        assert!(
            body.contains("\"choices\""),
            "request {label} body missing 'choices': {body}"
        );
        assert!(
            body.contains("\"finish_reason\""),
            "request {label} body missing 'finish_reason': {body}"
        );
    }
}
