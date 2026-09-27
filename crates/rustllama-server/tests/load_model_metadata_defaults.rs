//! Verifies that `POST /v1/models/load` picks a sensible `ctx_size`
//! from the model's GGUF metadata when the caller doesn't pin one
//! explicitly. Two cases:
//!
//!   - Model trained with ctx_train < 8192 (the legacy default) →
//!     resolved ctx_size matches ctx_train. Saves KV-cache memory
//!     on small-ctx models.
//!   - Explicit `ctx_size` in the request → metadata is ignored,
//!     the override wins. Existing scripts that pin a value keep
//!     working unchanged.
//!
//! Uses the synth-llama fixture so the test doesn't pay model
//! download / inference cost.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize};

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};
use rustllama_server::{router, AppState};
use tower::ServiceExt;

fn build_empty_state(_tag: &str) -> AppState {
    let _ = (Arc::<()>::new(()), AtomicUsize::new(0), AtomicU64::new(0));
    AppState::empty(
        "0.0.0-test".into(),
        rustllama_server::DEFAULT_MAX_LOADED_MODELS,
        rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
    )
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

async fn get_json(app: axum::Router, path: &str) -> (StatusCode, Vec<u8>) {
    let req = Request::builder()
        .method("GET")
        .uri(path)
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    (status, bytes.to_vec())
}

/// A small-ctx model (ctx_train < 8192) loaded without an explicit
/// ctx_size resolves to its trained context length, not the legacy
/// 8192 default. Verifies via /v1/metrics that the loaded model's
/// ctx_size matches.
#[tokio::test]
async fn load_without_ctx_size_picks_up_trained_context_length() {
    let tmp = std::env::temp_dir().join("rustllama-load-meta-ctx-train.gguf");
    let synth = SynthLlama {
        // Pick a non-default ctx so we can prove the metadata is
        // being read (default would be 32 which is too small for
        // anyone to want as ctx_size — the engine would error on
        // even a small prompt + max_tokens combo).
        ctx: 1024,
        ..SynthLlama::default()
    };
    write_synthetic_llama_gguf(&tmp, &synth);

    let state = build_empty_state("trained-ctx");
    let app = router(state.clone());
    let (status, body) = post_json(
        app,
        "/v1/models/load",
        serde_json::json!({ "path": tmp.to_string_lossy() }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "load failed: {}",
        String::from_utf8_lossy(&body)
    );
    let v: serde_json::Value = serde_json::from_slice(&body).expect("load response is json");
    let model_id = v["model_id"].as_str().expect("model_id").to_string();

    // /v1/metrics returns the active model's metrics including
    // ctx_size — the resolved value the engine actually allocated.
    let app = router(state);
    let (mstatus, mbody) = get_json(app, "/v1/metrics").await;
    assert_eq!(mstatus, StatusCode::OK);
    let m: serde_json::Value = serde_json::from_slice(&mbody).expect("metrics is json");
    assert_eq!(
        m["model_id"].as_str().unwrap_or_default(),
        model_id,
        "metrics is reporting the just-loaded model"
    );
    assert_eq!(
        m["ctx_size"].as_u64().expect("ctx_size"),
        1024,
        "ctx_size should match the model's trained context_length, not the 8192 default"
    );

    let _ = std::fs::remove_file(&tmp);
}

/// An explicit `ctx_size` in the load request overrides the
/// metadata-inferred default, so scripts that already pin a value
/// see the same behavior as before. Pinning this guards the
/// backward-compat path.
#[tokio::test]
async fn explicit_ctx_size_overrides_metadata_default() {
    let tmp = std::env::temp_dir().join("rustllama-load-meta-explicit.gguf");
    let synth = SynthLlama {
        ctx: 1024,
        ..SynthLlama::default()
    };
    write_synthetic_llama_gguf(&tmp, &synth);

    let state = build_empty_state("explicit");
    let app = router(state.clone());
    let (status, body) = post_json(
        app,
        "/v1/models/load",
        // Explicit override. Smaller than the metadata's 1024 so
        // we can prove the override is what ran.
        serde_json::json!({ "path": tmp.to_string_lossy(), "ctx_size": 256 }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "load failed: {}",
        String::from_utf8_lossy(&body)
    );

    let app = router(state);
    let (_, mbody) = get_json(app, "/v1/metrics").await;
    let m: serde_json::Value = serde_json::from_slice(&mbody).expect("metrics is json");
    assert_eq!(
        m["ctx_size"].as_u64().expect("ctx_size"),
        256,
        "explicit ctx_size must win over metadata-inferred default"
    );

    let _ = std::fs::remove_file(&tmp);
}
