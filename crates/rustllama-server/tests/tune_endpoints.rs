//! Integration tests for `POST /v1/tune/placement` + `POST
//! /v1/tune/batch_size`. The full measurement loop is engine-side
//! and exhaustively covered by
//! `crates/rustllama-engine/tests/measurement_module.rs`; this
//! file's contract is the HTTP boundary: request shape parsing,
//! model-path validation, and the structured-response envelope.
//!
//! Synthetic GGUF fixtures keep tests fast (~seconds vs. minutes
//! for a real 7B model). On mock-mode hosts the candidate
//! enumeration runs but persistence silently skips (no SYCL device
//! → no fingerprint → no cache file); the response still carries
//! the measurement results.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::MockEngine;
use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};
use rustllama_server::{router, AppState, ServingModel};
use tower::ServiceExt;

fn build_state(tag: &str) -> AppState {
    let _ = tag;
    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "tune-endpoint-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    AppState::new(serving, "0.0.0-test".into())
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

/// Bad `model_path` (file doesn't exist) → 400 with a clear error
/// message that names the missing path. Pins the boundary check
/// before any expensive engine-load work fires.
#[tokio::test]
async fn tune_placement_rejects_nonexistent_model_path() {
    let app = router(build_state("bad-path-placement"));
    let bogus_path = "C:/this/path/does/not/exist/nope.gguf";
    let (status, body) = post_json(
        app,
        "/v1/tune/placement",
        serde_json::json!({ "model_path": bogus_path }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("does not exist"),
        "400 body should name the failure mode: {body_str}"
    );
}

/// Same boundary check on the batch_size endpoint.
#[tokio::test]
async fn tune_batch_size_rejects_nonexistent_model_path() {
    let app = router(build_state("bad-path-batch"));
    let bogus_path = "C:/this/path/does/not/exist/nope.gguf";
    let (status, body) = post_json(
        app,
        "/v1/tune/batch_size",
        serde_json::json!({ "model_path": bogus_path, "candidates": [16, 32] }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body_str = String::from_utf8_lossy(&body);
    assert!(body_str.contains("does not exist"), "{body_str}");
}

/// Empty `candidates` array on the batch_size endpoint → 400. The
/// engine measurement function would handle it (returns None
/// winner), but it's wasteful to load a model for a zero-candidate
/// run; reject at the boundary.
#[tokio::test]
async fn tune_batch_size_rejects_empty_candidates() {
    let model = std::env::temp_dir().join("rustllama-tune-bs-empty.gguf");
    write_synthetic_llama_gguf(&model, &SynthLlama::default());

    let app = router(build_state("empty-candidates"));
    let (status, body) = post_json(
        app,
        "/v1/tune/batch_size",
        serde_json::json!({ "model_path": model.to_string_lossy(), "candidates": [] }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(
        String::from_utf8_lossy(&body).contains("non-empty"),
        "400 body must explain the candidates constraint"
    );

    let _ = std::fs::remove_file(&model);
}

/// Happy path: post to /v1/tune/batch_size with a synth GGUF + a
/// few candidates. Verify the response envelope shape — winner,
/// candidates array with the expected per-row fields, load_ms.
#[tokio::test]
async fn tune_batch_size_returns_full_envelope_on_synth_model() {
    let model = std::env::temp_dir().join("rustllama-tune-bs-happy.gguf");
    write_synthetic_llama_gguf(
        &model,
        &SynthLlama {
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 32,
            d_model: 64,
            d_ff: 128,
            vocab: 32,
            ctx: 128,
            ..SynthLlama::default()
        },
    );

    let app = router(build_state("happy-batch"));
    let (status, body) = post_json(
        app,
        "/v1/tune/batch_size",
        serde_json::json!({
            "model_path": model.to_string_lossy(),
            "candidates": [16, 32],
            "prompt_tokens": 32,
            "repeats": 1,
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body={}",
        String::from_utf8_lossy(&body)
    );
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");

    assert!(v["load_ms"].as_f64().is_some(), "load_ms is a float");
    assert!(
        v["candidates"].is_array(),
        "candidates is always an array (may have failed entries inline)"
    );
    let candidates = v["candidates"].as_array().unwrap();
    assert_eq!(candidates.len(), 2, "one report row per requested candidate");
    for c in candidates {
        assert!(c["batch_size"].as_u64().is_some(), "row has batch_size");
        assert!(c["warmup_ms"].as_f64().is_some(), "row has warmup_ms");
        // median_tps / max_tps / error are all nullable — at least
        // one of (median_tps, error) is populated per row.
        let has_result = c["median_tps"].as_f64().is_some() || c["error"].is_string();
        assert!(has_result, "every row carries either a measurement or an error: {c}");
    }
    // cache_path is null on mock-mode hosts (no SYCL → no
    // fingerprint → can't write the device-keyed file). Either
    // outcome (null or a string) is valid.
    assert!(
        v["cache_path"].is_string() || v["cache_path"].is_null(),
        "cache_path field present: {v}"
    );

    let _ = std::fs::remove_file(&model);
}

/// Neither `model_path` nor `model_name` → 400 with a message that
/// names both fields. The error path the GUI hits if it forgets to
/// pass either.
#[tokio::test]
async fn tune_placement_rejects_when_neither_path_nor_name_supplied() {
    let app = router(build_state("neither"));
    let (status, body) = post_json(app, "/v1/tune/placement", serde_json::json!({})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("model_path") && body_str.contains("model_name"),
        "400 must name both accepted fields: {body_str}"
    );
}

/// Both `model_path` AND `model_name` supplied → 400 with a "pick
/// one" message. Guards against ambiguous requests where the two
/// would disagree.
#[tokio::test]
async fn tune_placement_rejects_when_both_path_and_name_supplied() {
    let app = router(build_state("both"));
    let (status, body) = post_json(
        app,
        "/v1/tune/placement",
        serde_json::json!({
            "model_path": "C:/anything.gguf",
            "model_name": "anything",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("exactly one"),
        "400 should say 'exactly one': {body_str}"
    );
}

/// `model_name` that doesn't match any cached GGUF → 404 with a
/// message that echoes the name. Matches `/v1/models/load`'s
/// equivalent path so users get consistent diagnostics across the
/// model-resolution endpoints.
#[tokio::test]
async fn tune_placement_rejects_unknown_model_name_with_404() {
    let app = router(build_state("unknown-name"));
    let (status, body) = post_json(
        app,
        "/v1/tune/placement",
        serde_json::json!({ "model_name": "nope-no-such-cached-model-99999" }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("nope-no-such-cached-model-99999"),
        "404 should echo the requested name: {body_str}"
    );
}

/// Same neither/both/unknown gates on the batch_size endpoint —
/// shared resolver helper, but pin the contract on both routes.
#[tokio::test]
async fn tune_batch_size_rejects_when_neither_path_nor_name_supplied() {
    let app = router(build_state("bs-neither"));
    let (status, _) = post_json(
        app,
        "/v1/tune/batch_size",
        serde_json::json!({ "candidates": [16] }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

/// Happy path for placement: same envelope-shape contract. The
/// candidate set is constrained by the 4 GiB VRAM budget plus the
/// synth model's tiny dims, but at least one candidate (n_gpu=0
/// CPU fallback) always fits.
#[tokio::test]
async fn tune_placement_returns_full_envelope_on_synth_model() {
    let model = std::env::temp_dir().join("rustllama-tune-pl-happy.gguf");
    write_synthetic_llama_gguf(
        &model,
        &SynthLlama {
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 32,
            d_model: 64,
            d_ff: 128,
            vocab: 32,
            ctx: 64,
            ..SynthLlama::default()
        },
    );

    let app = router(build_state("happy-placement"));
    let (status, body) = post_json(
        app,
        "/v1/tune/placement",
        serde_json::json!({
            "model_path": model.to_string_lossy(),
            "ctx_size": 64,
            "prompt_tokens": 4,
            "decode_tokens": 4,
            "repeats": 1,
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body={}",
        String::from_utf8_lossy(&body)
    );
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert!(v["load_ms"].as_f64().is_some());
    assert!(v["candidates"].is_array());
    let candidates = v["candidates"].as_array().unwrap();
    assert!(
        !candidates.is_empty(),
        "synth model always has ≥1 fitting placement candidate"
    );
    for c in candidates {
        assert!(c["n_gpu_layers"].as_u64().is_some());
        assert!(c["warmup_ms"].as_f64().is_some());
    }

    let _ = std::fs::remove_file(&model);
}
