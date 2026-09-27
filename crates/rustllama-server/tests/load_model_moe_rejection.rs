//! Integration tests for the `POST /v1/models/load` MoE handling.
//!
//! Phase 1 of the MoE arc rejected MoE GGUFs at load time with a
//! structured 400 + diagnostic. Phase 2-C wired the forward pass so
//! MoE GGUFs now load and serve through the regular engine boundary.
//! These tests pin the post-phase-2-C contract: MoE loads succeed
//! (200), the loaded model registers as MoE-shaped, and the previous
//! `error_type: "moe_not_supported"` 400 path is no longer taken.
//!
//! The file name is preserved (`load_model_moe_rejection.rs`) so the
//! git history stays continuous with the phase-1 tests it replaces.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::MockEngine;
use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama, SynthMoe};
use rustllama_server::{router, AppState, ServingModel};
use tower::ServiceExt;

fn build_state() -> AppState {
    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "moe-load-mock".into(),
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
async fn loading_moe_gguf_succeeds_via_v1_models_load() {
    // Phase 2-C: the HTTP load path now succeeds for a Mixtral-shape
    // MoE GGUF. The response is the normal `/v1/models/load` 200
    // shape (model_id + loaded entry) — not the phase-1 400 with
    // structured diagnostic.
    let model_path = std::env::temp_dir().join("rustllama-load-moe-phase2c-mixtral.gguf");
    write_synthetic_llama_gguf(
        &model_path,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 4,
                n_experts_used: 2,
                n_experts_shared: 0,
            }),
            // Keep the synth small so the load completes quickly in
            // CI — n_layers / d_model defaults are already small.
            ..SynthLlama::default()
        },
    );

    let app = router(build_state());
    let req = Request::builder()
        .method("POST")
        .uri("/v1/models/load")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::json!({ "path": model_path.display().to_string() }).to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::OK,
        "phase 2-C: MoE load must succeed via the public HTTP entry"
    );

    let _ = std::fs::remove_file(&model_path);
}

#[tokio::test]
async fn loading_moe_gguf_with_shared_experts_succeeds() {
    // DeepSeek-V3 shape: routed experts + 1 shared. Shared-expert
    // tensors flow through the load path and the forward pass adds
    // them unconditionally on top of the routed contribution.
    let model_path =
        std::env::temp_dir().join("rustllama-load-moe-phase2c-deepseek.gguf");
    write_synthetic_llama_gguf(
        &model_path,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 4,
                n_experts_used: 2,
                n_experts_shared: 1,
            }),
            ..SynthLlama::default()
        },
    );
    let app = router(build_state());
    let req = Request::builder()
        .method("POST")
        .uri("/v1/models/load")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::json!({ "path": model_path.display().to_string() }).to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let _ = std::fs::remove_file(&model_path);
}

#[tokio::test]
async fn loading_nonexistent_path_keeps_500_path() {
    // Pin the non-MoE failure path — a missing GGUF still 500s,
    // since the file-not-found error isn't a client-input-shape
    // problem in the same way a wrong-architecture model would be.
    let app = router(build_state());
    let req = Request::builder()
        .method("POST")
        .uri("/v1/models/load")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::json!({
                "path": "/nonexistent/path/to/model.gguf",
            })
            .to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(
        resp.status(),
        StatusCode::INTERNAL_SERVER_ERROR,
        "missing-file load still 500s"
    );
}

/// After loading a MoE GGUF via `/v1/models/load`, `GET /v1/models`
/// must surface the new `moe` block on the loaded entry so editors /
/// the GUI can render an "8 routed, top-2" badge without hitting
/// `/v1/capabilities` separately.
#[tokio::test]
async fn list_models_surfaces_moe_block_for_loaded_moe_model() {
    let model_path = std::env::temp_dir().join("rustllama-list-models-moe.gguf");
    write_synthetic_llama_gguf(
        &model_path,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 8,
                n_experts_used: 2,
                n_experts_shared: 0,
            }),
            ..SynthLlama::default()
        },
    );
    let app = router(build_state());

    // Load the MoE GGUF first.
    let load_req = Request::builder()
        .method("POST")
        .uri("/v1/models/load")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::json!({
                "path": model_path.display().to_string(),
                "model_id": "moe-test-listed",
            })
            .to_string(),
        ))
        .unwrap();
    let load_resp = app.clone().oneshot(load_req).await.unwrap();
    assert_eq!(load_resp.status(), StatusCode::OK);

    // Now list and find the MoE entry.
    let list_req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let list_resp = app.oneshot(list_req).await.unwrap();
    assert_eq!(list_resp.status(), StatusCode::OK);
    let bytes = to_bytes(list_resp.into_body(), 1 << 20).await.unwrap();
    let body_str = String::from_utf8_lossy(&bytes);
    let v: serde_json::Value = serde_json::from_str(&body_str).expect("json");
    let data = v["data"].as_array().expect("data array");
    // The CpuEngine resolves the model id from the GGUF filename
    // stem, not the request's `model_id` field, so look up by
    // "has a moe block" rather than by id.
    let moe_entry = data
        .iter()
        .find(|e| e.get("moe").map(|m| !m.is_null()).unwrap_or(false))
        .unwrap_or_else(|| panic!("no entry with moe block: {body_str}"));
    let moe = &moe_entry["moe"];
    assert_eq!(
        moe["n_experts"].as_u64(),
        Some(8),
        "/v1/models must surface n_experts=8: {moe_entry}"
    );
    assert_eq!(moe["n_experts_used"].as_u64(), Some(2));
    assert_eq!(moe["n_experts_shared"].as_u64(), Some(0));

    // The mock-default entry seeded by build_state has no
    // cpu_engine (just MockEngine), so the lookup yields None and
    // the moe block is omitted via skip_serializing_if.
    let mock_entry = data
        .iter()
        .find(|e| e["id"].as_str() == Some("moe-load-mock"))
        .expect("mock default entry present");
    assert!(
        mock_entry.get("moe").map(|m| m.is_null()).unwrap_or(true),
        "mock (no cpu_engine) entry must omit moe block: {mock_entry}"
    );

    let _ = std::fs::remove_file(&model_path);
}

/// Defense-in-depth: the `moe_not_supported` JSON error shape
/// (phase-1 contract) is no longer in the response body. Any
/// client that pattern-matched on the JSON error_type field
/// should see a clean 200 instead.
#[tokio::test]
async fn loading_moe_gguf_response_does_not_contain_phase1_error_type() {
    let model_path = std::env::temp_dir().join("rustllama-load-moe-no-phase1-err.gguf");
    write_synthetic_llama_gguf(
        &model_path,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 4,
                n_experts_used: 1,
                n_experts_shared: 0,
            }),
            ..SynthLlama::default()
        },
    );
    let app = router(build_state());
    let req = Request::builder()
        .method("POST")
        .uri("/v1/models/load")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::json!({ "path": model_path.display().to_string() }).to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let body_str = String::from_utf8_lossy(&bytes);
    assert!(
        !body_str.contains("moe_not_supported"),
        "phase-1 error_type must NOT appear in phase-2-C responses: {body_str}"
    );
    let _ = std::fs::remove_file(&model_path);
}
