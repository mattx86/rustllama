//! Integration tests for `POST /api/gguf/inspect`.
//!
//! The endpoint is read-only and bypasses the model registry — it
//! mmap-opens a GGUF on disk, parses the header / metadata / tensor
//! table, and returns JSON. These tests pin the response shape that
//! the GUI Models page (and editors poking at the API directly)
//! depend on:
//!   - top-level architecture / tensor count / file size fields
//!   - alphabetized dtype histogram with bytes + counts
//!   - per-tensor list opt-in via `include_tensors: true`
//!   - 400 on missing body, 400 on a bad path
//!
//! Uses the synthetic Llama GGUF so the fixture is self-contained
//! and doesn't require a real model download.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::CpuEngine;
use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama, SynthMoe};
use rustllama_server::{router, AppState, ServingModel};
use tower::ServiceExt;

fn build_state(tag: &str) -> (AppState, std::path::PathBuf) {
    let tmp = std::env::temp_dir().join(format!("rustllama-inspect-{tag}.gguf"));
    write_synthetic_llama_gguf(&tmp, &SynthLlama::default());
    let cpu = CpuEngine::load_with_tokenizer(&tmp, 64).expect("load engine");
    let cpu = Arc::new(cpu);
    let serving = ServingModel {
        engine: cpu.clone() as Arc<dyn rustllama_engine::Engine>,
        cpu_engine: Some(cpu),
        model_id: "inspect-test".into(),
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

/// Inspect by absolute `path` returns 200 + the expected top-level
/// fields. Architecture is the synth model's `general.architecture`
/// metadata (`"llama"`); tensor_count > 0; dtype histogram is
/// populated.
#[tokio::test]
async fn inspect_by_path_returns_metadata_and_tensor_stats() {
    let (state, tmp) = build_state("by-path");
    let app = router(state);

    let (status, body) = post_json(
        app,
        "/api/gguf/inspect",
        serde_json::json!({
            "path": tmp.to_string_lossy(),
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
    assert_eq!(v["architecture"], "llama");
    assert!(
        v["tensor_count"].as_u64().unwrap_or(0) > 0,
        "expected at least one tensor in the synth GGUF: {v}"
    );
    assert!(
        v["file_bytes"].as_u64().unwrap_or(0) > 0,
        "file_bytes must reflect on-disk size"
    );
    assert!(
        v["total_tensor_bytes"].as_u64().unwrap_or(0) > 0,
        "total_tensor_bytes must be populated"
    );
    let dtypes = v["dtypes"].as_array().expect("dtypes array");
    assert!(!dtypes.is_empty(), "dtype histogram must have ≥1 entry");
    // Each bucket has the three fields the GUI renders.
    for bucket in dtypes {
        assert!(bucket["dtype"].is_string());
        assert!(bucket["tensor_count"].as_u64().is_some());
        assert!(bucket["bytes"].as_u64().is_some());
    }
    // Per-tensor list omitted by default — this keeps payload small.
    assert!(
        v.get("tensors").map(|t| t.is_null()).unwrap_or(true),
        "default response should not include `tensors`: {v}"
    );
    let _ = std::fs::remove_file(&tmp);
}

/// `include_tensors: true` fills the per-tensor list with one entry
/// per tensor, each carrying name + dtype + shape + bytes.
#[tokio::test]
async fn inspect_with_include_tensors_returns_full_list() {
    let (state, tmp) = build_state("with-tensors");
    let app = router(state);

    let (status, body) = post_json(
        app,
        "/api/gguf/inspect",
        serde_json::json!({
            "path": tmp.to_string_lossy(),
            "include_tensors": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    let tensors = v["tensors"].as_array().expect("tensors array");
    let tensor_count = v["tensor_count"].as_u64().unwrap();
    assert_eq!(
        tensors.len() as u64,
        tensor_count,
        "tensors[] length must match tensor_count"
    );
    let t0 = &tensors[0];
    assert!(t0["name"].is_string(), "tensor has name");
    assert!(t0["dtype"].is_string(), "tensor has dtype");
    assert!(t0["shape"].is_array(), "tensor has shape");
    assert!(t0["bytes"].as_u64().is_some(), "tensor has bytes");
    let _ = std::fs::remove_file(&tmp);
}

/// Missing body (no path/hub/name) returns 400 with a message that
/// names the three accepted keys so the GUI / editors can route the
/// error.
#[tokio::test]
async fn inspect_without_path_or_name_returns_400() {
    let (state, tmp) = build_state("no-body");
    let app = router(state);

    let (status, body) = post_json(app, "/api/gguf/inspect", serde_json::json!({})).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("path") && body_str.contains("name"),
        "400 should name the accepted keys: {body_str}"
    );
    let _ = std::fs::remove_file(&tmp);
}

/// Dense GGUFs must NOT carry MoE fields in the response —
/// `n_experts` / `n_experts_used` / `n_experts_shared` are
/// `skip_serializing_if = "Option::is_none"`, so a dense inspect
/// must omit them entirely (clients use field presence to branch
/// "dense" vs "MoE" UI).
#[tokio::test]
async fn inspect_dense_gguf_omits_moe_fields() {
    let (state, tmp) = build_state("dense-moe-omitted");
    let app = router(state);
    let (status, body) = post_json(
        app,
        "/api/gguf/inspect",
        serde_json::json!({ "path": tmp.to_string_lossy() }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    let obj = v.as_object().expect("object");
    assert!(
        !obj.contains_key("n_experts"),
        "dense GGUF must not include n_experts: {v}"
    );
    assert!(!obj.contains_key("n_experts_used"));
    assert!(!obj.contains_key("n_experts_shared"));
    let _ = std::fs::remove_file(&tmp);
}

/// MoE GGUFs surface `n_experts`, `n_experts_used`, `n_experts_shared`
/// so the GUI Models page can render an "8 experts, 2 routed" badge
/// without re-parsing the GGUF.
#[tokio::test]
async fn inspect_moe_gguf_surfaces_expert_counts() {
    let tmp =
        std::env::temp_dir().join("rustllama-inspect-moe-experts.gguf");
    write_synthetic_llama_gguf(
        &tmp,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 8,
                n_experts_used: 2,
                n_experts_shared: 0,
            }),
            ..SynthLlama::default()
        },
    );
    // Re-use the dense build_state engine — the inspect endpoint
    // doesn't consult the loaded engine for the MoE GGUF, it just
    // parses the file on disk. Build a minimal state to satisfy the
    // router constructor.
    let (state, dense_tmp) = build_state("moe-experts-state");
    let app = router(state);
    let (status, body) = post_json(
        app,
        "/api/gguf/inspect",
        serde_json::json!({ "path": tmp.to_string_lossy() }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body={}",
        String::from_utf8_lossy(&body)
    );
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(v["n_experts"].as_u64(), Some(8), "n_experts=8: {v}");
    assert_eq!(v["n_experts_used"].as_u64(), Some(2));
    assert_eq!(v["n_experts_shared"].as_u64(), Some(0));
    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_file(&dense_tmp);
}

/// DeepSeek-V3-style shared-expert MoE GGUF: `n_experts_shared` > 0.
#[tokio::test]
async fn inspect_deepseek_shape_shows_shared_expert_count() {
    let tmp =
        std::env::temp_dir().join("rustllama-inspect-moe-shared.gguf");
    write_synthetic_llama_gguf(
        &tmp,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 4,
                n_experts_used: 2,
                n_experts_shared: 1,
            }),
            ..SynthLlama::default()
        },
    );
    let (state, dense_tmp) = build_state("deepseek-shared-state");
    let app = router(state);
    let (status, body) = post_json(
        app,
        "/api/gguf/inspect",
        serde_json::json!({ "path": tmp.to_string_lossy() }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(v["n_experts_shared"].as_u64(), Some(1));
    let _ = std::fs::remove_file(&tmp);
    let _ = std::fs::remove_file(&dense_tmp);
}

/// A nonexistent path returns 400 (Gguf::open io error mapped to
/// BAD_REQUEST so editor clients can distinguish "file is bad" from
/// "server is broken").
#[tokio::test]
async fn inspect_with_bad_path_returns_400() {
    let (state, tmp) = build_state("bad-path");
    let app = router(state);

    let (status, _body) = post_json(
        app,
        "/api/gguf/inspect",
        serde_json::json!({
            "path": "C:/this/path/does/not/exist/nope.gguf",
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let _ = std::fs::remove_file(&tmp);
}
