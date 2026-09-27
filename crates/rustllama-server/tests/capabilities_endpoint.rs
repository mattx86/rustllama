//! Integration tests for `GET /v1/capabilities`. Pins the JSON
//! shape that editor / GUI clients use for feature-detection at
//! startup. Covers:
//!   - core fields always present (server_version, response_formats,
//!     endpoints, tools)
//!   - embeddings.configured flips with the slot enabled
//!   - rerank.configured flips with the slot enabled
//!   - audit_log.enabled flips with an attached sink
//!   - auth.required flips with an attached auth state
//!   - history.enabled reflects the feature gate

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use rustllama_engine::MockEngine;
use rustllama_server::{router, AppState, AuthState, ServingModel};
use tower::ServiceExt;

fn build_state() -> AppState {
    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "caps-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    AppState::new(serving, "0.0.0-test".into())
}

async fn get_caps(state: AppState) -> serde_json::Value {
    get_caps_with_auth(state, None).await
}

async fn get_caps_with_auth(state: AppState, bearer: Option<&str>) -> serde_json::Value {
    let app = router(state);
    let mut builder = Request::builder().method("GET").uri("/v1/capabilities");
    if let Some(token) = bearer {
        builder = builder.header("authorization", format!("Bearer {token}"));
    }
    let req = builder.body(Body::empty()).unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    serde_json::from_slice(&bytes).expect("json")
}

#[tokio::test]
async fn capabilities_returns_core_fields() {
    let v = get_caps(build_state()).await;
    assert_eq!(v["server_version"], "0.0.0-test");
    // response_formats must cover the full set the chat handler
    // dispatches on — clients use this list to know which
    // grammar kinds they can request.
    let formats: Vec<&str> = v["response_formats"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap())
        .collect();
    for expected in ["json_object", "json_schema", "code", "regex", "diff"] {
        assert!(
            formats.contains(&expected),
            "response_formats missing {expected}: {formats:?}"
        );
    }
    // Endpoints surface includes the major OpenAI + Ollama + extension routes.
    let endpoints: Vec<&str> = v["endpoints"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap())
        .collect();
    for expected in [
        "/v1/chat/completions",
        "/v1/embeddings",
        "/v1/rerank",
        "/v1/models",
        "/v1/capabilities",
        "/api/chat",
        "/api/embed",
        "/healthz",
    ] {
        assert!(
            endpoints.contains(&expected),
            "endpoints missing {expected}"
        );
    }
    // Tools always reported as supported (chat endpoint accepts the
    // OpenAI tools shape regardless of model).
    assert_eq!(v["tools"]["supported"], true);
    assert!(v["tools"]["max_iterations"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn capabilities_embeddings_off_when_slot_disabled() {
    let v = get_caps(build_state()).await;
    assert_eq!(v["embeddings"]["configured"], false);
    // Dimensions absent (skip_serializing_if=None) — clients keying
    // off presence get a clean signal.
    assert!(
        v["embeddings"].get("dimensions").is_none()
            || v["embeddings"]["dimensions"].is_null(),
        "dimensions must be absent when not configured: {}",
        v["embeddings"]
    );
}

#[tokio::test]
async fn capabilities_embeddings_on_when_slot_enabled() {
    let state = build_state().with_embedding_slot();
    let v = get_caps(state).await;
    assert_eq!(v["embeddings"]["configured"], true);
    // Pre-load: dimensions null (no model loaded yet).
    assert!(
        v["embeddings"].get("dimensions").is_none()
            || v["embeddings"]["dimensions"].is_null()
    );
}

#[tokio::test]
async fn capabilities_rerank_on_when_slot_enabled() {
    let state = build_state().with_reranker_slot();
    let v = get_caps(state).await;
    assert_eq!(v["rerank"]["configured"], true);
}

#[tokio::test]
async fn capabilities_rerank_off_when_slot_disabled() {
    let v = get_caps(build_state()).await;
    assert_eq!(v["rerank"]["configured"], false);
}

#[tokio::test]
async fn capabilities_auth_required_flips_with_attached_state() {
    let v = get_caps(build_state()).await;
    assert_eq!(v["auth"]["required"], false, "no auth → false");

    // When auth is attached, the middleware requires a bearer token
    // — clients hitting /v1/capabilities have to authenticate just
    // like every other route. Pass the configured key explicitly.
    let state = build_state().with_auth(AuthState::new("supersecret").unwrap());
    let v = get_caps_with_auth(state, Some("supersecret")).await;
    assert_eq!(v["auth"]["required"], true, "auth attached → true");
}

#[tokio::test]
async fn capabilities_audit_off_by_default() {
    // Audit log requires a real file sink; the default test state
    // has none attached. Clients use this to decide whether to
    // surface "your requests are being logged" UI.
    let v = get_caps(build_state()).await;
    assert_eq!(v["audit_log"]["enabled"], false);
}

#[tokio::test]
async fn capabilities_history_reflects_feature_gate() {
    // The history field is always present (so the response shape
    // is feature-flag-independent) but `enabled=false` when the
    // feature isn't compiled in OR no store was attached. This
    // test exercises the no-store-attached path; the feature-on
    // + store-attached path is covered by the history-specific
    // test suite that has the feature gate.
    let v = get_caps(build_state()).await;
    assert_eq!(
        v["history"]["enabled"],
        false,
        "no history store attached → enabled=false"
    );
}

// ---- FIM capability ---------------------------------------------------------

#[tokio::test]
async fn capabilities_fim_unavailable_on_mock_engine() {
    // Mock engine has no tokenizer → fim.available is false even
    // when a "model" is registered.
    let v = get_caps(build_state()).await;
    assert_eq!(v["fim"]["available"], false);
    // The ids fields are skip-serialize-on-None, so they're absent.
    assert!(
        v["fim"].get("prefix_id").is_none() || v["fim"]["prefix_id"].is_null(),
        "no model loaded → no FIM ids: {}",
        v["fim"]
    );
}

#[tokio::test]
async fn capabilities_fim_available_on_chat_model_with_fim_markers() {
    use rustllama_engine::CpuEngine;
    use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};

    let model_path =
        std::env::temp_dir().join("rustllama-caps-fim-on.gguf");
    write_synthetic_llama_gguf(
        &model_path,
        &SynthLlama {
            // include_fim_tokens: true → tokenizer.fim_tokens() returns
            // Some(Qwen2.5-Coder-style triplet), which is what the
            // capabilities probe surfaces.
            include_fim_tokens: true,
            ..SynthLlama::default()
        },
    );
    let cpu = Arc::new(CpuEngine::load_with_tokenizer(&model_path, 64).expect("load"));
    let serving = ServingModel {
        engine: cpu.clone() as Arc<dyn rustllama_engine::Engine>,
        cpu_engine: Some(cpu),
        model_id: "caps-fim-on".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    let state = AppState::new(serving, "0.0.0-test".into());
    let v = get_caps(state).await;
    assert_eq!(v["fim"]["available"], true);
    // All three ids must be present + numeric.
    assert!(v["fim"]["prefix_id"].is_u64(), "prefix_id: {}", v["fim"]);
    assert!(v["fim"]["suffix_id"].is_u64());
    assert!(v["fim"]["middle_id"].is_u64());
    let _ = std::fs::remove_file(&model_path);
}

// ---- MoE capability --------------------------------------------------------

#[tokio::test]
async fn capabilities_moe_supported_true_on_default_state() {
    // Phase 2-C: MoE inference is engine-supported regardless of
    // which model (if any) is currently loaded. Mock-engine servers
    // still advertise `supported: true` so editor clients can plan
    // around the capability without waiting for a load.
    let v = get_caps(build_state()).await;
    assert_eq!(v["moe"]["supported"], true);
    // No active model loaded → active_* fields absent (skip-serialize).
    assert!(
        v["moe"].get("active_n_experts").is_none()
            || v["moe"]["active_n_experts"].is_null(),
        "no model loaded → no active expert count: {}",
        v["moe"]
    );
}

#[tokio::test]
async fn capabilities_moe_active_counts_populated_for_mixtral_shape_model() {
    use rustllama_engine::CpuEngine;
    use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama, SynthMoe};

    let model_path = std::env::temp_dir().join("rustllama-caps-moe-mixtral.gguf");
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
    let cpu = Arc::new(CpuEngine::load_with_tokenizer(&model_path, 64).expect("load"));
    let serving = ServingModel {
        engine: cpu.clone() as Arc<dyn rustllama_engine::Engine>,
        cpu_engine: Some(cpu),
        model_id: "caps-moe-mixtral".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    let state = AppState::new(serving, "0.0.0-test".into());
    let v = get_caps(state).await;
    assert_eq!(v["moe"]["supported"], true);
    assert_eq!(v["moe"]["active_n_experts"], 8);
    assert_eq!(v["moe"]["active_n_experts_used"], 2);
    // Mixtral / Qwen3-MoE have 0 shared experts → field absent.
    assert!(
        v["moe"].get("active_n_experts_shared").is_none()
            || v["moe"]["active_n_experts_shared"].is_null(),
        "Mixtral-shape model: shared count must be absent (it's 0): {}",
        v["moe"]
    );
    let _ = std::fs::remove_file(&model_path);
}

#[tokio::test]
async fn capabilities_moe_active_shared_populated_for_deepseek_shape_model() {
    use rustllama_engine::CpuEngine;
    use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama, SynthMoe};

    let model_path = std::env::temp_dir().join("rustllama-caps-moe-deepseek.gguf");
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
    let cpu = Arc::new(CpuEngine::load_with_tokenizer(&model_path, 64).expect("load"));
    let serving = ServingModel {
        engine: cpu.clone() as Arc<dyn rustllama_engine::Engine>,
        cpu_engine: Some(cpu),
        model_id: "caps-moe-deepseek".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    let state = AppState::new(serving, "0.0.0-test".into());
    let v = get_caps(state).await;
    assert_eq!(v["moe"]["active_n_experts_shared"], 1);
    let _ = std::fs::remove_file(&model_path);
}

#[tokio::test]
async fn capabilities_moe_active_counts_absent_for_dense_model() {
    use rustllama_engine::CpuEngine;
    use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};

    let model_path = std::env::temp_dir().join("rustllama-caps-moe-dense.gguf");
    write_synthetic_llama_gguf(&model_path, &SynthLlama::default());
    let cpu = Arc::new(CpuEngine::load_with_tokenizer(&model_path, 64).expect("load"));
    let serving = ServingModel {
        engine: cpu.clone() as Arc<dyn rustllama_engine::Engine>,
        cpu_engine: Some(cpu),
        model_id: "caps-moe-dense".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    let state = AppState::new(serving, "0.0.0-test".into());
    let v = get_caps(state).await;
    assert_eq!(v["moe"]["supported"], true);
    // Dense model → no active expert counts.
    for field in ["active_n_experts", "active_n_experts_used", "active_n_experts_shared"] {
        assert!(
            v["moe"].get(field).is_none() || v["moe"][field].is_null(),
            "dense model: {field} must be absent: {}",
            v["moe"]
        );
    }
    let _ = std::fs::remove_file(&model_path);
}

#[tokio::test]
async fn capabilities_fim_unavailable_on_chat_model_without_fim_markers() {
    use rustllama_engine::CpuEngine;
    use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};

    let model_path =
        std::env::temp_dir().join("rustllama-caps-fim-off.gguf");
    write_synthetic_llama_gguf(
        &model_path,
        &SynthLlama {
            include_fim_tokens: false,
            ..SynthLlama::default()
        },
    );
    let cpu = Arc::new(CpuEngine::load_with_tokenizer(&model_path, 64).expect("load"));
    let serving = ServingModel {
        engine: cpu.clone() as Arc<dyn rustllama_engine::Engine>,
        cpu_engine: Some(cpu),
        model_id: "caps-fim-off".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    let state = AppState::new(serving, "0.0.0-test".into());
    let v = get_caps(state).await;
    assert_eq!(
        v["fim"]["available"],
        false,
        "chat model without FIM markers → fim.available=false"
    );
    let _ = std::fs::remove_file(&model_path);
}
