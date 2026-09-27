//! Integration tests for the Ollama-compatible HTTP surface.
//!
//! Uses `MockEngine` for the request/response shape checks. Endpoints
//! that require a real CpuEngine (`/api/show` needs the GGUF config,
//! `/api/pull` / `/api/delete` touch the cache directory) are covered
//! at the smoke level + the cancel test suite exercises the streaming
//! `/api/chat` and `/api/generate` paths.

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
    AppState::new(serving, "0.1.0-ollama-test".into())
}

async fn get_json(uri: &str) -> (StatusCode, serde_json::Value) {
    let app = router(build_state());
    let req = Request::builder()
        .method("GET")
        .uri(uri)
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let body = serde_json::from_slice::<serde_json::Value>(&bytes).unwrap_or_else(|_| {
        serde_json::Value::String(String::from_utf8_lossy(&bytes).to_string())
    });
    (status, body)
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

#[tokio::test]
async fn version_returns_server_version() {
    let (status, body) = get_json("/api/version").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["version"], "0.1.0-ollama-test");
}

#[tokio::test]
async fn tags_lists_loaded_models() {
    let (status, body) = get_json("/api/tags").await;
    assert_eq!(status, StatusCode::OK);
    let models = body["models"].as_array().expect("models array");
    // At minimum the loaded mock model should appear. The cache
    // listing may add more entries on a populated host, but the
    // loaded mock model is always there.
    let has_mock = models.iter().any(|m| m["name"] == "rustllama-mock");
    assert!(has_mock, "expected rustllama-mock in tags: {body}");
    // Each entry must carry the standard Ollama-tag schema.
    for m in models {
        assert!(m["name"].is_string());
        assert!(m["model"].is_string());
        assert!(m["modified_at"].is_string());
        assert!(m["size"].is_number());
        assert!(m["details"]["format"] == "gguf");
        assert!(m["details"]["families"].is_array());
    }
}

#[tokio::test]
async fn chat_non_streaming_returns_single_json_object_with_done() {
    let (status, raw) = post_text(
        "/api/chat",
        serde_json::json!({
            "model": "rustllama-mock",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {raw}");
    let body: serde_json::Value = serde_json::from_str(&raw).expect("non-stream is one JSON");
    // Ollama non-stream chat: { model, message: {role, content}, done: true, done_reason, ... }
    assert_eq!(body["model"], "rustllama-mock");
    assert_eq!(body["message"]["role"], "assistant");
    assert!(body["message"]["content"].is_string());
    assert_eq!(body["done"], true);
    assert!(body["done_reason"].is_string());
}

#[tokio::test]
async fn chat_streaming_emits_ndjson_with_final_done_true() {
    let (status, raw) = post_text(
        "/api/chat",
        serde_json::json!({
            "model": "rustllama-mock",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {raw}");
    // NDJSON: one JSON object per line.
    let lines: Vec<&str> = raw.lines().filter(|l| !l.is_empty()).collect();
    assert!(!lines.is_empty(), "expected at least one NDJSON line");
    // First line should be a chat delta with done=false.
    let first: serde_json::Value =
        serde_json::from_str(lines[0]).expect("first line is JSON");
    assert_eq!(first["model"], "rustllama-mock");
    assert!(first["message"]["role"] == "assistant");
    // Last line carries done=true plus done_reason.
    let last: serde_json::Value =
        serde_json::from_str(lines.last().unwrap()).expect("last line is JSON");
    assert_eq!(last["done"], true);
    assert!(last["done_reason"].is_string());
}

#[tokio::test]
async fn generate_non_streaming_returns_single_json_with_response_field() {
    let (status, raw) = post_text(
        "/api/generate",
        serde_json::json!({
            "model": "rustllama-mock",
            "prompt": "tell me a fact",
            "stream": false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {raw}");
    let body: serde_json::Value = serde_json::from_str(&raw).expect("non-stream is one JSON");
    assert_eq!(body["model"], "rustllama-mock");
    assert!(body["response"].is_string());
    assert_eq!(body["done"], true);
    assert!(body["done_reason"].is_string());
}

#[tokio::test]
async fn generate_streaming_emits_ndjson_with_final_done_true() {
    let (status, raw) = post_text(
        "/api/generate",
        serde_json::json!({
            "model": "rustllama-mock",
            "prompt": "tell me a fact",
            "stream": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {raw}");
    let lines: Vec<&str> = raw.lines().filter(|l| !l.is_empty()).collect();
    assert!(!lines.is_empty());
    let first: serde_json::Value = serde_json::from_str(lines[0]).expect("first JSON");
    assert!(first["response"].is_string());
    let last: serde_json::Value =
        serde_json::from_str(lines.last().unwrap()).expect("last JSON");
    assert_eq!(last["done"], true);
}

#[tokio::test]
async fn chat_unknown_model_returns_404() {
    let (status, body) = post_text(
        "/api/chat",
        serde_json::json!({
            "model": "no-such-model",
            "messages": [{"role": "user", "content": "hi"}],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body.contains("no-such-model"), "body: {body}");
}

#[tokio::test]
async fn show_requires_real_engine_returns_503_on_mock() {
    // `/api/show` needs the GGUF config (architecture, hyperparams, …)
    // which only `CpuEngine` exposes. The mock harness here doesn't
    // load a real engine, so the endpoint correctly degrades to 503.
    let (status, body) = post_text(
        "/api/show",
        serde_json::json!({"model": "rustllama-mock"}),
    )
    .await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "body: {body}");
}

// ----- /api/show against a real CpuEngine -----------------------------------
//
// Loads a synthetic Llama GGUF into a real [`CpuEngine`] and posts to
// `/api/show`. Verifies every documented response field that editors
// (Continue.dev, Cline, Cursor, Open WebUI) probe when they call this
// endpoint to learn a model's architecture / chat template / parameter
// scale. The mock-engine 503 sibling test above asserts the negative
// branch; this one asserts the positive shape.

fn build_state_with_real_cpu_engine(tag: &str) -> (rustllama_server::AppState, std::path::PathBuf) {
    use rustllama_engine::CpuEngine;
    use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};

    let tmp = std::env::temp_dir().join(format!("rustllama-ollama-show-{tag}.gguf"));
    write_synthetic_llama_gguf(&tmp, &SynthLlama::default());
    // `/api/show` surfaces the chat template embedded in the GGUF, so
    // load with a tokenizer (the bare `load` constructor skips it).
    let cpu = CpuEngine::load_with_tokenizer(&tmp, 16).expect("load cpu engine");
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
    let state = AppState::new(serving, "0.1.0-ollama-test".into());
    (state, tmp)
}

#[tokio::test]
async fn show_with_real_engine_returns_architecture_template_and_details() {
    let (state, tmp) = build_state_with_real_cpu_engine("primary");
    let app = router(state);

    // Empty body → `model: None`, `name: None`. Handler routes to the
    // default loaded model — easier than guessing the temp-file stem.
    let req = Request::builder()
        .method("POST")
        .uri("/api/show")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::json!({}).to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&bytes).expect("/api/show returned JSON");
    assert_eq!(status, StatusCode::OK, "body: {body}");

    // Top-level response fields per the Ollama /api/show shape.
    assert!(
        body["modelfile"]
            .as_str()
            .map(|s| s.contains("FROM"))
            .unwrap_or(false),
        "modelfile should carry a FROM directive: {body}"
    );
    assert!(body["parameters"].is_string());
    // The synth GGUF embeds a ChatML-style chat template; /api/show should
    // surface it as `template` so editors can preview prompt rendering.
    let template = body["template"].as_str().expect("template is string");
    assert!(
        template.contains("im_start") && template.contains("messages"),
        "template should be the synth ChatML, got: {template:?}"
    );

    // `details` block (used by clients to pick a model variant).
    let details = &body["details"];
    assert_eq!(details["format"], "gguf");
    assert!(details["family"].is_string());
    assert!(details["families"].is_array());
    assert!(details["parameter_size"].is_string());
    assert!(details["quantization_level"].is_string());

    // `model_info` block (architecture + dimensions, mirrors the GGUF
    // metadata keys real Ollama emits for ollama_show).
    let info = &body["model_info"];
    assert_eq!(info["general.architecture"], "llama");
    assert!(
        info["context_length"].as_u64().unwrap_or(0) > 0,
        "context_length should be > 0: {body}"
    );
    assert!(
        info["embedding_length"].as_u64().unwrap_or(0) > 0,
        "embedding_length should be > 0: {body}"
    );
    assert!(
        info["block_count"].as_u64().unwrap_or(0) > 0,
        "block_count should be > 0 (synth has 2 layers): {body}"
    );
    assert!(info["attention.head_count"].as_u64().unwrap_or(0) > 0);
    assert!(info["attention.head_count_kv"].as_u64().unwrap_or(0) > 0);

    let _ = std::fs::remove_file(&tmp);
}

#[tokio::test]
async fn show_accepts_name_alias_for_model() {
    // Real Ollama clients pass the model id under `name` OR `model`. The
    // adapter must accept either spelling. We post the actual derived
    // file-stem under `name` to confirm the alias is honored.
    let (state, tmp) = build_state_with_real_cpu_engine("name-alias");
    let derived_id = tmp.file_stem().and_then(|s| s.to_str()).unwrap().to_string();
    let app = router(state);

    let req = Request::builder()
        .method("POST")
        .uri("/api/show")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::json!({ "name": derived_id }).to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&bytes)
    );

    let _ = std::fs::remove_file(&tmp);
}

#[tokio::test]
async fn embeddings_returns_501_with_no_model_diagnostic() {
    // `/api/embeddings` (legacy Ollama spelling) shares the
    // `get_embedding_bundle` gating with `/v1/embeddings`: when
    // `[embeddings]` is unconfigured both surfaces return 501 with
    // the same structured diagnostic so clients can pattern-match
    // on a single error shape. Editors that probe `/api/embeddings`
    // for a capability check still get a stable signal.
    let (status, raw) = post_text(
        "/api/embeddings",
        serde_json::json!({
            "model": "rustllama-mock",
            "prompt": "embed me"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "body: {raw}");
    let body: serde_json::Value = serde_json::from_str(&raw).expect("handler returns JSON");
    assert!(
        body["error"]
            .as_str()
            .unwrap_or("")
            .contains("no embedding model configured"),
        "error field must surface the no-model-configured diagnostic: {body}"
    );
    assert!(
        body["detail"]
            .as_str()
            .unwrap_or("")
            .contains("[embeddings].path"),
        "detail field must point at the config knob: {body}"
    );
}

#[tokio::test]
async fn embed_returns_501_too() {
    // `/api/embed` is the modern Ollama spelling for the same shape.
    // Both routes should map to the same stub.
    let (status, _raw) = post_text(
        "/api/embed",
        serde_json::json!({ "model": "rustllama-mock", "input": "embed me" }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
}

/// `/api/ps` reports the loaded model with name, expires_at, size,
/// and details. Editor extensions hit this to verify a model is
/// resident before sending a chat request (`/api/tags` would also
/// match cached-but-cold models which would 503 on first use).
#[tokio::test]
async fn ps_lists_loaded_model_with_size_and_expires_at() {
    let (state, tmp) = build_state_with_real_cpu_engine("ps-loaded");
    let app = router(state);
    let req = Request::builder()
        .method("GET")
        .uri("/api/ps")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let body: serde_json::Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), 1 << 20).await.unwrap()).unwrap();
    assert_eq!(status, StatusCode::OK);
    let models = body["models"].as_array().expect("models array");
    assert!(
        !models.is_empty(),
        "/api/ps must list the loaded synth model: {body}"
    );
    let m = &models[0];
    assert!(m["name"].is_string(), "name field present");
    assert!(m["model"].is_string(), "model field present");
    assert!(
        m["size"].as_u64().unwrap_or(0) > 0,
        "size must reflect on-disk bytes: {m}"
    );
    // expires_at is a fixed far-future sentinel since rustllama
    // doesn't auto-expire models from the registry.
    assert!(
        m["expires_at"].as_str().unwrap_or("").starts_with("9999"),
        "expires_at must be far-future sentinel: {m}"
    );
    // details block mirrors /api/tags shape — clients reuse parsing.
    let details = &m["details"];
    assert_eq!(details["format"], "gguf");
    assert!(details["family"].is_string());
    let _ = std::fs::remove_file(&tmp);
}

/// `/api/ps` with no models resident → empty `models` array, not
/// an error. Editor extensions check `models.length === 0` to
/// decide whether to issue a `/api/pull`.
#[tokio::test]
async fn ps_with_mock_engine_skips_models_without_real_cpu_engine() {
    use rustllama_engine::MockEngine;
    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "mock-only".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    let state = AppState::new(serving, "0.0.0-ps-mock".into());
    let app = router(state);
    let req = Request::builder()
        .method("GET")
        .uri("/api/ps")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), 1 << 20).await.unwrap()).unwrap();
    let models = body["models"].as_array().expect("models array");
    // Mock-only entry has no cpu_engine, so it gets skipped.
    assert!(
        models.is_empty(),
        "/api/ps must skip mock entries without a CpuEngine: {body}"
    );
}

/// `/api/show` `modelfile` field carries a full Ollama-compatible
/// Modelfile: `FROM`, `TEMPLATE """..."""`, and `PARAMETER` lines.
/// Editor clients render this as the model's "config" — empty stubs
/// like just `FROM <id>` look broken vs. real Ollama output.
#[tokio::test]
async fn show_modelfile_carries_template_and_parameter_directives() {
    let (state, tmp) = build_state_with_real_cpu_engine("modelfile-block");
    let app = router(state);
    let req = Request::builder()
        .method("POST")
        .uri("/api/show")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::json!({}).to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), 1 << 20).await.unwrap()).unwrap();
    let modelfile = body["modelfile"].as_str().expect("modelfile string");
    assert!(modelfile.contains("FROM "), "modelfile must carry FROM: {modelfile:?}");
    assert!(
        modelfile.contains("TEMPLATE \"\"\""),
        "modelfile must carry triple-quoted TEMPLATE block: {modelfile:?}"
    );
    assert!(
        modelfile.contains("PARAMETER num_ctx"),
        "modelfile must carry PARAMETER num_ctx: {modelfile:?}"
    );
    assert!(
        modelfile.contains("PARAMETER stop"),
        "modelfile must carry PARAMETER stop: {modelfile:?}"
    );
    let _ = std::fs::remove_file(&tmp);
}

/// `/api/show` populates the `parameters` block with Ollama-style
/// `key value` pairs (`num_ctx`, `stop`). Empty strings would tell
/// editor clients "no advertisable defaults" — verify the synth
/// fixture exercises the populated path so the GUI / editor
/// integrations don't silently lose this info.
#[tokio::test]
async fn show_parameters_block_carries_num_ctx_and_stop() {
    let (state, tmp) = build_state_with_real_cpu_engine("parameters-block");
    let app = router(state);
    let req = Request::builder()
        .method("POST")
        .uri("/api/show")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::json!({}).to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), 1 << 20).await.unwrap()).unwrap();
    let params = body["parameters"].as_str().expect("parameters string");
    assert!(
        params.contains("num_ctx"),
        "/api/show parameters must include num_ctx line: {params:?}"
    );
    // `num_gpu` reflects the engine's hybrid-placement cutoff —
    // emitted even when "all on GPU" (clipped to n_layers).
    // Ollama clients display this in their model picker.
    assert!(
        params.contains("num_gpu"),
        "/api/show parameters must include num_gpu line: {params:?}"
    );
    // Synth tokenizer emits an EOS token, so `stop` should be present.
    assert!(
        params.contains("stop"),
        "/api/show parameters must include stop line when tokenizer has EOS: {params:?}"
    );
    // Stop values are JSON-encoded — confirm we get a quoted string,
    // not a raw word that would break Modelfile parsers when the EOS
    // contains whitespace or special characters.
    for line in params.lines() {
        if let Some(rest) = line.strip_prefix("stop ") {
            assert!(
                rest.starts_with('"') && rest.ends_with('"'),
                "stop value must be JSON-quoted: {line:?}"
            );
        }
    }
    let _ = std::fs::remove_file(&tmp);
}

/// Build state with a real CpuEngine loaded from a MoE synthetic
/// GGUF. Same shape as `build_state_with_real_cpu_engine` — the only
/// difference is the GGUF carries MoE expert metadata + tensors.
fn build_state_with_moe_cpu_engine(
    tag: &str,
    n_experts: u32,
    n_experts_used: u32,
    n_experts_shared: u32,
) -> (rustllama_server::AppState, std::path::PathBuf) {
    use rustllama_engine::CpuEngine;
    use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama, SynthMoe};

    let tmp = std::env::temp_dir().join(format!("rustllama-ollama-show-moe-{tag}.gguf"));
    write_synthetic_llama_gguf(
        &tmp,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts,
                n_experts_used,
                n_experts_shared,
            }),
            ..SynthLlama::default()
        },
    );
    let cpu = CpuEngine::load_with_tokenizer(&tmp, 16).expect("load moe cpu engine");
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
    let state = AppState::new(serving, "0.1.0-ollama-test".into());
    (state, tmp)
}

/// `/api/show` surfaces `{arch}.expert_count` / `expert_used_count` /
/// `expert_shared_count` in the `model_info` block when the loaded
/// model is MoE — matching the GGUF metadata convention that real
/// Ollama clients (which special-case Mixtral / DeepSeek-V3) expect.
#[tokio::test]
async fn show_moe_includes_expert_count_keys_in_model_info() {
    let (state, tmp) = build_state_with_moe_cpu_engine("expert-keys", 8, 2, 0);
    let app = router(state);
    let req = Request::builder()
        .method("POST")
        .uri("/api/show")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::json!({}).to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let body: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    assert_eq!(status, StatusCode::OK, "body: {body}");
    let info = &body["model_info"];
    assert_eq!(
        info["llama.expert_count"].as_u64(),
        Some(8),
        "/api/show must surface MoE expert_count: {body}"
    );
    assert_eq!(info["llama.expert_used_count"].as_u64(), Some(2));
    assert_eq!(info["llama.expert_shared_count"].as_u64(), Some(0));
    let _ = std::fs::remove_file(&tmp);
}

/// DeepSeek-V3-style: shared-expert count > 0 round-trips.
#[tokio::test]
async fn show_moe_with_shared_experts_surfaces_shared_count() {
    let (state, tmp) = build_state_with_moe_cpu_engine("deepseek-shared", 4, 2, 1);
    let app = router(state);
    let req = Request::builder()
        .method("POST")
        .uri("/api/show")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::json!({}).to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), 1 << 20).await.unwrap()).unwrap();
    assert_eq!(body["model_info"]["llama.expert_shared_count"].as_u64(), Some(1));
    let _ = std::fs::remove_file(&tmp);
}

/// Dense models must NOT carry the MoE keys. Clients use key
/// presence to branch on MoE-aware UI, so this is a stability gate
/// for that contract.
#[tokio::test]
async fn show_dense_omits_moe_keys_from_model_info() {
    let (state, tmp) = build_state_with_real_cpu_engine("dense-no-moe-keys");
    let app = router(state);
    let req = Request::builder()
        .method("POST")
        .uri("/api/show")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::json!({}).to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&to_bytes(resp.into_body(), 1 << 20).await.unwrap()).unwrap();
    let info = body["model_info"].as_object().unwrap();
    assert!(
        !info.contains_key("llama.expert_count"),
        "dense /api/show must not include expert_count: {body}"
    );
    assert!(!info.contains_key("llama.expert_used_count"));
    assert!(!info.contains_key("llama.expert_shared_count"));
    let _ = std::fs::remove_file(&tmp);
}

#[tokio::test]
async fn show_unknown_model_returns_404() {
    // Mismatched model id under the real-engine harness must still 404
    // (rather than e.g. silently substituting the default model).
    let (state, tmp) = build_state_with_real_cpu_engine("unknown-404");
    let app = router(state);
    let req = Request::builder()
        .method("POST")
        .uri("/api/show")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::json!({ "model": "definitely-not-loaded" }).to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let _ = std::fs::remove_file(&tmp);
}
