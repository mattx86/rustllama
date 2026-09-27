//! Integration tests for `POST /v1/embeddings` + the Ollama-shaped
//! `/api/embeddings` / `/api/embed`. End-to-end coverage of:
//!   - empty input → 400 with a clear message
//!   - no [embeddings] configured → 501 with diagnostic
//!   - configured + text input → 200 with real vectors (tokenizer
//!     loaded from the same GGUF as the model)
//!   - configured + pre-tokenized int input → 200 with real vectors
//!   - configured + batch of text inputs → 200 with per-input vectors
//!   - mid-batch forward error → 400 surfacing input_index

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::MockEngine;
use rustllama_server::{router, AppState, ServingModel};
use tower::ServiceExt;

fn build_state(tag: &str, config_toml: &str) -> (AppState, std::path::PathBuf) {
    let tmp = std::env::temp_dir().join(format!("rustllama-embeddings-{tag}.toml"));
    std::fs::write(&tmp, config_toml).expect("write config");
    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "embeddings-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    let state = AppState::new(serving, "0.0.0-test".into()).with_config_path(tmp.clone());
    (state, tmp)
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

/// `[embeddings]` block absent → 501 with "no embedding model
/// configured" + the config-field hint. The endpoint is reachable
/// (not 404) so clients probing for embedding capability get a
/// structured signal.
#[tokio::test]
async fn embeddings_returns_501_with_no_model_hint_when_unconfigured() {
    let (state, cfg) = build_state("unconfigured", "[server]\nport = 11434\n");
    let app = router(state);
    let (status, body) = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({ "input": "hello world", "model": "any" }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("no embedding model configured"),
        "body should name the missing config: {body_str}"
    );
    assert!(
        body_str.contains("[embeddings]"),
        "body should point at the right config field: {body_str}"
    );
    let _ = std::fs::remove_file(&cfg);
}

/// `[embeddings].path` set but slot not enabled in state → still
/// 501 "no embedding model configured" (slot is what activates
/// the load path). Pins the contract: state.with_embedding_slot()
/// is what flips the gate, not just the config field. This is
/// the path test fixtures hit when they construct state by hand
/// without calling `.with_embedding_slot()`.
#[tokio::test]
async fn embeddings_unconfigured_slot_returns_501_even_when_config_set() {
    let (state, cfg) = build_state(
        "config-set-slot-unset",
        r#"
[server]
port = 11434

[embeddings]
path = "C:/models/some-bert-model.gguf"
"#,
    );
    let app = router(state);
    let (status, body) = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({ "input": [1, 2, 3] }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("no embedding model configured"),
        "501 body: {body_str}"
    );
    let _ = std::fs::remove_file(&cfg);
}

/// Empty input → 400 (not 501). This is a wiring error the client
/// can fix without needing an embedding model loaded.
#[tokio::test]
async fn embeddings_rejects_empty_input_with_400() {
    let (state, cfg) = build_state("empty-batch", "[server]\nport = 11434\n");
    let app = router(state);
    let (status, body) = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({ "input": [] }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("non-empty"),
        "400 must explain why: {body_str}"
    );
    let _ = std::fs::remove_file(&cfg);
}

/// Both `"input": "single"` and `"input": ["batch"]` shapes parse
/// cleanly. The 501 body fires either way (no loader yet), but the
/// 400 path for malformed input is what we're guarding against
/// here — accepting either OpenAI-supported shape.
#[tokio::test]
async fn embeddings_accepts_both_single_and_batch_input_shapes() {
    let (state, cfg) = build_state("input-shapes", "[server]\nport = 11434\n");

    let app = router(state.clone());
    let (s1, _) = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({ "input": "single string" }),
    )
    .await;
    assert_eq!(s1, StatusCode::NOT_IMPLEMENTED, "single-string parses");

    let app = router(state);
    let (s2, _) = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({ "input": ["a", "b", "c"] }),
    )
    .await;
    assert_eq!(s2, StatusCode::NOT_IMPLEMENTED, "batch parses");

    let _ = std::fs::remove_file(&cfg);
}

/// Pre-tokenized int input on a real configured embedding model
/// → 200 with real vectors. The synth BERT GGUF + the
/// `forward_embed` implementation from the prior turn are the
/// engine; this test verifies the full path through the HTTP
/// endpoint + lazy-load + response shape.
#[tokio::test]
async fn embeddings_returns_real_vectors_for_pretokenized_input() {
    use rustllama_gguf::synth::{write_synthetic_bert_gguf, SynthBert};
    use rustllama_server::AppState;
    use std::sync::Arc;
    use rustllama_engine::MockEngine;
    use rustllama_server::ServingModel;

    let model_path = std::env::temp_dir().join("rustllama-embeddings-real-bert.gguf");
    let synth = SynthBert::default();
    write_synthetic_bert_gguf(&model_path, &synth);

    // Build a config that points at the synth BERT GGUF. Use
    // string-escape for the path so TOML accepts Windows
    // backslashes.
    let cfg_path = std::env::temp_dir().join("rustllama-embeddings-real-cfg.toml");
    let cfg_toml = format!(
        r#"
[server]
port = 11434

[embeddings]
path = "{}"
"#,
        model_path.display().to_string().replace('\\', "\\\\"),
    );
    std::fs::write(&cfg_path, cfg_toml).expect("write config");

    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "embeddings-real-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    let state = AppState::new(serving, "0.0.0-test".into())
        .with_config_path(cfg_path.clone())
        .with_embedding_slot();
    let app = router(state);

    let tokens = vec![0i32, 1, 2, 3, 4]; // within synth vocab (32) + ctx (32)
    let (status, body) = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({ "input": tokens, "model": "bge-test" }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(v["object"], "list");
    assert_eq!(v["model"], "bge-test", "model field echoes the request");
    let data = v["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1, "single input → single entry");
    assert_eq!(data[0]["object"], "embedding");
    assert_eq!(data[0]["index"], 0);
    let embedding = data[0]["embedding"].as_array().expect("embedding array");
    assert_eq!(
        embedding.len(),
        synth.d_model as usize,
        "embedding length matches model d_model"
    );
    for (i, val) in embedding.iter().enumerate() {
        let f = val.as_f64().expect("embedding entries are numbers");
        assert!(
            f.is_finite(),
            "embedding[{i}] = {f} is not finite — forward produced NaN/Inf"
        );
    }
    let usage = &v["usage"];
    assert_eq!(usage["prompt_tokens"], 5);
    assert_eq!(usage["total_tokens"], 5);

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&cfg_path);
}

/// Batch of pre-tokenized inputs → 200 with N entries, indices
/// matching input order, each carrying its own d_model vector.
#[tokio::test]
async fn embeddings_batch_pretokenized_returns_per_input_vectors() {
    use rustllama_gguf::synth::{write_synthetic_bert_gguf, SynthBert};
    use rustllama_server::AppState;
    use std::sync::Arc;
    use rustllama_engine::MockEngine;
    use rustllama_server::ServingModel;

    let model_path = std::env::temp_dir().join("rustllama-embeddings-batch-bert.gguf");
    let synth = SynthBert::default();
    write_synthetic_bert_gguf(&model_path, &synth);

    let cfg_path = std::env::temp_dir().join("rustllama-embeddings-batch-cfg.toml");
    std::fs::write(
        &cfg_path,
        format!(
            "[server]\nport = 11434\n\n[embeddings]\npath = \"{}\"\n",
            model_path.display().to_string().replace('\\', "\\\\"),
        ),
    )
    .expect("write config");

    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "embeddings-batch-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    let state = AppState::new(serving, "0.0.0-test".into())
        .with_config_path(cfg_path.clone())
        .with_embedding_slot();
    let app = router(state);

    let batch = vec![vec![1i32, 2, 3], vec![4, 5, 6, 7], vec![8]];
    let (status, body) = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({ "input": batch }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    let data = v["data"].as_array().unwrap();
    assert_eq!(data.len(), 3);
    for (i, entry) in data.iter().enumerate() {
        assert_eq!(entry["index"], i);
        assert_eq!(
            entry["embedding"].as_array().unwrap().len(),
            synth.d_model as usize
        );
    }
    // Total tokens = 3 + 4 + 1.
    assert_eq!(v["usage"]["total_tokens"], 8);

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&cfg_path);
}

/// Text input on a configured model → 200 with real vectors. The
/// WordPiece tokenizer loaded from the same GGUF runs the
/// `[CLS] … [SEP]` wrap + lookup, then the BERT forward + pooling
/// produces the vector. End-to-end exercise of the text path
/// the OpenAI-shape clients (Continue, Aider, Cline, langchain)
/// rely on.
#[tokio::test]
async fn embeddings_text_input_returns_real_vectors() {
    use rustllama_gguf::synth::{write_synthetic_bert_gguf, SynthBert};
    use rustllama_server::AppState;
    use std::sync::Arc;
    use rustllama_engine::MockEngine;
    use rustllama_server::ServingModel;

    let model_path = std::env::temp_dir().join("rustllama-embeddings-text-bert.gguf");
    let synth = SynthBert::default();
    write_synthetic_bert_gguf(&model_path, &synth);
    let cfg_path = std::env::temp_dir().join("rustllama-embeddings-text-cfg.toml");
    std::fs::write(
        &cfg_path,
        format!(
            "[server]\nport = 11434\n\n[embeddings]\npath = \"{}\"\n",
            model_path.display().to_string().replace('\\', "\\\\"),
        ),
    )
    .unwrap();

    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "embeddings-text-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    let state = AppState::new(serving, "0.0.0-test".into())
        .with_config_path(cfg_path.clone())
        .with_embedding_slot();
    let app = router(state);

    // "hello world" is in the synth vocab, so it round-trips
    // cleanly through WordPiece + the BERT forward pass.
    let (status, body) = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({ "input": "hello world", "model": "bge-text-test" }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body: {}",
        String::from_utf8_lossy(&body)
    );
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(v["object"], "list");
    assert_eq!(v["model"], "bge-text-test");
    let data = v["data"].as_array().expect("data array");
    assert_eq!(data.len(), 1, "single text input → one entry");
    let embedding = data[0]["embedding"].as_array().expect("embedding array");
    assert_eq!(embedding.len(), synth.d_model as usize);
    for (i, val) in embedding.iter().enumerate() {
        let f = val.as_f64().expect("embedding entries are numbers");
        assert!(f.is_finite(), "embedding[{i}] = {f} is not finite");
    }
    // Total tokens for "hello world" with [CLS] / [SEP] wrap = 4.
    let total = v["usage"]["total_tokens"].as_u64().expect("total_tokens u64");
    assert!(
        total >= 3,
        "expected at least [CLS] + word + [SEP] = 3 tokens, got {total}"
    );

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&cfg_path);
}

/// Batch of text inputs → 200 with per-input vectors.
#[tokio::test]
async fn embeddings_batch_text_returns_per_input_vectors() {
    use rustllama_gguf::synth::{write_synthetic_bert_gguf, SynthBert};
    use rustllama_server::AppState;
    use std::sync::Arc;
    use rustllama_engine::MockEngine;
    use rustllama_server::ServingModel;

    let model_path = std::env::temp_dir().join("rustllama-embeddings-batchtext-bert.gguf");
    let synth = SynthBert::default();
    write_synthetic_bert_gguf(&model_path, &synth);
    let cfg_path = std::env::temp_dir().join("rustllama-embeddings-batchtext-cfg.toml");
    std::fs::write(
        &cfg_path,
        format!(
            "[server]\nport = 11434\n\n[embeddings]\npath = \"{}\"\n",
            model_path.display().to_string().replace('\\', "\\\\"),
        ),
    )
    .unwrap();

    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "embeddings-batchtext-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    let state = AppState::new(serving, "0.0.0-test".into())
        .with_config_path(cfg_path.clone())
        .with_embedding_slot();
    let app = router(state);

    let (status, body) = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({ "input": ["hello world", "test input", "the quick brown fox"] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    let data = v["data"].as_array().unwrap();
    assert_eq!(data.len(), 3);
    for (i, entry) in data.iter().enumerate() {
        assert_eq!(entry["index"], i);
        assert_eq!(
            entry["embedding"].as_array().unwrap().len(),
            synth.d_model as usize
        );
    }

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&cfg_path);
}

/// Forward-pass-level errors (e.g. token id out of vocab) surface
/// as 400 with the input_index so the client knows which entry
/// failed without re-running the whole batch.
#[tokio::test]
async fn embeddings_forward_error_surfaces_input_index() {
    use rustllama_gguf::synth::{write_synthetic_bert_gguf, SynthBert};
    use rustllama_server::AppState;
    use std::sync::Arc;
    use rustllama_engine::MockEngine;
    use rustllama_server::ServingModel;

    let model_path = std::env::temp_dir().join("rustllama-embeddings-err-bert.gguf");
    let synth = SynthBert::default();
    write_synthetic_bert_gguf(&model_path, &synth);
    let cfg_path = std::env::temp_dir().join("rustllama-embeddings-err-cfg.toml");
    std::fs::write(
        &cfg_path,
        format!(
            "[server]\nport = 11434\n\n[embeddings]\npath = \"{}\"\n",
            model_path.display().to_string().replace('\\', "\\\\"),
        ),
    )
    .unwrap();

    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "embeddings-err-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    let state = AppState::new(serving, "0.0.0-test".into())
        .with_config_path(cfg_path.clone())
        .with_embedding_slot();
    let app = router(state);

    // Second entry has a token id beyond synth vocab (32) →
    // TokenOutOfVocab → 400 with input_index = 1.
    let batch = vec![vec![1i32, 2], vec![1, 999]];
    let (status, body) = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({ "input": batch }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(
        v["input_index"], 1,
        "error body must name which input failed"
    );

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&cfg_path);
}

/// The Ollama-shaped stubs (`/api/embeddings` + `/api/embed`)
/// share the same gating — when no model is configured they
/// return the "not configured" message, when configured they
/// return the "loader pending" message. Keeps consistency across
/// the two API surfaces so Ollama-protocol clients get the same
/// diagnostics as OpenAI-protocol ones.
#[tokio::test]
async fn ollama_embeddings_stub_reflects_embeddings_config() {
    let (state, cfg) = build_state(
        "ollama-configured",
        r#"
[server]
port = 11434

[embeddings]
hub = "BAAI/bge-small-en-v1.5:bge-small-en-v1.5-q4_k_m.gguf"
"#,
    );
    let app = router(state.clone());
    let (status, body) = post_json(
        app,
        "/api/embeddings",
        serde_json::json!({ "model": "any", "prompt": "hello" }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("no embedding model configured"),
        "ollama legacy stub must surface the no-model-configured diagnostic: {body_str}"
    );

    // /api/embed (current Ollama spelling) shares the same gating.
    let app = router(state);
    let (status, body) = post_json(
        app,
        "/api/embed",
        serde_json::json!({ "model": "any", "input": "hello" }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("no embedding model configured"),
        "ollama modern stub must surface the no-model-configured diagnostic: {body_str}"
    );

    let _ = std::fs::remove_file(&cfg);
}

// --- Ollama success-path adapters --------------------------------------------

/// Helper that mirrors the inline pattern in the success-path tests
/// above: writes a synth BERT GGUF + a config that points at it,
/// and returns the configured AppState with the embedding slot
/// enabled.
fn build_state_with_synth_bert(tag: &str) -> (AppState, std::path::PathBuf, std::path::PathBuf) {
    use rustllama_gguf::synth::{write_synthetic_bert_gguf, SynthBert};
    let model_path =
        std::env::temp_dir().join(format!("rustllama-embeddings-ollama-{tag}-bert.gguf"));
    write_synthetic_bert_gguf(&model_path, &SynthBert::default());
    let cfg_path = std::env::temp_dir().join(format!("rustllama-embeddings-ollama-{tag}-cfg.toml"));
    std::fs::write(
        &cfg_path,
        format!(
            "[server]\nport = 11434\n\n[embeddings]\npath = \"{}\"\n",
            model_path.display().to_string().replace('\\', "\\\\"),
        ),
    )
    .expect("write config");

    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: format!("embeddings-ollama-{tag}-mock"),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    let state = AppState::new(serving, "0.0.0-test".into())
        .with_config_path(cfg_path.clone())
        .with_embedding_slot();
    (state, model_path, cfg_path)
}

/// Legacy Ollama `POST /api/embeddings` returns the flat `{embedding}`
/// shape with a vector of d_model floats.
#[tokio::test]
async fn ollama_legacy_embeddings_returns_flat_vector() {
    let (state, model_path, cfg_path) = build_state_with_synth_bert("legacy-success");
    let app = router(state);
    let (status, body) = post_json(
        app,
        "/api/embeddings",
        serde_json::json!({ "model": "rustllama-embeddings", "prompt": "hello world" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    let embedding = v["embedding"].as_array().expect("flat embedding array");
    assert_eq!(embedding.len(), 64, "synth d_model is 64");
    for f in embedding {
        assert!(f.as_f64().unwrap().is_finite());
    }
    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&cfg_path);
}

/// Modern Ollama `POST /api/embed` with a single string returns
/// the batched `{embeddings: [[...]]}` shape + duration / token
/// count metadata.
#[tokio::test]
async fn ollama_modern_embed_single_returns_batched_shape() {
    let (state, model_path, cfg_path) = build_state_with_synth_bert("modern-single");
    let app = router(state);
    let (status, body) = post_json(
        app,
        "/api/embed",
        serde_json::json!({ "model": "rustllama-embeddings", "input": "hello world" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(v["model"], "rustllama-embeddings");
    let embeddings = v["embeddings"].as_array().expect("nested embeddings array");
    assert_eq!(embeddings.len(), 1, "single input → one nested vector");
    assert_eq!(embeddings[0].as_array().unwrap().len(), 64);
    // Metadata fields the Ollama clients (langchain, etc.) read.
    assert!(v["total_duration"].is_u64());
    assert!(v["load_duration"].is_u64());
    let count = v["prompt_eval_count"].as_u64().expect("prompt_eval_count u64");
    assert!(count >= 3, "[CLS] + tokens + [SEP] ≥ 3 tokens, got {count}");
    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&cfg_path);
}

/// Modern Ollama `POST /api/embed` with an input array returns one
/// vector per input. Order is preserved.
#[tokio::test]
async fn ollama_modern_embed_batch_returns_per_input_vectors() {
    let (state, model_path, cfg_path) = build_state_with_synth_bert("modern-batch");
    let app = router(state);
    let (status, body) = post_json(
        app,
        "/api/embed",
        serde_json::json!({
            "model": "rustllama-embeddings",
            "input": ["hello world", "the quick brown fox", "test input"],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    let embeddings = v["embeddings"].as_array().unwrap();
    assert_eq!(embeddings.len(), 3);
    for embedding in embeddings {
        assert_eq!(embedding.as_array().unwrap().len(), 64);
    }
    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&cfg_path);
}

/// Modern Ollama `POST /api/embed` with an empty input → 400, not
/// a forward-pass error. The legacy shape with an empty prompt
/// behaves the same.
#[tokio::test]
async fn ollama_embed_rejects_empty_input_with_400() {
    let (state, model_path, cfg_path) = build_state_with_synth_bert("empty-input");
    let app = router(state.clone());
    let (status, _) = post_json(
        app,
        "/api/embed",
        serde_json::json!({ "input": "" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let app = router(state.clone());
    let (status, _) = post_json(
        app,
        "/api/embed",
        serde_json::json!({ "input": [] }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let app = router(state);
    let (status, _) = post_json(
        app,
        "/api/embeddings",
        serde_json::json!({ "prompt": "" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&cfg_path);
}

// --- /v1/models capability advertisement -------------------------------------

/// `GET /v1/models` includes a `rustllama-embeddings` entry with
/// `purpose: "embedding"` when the slot is enabled. Lets clients
/// discover the endpoint without probing `/v1/embeddings`. The
/// `dimensions` field is `null` until first /v1/embeddings request
/// triggers the lazy load; we don't force the load on a discovery
/// call.
#[tokio::test]
async fn v1_models_advertises_embedding_capability_when_slot_enabled() {
    let (state, model_path, cfg_path) = build_state_with_synth_bert("v1-models-advertise");
    let app = router(state);
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    let data = v["data"].as_array().expect("data array");
    let embed_entry = data
        .iter()
        .find(|m| m["id"] == "rustllama-embeddings")
        .expect("embedding capability must be advertised");
    assert_eq!(embed_entry["purpose"], "embedding");
    assert_eq!(embed_entry["object"], "model");
    // Pre-load: dimensions is null (the field is skipped via
    // skip_serializing_if when None, so it's just absent).
    assert!(
        embed_entry.get("dimensions").is_none()
            || embed_entry["dimensions"].is_null(),
        "dimensions must be absent before first embed call: {embed_entry}"
    );

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&cfg_path);
}

/// After a `/v1/embeddings` call has loaded the model, `/v1/models`
/// surfaces `dimensions` so RAG clients can validate vector-store
/// dimension upfront.
#[tokio::test]
async fn v1_models_reports_dimensions_after_first_embed_load() {
    let (state, model_path, cfg_path) = build_state_with_synth_bert("v1-models-dims");

    // Force the lazy-load by calling /v1/embeddings once.
    let app = router(state.clone());
    let (status, _) = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({ "input": "hello world" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // Now /v1/models should report dimensions.
    let app = router(state);
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    let data = v["data"].as_array().unwrap();
    let embed_entry = data
        .iter()
        .find(|m| m["id"] == "rustllama-embeddings")
        .expect("embedding capability must be advertised");
    assert_eq!(embed_entry["dimensions"], 64, "synth d_model is 64");

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&cfg_path);
}

/// `/v1/models` omits the embedding entry entirely when the slot
/// is not enabled. Backwards-compat: chat-only clients see exactly
/// what they saw before this change.
#[tokio::test]
async fn v1_models_omits_embedding_entry_when_slot_disabled() {
    let (state, cfg) = build_state("v1-models-no-slot", "[server]\nport = 11434\n");
    let app = router(state);
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    let data = v["data"].as_array().unwrap();
    assert!(
        data.iter().all(|m| m["id"] != "rustllama-embeddings"),
        "embedding entry must NOT appear when slot disabled: {data:?}"
    );
    let _ = std::fs::remove_file(&cfg);
}

// --- OpenAI spec extras: encoding_format + dimensions ----------------------

/// `encoding_format: "base64"` returns each vector as a single LE-f32
/// base64 string instead of a JSON array. Decoding the string with
/// the standard base64 alphabet and re-interpreting as LE-f32 must
/// reproduce the same vector the "float" path would have returned.
#[tokio::test]
async fn embeddings_base64_encoding_returns_string_per_vector() {
    use rustllama_gguf::synth::{write_synthetic_bert_gguf, SynthBert};
    let model_path = std::env::temp_dir().join("rustllama-embeddings-b64-bert.gguf");
    let synth = SynthBert::default();
    write_synthetic_bert_gguf(&model_path, &synth);
    let cfg_path = std::env::temp_dir().join("rustllama-embeddings-b64-cfg.toml");
    std::fs::write(
        &cfg_path,
        format!(
            "[server]\nport = 11434\n\n[embeddings]\npath = \"{}\"\n",
            model_path.display().to_string().replace('\\', "\\\\"),
        ),
    )
    .unwrap();
    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "embeddings-b64-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    let state = AppState::new(serving, "0.0.0-test".into())
        .with_config_path(cfg_path.clone())
        .with_embedding_slot();

    // First call: float format (default). Capture the vector.
    let app = router(state.clone());
    let (status, body) = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({ "input": "hello world", "encoding_format": "float" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let float_vec: Vec<f64> = v["data"][0]["embedding"]
        .as_array()
        .expect("float format → JSON array")
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    assert_eq!(float_vec.len(), synth.d_model as usize);

    // Second call: same input, base64 format. Decode the string and
    // verify it byte-matches the float-format LE-f32 serialization.
    let app = router(state);
    let (status, body) = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({ "input": "hello world", "encoding_format": "base64" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let b64 = v["data"][0]["embedding"]
        .as_str()
        .expect("base64 format → JSON string");
    let decoded_bytes = decode_base64(b64);
    assert_eq!(
        decoded_bytes.len(),
        float_vec.len() * 4,
        "decoded base64 must hold one LE-f32 per dim"
    );
    for (i, chunk) in decoded_bytes.chunks_exact(4).enumerate() {
        let f = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let expected = float_vec[i] as f32;
        assert!(
            (f - expected).abs() < 1e-6,
            "vec[{i}]: base64-decoded {f} != float-form {expected}"
        );
    }

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&cfg_path);
}

/// `encoding_format` with an unknown value → 400 with a clear msg
/// naming the only two accepted values.
#[tokio::test]
async fn embeddings_rejects_unknown_encoding_format_with_400() {
    let (state, cfg) = build_state("bad-encoding", "[server]\nport = 11434\n");
    let app = router(state);
    let (status, body) = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({ "input": "hi", "encoding_format": "xml" }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("encoding_format"),
        "body must name the offending field: {body_str}"
    );
    let _ = std::fs::remove_file(&cfg);
}

/// `dimensions` truncates the vector to the first N entries and
/// L2-renormalizes (the standard Matryoshka / MRL convention).
#[tokio::test]
async fn embeddings_dimensions_truncates_and_renormalizes() {
    use rustllama_gguf::synth::{write_synthetic_bert_gguf, SynthBert};
    let model_path = std::env::temp_dir().join("rustllama-embeddings-mrl-bert.gguf");
    let synth = SynthBert::default();
    write_synthetic_bert_gguf(&model_path, &synth);
    let cfg_path = std::env::temp_dir().join("rustllama-embeddings-mrl-cfg.toml");
    std::fs::write(
        &cfg_path,
        format!(
            "[server]\nport = 11434\n\n[embeddings]\npath = \"{}\"\n",
            model_path.display().to_string().replace('\\', "\\\\"),
        ),
    )
    .unwrap();
    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "embeddings-mrl-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    let state = AppState::new(serving, "0.0.0-test".into())
        .with_config_path(cfg_path.clone())
        .with_embedding_slot();

    let app = router(state);
    let (status, body) = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({ "input": "hello world", "dimensions": 16 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let v: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let embedding: Vec<f64> = v["data"][0]["embedding"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    assert_eq!(embedding.len(), 16, "truncated to dimensions=16");
    let norm: f64 = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "truncated vector must be L2-renormalized; got norm {norm}"
    );
    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&cfg_path);
}

/// `dimensions = 0` or `dimensions > d_model` → 400 with a clear
/// message naming the valid range.
#[tokio::test]
async fn embeddings_rejects_invalid_dimensions_with_400() {
    use rustllama_gguf::synth::{write_synthetic_bert_gguf, SynthBert};
    let model_path = std::env::temp_dir().join("rustllama-embeddings-mrl-bad-bert.gguf");
    let synth = SynthBert::default();
    write_synthetic_bert_gguf(&model_path, &synth);
    let cfg_path = std::env::temp_dir().join("rustllama-embeddings-mrl-bad-cfg.toml");
    std::fs::write(
        &cfg_path,
        format!(
            "[server]\nport = 11434\n\n[embeddings]\npath = \"{}\"\n",
            model_path.display().to_string().replace('\\', "\\\\"),
        ),
    )
    .unwrap();
    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "embeddings-mrl-bad-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    let state = AppState::new(serving, "0.0.0-test".into())
        .with_config_path(cfg_path.clone())
        .with_embedding_slot();

    // dimensions = 0 → 400
    let app = router(state.clone());
    let (status, body) = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({ "input": "hi", "dimensions": 0 }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body_str = String::from_utf8_lossy(&body);
    assert!(body_str.contains("dimensions"), "{body_str}");

    // dimensions > d_model (synth is 64) → 400
    let app = router(state);
    let (status, body) = post_json(
        app,
        "/v1/embeddings",
        serde_json::json!({ "input": "hi", "dimensions": (synth.d_model as u32) + 1 }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body_str = String::from_utf8_lossy(&body);
    assert!(body_str.contains("dimensions"), "{body_str}");

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&cfg_path);
}

/// Decode a standard-base64 string into bytes. Inline so the test
/// file stays self-contained (the test crate has no base64 dep).
fn decode_base64(s: &str) -> Vec<u8> {
    fn val(b: u8) -> Option<u32> {
        match b {
            b'A'..=b'Z' => Some((b - b'A') as u32),
            b'a'..=b'z' => Some((b - b'a' + 26) as u32),
            b'0'..=b'9' => Some((b - b'0' + 52) as u32),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    let mut i = 0;
    while i + 4 <= bytes.len() {
        let mut n: u32 = 0;
        let mut pad = 0u32;
        for j in 0..4 {
            let b = bytes[i + j];
            if b == b'=' {
                pad += 1;
                n <<= 6;
            } else {
                n = (n << 6) | val(b).expect("valid base64 char");
            }
        }
        out.push(((n >> 16) & 0xff) as u8);
        if pad < 2 {
            out.push(((n >> 8) & 0xff) as u8);
        }
        if pad < 1 {
            out.push((n & 0xff) as u8);
        }
        i += 4;
    }
    out
}
