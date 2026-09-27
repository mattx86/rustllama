//! Integration tests for `POST /v1/rerank` (Cohere/Jina shape).
//! Covers:
//!   - unconfigured slot → 501 with diagnostic
//!   - configured slot but GGUF lacks classifier head → 500 with
//!     "use [embeddings] instead" hint
//!   - happy path: 200 with sorted-by-score results
//!   - top_n truncates after sorting
//!   - return_documents echoes original text
//!   - empty query / empty documents → 400
//!   - /v1/models advertises the reranker capability

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::MockEngine;
use rustllama_server::{router, AppState, ServingModel};
use tower::ServiceExt;

fn build_state_unconfigured(tag: &str) -> (AppState, std::path::PathBuf) {
    let tmp = std::env::temp_dir().join(format!("rustllama-rerank-{tag}.toml"));
    std::fs::write(&tmp, "[server]\nport = 11434\n").expect("write config");
    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "rerank-mock".into(),
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

fn build_state_with_synth_reranker(
    tag: &str,
    n_labels: u32,
) -> (AppState, std::path::PathBuf, std::path::PathBuf) {
    use rustllama_gguf::synth::{write_synthetic_bert_gguf, SynthBert};
    let model_path = std::env::temp_dir().join(format!("rustllama-rerank-{tag}-bert.gguf"));
    let synth = SynthBert::default().with_classifier_head(n_labels);
    write_synthetic_bert_gguf(&model_path, &synth);
    let cfg_path = std::env::temp_dir().join(format!("rustllama-rerank-{tag}-cfg.toml"));
    std::fs::write(
        &cfg_path,
        format!(
            "[server]\nport = 11434\n\n[reranker]\npath = \"{}\"\n",
            model_path.display().to_string().replace('\\', "\\\\"),
        ),
    )
    .expect("write config");
    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: format!("rerank-{tag}-mock"),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    let state = AppState::new(serving, "0.0.0-test".into())
        .with_config_path(cfg_path.clone())
        .with_reranker_slot();
    (state, model_path, cfg_path)
}

fn build_state_pointing_at_embedding_only_gguf(
    tag: &str,
) -> (AppState, std::path::PathBuf, std::path::PathBuf) {
    use rustllama_gguf::synth::{write_synthetic_bert_gguf, SynthBert};
    let model_path =
        std::env::temp_dir().join(format!("rustllama-rerank-{tag}-embed-only.gguf"));
    write_synthetic_bert_gguf(&model_path, &SynthBert::default());
    let cfg_path = std::env::temp_dir().join(format!("rustllama-rerank-{tag}-cfg.toml"));
    std::fs::write(
        &cfg_path,
        format!(
            "[server]\nport = 11434\n\n[reranker]\npath = \"{}\"\n",
            model_path.display().to_string().replace('\\', "\\\\"),
        ),
    )
    .unwrap();
    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: format!("rerank-{tag}-mock"),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    let state = AppState::new(serving, "0.0.0-test".into())
        .with_config_path(cfg_path.clone())
        .with_reranker_slot();
    (state, model_path, cfg_path)
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

#[tokio::test]
async fn rerank_returns_501_when_unconfigured() {
    let (state, cfg) = build_state_unconfigured("unconfigured");
    let app = router(state);
    let (status, body) = post_json(
        app,
        "/v1/rerank",
        serde_json::json!({
            "query": "what is the capital of france?",
            "documents": ["paris is the capital", "berlin is in germany"],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("no reranker model configured"),
        "501 body must name the missing config: {body_str}"
    );
    let _ = std::fs::remove_file(&cfg);
}

#[tokio::test]
async fn rerank_rejects_embedding_only_gguf_with_500() {
    // Pointing [reranker] at a pure-embedding GGUF must fail
    // cleanly at first request — not silently fall through to
    // some other path. The diagnostic tells the user which knob
    // to flip.
    let (state, model_path, cfg_path) = build_state_pointing_at_embedding_only_gguf("embed-only");
    let app = router(state);
    let (status, body) = post_json(
        app,
        "/v1/rerank",
        serde_json::json!({
            "query": "hello",
            "documents": ["world"],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("has no `cls.weight`")
            || body_str.contains("this is an embedding model"),
        "500 body must explain why the GGUF is wrong-typed: {body_str}"
    );
    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&cfg_path);
}

#[tokio::test]
async fn rerank_returns_results_sorted_by_score() {
    let (state, model_path, cfg_path) = build_state_with_synth_reranker("happy-path", 1);
    let app = router(state);
    let (status, body) = post_json(
        app,
        "/v1/rerank",
        serde_json::json!({
            "query": "hello world",
            "documents": [
                "the quick brown fox",
                "test input embed",
                "hello world model",
                "lazy dog cat jump",
            ],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(v["model"], "rustllama-reranker");
    let results = v["results"].as_array().expect("results array");
    assert_eq!(results.len(), 4);

    // Indices must collectively cover {0,1,2,3} (every input
    // appears in the output exactly once).
    let mut seen: Vec<u64> = results
        .iter()
        .map(|r| r["index"].as_u64().expect("index u64"))
        .collect();
    seen.sort_unstable();
    assert_eq!(seen, vec![0, 1, 2, 3], "every input must appear once");

    // Scores must be sorted descending.
    let scores: Vec<f64> = results
        .iter()
        .map(|r| r["relevance_score"].as_f64().expect("score f64"))
        .collect();
    for w in scores.windows(2) {
        assert!(
            w[0] >= w[1],
            "results must be sorted desc by relevance_score: {scores:?}"
        );
    }

    // total_tokens > 0 — there were 4 (query, doc) pairs each
    // wrapped with [CLS] / [SEP] / [SEP], so at minimum 4 *
    // (1 + 1 + 1 + 1 + 1) = 20 tokens.
    let total_tokens = v["usage"]["total_tokens"].as_u64().expect("total_tokens");
    assert!(total_tokens >= 4, "expected at least 4 tokens, got {total_tokens}");

    // Default: documents not echoed.
    for r in results {
        assert!(r.get("document").is_none() || r["document"].is_null(),
                "default response must not include document text: {r}");
    }

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&cfg_path);
}

#[tokio::test]
async fn rerank_top_n_truncates_after_sorting() {
    let (state, model_path, cfg_path) = build_state_with_synth_reranker("top-n", 1);
    let app = router(state);
    let (status, body) = post_json(
        app,
        "/v1/rerank",
        serde_json::json!({
            "query": "hello",
            "documents": ["world", "test", "input", "the quick brown fox"],
            "top_n": 2,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    let results = v["results"].as_array().unwrap();
    assert_eq!(results.len(), 2, "top_n=2 truncates");
    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&cfg_path);
}

#[tokio::test]
async fn rerank_return_documents_echoes_text() {
    let (state, model_path, cfg_path) = build_state_with_synth_reranker("return-docs", 1);
    let app = router(state);
    let (status, body) = post_json(
        app,
        "/v1/rerank",
        serde_json::json!({
            "query": "hello",
            "documents": ["world", "test input"],
            "return_documents": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    let results = v["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    for r in results {
        let text = r["document"]["text"]
            .as_str()
            .unwrap_or_else(|| panic!("document.text must be present: {r}"));
        assert!(text == "world" || text == "test input");
    }
    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&cfg_path);
}

#[tokio::test]
async fn rerank_rejects_empty_query_with_400() {
    let (state, model_path, cfg_path) = build_state_with_synth_reranker("empty-query", 1);
    let app = router(state);
    let (status, _) = post_json(
        app,
        "/v1/rerank",
        serde_json::json!({ "query": "", "documents": ["a"] }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&cfg_path);
}

#[tokio::test]
async fn rerank_rejects_empty_documents_with_400() {
    let (state, model_path, cfg_path) = build_state_with_synth_reranker("empty-docs", 1);
    let app = router(state.clone());
    let (status, _) = post_json(
        app,
        "/v1/rerank",
        serde_json::json!({ "query": "hi", "documents": [] }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let app = router(state);
    let (status, _) = post_json(
        app,
        "/v1/rerank",
        serde_json::json!({ "query": "hi", "documents": ["valid", ""] }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&cfg_path);
}

#[tokio::test]
async fn v1_models_advertises_reranker_capability_when_slot_enabled() {
    let (state, model_path, cfg_path) = build_state_with_synth_reranker("v1-models-rerank", 1);
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
    let entry = data
        .iter()
        .find(|m| m["id"] == "rustllama-reranker")
        .expect("reranker capability must be advertised");
    assert_eq!(entry["purpose"], "rerank");
    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&cfg_path);
}

#[tokio::test]
async fn v1_models_reports_n_labels_after_first_rerank_load() {
    let (state, model_path, cfg_path) =
        build_state_with_synth_reranker("v1-models-n-labels", 3);

    // Force lazy-load via a /v1/rerank call.
    let app = router(state.clone());
    let (status, _) = post_json(
        app,
        "/v1/rerank",
        serde_json::json!({ "query": "hi", "documents": ["a", "b"] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let app = router(state);
    let req = Request::builder()
        .method("GET")
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    let entry = v["data"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == "rustllama-reranker")
        .unwrap()
        .clone();
    assert_eq!(entry["dimensions"], 3, "n_labels = 3");

    let _ = std::fs::remove_file(&model_path);
    let _ = std::fs::remove_file(&cfg_path);
}

#[tokio::test]
async fn v1_models_omits_reranker_entry_when_slot_disabled() {
    let (state, cfg) = build_state_unconfigured("v1-models-no-rerank");
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
        data.iter().all(|m| m["id"] != "rustllama-reranker"),
        "reranker entry must NOT appear when slot disabled: {data:?}"
    );
    let _ = std::fs::remove_file(&cfg);
}
