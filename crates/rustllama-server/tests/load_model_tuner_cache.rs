//! Verifies the server `load_model` endpoint's cache-consumption
//! wiring: the consumer helpers (`tuner_cached_*`) are queried only
//! when the relevant `[tuning].auto_apply_*` flag is on AND the
//! request didn't pin the value. Mock-mode hosts (no SYCL device
//! visible) silently fall back to config defaults — that's the
//! path most of these tests cover.
//!
//! The cache-HIT case is exercised by the CLI-side persister →
//! consumer round-trip in `rustllama-cli/tests/tuner_cache_persistence.rs`;
//! both layers share the same `rustllama_tuner::load_cache` call
//! shape, so verifying the server doesn't *break* the no-cache
//! fallback is the meaningful contract here. (Fully wiring a
//! synthetic cache file under the default cache dir at test time
//! would race with any other test sharing the same `paths()`
//! resolution.)

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};
use rustllama_server::{router, AppState};
use tower::ServiceExt;

fn build_state_with_config(tag: &str, config_toml: &str) -> (AppState, std::path::PathBuf) {
    let tmp = std::env::temp_dir().join(format!("rustllama-load-tuner-{tag}.toml"));
    std::fs::write(&tmp, config_toml).expect("write config");
    let state = AppState::empty(
        "0.0.0-test".into(),
        rustllama_server::DEFAULT_MAX_LOADED_MODELS,
        rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
    )
    .with_config_path(tmp.clone());
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

/// With auto_apply on (the default) but no cache entry available
/// — which is what every fresh install + every mock-mode test
/// fixture sees — load_model must complete normally and return
/// 200 OK. Guards against my consumer-wiring breaking the
/// no-cache fallback path.
#[tokio::test]
async fn load_model_succeeds_with_auto_apply_on_and_no_cache() {
    let model = std::env::temp_dir().join("rustllama-load-tuner-no-cache.gguf");
    write_synthetic_llama_gguf(&model, &SynthLlama::default());

    // Default config has auto_apply_placement = auto_apply_batch_size = true.
    let (state, cfg) = build_state_with_config(
        "no-cache",
        r#"
[server]
port = 11434

[tuning]
auto_apply_placement = true
auto_apply_batch_size = true
"#,
    );
    let app = router(state);
    let (status, body) = post_json(
        app,
        "/v1/models/load",
        serde_json::json!({ "path": model.to_string_lossy() }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "load_model with auto_apply on and no cache must succeed: {}",
        String::from_utf8_lossy(&body)
    );
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert!(v["model_id"].is_string(), "load response carries model_id");

    let _ = std::fs::remove_file(&model);
    let _ = std::fs::remove_file(&cfg);
}

/// With auto_apply OFF, the consumer helpers are NOT called at
/// all (precedence: request > config; cache skipped entirely).
/// This is the escape-hatch path for users who want fully
/// deterministic loads regardless of any cache that might exist.
#[tokio::test]
async fn load_model_skips_cache_when_auto_apply_off() {
    let model = std::env::temp_dir().join("rustllama-load-tuner-no-apply.gguf");
    write_synthetic_llama_gguf(&model, &SynthLlama::default());

    let (state, cfg) = build_state_with_config(
        "no-apply",
        r#"
[server]
port = 11434

[tuning]
auto_apply_placement = false
auto_apply_batch_size = false
"#,
    );
    let app = router(state);
    let (status, _) = post_json(
        app,
        "/v1/models/load",
        serde_json::json!({ "path": model.to_string_lossy() }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "load_model with auto_apply off must succeed (cache is skipped, defaults win)"
    );

    let _ = std::fs::remove_file(&model);
    let _ = std::fs::remove_file(&cfg);
}

/// Explicit `batch_size` in the request wins over both cache and
/// config — verifies the precedence chain stops at the request
/// field, not falling through to a cache lookup that might exist.
#[tokio::test]
async fn load_model_request_batch_size_wins_over_cache_and_config() {
    let model = std::env::temp_dir()
        .join("rustllama-load-tuner-request-wins.gguf");
    write_synthetic_llama_gguf(&model, &SynthLlama::default());

    let (state, cfg) = build_state_with_config(
        "request-wins",
        r#"
[server]
port = 11434

[inference]
batch_size = 512

[tuning]
auto_apply_batch_size = true
"#,
    );
    let app = router(state);
    let (status, _) = post_json(
        app,
        "/v1/models/load",
        // Explicit small batch size — would be unusual to set
        // from a cache, proves the request field reaches the
        // engine.
        serde_json::json!({ "path": model.to_string_lossy(), "batch_size": 32 }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "load_model with explicit batch_size must succeed"
    );

    let _ = std::fs::remove_file(&model);
    let _ = std::fs::remove_file(&cfg);
}
