//! Integration test for the system-prompt library: a config file
//! containing `[[system_prompts]]` blocks round-trips intact through
//! `GET /v1/config`. The GUI Chat page consumes this list to populate
//! its dropdown; missing/empty list keeps the dropdown hidden.
//!
//! Pinning the round-trip catches two regression classes:
//!   - schema drift (renaming a field on `SystemPrompt` would break
//!     existing config.toml files in the wild)
//!   - serialization drift (toml ↔ serde_json: the GUI reads JSON
//!     from `/v1/config`, but the file on disk is TOML; both paths
//!     have to agree on the same field names)

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::MockEngine;
use rustllama_server::{router, AppState, ServingModel};
use tower::ServiceExt;

fn build_state(tag: &str, config_toml: &str) -> (AppState, std::path::PathBuf) {
    let tmp = std::env::temp_dir().join(format!("rustllama-sysprompts-{tag}.toml"));
    std::fs::write(&tmp, config_toml).expect("write config");
    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "sysprompt-mock".into(),
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

async fn get_json(app: axum::Router, path: &str) -> (StatusCode, Vec<u8>) {
    let req = Request::builder()
        .method("GET")
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    (status, bytes.to_vec())
}

const CONFIG_WITH_SYSTEM_PROMPTS: &str = r#"
[server]
port = 11434

[[system_prompts]]
name = "Senior Rust reviewer"
body = "You are a senior Rust engineer reviewing the user's code. Flag UB, lifetime errors, and missing error handling."
default_for_model = "qwen2.5-coder-7b-instruct-q4_k_m"

[[system_prompts]]
name = "Concise"
body = "Answer in two sentences or fewer."
"#;

/// `[[system_prompts]]` in config.toml round-trips through
/// `GET /v1/config` with the right shape — name + body + optional
/// `default_for_model` — so the Chat page's dropdown picks up
/// everything the user defined.
#[tokio::test]
async fn system_prompts_round_trip_through_get_config() {
    let (state, tmp) = build_state("round-trip", CONFIG_WITH_SYSTEM_PROMPTS);
    let app = router(state);

    let (status, body) = get_json(app, "/v1/config").await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body={}",
        String::from_utf8_lossy(&body)
    );

    let env: serde_json::Value = serde_json::from_slice(&body).expect("json");
    let prompts = env["config"]["system_prompts"]
        .as_array()
        .expect("system_prompts array present in config envelope");
    assert_eq!(prompts.len(), 2, "two prompts defined in fixture");

    // First entry has all three fields populated.
    assert_eq!(prompts[0]["name"], "Senior Rust reviewer");
    assert!(
        prompts[0]["body"]
            .as_str()
            .unwrap_or_default()
            .contains("senior Rust engineer"),
        "prompt body preserved verbatim: {}",
        prompts[0]
    );
    assert_eq!(
        prompts[0]["default_for_model"], "qwen2.5-coder-7b-instruct-q4_k_m",
        "default_for_model surfaces so the GUI can auto-select"
    );

    // Second entry has the optional `default_for_model` defaulted to "".
    assert_eq!(prompts[1]["name"], "Concise");
    assert_eq!(prompts[1]["default_for_model"], "");

    let _ = std::fs::remove_file(&tmp);
}

/// A config with no `[[system_prompts]]` blocks reports an empty
/// array, NOT a missing field. The Chat page checks `length > 0` to
/// decide whether to render the dropdown — a missing field instead
/// of an empty array would force every consumer to handle two
/// shapes.
#[tokio::test]
async fn system_prompts_defaults_to_empty_array_when_absent() {
    let (state, tmp) = build_state(
        "absent",
        r#"
[server]
port = 11434
"#,
    );
    let app = router(state);

    let (status, body) = get_json(app, "/v1/config").await;
    assert_eq!(status, StatusCode::OK);
    let env: serde_json::Value = serde_json::from_slice(&body).expect("json");
    let prompts = env["config"]["system_prompts"]
        .as_array()
        .expect("system_prompts must always be an array, never null/missing");
    assert!(prompts.is_empty(), "no prompts defined → empty array");
    let _ = std::fs::remove_file(&tmp);
}
