//! Integration tests for `POST /v1/config/profile/apply`.
//!
//! The endpoint reads on-disk `config.toml`, applies the named
//! profile's sparse overrides, writes the merged result back, and
//! returns the same `{ changes, requires_*_reload }` shape as PUT
//! `/v1/config`. These tests pin:
//!   - applying a server-only profile flips `changes.server` +
//!     `requires_server_restart`
//!   - applying an inference-only profile flips `changes.inference`
//!     + `requires_model_reload`
//!   - the merged config is actually written to disk
//!   - an unknown profile name returns 404 with the available list
//!
//! Uses MockEngine so the test doesn't pay model-load latency; the
//! config endpoints are engine-agnostic.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::MockEngine;
use rustllama_server::{router, AppState, ServingModel};
use tower::ServiceExt;

fn build_state(tag: &str, config_toml: &str) -> (AppState, std::path::PathBuf) {
    let tmp = std::env::temp_dir().join(format!("rustllama-cfg-profile-{tag}.toml"));
    std::fs::write(&tmp, config_toml).expect("write config");
    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "cfg-mock".into(),
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

const CONFIG_WITH_PROFILES: &str = r#"
[server]
bind_addr = "127.0.0.1"
port = 11434

[[profiles]]
name = "lan"
[profiles.server]
bind_addr = "0.0.0.0"
port = 11500

[[profiles]]
name = "small"
[profiles.inference]
ctx_size = 4096
batch_size = 256
"#;

/// Applying a server-only profile flips `changes.server` +
/// `requires_server_restart` and writes the new bind/port to disk.
#[tokio::test]
async fn apply_server_profile_flips_server_changes_and_persists_to_disk() {
    let (state, tmp) = build_state("server-profile", CONFIG_WITH_PROFILES);
    let app = router(state);

    let (status, body) = post_json(
        app,
        "/v1/config/profile/apply",
        serde_json::json!({ "name": "lan" }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body={}",
        String::from_utf8_lossy(&body)
    );
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(v["changes"]["server"], true, "server section changed: {v}");
    assert_eq!(v["requires_server_restart"], true);
    assert_eq!(v["changes"]["inference"], false);
    assert_eq!(v["requires_model_reload"], false);

    // File on disk reflects the merged config — load + check.
    let merged = rustllama_config::load(&tmp).expect("reload merged config");
    assert_eq!(merged.server.bind_addr, "0.0.0.0");
    assert_eq!(merged.server.port, 11500);

    let _ = std::fs::remove_file(&tmp);
}

/// Applying an inference-only profile flips `changes.inference` +
/// `requires_model_reload` (server changes false).
#[tokio::test]
async fn apply_inference_profile_flips_inference_changes() {
    let (state, tmp) = build_state("inf-profile", CONFIG_WITH_PROFILES);
    let app = router(state);

    let (status, body) = post_json(
        app,
        "/v1/config/profile/apply",
        serde_json::json!({ "name": "small" }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(v["changes"]["inference"], true);
    assert_eq!(v["requires_model_reload"], true);
    assert_eq!(v["changes"]["server"], false);
    assert_eq!(v["requires_server_restart"], false);

    let merged = rustllama_config::load(&tmp).expect("reload merged config");
    assert_eq!(merged.inference.ctx_size, 4096);
    assert_eq!(merged.inference.batch_size, 256);

    let _ = std::fs::remove_file(&tmp);
}

/// Unknown profile name → 404 with a body that names the available
/// profiles, so the GUI dropdown's stale selection produces an error
/// the user can act on.
#[tokio::test]
async fn apply_unknown_profile_returns_404_with_available_list() {
    let (state, tmp) = build_state("unknown-profile", CONFIG_WITH_PROFILES);
    let app = router(state);

    let (status, body) = post_json(
        app,
        "/v1/config/profile/apply",
        serde_json::json!({ "name": "nope-no-such-profile" }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("nope-no-such-profile"),
        "404 echoes the requested name: {body_str}"
    );
    assert!(
        body_str.contains("lan") && body_str.contains("small"),
        "404 lists the available profiles: {body_str}"
    );
    let _ = std::fs::remove_file(&tmp);
}

/// Config without any `[[profiles]]` blocks → unknown-profile request
/// returns 404 with a "<none defined>" hint so the UI can render an
/// "add profiles to config.toml" message instead of just an error.
#[tokio::test]
async fn apply_profile_with_no_profiles_defined_returns_404_with_none_hint() {
    let (state, tmp) = build_state(
        "no-profiles",
        r#"
[server]
bind_addr = "127.0.0.1"
port = 11434
"#,
    );
    let app = router(state);

    let (status, body) = post_json(
        app,
        "/v1/config/profile/apply",
        serde_json::json!({ "name": "anything" }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.contains("none defined"),
        "404 should note that no profiles are defined: {body_str}"
    );
    let _ = std::fs::remove_file(&tmp);
}
