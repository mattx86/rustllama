//! Integration tests for `GET /v1/tuning_summary` — the visibility
//! endpoint backing the GUI Status page's Tuner panel.
//!
//! Tests cover the always-reachable shape contract: the endpoint
//! returns the auto_apply flags from config + a stable JSON envelope
//! whether or not a SYCL device + cache file are present. Mock-mode
//! hosts get `device: null` + empty cache fields; SYCL hosts get
//! the device fingerprint + whatever the cache holds.
//!
//! The cache-HIT shape (placement entries, batch_size winner) is
//! exercised by the CLI-side persister tests; this layer's contract
//! is "endpoint returns the expected envelope without errors and
//! reflects the config's auto_apply flags."

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::MockEngine;
use rustllama_server::{router, AppState, ServingModel};
use tower::ServiceExt;

fn build_state_with_config(tag: &str, config_toml: &str) -> (AppState, std::path::PathBuf) {
    let tmp = std::env::temp_dir().join(format!("rustllama-tuning-summary-{tag}.toml"));
    std::fs::write(&tmp, config_toml).expect("write config");
    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "tuning-summary-mock".into(),
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

/// Endpoint returns 200 with the expected envelope shape. Every
/// field the GUI panel reads from must be present (even when null)
/// so the renderer doesn't have to handle `undefined` paths.
#[tokio::test]
async fn tuning_summary_returns_full_envelope_with_default_config() {
    let (state, cfg) = build_state_with_config("default-cfg", "[server]\nport = 11434\n");
    let app = router(state);
    let (status, body) = get_json(app, "/v1/tuning_summary").await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body={}",
        String::from_utf8_lossy(&body)
    );
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");

    // Required fields — present (possibly null/empty) on every
    // response. The GUI relies on this stability.
    assert!(v.get("device").is_some(), "device field present");
    assert!(v.get("cache_path").is_some(), "cache_path field present");
    assert!(
        v["cache_present"].is_boolean(),
        "cache_present is a boolean: {v}"
    );
    assert!(v.get("last_tuned").is_some(), "last_tuned field present");
    assert!(
        v["kernel_entry_count"].as_u64().is_some(),
        "kernel_entry_count is a number: {v}"
    );
    assert!(
        v["placement"].is_array(),
        "placement is always an array: {v}"
    );
    assert!(
        v["kernels"].is_array(),
        "kernels is always an array (may be empty): {v}"
    );
    assert!(
        v.get("batch_size").is_some(),
        "batch_size field present (may be null)"
    );

    // Auto-apply defaults are `true` per the config schema.
    assert_eq!(v["auto_apply_placement"], true);
    assert_eq!(v["auto_apply_batch_size"], true);

    let _ = std::fs::remove_file(&cfg);
}

/// `[tuning].auto_apply_*` flags round-trip via the endpoint —
/// the GUI uses them to label the panel "auto-applied" vs "stored,
/// not applied". Setting both to false in config must surface.
#[tokio::test]
async fn tuning_summary_reflects_auto_apply_flags() {
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
    let (status, body) = get_json(app, "/v1/tuning_summary").await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(v["auto_apply_placement"], false);
    assert_eq!(v["auto_apply_batch_size"], false);

    let _ = std::fs::remove_file(&cfg);
}

/// Mock-mode (no SYCL) → `device: null`, `cache_path` may or may
/// not be present depending on whether `default_cache_dir()`
/// resolves. The endpoint must NOT error in this case — it's the
/// state every non-SYCL test host hits.
#[tokio::test]
async fn tuning_summary_handles_mock_mode_gracefully() {
    let (state, cfg) = build_state_with_config("mock", "[server]\nport = 11434\n");
    let app = router(state);
    let (status, body) = get_json(app, "/v1/tuning_summary").await;
    assert_eq!(
        status,
        StatusCode::OK,
        "mock-mode hosts must get 200, not 5xx"
    );
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    // On a mock-mode build with no SYCL device, fingerprint_device(0)
    // returns None and the endpoint's match arm produces null +
    // empty fields uniformly.
    if v["device"].is_null() {
        assert_eq!(v["kernel_entry_count"], 0);
        assert_eq!(
            v["placement"].as_array().map(|a| a.len()),
            Some(0),
            "no-device → no placement entries"
        );
        assert_eq!(
            v["kernels"].as_array().map(|a| a.len()),
            Some(0),
            "no-device → no kernel-LWS entries"
        );
        assert_eq!(v["cache_present"], false);
    }
    // (When a real SYCL device is present, the inverse-assertion
    // path is harder to guard against without seeding a cache file
    // at the resolved default cache dir, which would race with any
    // other tests sharing that path.)

    let _ = std::fs::remove_file(&cfg);
}
