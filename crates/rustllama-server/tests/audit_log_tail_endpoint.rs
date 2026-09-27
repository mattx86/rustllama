//! Integration tests for `GET /v1/audit_log/tail`. The audit-log
//! writer is exercised by `audit_log_middleware.rs`; this layer's
//! contract is the read-side endpoint:
//!
//!   - returns `{path: null, entries: []}` when the file doesn't
//!     exist yet (fresh install / audit_log was never enabled)
//!   - returns the last N entries in newest-first order when the
//!     file does exist
//!   - `total_entries` matches the full file's entry count even
//!     when `n` is smaller
//!   - `n` is hard-capped at 1000 to prevent the endpoint from
//!     pinning a huge audit log into memory

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::MockEngine;
use rustllama_server::{router, AppState, ServingModel};
use tower::ServiceExt;

fn build_state_with_config(tag: &str, config_toml: &str) -> (AppState, std::path::PathBuf) {
    let tmp = std::env::temp_dir().join(format!("rustllama-audit-tail-{tag}.toml"));
    std::fs::write(&tmp, config_toml).expect("write config");
    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "audit-tail-mock".into(),
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

/// Missing audit-log file → returns 200 with `path: null` and
/// `entries: []`. Matches the fresh-install path; GUI renders the
/// "no audit log on disk yet" hint.
#[tokio::test]
async fn tail_returns_empty_envelope_when_no_file_exists() {
    let audit_log = std::env::temp_dir()
        .join("rustllama-audit-tail-no-file-must-not-exist.jsonl");
    let _ = std::fs::remove_file(&audit_log);
    let cfg_toml = format!(
        r#"
[server]
port = 11434
audit_log = false
audit_log_path = "{}"
"#,
        audit_log.display().to_string().replace('\\', "\\\\"),
    );
    let (state, cfg) = build_state_with_config("no-file", &cfg_toml);
    let app = router(state);
    let (status, body) = get_json(app, "/v1/audit_log/tail").await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert!(v["path"].is_null(), "absent file → path is null");
    assert_eq!(v["total_entries"], 0);
    assert_eq!(v["entries"].as_array().map(|a| a.len()), Some(0));
    let _ = std::fs::remove_file(&cfg);
}

/// Populated file → entries come back newest-first + `total_entries`
/// reflects the full count regardless of `n`.
#[tokio::test]
async fn tail_returns_newest_first_with_total_count() {
    let audit_log = std::env::temp_dir().join("rustllama-audit-tail-populated.jsonl");
    // Seed five entries with distinct paths so we can identify
    // their order in the response.
    let body = "\
{\"ts_ms\":1000,\"method\":\"GET\",\"path\":\"/a\",\"query\":\"\",\"status\":200,\"latency_ms\":1}
{\"ts_ms\":2000,\"method\":\"GET\",\"path\":\"/b\",\"query\":\"\",\"status\":200,\"latency_ms\":2}
{\"ts_ms\":3000,\"method\":\"GET\",\"path\":\"/c\",\"query\":\"\",\"status\":404,\"latency_ms\":3}
{\"ts_ms\":4000,\"method\":\"POST\",\"path\":\"/d\",\"query\":\"\",\"status\":500,\"latency_ms\":4}
{\"ts_ms\":5000,\"method\":\"GET\",\"path\":\"/e\",\"query\":\"\",\"status\":200,\"latency_ms\":5}
";
    std::fs::write(&audit_log, body).expect("seed audit log");

    let cfg_toml = format!(
        r#"
[server]
audit_log_path = "{}"
"#,
        audit_log.display().to_string().replace('\\', "\\\\"),
    );
    let (state, cfg) = build_state_with_config("populated", &cfg_toml);
    let app = router(state);
    let (status, resp_body) = get_json(app, "/v1/audit_log/tail?n=3").await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&resp_body).expect("json");
    assert_eq!(v["total_entries"], 5, "total_entries reflects full file");
    let entries = v["entries"].as_array().expect("entries array");
    assert_eq!(entries.len(), 3, "n=3 caps to 3 most-recent");
    // Newest first: e, d, c (descending ts_ms).
    assert_eq!(entries[0]["path"], "/e");
    assert_eq!(entries[1]["path"], "/d");
    assert_eq!(entries[2]["path"], "/c");
    assert!(
        v["path"].as_str().unwrap().contains("rustllama-audit-tail-populated"),
        "path field carries the resolved file path"
    );

    let _ = std::fs::remove_file(&audit_log);
    let _ = std::fs::remove_file(&cfg);
}

/// Default `n` (no query param) returns up to 50. Pinning the
/// default so the GUI doesn't have to send the param explicitly.
#[tokio::test]
async fn tail_defaults_to_50_entries_when_n_unset() {
    let audit_log = std::env::temp_dir().join("rustllama-audit-tail-default-n.jsonl");
    // Seed 70 entries — more than the default 50.
    let mut body = String::new();
    for i in 0..70u64 {
        body.push_str(&format!(
            "{{\"ts_ms\":{i},\"method\":\"GET\",\"path\":\"/i{i}\",\"query\":\"\",\"status\":200,\"latency_ms\":1}}\n"
        ));
    }
    std::fs::write(&audit_log, &body).expect("seed");

    let cfg_toml = format!(
        r#"
[server]
audit_log_path = "{}"
"#,
        audit_log.display().to_string().replace('\\', "\\\\"),
    );
    let (state, cfg) = build_state_with_config("default-n", &cfg_toml);
    let app = router(state);
    let (status, resp_body) = get_json(app, "/v1/audit_log/tail").await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&resp_body).expect("json");
    assert_eq!(v["total_entries"], 70);
    assert_eq!(
        v["entries"].as_array().map(|a| a.len()),
        Some(50),
        "default n=50 caps to 50 even though 70 exist"
    );

    let _ = std::fs::remove_file(&audit_log);
    let _ = std::fs::remove_file(&cfg);
}

/// Malformed JSONL lines are silently skipped (not the typical
/// state but the file format isn't strictly enforced). The
/// endpoint returns the valid entries it found.
#[tokio::test]
async fn tail_skips_malformed_lines_without_erroring() {
    let audit_log = std::env::temp_dir().join("rustllama-audit-tail-malformed.jsonl");
    let body = "\
{\"ts_ms\":1,\"method\":\"GET\",\"path\":\"/ok1\",\"query\":\"\",\"status\":200,\"latency_ms\":1}
this is not json
{\"ts_ms\":2,\"method\":\"GET\",\"path\":\"/ok2\",\"query\":\"\",\"status\":200,\"latency_ms\":1}

{ incomplete json
{\"ts_ms\":3,\"method\":\"GET\",\"path\":\"/ok3\",\"query\":\"\",\"status\":200,\"latency_ms\":1}
";
    std::fs::write(&audit_log, body).expect("seed");

    let cfg_toml = format!(
        r#"
[server]
audit_log_path = "{}"
"#,
        audit_log.display().to_string().replace('\\', "\\\\"),
    );
    let (state, cfg) = build_state_with_config("malformed", &cfg_toml);
    let app = router(state);
    let (status, resp_body) = get_json(app, "/v1/audit_log/tail").await;
    assert_eq!(status, StatusCode::OK, "malformed lines must not 5xx");
    let v: serde_json::Value = serde_json::from_slice(&resp_body).expect("json");
    // Three valid lines, two malformed → total_entries = 3.
    assert_eq!(v["total_entries"], 3, "malformed lines skipped, valid counted");
    let entries = v["entries"].as_array().unwrap();
    let paths: Vec<&str> = entries.iter().map(|e| e["path"].as_str().unwrap()).collect();
    assert_eq!(paths, vec!["/ok3", "/ok2", "/ok1"]);

    let _ = std::fs::remove_file(&audit_log);
    let _ = std::fs::remove_file(&cfg);
}
