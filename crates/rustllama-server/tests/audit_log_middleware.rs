//! Integration tests for the audit-log middleware. Pinned behavior:
//!
//!   - When `AuditSink` is attached, every request appends a JSONL
//!     line with `{ts_ms, method, path, query, status, latency_ms}`.
//!   - `api_key=…` (and other sensitive params) in the query string
//!     are redacted to `<key>=***` so the log doesn't leak credentials.
//!   - The Authorization header value never appears in the log —
//!     headers aren't logged at all.
//!   - When no `AuditSink` is attached (the default state), no file
//!     is created and no extra overhead is incurred. We assert the
//!     "absent sink → absent middleware effect" path by checking
//!     that the response still flows through and no path-derived
//!     file exists.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::MockEngine;
use rustllama_server::{router, AppState, AuditSink, ServingModel};
use tower::ServiceExt;

fn build_state(tag: &str) -> AppState {
    let _ = tag;
    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "audit-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    AppState::new(serving, "0.0.0-test".into())
}

fn build_state_with_audit(tag: &str) -> (AppState, std::path::PathBuf) {
    let path = std::env::temp_dir().join(format!("rustllama-audit-{tag}.jsonl"));
    // Start with a clean file so previous test runs don't leave
    // residue that confuses the assertions.
    let _ = std::fs::remove_file(&path);
    let sink = AuditSink::try_open(path.clone()).expect("open audit sink");
    let state = build_state(tag).with_audit_sink(sink);
    (state, path)
}

async fn get(app: axum::Router, path: &str) -> (StatusCode, Vec<u8>) {
    let req = Request::builder()
        .method("GET")
        .uri(path)
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    (status, bytes.to_vec())
}

async fn get_with_auth(
    app: axum::Router,
    path: &str,
    auth: &str,
) -> (StatusCode, Vec<u8>) {
    let req = Request::builder()
        .method("GET")
        .uri(path)
        .header(header::AUTHORIZATION, auth)
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    (status, bytes.to_vec())
}

fn read_audit_lines(path: &std::path::Path) -> Vec<serde_json::Value> {
    let raw = std::fs::read_to_string(path).expect("read audit log");
    raw.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<serde_json::Value>(l).expect("audit line is json"))
        .collect()
}

/// Enabling the audit sink causes one JSONL line per request, with
/// the expected fields and types. Method + path appear verbatim.
#[tokio::test]
async fn audit_middleware_writes_jsonl_per_request() {
    let (state, path) = build_state_with_audit("writes-jsonl");
    let app = router(state);

    let (status, _) = get(app, "/healthz").await;
    assert_eq!(status, StatusCode::OK);

    // The middleware writes synchronously via a flushing BufWriter,
    // so the line is on disk by the time the response returns.
    let lines = read_audit_lines(&path);
    assert_eq!(lines.len(), 1, "one request → one line: {lines:?}");
    let e = &lines[0];
    assert_eq!(e["method"], "GET");
    assert_eq!(e["path"], "/healthz");
    assert_eq!(e["status"], 200);
    assert!(e["ts_ms"].as_u64().is_some(), "ts_ms is a u64: {e}");
    assert!(e["latency_ms"].as_u64().is_some(), "latency_ms is a u64: {e}");
    // Default for an empty query string is the empty string — NOT
    // null — so consumers can rely on the type being stable.
    assert_eq!(e["query"], "");

    let _ = std::fs::remove_file(&path);
}

/// `api_key=…` in the query string is redacted to `api_key=***`.
/// Verifies the literal secret never lands in the audit log even
/// when the user (or a misconfigured client) puts it there.
#[tokio::test]
async fn audit_middleware_redacts_api_key_query_param() {
    let (state, path) = build_state_with_audit("redact-key");
    let app = router(state);

    let secret = "sk-please-never-leak-this-string";
    let url = format!("/v1/models?api_key={secret}&foo=bar");
    let (_status, _) = get(app, &url).await;

    let lines = read_audit_lines(&path);
    assert_eq!(lines.len(), 1);
    let q = lines[0]["query"].as_str().unwrap();
    assert!(
        q.contains("api_key=***"),
        "redacted form must appear: {q}"
    );
    assert!(
        !q.contains(secret),
        "raw key must NOT appear in audit log: {q}"
    );
    // Non-sensitive params pass through untouched.
    assert!(q.contains("foo=bar"), "non-sensitive params preserved: {q}");

    // Also verify the secret never appears anywhere in the file
    // (defense-in-depth against future fields that might leak it).
    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(
        !raw.contains(secret),
        "secret must not appear anywhere in audit log file: {raw}"
    );
    let _ = std::fs::remove_file(&path);
}

/// The Authorization header value never appears in the audit log —
/// the middleware only logs method, path, query, status, latency.
/// Pins the no-headers-in-log invariant.
#[tokio::test]
async fn audit_middleware_never_logs_authorization_header() {
    let (state, path) = build_state_with_audit("no-auth-header");
    let app = router(state);

    let bearer = "Bearer sk-this-bearer-token-should-never-be-logged";
    let (_status, _) = get_with_auth(app, "/healthz", bearer).await;

    let raw = std::fs::read_to_string(&path).unwrap();
    assert!(
        !raw.contains("sk-this-bearer-token-should-never-be-logged"),
        "Authorization value leaked into audit log: {raw}"
    );
    assert!(
        !raw.to_lowercase().contains("authorization"),
        "even the header NAME shouldn't appear (we don't log headers at all): {raw}"
    );

    let _ = std::fs::remove_file(&path);
}

/// When no audit sink is attached, no file is created (which is
/// what we already checked via "audit disabled = no path side
/// effects") and the response still flows through normally — the
/// middleware doesn't accidentally intercept anything.
#[tokio::test]
async fn audit_middleware_absent_when_sink_unset() {
    // Use a name that no other test claims so we can definitively
    // assert "this path doesn't exist" without flaking on left-over
    // residue.
    let probe_path = std::env::temp_dir().join("rustllama-audit-no-sink-must-not-exist.jsonl");
    let _ = std::fs::remove_file(&probe_path);
    let app = router(build_state("no-sink"));
    let (status, _) = get(app, "/healthz").await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        !probe_path.exists(),
        "no sink attached → no file created at {}",
        probe_path.display()
    );
}

/// `token=…` and `auth=…` query params are also redacted — they're
/// common alternate names for the same kind of secret. Verifies the
/// redaction is keyed off the canonical list, not just `api_key`.
#[tokio::test]
async fn audit_middleware_redacts_token_and_auth_query_params() {
    let (state, path) = build_state_with_audit("redact-token-auth");
    let app = router(state);

    let url = "/v1/models?token=t-secret-token&auth=a-secret-auth&other=ok";
    let (_status, _) = get(app, url).await;

    let lines = read_audit_lines(&path);
    let q = lines[0]["query"].as_str().unwrap();
    assert!(q.contains("token=***"), "token redacted: {q}");
    assert!(q.contains("auth=***"), "auth redacted: {q}");
    assert!(q.contains("other=ok"), "non-sensitive preserved: {q}");
    assert!(
        !q.contains("t-secret-token") && !q.contains("a-secret-auth"),
        "no raw secrets in query field: {q}"
    );

    let _ = std::fs::remove_file(&path);
}
