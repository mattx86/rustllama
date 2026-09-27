//! Integration tests for `/v1/crash_logs` — the GUI's surface over
//! the runtime panic hook's per-panic file dumps.
//!
//! Tests seed real fixtures in the resolved `crash_log_dir` (the
//! one `rustllama_runtime::paths()` hands out — usually
//! `%LOCALAPPDATA%\rustllama\logs` on Windows). Fixture filenames
//! use synthetic-but-distinctive pid values (`9999999X`) so they
//! sort and clean up without colliding with any real panics that
//! might already live in the dir from prior runs.

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
        model_id: "crash-logs-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    AppState::new(serving, "0.0.0-test".into())
}

/// Pid suffix used by all test fixtures so we can find + remove
/// them without disturbing real panic dumps. Picked well above any
/// realistic PID on Windows (which caps at 2^32-1 but in practice
/// stays under ~32K for typical sessions).
const TEST_PID: u32 = 99_999_900;

fn fixture_path(timestamp: u64) -> std::path::PathBuf {
    let dir = rustllama_runtime::paths().crash_log_dir.clone();
    let _ = std::fs::create_dir_all(&dir);
    dir.join(format!("crash-{timestamp}-{TEST_PID}.log"))
}

fn write_fixture(timestamp: u64, body: &str) -> String {
    let path = fixture_path(timestamp);
    std::fs::write(&path, body).expect("write crash fixture");
    path.file_name().unwrap().to_string_lossy().to_string()
}

fn cleanup_fixtures(timestamps: &[u64]) {
    for ts in timestamps {
        let _ = std::fs::remove_file(fixture_path(*ts));
    }
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

async fn get_with_content_type(app: axum::Router, path: &str) -> (StatusCode, String, String) {
    let req = Request::builder()
        .method("GET")
        .uri(path)
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let ctype = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .map(|v| v.to_str().unwrap_or("").to_string())
        .unwrap_or_default();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    (status, ctype, String::from_utf8_lossy(&bytes).into_owned())
}

async fn delete(app: axum::Router, path: &str) -> StatusCode {
    let req = Request::builder()
        .method("DELETE")
        .uri(path)
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    resp.status()
}

/// `GET /v1/crash_logs` returns the seeded fixtures with the right
/// shape — name, path, size_bytes, epoch_secs. Newest first.
#[tokio::test]
async fn list_returns_seeded_fixtures_newest_first() {
    // Stamp the fixtures ahead of wall-clock time so they always
    // survive the endpoint's newest-50 truncation — a dev machine's
    // crash dir can hold dozens of real panic dumps, all newer than
    // any fixed historical timestamp (this test used to seed Nov-2023
    // stamps and started failing once 50+ real crashes accumulated).
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let ts_old = now + 100_000;
    let ts_new = now + 100_100;
    let name_old = write_fixture(ts_old, "old panic body");
    let name_new = write_fixture(ts_new, "newer panic body");

    let app = router(build_state());
    let (status, body) = get(app, "/v1/crash_logs").await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    let entries = v["entries"].as_array().expect("entries array");

    // Locate our fixtures (other panics from real runs may share the dir).
    let found_old = entries.iter().find(|e| e["name"] == name_old);
    let found_new = entries.iter().find(|e| e["name"] == name_new);
    assert!(found_old.is_some(), "old fixture missing: {entries:?}");
    assert!(found_new.is_some(), "new fixture missing: {entries:?}");

    // Newest-first ordering: in the absolute output, the new fixture's
    // index must be lower than the old fixture's.
    let idx_new = entries.iter().position(|e| e["name"] == name_new).unwrap();
    let idx_old = entries.iter().position(|e| e["name"] == name_old).unwrap();
    assert!(idx_new < idx_old, "newest must come first");

    let new_entry = found_new.unwrap();
    assert_eq!(new_entry["epoch_secs"], ts_new);
    assert_eq!(new_entry["size_bytes"], "newer panic body".len() as u64);
    assert!(new_entry["path"].as_str().unwrap().ends_with(&name_new));

    // dir field surfaces the absolute crash log directory.
    assert!(v["dir"].is_string());

    cleanup_fixtures(&[ts_old, ts_new]);
}

/// `GET /v1/crash_logs/:name` returns the file body verbatim with
/// `text/plain; charset=utf-8`. The Settings panel renders this in a
/// `<pre>` block.
#[tokio::test]
async fn get_returns_body_with_text_plain_content_type() {
    let ts = 1_700_000_200u64;
    let payload = "rustllama crash log\npanic: synthetic test\nlocation: nowhere.rs:1";
    let name = write_fixture(ts, payload);

    let app = router(build_state());
    let (status, ctype, body) = get_with_content_type(app, &format!("/v1/crash_logs/{name}")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        ctype.starts_with("text/plain"),
        "content-type must be text/plain so the GUI doesn't auto-download: {ctype}"
    );
    assert_eq!(body, payload, "body must round-trip verbatim");

    cleanup_fixtures(&[ts]);
}

/// Path traversal via `..` is rejected at the name-validation step
/// (BAD_REQUEST), never reaching `std::fs::read_to_string`. Pins the
/// security barrier.
#[tokio::test]
async fn get_rejects_path_traversal_attempts() {
    // Names that don't match the `crash-<digits>-<digits>.log` shape
    // are rejected before any filesystem access. Axum's router does
    // its own path decoding so `..` reaching the handler is the
    // path the validation must defend against. Each iteration builds
    // a fresh `app` because `oneshot` consumes the router.
    for bad in &[
        "../some-other-file.log",
        "crash-..-1234.log",
        "crash-1234-1234.txt",
        "not-a-crash.log",
        "crash--1234.log",
        "crash-1234-.log",
    ] {
        let app = router(build_state());
        let url = format!("/v1/crash_logs/{}", urlencoding(bad));
        let (status, _) = get(app, &url).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "{bad} must be rejected as a bad name, got {status}"
        );
    }

    // For completeness, a valid-shaped but nonexistent name returns
    // 404 — not 200, not 500.
    let fresh_app = router(build_state());
    let (status, _) = get(fresh_app, "/v1/crash_logs/crash-1-1.log").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

/// Minimal percent-encoder for path-segment characters used by the
/// traversal test. axum decodes `%2e%2e` back to `..` so we can
/// verify the validator catches it regardless of how the client
/// encoded the bad input.
fn urlencoding(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        let safe = b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~');
        if safe {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// `DELETE /v1/crash_logs/:name` removes the file from disk and a
/// subsequent GET returns 404.
#[tokio::test]
async fn delete_removes_file_from_disk() {
    let ts = 1_700_000_300u64;
    let name = write_fixture(ts, "panic for delete test");
    let abs = fixture_path(ts);
    assert!(abs.exists(), "fixture seeded");

    let app = router(build_state());
    let status = delete(app, &format!("/v1/crash_logs/{name}")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(!abs.exists(), "file removed from disk after DELETE");

    // Second delete returns 404 — file is already gone.
    let app = router(build_state());
    let status = delete(app, &format!("/v1/crash_logs/{name}")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}
