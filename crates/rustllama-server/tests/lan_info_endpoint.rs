//! Integration tests for `GET /v1/lan_info` — the endpoint that
//! powers the Settings page's "LAN access" panel. Pinned behaviors:
//!
//!   - Loopback-bound config returns `url: null` + `qr_svg: null`
//!     so the GUI renders the "local-only" hint, not a misleading
//!     QR for 127.0.0.1.
//!   - Non-loopback bind with a discoverable LAN IP returns a URL
//!     of the form `http://<ip>:<port>` and an SVG that starts with
//!     `<?xml` (the qrcode-rs `svg::Color` renderer's preamble).
//!   - `api_key_set` reflects whether `[server].api_key` is
//!     non-empty without leaking the key itself — the GUI uses this
//!     to decide whether to surface the "open server on LAN" warning.
//!
//! Tests run on the host's real network, so the routable-IP path is
//! environment-sensitive. We assert only on what the endpoint
//! *must* return for the configured shape, not on a specific IP.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::MockEngine;
use rustllama_server::{router, AppState, ServingModel};
use tower::ServiceExt;

fn build_state(tag: &str, config_toml: &str) -> (AppState, std::path::PathBuf) {
    let tmp = std::env::temp_dir().join(format!("rustllama-lan-{tag}.toml"));
    std::fs::write(&tmp, config_toml).expect("write config");
    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "lan-mock".into(),
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

/// Loopback-bound config → `url` and `qr_svg` are both null. The
/// GUI uses these as the signal that LAN access is disabled and
/// surfaces the "change bind_addr to 0.0.0.0" hint instead of
/// a stale QR.
#[tokio::test]
async fn lan_info_loopback_bind_returns_null_url_and_qr() {
    let (state, tmp) = build_state(
        "loopback",
        r#"
[server]
bind_addr = "127.0.0.1"
port = 11434
api_key = ""
"#,
    );
    let app = router(state);

    let (status, body) = get_json(app, "/v1/lan_info").await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(v["bind_addr"], "127.0.0.1");
    assert_eq!(v["port"], 11434);
    assert!(
        v["url"].is_null(),
        "loopback bind must return null url: {v}"
    );
    assert!(v["qr_svg"].is_null(), "no url → no qr: {v}");
    assert_eq!(v["api_key_set"], false);
    let _ = std::fs::remove_file(&tmp);
}

/// Non-loopback bind on a host with a routable interface → URL +
/// SVG QR are populated. We can't assert a specific IP (depends on
/// the test runner's network), but we can pin:
///   - if `primary_lan_ip` is non-null, `url` is `http://<ip>:<port>`
///     and `qr_svg` is non-null SVG markup
///   - `api_key_set` reflects the config without leaking the key
#[tokio::test]
async fn lan_info_open_bind_returns_url_and_qr_when_routable() {
    let (state, tmp) = build_state(
        "open-bind",
        r#"
[server]
bind_addr = "0.0.0.0"
port = 12345
api_key = "sk-secret-do-not-leak"
"#,
    );
    let app = router(state);

    let (status, body) = get_json(app, "/v1/lan_info").await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(v["bind_addr"], "0.0.0.0");
    assert_eq!(v["port"], 12345);
    assert_eq!(
        v["api_key_set"], true,
        "api_key_set should reflect non-empty key"
    );
    // The api_key string itself must not appear anywhere in the
    // response — leaking it on a public endpoint would be a real
    // security regression.
    let raw = String::from_utf8_lossy(&body);
    assert!(
        !raw.contains("sk-secret-do-not-leak"),
        "api_key value leaked in response body: {raw}"
    );

    // If discovery succeeded (almost always — even isolated hosts
    // have at least one routable interface), both url + qr should
    // be populated. If discovery failed, both should be null —
    // partial responses are a bug.
    let url_is_null = v["url"].is_null();
    let qr_is_null = v["qr_svg"].is_null();
    assert_eq!(
        url_is_null, qr_is_null,
        "url and qr_svg must agree on null/non-null: {v}"
    );
    if !url_is_null {
        let url = v["url"].as_str().unwrap();
        assert!(
            url.starts_with("http://") && url.ends_with(":12345"),
            "url shape must be http://<ip>:12345 — got {url}"
        );
        let qr = v["qr_svg"].as_str().unwrap();
        // qrcode-rs SVG renderer prefixes output with the standard
        // XML declaration. Pinning this guards against an
        // accidental switch to a different renderer (PNG / unicode)
        // that would break the GUI's dangerouslySetInnerHTML path.
        assert!(
            qr.starts_with("<?xml"),
            "qr_svg should be SVG markup: {}",
            &qr[..qr.len().min(80)]
        );
        assert!(
            qr.contains("<svg"),
            "qr_svg must contain an <svg> element: {}",
            &qr[..qr.len().min(80)]
        );
    }
    let _ = std::fs::remove_file(&tmp);
}

/// Missing `[server].api_key` (or empty) → `api_key_set: false`.
/// The GUI uses this to trigger the "open server on LAN" warning.
#[tokio::test]
async fn lan_info_reports_api_key_not_set_when_blank() {
    let (state, tmp) = build_state(
        "no-key",
        r#"
[server]
bind_addr = "0.0.0.0"
port = 11434
api_key = ""
"#,
    );
    let app = router(state);

    let (status, body) = get_json(app, "/v1/lan_info").await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    assert_eq!(v["api_key_set"], false);
    let _ = std::fs::remove_file(&tmp);
}
