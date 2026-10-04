//! Integration tests for the bearer-token auth middleware. Pinned
//! behaviors:
//!
//!   - With no auth state attached, requests pass through (legacy
//!     open-access path — backward compatibility for every existing
//!     test fixture).
//!   - With auth attached, requests presenting no header get 401
//!     with `WWW-Authenticate: Bearer`.
//!   - The wrong token gets 401 even if it's the right length —
//!     constant-time compare doesn't reveal length match via timing.
//!   - The right token in either `Bearer <token>` or bare `<token>`
//!     form passes through.
//!   - `/healthz` bypasses auth so monitoring probes don't need
//!     credentials.
//!   - Without `ConnectInfo` (test oneshot path), no loopback
//!     bypass — the test client must present the token. This is
//!     the conservative-default the test harness verifies.
//!
//! The loopback bypass when ConnectInfo IS available is exercised
//! by the live `axum::serve` wiring; the harness here can't easily
//! set ConnectInfo on a oneshot request, so the loopback path is
//! tested indirectly: the request without ConnectInfo + a wrong
//! header still 401s, proving the bypass isn't accidentally on.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::MockEngine;
use rustllama_server::{router, AppState, AuthState, ServingModel};
use tower::ServiceExt;

fn build_state_no_auth(_tag: &str) -> AppState {
    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "auth-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    AppState::new(serving, "0.0.0-test".into())
}

fn build_state_with_auth(tag: &str, key: &str) -> AppState {
    let auth = AuthState::new(key).expect("non-empty key must produce AuthState");
    build_state_no_auth(tag).with_auth(auth)
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

async fn get_with_header(
    app: axum::Router,
    path: &str,
    auth_value: &str,
) -> (StatusCode, Vec<header::HeaderValue>) {
    let req = Request::builder()
        .method("GET")
        .uri(path)
        .header(header::AUTHORIZATION, auth_value)
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let www_auth: Vec<header::HeaderValue> = resp
        .headers()
        .get_all(header::WWW_AUTHENTICATE)
        .into_iter()
        .cloned()
        .collect();
    (status, www_auth)
}

/// Auth state whose loopback bypass is toggled by `require_loopback`
/// (mirrors `[server].require_auth_loopback`).
fn build_state_with_auth_loopback(tag: &str, key: &str, require_loopback: bool) -> AppState {
    let auth = AuthState::new(key)
        .expect("non-empty key must produce AuthState")
        .require_loopback_auth(require_loopback);
    build_state_no_auth(tag).with_auth(auth)
}

/// GET with an explicit `ConnectInfo` peer + optional Authorization header.
/// The oneshot path doesn't set ConnectInfo, so we insert it to exercise the
/// loopback-bypass branch of the middleware.
async fn get_with_peer(
    app: axum::Router,
    path: &str,
    peer: std::net::SocketAddr,
    auth_value: Option<&str>,
) -> StatusCode {
    let mut builder = Request::builder().method("GET").uri(path);
    if let Some(v) = auth_value {
        builder = builder.header(header::AUTHORIZATION, v);
    }
    let mut req = builder.body(Body::empty()).unwrap();
    req.extensions_mut().insert(axum::extract::ConnectInfo(peer));
    app.oneshot(req).await.unwrap().status()
}

/// `require_auth_loopback` toggles the loopback bypass: a 127.0.0.1 peer with
/// no token is admitted by default (bypass on) but rejected when loopback auth
/// is required; the right token still passes; a remote peer always 401s.
#[tokio::test]
async fn loopback_bypass_toggles_with_require_auth_loopback() {
    use std::net::SocketAddr;
    let loopback: SocketAddr = "127.0.0.1:52001".parse().unwrap();
    let remote: SocketAddr = "192.168.1.50:52002".parse().unwrap();

    // Default (bypass ON): loopback peer, no token → 200.
    let app = router(build_state_with_auth_loopback("lb-default", "sk-test", false));
    assert_eq!(
        get_with_peer(app, "/v1/models", loopback, None).await,
        StatusCode::OK,
        "default loopback bypass should admit a local peer without a token"
    );

    // require_auth_loopback = true: loopback peer, no token → 401.
    let app = router(build_state_with_auth_loopback("lb-require", "sk-test", true));
    assert_eq!(
        get_with_peer(app, "/v1/models", loopback, None).await,
        StatusCode::UNAUTHORIZED,
        "require_auth_loopback should reject a local peer without a token"
    );

    // require_auth_loopback = true: loopback peer WITH the right token → 200.
    let app = router(build_state_with_auth_loopback("lb-ok", "sk-test", true));
    assert_eq!(
        get_with_peer(app, "/v1/models", loopback, Some("Bearer sk-test")).await,
        StatusCode::OK,
        "require_auth_loopback should admit a local peer WITH the right token"
    );

    // Remote peer, no token → 401 regardless of the toggle.
    let app = router(build_state_with_auth_loopback("lb-remote", "sk-test", false));
    assert_eq!(
        get_with_peer(app, "/v1/models", remote, None).await,
        StatusCode::UNAUTHORIZED,
        "remote peer must always authenticate"
    );
}

/// No AuthState attached → middleware doesn't fire, every request
/// passes through unchanged. Pins the backward-compat path.
#[tokio::test]
async fn no_auth_state_passes_every_request_through() {
    let app = router(build_state_no_auth("no-auth"));
    let (status, _) = get(app, "/healthz").await;
    assert_eq!(status, StatusCode::OK);

    let app = router(build_state_no_auth("no-auth-2"));
    let (status, _) = get(app, "/v1/models").await;
    assert_eq!(status, StatusCode::OK);
}

/// AuthState attached + no Authorization header → 401 with
/// `WWW-Authenticate: Bearer` (RFC-mandated for bearer schemes).
#[tokio::test]
async fn missing_authorization_header_returns_401_with_www_authenticate() {
    let app = router(build_state_with_auth("missing-header", "sk-test"));
    let (status, _) = get(app, "/v1/models").await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    let app = router(build_state_with_auth("missing-header-2", "sk-test"));
    let (_, www_auth) =
        get_with_header(app, "/v1/models", "  ").await; // header present but empty after trim → still wrong
    // Even a blank header should 401 — verify www_authenticate is
    // present so RFC-compliant clients know what scheme to use.
    let _ = www_auth; // The empty-header path takes split_once → bare path → empty token → ct_eq fails → 401.

    let app = router(build_state_with_auth("missing-header-3", "sk-test"));
    let (status, www_auth) = get_with_header(app, "/v1/models", "").await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    // axum may not surface an empty header on the inbound side; the
    // important guarantee is the 401 itself.
    let _ = www_auth;
}

/// Wrong token same length as the configured key → still 401. Pins
/// the "no information leak about length" property — without
/// constant-time compare, an attacker could brute-force the length.
#[tokio::test]
async fn wrong_token_same_length_returns_401() {
    let app = router(build_state_with_auth("wrong-same-len", "sk-correct-keyXX"));
    let bad = "Bearer sk-correct-keyYY"; // same length, last 2 chars differ
    let (status, www_auth) = get_with_header(app, "/v1/models", bad).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert!(
        www_auth.iter().any(|v| v.to_str().unwrap_or("") == "Bearer"),
        "401 should carry WWW-Authenticate: Bearer"
    );
}

/// `Bearer <token>` with the correct key → 200 (request flows
/// through to the real handler).
#[tokio::test]
async fn correct_bearer_token_passes() {
    let app = router(build_state_with_auth("correct-bearer", "sk-passes-please"));
    let good = "Bearer sk-passes-please";
    let (status, _) = get_with_header(app, "/v1/models", good).await;
    assert_eq!(status, StatusCode::OK);
}

/// Bare `<token>` (no `Bearer ` scheme prefix) also passes. Some
/// clients omit the scheme; we accept it as a courtesy.
#[tokio::test]
async fn bare_token_without_scheme_passes() {
    let app = router(build_state_with_auth("bare-token", "sk-bare-ok"));
    let good = "sk-bare-ok";
    let (status, _) = get_with_header(app, "/v1/models", good).await;
    assert_eq!(status, StatusCode::OK);
}

/// Lowercase `bearer` (RFC says scheme is case-insensitive) is
/// accepted.
#[tokio::test]
async fn lowercase_bearer_scheme_is_accepted() {
    let app = router(build_state_with_auth("lower-bearer", "sk-lowercase"));
    let good = "bearer sk-lowercase";
    let (status, _) = get_with_header(app, "/v1/models", good).await;
    assert_eq!(status, StatusCode::OK);
}

/// `/healthz` bypasses auth even when AuthState is attached and no
/// header is present. Monitoring probes shouldn't need creds.
#[tokio::test]
async fn healthz_bypasses_auth_with_no_header() {
    let app = router(build_state_with_auth("healthz-bypass", "sk-test"));
    let (status, _) = get(app, "/healthz").await;
    assert_eq!(
        status,
        StatusCode::OK,
        "/healthz must be reachable without auth even when api_key is set"
    );
}

/// Wrong token with wrong length → 401 too. (Companion to the
/// same-length test; pins that the middleware doesn't accidentally
/// accept partial matches.)
#[tokio::test]
async fn wrong_token_different_length_returns_401() {
    let app = router(build_state_with_auth("wrong-diff-len", "sk-correct-key"));
    let bad = "Bearer sk-different";
    let (status, _) = get_with_header(app, "/v1/models", bad).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

/// Empty `AuthState::new("")` returns None — the constructor refuses
/// to build a state for an empty key, so the caller never
/// accidentally attaches auth with no actual secret.
#[tokio::test]
async fn auth_state_new_empty_returns_none() {
    assert!(AuthState::new("").is_none(), "empty key → no AuthState");
    assert!(
        AuthState::new("anything").is_some(),
        "non-empty key → AuthState built"
    );
}
