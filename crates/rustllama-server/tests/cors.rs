//! CORS layer integration: `router_with_cors` should attach the
//! `Access-Control-Allow-*` headers when the supplied origin list is
//! non-empty, and leave them absent when the list is empty (the
//! same-origin / curl default).

use std::sync::Arc;

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use rustllama_engine::MockEngine;
use rustllama_server::{router, router_with_cors, AppState, ServingModel};
use tower::ServiceExt;

fn build_state() -> AppState {
    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "cors-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    AppState::new(serving, "0.0.0-test".into())
}

#[tokio::test]
async fn empty_origin_list_does_not_attach_cors_headers() {
    // Default `router(state)` is equivalent to no CORS — `OPTIONS`
    // preflight requests don't get an Allow-Origin header. Editors
    // running on a separate host would have their fetch() rejected.
    let app = router(build_state());
    let req = Request::builder()
        .method(Method::OPTIONS)
        .uri("/healthz")
        .header(header::ORIGIN, "http://example.test")
        .header(header::ACCESS_CONTROL_REQUEST_METHOD, "GET")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    // No CORS layer → no Allow-Origin header on the response.
    assert!(
        resp.headers().get(header::ACCESS_CONTROL_ALLOW_ORIGIN).is_none(),
        "no CORS configured → no Access-Control-Allow-Origin header"
    );
}

#[tokio::test]
async fn wildcard_origin_attaches_allow_origin_star() {
    // `cors_origins = ["*"]` → tower-http's CorsLayer with `Any`
    // surfaces `Access-Control-Allow-Origin: *` on preflight responses.
    let app = router_with_cors(build_state(), &["*".to_string()]);
    let req = Request::builder()
        .method(Method::OPTIONS)
        .uri("/healthz")
        .header(header::ORIGIN, "http://example.test")
        .header(header::ACCESS_CONTROL_REQUEST_METHOD, "GET")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let allow = resp
        .headers()
        .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");
    assert_eq!(allow, "*");
}

#[tokio::test]
async fn exact_origin_match_attaches_allow_origin_echo() {
    // Allow-list mode: only requests whose Origin matches an entry
    // get the allow header echoed back. Requests from other origins
    // are silently dropped at the browser layer (server still
    // responds, but without the allow header).
    let allowed = "http://allowed.example".to_string();
    let app = router_with_cors(build_state(), &[allowed.clone()]);

    // Matching origin: header echoed.
    let req_ok = Request::builder()
        .method(Method::OPTIONS)
        .uri("/healthz")
        .header(header::ORIGIN, &allowed)
        .header(header::ACCESS_CONTROL_REQUEST_METHOD, "GET")
        .body(Body::empty())
        .unwrap();
    let resp_ok = app.clone().oneshot(req_ok).await.unwrap();
    let allow = resp_ok
        .headers()
        .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");
    assert_eq!(allow, allowed.as_str());

    // Non-matching origin: no allow header.
    let req_bad = Request::builder()
        .method(Method::OPTIONS)
        .uri("/healthz")
        .header(header::ORIGIN, "http://other.example")
        .header(header::ACCESS_CONTROL_REQUEST_METHOD, "GET")
        .body(Body::empty())
        .unwrap();
    let resp_bad = app.oneshot(req_bad).await.unwrap();
    assert!(
        resp_bad
            .headers()
            .get(header::ACCESS_CONTROL_ALLOW_ORIGIN)
            .is_none(),
        "non-matching origin must not receive an allow header"
    );
}
