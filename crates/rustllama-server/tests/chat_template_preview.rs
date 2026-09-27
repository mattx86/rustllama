//! Integration tests for `POST /v1/chat/template/preview` — the
//! endpoint that powers the Settings page's chat_template live
//! preview. Pure render path, no tokenize / engine load.
//!
//! Pinned behaviors:
//!   - A real ChatML template renders the sample messages with the
//!     expected `<|im_start|>role` / `<|im_end|>` markers.
//!   - `add_generation_prompt: true` appends the assistant primer;
//!     `false` omits it.
//!   - A syntactically-broken template returns 400 with the parse
//!     error in the body so the Settings page can surface it inline.
//!   - `used_engine_specials` is false when no real engine is loaded
//!     (MockEngine fixture has no tokenizer surface), matching what
//!     the GUI hint claims.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::MockEngine;
use rustllama_server::{router, AppState, ServingModel};
use tower::ServiceExt;

fn build_state(tag: &str) -> AppState {
    let _ = tag;
    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "preview-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    AppState::new(serving, "0.0.0-test".into())
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

// A minimal ChatML-shaped template — the format Qwen2.5-Coder /
// many small coding models use. Wrapping each message in
// `<|im_start|>role` ... `<|im_end|>` is what the GUI preview shows.
const CHATML_TEMPLATE: &str = "{%- for m in messages -%}\
<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n\
{%- endfor -%}\
{%- if add_generation_prompt -%}<|im_start|>assistant\n{%- endif -%}";

/// A real ChatML template renders the supplied messages with the
/// expected `<|im_start|>`/`<|im_end|>` chrome, and appends the
/// assistant primer when `add_generation_prompt: true`. Without an
/// engine loaded, `used_engine_specials` is false — matching what
/// the GUI hint promises the user.
#[tokio::test]
async fn preview_renders_chatml_with_generation_prompt() {
    let app = router(build_state("chatml-with-gen"));
    let (status, body) = post_json(
        app,
        "/v1/chat/template/preview",
        serde_json::json!({
            "template": CHATML_TEMPLATE,
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            "add_generation_prompt": true,
        }),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::OK,
        "body={}",
        String::from_utf8_lossy(&body)
    );
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    let rendered = v["rendered"].as_str().expect("rendered string");
    // Each of the three roles appears with the ChatML im_start chrome.
    assert!(rendered.contains("<|im_start|>system\nbe terse<|im_end|>"));
    assert!(rendered.contains("<|im_start|>user\nhi<|im_end|>"));
    assert!(rendered.contains("<|im_start|>assistant\nhello<|im_end|>"));
    // The generation primer is appended at the tail. minijinja's
    // `{%- ... -%}` whitespace control strips the trailing newline
    // inside the block, so we match on the primer prefix without
    // mandating a trailing `\n`.
    assert!(
        rendered.ends_with("<|im_start|>assistant"),
        "trailing assistant primer must be present: {rendered:?}"
    );
    // MockEngine has no tokenizer, so the special-token forwarding
    // path returns false.
    assert_eq!(v["used_engine_specials"], false);
}

/// `add_generation_prompt: false` omits the assistant primer. Used
/// by FIM/completion editors that don't want the chat shape.
#[tokio::test]
async fn preview_without_generation_prompt_omits_assistant_primer() {
    let app = router(build_state("chatml-no-gen"));
    let (status, body) = post_json(
        app,
        "/v1/chat/template/preview",
        serde_json::json!({
            "template": CHATML_TEMPLATE,
            "messages": [
                {"role": "user", "content": "hi"},
            ],
            "add_generation_prompt": false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    let rendered = v["rendered"].as_str().expect("rendered string");
    assert!(rendered.contains("<|im_start|>user\nhi<|im_end|>"));
    assert!(
        !rendered.contains("<|im_start|>assistant"),
        "no generation prompt → no assistant primer: {rendered:?}"
    );
}

/// A syntactically-invalid template returns 400 with the parse
/// error so the Settings page can render it inline rather than
/// silently swallowing the typo.
#[tokio::test]
async fn preview_invalid_template_returns_400_with_error_body() {
    let app = router(build_state("bad-tpl"));
    let (status, body) = post_json(
        app,
        "/v1/chat/template/preview",
        serde_json::json!({
            // Unterminated `{% for %}` — Jinja parse error.
            "template": "{% for m in messages %}{{ m.role }}",
            "messages": [],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let body_str = String::from_utf8_lossy(&body);
    assert!(
        body_str.to_lowercase().contains("template")
            || body_str.to_lowercase().contains("render")
            || body_str.to_lowercase().contains("syntax"),
        "400 body should describe the template failure: {body_str}"
    );
}

/// An empty messages array is legal — the template runs against
/// whatever pre/post chrome it produces standalone (some templates
/// emit a leading system block, etc). The endpoint must not 400
/// on this — the GUI hits this state while the user is still typing.
#[tokio::test]
async fn preview_empty_messages_renders_chrome_only() {
    let app = router(build_state("empty-msgs"));
    let (status, body) = post_json(
        app,
        "/v1/chat/template/preview",
        serde_json::json!({
            "template": CHATML_TEMPLATE,
            "messages": [],
            "add_generation_prompt": true,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_slice(&body).expect("json");
    // With 0 messages and add_generation_prompt=true, only the
    // assistant primer is emitted. minijinja strips the trailing
    // newline via the `{%- -%}` whitespace control.
    assert_eq!(
        v["rendered"].as_str().unwrap_or_default(),
        "<|im_start|>assistant"
    );
}
