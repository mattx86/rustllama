//! Integration tests for the Anthropic Messages adapter (`POST /v1/messages`).
//!
//! Uses `MockEngine` so the tests verify request/response shape + SSE
//! event flow without needing a real model. Tool-call shape coverage
//! (Anthropic `tools` → engine grammar → `tool_use` content blocks)
//! is exercised indirectly by [`anthropic::messages`] + the chat module
//! when a real CpuEngine is present; the cancel test in `cancel.rs`
//! also goes through the same code path.

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
        model_id: "rustllama-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    AppState::new(serving, "0.0.0-test".into())
}

async fn post_json(uri: &str, body: serde_json::Value) -> (StatusCode, serde_json::Value) {
    let app = router(build_state());
    let req = Request::builder()
        .method("POST")
        .uri(uri)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let body = serde_json::from_slice::<serde_json::Value>(&bytes).unwrap_or_else(|_| {
        serde_json::Value::String(String::from_utf8_lossy(&bytes).to_string())
    });
    (status, body)
}

fn build_state_with_real_cpu_engine(tag: &str) -> (rustllama_server::AppState, std::path::PathBuf) {
    use rustllama_engine::CpuEngine;
    use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};

    let tmp = std::env::temp_dir().join(format!("rustllama-anthropic-usage-{tag}.gguf"));
    write_synthetic_llama_gguf(&tmp, &SynthLlama::default());
    let cpu = CpuEngine::load_with_tokenizer(&tmp, 16).expect("load cpu engine");
    let model_id = cpu.model_id().to_string();
    let cpu = Arc::new(cpu);
    let serving = ServingModel {
        engine: cpu.clone() as Arc<dyn rustllama_engine::Engine>,
        cpu_engine: Some(cpu),
        model_id,
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    let state = rustllama_server::AppState::new(serving, "0.0.0-test".into());
    (state, tmp)
}

#[tokio::test]
async fn messages_non_streaming_returns_anthropic_shaped_response() {
    let (status, body) = post_json(
        "/v1/messages",
        serde_json::json!({
            "model": "rustllama-mock",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 16,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "response: {body}");
    // Top-level shape.
    assert_eq!(body["type"], "message");
    assert_eq!(body["role"], "assistant");
    assert_eq!(body["model"], "rustllama-mock");
    let id = body["id"].as_str().expect("id is string");
    assert!(id.starts_with("msg_"), "id prefix: {id}");
    // content is an array of blocks with type=text on the mock path.
    let content = body["content"].as_array().expect("content array");
    assert!(!content.is_empty());
    assert_eq!(content[0]["type"], "text");
    assert!(content[0]["text"].as_str().is_some());
    // stop_reason is one of the standard values.
    let stop_reason = body["stop_reason"].as_str().expect("stop_reason");
    assert!(
        matches!(stop_reason, "end_turn" | "max_tokens" | "stop_sequence" | "tool_use"),
        "unexpected stop_reason: {stop_reason}"
    );
    // usage block has the two required counts.
    assert!(body["usage"]["input_tokens"].is_number());
    assert!(body["usage"]["output_tokens"].is_number());
}

#[tokio::test]
async fn messages_system_top_level_field_renders_into_chat() {
    // Anthropic puts `system` at the top level, not inside `messages`. The
    // adapter should pass it through to the chat template as a system
    // message. We can't directly verify the rendered prompt with MockEngine,
    // but we can verify the request is accepted (no 4xx) and the response
    // is well-shaped.
    let (status, body) = post_json(
        "/v1/messages",
        serde_json::json!({
            "model": "rustllama-mock",
            "system": "You are a terse assistant.",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "response: {body}");
    assert_eq!(body["type"], "message");
}

#[tokio::test]
async fn messages_system_blocks_form_also_accepted() {
    // Anthropic also allows `system` to be `[{"type":"text","text":"..."}]`.
    let (status, body) = post_json(
        "/v1/messages",
        serde_json::json!({
            "model": "rustllama-mock",
            "system": [{"type": "text", "text": "You are a terse assistant."}],
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "response: {body}");
}

#[tokio::test]
async fn messages_accepts_image_content_blocks_as_text_passthrough() {
    // Multimodal Claude SDK calls (Claude Code with screenshot attached,
    // anthropic-sdk-* clients constructing image blocks) must not 400
    // even though we don't have a VLM. The adapter renders image
    // blocks as a `[image: <media_type>]` placeholder so the
    // text-only model still gets a coherent prompt.
    let (status, body) = post_json(
        "/v1/messages",
        serde_json::json!({
            "model": "rustllama-mock",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Look at this: "},
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": "fake-base64-data"
                    }},
                    {"type": "text", "text": " — what is it?"}
                ]
            }],
            "max_tokens": 16
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(body["type"], "message");
}

#[tokio::test]
async fn messages_accepts_openai_shape_image_url_blocks() {
    // Some clients send the OpenAI `image_url` shape through Anthropic-
    // shaped APIs anyway. The adapter should handle both gracefully.
    let (status, _) = post_json(
        "/v1/messages",
        serde_json::json!({
            "model": "rustllama-mock",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe: "},
                    {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
                ]
            }],
            "max_tokens": 8
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn messages_unknown_model_returns_404() {
    let (status, body) = post_json(
        "/v1/messages",
        serde_json::json!({
            "model": "no-such-model",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    // body should mention the unknown model.
    let text = body.as_str().unwrap_or("");
    assert!(text.contains("no-such-model"), "body: {text}");
}

#[tokio::test]
async fn messages_non_streaming_omits_usage_extension_fields_under_mock_engine() {
    // Negative: without a real engine, `usage` should carry just the
    // two required Anthropic fields. The extension fields use
    // `skip_serializing_if = "Option::is_none"` so they're absent.
    // This catches a regression where someone accidentally hard-codes
    // `Some(0.0)` instead of `None` for the mock path.
    let (status, body) = post_json(
        "/v1/messages",
        serde_json::json!({
            "model": "rustllama-mock",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    let usage = &body["usage"];
    assert!(usage["input_tokens"].is_number());
    assert!(usage["output_tokens"].is_number());
    assert!(
        usage.get("prefill_ms").is_none(),
        "prefill_ms should be absent under MockEngine, got {usage}"
    );
    assert!(usage.get("decode_ms").is_none());
    assert!(usage.get("cache_hit_tokens").is_none());
}

#[tokio::test]
async fn messages_non_streaming_surfaces_usage_extension_fields_with_real_engine() {
    // With a real CpuEngine, the non-stream response must populate
    // the rustllama extension fields (`prefill_ms`, `decode_ms`,
    // `cache_hit_tokens`, `tokens_prefilled`) on `usage` so editor
    // UIs that show TTFT have the data they need.
    let (state, tmp) = build_state_with_real_cpu_engine("non-stream");
    let app = rustllama_server::router(state);
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
    })
    .to_string();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let body: serde_json::Value =
        serde_json::from_slice(&bytes).expect("response is JSON");
    let usage = &body["usage"];
    assert!(usage["input_tokens"].is_number());
    assert!(usage["output_tokens"].is_number());
    // The four extension fields must all be present.
    assert!(
        usage["prefill_ms"].is_number(),
        "prefill_ms missing: {body}"
    );
    assert!(usage["decode_ms"].is_number(), "decode_ms missing: {body}");
    assert!(
        usage["cache_hit_tokens"].is_number(),
        "cache_hit_tokens missing: {body}"
    );
    assert!(
        usage["tokens_prefilled"].is_number(),
        "tokens_prefilled missing: {body}"
    );
    let _ = std::fs::remove_file(&tmp);
}

#[tokio::test]
async fn messages_streaming_surfaces_usage_extension_fields_with_real_engine() {
    // Same as above for the SSE path: the `message_delta` event
    // carries `usage.prefill_ms` / etc. when a real engine is loaded.
    let (state, tmp) = build_state_with_real_cpu_engine("stream");
    let app = rustllama_server::router(state);
    let body = serde_json::json!({
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "stream": true,
    })
    .to_string();
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let text = String::from_utf8_lossy(&bytes).to_string();

    // Find the message_delta event's `data:` line and parse it.
    let mut lines = text.lines().peekable();
    let mut found_usage = false;
    while let Some(line) = lines.next() {
        if line == "event: message_delta" {
            // Next non-empty line is the matching `data: {...}`.
            for data_line in lines.by_ref() {
                if let Some(payload) = data_line.strip_prefix("data: ") {
                    let v: serde_json::Value =
                        serde_json::from_str(payload).expect("delta JSON");
                    let usage = &v["usage"];
                    assert!(
                        usage["prefill_ms"].is_number(),
                        "stream prefill_ms missing: {v}"
                    );
                    assert!(usage["decode_ms"].is_number(), "stream decode_ms missing: {v}");
                    assert!(
                        usage["cache_hit_tokens"].is_number(),
                        "stream cache_hit_tokens missing: {v}"
                    );
                    assert!(
                        usage["tokens_prefilled"].is_number(),
                        "stream tokens_prefilled missing: {v}"
                    );
                    found_usage = true;
                    break;
                }
            }
            break;
        }
    }
    assert!(found_usage, "no message_delta event with usage: {text}");
    let _ = std::fs::remove_file(&tmp);
}

#[tokio::test]
async fn messages_streaming_emits_full_event_sequence() {
    // Anthropic SSE stream order:
    //   message_start
    //   content_block_start
    //   content_block_delta+
    //   content_block_stop
    //   message_delta
    //   message_stop
    let app = router(build_state());
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::json!({
                "model": "rustllama-mock",
                "messages": [{"role": "user", "content": "tell a tiny story"}],
                "max_tokens": 8,
                "stream": true,
            })
            .to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();

    // Extract every `event:` line, in order.
    let events: Vec<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("event: "))
        .collect();

    assert!(
        events.first().map(|e| *e == "message_start").unwrap_or(false),
        "first event should be message_start, got: {events:?}"
    );
    assert!(
        events.iter().any(|e| *e == "content_block_start"),
        "no content_block_start in: {events:?}"
    );
    assert!(
        events.iter().any(|e| *e == "content_block_delta"),
        "no content_block_delta in: {events:?}"
    );
    assert!(
        events.iter().any(|e| *e == "content_block_stop"),
        "no content_block_stop in: {events:?}"
    );
    assert!(
        events.iter().any(|e| *e == "message_delta"),
        "no message_delta in: {events:?}"
    );
    assert!(
        events.last().map(|e| *e == "message_stop").unwrap_or(false),
        "last event should be message_stop, got: {events:?}"
    );
}

// ---- cache_control acceptance + Anthropic-native usage fields ---------------

#[tokio::test]
async fn cache_control_on_system_text_block_parses_cleanly() {
    // Claude Code / Claude Desktop / langchain-anthropic auto-annotate
    // long system prompts with `cache_control: {type: "ephemeral"}`.
    // The request must parse and the response must come back 200 —
    // the marker is informational today (engine uses implicit LCP)
    // but the wire shape MUST accept it.
    let (status, body) = post_json(
        "/v1/messages",
        serde_json::json!({
            "model": "rustllama-mock",
            "max_tokens": 16,
            "system": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant.",
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            "messages": [
                {"role": "user", "content": "hi"}
            ],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    // Sanity: response shape unchanged.
    assert_eq!(body["type"], "message");
}

#[tokio::test]
async fn cache_control_on_user_content_block_parses_cleanly() {
    // The annotation also commonly appears on user-message content
    // blocks when the client is caching long context (e.g. a
    // codebase dump). Must accept it the same way.
    let (status, body) = post_json(
        "/v1/messages",
        serde_json::json!({
            "model": "rustllama-mock",
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "context block ...",
                            "cache_control": {"type": "ephemeral"}
                        },
                        {"type": "text", "text": "now my question"}
                    ]
                }
            ],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "body: {body}");
    assert_eq!(body["type"], "message");
}

#[tokio::test]
async fn cache_control_with_unknown_type_still_accepted() {
    // Forward-compat: a future Anthropic-side type like
    // `"persistent"` must NOT 400. We accept the field with any
    // string in `type` so the server doesn't reject requests when
    // upstream extends the marker.
    let (status, _) = post_json(
        "/v1/messages",
        serde_json::json!({
            "model": "rustllama-mock",
            "max_tokens": 16,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "hi",
                            "cache_control": {"type": "future_persistent_variant"}
                        }
                    ]
                }
            ],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn anthropic_usage_includes_cache_read_input_tokens_on_real_engine() {
    // Real CpuEngine: the engine's RequestStats.cache_hit_tokens
    // flows into AnthropicUsage as cache_read_input_tokens under
    // the Anthropic-native field name. Even on a single request
    // (no prior cache state) the field MUST be present and
    // numeric so Anthropic SDK consumers can rely on it.
    let (state, tmp) = build_state_with_real_cpu_engine("cache-read-shape");
    let app = router(state);
    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            // Omit `model` so the request hits the default-registered
            // synth engine — matches the pattern used by the existing
            // real-engine usage tests above.
            serde_json::json!({
                "max_tokens": 4,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": "<|tok_5|>"}]
            })
            .to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("json");

    let usage = &v["usage"];
    assert!(
        usage["cache_read_input_tokens"].is_u64(),
        "cache_read_input_tokens must be a number: {usage}"
    );
    // First request → 0 (nothing in cache yet). The field's *presence*
    // is what Anthropic SDK consumers check; the value confirms the
    // mapping from RequestStats.cache_hit_tokens is wired.
    assert_eq!(
        usage["cache_read_input_tokens"].as_u64().unwrap(),
        0,
        "first request has no cache hits"
    );
    // cache_creation_input_tokens is reported as 0 until per-marker
    // boundaries land — pin that here so a future change is
    // intentional.
    assert_eq!(
        usage["cache_creation_input_tokens"].as_u64().unwrap(),
        0,
        "cache_creation always 0 until per-marker caching lands"
    );

    let _ = std::fs::remove_file(&tmp);
}
