//! Tool-calling on the continuous-batching (paged) engine.
//!
//! Regression guard for the fix that lets `/v1/chat/completions` tool/function
//! requests run on the `PagedBatchEngine`. The old server tools handlers
//! rendered the chat prompt + built the tool-call grammar SERVER-SIDE off the
//! CONCRETE `serving.cpu_engine` / `handle.cpu_engine`, which is `None` when a
//! model is served by the paged engine — so every tools request against a
//! paged model got a hard `503 "tools/functions requires a real engine (not
//! the MockEngine)"`. The handlers now source the tokenizer + context window
//! GENERICALLY via the `Engine` trait (`tokenizer_arc()` / `max_ctx()`), and
//! generation already ran through the generic `handle.engine.generate(...)`.
//! The paged engine's per-slot `build_paged_grammar` carries the tool-call
//! grammar exactly like the `ContiguousEngine`, so nothing downstream had to change.
//!
//! What is validatable HERE (no GPU / real tokenizer needed):
//!   - A tools request against a PAGED model is NOT rejected by the removed
//!     503 guard — it reaches the generic engine's `generate()`. The synthetic
//!     GGUF's `<tok_N>`-only vocab can't encode the rendered ChatML prompt, so
//!     the paged engine then bails with "prompt tokenized to zero tokens"
//!     (HTTP 500) — which PROVES the request traversed the full tools pipeline
//!     (guard → template render → grammar build → gate → `generate`) rather
//!     than being turned away at the guard. A real tokenizer yields a 200.
//!   - A genuinely model-less server (MockEngine, no tokenizer) still returns
//!     the reworded graceful 503 ("tool-calling requires a loaded model").

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::paged_batch::PagedBatchEngine;
use rustllama_engine::MockEngine;
use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};
use rustllama_server::{router, AppState, ServingModel, DEFAULT_MAX_PENDING_PER_MODEL};
use tower::ServiceExt;

/// A single `get_weather` tool in OpenAI shape.
fn weather_tools() -> serde_json::Value {
    serde_json::json!([{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": { "location": { "type": "string" } },
                "required": ["location"]
            }
        }
    }])
}

async fn post_chat(app: axum::Router, body: serde_json::Value) -> (StatusCode, String) {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    (status, String::from_utf8_lossy(&bytes).to_string())
}

/// A paged-engine tools request is NOT turned away by the removed MockEngine
/// 503 guard: it reaches the generic engine's `generate()`, so the tool-call
/// grammar path is live on the continuous-batching engine.
#[tokio::test]
async fn paged_engine_tools_request_reaches_generation_not_503() {
    let tmp = std::env::temp_dir().join("rustllama-paged-tools.gguf");
    write_synthetic_llama_gguf(&tmp, &SynthLlama::default());

    let paged = Arc::new(PagedBatchEngine::load(&tmp, 32, 2).expect("load paged engine"));
    let model_id = paged.model_id().to_string();
    let serving = ServingModel::new_fused_decode_paged(
        paged,
        model_id.clone(),
        DEFAULT_MAX_PENDING_PER_MODEL,
        2,
    );
    let state = AppState::new(serving, "0.0.0-test".into());
    let app = router(state);

    let (status, body) = post_chat(
        app,
        serde_json::json!({
            "model": model_id,
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "tools": weather_tools(),
            "tool_choice": "auto",
            "stream": false,
            "max_tokens": 8,
            "temperature": 0.0,
        }),
    )
    .await;

    // The fix: no more hard 503 on the paged engine.
    assert_ne!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "paged tools request must not 503 (the removed cpu_engine guard); body={body}"
    );
    assert!(
        !body.contains("requires a real engine"),
        "the old MockEngine guard message must be gone; body={body}"
    );
    // The request traversed the full tools pipeline to the generic engine's
    // generate(). With a real tokenizer this is a 200 chat/tool_calls; with
    // the synth `<tok_N>` vocab the paged engine bails on the zero-token encode
    // (HTTP 500) — either outcome proves the guard no longer short-circuits it
    // (the removed guard was a 503, as are backpressure refusals). The exact
    // synth failure mode ("prompt tokenized to zero tokens") is asserted when
    // we see the 500.
    assert!(
        status == StatusCode::OK || status == StatusCode::INTERNAL_SERVER_ERROR,
        "paged tools request should reach generation (200) or hit the synth \
         tokenizer's zero-token limit (500), not be turned away; got {status}, body={body}"
    );
    if status == StatusCode::INTERNAL_SERVER_ERROR {
        assert!(
            body.contains("zero tokens"),
            "the 500 must come from the synth engine's generate(), proving the \
             tools path reached it; body={body}"
        );
    }

    let _ = std::fs::remove_file(&tmp);
}

/// The streaming tools path takes the same generic sourcing — it must also
/// clear the removed guard rather than 503.
#[tokio::test]
async fn paged_engine_streaming_tools_request_not_503() {
    let tmp = std::env::temp_dir().join("rustllama-paged-tools-stream.gguf");
    write_synthetic_llama_gguf(&tmp, &SynthLlama::default());

    let paged = Arc::new(PagedBatchEngine::load(&tmp, 32, 2).expect("load paged engine"));
    let model_id = paged.model_id().to_string();
    let serving = ServingModel::new_fused_decode_paged(
        paged,
        model_id.clone(),
        DEFAULT_MAX_PENDING_PER_MODEL,
        2,
    );
    let state = AppState::new(serving, "0.0.0-test".into());
    let app = router(state);

    let (status, body) = post_chat(
        app,
        serde_json::json!({
            "model": model_id,
            "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
            "tools": weather_tools(),
            "tool_choice": "auto",
            "stream": true,
            "max_tokens": 8,
            "temperature": 0.0,
        }),
    )
    .await;

    assert_ne!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "paged streaming tools request must not 503; body={body}"
    );
    assert!(
        !body.contains("requires a real engine"),
        "the old MockEngine guard message must be gone; body={body}"
    );
}

/// A genuinely model-less server (MockEngine, no tokenizer) still returns the
/// reworded graceful 503 — the only case the tools path legitimately refuses.
#[tokio::test]
async fn modelless_tools_request_returns_reworded_503() {
    let serving = ServingModel {
        engine: Arc::new(MockEngine),
        cpu_engine: None,
        model_id: "tools-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    let state = AppState::new(serving, "0.0.0-test".into());
    let app = router(state);

    let (status, body) = post_chat(
        app,
        serde_json::json!({
            "model": "tools-mock",
            "messages": [{"role": "user", "content": "What is the weather?"}],
            "tools": weather_tools(),
            "stream": false,
            "max_tokens": 8,
        }),
    )
    .await;

    assert_eq!(
        status,
        StatusCode::SERVICE_UNAVAILABLE,
        "a model-less (tokenizer-less) server must refuse tools; body={body}"
    );
    assert!(
        body.contains("tool-calling requires a loaded model"),
        "503 must carry the reworded graceful message; body={body}"
    );
    // The old wording must be gone.
    assert!(
        !body.contains("MockEngine"),
        "reworded 503 must not mention the MockEngine; body={body}"
    );
}
