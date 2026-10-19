//! P11 — per-request LoRA adapter routing on the OLLAMA (`/api/*`) and
//! ANTHROPIC (`/v1/messages`) HTTP surfaces (the OpenAI `/v1/chat/completions`
//! + `/v1/completions` surfaces are covered against a real engine in
//! `lora_routing.rs`).
//!
//! These surfaces don't expose a per-token logprob array, so instead of
//! comparing output distributions against a real `ContiguousEngine` (the technique
//! `lora_routing.rs` uses), a `RecordingEngine` captures the
//! `SamplingParams.lora_adapter` that reaches the engine. That directly proves
//! the three contract points per surface:
//!   - a request naming a resident adapter is served WITH it — the adapter name
//!     reaches `sampling.lora_adapter` (via the surface's selector: Ollama's
//!     `options.lora` field or the `model: "<base>:<adapter>"` convention;
//!     Anthropic's `model: "<base>:<adapter>"` — its only natural carrier);
//!   - a base request (no selection) reaches the engine with `lora_adapter ==
//!     None` (byte-identical default);
//!   - an unknown adapter is a clean 4xx and the engine is NEVER invoked (no
//!     silent base serve).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::{
    ChatMessage, Engine, Metrics, SamplingParams, SubmitOpts, Token, TokenStream,
};
use rustllama_server::{router, AppState, ServingModel};
use tower::ServiceExt;

/// Captures the per-request LoRA selection that reaches the engine, plus
/// whether the engine was invoked at all (so the unknown-adapter tests can
/// assert "never a silent base serve").
#[derive(Clone, Default)]
struct LoraRecorder {
    called: Arc<AtomicBool>,
    last_adapter: Arc<Mutex<Option<String>>>,
}

struct LoraRecordingEngine {
    rec: LoraRecorder,
}

impl LoraRecordingEngine {
    fn record_and_stream(&self, s: &SamplingParams) -> TokenStream {
        self.rec.called.store(true, Ordering::Release);
        *self.rec.last_adapter.lock().unwrap() = s.lora_adapter.clone();
        let stream = async_stream::stream! {
            yield Ok(Token { id: 1, text: "x".into(), logprobs: None });
        };
        Box::pin(stream)
    }
}

impl Engine for LoraRecordingEngine {
    fn metrics(&self) -> Metrics {
        Metrics {
            tokens_per_second: 0.0,
            context_used: 0,
            vram_estimate_mb: 0,
            ram_estimate_mb: 0,
            ..Default::default()
        }
    }
    fn n_ctx(&self) -> u32 {
        2048
    }
    fn vocab_size(&self) -> usize {
        256
    }
    fn tokenize(&self, text: &str) -> rustllama_engine::Result<Vec<u32>> {
        Ok(text.bytes().map(|b| b as u32).collect())
    }
    fn chat(
        &self,
        _msgs: &[ChatMessage],
        s: &SamplingParams,
    ) -> rustllama_engine::Result<TokenStream> {
        Ok(self.record_and_stream(s))
    }
    fn generate(&self, _p: &str, s: &SamplingParams) -> rustllama_engine::Result<TokenStream> {
        Ok(self.record_and_stream(s))
    }
    fn chat_scheduled(
        &self,
        _msgs: &[ChatMessage],
        s: &SamplingParams,
        _opts: SubmitOpts,
    ) -> rustllama_engine::Result<TokenStream> {
        Ok(self.record_and_stream(s))
    }
    fn generate_scheduled(
        &self,
        _p: &str,
        s: &SamplingParams,
        _opts: SubmitOpts,
    ) -> rustllama_engine::Result<TokenStream> {
        Ok(self.record_and_stream(s))
    }
    /// Exactly one resident adapter, named `tune-a` — the selector the routing
    /// code validates against before serving.
    fn has_lora_adapter(&self, name: &str) -> bool {
        name == "tune-a"
    }
}

fn build_state(rec: LoraRecorder) -> AppState {
    let serving = ServingModel {
        engine: Arc::new(LoraRecordingEngine { rec }),
        cpu_engine: None,
        model_id: "rec-mock".into(),
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
) -> (StatusCode, serde_json::Value) {
    let req = Request::builder()
        .method("POST")
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let v = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, v)
}

fn last_adapter(rec: &LoraRecorder) -> Option<String> {
    rec.last_adapter.lock().unwrap().clone()
}

// ---------------------------------------------------------------------------
// Ollama /api/generate — `options.lora` field + the `model:adapter` convention
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ollama_generate_options_lora_reaches_sampling() {
    let rec = LoraRecorder::default();
    let state = build_state(rec.clone());
    let (status, v) = post_json(
        router(state),
        "/api/generate",
        serde_json::json!({
            "model": "rec-mock",
            "prompt": "hi",
            "stream": false,
            "options": { "lora": "tune-a" },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert!(rec.called.load(Ordering::Acquire), "engine invoked");
    assert_eq!(
        last_adapter(&rec),
        Some("tune-a".into()),
        "options.lora must reach SamplingParams.lora_adapter"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ollama_generate_model_suffix_convention_reaches_sampling() {
    let rec = LoraRecorder::default();
    let state = build_state(rec.clone());
    let (status, v) = post_json(
        router(state),
        "/api/generate",
        serde_json::json!({
            "model": "rec-mock:tune-a",
            "prompt": "hi",
            "stream": false,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(
        last_adapter(&rec),
        Some("tune-a".into()),
        "`model: base:adapter` must reach SamplingParams.lora_adapter"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ollama_generate_unknown_adapter_is_404_and_never_serves_base() {
    let rec = LoraRecorder::default();
    let state = build_state(rec.clone());
    let (status, _v) = post_json(
        router(state),
        "/api/generate",
        serde_json::json!({
            "model": "rec-mock",
            "prompt": "hi",
            "stream": false,
            "options": { "lora": "ghost" },
        }),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND, "unknown adapter must 404");
    assert!(
        !rec.called.load(Ordering::Acquire),
        "the engine must NOT be invoked on an unknown adapter (no silent base serve)"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ollama_generate_base_request_has_no_adapter() {
    let rec = LoraRecorder::default();
    let state = build_state(rec.clone());
    let (status, _v) = post_json(
        router(state),
        "/api/generate",
        serde_json::json!({ "model": "rec-mock", "prompt": "hi", "stream": false }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(rec.called.load(Ordering::Acquire));
    assert_eq!(
        last_adapter(&rec),
        None,
        "a base request must reach the engine with no adapter (byte-identical default)"
    );
}

// ---------------------------------------------------------------------------
// Anthropic /v1/messages — the `model: "<base>:<adapter>"` convention
// ---------------------------------------------------------------------------

fn anthropic_body(model: &str) -> serde_json::Value {
    serde_json::json!({
        "model": model,
        "max_tokens": 8,
        "messages": [{ "role": "user", "content": "hi" }],
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn anthropic_messages_model_suffix_convention_reaches_sampling() {
    let rec = LoraRecorder::default();
    let state = build_state(rec.clone());
    let (status, v) = post_json(
        router(state),
        "/v1/messages",
        anthropic_body("rec-mock:tune-a"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{v}");
    assert_eq!(
        last_adapter(&rec),
        Some("tune-a".into()),
        "`model: base:adapter` must reach SamplingParams.lora_adapter on /v1/messages"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn anthropic_messages_unknown_adapter_is_404_and_never_serves_base() {
    let rec = LoraRecorder::default();
    let state = build_state(rec.clone());
    let (status, _v) = post_json(
        router(state),
        "/v1/messages",
        anthropic_body("rec-mock:ghost"),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "a known base with an unknown adapter must 404"
    );
    assert!(
        !rec.called.load(Ordering::Acquire),
        "the engine must NOT be invoked on an unknown adapter (no silent base serve)"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn anthropic_messages_base_request_has_no_adapter() {
    let rec = LoraRecorder::default();
    let state = build_state(rec.clone());
    let (status, _v) = post_json(router(state), "/v1/messages", anthropic_body("rec-mock")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(rec.called.load(Ordering::Acquire));
    assert_eq!(
        last_adapter(&rec),
        None,
        "a base request must reach the engine with no adapter (byte-identical default)"
    );
}
