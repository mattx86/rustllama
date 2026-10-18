//! P14-http regression tests: the `X-RustLlama-Priority` /
//! `X-RustLlama-Deadline-Ms` request headers are parsed and threaded through
//! the chat handler into the SUBMITTED engine request's `SubmitOpts`
//! (admission `priority` + per-request `deadline`), so the paged driver's
//! priority heap + deadline enforcement actually see them.
//!
//! The engine-layer behavior (heap ordering, deadline-cancels-with-timeout) is
//! pinned in `rustllama-engine/tests/paged_batch_engine.rs`. These tests pin
//! the HTTP SURFACE: (a) a `RecordingEngine` captures the `SubmitOpts` that
//! reach `Engine::chat_scheduled`, proving the headers arrive at the engine
//! request; (b) a request with no headers is byte-identical (DEFAULT priority,
//! no deadline); (c) an already-expired deadline terminates the stream (the
//! RecordingEngine mirrors the driver's `DeadlineExceeded` terminal), so a
//! too-slow request is time-limited rather than running unbounded.

use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Instant;

use axum::body::{to_bytes, Body};
use axum::http::{header, HeaderMap, Request, StatusCode};
use rustllama_engine::batch_scheduler::Priority;
use rustllama_engine::{
    ChatMessage, Engine, EngineError, Metrics, SamplingParams, SubmitOpts, Token, TokenStream,
};
use rustllama_server::{router, AppState, ReqSchedule, ServingModel};
use tower::ServiceExt;

// ---------------------------------------------------------------------------
// Unit: ReqSchedule::from_headers
// ---------------------------------------------------------------------------

fn hdrs(pairs: &[(&str, &str)]) -> HeaderMap {
    let mut h = HeaderMap::new();
    for (k, v) in pairs {
        h.insert(
            axum::http::HeaderName::from_bytes(k.as_bytes()).unwrap(),
            v.parse().unwrap(),
        );
    }
    h
}

#[test]
fn from_headers_defaults_when_absent() {
    let s = ReqSchedule::from_headers(&HeaderMap::new());
    assert!(s.is_default(), "no headers ⇒ default schedule");
    assert_eq!(s.priority(), Priority::DEFAULT);
    // to_submit_opts(None) must equal SubmitOpts::default() on the default path.
    let opts = s.to_submit_opts(None);
    assert_eq!(opts.priority, Priority::DEFAULT);
    assert!(opts.deadline.is_none());
    assert!(opts.cancel_flag.is_none());
}

#[test]
fn from_headers_parses_named_and_numeric_priority() {
    assert_eq!(
        ReqSchedule::from_headers(&hdrs(&[("x-rustllama-priority", "high")])).priority(),
        Priority::HIGH
    );
    assert_eq!(
        ReqSchedule::from_headers(&hdrs(&[("x-rustllama-priority", "low")])).priority(),
        Priority::LOW
    );
    assert_eq!(
        ReqSchedule::from_headers(&hdrs(&[("x-rustllama-priority", "7")])).priority(),
        Priority(7)
    );
    // Garbage falls through to DEFAULT (advisory hint, never a 400).
    assert_eq!(
        ReqSchedule::from_headers(&hdrs(&[("x-rustllama-priority", "nonsense")])).priority(),
        Priority::DEFAULT
    );
}

#[test]
fn from_headers_parses_deadline_ms() {
    let s = ReqSchedule::from_headers(&hdrs(&[("x-rustllama-deadline-ms", "5000")]));
    assert!(!s.is_default());
    let before = Instant::now();
    let opts = s.to_submit_opts(None);
    let d = opts.deadline.expect("deadline present");
    // Deadline resolves to an absolute Instant ~5s out.
    assert!(d > before, "deadline must be in the future");
    assert!(d <= before + std::time::Duration::from_millis(5_050));
    // A zero-ms deadline is kept (already-expired ⇒ cancelled immediately).
    let z = ReqSchedule::from_headers(&hdrs(&[("x-rustllama-deadline-ms", "0")]));
    assert!(z.to_submit_opts(None).deadline.is_some());
    // Non-numeric ⇒ no deadline.
    let bad = ReqSchedule::from_headers(&hdrs(&[("x-rustllama-deadline-ms", "soon")]));
    assert!(bad.to_submit_opts(None).deadline.is_none());
}

// ---------------------------------------------------------------------------
// Integration: headers reach the engine request via chat_scheduled
// ---------------------------------------------------------------------------

/// Records the `SubmitOpts` that reach `chat_scheduled` / `generate_scheduled`
/// and mirrors the paged driver's two relevant behaviors: a normal request
/// streams a few tokens; an already-expired deadline ends immediately with the
/// `DeadlineExceeded` terminal.
#[derive(Clone, Default)]
struct Recorder {
    called: Arc<AtomicBool>,
    priority: Arc<AtomicU8>,
    deadline_seen: Arc<AtomicBool>,
}

struct RecordingEngine {
    rec: Recorder,
}

impl RecordingEngine {
    fn stream_for(&self, opts: &SubmitOpts) -> TokenStream {
        self.rec.called.store(true, Ordering::Release);
        self.rec.priority.store(opts.priority.0, Ordering::Release);
        self.rec
            .deadline_seen
            .store(opts.deadline.is_some(), Ordering::Release);
        // Mirror the driver: an already-elapsed deadline ⇒ immediate timeout
        // terminal, no tokens. Otherwise stream a few tokens.
        let expired = opts.deadline.is_some_and(|d| Instant::now() >= d);
        if expired {
            let s = async_stream::stream! {
                yield Err(EngineError::DeadlineExceeded);
            };
            Box::pin(s)
        } else {
            let s = async_stream::stream! {
                for i in 0..4u32 {
                    yield Ok(Token { id: i, text: format!("t{i} "), logprobs: None });
                }
            };
            Box::pin(s)
        }
    }
}

impl Engine for RecordingEngine {
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
        msgs: &[ChatMessage],
        s: &SamplingParams,
    ) -> rustllama_engine::Result<TokenStream> {
        // Default (no-opts) path delegates to chat_scheduled with default opts —
        // so even a client that never sends headers is recorded as DEFAULT.
        self.chat_scheduled(msgs, s, SubmitOpts::default())
    }
    fn generate(&self, p: &str, s: &SamplingParams) -> rustllama_engine::Result<TokenStream> {
        self.generate_scheduled(p, s, SubmitOpts::default())
    }
    fn chat_scheduled(
        &self,
        _msgs: &[ChatMessage],
        _s: &SamplingParams,
        opts: SubmitOpts,
    ) -> rustllama_engine::Result<TokenStream> {
        Ok(self.stream_for(&opts))
    }
    fn generate_scheduled(
        &self,
        _p: &str,
        _s: &SamplingParams,
        opts: SubmitOpts,
    ) -> rustllama_engine::Result<TokenStream> {
        Ok(self.stream_for(&opts))
    }
}

fn build_state(rec: Recorder) -> AppState {
    let serving = ServingModel {
        engine: Arc::new(RecordingEngine { rec }),
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

async fn chat_request(app: axum::Router, extra: &[(&str, &str)]) -> (StatusCode, String) {
    let mut b = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json");
    for (k, v) in extra {
        b = b.header(*k, *v);
    }
    let req = b
        .body(Body::from(
            serde_json::json!({
                "model": "rec-mock",
                "messages": [{"role":"user","content":"hi"}],
                "stream": true,
                "max_tokens": 8,
            })
            .to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    (status, String::from_utf8_lossy(&bytes).to_string())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn priority_and_deadline_headers_reach_the_engine_request() {
    let rec = Recorder::default();
    let state = build_state(rec.clone());
    let app = router(state);

    let (status, _body) = chat_request(
        app,
        &[
            ("x-rustllama-priority", "high"),
            ("x-rustllama-deadline-ms", "5000"),
        ],
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    assert!(rec.called.load(Ordering::Acquire), "engine was invoked");
    assert_eq!(
        rec.priority.load(Ordering::Acquire),
        Priority::HIGH.0,
        "the X-RustLlama-Priority header must reach SubmitOpts.priority"
    );
    assert!(
        rec.deadline_seen.load(Ordering::Acquire),
        "the X-RustLlama-Deadline-Ms header must reach SubmitOpts.deadline"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_headers_submit_at_default_priority_no_deadline() {
    let rec = Recorder::default();
    let state = build_state(rec.clone());
    let app = router(state);

    let (status, _body) = chat_request(app, &[]).await;
    assert_eq!(status, StatusCode::OK);

    assert!(rec.called.load(Ordering::Acquire));
    assert_eq!(
        rec.priority.load(Ordering::Acquire),
        Priority::DEFAULT.0,
        "a request with no scheduling headers submits at DEFAULT priority"
    );
    assert!(
        !rec.deadline_seen.load(Ordering::Acquire),
        "a request with no deadline header submits with no deadline (byte-identical default)"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn already_expired_deadline_time_limits_the_stream() {
    let rec = Recorder::default();
    let state = build_state(rec.clone());
    let app = router(state);

    // 0 ms ⇒ already expired when the engine inspects it. The RecordingEngine
    // mirrors the driver's `DeadlineExceeded` terminal, so the server's stream
    // loop ends immediately (no generated tokens), and the SSE finishes quickly
    // rather than streaming the 4-token normal body.
    let (status, body) = chat_request(app, &[("x-rustllama-deadline-ms", "0")]).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        rec.deadline_seen.load(Ordering::Acquire),
        "the deadline must reach the engine"
    );
    // The normal path streams `t0 t1 t2 t3`; an expired-deadline request must
    // NOT contain those content deltas (it ended on the timeout terminal).
    assert!(
        !body.contains("t0") && !body.contains("t3"),
        "an already-expired request must not stream the normal tokens: {body}"
    );
    assert!(
        body.contains("[DONE]"),
        "stream still terminates cleanly: {body}"
    );
}
