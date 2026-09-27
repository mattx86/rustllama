//! Multi-flight serving acceptance: with `[server].concurrency = N`,
//! the ServingModel allocates `N` engine forks and round-robin
//! distributes incoming requests across them. Two concurrent requests
//! against a concurrency=2 ServingModel must run their handler logic
//! in parallel (not serialize on the gate), which we observe by
//! measuring elapsed wall-clock time against the engine's per-token
//! sleep budget.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::{
    ChatMessage, Engine, Metrics, SamplingParams, Token, TokenStream,
};
use rustllama_server::{router, AppState, MultiFlightPool, ServingModel};
use tower::ServiceExt;

/// 6 tokens × 30ms each = ~180ms per request. With concurrency=2 and
/// two parallel requests, the wall-clock should be ~180ms (overlap),
/// not ~360ms (serialized). We use a generous margin (≤300ms) to
/// avoid CI flakiness while still rejecting the serialized case.
const TOKENS_PER_REQUEST: u32 = 6;
const MS_PER_TOKEN: u64 = 30;

/// Tracks active in-flight requests so the test can assert that
/// during a parallel run, the count actually reached >= 2 (and not
/// just "request A finished, then request B started" — which would
/// also produce a fast wall-clock).
#[derive(Clone, Default)]
struct OverlapWitness {
    active: Arc<AtomicUsize>,
    peak: Arc<AtomicUsize>,
}

struct SlowEngine {
    witness: OverlapWitness,
}

impl Engine for SlowEngine {
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
        _s: &SamplingParams,
    ) -> rustllama_engine::Result<TokenStream> {
        Ok(slow_stream(self.witness.clone()))
    }
    fn generate(
        &self,
        _prompt: &str,
        _s: &SamplingParams,
    ) -> rustllama_engine::Result<TokenStream> {
        Ok(slow_stream(self.witness.clone()))
    }
}

fn slow_stream(w: OverlapWitness) -> TokenStream {
    let s = async_stream::stream! {
        // Mark the request as in-flight; track peak concurrency so
        // the test can assert real overlap rather than just fast
        // sequential turnaround.
        let cur = w.active.fetch_add(1, Ordering::AcqRel) + 1;
        let mut peak = w.peak.load(Ordering::Acquire);
        while cur > peak {
            match w.peak.compare_exchange(
                peak,
                cur,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }

        for i in 0..TOKENS_PER_REQUEST {
            tokio::time::sleep(Duration::from_millis(MS_PER_TOKEN)).await;
            yield Ok(Token {
                id: i,
                text: format!("t{i} "),
                logprobs: None,
            });
        }

        w.active.fetch_sub(1, Ordering::AcqRel);
    };
    Box::pin(s)
}

/// Synthesize a ServingModel with `concurrency` mock engines in the
/// multi-flight pool. We can't use the production `with_concurrency`
/// path because it requires a real CpuEngine to call
/// `fork_for_concurrent_use`. Instead, we build the pool by hand
/// with N distinct `SlowEngine` instances sharing the same witness
/// — proving the round-robin path picks each pool entry.
fn build_state(concurrency: usize) -> (AppState, OverlapWitness) {
    let witness = OverlapWitness::default();
    let engines: Vec<Arc<dyn Engine>> = (0..concurrency)
        .map(|_| Arc::new(SlowEngine { witness: witness.clone() }) as Arc<dyn Engine>)
        .collect();
    let pool = MultiFlightPool {
        engines: engines.clone(),
        cpu_engines: Vec::new(), // mock path — no CpuEngine forks
        next: Arc::new(AtomicUsize::new(0)),
    };
    let serving = ServingModel {
        engine: engines[0].clone(),
        cpu_engine: None,
        model_id: "multi-mock".into(),
        gate: ServingModel::new_gate_for_concurrency(concurrency),
        scheduler: ServingModel::new_scheduler_for_concurrency(concurrency),
        pending: Arc::new(AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: Some(pool),
    };
    (AppState::new(serving, "0.0.0-test".into()), witness)
}

async fn one_chat_request(app: axum::Router) -> StatusCode {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(
            serde_json::json!({
                "model": "multi-mock",
                "messages": [{"role":"user","content":"hi"}],
                "stream": false,
                "max_tokens": TOKENS_PER_REQUEST,
            })
            .to_string(),
        ))
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let _ = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    status
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrency_two_runs_two_requests_in_parallel() {
    // With concurrency=2, two parallel requests should finish in
    // ~one-request worth of wall-clock time, not two. We also assert
    // peak in-flight >= 2 to rule out the "fast serial turnaround"
    // failure mode.
    let (state, witness) = build_state(2);
    let app = router(state);

    let start = Instant::now();
    let (a, b) = tokio::join!(
        one_chat_request(app.clone()),
        one_chat_request(app.clone()),
    );
    let elapsed = start.elapsed();

    assert_eq!(a, StatusCode::OK, "request a should succeed");
    assert_eq!(b, StatusCode::OK, "request b should succeed");

    // One request is ~180ms (6 * 30ms). Serialized would be ~360ms.
    // The threshold sits in the middle — comfortably faster than
    // serial but loose enough to absorb runtime scheduling noise.
    assert!(
        elapsed < Duration::from_millis(300),
        "parallel requests took {elapsed:?} — looks serialized \
         (expected <300ms with concurrency=2; serial would be ~360ms)",
    );

    // Peak observed concurrency must be exactly 2 (the gate permits)
    // — if it stayed at 1, the gate was still single-flight.
    let peak = witness.peak.load(Ordering::Acquire);
    assert_eq!(
        peak, 2,
        "expected peak in-flight = 2 (concurrency permit), got {peak}",
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrency_one_serializes() {
    // Regression: with concurrency=1 (single-flight), the same
    // two-request workload MUST serialize. This pins that
    // the multi-flight wiring doesn't accidentally let single-flight
    // requests overlap (which would race on the engine's KV cache
    // in production).
    let (state, witness) = build_state(1);
    let app = router(state);

    let (a, b) = tokio::join!(
        one_chat_request(app.clone()),
        one_chat_request(app.clone()),
    );
    assert_eq!(a, StatusCode::OK);
    assert_eq!(b, StatusCode::OK);

    let peak = witness.peak.load(Ordering::Acquire);
    assert_eq!(
        peak, 1,
        "single-flight must serialize — peak in-flight should be 1, got {peak}",
    );
}
