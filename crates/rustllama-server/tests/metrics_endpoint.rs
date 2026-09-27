//! Verifies `GET /v1/metrics` returns the expected JSON shape and
//! reflects the current default model.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use rustllama_engine::{
    ChatMessage, Engine, Metrics, SamplingParams, Token, TokenStream,
};
use rustllama_server::{router, AppState, ServingModel};
use tower::ServiceExt;

/// Identical to the mock in `single_flight_gate.rs` but trimmed —
/// metrics test doesn't care about generation speed; it only checks
/// the JSON shape, which queries `engine.n_ctx()` / `engine.metrics()`.
struct MetricsMock;

impl Engine for MetricsMock {
    fn metrics(&self) -> Metrics {
        Metrics {
            tokens_per_second: 0.0,
            context_used: 7,
            vram_estimate_mb: 0,
            ram_estimate_mb: 0,
            ..Default::default()
        }
    }
    fn n_ctx(&self) -> u32 {
        4096
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
        let s = async_stream::stream! {
            yield Ok(Token { id: 0, text: "hi".to_string(), logprobs: None });
        };
        Ok(Box::pin(s))
    }
    fn generate(
        &self,
        _prompt: &str,
        _s: &SamplingParams,
    ) -> rustllama_engine::Result<TokenStream> {
        let s = async_stream::stream! {
            yield Ok(Token { id: 0, text: "hi".to_string(), logprobs: None });
        };
        Ok(Box::pin(s))
    }
}

fn build_state() -> AppState {
    let serving = ServingModel {
        engine: Arc::new(MetricsMock),
        cpu_engine: None,
        model_id: "metrics-mock".into(),
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
async fn metrics_endpoint_returns_expected_shape() {
    // GET /v1/metrics should return 200 + JSON containing every key
    // the GUI Status page binds to. Pins the contract since any
    // missing key would render as "—" rather than the real value.
    let state = build_state();
    let app = router(state);
    let req = Request::builder()
        .method("GET")
        .uri("/v1/metrics")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("json");

    assert_eq!(v["model_id"], "metrics-mock");
    assert_eq!(v["ctx_size"], 4096);
    assert_eq!(v["ctx_used"], 7);
    assert_eq!(v["pending"], 0);
    assert_eq!(
        v["max_pending"],
        rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL
    );
    // No request has run yet so the `last_*` fields are absent (None
    // — the engine has no CpuEngine attached in this mock setup).
    assert!(v["last_tok_s"].is_null());
    assert!(v["last_prefill_ms"].is_null());
    assert!(v["last_decode_ms"].is_null());
    assert!(v["last_tokens_generated"].is_null());
    assert!(v["last_cache_hit_tokens"].is_null());
    // Uptime present and >= 0.
    assert!(v["uptime_s"].is_u64());
    // Concurrency: at least 1, even on the mock-engine path.
    assert!(v["concurrency"].as_u64().unwrap_or(0) >= 1);
    // kv_dtype: mock engines have no CpuEngine attached so this is
    // null. Pinning that here so the GUI's "—" fallback is exercised.
    assert!(v["kv_dtype"].is_null());
    // Memory fields: present and nonzero (sysinfo on any reasonable
    // host reports >0 total RAM).
    assert!(v["ram_total_bytes"].as_u64().unwrap_or(0) > 0);
    assert!(v["ram_available_bytes"].as_u64().is_some());
    // SYCL device count: 0 when no SYCL device is present at
    // runtime; >=0 when a SYCL device enumerates. Either way the
    // field is a u64.
    assert!(v["sycl_device_count"].as_u64().is_some());

    // Paged KV fields — present as u64 even on the mock engine
    // (which reports 0 for all three). Pins the JSON shape the GUI
    // Status page reads when deciding whether to render the
    // "Paged KV pool" panel.
    assert_eq!(v["paged_total_pages"].as_u64(), Some(0));
    assert_eq!(v["paged_free_pages"].as_u64(), Some(0));
    assert_eq!(v["paged_active_slots"].as_u64(), Some(0));

    // Cumulative-stats field: skipped on mock engines (no
    // `cumulative_stats_snapshot` impl), so it's absent in the
    // serialized JSON. The GUI keys off presence to decide
    // whether to render the "Lifetime stats" panel.
    assert!(
        v.get("cumulative").is_none() || v["cumulative"].is_null(),
        "mock engine must not surface cumulative stats: {v}"
    );
}

#[tokio::test]
async fn metrics_endpoint_surfaces_cumulative_stats_when_engine_tracks_them() {
    // Engine impl that opts in to cumulative tracking by returning
    // a hand-crafted snapshot — exercises the JSON-serialization
    // path without needing a real CpuEngine.
    struct CumulativeMock;
    impl Engine for CumulativeMock {
        fn metrics(&self) -> Metrics {
            Metrics::default()
        }
        fn n_ctx(&self) -> u32 {
            2048
        }
        fn vocab_size(&self) -> usize {
            256
        }
        fn tokenize(&self, _: &str) -> rustllama_engine::Result<Vec<u32>> {
            Ok(vec![])
        }
        fn chat(
            &self,
            _: &[ChatMessage],
            _: &SamplingParams,
        ) -> rustllama_engine::Result<TokenStream> {
            unimplemented!()
        }
        fn generate(
            &self,
            _: &str,
            _: &SamplingParams,
        ) -> rustllama_engine::Result<TokenStream> {
            unimplemented!()
        }
        fn cumulative_stats_snapshot(
            &self,
        ) -> Option<rustllama_engine::CumulativeStatsSnapshot> {
            Some(rustllama_engine::CumulativeStatsSnapshot {
                total_requests: 7,
                total_tokens_prefilled: 100,
                total_cache_hit_tokens: 400,
                total_tokens_generated: 250,
                total_prefill_ms: 123.5,
                total_decode_ms: 4500.25,
                // hit rate = 400 / (400 + 100) = 0.8
                prefix_cache_hit_rate: 0.8,
                total_spec_rounds: 3,
                total_spec_drafted: 24,
                total_spec_accepted: 18,
                // acceptance = 18 / 24 = 0.75
                spec_acceptance_rate: 0.75,
            })
        }
    }
    let serving = ServingModel {
        engine: Arc::new(CumulativeMock),
        cpu_engine: None,
        model_id: "cumulative-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    let app = router(AppState::new(serving, "0.0.0-test".into()));
    let req = Request::builder()
        .method("GET")
        .uri("/v1/metrics")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    let c = &v["cumulative"];
    assert!(
        c.is_object(),
        "cumulative must serialize as a JSON object when the engine opts in: {v}"
    );
    assert_eq!(c["total_requests"], 7);
    assert_eq!(c["total_tokens_prefilled"], 100);
    assert_eq!(c["total_cache_hit_tokens"], 400);
    assert_eq!(c["total_tokens_generated"], 250);
    // Float comparisons through JSON go through f64 — exact-equal
    // is safe for these literals.
    assert!((c["total_prefill_ms"].as_f64().unwrap() - 123.5).abs() < 1e-9);
    assert!((c["total_decode_ms"].as_f64().unwrap() - 4500.25).abs() < 1e-9);
    assert!((c["prefix_cache_hit_rate"].as_f64().unwrap() - 0.8).abs() < 1e-9);
    // Speculation counters pass through verbatim too.
    assert_eq!(c["total_spec_rounds"], 3);
    assert_eq!(c["total_spec_drafted"], 24);
    assert_eq!(c["total_spec_accepted"], 18);
    assert!((c["spec_acceptance_rate"].as_f64().unwrap() - 0.75).abs() < 1e-9);
}

/// When the engine reports nonzero paged_* values (a `PagedBatchEngine`
/// or a `CpuEngine` with `kv_cache_layout = "paged"`), the
/// /v1/metrics JSON surfaces them unchanged — the conversion in
/// the handler is a verbatim pass-through.
#[tokio::test]
async fn metrics_endpoint_surfaces_paged_fields_from_engine() {
    struct PagedMock;
    impl Engine for PagedMock {
        fn metrics(&self) -> Metrics {
            Metrics {
                paged_total_pages: 128,
                paged_free_pages: 96,
                paged_active_slots: 2,
                ..Default::default()
            }
        }
        fn n_ctx(&self) -> u32 {
            2048
        }
        fn vocab_size(&self) -> usize {
            256
        }
        fn tokenize(&self, _: &str) -> rustllama_engine::Result<Vec<u32>> {
            Ok(vec![])
        }
        fn chat(
            &self,
            _: &[ChatMessage],
            _: &SamplingParams,
        ) -> rustllama_engine::Result<TokenStream> {
            unimplemented!()
        }
        fn generate(
            &self,
            _: &str,
            _: &SamplingParams,
        ) -> rustllama_engine::Result<TokenStream> {
            unimplemented!()
        }
    }
    let serving = ServingModel {
        engine: Arc::new(PagedMock),
        cpu_engine: None,
        model_id: "paged-mock".into(),
        gate: ServingModel::new_gate(),
        scheduler: ServingModel::new_scheduler(),
        pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_pending: rustllama_server::DEFAULT_MAX_PENDING_PER_MODEL,
        last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        multi: None,
    };
    let app = router(AppState::new(serving, "0.0.0-test".into()));
    let req = Request::builder()
        .method("GET")
        .uri("/v1/metrics")
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).expect("json");
    assert_eq!(v["paged_total_pages"].as_u64(), Some(128));
    assert_eq!(v["paged_free_pages"].as_u64(), Some(96));
    assert_eq!(v["paged_active_slots"].as_u64(), Some(2));
}
