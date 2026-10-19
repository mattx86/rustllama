//! P11 Phase 3 — HTTP routing of per-request LoRA adapter selection.
//!
//! Anchors (via `/v1/completions` against a real `ContiguousEngine` + synth model):
//!   - a request naming a resident adapter (explicit `lora` field OR the
//!     `model: "<base>:<adapter>"` convention) is served WITH it — the output
//!     distribution differs from the un-adapted base;
//!   - a base request (no selection) is unchanged;
//!   - an unknown adapter name is a clean 4xx (never a silent base serve);
//!   - the `POST /v1/lora/load` + `GET /v1/lora` endpoints attach + list.
//!
//! Observable: the synthetic model's vocab is special `<|tok_i|>` tokens, so the
//! completion TEXT decodes to empty (special tokens skipped). The per-token
//! `logprobs` array, however, reflects the output DISTRIBUTION — applying an
//! adapter perturbs the logits and therefore the logprobs, which is the robust
//! signal here (independent of argmax / detokenization). A moderate adapter
//! alpha keeps the softmax from saturating so the numbers stay informative.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{header, Request, StatusCode};
use rustllama_engine::ContiguousEngine;
use rustllama_gguf::synth::{
    write_synthetic_llama_gguf, write_synthetic_lora_adapter_gguf, SynthLlama, SynthLoraAdapter,
};
use rustllama_server::{router, AppState, ServingModel};
use tower::ServiceExt;

fn fixtures(tag: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    // The synth GPT2 tokenizer cannot encode a plain string into its special
    // `<|tok_i|>` vocab, so a text prompt yields 0 tokens (no generation). The
    // FIM path injects real FIM marker tokens, which DO generate — so the base
    // model ships the FIM markers and the request below is a FIM completion.
    let base = std::env::temp_dir().join(format!("rustllama-lora-routing-{tag}-base.gguf"));
    write_synthetic_llama_gguf(
        &base,
        &SynthLlama {
            include_fim_tokens: true,
            ..SynthLlama::default()
        },
    );
    let adapter = std::env::temp_dir().join(format!("rustllama-lora-routing-{tag}-ad.gguf"));
    write_synthetic_lora_adapter_gguf(
        &adapter,
        &SynthLoraAdapter {
            base: SynthLlama::default(),
            rank: 8,
            alpha: 32.0, // moderate: shifts the distribution without saturating it
            seed: 0xC0FFEE,
        },
    );
    (base, adapter)
}

/// Build serving state. When `attach`, the adapter is attached to the engine
/// under `tune-a` before the state is built (so a request can select it).
fn build_state(base: &std::path::Path, adapter: Option<&std::path::Path>) -> AppState {
    let cpu = ContiguousEngine::load_with_tokenizer(base, 64).expect("load engine");
    if let Some(ad) = adapter {
        cpu.attach_lora_adapter("tune-a", ad)
            .expect("attach adapter");
    }
    let cpu = Arc::new(cpu);
    let serving = ServingModel {
        engine: cpu.clone() as Arc<dyn rustllama_engine::Engine>,
        cpu_engine: Some(cpu),
        model_id: "lora-test".into(),
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

async fn get_json(app: axum::Router, path: &str) -> (StatusCode, serde_json::Value) {
    let req = Request::builder()
        .method("GET")
        .uri(path)
        .body(Body::empty())
        .unwrap();
    let resp = app.oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = to_bytes(resp.into_body(), 1 << 20).await.unwrap();
    let v = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, v)
}

/// The per-token logprob array of the first choice — the distribution
/// fingerprint used to tell an adapted completion from the base.
fn logprobs_of(v: &serde_json::Value) -> Vec<f64> {
    v["choices"][0]["logprobs"]["token_logprobs"]
        .as_array()
        .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
        .unwrap_or_default()
}

fn req_body(model: &str, lora: Option<&str>) -> serde_json::Value {
    // A FIM request (prompt + suffix) so the FIM markers tokenize to a real
    // prompt and the model actually generates (see `fixtures`). `logprobs`
    // surfaces the per-token distribution we compare.
    let mut b = serde_json::json!({
        "model": model,
        "prompt": "<|tok_3|>",
        "suffix": "<|tok_7|>",
        "max_tokens": 4,
        "temperature": 0.0,
        "logprobs": 1,
        "stream": false,
    });
    if let Some(l) = lora {
        b["lora"] = serde_json::Value::String(l.to_string());
    }
    b
}

#[tokio::test]
async fn lora_explicit_field_selects_adapter_and_changes_output() {
    let (base, adapter) = fixtures("explicit");
    let state = build_state(&base, Some(&adapter));

    // Base (no selection).
    let (s0, base_v) = post_json(
        router(state.clone()),
        "/v1/completions",
        req_body("lora-test", None),
    )
    .await;
    assert_eq!(s0, StatusCode::OK, "base: {base_v}");
    let base_lp = logprobs_of(&base_v);
    assert!(!base_lp.is_empty(), "base produced logprobs: {base_v}");

    // Adapter selected via the explicit `lora` field.
    let (s1, lora_v) = post_json(
        router(state.clone()),
        "/v1/completions",
        req_body("lora-test", Some("tune-a")),
    )
    .await;
    assert_eq!(s1, StatusCode::OK, "lora: {lora_v}");
    assert_ne!(
        base_lp,
        logprobs_of(&lora_v),
        "selecting a resident adapter must change the output distribution"
    );

    // Base is unchanged on a repeat (no leaked selection).
    let (_, base_again) = post_json(
        router(state.clone()),
        "/v1/completions",
        req_body("lora-test", None),
    )
    .await;
    assert_eq!(
        base_lp,
        logprobs_of(&base_again),
        "base must stay unchanged"
    );

    // Unknown adapter name ⇒ 404 (never a silent base serve).
    let (s2, err_v) = post_json(
        router(state.clone()),
        "/v1/completions",
        req_body("lora-test", Some("ghost")),
    )
    .await;
    assert_eq!(
        s2,
        StatusCode::NOT_FOUND,
        "unknown adapter must 404: {err_v}"
    );

    for p in [&base, &adapter] {
        let _ = std::fs::remove_file(p);
    }
}

#[tokio::test]
async fn lora_model_suffix_convention_selects_adapter() {
    let (base, adapter) = fixtures("suffix");
    let state = build_state(&base, Some(&adapter));

    let base_lp = {
        let (s, v) = post_json(
            router(state.clone()),
            "/v1/completions",
            req_body("lora-test", None),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        logprobs_of(&v)
    };

    // `model: "<base>:<adapter>"` convention.
    let (s, v) = post_json(
        router(state.clone()),
        "/v1/completions",
        req_body("lora-test:tune-a", None),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "suffix convention: {v}");
    assert_ne!(
        base_lp,
        logprobs_of(&v),
        "`model: base:adapter` must select the adapter"
    );

    // `base:unknown` ⇒ 404 (base resolves, adapter does not).
    let (s_bad, _) = post_json(
        router(state.clone()),
        "/v1/completions",
        req_body("lora-test:ghost", None),
    )
    .await;
    assert_eq!(
        s_bad,
        StatusCode::NOT_FOUND,
        "a known base with an unknown adapter must 404"
    );

    for p in [&base, &adapter] {
        let _ = std::fs::remove_file(p);
    }
}

#[tokio::test]
async fn lora_load_and_list_endpoints() {
    let (base, adapter) = fixtures("endpoints");
    // No adapter pre-attached — attach it over HTTP.
    let state = build_state(&base, None);

    // Empty to start.
    let (s, v) = get_json(router(state.clone()), "/v1/lora").await;
    assert_eq!(s, StatusCode::OK, "list: {v}");
    assert_eq!(v["data"].as_array().map(|a| a.len()), Some(0));

    // Load it.
    let (s, v) = post_json(
        router(state.clone()),
        "/v1/lora/load",
        serde_json::json!({ "name": "tune-a", "path": adapter.to_string_lossy() }),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "load: {v}");
    assert_eq!(v["name"], "tune-a");
    assert!(v["adapted_tensors"].as_u64().unwrap_or(0) > 0);

    // Now it lists.
    let (s, v) = get_json(router(state.clone()), "/v1/lora").await;
    assert_eq!(s, StatusCode::OK);
    let names: Vec<String> = v["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_str().unwrap().to_string())
        .collect();
    assert_eq!(names, vec!["tune-a".to_string()]);

    // And a request can now select it (distribution differs from base).
    let (_, base_v) = post_json(
        router(state.clone()),
        "/v1/completions",
        req_body("lora-test", None),
    )
    .await;
    let (s, lora_v) = post_json(
        router(state.clone()),
        "/v1/completions",
        req_body("lora-test", Some("tune-a")),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert_ne!(
        logprobs_of(&base_v),
        logprobs_of(&lora_v),
        "an HTTP-loaded adapter must apply when selected"
    );

    for p in [&base, &adapter] {
        let _ = std::fs::remove_file(p);
    }
}
