//! Speculative-verification equivalence tests (ID level).
//!
//! At temperature 0 the accept/reject math is exact: a drafted token
//! is accepted iff it IS the target's argmax at that position. So the
//! argmax chain of `verify_speculation`'s K+1 batched distributions
//! must reproduce the serial greedy continuation token-for-token —
//! the drafter only changes *speed*, never output. (The synth
//! tokenizer cannot encode text — its vocab is literal `<tok_N>`
//! strings with no merges — so these tests drive the ID-level API;
//! the text-level stream equivalence is exercised on real models in
//! the release smoke.)

use rustllama_engine::{CpuEngine, KvDtype, SamplingParams};
use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};

fn greedy(n: u32) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_p: 0.0,
        top_k: 0,
        repeat_penalty: 1.0,
        max_tokens: n,
        stop: vec![],
        seed: 7,
        ..SamplingParams::default()
    }
}

fn argmax(dist: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in dist.iter().enumerate() {
        if v > dist[best] {
            best = i;
        }
    }
    best as u32
}

/// Serial greedy continuation, then verify the SAME continuation as
/// speculative candidates: every batched distribution's argmax must
/// equal the serially-decoded token at that position (the "all
/// accepted" property that makes temp-0 speculation lossless).
fn assert_verify_matches_serial(engine: &CpuEngine, prompt: &[i32], k: usize) {
    let serial = engine
        .generate_token_ids(prompt, (k + 1) as u32, &greedy((k + 1) as u32))
        .expect("serial greedy");
    assert!(
        serial.len() > k,
        "synth model stopped early ({} tokens, need {})",
        serial.len(),
        k + 1
    );

    let candidates: Vec<u32> = serial[..k].to_vec();
    let dists = engine
        .verify_speculation(prompt, &candidates)
        .expect("verify");
    assert_eq!(dists.len(), k + 1, "expected K+1 distributions");
    for (i, dist) in dists.iter().enumerate() {
        assert_eq!(
            argmax(dist),
            serial[i],
            "position {i}: batched verify argmax diverged from serial greedy"
        );
    }
}

#[test]
fn batched_verify_matches_serial_greedy_f32() {
    let tmp = std::env::temp_dir().join("rustllama-spec-verify-f32.gguf");
    write_synthetic_llama_gguf(&tmp, &SynthLlama::default());
    let engine = CpuEngine::load_with_options(&tmp, 64, true, KvDtype::F32).expect("load");
    assert_verify_matches_serial(&engine, &[5, 9, 7, 5, 9, 7, 5], 4);
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn batched_verify_matches_serial_greedy_q4_0() {
    // Quantized KV + active whitening gate (head_dim 64) — the shape
    // the real Bonsai speculation deployment runs with.
    let tmp = std::env::temp_dir().join("rustllama-spec-verify-q40.gguf");
    let mut synth = SynthLlama::default();
    synth.head_dim = 64;
    synth.d_model = synth.n_heads * 64;
    write_synthetic_llama_gguf(&tmp, &synth);
    let engine = CpuEngine::load_with_options(&tmp, 64, true, KvDtype::Q4_0).expect("load");
    assert_verify_matches_serial(&engine, &[4, 11, 4, 11, 4, 11, 4], 4);
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn verify_is_state_neutral() {
    // Two verifies with different candidates must not disturb each
    // other — verification is a pure read (the property the hybrid
    // DeltaNet snapshot/restore extends to recurrent state).
    let tmp = std::env::temp_dir().join("rustllama-spec-verify-neutral.gguf");
    write_synthetic_llama_gguf(&tmp, &SynthLlama::default());
    let engine = CpuEngine::load_with_options(&tmp, 64, true, KvDtype::F32).expect("load");
    let prompt = [5i32, 9, 7, 5, 9];
    let a1 = engine.verify_speculation(&prompt, &[1, 2, 3]).expect("a1");
    let _ = engine.verify_speculation(&prompt, &[30, 29]).expect("mid");
    let a2 = engine.verify_speculation(&prompt, &[1, 2, 3]).expect("a2");
    assert_eq!(a1, a2, "verify must be state-neutral / repeatable");
    let _ = std::fs::remove_file(&tmp);
}

/// End-to-end stream equivalence at temperature 0, ID level: the
/// n-gram speculative stream must produce EXACTLY the plain greedy
/// continuation (the greedy accept gate makes acceptance
/// deterministic), while committing state incrementally between
/// rounds (the fix for the quadratic re-prefill "spec hang": the old
/// pure-read verify restored everything, so every round re-prefilled
/// prompt+committed from a stale LCP). The repetitive prompt makes
/// the drafter fire so both the accept and reject commit arms run.
async fn assert_ngram_stream_matches_plain(engine: &CpuEngine, prompt: &[i32], n: u32) {
    use futures::StreamExt;
    use rustllama_engine::speculative::NgramDrafterConfig;

    let cfg = NgramDrafterConfig {
        n_match: 2,
        n_draft: 3,
        ..Default::default()
    };
    let mut stream = engine
        .speculate_ngram_stream_from_ids(prompt.to_vec(), cfg, greedy(n))
        .expect("spec stream");
    let mut spec_ids: Vec<u32> = Vec::new();
    while let Some(item) = stream.next().await {
        spec_ids.push(item.expect("stream token").id);
    }
    assert!(
        !spec_ids.is_empty(),
        "speculative stream produced no tokens"
    );

    let plain = engine
        .generate_token_ids(prompt, n, &greedy(n))
        .expect("plain greedy");
    let common = spec_ids.len().min(plain.len());
    assert_eq!(
        &spec_ids[..common],
        &plain[..common],
        "speculative greedy must equal plain greedy token-for-token"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ngram_stream_from_ids_matches_plain_greedy_f32() {
    let tmp = std::env::temp_dir().join("rustllama-spec-stream-f32.gguf");
    write_synthetic_llama_gguf(&tmp, &SynthLlama::default());
    let engine = CpuEngine::load_with_options(&tmp, 64, true, KvDtype::F32).expect("load");
    assert_ngram_stream_matches_plain(&engine, &[5, 9, 7, 5, 9, 7, 5], 12).await;
    let _ = std::fs::remove_file(&tmp);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ngram_stream_from_ids_matches_plain_greedy_q4_0() {
    let tmp = std::env::temp_dir().join("rustllama-spec-stream-q40.gguf");
    let mut synth = SynthLlama::default();
    synth.head_dim = 64;
    synth.d_model = synth.n_heads * 64;
    write_synthetic_llama_gguf(&tmp, &synth);
    let engine = CpuEngine::load_with_options(&tmp, 64, true, KvDtype::Q4_0).expect("load");
    assert_ngram_stream_matches_plain(&engine, &[4, 11, 4, 11, 4, 11, 4], 12).await;
    let _ = std::fs::remove_file(&tmp);
}
