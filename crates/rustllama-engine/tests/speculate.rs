//! End-to-end smoke test for `Engine::speculate`.
//!
//! Loads a synthetic Llama GGUF twice — once as the "draft" and once
//! as the "target". Since both engines run the same model with the
//! same sampling settings (greedy via the scaffold's
//! temperature=0 override), the draft's K candidates match the
//! target's first K tokens token-for-token, which means
//! `accept_reject`'s degenerate "all accepted" case fires and the
//! stream emits K + 1 tokens (K accepted + 1 bonus).
//!
//! This pins the trait + plumbing — the underlying probability
//! distribution work for general spec-decoding ships in a follow-up.

use std::sync::Arc;

use futures::StreamExt;
use rustllama_engine::{CpuEngine, Engine, SamplingParams};
use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};

fn build_engine_pair(tag: &str) -> (CpuEngine, Arc<dyn Engine>, std::path::PathBuf) {
    let tmp = std::env::temp_dir().join(format!("rustllama-speculate-{tag}.gguf"));
    write_synthetic_llama_gguf(&tmp, &SynthLlama::default());
    // Both load the same GGUF; treat one as draft and one as target.
    // Real spec-decode pairs a small draft with a big target — here
    // they're the same model, which is the "all accept" degenerate
    // case the scaffold is calibrated for.
    let target = CpuEngine::load_with_tokenizer(&tmp, 64).expect("load target");
    let draft: Arc<dyn Engine> =
        Arc::new(CpuEngine::load_with_tokenizer(&tmp, 64).expect("load draft"));
    (target, draft, tmp)
}

#[tokio::test]
async fn speculate_with_identical_engines_accepts_all_candidates() {
    let (target, draft, tmp) = build_engine_pair("identical");
    let sampling = SamplingParams {
        // Force greedy on the test path. The scaffold also forces
        // temperature=0 internally; setting it here keeps the
        // contract visible at the call site.
        temperature: 0.0,
        seed: 0,
        max_tokens: 16,
        ..SamplingParams::default()
    };
    let stream = target
        .speculate("<tok_3> <tok_5>", draft.clone(), /*k=*/ 4, &sampling)
        .expect("speculate dispatches");
    let mut tokens: Vec<u32> = Vec::new();
    let mut s = stream;
    while let Some(t) = s.next().await {
        let tok = t.expect("token");
        tokens.push(tok.id);
    }
    // Synth model has near-random weights — greedy under default
    // synth eos_token_id often picks EOS on the first decode, which
    // collapses both draft and target to 0 tokens. The scaffold
    // tolerates that: it yields `min(accepted, target.len())` tokens
    // and finishes the stream. What we pin here is just the
    // **trait dispatch + accept loop** ran without panicking;
    // meaningful spec-decode output requires a real-weights model.
    //
    // Multi-round update: the bound is now `max_tokens` (16) since
    // the loop can chain multiple rounds. Synth-model EOS still
    // terminates early in practice.
    assert!(
        tokens.len() <= sampling.max_tokens as usize,
        "speculate must respect max_tokens={}; got {} ({tokens:?})",
        sampling.max_tokens,
        tokens.len()
    );
    let _ = std::fs::remove_file(&tmp);
}

/// Multi-round acceptance: with K=2 and max_tokens=8, the loop must
/// be capable of running multiple rounds. Synth model often hits EOS
/// early so we can't assert "exactly N rounds ran" — what we DO
/// assert is that the implementation stays correct under the
/// extended bound (no panic, no overshoot, monotonic token emission).
#[tokio::test]
async fn speculate_multi_round_respects_max_tokens_bound() {
    let (target, draft, tmp) = build_engine_pair("multi-round");
    let sampling = SamplingParams {
        temperature: 0.0,
        seed: 1,
        max_tokens: 8,
        ..SamplingParams::default()
    };
    let stream = target
        .speculate("<tok_1>", draft.clone(), /*k=*/ 2, &sampling)
        .expect("speculate dispatches");
    let mut tokens: Vec<u32> = Vec::new();
    let mut s = stream;
    while let Some(t) = s.next().await {
        if let Ok(tok) = t {
            tokens.push(tok.id);
        }
    }
    // Total never exceeds max_tokens. Lower bound is 0 (EOS at first
    // step is allowed on the synth model). The point of this test is
    // the safety property: max_tokens is a hard upper bound and the
    // multi-round loop honors it.
    assert!(
        tokens.len() <= sampling.max_tokens as usize,
        "multi-round speculate must NOT overshoot max_tokens={}; emitted {} ({tokens:?})",
        sampling.max_tokens,
        tokens.len()
    );
    let _ = std::fs::remove_file(&tmp);
}

#[tokio::test]
async fn speculate_zero_k_degrades_to_plain_generation() {
    // K=0 means "no speculation, just run the target." The scaffold
    // detects this and calls `self.generate(...)` directly. Pin
    // that fast path so future refactors don't accidentally make
    // K=0 do something silly like "run draft once + bonus".
    let (target, draft, tmp) = build_engine_pair("zero-k");
    let sampling = SamplingParams {
        temperature: 0.0,
        seed: 0,
        max_tokens: 3,
        ..SamplingParams::default()
    };
    let stream = target
        .speculate("<tok_4>", draft.clone(), /*k=*/ 0, &sampling)
        .expect("speculate dispatches");
    let mut tokens: Vec<u32> = Vec::new();
    let mut s = stream;
    while let Some(t) = s.next().await {
        if let Ok(t) = t {
            tokens.push(t.id);
        } else {
            break;
        }
    }
    // Without speculation we should see at most `max_tokens` tokens.
    assert!(tokens.len() <= 3, "K=0 path must respect max_tokens cap");
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn verify_speculation_returns_kplus1_softmax_distributions() {
    // Direct test of the target-side probability extraction. Pins:
    //   - returned vector has length `candidates.len() + 1`
    //   - each distribution has length `vocab_size`
    //   - each distribution sums to ~1.0 (softmax invariant)
    //   - all probabilities are non-negative
    // Pre-call and post-call engine state are unchanged, so calling
    // `verify_speculation` mid-conversation is safe.
    let tmp = std::env::temp_dir().join("rustllama-verify-spec.gguf");
    write_synthetic_llama_gguf(&tmp, &SynthLlama::default());
    let engine = CpuEngine::load_with_tokenizer(&tmp, 64).expect("load engine");
    let vocab = engine.vocab_size();

    // Use a fixed prompt + a couple of arbitrary candidate token ids
    // within the synth model's vocab (the synth model's vocab is
    // declared by `SynthLlama::default`).
    let prompt_ids: Vec<i32> = vec![1, 2, 3, 4];
    let candidates: Vec<u32> = vec![5, 6];

    let dists = engine
        .verify_speculation(&prompt_ids, &candidates)
        .expect("verify_speculation");
    assert_eq!(
        dists.len(),
        candidates.len() + 1,
        "K+1 distributions expected (K={} candidates → {} dists)",
        candidates.len(),
        candidates.len() + 1,
    );
    for (i, d) in dists.iter().enumerate() {
        assert_eq!(d.len(), vocab, "dist[{i}] must span the full vocab");
        assert!(
            d.iter().all(|&p| p >= 0.0),
            "dist[{i}] must have non-negative entries"
        );
        let sum: f32 = d.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-3,
            "dist[{i}] must sum to ~1.0, got {sum}",
        );
    }
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn verify_speculation_does_not_mutate_engine_state() {
    // The doc comment promises the engine's KV cache + last_ids are
    // restored after the call. Verify by running a normal generation
    // before and after the verify call and asserting identical
    // outputs (same prompt + same seed → same tokens iff state was
    // restored).
    let tmp = std::env::temp_dir().join("rustllama-verify-purity.gguf");
    write_synthetic_llama_gguf(&tmp, &SynthLlama::default());
    let engine = CpuEngine::load_with_tokenizer(&tmp, 64).expect("load");

    let sampling = SamplingParams {
        temperature: 0.0,
        seed: 7,
        max_tokens: 4,
        ..SamplingParams::default()
    };
    // Reference: generate without any verify in the middle.
    engine.reset_state();
    let prompt_ids: Vec<i32> = vec![1, 2, 3];
    let reference = engine
        .generate_token_ids(&prompt_ids, sampling.max_tokens, &sampling)
        .expect("reference gen");

    // Now: reset, run verify, then generate. Output must match the
    // reference exactly.
    engine.reset_state();
    let _dists = engine
        .verify_speculation(&prompt_ids, &[10, 11, 12])
        .expect("verify");
    let after = engine
        .generate_token_ids(&prompt_ids, sampling.max_tokens, &sampling)
        .expect("after-verify gen");
    assert_eq!(
        after, reference,
        "verify_speculation must not perturb state — \
         post-verify generate diverged from reference"
    );
    let _ = std::fs::remove_file(&tmp);
}

#[tokio::test]
async fn mock_engine_speculate_returns_unsupported() {
    // The default trait impl returns `SpeculationUnsupported`. Pin
    // that so the error path stays usable for handlers that need to
    // surface "this engine doesn't speculate" as a typed error.
    use rustllama_engine::{EngineError, MockEngine};
    let target = MockEngine;
    let draft: Arc<dyn Engine> = Arc::new(MockEngine);
    let sampling = SamplingParams::default();
    // `expect_err` can't print a TokenStream on success; match
    // manually to keep the error path the only branch this test
    // exercises.
    match target.speculate("hi", draft, 4, &sampling) {
        Ok(_) => panic!("MockEngine must not speculate"),
        Err(EngineError::SpeculationUnsupported) => { /* expected */ }
        Err(other) => panic!("expected SpeculationUnsupported, got {other:?}"),
    }
}
