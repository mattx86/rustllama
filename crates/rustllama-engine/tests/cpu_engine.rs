//! Integration test for `CpuEngine` against a synthetic Llama GGUF.
//!
//! Verifies that the prefill + decode + sampler path produces:
//!   - deterministic greedy continuations
//!   - identical results across two equivalent invocations
//!   - sane in-bounds token ids
//! Phase 1 acceptance for the engine layer.

use rustllama_engine::{BeamParams, CpuEngine, SamplingParams};
use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};

fn build_engine(tag: &str) -> (CpuEngine, std::path::PathBuf) {
    let tmp = std::env::temp_dir().join(format!("rustllama-cpu-engine-{tag}.gguf"));
    let params = SynthLlama::default();
    write_synthetic_llama_gguf(&tmp, &params);
    let engine = CpuEngine::load(&tmp, 16).expect("load engine");
    (engine, tmp)
}

#[test]
fn greedy_decode_is_deterministic_and_in_vocab() {
    let (engine, tmp) = build_engine("greedy");
    let vocab = engine.vocab_size() as u32;

    let greedy_params = SamplingParams {
        temperature: 0.0,
        top_p: 0.0,
        top_k: 0,
        repeat_penalty: 1.0,
        max_tokens: 4,
        stop: vec![],
        seed: 0,
        ..SamplingParams::default()
    };

    let prompt = [3i32, 5, 7];
    let out1 = engine
        .generate_token_ids(&prompt, 4, &greedy_params)
        .expect("gen 1");
    let out2 = engine
        .generate_token_ids(&prompt, 4, &greedy_params)
        .expect("gen 2");

    assert_eq!(out1.len(), 4);
    assert_eq!(out1, out2, "greedy must be deterministic across runs");
    for t in &out1 {
        assert!(*t < vocab, "token {t} out of vocab {vocab}");
    }

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn seeded_sampling_is_reproducible() {
    let (engine, tmp) = build_engine("seeded");

    let sample_params = SamplingParams {
        temperature: 0.8,
        top_p: 0.9,
        top_k: 20,
        repeat_penalty: 1.1,
        max_tokens: 6,
        stop: vec![],
        seed: 0xDEAD_BEEF,
        ..SamplingParams::default()
    };

    let prompt = [3i32, 5, 7];
    let a = engine
        .generate_token_ids(&prompt, 6, &sample_params)
        .expect("a");
    let b = engine
        .generate_token_ids(&prompt, 6, &sample_params)
        .expect("b");

    assert_eq!(a, b, "same seed must yield same tokens");
    assert!(!a.is_empty() && a.len() <= 6);

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn prompt_too_long_is_rejected() {
    let (engine, tmp) = build_engine("toolong");
    let max = engine.max_ctx();
    let prompt: Vec<i32> = (0..max as i32).collect();

    let p = SamplingParams::default();
    let err = engine.generate_token_ids(&prompt, 4, &p);
    assert!(matches!(
        err,
        Err(rustllama_engine::CpuEngineError::PromptTooLong { .. })
    ));

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn prefix_cache_pool_preserves_outputs_across_alternating_threads() {
    // Reproduce a multi-conversation scenario:
    //   - Thread A: generate from prompt_a (cache is now warmed on A).
    //   - Thread B: generate from prompt_b (different prefix, evicts A
    //     from the live state but the pool should retain A's snapshot).
    //   - Back to thread A: generate from prompt_a again. With the
    //     multi-snapshot pool, the engine should restore A's saved
    //     snapshot rather than re-prefill A from scratch.
    //
    // Determinism is the load-bearing check: regardless of whether the
    // engine reused the snapshot or re-prefilled, the greedy output
    // must match a freshly-restarted engine's output for the same prompt.
    let (mut engine, tmp) = build_engine("prefix-pool");
    engine.set_prefix_cache_max_snapshots(4);

    let greedy = SamplingParams {
        temperature: 0.0,
        top_p: 0.0,
        top_k: 0,
        repeat_penalty: 1.0,
        max_tokens: 4,
        stop: vec![],
        seed: 0,
        ..SamplingParams::default()
    };

    // Two prompts long enough to clear the PREFIX_REUSE_MIN_TOKENS=8 gate
    // and with no shared prefix (first token differs).
    let prompt_a = [3i32, 5, 7, 9, 11, 13, 15, 17, 19];
    let prompt_b = [4i32, 6, 8, 10, 12, 14, 16, 18, 20];

    // Reference output for A from a clean engine (cold-start, no cache).
    let (mut cold, tmp2) = build_engine("prefix-pool-cold");
    cold.set_prefix_cache_max_snapshots(0);
    let cold_a = cold
        .generate_token_ids(&prompt_a, 4, &greedy)
        .expect("cold A");
    let _ = std::fs::remove_file(&tmp2);

    // Warm thread A.
    let warm_a = engine
        .generate_token_ids(&prompt_a, 4, &greedy)
        .expect("warm A");
    assert_eq!(warm_a, cold_a, "first generation must match cold-start");
    assert!(
        engine.prefix_cache_snapshot_count() >= 1,
        "A should be in the pool after gen"
    );

    // Run B; this should also land its own snapshot in the pool.
    let _warm_b = engine
        .generate_token_ids(&prompt_b, 4, &greedy)
        .expect("warm B");
    assert!(
        engine.prefix_cache_snapshot_count() >= 2,
        "pool should now hold both A and B snapshots"
    );

    // Now re-run A. With a single-snapshot cache this would have LCP=0
    // against B's tokens and re-prefill from scratch; with the pool the
    // engine restores A's snapshot. Either way the output must match
    // the cold-start reference exactly.
    let warm_a_again = engine
        .generate_token_ids(&prompt_a, 4, &greedy)
        .expect("warm A again");
    assert_eq!(warm_a_again, cold_a, "pool-restored gen must match cold");

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn prefix_cache_disabled_pool_zero_capacity() {
    // With prefix_cache_max_snapshots=0 the pool should never grow, even
    // though `set_prefix_cache(true)` is left enabled — the single-snapshot
    // LCP via `last_ids` is what survives.
    let (mut engine, tmp) = build_engine("prefix-pool-off");
    engine.set_prefix_cache_max_snapshots(0);

    let greedy = SamplingParams {
        temperature: 0.0,
        top_p: 0.0,
        top_k: 0,
        repeat_penalty: 1.0,
        max_tokens: 4,
        stop: vec![],
        seed: 0,
        ..SamplingParams::default()
    };
    let prompt = [3i32, 5, 7, 9, 11, 13, 15, 17, 19];
    let _ = engine.generate_token_ids(&prompt, 4, &greedy).expect("gen");
    assert_eq!(
        engine.prefix_cache_snapshot_count(),
        0,
        "zero-capacity pool must stay empty"
    );

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn last_request_stats_populated_after_generate() {
    // After a generation, the engine must publish stats reflecting the
    // request: at least one decode forward, and the prompt should have
    // turned into either prefill forwards or cache hits (sum == prompt
    // length minus the held-back last token).
    let (engine, tmp) = build_engine("stats");
    let greedy = SamplingParams {
        temperature: 0.0,
        top_p: 0.0,
        top_k: 0,
        repeat_penalty: 1.0,
        max_tokens: 3,
        stop: vec![],
        seed: 0,
        ..SamplingParams::default()
    };
    let prompt = [3i32, 5, 7, 9, 11, 13, 15, 17, 19];
    let _ = engine.generate_token_ids(&prompt, 3, &greedy).expect("gen");

    let s = engine.last_request_stats();
    assert!(
        s.tokens_generated > 0,
        "must record at least 1 decode forward"
    );
    assert!(s.decode_ms >= 0.0);
    assert!(s.prefill_ms >= 0.0);
    // First run: cache empty → all prompt tokens (except the held-back
    // last) become prefill forwards, no cache hits.
    assert_eq!(s.cache_hit_tokens, 0);
    assert_eq!(s.tokens_prefilled, (prompt.len() - 1) as u32);

    // Second run with the same prompt should hit the prefix cache.
    let _ = engine
        .generate_token_ids(&prompt, 3, &greedy)
        .expect("gen 2");
    let s2 = engine.last_request_stats();
    assert!(
        s2.cache_hit_tokens > 0,
        "second identical run must hit cache, got {s2:?}"
    );

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn clear_prefix_cache_wipes_pool_and_live_state() {
    let (mut engine, tmp) = build_engine("prefix-pool-clear");
    engine.set_prefix_cache_max_snapshots(4);

    let greedy = SamplingParams {
        temperature: 0.0,
        top_p: 0.0,
        top_k: 0,
        repeat_penalty: 1.0,
        max_tokens: 4,
        stop: vec![],
        seed: 0,
        ..SamplingParams::default()
    };
    let prompt = [3i32, 5, 7, 9, 11, 13, 15, 17, 19];
    let _ = engine.generate_token_ids(&prompt, 4, &greedy).expect("gen");
    assert!(engine.prefix_cache_snapshot_count() >= 1);

    engine.clear_prefix_cache();
    assert_eq!(engine.prefix_cache_snapshot_count(), 0);

    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn fork_for_concurrent_use_has_independent_state() {
    // The fork primitive must produce an engine that:
    //   1. Reproduces the parent's greedy output bit-for-bit (so we
    //      know the model weights + tokenizer are correctly shared
    //      via Arc and haven't been corrupted by the fork).
    //   2. Has independent generation state — calling generate on the
    //      fork does NOT change the parent's prefix cache snapshot
    //      count.
    // This pins the multi-flight-serving invariant: concurrent slots
    // can each run on their own fork without colliding.
    let (parent, tmp) = build_engine("fork");

    let greedy = SamplingParams {
        temperature: 0.0,
        top_p: 0.0,
        top_k: 0,
        repeat_penalty: 1.0,
        max_tokens: 4,
        stop: vec![],
        seed: 0,
        ..SamplingParams::default()
    };
    let prompt = [3i32, 5, 7, 9];

    // Reference output from the parent.
    parent.clear_prefix_cache();
    let parent_out = parent
        .generate_token_ids(&prompt, 4, &greedy)
        .expect("parent gen");
    let parent_snap_count_after = parent.prefix_cache_snapshot_count();

    // Fresh fork sees no shared state from the parent's generation.
    let fork = parent.fork_for_concurrent_use();
    assert_eq!(
        fork.prefix_cache_snapshot_count(),
        0,
        "fresh fork must start with an empty prefix-cache pool — \
         cross-fork pool sharing isn't enabled in v1",
    );

    // Fork reproduces parent's greedy output (same model, same seed,
    // same prompt → same tokens).
    let fork_out = fork
        .generate_token_ids(&prompt, 4, &greedy)
        .expect("fork gen");
    assert_eq!(
        fork_out, parent_out,
        "fork must reproduce parent greedy output — \
         model weights are Arc-shared"
    );

    // After fork generates, parent's prefix-cache snapshot count is
    // unchanged. This proves the state Arc isn't shared.
    assert_eq!(
        parent.prefix_cache_snapshot_count(),
        parent_snap_count_after,
        "fork generation must not bump parent's prefix-cache pool",
    );

    let _ = std::fs::remove_file(&tmp);
}

// ---- P12b: beam search -------------------------------------------------

/// Anchor: `num_beams == 1` is byte-identical to the greedy argmax decode.
/// This proves the beam machinery (candidate build, selection, KV
/// snapshot/restore fork, finalization) doesn't perturb the degenerate case.
#[test]
fn beam_one_equals_greedy_byte_identical() {
    let (engine, tmp) = build_engine("beam-greedy");
    let greedy_params = SamplingParams {
        temperature: 0.0,
        top_p: 0.0,
        top_k: 0,
        repeat_penalty: 1.0,
        max_tokens: 6,
        stop: vec![],
        seed: 0,
        ..SamplingParams::default()
    };
    let prompt = [3i32, 5, 7];
    let greedy = engine
        .generate_token_ids(&prompt, 6, &greedy_params)
        .expect("greedy decode");

    let bp = BeamParams {
        num_beams: 1,
        length_penalty: 1.0,
        early_stopping: false,
        num_return: 1,
    };
    let beams = engine
        .generate_beam_token_ids(&prompt, 6, &bp)
        .expect("beam decode");
    assert_eq!(beams.len(), 1, "num_beams=1 returns exactly one hypothesis");
    assert_eq!(
        beams[0].tokens, greedy,
        "num_beams=1 beam search must equal greedy decode token-for-token"
    );

    // Same under early_stopping=true (should not change the degenerate case).
    let bp_es = BeamParams {
        early_stopping: true,
        ..bp
    };
    let beams_es = engine
        .generate_beam_token_ids(&prompt, 6, &bp_es)
        .expect("beam decode es");
    assert_eq!(
        beams_es[0].tokens, greedy,
        "num_beams=1 with early_stopping must still equal greedy"
    );

    let _ = std::fs::remove_file(&tmp);
}

/// Anchor: with `num_beams == 2` the returned beams are distinct sequences,
/// ordered by the length-penalty-adjusted cumulative log-prob (best first).
#[test]
fn two_beams_distinct_and_ranked() {
    let (engine, tmp) = build_engine("beam-two");
    let prompt = [3i32, 5, 7];
    let bp = BeamParams {
        num_beams: 2,
        length_penalty: 1.0,
        early_stopping: false,
        num_return: 2,
    };
    let beams = engine
        .generate_beam_token_ids(&prompt, 5, &bp)
        .expect("beam decode");
    assert_eq!(beams.len(), 2, "num_return=2 returns two hypotheses");
    assert_ne!(
        beams[0].tokens, beams[1].tokens,
        "beam search must return distinct sequences"
    );
    assert!(
        beams[0].score >= beams[1].score,
        "returned beams must be ordered by adjusted score (desc): {} vs {}",
        beams[0].score,
        beams[1].score
    );

    // Deterministic: same prompt → same beams.
    let beams2 = engine
        .generate_beam_token_ids(&prompt, 5, &bp)
        .expect("beam decode 2");
    assert_eq!(beams.len(), beams2.len());
    for (a, b) in beams.iter().zip(beams2.iter()) {
        assert_eq!(a.tokens, b.tokens, "beam search must be deterministic");
    }

    let _ = std::fs::remove_file(&tmp);
}

/// Beam search runs with larger `num_beams` without panicking and still
/// returns distinct, correctly-ranked hypotheses (a 4-beam smoke test on the
/// KV fork/restore machinery).
#[test]
fn four_beams_run_and_rank() {
    let (engine, tmp) = build_engine("beam-four");
    let prompt = [3i32, 5, 7];
    let bp = BeamParams {
        num_beams: 4,
        length_penalty: 1.0,
        early_stopping: false,
        num_return: 3,
    };
    let beams = engine
        .generate_beam_token_ids(&prompt, 6, &bp)
        .expect("4-beam");
    assert_eq!(beams.len(), 3, "num_return=3 returns three hypotheses");
    // Distinct + ranked by adjusted score (descending).
    for i in 1..beams.len() {
        assert!(
            beams[i - 1].score >= beams[i].score,
            "beams must be ranked by adjusted score"
        );
        assert_ne!(
            beams[i - 1].tokens,
            beams[i].tokens,
            "returned beams must be distinct"
        );
    }

    let _ = std::fs::remove_file(&tmp);
}

/// P15 (echo + logprobs): `prompt_token_logprobs` returns one entry per prompt
/// token — the FIRST is `None` (no preceding context, OpenAI's `null`), and
/// every later entry carries the real `logP(prompt[i] | prompt[0..i])` plus the
/// top-k alternatives. Token-id based (not text) so it is deterministic on the
/// synth model regardless of the tokenizer, and it exercises the exact method
/// the server's `echo:true + logprobs:N` path prepends to the completion.
#[test]
fn prompt_token_logprobs_are_null_first_then_real() {
    let (engine, tmp) = build_engine("prompt-logprobs");
    let vocab = engine.vocab_size() as u32;

    let prompt = [3i32, 5, 7, 9];
    let lps = engine
        .prompt_token_logprobs(&prompt, 3)
        .expect("prompt logprobs");

    // One entry per prompt token.
    assert_eq!(lps.len(), prompt.len(), "one logprob slot per prompt token");
    // First token has no conditioning context → null.
    assert!(
        lps[0].is_none(),
        "first prompt token's logprob must be None (no context)"
    );
    // Every subsequent token carries a finite conditional logprob (<= 0 in log
    // space) and its top-k alternatives are in-vocab and descending.
    for (i, slot) in lps.iter().enumerate().skip(1) {
        let lp = slot
            .as_ref()
            .unwrap_or_else(|| panic!("prompt token {i} must have a logprob"));
        assert!(
            lp.logprob.is_finite() && lp.logprob <= 1e-4,
            "token {i} logprob {} must be a finite log-prob",
            lp.logprob
        );
        assert!(lp.top.len() <= 3, "top-k respects k=3");
        for alt in &lp.top {
            assert!(alt.id < vocab, "alt id {} out of vocab {vocab}", alt.id);
        }
        for w in lp.top.windows(2) {
            assert!(
                w[0].logprob >= w[1].logprob,
                "top-k alternatives must be descending"
            );
        }
    }

    // Empty prompt → empty result (defensive).
    assert!(engine
        .prompt_token_logprobs(&[], 1)
        .expect("empty ok")
        .is_empty());

    let _ = std::fs::remove_file(&tmp);
}
