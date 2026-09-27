//! Integration test for `measure_batch_size_candidates` — the
//! dynamic half of `rustllama tune --batch-size`. Drives a synth
//! GGUF with a long-ish synthetic prompt through several batch
//! sizes, verifies a winner is picked by prefill tok/s.
//!
//! The candidates are small (8/16/32) so the synth model's
//! prefill phase produces measurable timings without ballooning
//! the test runtime — the production defaults are 128/256/512/
//! 1024/2048 but those would all process the synth's tiny prompt
//! in one chunk and the comparison would be meaningless.

use rustllama_cli::measure_batch_size_candidates;
use rustllama_config::Config;
use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};

#[test]
fn measure_batch_size_returns_a_winner_for_synth_model() {
    let tmp = std::env::temp_dir().join("rustllama-batch-size-winner.gguf");
    let synth = SynthLlama {
        n_layers: 2,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 32,
        d_model: 64,
        d_ff: 128,
        vocab: 32,
        // Big enough ctx that even the largest candidate's prompt
        // fits with room for the decode.
        ctx: 128,
        ..SynthLlama::default()
    };
    write_synthetic_llama_gguf(&tmp, &synth);

    // Three candidates that meaningfully partition a 64-token prompt:
    //   - 16 → 4 chunks
    //   - 32 → 2 chunks
    //   - 64 → 1 chunk
    let candidates = vec![16usize, 32, 64];

    let cfg = Config::default();
    let winner = measure_batch_size_candidates(
        &tmp,
        &candidates,
        64, // prompt_tokens — fits well within synth's 128-ctx
        2,  // repeats — minimum for median
        &cfg,
    )
    .expect("measurement loop must not error on synth model");

    let b = winner.expect("at least one candidate should succeed");
    assert!(
        candidates.contains(&b),
        "winner batch_size={b} must be one of {candidates:?}"
    );

    let _ = std::fs::remove_file(&tmp);
}

/// Single-candidate sweep returns that candidate (or None if it
/// failed). Guards against an off-by-one in the comparison logic.
#[test]
fn measure_batch_size_single_candidate_returns_it() {
    let tmp = std::env::temp_dir().join("rustllama-batch-size-single.gguf");
    write_synthetic_llama_gguf(&tmp, &SynthLlama::default());

    let cfg = Config::default();
    let winner = measure_batch_size_candidates(&tmp, &[32usize], 16, 1, &cfg)
        .expect("single candidate must not error");
    assert_eq!(winner, Some(32), "the only candidate is the winner");

    let _ = std::fs::remove_file(&tmp);
}
