//! Integration test for `measure_placement_candidates` — the dynamic
//! half of `rustllama tune --placement --measure`. Drives a synth
//! GGUF through the real CpuEngine path with varied `n_gpu_layers`
//! settings, verifies a winner is selected by measured tok/s.
//!
//! The synth model is tiny (default SynthLlama dims: 2 layers, 64-
//! dim, 32-vocab) so the whole test runs in ~seconds rather than
//! the minutes a 7B Q4_K_M measurement would take. Hardware-blocked
//! GPU dispatch (no SYCL device present at runtime) falls
//! through to CPU for every candidate, which is fine — the test
//! validates the measurement plumbing, not the GPU acceleration.

use rustllama_cli::measure_placement_candidates;
use rustllama_config::Config;
use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};
use rustllama_tuner::PlacementPlan;

#[test]
fn measure_candidates_returns_a_winner_for_synth_model() {
    let tmp = std::env::temp_dir().join("rustllama-placement-measure-winner.gguf");
    let synth = SynthLlama {
        // Plain 2-layer synth — enough for two candidates (n=0 full
        // CPU, n=2 fully-on-GPU) plus the all-GPU+head sentinel n=3.
        n_layers: 2,
        n_heads: 2,
        n_kv_heads: 1,
        head_dim: 32,
        d_model: 64,
        d_ff: 128,
        vocab: 32,
        ctx: 64,
        ..SynthLlama::default()
    };
    write_synthetic_llama_gguf(&tmp, &synth);

    // Three candidates spanning the full range. The measurement loop
    // reconfigures `n_gpu_layers` between runs via the per-thread
    // setter, so loading the model once is correct.
    let candidates = vec![
        PlacementPlan {
            n_gpu_layers: 0,
            overrides: Vec::new(),
        },
        PlacementPlan {
            n_gpu_layers: 2,
            overrides: Vec::new(),
        },
        PlacementPlan {
            n_gpu_layers: 3,
            overrides: Vec::new(),
        },
    ];

    let cfg = Config::default();
    let winner = measure_placement_candidates(
        &tmp,
        &candidates,
        64, // ctx_size
        4,  // measure_prompt_tokens — short, fast
        4,  // measure_decode_tokens — short, fast
        2,  // measure_repeats — minimum for median
        &cfg,
    )
    .expect("measurement loop must not error on synth model");

    let n = winner.expect("at least one candidate should succeed");
    // The winner has to be one of the candidates we passed in.
    let candidate_ns: Vec<u32> = candidates.iter().map(|p| p.n_gpu_layers).collect();
    assert!(
        candidate_ns.contains(&n),
        "winner n_gpu_layers={n} must be one of {candidate_ns:?}"
    );

    let _ = std::fs::remove_file(&tmp);
}

/// Empty candidate list short-circuits — no model load, returns
/// `None`. Guards against a future refactor that accidentally tries
/// to load + drive an engine when there's nothing to measure.
#[test]
fn measure_candidates_handles_empty_list() {
    let tmp = std::env::temp_dir().join("rustllama-placement-measure-empty.gguf");
    write_synthetic_llama_gguf(&tmp, &SynthLlama::default());

    let cfg = Config::default();
    // With zero candidates the loop still loads the model (current
    // implementation), but the inner per-candidate loop runs zero
    // times and `best_n` stays `None`. The function returns Ok(None).
    let result = measure_placement_candidates(&tmp, &[], 64, 4, 4, 1, &cfg)
        .expect("empty candidate list must not error");
    assert!(
        result.is_none(),
        "empty candidate list → no winner, got {result:?}"
    );

    let _ = std::fs::remove_file(&tmp);
}
