//! Direct integration tests for `rustllama_engine::measurement`.
//! Drives the structured-return API end-to-end against the synth
//! Llama fixture, validating that the report shape is what server +
//! CLI consumers will see.
//!
//! The CLI-side tests (`crates/rustllama-cli/tests/placement_measurement.rs`,
//! `batch_size_measurement.rs`) exercise the same logic through
//! cli wrappers that delegate here — this file pins the engine's
//! pub API contract independently of CLI formatting.

use rustllama_engine::measurement::{
    measure_batch_size_candidates, measure_placement_candidates, MeasurementConfig,
};
use rustllama_engine::KvDtype;
use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};
use rustllama_tuner::PlacementPlan;

fn default_measurement_cfg() -> MeasurementConfig {
    MeasurementConfig {
        kv_dtype: KvDtype::F32,
        kv_cache_layout: "contiguous".into(),
        flash_attention: false,
        n_gpu_layers: 999,
    }
}

#[test]
fn placement_report_has_one_candidate_entry_per_input() {
    let tmp = std::env::temp_dir().join("rustllama-engine-meas-placement-shape.gguf");
    write_synthetic_llama_gguf(
        &tmp,
        &SynthLlama {
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 32,
            d_model: 64,
            d_ff: 128,
            vocab: 32,
            ctx: 64,
            ..SynthLlama::default()
        },
    );
    let candidates = vec![
        PlacementPlan { n_gpu_layers: 0, overrides: Vec::new() },
        PlacementPlan { n_gpu_layers: 2, overrides: Vec::new() },
    ];
    let report = measure_placement_candidates(
        &tmp, &candidates, 64, 4, 4, 2, &default_measurement_cfg(),
    )
    .expect("measurement must not error on synth model");

    assert_eq!(
        report.candidates.len(),
        candidates.len(),
        "one report entry per input candidate"
    );
    assert!(report.load_ms > 0.0, "load_ms must be populated");
    // At least one candidate should succeed → winner Some(n) +
    // winner_tps > 0.
    if report.winner.is_some() {
        assert!(report.winner_tps > 0.0);
        let winner_n = report.winner.unwrap();
        let n_gpu_in_candidates: Vec<u32> =
            candidates.iter().map(|p| p.n_gpu_layers).collect();
        assert!(
            n_gpu_in_candidates.contains(&winner_n),
            "winner must be one of the candidates"
        );
    }
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn placement_empty_candidates_returns_no_winner() {
    let tmp = std::env::temp_dir().join("rustllama-engine-meas-placement-empty.gguf");
    write_synthetic_llama_gguf(&tmp, &SynthLlama::default());
    let report =
        measure_placement_candidates(&tmp, &[], 64, 4, 4, 1, &default_measurement_cfg())
            .expect("empty candidates must not error");
    assert!(report.winner.is_none());
    assert_eq!(report.winner_tps, 0.0);
    assert!(report.candidates.is_empty());
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn batch_size_report_has_one_candidate_entry_per_input() {
    let tmp = std::env::temp_dir().join("rustllama-engine-meas-batch-shape.gguf");
    write_synthetic_llama_gguf(
        &tmp,
        &SynthLlama {
            n_layers: 2,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 32,
            d_model: 64,
            d_ff: 128,
            vocab: 32,
            ctx: 128,
            ..SynthLlama::default()
        },
    );
    let candidates = vec![16usize, 32, 64];
    let report = measure_batch_size_candidates(&tmp, &candidates, 64, 2, &default_measurement_cfg())
        .expect("measurement must not error on synth model");

    assert_eq!(report.candidates.len(), candidates.len());
    if let Some(b) = report.winner {
        assert!(candidates.contains(&b), "winner must be in candidate list");
        assert!(report.winner_tps > 0.0);
    }
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn batch_size_single_candidate_is_the_winner() {
    let tmp = std::env::temp_dir().join("rustllama-engine-meas-batch-single.gguf");
    write_synthetic_llama_gguf(&tmp, &SynthLlama::default());
    let report = measure_batch_size_candidates(&tmp, &[32usize], 16, 1, &default_measurement_cfg())
        .expect("single-candidate must not error");
    assert_eq!(report.winner, Some(32));
    let _ = std::fs::remove_file(&tmp);
}

/// Failed-candidate reporting: a successful neighbor + a failure
/// should produce a report with the failure's `error` populated +
/// `median_tps` = None, while the success has `median_tps` populated.
/// We can't easily synthesize a failure on the happy synth path, so
/// this test just pins that on the success path `error` is always
/// `None`.
#[test]
fn successful_candidates_have_no_error_field() {
    let tmp = std::env::temp_dir().join("rustllama-engine-meas-no-errors.gguf");
    write_synthetic_llama_gguf(&tmp, &SynthLlama::default());
    let report = measure_batch_size_candidates(&tmp, &[16usize, 32], 16, 1, &default_measurement_cfg())
        .expect("must succeed");
    for c in &report.candidates {
        if c.median_tps.is_some() {
            assert!(
                c.error.is_none(),
                "successful candidate must not have an error field set"
            );
        }
    }
    let _ = std::fs::remove_file(&tmp);
}
