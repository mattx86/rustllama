//! F9 — `ContiguousEngine` loads + runs a PURE Mamba/SSM model (all recurrent
//! layers, zero attention) through the hybrid forward (`forward_one_hybrid`
//! with no full-attention layers); the recurrent `DeltaNetCache` is the only
//! cache. Before F9 the engine refused pure-SSM checkpoints ("no MambaEngine
//! exists"); now a model carrying the hybrid DeltaNet layout with zero
//! attention layers routes straight to the hybrid path.
//!
//! F10 default-off — this file also pins that lookahead speculative decoding
//! is OFF unless explicitly set, so leaving it off is byte-identical.

use rustllama_engine::speculative::LookaheadDrafterConfig;
use rustllama_engine::{ContiguousEngine, SamplingParams};
use rustllama_gguf::synth::{
    write_synthetic_llama_gguf, write_synthetic_mamba_gguf, SynthHybrid, SynthLlama, SynthMamba,
};
use rustllama_models::llama_arch::KvDtype;

fn greedy(n: u32) -> SamplingParams {
    SamplingParams {
        temperature: 0.0,
        top_p: 0.0,
        top_k: 0,
        repeat_penalty: 1.0,
        max_tokens: n,
        stop: vec![],
        seed: 0,
        ..SamplingParams::default()
    }
}

fn write_pure_ssm(tag: &str, n_layers: u32) -> std::path::PathBuf {
    let tmp = std::env::temp_dir().join(format!("rustllama-pure-ssm-eng-{tag}.gguf"));
    let params = SynthLlama {
        n_layers,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 8,
        d_model: 32,
        d_ff: 64,
        vocab: 32,
        ctx: 64,
        hybrid: Some(SynthHybrid::pure_ssm(n_layers, 32)),
        ..SynthLlama::default()
    };
    write_synthetic_llama_gguf(&tmp, &params);
    tmp
}

/// F9 anchor: `ContiguousEngine` loads a pure-SSM model (NOT refused) and decodes it
/// to a finite, deterministic token sequence through `forward_one_hybrid`.
#[test]
fn cpu_engine_loads_and_runs_pure_ssm() {
    let tmp = write_pure_ssm("run", 4);

    let mut cpu =
        ContiguousEngine::load_with_options_and_layout(&tmp, 64, true, KvDtype::F32, "contiguous")
            .expect(
                "pure-SSM model must load via ContiguousEngine (hybrid forward, zero attention)",
            );
    cpu.set_prefix_cache(false);

    // F10 default-off: lookahead spec is None on a freshly loaded engine.
    assert!(
        cpu.lookahead_speculative().is_none(),
        "lookahead speculative must be OFF by default (byte-identical)"
    );

    let prompt: Vec<i32> = vec![3, 1, 4, 1, 5, 9];
    let out1 = cpu
        .generate_token_ids(&prompt, 5, &greedy(5))
        .expect("pure-SSM greedy generate");
    assert!(!out1.is_empty(), "pure-SSM decode produced no tokens");
    assert!(
        out1.iter().all(|&t| (t as usize) < cpu.vocab_size()),
        "token id out of vocab range"
    );

    // Determinism: a second identical greedy run must match token-for-token.
    let out2 = cpu
        .generate_token_ids(&prompt, 5, &greedy(5))
        .expect("pure-SSM greedy generate (repeat)");
    assert_eq!(out1, out2, "pure-SSM greedy decode is not deterministic");

    let _ = std::fs::remove_file(&tmp);
}

fn write_canonical_mamba(tag: &str, n_layers: u32) -> std::path::PathBuf {
    let tmp = std::env::temp_dir().join(format!("rustllama-canon-mamba-eng-{tag}.gguf"));
    write_synthetic_mamba_gguf(
        &tmp,
        &SynthMamba {
            n_layers,
            ..SynthMamba::default()
        },
    );
    tmp
}

/// F9-tail anchor: a CANONICAL Mamba/SSM GGUF (arch `mamba`, the selective-scan
/// recurrence — NOT the DeltaNet hybrid layout) LOADS through `ContiguousEngine` (it is
/// no longer refused) and greedily decodes to a finite, deterministic, in-vocab
/// token sequence through the wired `mamba_arch::MambaModel` selective scan.
#[test]
fn cpu_engine_loads_and_runs_canonical_mamba() {
    let tmp = write_canonical_mamba("run", 3);

    let mut cpu =
        ContiguousEngine::load_with_options_and_layout(&tmp, 64, true, KvDtype::F32, "contiguous")
            .expect(
            "canonical Mamba model must load via ContiguousEngine (selective-scan engine wired, \
             not refused)",
        );
    cpu.set_prefix_cache(false);

    let prompt: Vec<i32> = vec![3, 1, 4, 1, 5, 9];
    let out1 = cpu
        .generate_token_ids(&prompt, 5, &greedy(5))
        .expect("canonical Mamba greedy generate");
    assert!(
        !out1.is_empty(),
        "canonical Mamba decode produced no tokens"
    );
    assert!(
        out1.iter().all(|&t| (t as usize) < cpu.vocab_size()),
        "token id out of vocab range"
    );

    // Determinism: a second identical greedy run must match token-for-token
    // (a fresh selective-scan cache per call starts from zero state).
    let out2 = cpu
        .generate_token_ids(&prompt, 5, &greedy(5))
        .expect("canonical Mamba greedy generate (repeat)");
    assert_eq!(
        out1, out2,
        "canonical Mamba greedy decode is not deterministic"
    );

    let _ = std::fs::remove_file(&tmp);
}

/// F10 knob: `set_lookahead_speculative` round-trips, so the serve / HTTP load
/// paths can select it per the `[inference].speculative_lookahead` config.
#[test]
fn lookahead_knob_round_trips() {
    let tmp = write_pure_ssm("knob", 2);
    let mut cpu =
        ContiguousEngine::load_with_options_and_layout(&tmp, 64, true, KvDtype::F32, "contiguous")
            .expect("load");
    assert!(cpu.lookahead_speculative().is_none());
    cpu.set_lookahead_speculative(Some(LookaheadDrafterConfig {
        ngram: 3,
        window: 4,
        q: 0.5,
    }));
    let got = cpu.lookahead_speculative().expect("lookahead now set");
    assert_eq!(got.ngram, 3);
    assert_eq!(got.window, 4);
    let _ = std::fs::remove_file(&tmp);
}
