//! TurboQuant + NVFP4 KV-cache integration tests on the synthetic
//! Llama model.
//!
//! The F32 baseline is the reference; quantized paths must:
//!   1. Not panic during load + first decode (smoke).
//!   2. Be deterministic across runs (`run(prompt) == run(prompt)`).
//!   3. Run to completion with the same `max_tokens` cap.
//!
//! Exact greedy-output parity with F32 is NOT asserted for all
//! variants: tq1 / tq2 carry far too little resolution to round-trip
//! greedy decisions on a near-random synth model, and tq4 + nvfp4
//! sit on the boundary. Q8_0 has a dedicated parity test (see
//! `kv_q8_0.rs`). The fully-fused dequant-in-attention path and SIMD
//! variants land in a follow-up.

use rustllama_engine::{CpuEngine, KvDtype, SamplingParams};
use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};

fn synth_gguf(tag: &str) -> std::path::PathBuf {
    let tmp = std::env::temp_dir().join(format!("rustllama-kv-{tag}.gguf"));
    write_synthetic_llama_gguf(&tmp, &SynthLlama::default());
    tmp
}

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

#[test]
fn tq8_kv_load_decode_and_determinism() {
    // tq8 (TurboQuant 8-bit) should run end-to-end without panics
    // and produce identical output on two back-to-back runs.
    let path = synth_gguf("tq8-determ");
    let engine = CpuEngine::load_with_options(&path, 32, false, KvDtype::Tq(8))
        .expect("load tq8");
    let prompt: Vec<i32> = (0..8).collect();
    let a = engine
        .generate_token_ids(&prompt, 4, &greedy(4))
        .expect("a");
    let b = engine
        .generate_token_ids(&prompt, 4, &greedy(4))
        .expect("b");
    assert_eq!(a, b, "tq8 KV must be deterministic");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn tq4_kv_load_decode_and_determinism() {
    let path = synth_gguf("tq4-determ");
    let engine = CpuEngine::load_with_options(&path, 32, false, KvDtype::Tq(4))
        .expect("load tq4");
    let prompt: Vec<i32> = (0..8).collect();
    let a = engine
        .generate_token_ids(&prompt, 4, &greedy(4))
        .expect("a");
    let b = engine
        .generate_token_ids(&prompt, 4, &greedy(4))
        .expect("b");
    assert_eq!(a, b, "tq4 KV must be deterministic");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn tq2_kv_load_decode_does_not_panic() {
    // tq2 is the most aggressive (3 levels per element); we just
    // pin that it runs to completion.
    let path = synth_gguf("tq2-smoke");
    let engine = CpuEngine::load_with_options(&path, 32, false, KvDtype::Tq(2))
        .expect("load tq2");
    let prompt: Vec<i32> = (0..8).collect();
    let out = engine
        .generate_token_ids(&prompt, 4, &greedy(4))
        .expect("tq2 gen");
    assert!(out.len() <= 4);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn tq1_kv_load_decode_does_not_panic() {
    // tq1 is signed-binary. Same smoke check.
    let path = synth_gguf("tq1-smoke");
    let engine = CpuEngine::load_with_options(&path, 32, false, KvDtype::Tq(1))
        .expect("load tq1");
    let prompt: Vec<i32> = (0..8).collect();
    let out = engine
        .generate_token_ids(&prompt, 4, &greedy(4))
        .expect("tq1 gen");
    assert!(out.len() <= 4);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn nvfp4_kv_load_decode_and_determinism() {
    // NVFP4 requires head_dim divisible by 16. The synth default
    // has head_dim=32 which satisfies the constraint.
    let path = synth_gguf("nvfp4-determ");
    let engine = CpuEngine::load_with_options(&path, 32, false, KvDtype::Nvfp4)
        .expect("load nvfp4");
    let prompt: Vec<i32> = (0..8).collect();
    let a = engine
        .generate_token_ids(&prompt, 4, &greedy(4))
        .expect("a");
    let b = engine
        .generate_token_ids(&prompt, 4, &greedy(4))
        .expect("b");
    assert_eq!(a, b, "NVFP4 KV must be deterministic");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn q4_0_kv_load_decode_and_determinism() {
    // Q4_0 (ggml blocks, the Prism-compatible 4-bit KV) requires
    // head_dim divisible by 32; the synth default head_dim=32
    // satisfies the constraint exactly.
    let path = synth_gguf("q4-0-determ");
    let engine = CpuEngine::load_with_options(&path, 32, false, KvDtype::Q4_0)
        .expect("load q4_0");
    let prompt: Vec<i32> = (0..8).collect();
    let a = engine
        .generate_token_ids(&prompt, 4, &greedy(4))
        .expect("a");
    let b = engine
        .generate_token_ids(&prompt, 4, &greedy(4))
        .expect("b");
    assert_eq!(a, b, "Q4_0 KV must be deterministic");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn q4_0_kv_whitening_active_matches_f32_greedy() {
    // head_dim 64 activates the KV whitening gate (head_dim % 64 == 0,
    // default ON for Q4_0). The whitened-Q4_0 greedy path must be
    // deterministic AND agree with the F32-KV greedy tokens on this
    // low-entropy synth model — whitening is orthonormal, so only
    // quantization noise (which it *reduces*) separates the two.
    let mut synth = SynthLlama::default();
    synth.head_dim = 64;
    synth.d_model = synth.n_heads * 64;
    let tmp = std::env::temp_dir().join("rustllama-kv-q4-0-whiten.gguf");
    write_synthetic_llama_gguf(&tmp, &synth);

    let prompt: Vec<i32> = (0..8).collect();
    let q4 = CpuEngine::load_with_options(&tmp, 32, false, KvDtype::Q4_0)
        .expect("load q4_0");
    let a = q4.generate_token_ids(&prompt, 4, &greedy(4)).expect("a");
    let b = q4.generate_token_ids(&prompt, 4, &greedy(4)).expect("b");
    assert_eq!(a, b, "whitened Q4_0 KV must be deterministic");

    let f32e = CpuEngine::load_with_options(&tmp, 32, false, KvDtype::F32)
        .expect("load f32");
    let f = f32e.generate_token_ids(&prompt, 4, &greedy(4)).expect("f");
    assert_eq!(a, f, "whitened Q4_0 greedy must match F32 greedy on the synth model");
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn kv_dtype_parse_handles_all_variants() {
    use rustllama_engine::KvDtype;
    assert_eq!(KvDtype::parse("f32"), Some(KvDtype::F32));
    assert_eq!(KvDtype::parse("F32"), Some(KvDtype::F32));
    assert_eq!(KvDtype::parse("q8_0"), Some(KvDtype::Q8_0));
    assert_eq!(KvDtype::parse("q4_0"), Some(KvDtype::Q4_0));
    assert_eq!(KvDtype::parse("Q4_0"), Some(KvDtype::Q4_0));
    assert_eq!(KvDtype::parse("tq1"), Some(KvDtype::Tq(1)));
    assert_eq!(KvDtype::parse("tq2"), Some(KvDtype::Tq(2)));
    assert_eq!(KvDtype::parse("tq4"), Some(KvDtype::Tq(4)));
    assert_eq!(KvDtype::parse("tq8"), Some(KvDtype::Tq(8)));
    assert_eq!(KvDtype::parse("TQ8"), Some(KvDtype::Tq(8)));
    assert_eq!(KvDtype::parse("nvfp4"), Some(KvDtype::Nvfp4));
    assert_eq!(KvDtype::parse("NVFP4"), Some(KvDtype::Nvfp4));
    assert_eq!(KvDtype::parse("bogus"), None);
}
