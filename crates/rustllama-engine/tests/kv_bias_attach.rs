//! K-cache mean-centering bias — engine integration tests.
//!
//! The subtraction is exactly softmax-invariant, so a ZERO bias must
//! leave greedy output bitwise unchanged — that pins the whole
//! plumbing (sidecar load → attach → Q4_0 write-arm subtract) without
//! making quality claims. Basis-mismatch refusal mirrors the fork.

use rustllama_engine::kv_bias::KvBiasData;
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
        seed: 0,
        ..SamplingParams::default()
    }
}

/// head_dim 64 so the whitening gate is ACTIVE (matches the real
/// Bonsai deployment shape of the bias: whiten → subtract → quantize).
fn synth_64(tag: &str) -> (std::path::PathBuf, SynthLlama) {
    let mut synth = SynthLlama::default();
    synth.head_dim = 64;
    synth.d_model = synth.n_heads * 64;
    let tmp = std::env::temp_dir().join(format!("rustllama-kvbias-{tag}.gguf"));
    write_synthetic_llama_gguf(&tmp, &synth);
    (tmp, synth)
}

#[test]
fn zero_bias_attach_is_greedy_identity() {
    let (model, synth) = synth_64("zero-ident");
    let prompt: Vec<i32> = (0..8).collect();

    let baseline = {
        let e = CpuEngine::load_with_options(&model, 32, false, KvDtype::Q4_0).expect("load");
        e.generate_token_ids(&prompt, 4, &greedy(4)).expect("gen")
    };

    // Zero bias for every layer, basis matching the active whitening
    // (head_dim 64 → whitening on unless the env lever disabled it).
    let whitening =
        rustllama_engine::kv_whitening_active_for(KvDtype::Q4_0, synth.head_dim as usize);
    let d_kv = (synth.n_kv_heads * synth.head_dim) as usize;
    let bias = KvBiasData {
        per_layer: (0..synth.n_layers as usize)
            .map(|_| Some(vec![0f32; d_kv]))
            .collect(),
        k_rot: Some(whitening),
    };
    let sidecar = model.with_extension("kvbias.gguf");
    bias.save_with_geometry(&sidecar, synth.head_dim as usize, synth.n_kv_heads as usize)
        .expect("save sidecar");

    let e = CpuEngine::load_with_options(&model, 32, false, KvDtype::Q4_0).expect("load2");
    // Auto-discovery: sidecar sits beside the model.
    let attached = e.attach_kv_bias(None).expect("attach");
    assert!(attached, "sidecar beside the model must auto-attach on q4_0 KV");
    let with_bias = e.generate_token_ids(&prompt, 4, &greedy(4)).expect("gen2");

    assert_eq!(
        baseline, with_bias,
        "a zero bias must be a bitwise no-op (softmax invariance + zero subtract)"
    );
    let _ = std::fs::remove_file(&sidecar);
    let _ = std::fs::remove_file(&model);
}

#[test]
fn basis_mismatch_is_refused() {
    let (model, synth) = synth_64("basis-mismatch");
    let whitening =
        rustllama_engine::kv_whitening_active_for(KvDtype::Q4_0, synth.head_dim as usize);
    let d_kv = (synth.n_kv_heads * synth.head_dim) as usize;
    let bias = KvBiasData {
        per_layer: (0..synth.n_layers as usize)
            .map(|_| Some(vec![0.25f32; d_kv]))
            .collect(),
        // Deliberately the WRONG basis.
        k_rot: Some(!whitening),
    };
    let sidecar = model.with_extension("kvbias.gguf");
    bias.save_with_geometry(&sidecar, synth.head_dim as usize, synth.n_kv_heads as usize)
        .expect("save sidecar");

    let e = CpuEngine::load_with_options(&model, 32, false, KvDtype::Q4_0).expect("load");
    let err = e
        .attach_kv_bias(Some(&sidecar))
        .expect_err("mismatched basis must refuse to attach");
    let msg = format!("{err}");
    assert!(msg.contains("recalibrate"), "unexpected error text: {msg}");
    let _ = std::fs::remove_file(&sidecar);
    let _ = std::fs::remove_file(&model);
}

#[test]
fn no_sidecar_auto_mode_is_quiet_noop() {
    let (model, _synth) = synth_64("no-sidecar");
    let e = CpuEngine::load_with_options(&model, 32, false, KvDtype::Q4_0).expect("load");
    assert!(!e.attach_kv_bias(None).expect("attach"), "no sidecar → Ok(false)");
    let _ = std::fs::remove_file(&model);
}

#[test]
fn non_q4_0_cache_with_explicit_path_errors() {
    let (model, synth) = synth_64("f32-explicit");
    let d_kv = (synth.n_kv_heads * synth.head_dim) as usize;
    let bias = KvBiasData {
        per_layer: (0..synth.n_layers as usize)
            .map(|_| Some(vec![0f32; d_kv]))
            .collect(),
        k_rot: Some(false),
    };
    let sidecar = model.with_extension("kvbias.gguf");
    bias.save_with_geometry(&sidecar, synth.head_dim as usize, synth.n_kv_heads as usize)
        .expect("save sidecar");
    let e = CpuEngine::load_with_options(&model, 32, false, KvDtype::F32).expect("load");
    // Auto mode: quiet skip on non-Q4_0.
    assert!(!e.attach_kv_bias(None).expect("auto"), "auto + f32 KV → quiet skip");
    // Explicit: hard error.
    e.attach_kv_bias(Some(&sidecar))
        .expect_err("explicit kv_bias_path + non-q4_0 cache must error");
    let _ = std::fs::remove_file(&sidecar);
    let _ = std::fs::remove_file(&model);
}
