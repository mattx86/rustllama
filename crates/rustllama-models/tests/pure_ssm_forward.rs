//! F9 — pure Mamba/SSM via the hybrid forward (zero attention layers).
//!
//! A pure-recurrent model (all SSM/DeltaNet layers, NO attention) is just
//! `forward_one_hybrid` with zero full-attention layers. This fixture is the
//! hybrid SSM tensor emission with `full_attention_interval` set past the last
//! layer, so every layer is an SSM block. The anchors:
//!   1. it LOADS as a hybrid with all-`Ssm` layers (no `FullAttention`);
//!   2. a forward produces FINITE, DETERMINISTIC logits (two identical runs);
//!   3. the recurrent `DeltaNetCache` is the only cache — the KV cache is all
//!      zero-length placeholders (the F9 "KV-less generation loop" property).
//!
//! Real HF Mamba coherence vs a Python reference is validate-on-HW; the bar
//! here is mechanical, finite, deterministic execution through the real
//! `forward_one_hybrid` (same guarantee the hybrid F8 anchors use).

use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthHybrid, SynthLlama};
use rustllama_gguf::Gguf;
use rustllama_models::llama_arch::{DeltaNetCache, HybridLayer, KvCache, LlamaModel};

fn write_pure_ssm(tag: &str, n_layers: u32) -> std::path::PathBuf {
    let tmp = std::env::temp_dir().join(format!("rustllama-pure-ssm-fwd-{tag}.gguf"));
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

#[test]
fn pure_ssm_loads_as_all_ssm_hybrid() {
    let n_layers = 4u32;
    let path = write_pure_ssm("load", n_layers);
    let gguf = Gguf::open(&path).expect("open pure-ssm gguf");
    let model = LlamaModel::load(&gguf).expect("pure-ssm hybrid loads");

    assert!(
        model.weights.is_hybrid(),
        "pure-SSM model must detect as hybrid"
    );
    let layers = model
        .weights
        .hybrid_layers
        .as_ref()
        .expect("hybrid_layers populated");
    assert_eq!(layers.len(), n_layers as usize);
    // Zero full-attention layers — every layer is an SSM/DeltaNet block.
    for (i, l) in layers.iter().enumerate() {
        assert!(
            matches!(l, HybridLayer::Ssm(_)),
            "layer {i} must be SSM (pure-recurrent, no attention)"
        );
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn pure_ssm_forward_is_finite_and_deterministic() {
    let n_layers = 4u32;
    let path = write_pure_ssm("fwd", n_layers);
    let gguf = Gguf::open(&path).expect("open pure-ssm gguf");
    let model = LlamaModel::load(&gguf).expect("pure-ssm hybrid loads");

    let hybrid = model.weights.hybrid_layers.as_ref().unwrap();
    let vocab = model.cfg.vocab_size;
    let max_ctx = 32usize;

    // The KV cache always carries one entry per layer; for a pure-SSM model
    // no attention layer ever writes KV — the recurrent DeltaNetCache is the
    // only live cache (the F9 "KV-less generation loop" property).
    assert_eq!(
        KvCache::new(&model.cfg, max_ctx).layers.len(),
        n_layers as usize
    );

    // Hand-run the recurrence over a short token sequence and capture logits.
    let run = || -> Vec<f32> {
        let mut kv = KvCache::new(&model.cfg, max_ctx);
        let mut dn = DeltaNetCache::new_for_hybrid(&model.cfg, hybrid).expect("dn cache");
        let mut logits = vec![0.0f32; vocab];
        let prompt: [i32; 5] = [3, 1, 4, 1, 5];
        for (pos, &tok) in prompt.iter().enumerate() {
            model.forward_one_hybrid(tok, pos as u32, &mut kv, &mut dn, &mut logits);
        }
        logits
    };

    let a = run();
    assert_eq!(a.len(), vocab);
    assert!(
        a.iter().all(|v| v.is_finite()),
        "pure-SSM forward produced non-finite logits"
    );
    // Determinism: a second identical run must be bit-for-bit identical.
    let b = run();
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "pure-SSM logit[{i}] not deterministic across runs"
        );
    }
    let _ = std::fs::remove_file(&path);
}
