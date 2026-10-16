//! P10b — EAGLE draft-head loader + autoregressive drafter (model level).
//!
//! Model-blind: no public EAGLE head exists for a supported base, so these
//! anchors use a SYNTHETIC head (real EAGLE STRUCTURE — fusion + decoder block +
//! tied LM head — with random weights). They pin:
//!   * the companion-GGUF loader (`EagleHead::from_gguf` / `from_path`) parses
//!     the `eagle.*` namespace into the expected shapes, and rejects a non-eagle
//!     file (fail-closed);
//!   * the autoregressive drafter (`forward_eagle_head`) grows a well-formed,
//!     finite, DETERMINISTIC set of per-depth logit rows from a captured base
//!     hidden — the drop-in for the MTP sidecar logits the tree builder consumes.
//! End-to-end coherence is validated by the engine-level chain ≡ linear anchor
//! (`rustllama-engine/tests/eagle.rs`), exactly like the rest of the tree-spec
//! stack.

use rustllama_gguf::synth::{
    write_synthetic_eagle_head_gguf, write_synthetic_llama_gguf, SynthEagleHead, SynthLlama,
};
use rustllama_gguf::Gguf;
use rustllama_models::llama_arch::{EagleHead, KvCache, LlamaModel};

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "rustllama-eagle-model-{name}-{}.gguf",
        std::process::id()
    ))
}

/// The loader parses the clean-room `eagle.*` namespace into the expected
/// shapes (fusion [2·d → d] + optional norms + decoder block), tied to the base
/// LM head by default.
#[test]
fn eagle_head_loads_from_companion_gguf() {
    let base = SynthLlama::default();
    let head_path = tmp("load");
    write_synthetic_eagle_head_gguf(
        &head_path,
        &SynthEagleHead {
            base: base.clone(),
            ..SynthEagleHead::default()
        },
    );
    // Load the base to get its config (the head is validated against it).
    let base_path = tmp("load-base");
    write_synthetic_llama_gguf(&base_path, &base);
    let g = Gguf::open(&base_path).expect("open base");
    let model = LlamaModel::load(&g).expect("load base");
    let d = model.cfg.d_model;

    let head = EagleHead::from_path(&head_path, &model.cfg).expect("load eagle head");
    assert_eq!(head.fc.out_dim(), d, "fusion out = d_model");
    assert_eq!(head.fc.in_dim(), 2 * d, "fusion in = 2·d_model");
    assert!(head.fc_bias.is_some(), "synth head emits fc.bias");
    assert!(head.embed_norm.is_some() && head.hidden_norm.is_some(), "pre-fusion norms");
    assert_eq!(head.block.attn_norm.len(), d, "block attn_norm = d_model");
    // Default synth head ties to the base LM head / embedding.
    assert!(head.lm_head.is_none(), "default head ties to the base output");
    assert!(head.token_embd.is_none(), "default head reuses the base embedding");

    let _ = std::fs::remove_file(&head_path);
    let _ = std::fs::remove_file(&base_path);
}

/// `own_lm_head` emits a dedicated output norm + LM head + embedding.
#[test]
fn eagle_head_with_own_lm_head_loads_its_tensors() {
    let base = SynthLlama::default();
    let head_path = tmp("own");
    write_synthetic_eagle_head_gguf(
        &head_path,
        &SynthEagleHead {
            base: base.clone(),
            own_lm_head: true,
            ..SynthEagleHead::default()
        },
    );
    let cfg = {
        let base_path = tmp("own-base");
        write_synthetic_llama_gguf(&base_path, &base);
        let g = Gguf::open(&base_path).expect("open base");
        let m = LlamaModel::load(&g).expect("load base");
        let _ = std::fs::remove_file(&base_path);
        m.cfg
    };
    let head = EagleHead::from_path(&head_path, &cfg).expect("load eagle head");
    assert!(head.output_norm.is_some(), "own output_norm");
    assert!(head.lm_head.is_some(), "own LM head");
    assert!(head.token_embd.is_some(), "own embedding");
    let _ = std::fs::remove_file(&head_path);
}

/// A plain base GGUF (no `adapter.type = "eagle"`) is rejected — fail-closed.
#[test]
fn non_eagle_gguf_is_rejected() {
    let base = SynthLlama::default();
    let base_path = tmp("reject");
    write_synthetic_llama_gguf(&base_path, &base);
    let g = Gguf::open(&base_path).expect("open base");
    let model = LlamaModel::load(&g).expect("load base");
    let err = EagleHead::from_gguf(&g, &model.cfg);
    assert!(err.is_err(), "a non-eagle GGUF must be rejected");
    let _ = std::fs::remove_file(&base_path);
}

/// The autoregressive drafter grows a well-formed, finite, DETERMINISTIC set of
/// per-depth logit rows from a captured base hidden.
#[test]
fn forward_eagle_head_drafts_deterministic_rows() {
    let base = SynthLlama::default();
    let base_path = tmp("draft-base");
    let head_path = tmp("draft-head");
    write_synthetic_llama_gguf(&base_path, &base);
    write_synthetic_eagle_head_gguf(
        &head_path,
        &SynthEagleHead {
            base: base.clone(),
            ..SynthEagleHead::default()
        },
    );
    let g = Gguf::open(&base_path).expect("open base");
    let model = LlamaModel::load(&g).expect("load base");
    let head = EagleHead::from_path(&head_path, &model.cfg).expect("load head");
    let vocab = model.cfg.vocab_size;
    let d = model.cfg.d_model;

    // Capture a base hidden exactly how the EAGLE round does (the tip forward).
    let mut kv = KvCache::new(&model.cfg, 16);
    let mut main = vec![0f32; vocab];
    let mut hidden: Vec<f32> = Vec::new();
    model.forward_one_with_hidden(5, 0, &mut kv, &mut main, &mut hidden);
    assert_eq!(hidden.len(), d, "captured hidden is d_model");

    let depth = 4usize;
    let rows = model.forward_eagle_head(&head, &hidden, 5, depth);
    assert_eq!(rows.len(), depth, "one logit row per draft depth");
    for (i, r) in rows.iter().enumerate() {
        assert_eq!(r.len(), vocab, "row {i} is full-vocab");
        assert!(r.iter().all(|x| x.is_finite()), "row {i} must be finite");
    }
    // Deterministic for identical inputs (argmax advance, no RNG).
    let rows2 = model.forward_eagle_head(&head, &hidden, 5, depth);
    assert_eq!(rows, rows2, "forward_eagle_head must be deterministic");

    // depth 0 ⇒ no rows (degenerate guard).
    assert!(model.forward_eagle_head(&head, &hidden, 5, 0).is_empty());

    let _ = std::fs::remove_file(&base_path);
    let _ = std::fs::remove_file(&head_path);
}
