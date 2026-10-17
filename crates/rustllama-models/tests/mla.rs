//! M1 — Multi-head Latent Attention (DeepSeek-V2/V3) load + forward.
//!
//! Contracts pinned here:
//!   1. A `deepseek2`/MLA GGUF LOADS (no `attn_k.weight` error — the old
//!      failure mode was routing it through the plain-Llama GQA path).
//!   2. The MLA config parses (low-rank Q, compressed-latent KV, decoupled
//!      RoPE dims) and the model reports `is_mla()`.
//!   3. `forward_one` (absorbed decode) produces FINITE logits.
//!   4. **naive ≡ absorbed**: the naive decode (materialize per-head K/V
//!      from the stored latent, standard attention) equals the absorbed
//!      decode (fold `attn_kv_b`, score against the latent) within 1e-3.
//!      This is the correctness proof for the memory-saving absorbed path.
//!   5. Covers the low-rank-Q, direct-Q (V2-Lite), and MoE-FFN shapes.

use rustllama_gguf::synth::{write_synthetic_deepseek_mla_gguf, SynthMla, SynthMoe};
use rustllama_gguf::Gguf;
use rustllama_models::llama_arch::{KvCache, KvDtype, LlamaModel};
use rustllama_models::llama_config::LlamaConfig;

fn synth_path(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("rustllama-mla-fwd-{tag}.gguf"))
}

/// Run the given token sequence from a fresh cache, returning each step's
/// full logit vector. `naive` picks the naive vs absorbed decode.
fn run_sequence(
    model: &LlamaModel,
    cfg: &LlamaConfig,
    tokens: &[i32],
    naive: bool,
) -> Vec<Vec<f32>> {
    let mut kv = KvCache::new_with_dtype(cfg, cfg.ctx_train, KvDtype::F32);
    assert!(kv.mla.is_some(), "MLA model must build an MLA KV cache");
    let mut out = Vec::new();
    for (i, &tok) in tokens.iter().enumerate() {
        let mut logits = vec![0f32; cfg.vocab_size];
        if naive {
            model.forward_one_mla_naive(tok, i as u32, &mut kv, &mut logits);
        } else {
            model.forward_one(tok, i as u32, &mut kv, &mut logits);
        }
        out.push(logits);
    }
    out
}

fn assert_all_finite(steps: &[Vec<f32>], label: &str) {
    for (s, logits) in steps.iter().enumerate() {
        for (i, v) in logits.iter().enumerate() {
            assert!(
                v.is_finite(),
                "{label}: step {s} logit[{i}] = {v} not finite"
            );
        }
    }
}

/// Load + run an MLA fixture, asserting: load succeeds, is_mla, finite
/// logits, and naive ≡ absorbed within 1e-3 over a multi-token sequence.
fn load_run_and_check_parity(params: &SynthMla, tag: &str) {
    let path = synth_path(tag);
    write_synthetic_deepseek_mla_gguf(&path, params);
    let gguf = Gguf::open(&path).expect("open MLA gguf");
    let cfg = LlamaConfig::from_gguf(&gguf).expect("parse MLA config");

    // (2) MLA config parsed.
    let mla = cfg
        .mla
        .as_ref()
        .expect("cfg.mla must be Some for a deepseek2/MLA GGUF");
    assert_eq!(mla.kv_lora_rank as u32, params.kv_lora_rank);
    assert_eq!(mla.qk_rope_head_dim as u32, params.qk_rope_head_dim);
    assert_eq!(mla.v_head_dim as u32, params.v_head_dim);
    assert_eq!(mla.qk_nope_head_dim as u32, params.qk_nope_head_dim);
    assert_eq!(
        mla.q_lora_rank.map(|r| r as u32),
        params.q_lora_rank,
        "q_lora_rank parse"
    );

    // (1) LOADS (would previously die at `attn_k.weight`).
    let model = LlamaModel::load(&gguf).expect("MLA GGUF must load, not die at attn_k.weight");
    assert!(model.weights.is_mla(), "model must report is_mla()");

    // (3) + (4): run a multi-token sequence both ways.
    let tokens: Vec<i32> = (0..5).map(|i| i % cfg.vocab_size as i32).collect();
    let absorbed = run_sequence(&model, &cfg, &tokens, false);
    let naive = run_sequence(&model, &cfg, &tokens, true);
    assert_all_finite(&absorbed, "absorbed");
    assert_all_finite(&naive, "naive");

    let mut max_abs = 0.0f32;
    for (a_step, n_step) in absorbed.iter().zip(naive.iter()) {
        for (a, n) in a_step.iter().zip(n_step.iter()) {
            max_abs = max_abs.max((a - n).abs());
        }
    }
    assert!(
        max_abs < 1e-3,
        "{tag}: naive vs absorbed max |Δlogit| = {max_abs} exceeds 1e-3"
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn mla_low_rank_q_dense_ffn_loads_runs_and_naive_equals_absorbed() {
    load_run_and_check_parity(&SynthMla::default(), "lowrank-dense");
}

#[test]
fn mla_direct_q_v2lite_shape_loads_runs_and_naive_equals_absorbed() {
    // DeepSeek-V2-Lite shape: no low-rank Q (single direct `attn_q`).
    load_run_and_check_parity(
        &SynthMla {
            q_lora_rank: None,
            ..SynthMla::default()
        },
        "directq-dense",
    );
}

#[test]
fn mla_with_moe_ffn_and_dense_prefix_loads_runs_and_naive_equals_absorbed() {
    // DeepSeek-V2 full shape: MLA attention + MoE FFN with a dense first
    // layer (first_k_dense_replace) + shared experts.
    load_run_and_check_parity(
        &SynthMla {
            n_layers: 3,
            n_dense_layers: 1,
            moe: Some(SynthMoe {
                n_experts: 4,
                n_experts_used: 2,
                n_experts_shared: 1,
            }),
            ..SynthMla::default()
        },
        "moe-dense-prefix",
    );
}

#[test]
fn mla_asymmetric_head_dims_loads_and_parity_holds() {
    // qk_nope != qk_rope != v_head_dim, multiple heads — exercises the
    // per-head weight-fold indexing in the absorbed path.
    load_run_and_check_parity(
        &SynthMla {
            n_heads: 3,
            d_model: 48,
            q_lora_rank: Some(24),
            kv_lora_rank: 20,
            qk_nope_head_dim: 12,
            qk_rope_head_dim: 8,
            v_head_dim: 16,
            ..SynthMla::default()
        },
        "asym",
    );
}
