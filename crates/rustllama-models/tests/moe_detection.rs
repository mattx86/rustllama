//! MoE phase-1 contract: detect MoE GGUFs at load time and surface a
//! clear `LlamaLoadError::UnsupportedMoe` error instead of the cryptic
//! "missing `ffn_gate.weight`" you'd otherwise see when the dense FFN
//! load path runs against a MoE model.
//!
//! Phase 2 will land tensor binding + forward pass for MoE; this
//! turn's contract is just: detect + reject cleanly.

use rustllama_gguf::synth::{
    write_synthetic_llama_gguf, SynthHybrid, SynthLlama, SynthMoe,
};
use rustllama_gguf::Gguf;
use rustllama_models::llama_arch::{LlamaLoadError, LlamaWeights};
use rustllama_models::llama_config::LlamaConfig;

fn synth_path(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("rustllama-synth-moe-{tag}.gguf"))
}

#[test]
fn dense_gguf_has_no_moe_config() {
    // Sanity: the existing dense default SynthLlama doesn't emit MoE
    // keys, so the config's moe field is None.
    let path = synth_path("dense-sanity");
    write_synthetic_llama_gguf(&path, &SynthLlama::default());
    let gguf = Gguf::open(&path).expect("open dense gguf");
    let cfg = LlamaConfig::from_gguf(&gguf).expect("dense gguf parses");
    assert!(cfg.moe.is_none(), "dense GGUF must not carry MoE config");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn moe_gguf_parses_with_mixtral_shape_metadata() {
    // Mixtral-8x7B: 8 experts, top-2 routed, 0 shared.
    let path = synth_path("mixtral-shape");
    write_synthetic_llama_gguf(
        &path,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 8,
                n_experts_used: 2,
                n_experts_shared: 0,
            }),
            ..SynthLlama::default()
        },
    );
    let gguf = Gguf::open(&path).expect("open moe gguf");
    let cfg = LlamaConfig::from_gguf(&gguf).expect("moe metadata parses");
    let moe = cfg.moe.expect("moe field must be populated");
    assert_eq!(moe.n_experts, 8);
    assert_eq!(moe.n_experts_used, 2);
    assert_eq!(moe.n_experts_shared, 0);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn hybrid_gguf_populates_hybrid_config_with_qwen35moe_shape() {
    // qwen35moe Phase 2 contract: when the GGUF declares
    // `{arch}.full_attention_interval` + `{arch}.ssm.*`, the loader
    // populates `LlamaConfig::hybrid` with all five SSM params +
    // the shared-expert FFN dim + the NextN head count. Layers can
    // then be classified into full-attention vs SSM via
    // `(i + 1) % full_attention_interval == 0`.
    let path = synth_path("qwen35moe-hybrid-cfg");
    write_synthetic_llama_gguf(
        &path,
        &SynthLlama {
            n_layers: 8,
            moe: Some(SynthMoe {
                n_experts: 256,
                n_experts_used: 8,
                n_experts_shared: 0,
            }),
            hybrid: Some(SynthHybrid {
                full_attention_interval: 4,
                ssm_state_size: 128,
                ssm_conv_kernel: 4,
                ssm_group_count: 16,
                ssm_time_step_rank: 32,
                ssm_inner_size: 4096,
                shared_expert_feed_forward_length: Some(512),
                nextn_predict_layers: 1,
            }),
            ..SynthLlama::default()
        },
    );
    let gguf = Gguf::open(&path).expect("open hybrid gguf");
    let cfg = LlamaConfig::from_gguf(&gguf).expect("hybrid metadata parses");
    let hyb = cfg.hybrid.as_ref().expect("hybrid field must be populated");
    assert_eq!(hyb.full_attention_interval, 4);
    assert_eq!(hyb.ssm_state_size, 128);
    assert_eq!(hyb.ssm_conv_kernel, 4);
    assert_eq!(hyb.ssm_group_count, 16);
    assert_eq!(hyb.ssm_time_step_rank, 32);
    assert_eq!(hyb.ssm_inner_size, 4096);
    assert_eq!(hyb.shared_expert_feed_forward_length, Some(512));
    assert_eq!(hyb.nextn_predict_layers, 1);

    // Layer-classification sanity: with 8 layers + interval=4,
    // full-attention layers are indices where (i+1) % 4 == 0 →
    // 3 and 7 (two of eight). The rest are SSM.
    let attn_layers: Vec<usize> = (0..cfg.n_layers)
        .filter(|i| (i + 1) % hyb.full_attention_interval as usize == 0)
        .collect();
    assert_eq!(attn_layers, vec![3, 7]);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn dense_gguf_has_no_hybrid_config() {
    // Sanity: a plain dense or MoE GGUF without SSM keys must NOT
    // accidentally populate `hybrid`. Detection requires BOTH
    // `full_attention_interval > 0` AND `ssm.state_size` present.
    let path = synth_path("dense-no-hybrid");
    write_synthetic_llama_gguf(&path, &SynthLlama::default());
    let gguf = Gguf::open(&path).expect("open dense gguf");
    let cfg = LlamaConfig::from_gguf(&gguf).expect("dense gguf parses");
    assert!(cfg.hybrid.is_none(), "dense GGUF must not carry hybrid config");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn moe_gguf_with_shared_experts_parses_deepseek_v3_shape() {
    // DeepSeek-V3: 256 routed experts + 1 shared (the v3 paper uses
    // shared-expert design to capture common patterns). The exact
    // numbers here are illustrative — we just verify shared > 0
    // propagates through the loader.
    let path = synth_path("deepseek-shared");
    write_synthetic_llama_gguf(
        &path,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 256,
                n_experts_used: 8,
                n_experts_shared: 1,
            }),
            ..SynthLlama::default()
        },
    );
    let gguf = Gguf::open(&path).expect("open moe gguf");
    let cfg = LlamaConfig::from_gguf(&gguf).expect("moe metadata parses");
    let moe = cfg.moe.expect("moe field");
    assert_eq!(moe.n_experts_shared, 1, "shared-expert count round-trips");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn moe_gguf_load_succeeds_phase_2c() {
    // Phase 2-C contract: MoE GGUFs now load through the regular
    // `LlamaModel::load` entry point. Phase 1 + 2-A + 2-B-2 all
    // gated MoE behind a clean rejection pending forward-pass
    // support; phase 2-C wires the serial-prefill MoE path so the
    // chat/completion endpoints work end-to-end against a real
    // Mixtral-shape GGUF.
    let path = synth_path("phase2c-mixtral-loads");
    write_synthetic_llama_gguf(
        &path,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 8,
                n_experts_used: 2,
                n_experts_shared: 0,
            }),
            ..SynthLlama::default()
        },
    );
    let gguf = Gguf::open(&path).expect("open");
    let model = rustllama_models::llama_arch::LlamaModel::load(&gguf)
        .expect("phase 2-C: MoE must load through the public entry point");
    assert!(model.weights.is_moe(), "loaded model must report is_moe");
    let _ = std::fs::remove_file(&path);
}

#[test]
fn unsupported_moe_error_variant_still_displays_for_future_use() {
    // Phase 2-C removed the active `LlamaModel::load` rejection
    // but the `LlamaLoadError::UnsupportedMoe` variant stays —
    // future MoE-architecture variants we don't support can use
    // the same error shape. Construct one directly and verify
    // the Display format clients pattern-match on still works.
    let err = LlamaLoadError::UnsupportedMoe {
        arch: "future_moe_v2".into(),
        n_experts: 64,
        n_experts_used: 8,
        shared_msg: ", 2 shared".into(),
    };
    let msg = err.to_string();
    assert!(msg.contains("MoE"));
    assert!(msg.contains("64 experts"));
    assert!(msg.contains("top-8"));
    assert!(msg.contains("2 shared"));
}

// ---- Phase 2-A: tensor binding succeeds via LlamaWeights::from_gguf ---------
//
// The engine-facing `LlamaModel::load` still rejects MoE pending the
// phase 2-B forward pass, but inspection / introspection tools that
// just want to read tensor shapes (the GGUF dumper, `/api/show`,
// future MoE-aware engines) can call `LlamaWeights::from_gguf`
// directly and walk the bound expert tensors.

#[test]
fn moe_weights_from_gguf_succeeds_and_binds_router_plus_expert_tensors() {
    let path = synth_path("phase2a-mixtral-binding");
    let synth = SynthMoe {
        n_experts: 8,
        n_experts_used: 2,
        n_experts_shared: 0,
    };
    write_synthetic_llama_gguf(
        &path,
        &SynthLlama {
            moe: Some(synth),
            ..SynthLlama::default()
        },
    );
    let gguf = Gguf::open(&path).expect("open");
    let cfg = LlamaConfig::from_gguf(&gguf).expect("config");
    let weights = LlamaWeights::from_gguf(&gguf, &cfg).expect(
        "phase 2-A: MoE weights must bind successfully through from_gguf \
         even though LlamaModel::load still rejects pending forward",
    );

    assert!(weights.is_moe(), "is_moe() must reflect MoE binding");
    assert!(
        weights.blocks.is_empty(),
        "dense `blocks` must stay empty when MoE binds"
    );
    let mbs = weights.moe_blocks.as_ref().expect("moe_blocks populated");
    assert_eq!(
        mbs.len(),
        cfg.n_layers,
        "one MoE block per transformer layer"
    );

    // Spot-check the first layer: router + 3 expert tensors must
    // all be bound, shared-expert tensors must NOT be (non-shared
    // synth fixture).
    let blk0 = &mbs[0];
    // Just verify the fields exist with the expected emptiness/
    // presence — fine-grained shape inspection lands in phase 2-B
    // when the forward pass actually consumes the tensors.
    let _ = &blk0.router;
    let _ = &blk0.w_gate_exps;
    let _ = &blk0.w_up_exps;
    let _ = &blk0.w_down_exps;
    assert!(blk0.w_gate_shared.is_none(), "non-shared MoE: gate_shared absent");
    assert!(blk0.w_up_shared.is_none());
    assert!(blk0.w_down_shared.is_none());

    let _ = std::fs::remove_file(&path);
}

#[test]
fn moe_weights_from_gguf_binds_shared_expert_tensors_on_deepseek_shape() {
    let path = synth_path("phase2a-deepseek-shared");
    write_synthetic_llama_gguf(
        &path,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 256,
                n_experts_used: 8,
                n_experts_shared: 1,
            }),
            ..SynthLlama::default()
        },
    );
    let gguf = Gguf::open(&path).expect("open");
    let cfg = LlamaConfig::from_gguf(&gguf).expect("config");
    let weights = LlamaWeights::from_gguf(&gguf, &cfg).expect("load MoE weights");
    let mbs = weights.moe_blocks.as_ref().expect("moe_blocks populated");
    let blk0 = &mbs[0];
    assert!(
        blk0.w_gate_shared.is_some(),
        "DeepSeek-V3-style shared experts: gate_shared must be bound"
    );
    assert!(blk0.w_up_shared.is_some());
    assert!(blk0.w_down_shared.is_some());
    let _ = std::fs::remove_file(&path);
}

#[test]
fn moe_load_pre_computes_per_expert_tensor_handles_phase_2d() {
    // Phase 2-D: `LlamaWeights::from_gguf` MoE branch slices
    // per-expert tensor views from the 3D parent at load time
    // (each view shares storage via Storage::CpuOwnedSlice).
    // Verify the vectors are populated with exactly n_experts
    // entries per layer.
    let path = synth_path("phase2d-per-expert-handles");
    let n_experts = 4;
    write_synthetic_llama_gguf(
        &path,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts,
                n_experts_used: 2,
                n_experts_shared: 0,
            }),
            ..SynthLlama::default()
        },
    );
    let gguf = Gguf::open(&path).unwrap();
    let cfg = LlamaConfig::from_gguf(&gguf).unwrap();
    let weights = LlamaWeights::from_gguf(&gguf, &cfg).unwrap();
    let mbs = weights.moe_blocks.as_ref().expect("moe_blocks bound");
    for (i, blk) in mbs.iter().enumerate() {
        assert_eq!(
            blk.gate_per_expert.len(),
            n_experts as usize,
            "layer {i}: gate_per_expert must have one entry per expert"
        );
        assert_eq!(blk.up_per_expert.len(), n_experts as usize);
        assert_eq!(blk.down_per_expert.len(), n_experts as usize);
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn dense_weights_from_gguf_has_no_moe_blocks() {
    // Dense models must continue to populate `blocks` (not
    // `moe_blocks`) — pin the mutual-exclusivity invariant so a
    // future loader refactor can't accidentally populate both.
    let path = synth_path("phase2a-dense-no-moe");
    write_synthetic_llama_gguf(&path, &SynthLlama::default());
    let gguf = Gguf::open(&path).unwrap();
    let cfg = LlamaConfig::from_gguf(&gguf).unwrap();
    let weights = LlamaWeights::from_gguf(&gguf, &cfg).expect("dense binds");
    assert!(!weights.is_moe(), "dense model must not report is_moe");
    assert!(weights.moe_blocks.is_none());
    assert!(!weights.blocks.is_empty(), "dense `blocks` must be populated");
    let _ = std::fs::remove_file(&path);
}
