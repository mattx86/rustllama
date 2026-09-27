//! MoE phase 2-B-2 contract: `LlamaModel::forward_one` produces
//! finite logits for a MoE GGUF when constructed via the
//! `load_allow_moe` bypass.
//!
//! The engine boundary (`LlamaModel::load`) still rejects MoE
//! pending prefill support in phase 2-C — but the decode-mode
//! forward path now actually runs, so a future MoE-aware
//! generation driver could already use it for single-token
//! produce-one-token-at-a-time generation.

use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama, SynthMoe};
use rustllama_gguf::Gguf;
use rustllama_models::llama_arch::{KvCache, KvDtype, LlamaModel};
use rustllama_models::llama_config::LlamaConfig;

fn synth_path(tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("rustllama-moe-fwd-{tag}.gguf"))
}

#[test]
fn forward_one_on_mixtral_shape_produces_finite_logits() {
    let path = synth_path("mixtral-fwd");
    write_synthetic_llama_gguf(
        &path,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 4,
                n_experts_used: 2,
                n_experts_shared: 0,
            }),
            ..SynthLlama::default()
        },
    );
    let gguf = Gguf::open(&path).expect("open");
    let cfg = LlamaConfig::from_gguf(&gguf).expect("config");
    let model = LlamaModel::load_allow_moe(&gguf)
        .expect("phase 2-B: MoE load_allow_moe must succeed");
    assert!(
        model.weights.is_moe(),
        "model must report is_moe after load_allow_moe on a MoE GGUF"
    );

    // Allocate a KV cache. The MoE path uses the standard
    // contiguous F32 backend — same code as dense.
    let max_ctx = cfg.ctx_train;
    let mut kv = KvCache::new_with_dtype(&cfg, max_ctx, KvDtype::F32);
    let mut logits = vec![0f32; cfg.vocab_size];

    // Feed token 0 at position 0. Synth weights are RNG-seeded so
    // we can't pin specific logit values, but the contract is:
    // every logit is finite (the routing math + per-expert SwiGLU
    // didn't NaN-out the activations).
    model.forward_one(0i32, 0u32, &mut kv, &mut logits);

    for (i, v) in logits.iter().enumerate() {
        assert!(
            v.is_finite(),
            "logit[{i}] = {v} is not finite (MoE forward NaN'd)"
        );
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn forward_one_on_deepseek_shape_with_shared_experts_is_finite() {
    // Add a shared-expert FFN on top of the routed top-K. The
    // shared expert is always-active and contributes
    // unconditionally to the output — verify the math doesn't
    // diverge / NaN even with two FFN contributions per layer.
    let path = synth_path("deepseek-shared-fwd");
    write_synthetic_llama_gguf(
        &path,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 4,
                n_experts_used: 2,
                n_experts_shared: 1,
            }),
            ..SynthLlama::default()
        },
    );
    let gguf = Gguf::open(&path).expect("open");
    let cfg = LlamaConfig::from_gguf(&gguf).expect("config");
    let model = LlamaModel::load_allow_moe(&gguf).expect("load");
    let mut kv = KvCache::new_with_dtype(&cfg, cfg.ctx_train, KvDtype::F32);
    let mut logits = vec![0f32; cfg.vocab_size];
    model.forward_one(0i32, 0u32, &mut kv, &mut logits);
    for (i, v) in logits.iter().enumerate() {
        assert!(v.is_finite(), "logit[{i}] = {v} (shared-expert MoE NaN'd)");
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn forward_one_on_dense_still_works_after_moe_refactor() {
    // Regression: the phase 2-B-2 refactor introduced an
    // AttnBlock trait + per-layer Vec<&dyn AttnBlock> indirection.
    // Pin that the dense forward path still produces finite logits.
    let path = synth_path("dense-regression");
    write_synthetic_llama_gguf(&path, &SynthLlama::default());
    let gguf = Gguf::open(&path).unwrap();
    let cfg = LlamaConfig::from_gguf(&gguf).unwrap();
    let model = LlamaModel::load(&gguf).expect("dense load");
    let mut kv = KvCache::new_with_dtype(&cfg, cfg.ctx_train, KvDtype::F32);
    let mut logits = vec![0f32; cfg.vocab_size];
    model.forward_one(0i32, 0u32, &mut kv, &mut logits);
    for v in &logits {
        assert!(v.is_finite());
    }
    let _ = std::fs::remove_file(&path);
}

#[test]
fn forward_prefill_batched_f32_moe_matches_serial_forward_one() {
    // Phase 2-D-4 contract: the batched F32 prefill path now
    // handles MoE (attention batched via AttnBlock trait; FFN
    // per-token through moe_ffn_one_into). Pin parity against the
    // serial forward_one loop on the same MoE GGUF — both paths
    // see the same router + per-expert SwiGLU math, only the
    // attention reduction order differs.
    let path = synth_path("phase2d-batched-parity");
    write_synthetic_llama_gguf(
        &path,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 4,
                n_experts_used: 2,
                n_experts_shared: 0,
            }),
            ..SynthLlama::default()
        },
    );
    let gguf = Gguf::open(&path).expect("open");
    let cfg = LlamaConfig::from_gguf(&gguf).expect("config");
    let model = LlamaModel::load(&gguf).expect("MoE load (phase 2-C public entry)");
    assert!(model.weights.is_moe());

    let tokens = [3i32, 5, 7, 11, 13, 17];

    // Serial baseline: forward_one per token into a fresh KV cache.
    let mut kv_serial = KvCache::new_with_dtype(&cfg, 32, KvDtype::F32);
    let mut logits_serial = vec![0f32; cfg.vocab_size];
    for (i, &tok) in tokens.iter().enumerate() {
        model.forward_one(tok, i as u32, &mut kv_serial, &mut logits_serial);
    }

    // Batched path — call directly to avoid env-var races.
    let mut kv_batched = KvCache::new_with_dtype(&cfg, 32, KvDtype::F32);
    let logits_batched = model.forward_prefill_batched_f32(&tokens, 0, &mut kv_batched);

    assert_eq!(logits_batched.len(), logits_serial.len());
    let mut max_err = 0f32;
    for (b, s) in logits_batched.iter().zip(logits_serial.iter()) {
        assert!(b.is_finite() && s.is_finite(), "non-finite logit");
        let e = (b - s).abs();
        if e > max_err {
            max_err = e;
        }
    }
    assert!(
        max_err < 1e-4,
        "MoE batched vs serial prefill max abs error {max_err} > 1e-4"
    );
    assert_eq!(kv_batched.seq_len, kv_serial.seq_len);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn forward_prefill_batched_q8_0_moe_matches_serial_forward_one() {
    // Phase 2-D-5: same MoE parity check as the F32 variant, but
    // through the Q8_0 KV path. Tighter error bound than TQ/NVFP4
    // because Q8_0 quantization noise is small.
    let path = synth_path("phase2d5-q8-batched-moe");
    write_synthetic_llama_gguf(
        &path,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 4,
                n_experts_used: 2,
                n_experts_shared: 0,
            }),
            ..SynthLlama::default()
        },
    );
    let gguf = Gguf::open(&path).expect("open");
    let cfg = LlamaConfig::from_gguf(&gguf).expect("config");
    let model = LlamaModel::load(&gguf).expect("MoE load");

    let tokens = [3i32, 5, 7, 11, 13, 17];

    let mut kv_serial = KvCache::new_with_dtype(&cfg, 32, KvDtype::Q8_0);
    let mut logits_serial = vec![0f32; cfg.vocab_size];
    for (i, &tok) in tokens.iter().enumerate() {
        model.forward_one(tok, i as u32, &mut kv_serial, &mut logits_serial);
    }
    let mut kv_batched = KvCache::new_with_dtype(&cfg, 32, KvDtype::Q8_0);
    let logits_batched = model.forward_prefill_batched_q8_0(&tokens, 0, &mut kv_batched);

    let mut max_err = 0f32;
    for (b, s) in logits_batched.iter().zip(logits_serial.iter()) {
        assert!(b.is_finite() && s.is_finite());
        max_err = max_err.max((b - s).abs());
    }
    assert!(
        max_err < 1e-3,
        "Q8_0 MoE batched vs serial max err {max_err} > 1e-3"
    );
    assert_eq!(kv_batched.seq_len, kv_serial.seq_len);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn forward_prefill_batched_tq_moe_matches_serial_forward_one() {
    let path = synth_path("phase2d5-tq-batched-moe");
    write_synthetic_llama_gguf(
        &path,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 4,
                n_experts_used: 2,
                n_experts_shared: 0,
            }),
            ..SynthLlama::default()
        },
    );
    let gguf = Gguf::open(&path).expect("open");
    let cfg = LlamaConfig::from_gguf(&gguf).expect("config");
    let model = LlamaModel::load(&gguf).expect("MoE load");

    let tokens = [3i32, 5, 7, 11, 13, 17];

    let mut kv_serial = KvCache::new_with_dtype(&cfg, 32, KvDtype::Tq(4));
    let mut logits_serial = vec![0f32; cfg.vocab_size];
    for (i, &tok) in tokens.iter().enumerate() {
        model.forward_one(tok, i as u32, &mut kv_serial, &mut logits_serial);
    }
    let mut kv_batched = KvCache::new_with_dtype(&cfg, 32, KvDtype::Tq(4));
    let logits_batched = model.forward_prefill_batched_tq(&tokens, 0, &mut kv_batched);

    let mut max_err = 0f32;
    for (b, s) in logits_batched.iter().zip(logits_serial.iter()) {
        assert!(b.is_finite() && s.is_finite());
        max_err = max_err.max((b - s).abs());
    }
    // TQ4 quant noise dominates here — same bound as the dense TQ
    // parity test (1e-2) since both paths see identical TQ noise.
    assert!(
        max_err < 1e-2,
        "TQ MoE batched vs serial max err {max_err} > 1e-2"
    );
    assert_eq!(kv_batched.seq_len, kv_serial.seq_len);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn forward_prefill_batched_nvfp4_moe_matches_serial_forward_one() {
    let path = synth_path("phase2d5-nvfp4-batched-moe");
    write_synthetic_llama_gguf(
        &path,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 4,
                n_experts_used: 2,
                n_experts_shared: 0,
            }),
            ..SynthLlama::default()
        },
    );
    let gguf = Gguf::open(&path).expect("open");
    let cfg = LlamaConfig::from_gguf(&gguf).expect("config");
    let model = LlamaModel::load(&gguf).expect("MoE load");
    assert_eq!(cfg.head_dim % 16, 0, "NVFP4 needs head_dim % 16 == 0");

    let tokens = [3i32, 5, 7, 11, 13, 17];

    let mut kv_serial = KvCache::new_with_dtype(&cfg, 32, KvDtype::Nvfp4);
    let mut logits_serial = vec![0f32; cfg.vocab_size];
    for (i, &tok) in tokens.iter().enumerate() {
        model.forward_one(tok, i as u32, &mut kv_serial, &mut logits_serial);
    }
    let mut kv_batched = KvCache::new_with_dtype(&cfg, 32, KvDtype::Nvfp4);
    let logits_batched = model.forward_prefill_batched_nvfp4(&tokens, 0, &mut kv_batched);

    let mut max_err = 0f32;
    for (b, s) in logits_batched.iter().zip(logits_serial.iter()) {
        assert!(b.is_finite() && s.is_finite());
        max_err = max_err.max((b - s).abs());
    }
    assert!(
        max_err < 3e-3,
        "NVFP4 MoE batched vs serial max err {max_err} > 3e-3"
    );
    assert_eq!(kv_batched.seq_len, kv_serial.seq_len);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn forward_prefill_batched_f32_moe_with_shared_experts_matches_serial() {
    // DeepSeek-V3-style shared expert: always-active FFN added on
    // top of the routed top-K. Verify the batched path's per-token
    // moe_ffn_one_into call handles the shared-expert addition the
    // same way the serial forward_one does.
    let path = synth_path("phase2d-batched-shared-parity");
    write_synthetic_llama_gguf(
        &path,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 4,
                n_experts_used: 2,
                n_experts_shared: 1,
            }),
            ..SynthLlama::default()
        },
    );
    let gguf = Gguf::open(&path).expect("open");
    let cfg = LlamaConfig::from_gguf(&gguf).expect("config");
    let model = LlamaModel::load(&gguf).expect("load");

    let tokens = [2i32, 4, 6, 8];

    let mut kv_serial = KvCache::new_with_dtype(&cfg, 16, KvDtype::F32);
    let mut logits_serial = vec![0f32; cfg.vocab_size];
    for (i, &tok) in tokens.iter().enumerate() {
        model.forward_one(tok, i as u32, &mut kv_serial, &mut logits_serial);
    }
    let mut kv_batched = KvCache::new_with_dtype(&cfg, 16, KvDtype::F32);
    let logits_batched = model.forward_prefill_batched_f32(&tokens, 0, &mut kv_batched);

    let mut max_err = 0f32;
    for (b, s) in logits_batched.iter().zip(logits_serial.iter()) {
        assert!(b.is_finite() && s.is_finite());
        max_err = max_err.max((b - s).abs());
    }
    assert!(
        max_err < 1e-4,
        "shared-expert batched vs serial max err {max_err} > 1e-4"
    );
    let _ = std::fs::remove_file(&path);
}

/// MoE parity through the paged KV cache: `forward_one_paged_f32`
/// matches `forward_one` (contiguous KV) bit-for-bit on a MoE GGUF.
/// Phase 2-D-6 wired MoE through both paged paths so continuous-
/// batching servers can serve MoE models.
#[test]
fn forward_one_paged_f32_moe_matches_contiguous_forward_one() {
    use rustllama_models::page_table::PageTable;
    use rustllama_models::paged_kv_cache::PagedKvCache;
    use rustllama_models::paged_kv_store::PagedKvStore;

    let path = synth_path("phase2d6-paged-moe-forward-one");
    write_synthetic_llama_gguf(
        &path,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 4,
                n_experts_used: 2,
                n_experts_shared: 0,
            }),
            ..SynthLlama::default()
        },
    );
    let gguf = Gguf::open(&path).expect("open");
    let cfg = LlamaConfig::from_gguf(&gguf).expect("config");
    let model = LlamaModel::load(&gguf).expect("MoE load");
    let tokens: Vec<i32> = vec![3, 5, 7, 11, 13];
    let n_prompt = tokens.len();

    // Contiguous reference.
    let mut contig_kv = KvCache::new_with_dtype(&cfg, 16, KvDtype::F32);
    let mut contig_logits = vec![0f32; cfg.vocab_size];
    for (i, &t) in tokens.iter().enumerate() {
        model.forward_one(t, i as u32, &mut contig_kv, &mut contig_logits);
    }

    // Paged path.
    let page_size = 4u32;
    let pages_needed = (n_prompt as u32).div_ceil(page_size);
    let mut store = PagedKvStore::new(
        pages_needed,
        cfg.n_layers as u32,
        cfg.n_kv_heads as u32,
        page_size,
        cfg.head_dim as u32,
    )
    .expect("paged store");
    let mut table = PageTable::new(pages_needed, page_size);
    let mut paged = PagedKvCache::new_for(&store);
    paged.ensure_capacity(&mut table, n_prompt as u32).expect("capacity");
    let mut paged_logits = vec![0f32; cfg.vocab_size];
    for (i, &t) in tokens.iter().enumerate() {
        model.forward_one_paged_f32(t, i as u32, &mut paged, &mut store, &mut paged_logits);
    }

    // Same matmuls + same MoE routing math → bit-identical. (Paged
    // gathers reconstruct the same K/V slab the contiguous path
    // indexes into directly.)
    let mut max_err = 0f32;
    for (a, b) in contig_logits.iter().zip(paged_logits.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    assert!(
        max_err < 1e-5,
        "MoE paged vs contiguous max err {max_err} > 1e-5"
    );
    let _ = std::fs::remove_file(&path);
}

/// MoE parity through the multi-slot batched paged decode:
/// `forward_decode_paged_batched_f32` with 2 slots matches 2 independent
/// `forward_one_paged_f32` calls. Critical for continuous-batching
/// MoE serving — proves slot state doesn't cross-contaminate through
/// the batched matmuls + per-slot MoE routing.
#[test]
fn forward_decode_paged_batched_f32_moe_matches_serial_paged() {
    use rustllama_models::llama_arch::DecodeSlot;
    use rustllama_models::page_table::PageTable;
    use rustllama_models::paged_kv_cache::PagedKvCache;
    use rustllama_models::paged_kv_store::PagedKvStore;
    use rustllama_models::shared_paged_kv::SharedPagedKv;

    let path = synth_path("phase2d6-paged-batched-moe");
    write_synthetic_llama_gguf(
        &path,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 4,
                n_experts_used: 2,
                n_experts_shared: 0,
            }),
            ..SynthLlama::default()
        },
    );
    let gguf = Gguf::open(&path).expect("open");
    let cfg = LlamaConfig::from_gguf(&gguf).expect("config");
    let model = LlamaModel::load(&gguf).expect("MoE load");
    let prompt_a: Vec<i32> = vec![5, 1, 4, 2];
    let prompt_b: Vec<i32> = vec![8, 0, 3, 7];
    let n_prompt = prompt_a.len();
    let total_capacity = (n_prompt + 1) as u32;
    let page_size = 4u32;
    let pages_per_cache = total_capacity.div_ceil(page_size);

    // Reference: two independent paged forwards.
    let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(2);
    for prompt in [&prompt_a, &prompt_b] {
        let mut store = PagedKvStore::new(
            pages_per_cache,
            cfg.n_layers as u32,
            cfg.n_kv_heads as u32,
            page_size,
            cfg.head_dim as u32,
        )
        .expect("ref store");
        let mut table = PageTable::new(pages_per_cache, page_size);
        let mut cache = PagedKvCache::new_for(&store);
        cache.ensure_capacity(&mut table, total_capacity).expect("capacity");
        for (i, &t) in prompt.iter().enumerate() {
            let mut tmp = vec![0f32; cfg.vocab_size];
            model.forward_one_paged_f32(t, i as u32, &mut cache, &mut store, &mut tmp);
        }
        let mut logits = vec![0f32; cfg.vocab_size];
        model.forward_one_paged_f32(
            prompt[n_prompt - 1],
            n_prompt as u32,
            &mut cache,
            &mut store,
            &mut logits,
        );
        ref_logits.push(logits);
    }

    // Fused: shared store, 2 slots, one batched decode call.
    let total_pages_shared = pages_per_cache * 2;
    let store = PagedKvStore::new(
        total_pages_shared,
        cfg.n_layers as u32,
        cfg.n_kv_heads as u32,
        page_size,
        cfg.head_dim as u32,
    )
    .expect("shared store");
    let table = PageTable::new(total_pages_shared, page_size);
    let shared = SharedPagedKv::new(store, table);

    // Throwaway store just for the geometry borrow PagedKvCache::new_for needs.
    let geom_store = PagedKvStore::new(
        total_pages_shared,
        cfg.n_layers as u32,
        cfg.n_kv_heads as u32,
        page_size,
        cfg.head_dim as u32,
    )
    .unwrap();
    let mut cache_a = PagedKvCache::new_for(&geom_store);
    let mut cache_b = PagedKvCache::new_for(&geom_store);
    cache_a.ensure_capacity_shared(&shared, total_capacity).expect("a");
    cache_b.ensure_capacity_shared(&shared, total_capacity).expect("b");

    // Prefill both slots through the shared store.
    for (prompt, cache) in [(&prompt_a, &mut cache_a), (&prompt_b, &mut cache_b)] {
        shared.with_store_mut(|store| {
            for (i, &t) in prompt.iter().enumerate() {
                let mut tmp = vec![0f32; cfg.vocab_size];
                model.forward_one_paged_f32(t, i as u32, cache, store, &mut tmp);
            }
        });
    }

    let mut logits_a = vec![0f32; cfg.vocab_size];
    let mut logits_b = vec![0f32; cfg.vocab_size];
    {
        let mut slots = vec![
            DecodeSlot {
                token_id: prompt_a[n_prompt - 1],
                pos: n_prompt as u32,
                cache: &mut cache_a,
                logits_out: &mut logits_a,
            },
            DecodeSlot {
                token_id: prompt_b[n_prompt - 1],
                pos: n_prompt as u32,
                cache: &mut cache_b,
                logits_out: &mut logits_b,
            },
        ];
        model.forward_decode_paged_batched_f32(&mut slots, &shared);
        drop(slots);
    }

    let mut max_err = 0f32;
    for (a, b) in ref_logits[0].iter().zip(logits_a.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    for (a, b) in ref_logits[1].iter().zip(logits_b.iter()) {
        max_err = max_err.max((a - b).abs());
    }
    assert!(
        max_err < 1e-5,
        "MoE paged-batched vs serial-paged max err {max_err} > 1e-5"
    );
    let _ = std::fs::remove_file(&path);
}

#[test]
fn forward_one_moe_two_positions_advances_kv_cache_seq_len() {
    // The MoE FFN path doesn't touch KV cache (only attention
    // does), but the loop body still advances seq_len. Pin that
    // contract — two sequential forward_one calls must report
    // increasing seq_len.
    let path = synth_path("kv-advance");
    write_synthetic_llama_gguf(
        &path,
        &SynthLlama {
            moe: Some(SynthMoe {
                n_experts: 4,
                n_experts_used: 1,
                n_experts_shared: 0,
            }),
            ..SynthLlama::default()
        },
    );
    let gguf = Gguf::open(&path).expect("open");
    let cfg = LlamaConfig::from_gguf(&gguf).expect("config");
    let model = LlamaModel::load_allow_moe(&gguf).unwrap();
    let mut kv = KvCache::new_with_dtype(&cfg, cfg.ctx_train, KvDtype::F32);
    let mut logits = vec![0f32; cfg.vocab_size];
    model.forward_one(0i32, 0u32, &mut kv, &mut logits);
    assert_eq!(kv.seq_len, 1, "after first forward_one, seq_len = 1");
    model.forward_one(1i32, 1u32, &mut kv, &mut logits);
    assert_eq!(kv.seq_len, 2, "after second forward_one, seq_len = 2");
    for v in &logits {
        assert!(v.is_finite());
    }
    let _ = std::fs::remove_file(&path);
}
