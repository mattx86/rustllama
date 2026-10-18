use super::*;

/// A4 anchor: the load-time fused QKV weight (`concat_qkv_f32`)
/// dispatched as ONE matvec must be byte-identical to three separate
/// `matvec_tensor_dispatch` calls on a GQA shape (where Q's output
/// width differs from K/V's — the case the old `cols`-match gate
/// wrongly rejected). Also exercises the DimOrder accessors end to
/// end: GgmlNe source weights `[in, out]` fused into a RowMajor
/// `[sum_out, d_model]`, both dispatched without tripping C1b.
#[test]
fn fused_qkv_matches_three_separate_matvecs_on_gqa_shape() {
    // Build a GGUF-shaped F32 weight: shape == [in, out] (GgmlNe),
    // physical data row-major [out, in] — exactly `from_gguf`'s output.
    fn gguf_f32(name: &str, in_dim: usize, out_dim: usize, seed: u32) -> Tensor {
        let mut data = vec![0f32; out_dim * in_dim];
        for o in 0..out_dim {
            for i in 0..in_dim {
                let x = (seed as u64)
                    .wrapping_mul(1_000_003)
                    .wrapping_add((o * in_dim + i) as u64);
                data[o * in_dim + i] = (((x ^ (x >> 13)) & 0x3ff) as f32) / 1024.0 - 0.5;
            }
        }
        let mut t = Tensor::from_vec_f32(name, vec![in_dim as u64, out_dim as u64], data);
        t.dim_order = rustllama_tensor::DimOrder::GgmlNe;
        t
    }

    // GQA: 4 query heads, 2 kv heads, head_dim 8 → d_q=32, d_kv=16,
    // and a non-square d_model so a silent axis swap would diverge.
    let d_model = 24usize;
    let d_q = 32usize;
    let d_kv = 16usize;

    let w_q = gguf_f32("blk.0.attn_q.weight", d_model, d_q, 1);
    let w_k = gguf_f32("blk.0.attn_k.weight", d_model, d_kv, 2);
    let w_v = gguf_f32("blk.0.attn_v.weight", d_model, d_kv, 3);

    // The fused weight must build (the gate compares in_dim == d_model
    // across q/k/v; GQA output widths differ but must NOT block it).
    let fused = concat_qkv_f32(&w_q, &w_k, &w_v).expect("GQA QKV must fuse");
    assert_eq!(fused.dim_order, rustllama_tensor::DimOrder::RowMajor);
    assert_eq!(fused.out_dim(), d_q + 2 * d_kv);
    assert_eq!(fused.in_dim(), d_model);

    let h: Vec<f32> = (0..d_model).map(|i| (i as f32) * 0.03 - 0.4).collect();

    // Three separate matvecs (the fall-through path).
    let mut q_out = vec![0f32; d_q];
    let mut k_out = vec![0f32; d_kv];
    let mut v_out = vec![0f32; d_kv];
    matvec_tensor_dispatch(&w_q, &h, &mut q_out, d_q, d_model);
    matvec_tensor_dispatch(&w_k, &h, &mut k_out, d_kv, d_model);
    matvec_tensor_dispatch(&w_v, &h, &mut v_out, d_kv, d_model);

    // One fused matvec over the stacked rows.
    let mut qkv_out = vec![0f32; d_q + 2 * d_kv];
    matvec_tensor_dispatch(&fused, &h, &mut qkv_out, d_q + 2 * d_kv, d_model);

    // Byte-identical, slice by slice.
    let bits = |s: &[f32]| s.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&qkv_out[..d_q]), bits(&q_out), "Q slice differs");
    assert_eq!(
        bits(&qkv_out[d_q..d_q + d_kv]),
        bits(&k_out),
        "K slice differs"
    );
    assert_eq!(
        bits(&qkv_out[d_q + d_kv..]),
        bits(&v_out),
        "V slice differs"
    );
}

/// Diagnostic (V-4): on a REAL hybrid GGUF named by
/// RUSTLLAMA_REAL_HYBRID_GGUF, the from-embeds hybrid prefill fed
/// `embed_tokens_primal` rows must reproduce the token prefill's
/// logits BITWISE — pinning the primal-basis contract that the
/// VLM splice path relies on (Hadamard inverse applied exactly
/// once).
#[test]
#[ignore]
fn hybrid_from_embeds_matches_token_prefill_on_real_model() {
    let Ok(path) = std::env::var("RUSTLLAMA_REAL_HYBRID_GGUF") else {
        eprintln!("RUSTLLAMA_REAL_HYBRID_GGUF not set — skipping");
        return;
    };
    let gguf = Gguf::open(&path).expect("open gguf");
    let cfg = crate::llama_config::LlamaConfig::from_gguf(&gguf).expect("config");
    assert!(cfg.hybrid.is_some(), "fixture must be a hybrid model");
    let weights = LlamaWeights::from_gguf(&gguf, &cfg).expect("weights");
    let model = LlamaModel::from_parts(cfg, weights);
    let hybrid = model.weights.hybrid_layers.as_ref().expect("hybrid layers");
    let tokens: Vec<i32> = vec![3, 1000, 25, 4096, 77];

    let mut kv_a = KvCache::new_with_dtype(&model.cfg, 64, KvDtype::F32);
    let mut dn_a = DeltaNetCache::new_for_hybrid(&model.cfg, hybrid).expect("dn");
    let logits_a = model.forward_prefill_hybrid(&tokens, 0, &mut kv_a, &mut dn_a);

    let mut kv_b = KvCache::new_with_dtype(&model.cfg, 64, KvDtype::F32);
    let mut dn_b = DeltaNetCache::new_for_hybrid(&model.cfg, hybrid).expect("dn");
    let rows = model.embed_tokens_primal(&tokens);
    let logits_b = model.forward_prefill_hybrid_from_embeds(&rows, 0, &mut kv_b, &mut dn_b);

    assert_eq!(
        logits_a, logits_b,
        "from-embeds prefill must be bitwise-identical to token prefill"
    );
    eprintln!("hybrid from-embeds parity OK ({} logits)", logits_a.len());
}

fn synth_cfg(n_layers: usize, n_kv_heads: usize, head_dim: usize) -> LlamaConfig {
    LlamaConfig {
        arch: "llama".into(),
        n_layers,
        n_heads: n_kv_heads,
        n_kv_heads,
        d_model: head_dim * n_kv_heads,
        d_ff: head_dim * n_kv_heads * 2,
        head_dim,
        rope_dim: head_dim,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
        vocab_size: 256,
        ctx_train: 32,
        bos_token_id: None,
        eos_token_id: None,
        tie_word_embeddings: false,
        n_mtp_heads: 0,
        moe: None,
        hybrid: None,
        hadamard: None,
        ..Default::default()
    }
}

#[test]
fn q8_0_storage_is_roughly_one_quarter_of_f32() {
    // 4 layers × 4 kv-heads × 32 ctx × 64 head_dim = 32 768 entries
    // F32: 32k * 4 bytes * 2 (K+V) = 256 KiB per layer × 4 = 1 MiB
    // Q8_0: 32k * 1 byte (q) + 4k * 4 bytes (scales) = 32k+16k = 48 KiB,
    //   times 2 (K+V) = 96 KiB per layer × 4 = 384 KiB.
    // Ratio: 384/1024 = 0.375 — a bit over a quarter due to the
    // per-row scale overhead. Still a 2.6x reduction in storage.
    let cfg = synth_cfg(4, 4, 64);
    let max_ctx = 32;

    let f32_cache = KvCache::new(&cfg, max_ctx);
    let q8_cache = KvCache::new_with_dtype(&cfg, max_ctx, KvDtype::Q8_0);

    let f32_bytes: usize = f32_cache
        .layers
        .iter()
        .map(|l| match l {
            KvLayer::F32 { k, v } => k.len() * 4 + v.len() * 4,
            _ => unreachable!(),
        })
        .sum();
    let q8_bytes: usize = q8_cache
        .layers
        .iter()
        .map(|l| match l {
            KvLayer::Q8_0 {
                k_q,
                k_scales,
                v_q,
                v_scales,
            } => k_q.len() + k_scales.len() * 4 + v_q.len() + v_scales.len() * 4,
            _ => unreachable!(),
        })
        .sum();

    // For head_dim=64, expected ratio is ~(1 + 4/64) / 4 = 0.266.
    let ratio = q8_bytes as f64 / f32_bytes as f64;
    assert!(
        ratio < 0.30,
        "Q8_0 should be < 30% of F32 size for head_dim=64, got ratio={ratio:.3}"
    );
    assert!(
        ratio > 0.20,
        "Q8_0 shouldn't be impossibly small; got ratio={ratio:.3}"
    );
}

#[test]
fn quantize_row_q8_0_dequant_round_trips_extremes() {
    let mut q = vec![0i8; 8];
    let row = vec![-3.0, -1.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0];
    let scale = quantize_row_q8_0(&row, &mut q);
    // The absmax in the row is 3.0; scale should be 3/127.
    let expected_scale = 3.0_f32 / 127.0;
    assert!((scale - expected_scale).abs() < 1e-6);
    // Dequantize and confirm every value is within scale/2 of the
    // original. Endpoints (-3, 3) should dequant exactly.
    let dq: Vec<f32> = q.iter().map(|&x| x as f32 * scale).collect();
    assert!(
        (dq[0] - (-3.0)).abs() < 1e-5,
        "expected -3.0, got {}",
        dq[0]
    );
    assert!((dq[7] - 3.0).abs() < 1e-5, "expected 3.0, got {}", dq[7]);
    for (i, (&orig, &deq)) in row.iter().zip(dq.iter()).enumerate() {
        assert!(
            (orig - deq).abs() < scale,
            "i={i}: |{orig} - {deq}| = {} >= scale={scale}",
            (orig - deq).abs()
        );
    }
}

#[test]
fn quantize_row_q8_0_handles_all_zeros() {
    let mut q = vec![99i8; 8];
    let row = vec![0.0; 8];
    let scale = quantize_row_q8_0(&row, &mut q);
    // Scale convention for the all-zero row is 1.0 (any nonzero
    // value would do; the i8 row is all-zeros either way).
    assert_eq!(scale, 1.0);
    assert!(q.iter().all(|&x| x == 0));
}

/// Build a tiny synthetic Llama model with deterministic F32
/// weights. Used by the paged-vs-contiguous parity test below.
/// All values come from a cheap hash so any silently re-ordered
/// or skipped axis would surface as a non-trivial diff.
fn synth_llama_model(cfg: &LlamaConfig) -> LlamaModel {
    use rustllama_tensor::{Shape, Tensor};
    fn fill(n: usize, seed: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (seed as u64).wrapping_mul(1_000_003).wrapping_add(i as u64);
                // Map u64 → small f32 in [-0.5, 0.5] with enough
                // variation per cell to make any silent axis swap
                // produce a different bit pattern.
                let bits = (x ^ (x >> 17)) & 0xFFF;
                (bits as f32 / 4096.0) - 0.5
            })
            .collect()
    }
    fn mat(name: &str, rows: u64, cols: u64, seed: u32) -> Tensor {
        let n = (rows * cols) as usize;
        Tensor::from_vec_f32(name, vec![rows, cols] as Shape, fill(n, seed))
    }
    let d = cfg.d_model as u64;
    let d_ff = cfg.d_ff as u64;
    let d_q = (cfg.n_heads * cfg.head_dim) as u64;
    let d_kv = (cfg.n_kv_heads * cfg.head_dim) as u64;
    let v = cfg.vocab_size as u64;
    let token_embd = mat("token_embd", v, d, 1);
    let output_norm: Vec<f32> = (0..d).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let mut blocks = Vec::with_capacity(cfg.n_layers);
    for i in 0..cfg.n_layers {
        let base = 100 + (i as u32) * 13;
        let attn_norm: Vec<f32> = (0..d)
            .map(|j| 1.0 + ((i as u32 + j as u32) % 7) as f32 * 0.01)
            .collect();
        let ffn_norm: Vec<f32> = (0..d)
            .map(|j| 1.0 + ((i as u32 + j as u32) % 5) as f32 * 0.01)
            .collect();
        blocks.push(LlamaBlockWeights {
            attn_norm,
            w_q: mat("w_q", d_q, d, base),
            w_k: mat("w_k", d_kv, d, base + 1),
            w_v: mat("w_v", d_kv, d, base + 2),
            w_o: mat("w_o", d, d_q, base + 3),
            b_q: None,
            b_k: None,
            b_v: None,
            ffn_norm,
            w_gate: mat("w_gate", d_ff, d, base + 4),
            w_up: mat("w_up", d_ff, d, base + 5),
            w_down: mat("w_down", d, d_ff, base + 6),
            w_qkv_fused: None,
            q_norm: None,
            k_norm: None,
        });
    }
    LlamaModel::from_parts(
        cfg.clone(),
        LlamaWeights {
            token_embd,
            blocks,
            moe_blocks: None,
            hybrid_layers: None,
            mla_blocks: None,
            output_norm,
            output: None, // tie embeddings
            mtp_heads: None,
            nextn_head: None,
            hadamard: None,
        },
    )
}

/// P2 helper: a sliding-window variant of [`synth_cfg`]. `pattern`
/// = `None` makes every layer SWA (Mistral-style); `Some(p)` makes
/// every `p`-th layer a full-attention "global" layer (Gemma2-style).
fn swa_cfg(
    n_layers: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: u32,
    pattern: Option<u32>,
) -> LlamaConfig {
    let mut c = synth_cfg(n_layers, n_kv_heads, head_dim);
    c.sliding_window = Some(window);
    c.sliding_window_pattern = pattern;
    c
}

/// P2 ANCHOR: the bounded SWA ring's decode logits equal the
/// unbounded-but-masked SWA path's logits at every position —
/// including well past the window, where the ring wraps repeatedly —
/// AND the ring actually uses less memory. The reference cache
/// ([`KvCache::new_with_dtype_no_ring`]) keeps full `max_ctx` slabs and
/// masks; the ring cache ([`KvCache::new`]) stores only `window`
/// positions per SWA layer and gathers.
#[test]
fn swa_ring_decode_matches_masked_unbounded() {
    let window = 4u32;
    let max_ctx = 24usize; // decode 6× past the window
    for pattern in [None, Some(2u32)] {
        let cfg = swa_cfg(4, 2, 8, window, pattern);
        let model = synth_llama_model(&cfg);
        let mut ring = KvCache::new(&cfg, max_ctx);
        let mut masked = KvCache::new_with_dtype_no_ring(&cfg, max_ctx, KvDtype::F32);
        assert!(
            ring.has_ring_layers(),
            "SWA model (pattern={pattern:?}) must produce at least one ring layer"
        );
        assert!(
            !masked.has_ring_layers(),
            "no-ring reference must be full-slab"
        );

        let vocab = cfg.vocab_size;
        let toks: Vec<i32> = (0..max_ctx as i32).map(|i| (i * 7 + 3) % 256).collect();
        let mut lr = vec![0f32; vocab];
        let mut lm = vec![0f32; vocab];
        for (pos, &t) in toks.iter().enumerate() {
            model.forward_one(t, pos as u32, &mut ring, &mut lr);
            model.forward_one(t, pos as u32, &mut masked, &mut lm);
            let maxdiff = lr
                .iter()
                .zip(lm.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                maxdiff <= 1e-4,
                "ring vs masked diverged at pos {pos} (pattern={pattern:?}): max|Δ|={maxdiff}"
            );
        }
        assert!(
            ring.approx_bytes() < masked.approx_bytes(),
            "ring must use less KV memory (pattern={pattern:?}): ring={} masked={}",
            ring.approx_bytes(),
            masked.approx_bytes(),
        );
    }
}

/// P2: the ring is per-LAYER — on an alternating (Gemma2-style) model
/// only the SWA layers are bounded; the global full-attention layers
/// keep the full `max_ctx` slab.
#[test]
fn swa_ring_is_per_layer_on_alternating_pattern() {
    let window = 4u32;
    let max_ctx = 24usize;
    // pattern=2 ⇒ layer `(idx+1)%2==0` is GLOBAL: layers 1,3 global; 0,2 SWA.
    let cfg = swa_cfg(4, 2, 8, window, Some(2));
    let kv = KvCache::new(&cfg, max_ctx);
    assert!(kv.is_ring_layer(0) && !kv.is_ring_layer(1));
    assert!(kv.is_ring_layer(2) && !kv.is_ring_layer(3));
    assert_eq!(kv.layer_cap(0), window as usize);
    assert_eq!(kv.layer_cap(1), max_ctx);
}

/// P2: a non-SWA model NEVER rings, and the ring code path leaves its
/// decode byte-for-byte identical (same allocation, same forward).
#[test]
fn non_swa_model_never_rings_and_decode_byte_identical() {
    let max_ctx = 24usize;
    let cfg = synth_cfg(3, 2, 8); // sliding_window = None
    let model = synth_llama_model(&cfg);
    assert!(!KvCache::new(&cfg, max_ctx).has_ring_layers());

    let mut k1 = KvCache::new(&cfg, max_ctx);
    let mut k2 = KvCache::new_with_dtype_no_ring(&cfg, max_ctx, KvDtype::F32);
    assert_eq!(
        k1.approx_bytes(),
        k2.approx_bytes(),
        "non-SWA ring vs no-ring allocation must match exactly"
    );
    let vocab = cfg.vocab_size;
    let toks: Vec<i32> = (0..12i32).map(|i| (i * 5 + 1) % 256).collect();
    let mut l1 = vec![0f32; vocab];
    let mut l2 = vec![0f32; vocab];
    for (pos, &t) in toks.iter().enumerate() {
        model.forward_one(t, pos as u32, &mut k1, &mut l1);
        model.forward_one(t, pos as u32, &mut k2, &mut l2);
        assert_eq!(l1, l2, "non-SWA decode must be byte-identical at pos {pos}");
    }
}

/// P1 ANCHOR: a context shift leaves the KV ≡ a fresh re-prefill of the
/// retained token sequence at the shifted positions, so continuation is
/// coherent. Proven on a 1-LAYER model, where every layer's K/V is a pure
/// function of its own token (layer-0 K/V don't depend on attention
/// history) — so dropping the middle chunk and re-RoPE-ing the suffix
/// reproduces exactly what a re-prefill of the retained tokens would build.
/// Exercised for full-attention AND an all-SWA ring model (the ring layer's
/// re-slot + re-RoPE composes with the shift).
#[test]
fn context_shift_matches_reprefill_of_retained_tokens() {
    let max_ctx = 32usize;
    let n = 20usize;
    let keep = 3usize;
    let discard = 6usize;
    for swa in [None, Some(4u32)] {
        let cfg = match swa {
            Some(w) => swa_cfg(1, 2, 8, w, None),
            None => synth_cfg(1, 2, 8),
        };
        let model = synth_llama_model(&cfg);
        let theta = cfg.rope_theta;
        let vocab = cfg.vocab_size;
        let toks: Vec<i32> = (0..n as i32).map(|i| (i * 11 + 5) % 256).collect();
        let probe = 123i32;

        // SHIFTED: prefill all, context-shift, probe at the new position.
        let mut kv = KvCache::new(&cfg, max_ctx);
        let mut scratch = vec![0f32; vocab];
        for (p, &t) in toks.iter().enumerate() {
            model.forward_one(t, p as u32, &mut kv, &mut scratch);
        }
        assert!(
            kv.context_shift(keep, discard, theta),
            "context_shift should apply (swa={swa:?})"
        );
        assert_eq!(kv.seq_len, n - discard);
        let mut l_shift = vec![0f32; vocab];
        model.forward_one(probe, (n - discard) as u32, &mut kv, &mut l_shift);

        // REFERENCE: re-prefill the retained tokens, probe at the same pos.
        let mut ret: Vec<i32> = toks[..keep].to_vec();
        ret.extend_from_slice(&toks[keep + discard..]);
        assert_eq!(ret.len(), n - discard);
        let mut kv2 = KvCache::new(&cfg, max_ctx);
        let mut scratch2 = vec![0f32; vocab];
        for (p, &t) in ret.iter().enumerate() {
            model.forward_one(t, p as u32, &mut kv2, &mut scratch2);
        }
        let mut l_ref = vec![0f32; vocab];
        model.forward_one(probe, ret.len() as u32, &mut kv2, &mut l_ref);

        let maxdiff = l_shift
            .iter()
            .zip(l_ref.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            maxdiff < 1e-4,
            "context_shift logits must match a re-prefill (swa={swa:?}): max|Δ|={maxdiff}"
        );
    }
}

/// P1: `context_shift` is fail-safe — it returns `false` WITHOUT mutating
/// the cache for out-of-range parameters (so the engine keeps today's
/// hard-stop), and a ring whose live window would straddle the kept head /
/// dropped gap is refused rather than corrupted.
#[test]
fn context_shift_refuses_bad_params_without_mutating() {
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let mut kv = KvCache::new(&cfg, 32);
    let mut l = vec![0f32; cfg.vocab_size];
    for p in 0..10u32 {
        model.forward_one((p as i32 + 1) % 256, p, &mut kv, &mut l);
    }
    assert!(
        !kv.context_shift(0, 0, cfg.rope_theta),
        "discard=0 must refuse"
    );
    assert!(
        !kv.context_shift(10, 5, cfg.rope_theta),
        "keep>=seq must refuse"
    );
    assert_eq!(kv.seq_len, 10, "a refused shift must not change seq_len");
}

/// Parity probe: `forward_prefill_paged_f32` returns bit-
/// identical logits to `forward_prefill_batched_f32` for the
/// same model + tokens. Proves the gather path's data layout
/// agrees with the contiguous slab the existing attention
/// kernels expect — any axis swap inside `PagedKvCache::write_token`
/// or `gather_layer` would surface as a non-zero diff somewhere
/// in the 256-wide logits vector.
///
/// Runs in mock mode (no SYCL); both paths fall through to the
/// CPU `gqa_attention_flash_prefill`, so the bit-for-bit
/// equality is the strict assertion the test wants.
#[test]
fn forward_prefill_paged_matches_contiguous_for_same_input() {
    // Small model: 2 layers, 2 heads (no GQA grouping for v1
    // test simplicity — covered separately by the page store's
    // gather tests, which exercise n_kv_heads=3 too).
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let tokens: Vec<i32> = vec![1, 4, 17, 9, 33];
    let n_new = tokens.len();
    let max_ctx = 16;

    // Contiguous path — uses the existing F32 KvCache layout.
    let mut contig_kv = KvCache::new(&cfg, max_ctx);
    let logits_contig = model.forward_prefill_batched_f32(&tokens, 0, &mut contig_kv);

    // Paged path — fresh store + cache, grown to fit n_new
    // tokens for one slot.
    let page_size = 4;
    let pages_for_kv = (n_new as u32).div_ceil(page_size);
    let mut store = crate::paged_kv_store::PagedKvStore::new(
        pages_for_kv,
        cfg.n_layers as u32,
        cfg.n_kv_heads as u32,
        page_size,
        cfg.head_dim as u32,
    )
    .expect("paged store");
    let mut table = crate::page_table::PageTable::new(pages_for_kv, page_size);
    let mut paged_cache = crate::paged_kv_cache::PagedKvCache::new_for(&store);
    paged_cache
        .ensure_capacity(&mut table, n_new as u32)
        .expect("ensure_capacity");
    let logits_paged = model.forward_prefill_paged_f32(&tokens, 0, &mut paged_cache, &mut store);

    // Bit-identical: same inputs, same kernels — gather just
    // routes the data through a different storage layout.
    assert_eq!(logits_contig.len(), logits_paged.len(), "logits length");
    for (i, (&a, &b)) in logits_contig.iter().zip(logits_paged.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "logit[{i}] diverges: contiguous={a} paged={b}"
        );
    }
}

/// Parity probe across prefill + multiple decode steps:
/// `forward_one_paged_f32` produces bit-identical logits to
/// `forward_one` (F32 path) at every decode step after a paged
/// prefill. This exercises the full single-request generation
/// flow through paged storage — write+gather correctness
/// across both the batched-prefill code path AND the per-step
/// decode code path, with the cache growing as more tokens
/// land.
#[test]
fn forward_one_paged_matches_contiguous_across_decode_steps() {
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let prompt: Vec<i32> = vec![3, 1, 4, 1, 5];
    let n_prompt = prompt.len();
    let n_decode = 4; // generate 4 more tokens after prefill
    let total = n_prompt + n_decode;
    let max_ctx = 16;
    let mut decode_inputs: Vec<i32> = vec![7, 11, 13, 17];

    // Contiguous reference: prefill + n_decode forward_ones.
    let mut contig_kv = KvCache::new(&cfg, max_ctx);
    let _ = model.forward_prefill_batched_f32(&prompt, 0, &mut contig_kv);
    let mut contig_decode_logits: Vec<Vec<f32>> = Vec::with_capacity(n_decode);
    for (i, &tok) in decode_inputs.iter().enumerate() {
        let mut logits = vec![0f32; cfg.vocab_size];
        model.forward_one(tok, (n_prompt + i) as u32, &mut contig_kv, &mut logits);
        contig_decode_logits.push(logits);
    }

    // Paged path: same model, same inputs.
    let page_size = 4;
    let pages_needed = (total as u32).div_ceil(page_size);
    let mut store = crate::paged_kv_store::PagedKvStore::new(
        pages_needed,
        cfg.n_layers as u32,
        cfg.n_kv_heads as u32,
        page_size,
        cfg.head_dim as u32,
    )
    .expect("paged store");
    let mut table = crate::page_table::PageTable::new(pages_needed, page_size);
    let mut paged_cache = crate::paged_kv_cache::PagedKvCache::new_for(&store);
    paged_cache
        .ensure_capacity(&mut table, total as u32)
        .expect("ensure_capacity");
    let _ = model.forward_prefill_paged_f32(&prompt, 0, &mut paged_cache, &mut store);
    let mut paged_decode_logits: Vec<Vec<f32>> = Vec::with_capacity(n_decode);
    for (i, &tok) in decode_inputs.iter().enumerate() {
        let mut logits = vec![0f32; cfg.vocab_size];
        model.forward_one_paged_f32(
            tok,
            (n_prompt + i) as u32,
            &mut paged_cache,
            &mut store,
            &mut logits,
        );
        paged_decode_logits.push(logits);
    }

    // Bit-identical at every decode step.
    for (step, (a_logits, b_logits)) in contig_decode_logits
        .iter()
        .zip(paged_decode_logits.iter())
        .enumerate()
    {
        for (i, (&a, &b)) in a_logits.iter().zip(b_logits.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "step {step} logit[{i}] diverges: contiguous={a} paged={b}"
            );
        }
    }

    // Borrow checker silencer: make sure we used decode_inputs
    // after the parity assertions so the test's intent is clear.
    decode_inputs.clear();
    assert!(decode_inputs.is_empty());
}

/// Parity probe: one fused M-slot decode call produces bit-
/// identical logits to M independent `forward_one_paged_f32`
/// calls (one per slot). Each slot is independently prefilled
/// with a different prompt so the test exercises the actual
/// "M slots with different KV histories" case, not just the
/// trivial "same prompt M times" case.
///
/// This is the milestone test for Item 3.7c: proves the
/// batched-matmul path doesn't cross-contaminate slot state,
/// and that the per-slot serial attention reads the right
/// slot's KV history from the shared store.
#[test]
fn forward_decode_paged_batched_matches_serial_forward_one_paged() {
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    // Two slots with different prompts so their KV state
    // diverges before the decode step under test.
    let prompt_a: Vec<i32> = vec![5, 1, 4, 2];
    let prompt_b: Vec<i32> = vec![8, 0, 3, 7];
    assert_eq!(
        prompt_a.len(),
        prompt_b.len(),
        "prompts equal length to share scratch"
    );
    let n_prompt = prompt_a.len();
    let total_capacity = (n_prompt + 1) as u32; // prefill + 1 decode step

    // ---- Reference path: two independent single-owner caches ----
    // Each slot prefills + decodes one token through
    // `forward_one_paged_f32` (no batching, no sharing). The
    // logits this produces are the ground truth the fused
    // call must match.
    let page_size = 4;
    let pages_per_cache = total_capacity.div_ceil(page_size);
    let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(2);
    for prompt in [&prompt_a, &prompt_b] {
        let mut store = crate::paged_kv_store::PagedKvStore::new(
            pages_per_cache,
            cfg.n_layers as u32,
            cfg.n_kv_heads as u32,
            page_size,
            cfg.head_dim as u32,
        )
        .expect("ref store");
        let mut table = crate::page_table::PageTable::new(pages_per_cache, page_size);
        let mut cache = crate::paged_kv_cache::PagedKvCache::new_for(&store);
        cache
            .ensure_capacity(&mut table, total_capacity)
            .expect("ensure_capacity");
        let _ = model.forward_prefill_paged_f32(prompt, 0, &mut cache, &mut store);
        // One decode step: the last token of the prompt is
        // the "next input" the sampler would have produced.
        // We use prompt[n_prompt-1] for determinism in the
        // test; the engine's real generate loop samples
        // here.
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

    // ---- Fused path: shared store, two PagedKvCaches, one
    // forward_decode_paged_batched_f32 call ----
    // Build a shared store sized to hold both slots'
    // capacity. `pages_per_cache * 2` ensures the table can
    // satisfy both ensure_capacity_shared calls without
    // contention.
    let total_pages_shared = pages_per_cache * 2;
    let store = crate::paged_kv_store::PagedKvStore::new(
        total_pages_shared,
        cfg.n_layers as u32,
        cfg.n_kv_heads as u32,
        page_size,
        cfg.head_dim as u32,
    )
    .expect("shared store");
    let table = crate::page_table::PageTable::new(total_pages_shared, page_size);
    let shared = crate::shared_paged_kv::SharedPagedKv::new(store, table);

    // Per-slot caches that all point at `shared`. Build a
    // throwaway PagedKvStore just for the geometry borrow
    // `PagedKvCache::new_for` needs — it only reads page_size,
    // so the throwaway is cheap (4 pages × 2 layers × 2 heads
    // × 4 page_size × 8 head_dim = 6 KiB).
    let geom_store = crate::paged_kv_store::PagedKvStore::new(
        total_pages_shared,
        cfg.n_layers as u32,
        cfg.n_kv_heads as u32,
        page_size,
        cfg.head_dim as u32,
    )
    .unwrap();
    let mut cache_a = crate::paged_kv_cache::PagedKvCache::new_for(&geom_store);
    let mut cache_b = crate::paged_kv_cache::PagedKvCache::new_for(&geom_store);
    cache_a
        .ensure_capacity_shared(&shared, total_capacity)
        .expect("ensure A");
    cache_b
        .ensure_capacity_shared(&shared, total_capacity)
        .expect("ensure B");

    // Prefill both slots through the shared store using the
    // shared-access methods directly. `forward_prefill_paged_f32`
    // only takes single-owner `&mut PagedKvStore`, so we
    // bridge by locking the shared store for the entire
    // prefill of each slot. Prefill is per-slot serial here —
    // that's fine, this is test setup, not the path under
    // test.
    for (prompt, cache) in [(&prompt_a, &mut cache_a), (&prompt_b, &mut cache_b)] {
        // The shared store's with_store_mut helper hands us a
        // &mut PagedKvStore for the duration of the closure.
        shared.with_store_mut(|store| {
            let _ = model.forward_prefill_paged_f32(prompt, 0, cache, store);
        });
    }

    // The decode step under test: two slots, one fused call.
    let mut logits_a = vec![0f32; cfg.vocab_size];
    let mut logits_b = vec![0f32; cfg.vocab_size];
    let mut slots = vec![
        crate::llama_arch::DecodeSlot {
            token_id: prompt_a[n_prompt - 1],
            pos: n_prompt as u32,
            cache: &mut cache_a,
            logits_out: &mut logits_a,
        },
        crate::llama_arch::DecodeSlot {
            token_id: prompt_b[n_prompt - 1],
            pos: n_prompt as u32,
            cache: &mut cache_b,
            logits_out: &mut logits_b,
        },
    ];
    model.forward_decode_paged_batched_f32(&mut slots, &shared);
    drop(slots);

    // Bit-identical per-slot.
    let fused_logits = [&logits_a, &logits_b];
    for (slot_idx, (&fused, reference)) in fused_logits.iter().zip(ref_logits.iter()).enumerate() {
        for (i, (&a, &b)) in fused.iter().zip(reference.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "slot {slot_idx} logit[{i}] diverges: fused={a} ref={b}"
            );
        }
    }
}

/// Single-slot edge: `forward_decode_paged_batched_f32` with
/// `m=1` produces bit-identical output to a single
/// `forward_one_paged_f32` call. Catches off-by-one bugs in
/// the batched-scratch sizing (size depends on `m`) that
/// wouldn't show up in the multi-slot case.
#[test]
fn forward_decode_paged_batched_single_slot_matches_forward_one_paged() {
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let prompt: Vec<i32> = vec![3, 14, 1, 5];
    let n_prompt = prompt.len();
    let total_capacity = (n_prompt + 1) as u32;
    let page_size = 4;
    let pages = total_capacity.div_ceil(page_size);

    // Reference via single-owner forward_one_paged_f32.
    let mut ref_store = crate::paged_kv_store::PagedKvStore::new(
        pages,
        cfg.n_layers as u32,
        cfg.n_kv_heads as u32,
        page_size,
        cfg.head_dim as u32,
    )
    .unwrap();
    let mut ref_table = crate::page_table::PageTable::new(pages, page_size);
    let mut ref_cache = crate::paged_kv_cache::PagedKvCache::new_for(&ref_store);
    ref_cache
        .ensure_capacity(&mut ref_table, total_capacity)
        .unwrap();
    let _ = model.forward_prefill_paged_f32(&prompt, 0, &mut ref_cache, &mut ref_store);
    let mut ref_logits = vec![0f32; cfg.vocab_size];
    model.forward_one_paged_f32(
        prompt[n_prompt - 1],
        n_prompt as u32,
        &mut ref_cache,
        &mut ref_store,
        &mut ref_logits,
    );

    // Fused single-slot path.
    let shared_store = crate::paged_kv_store::PagedKvStore::new(
        pages,
        cfg.n_layers as u32,
        cfg.n_kv_heads as u32,
        page_size,
        cfg.head_dim as u32,
    )
    .unwrap();
    let shared_table = crate::page_table::PageTable::new(pages, page_size);
    let shared = crate::shared_paged_kv::SharedPagedKv::new(shared_store, shared_table);
    let geom_store = crate::paged_kv_store::PagedKvStore::new(
        pages,
        cfg.n_layers as u32,
        cfg.n_kv_heads as u32,
        page_size,
        cfg.head_dim as u32,
    )
    .unwrap();
    let mut cache = crate::paged_kv_cache::PagedKvCache::new_for(&geom_store);
    cache
        .ensure_capacity_shared(&shared, total_capacity)
        .unwrap();
    shared.with_store_mut(|store| {
        let _ = model.forward_prefill_paged_f32(&prompt, 0, &mut cache, store);
    });
    let mut fused_logits = vec![0f32; cfg.vocab_size];
    let mut slots = vec![crate::llama_arch::DecodeSlot {
        token_id: prompt[n_prompt - 1],
        pos: n_prompt as u32,
        cache: &mut cache,
        logits_out: &mut fused_logits,
    }];
    model.forward_decode_paged_batched_f32(&mut slots, &shared);
    drop(slots);

    for (i, (&a, &b)) in fused_logits.iter().zip(ref_logits.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "single-slot fused logit[{i}] diverges: fused={a} ref={b}"
        );
    }
}

/// Same parity probe at a non-zero `start_pos`: the paged path
/// must also handle the case where the cache already holds
/// previously-prefilled tokens and the new batch extends it.
/// Catches off-by-one bugs in `kv_len_base` / `seq_len`
/// handling that wouldn't show up in the start_pos=0 case.
#[test]
fn forward_prefill_paged_matches_contiguous_at_nonzero_start_pos() {
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let first_batch: Vec<i32> = vec![5, 11, 2];
    let second_batch: Vec<i32> = vec![19, 7];
    let total = first_batch.len() + second_batch.len();
    let max_ctx = 16;

    let mut contig_kv = KvCache::new(&cfg, max_ctx);
    let _ = model.forward_prefill_batched_f32(&first_batch, 0, &mut contig_kv);
    let logits_contig =
        model.forward_prefill_batched_f32(&second_batch, first_batch.len() as u32, &mut contig_kv);

    let page_size = 4;
    let pages_needed = (total as u32).div_ceil(page_size);
    let mut store = crate::paged_kv_store::PagedKvStore::new(
        pages_needed,
        cfg.n_layers as u32,
        cfg.n_kv_heads as u32,
        page_size,
        cfg.head_dim as u32,
    )
    .expect("paged store");
    let mut table = crate::page_table::PageTable::new(pages_needed, page_size);
    let mut paged_cache = crate::paged_kv_cache::PagedKvCache::new_for(&store);
    paged_cache
        .ensure_capacity(&mut table, total as u32)
        .expect("ensure_capacity");
    let _ = model.forward_prefill_paged_f32(&first_batch, 0, &mut paged_cache, &mut store);
    let logits_paged = model.forward_prefill_paged_f32(
        &second_batch,
        first_batch.len() as u32,
        &mut paged_cache,
        &mut store,
    );

    for (i, (&a, &b)) in logits_contig.iter().zip(logits_paged.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "logit[{i}] diverges at start_pos>0: contiguous={a} paged={b}"
        );
    }
}

/// D1b core anchor: a paged-F32 forward (inline prefill + single-token decode
/// + batched decode) is BIT-IDENTICAL to the contiguous-F32 forward for the
/// same tokens, because both run the same `*_capped_windowed` CPU kernels. At
/// `window=0 / softcap=0` (the overwhelming common case) these delegate to the
/// plain dispatchers, so this is also the "ordinary model is byte-identical"
/// regression guard. With a sliding window / Gemma2 soft-cap set, it pins the
/// new correctness path: paged SWA/softcap ≡ contiguous SWA/softcap.
fn assert_paged_matches_contiguous(cfg: &LlamaConfig) {
    let model = synth_llama_model(cfg);
    // Prompt longer than the test window (3) so the window actually trims.
    let prompt: Vec<i32> = vec![5, 11, 2, 19, 7, 1];
    let n_prompt = prompt.len();
    let last_tok = prompt[n_prompt - 1];
    let max_ctx = 16;

    // --- Contiguous reference (windowed/capped kernels). ---
    // P2: the reference is the FULL-slab masked path (what the paged side
    // computes — full history, windowed) via `new_with_dtype_no_ring`, not
    // the bounded SWA ring. The ring's equivalence to this masked path is
    // covered separately by `swa_ring_decode_matches_masked_unbounded`; the
    // ring also can't run the batched prefill this test calls directly.
    let mut contig_kv = KvCache::new_with_dtype_no_ring(cfg, max_ctx, KvDtype::F32);
    let prefill_contig = model.forward_prefill_batched_f32(&prompt, 0, &mut contig_kv);
    let mut decode_contig = vec![0f32; cfg.vocab_size];
    model.forward_one(
        last_tok,
        n_prompt as u32,
        &mut contig_kv,
        &mut decode_contig,
    );

    // --- Paged (single-owner inline prefill + single-token decode). ---
    let total_cap = (n_prompt + 1) as u32;
    let page_size = 4;
    let pages = total_cap.div_ceil(page_size);
    let mk_store = || {
        crate::paged_kv_store::PagedKvStore::new(
            pages,
            cfg.n_layers as u32,
            cfg.n_kv_heads as u32,
            page_size,
            cfg.head_dim as u32,
        )
        .unwrap()
    };
    let mut store = mk_store();
    let mut table = crate::page_table::PageTable::new(pages, page_size);
    let mut cache = crate::paged_kv_cache::PagedKvCache::new_for(&store);
    cache.ensure_capacity(&mut table, total_cap).unwrap();
    let prefill_paged = model.forward_prefill_paged_f32(&prompt, 0, &mut cache, &mut store);
    let mut decode_paged = vec![0f32; cfg.vocab_size];
    model.forward_one_paged_f32(
        last_tok,
        n_prompt as u32,
        &mut cache,
        &mut store,
        &mut decode_paged,
    );

    // Inline-prefill final-position logits must match bit-for-bit.
    for (i, (&a, &b)) in prefill_contig.iter().zip(prefill_paged.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "prefill logit[{i}]: contiguous={a} paged={b}"
        );
    }
    // Single-token paged decode must match the contiguous decode bit-for-bit.
    for (i, (&a, &b)) in decode_contig.iter().zip(decode_paged.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "decode logit[{i}]: contiguous={a} paged={b}"
        );
    }

    // --- Batched paged decode (single slot) must equal the contiguous decode. ---
    let shared = crate::shared_paged_kv::SharedPagedKv::new(
        mk_store(),
        crate::page_table::PageTable::new(pages, page_size),
    );
    let geom_store = mk_store();
    let mut bcache = crate::paged_kv_cache::PagedKvCache::new_for(&geom_store);
    bcache.ensure_capacity_shared(&shared, total_cap).unwrap();
    shared.with_store_mut(|store| {
        let _ = model.forward_prefill_paged_f32(&prompt, 0, &mut bcache, store);
    });
    let mut decode_batched = vec![0f32; cfg.vocab_size];
    let mut slots = vec![crate::llama_arch::DecodeSlot {
        token_id: last_tok,
        pos: n_prompt as u32,
        cache: &mut bcache,
        logits_out: &mut decode_batched,
    }];
    model.forward_decode_paged_batched_f32(&mut slots, &shared);
    drop(slots);
    for (i, (&a, &b)) in decode_contig.iter().zip(decode_batched.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "batched decode logit[{i}]: contiguous={a} batched={b}"
        );
    }
}

/// The critical regression guard: at `window=0 / softcap=0` (every ordinary
/// model), the D1b change is BYTE-IDENTICAL to the contiguous forward.
#[test]
fn paged_window_zero_byte_identical_to_contiguous() {
    assert_paged_matches_contiguous(&synth_cfg(2, 2, 8));
}

/// Paged SWA decode/prefill ≡ contiguous windowed, all-SWA (Mistral-style).
#[test]
fn paged_swa_all_layers_matches_contiguous_windowed() {
    let mut cfg = synth_cfg(2, 2, 8);
    cfg.sliding_window = Some(3);
    assert_paged_matches_contiguous(&cfg);
}

/// Paged SWA ≡ contiguous with the Gemma2-style alternating pattern (every
/// `p`-th layer global, the rest windowed).
#[test]
fn paged_swa_pattern_matches_contiguous_windowed() {
    let mut cfg = synth_cfg(4, 2, 8);
    cfg.sliding_window = Some(3);
    cfg.sliding_window_pattern = Some(2);
    assert_paged_matches_contiguous(&cfg);
}

/// Paged Gemma2 attention-logit soft-cap ≡ contiguous capped (no window).
#[test]
fn paged_softcap_matches_contiguous_capped() {
    let mut cfg = synth_cfg(2, 2, 8);
    cfg.attn_logit_softcapping = Some(50.0);
    assert_paged_matches_contiguous(&cfg);
}

/// Paged SWA + softcap together (full Gemma2 shape) ≡ contiguous.
#[test]
fn paged_swa_and_softcap_matches_contiguous() {
    let mut cfg = synth_cfg(4, 2, 8);
    cfg.sliding_window = Some(3);
    cfg.sliding_window_pattern = Some(2);
    cfg.attn_logit_softcapping = Some(50.0);
    assert_paged_matches_contiguous(&cfg);
}

/// Hybrid placement parity: the forward output is invariant to
/// `n_gpu_layers` in mock mode. Both `n_gpu_layers=u32::MAX`
/// (all GPU, all USM helpers run their full body) and
/// `n_gpu_layers=0` (all CPU, every helper short-circuits and
/// falls through) must produce bit-identical logits, because
/// in mock mode the GPU helpers also return false (no SYCL
/// device) → they fall through to the same CPU kernels. This
/// pins the contract that the placement gate is a *routing*
/// decision, not a numerical one.
///
/// Real-hardware perf wins from the cutoff land independently
/// — they don't change correctness.
#[test]
fn forward_one_invariant_to_n_gpu_layers_in_mock_mode() {
    let cfg = synth_cfg(3, 2, 8);
    let model = synth_llama_model(&cfg);
    let prompt: Vec<i32> = vec![2, 9, 4, 1];
    let max_ctx = 16;

    // All-CPU run: n_gpu_layers = 0.
    crate::accel::set_n_gpu_layers(0);
    let mut kv = KvCache::new(&cfg, max_ctx);
    let logits_cpu = model.forward_prefill_batched_f32(&prompt, 0, &mut kv);

    // All-GPU run (well, all "would be GPU if a device existed"):
    // in mock mode the same CPU fallbacks fire.
    crate::accel::set_n_gpu_layers(u32::MAX);
    let mut kv = KvCache::new(&cfg, max_ctx);
    let logits_gpu = model.forward_prefill_batched_f32(&prompt, 0, &mut kv);

    // Half-and-half: 1 GPU layer, 2 CPU layers.
    crate::accel::set_n_gpu_layers(1);
    let mut kv = KvCache::new(&cfg, max_ctx);
    let logits_hybrid = model.forward_prefill_batched_f32(&prompt, 0, &mut kv);

    for (i, (&a, &b)) in logits_cpu.iter().zip(logits_gpu.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "logit[{i}] differs between n_gpu_layers=0 and n_gpu_layers=MAX",
        );
    }
    for (i, (&a, &b)) in logits_cpu.iter().zip(logits_hybrid.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "logit[{i}] differs between n_gpu_layers=0 and n_gpu_layers=1",
        );
    }
    // Restore TLS for any sibling tests on this thread.
    crate::accel::set_n_gpu_layers(u32::MAX);
    crate::accel::set_current_layer_idx(0);
}

/// `preload_packed_weights_to_usm_with_cutoff` returns a
/// consistent total tensor count across every cutoff: in mock
/// mode every tensor reports as "skipped" (no real USM
/// upload happens), so the function should report the same
/// `uploaded + skipped` total regardless of cutoff. The
/// **real**-hardware test for "cutoff actually shrinks the
/// upload count" requires SYCL and lives in `real_model.rs`-
/// style gated tests — that's the perf-measurement side and
/// not what this unit test asserts. What this test guards is
/// that the cutoff branch doesn't double-count or miss a
/// tensor.
#[test]
fn preload_cutoff_keeps_total_tensor_count_consistent() {
    let cfg = synth_cfg(4, 2, 8); // 4 transformer blocks
    let model = synth_llama_model(&cfg);
    // 4 blocks × 7 per-block tensors + token_embd = 29 tensors.
    // (Tied output weights — no separate `output` tensor.)
    const EXPECTED_TOTAL: usize = 4 * 7 + 1;
    for cutoff in [0u32, 1, 2, 3, 4, 999, u32::MAX] {
        let (uploaded, skipped, _) = model.preload_packed_weights_to_usm_with_cutoff(cutoff);
        assert_eq!(
            uploaded + skipped,
            EXPECTED_TOTAL,
            "cutoff={cutoff}: uploaded={uploaded} + skipped={skipped} != expected {EXPECTED_TOTAL}",
        );
    }
    // And the default `preload_packed_weights_to_usm()`
    // behaves identically to `with_cutoff(u32::MAX)`.
    let (a_up, a_sk, _) = model.preload_packed_weights_to_usm();
    let (b_up, b_sk, _) = model.preload_packed_weights_to_usm_with_cutoff(u32::MAX);
    assert_eq!(a_up, b_up);
    assert_eq!(a_sk, b_sk);
}

// ---- V-6b-3a: embed_tokens public surface --------------------

/// `embed_tokens` returns `[ids.len(), d_model]` row-major f32 —
/// pin the contract the V-5 splice depends on.
#[test]
fn embed_tokens_returns_d_model_row_per_id() {
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let ids: Vec<i32> = vec![3, 7, 2, 11];
    let out = model.embed_tokens(&ids);
    assert_eq!(out.len(), ids.len() * cfg.d_model);
}

/// Empty input yields empty output — no kernel call, no panic.
/// A VLM-aware engine can call this unconditionally for every
/// chat request (text-only requests just produce nothing).
#[test]
fn embed_tokens_empty_input_returns_empty() {
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let out = model.embed_tokens(&[]);
    assert!(out.is_empty());
}

/// Each row matches the same token's per-id lookup. Guarantees
/// the batched dispatch produces the same rows as the per-token
/// path the transformer body uses internally.
#[test]
fn embed_tokens_rows_match_per_token_lookup() {
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let d = cfg.d_model;
    let ids: Vec<i32> = vec![5, 100, 0, 42];
    let batched = model.embed_tokens(&ids);
    for (i, &id) in ids.iter().enumerate() {
        let single = model.embed_tokens(&[id]);
        assert_eq!(single.len(), d);
        assert_eq!(
            &batched[i * d..(i + 1) * d],
            &single[..],
            "row {i} (token {id}) diverges from per-token lookup",
        );
    }
}

/// Distinct token ids must yield distinct rows (otherwise the
/// splice loses information when an image-placeholder embedding
/// is silently identical to a text token).
#[test]
fn embed_tokens_distinct_ids_yield_distinct_rows() {
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let d = cfg.d_model;
    let out_a = model.embed_tokens(&[3]);
    let out_b = model.embed_tokens(&[7]);
    assert_eq!(out_a.len(), d);
    assert_eq!(out_b.len(), d);
    assert_ne!(out_a, out_b, "embedding table rows should differ per id");
}

/// `embed_tokens` is deterministic: calling twice with the same
/// ids produces bit-identical output. (No hidden RNG, no thread-
/// local state that mutates across calls.)
#[test]
fn embed_tokens_is_deterministic() {
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let ids: Vec<i32> = vec![1, 2, 3, 4];
    let a = model.embed_tokens(&ids);
    let b = model.embed_tokens(&ids);
    assert_eq!(a, b);
}

// ---- V-6b-3b: forward_one_from_embed / forward_prefill_from_embeds ----

/// `forward_one_from_embed(embed_tokens(&[id])[..])` produces
/// bit-identical logits to `forward_one(id, ...)`. This is the
/// core correctness claim of V-6b-3b: feeding a pre-embedded row
/// through the splice path matches the token-id path for the
/// no-image case, so V-5's splice can mix text and image rows
/// without bias for text tokens.
#[test]
fn forward_one_from_embed_matches_forward_one_for_text_token() {
    let cfg = synth_cfg(3, 2, 8);
    let model = synth_llama_model(&cfg);
    let token: i32 = 17;
    let max_ctx = 8;

    let mut kv_tok = KvCache::new(&cfg, max_ctx);
    let mut logits_tok = vec![0.0f32; cfg.vocab_size];
    model.forward_one(token, 0, &mut kv_tok, &mut logits_tok);

    let embed = model.embed_tokens(&[token]);
    let mut kv_emb = KvCache::new(&cfg, max_ctx);
    let mut logits_emb = vec![0.0f32; cfg.vocab_size];
    model.forward_one_from_embed(&embed, 0, &mut kv_emb, &mut logits_emb);

    for (i, (&a, &b)) in logits_tok.iter().zip(logits_emb.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "logit[{i}] differs between forward_one and forward_one_from_embed",
        );
    }
}

/// Prefill parity: feeding the same prompt either as token ids
/// (forward_prefill) or as pre-embedded rows (forward_prefill_from_embeds)
/// must produce the same final logits.
#[test]
fn forward_prefill_from_embeds_matches_forward_prefill_for_text_prompt() {
    let cfg = synth_cfg(3, 2, 8);
    let model = synth_llama_model(&cfg);
    let prompt: Vec<i32> = vec![3, 14, 1, 5, 9];
    let max_ctx = 16;

    let mut kv_tok = KvCache::new(&cfg, max_ctx);
    let logits_tok = model.forward_prefill(&prompt, 0, &mut kv_tok);

    let embeds = model.embed_tokens(&prompt);
    let mut kv_emb = KvCache::new(&cfg, max_ctx);
    let logits_emb = model.forward_prefill_from_embeds(&embeds, 0, &mut kv_emb);

    for (i, (&a, &b)) in logits_tok.iter().zip(logits_emb.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "logit[{i}] differs between forward_prefill and forward_prefill_from_embeds",
        );
    }
}

/// DeltaNetCache: SSM layers get sized buffers; full-attention
/// layers get empty sentinels; layer count matches `cfg.n_layers`.
/// Pins the Phase 3.7a contract.
#[test]
fn deltanet_cache_sizes_ssm_layers_only() {
    // Build a tiny synth model that's `qwen35moe`-shaped: 4 layers
    // with full_attention_interval=2 → layer 1 and layer 3 are
    // full attention; layers 0 and 2 are SSM. We bypass the GGUF
    // path by constructing config + a stub `hybrid_layers` vec
    // directly — the cache only inspects shapes, not weights.
    let mut cfg = synth_cfg(4, 2, 4);
    cfg.hybrid = Some(crate::llama_config::HybridConfig {
        full_attention_interval: 2,
        ssm_state_size: 16,
        ssm_conv_kernel: 4,
        ssm_group_count: 1,
        ssm_time_step_rank: 2,
        ssm_inner_size: 16, // 2 V heads × 8
        shared_expert_feed_forward_length: None,
        nextn_predict_layers: 0,
    });
    let n_v_heads = 2_usize;
    // Build stub HybridLayer entries. SSM layers need `ssm_a` of
    // length n_v_heads for the cache's head-count derivation.
    let mk_dense_ffn = || HybridFfn::Dense {
        w_gate: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
        w_up: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
        w_down: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
    };
    let mk_ssm = || {
        HybridLayer::Ssm(SsmBlockWeights {
            attn_norm: vec![],
            attn_qkv: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
            attn_gate: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
            post_attention_norm: vec![],
            // The cache derives the conv channel count (= fused-QKV
            // width) from this tensor's last dim: kernel 4 over
            // 2·ssm_inner = 32 channels for the legacy-shaped stub.
            ssm_conv1d: Tensor::zeros_cpu(Dtype::F32, vec![4u64, 32u64]),
            ssm_conv1d_f32: vec![],
            ssm_dt_bias: vec![],
            ssm_a: vec![0.0; n_v_heads],
            ssm_alpha: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
            ssm_beta: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
            ssm_norm: vec![],
            ssm_out: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
            ffn: mk_dense_ffn(),
        })
    };
    let mk_attn = || {
        HybridLayer::FullAttention(HybridAttnBlockWeights {
            attn_norm: vec![],
            w_q: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
            w_k: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
            w_v: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
            w_o: Tensor::zeros_cpu(Dtype::F32, vec![0u64]),
            q_norm: None,
            k_norm: None,
            post_attention_norm: vec![],
            ffn: mk_dense_ffn(),
        })
    };
    // Layers 0, 2 = SSM; layers 1, 3 = full attention (per
    // (i + 1) % 2 == 0 → layers 1, 3).
    let hybrid_layers = vec![mk_ssm(), mk_attn(), mk_ssm(), mk_attn()];
    let cache = DeltaNetCache::new_for_hybrid(&cfg, &hybrid_layers).expect("hybrid config present");
    assert_eq!(cache.layers.len(), 4, "one entry per layer");
    // SSM layers (0, 2) have sized buffers.
    for &i in &[0_usize, 2] {
        let l = &cache.layers[i];
        assert!(!l.is_empty(), "SSM layer {i} should have allocated state");
        // conv_state size: (kernel-1) * 2*ssm_inner = 3 * 32 = 96
        assert_eq!(l.conv_state.len(), 3 * 32);
        // recurrent_state: n_v_heads * head_qk_dim * head_v_dim
        // = 2 * 8 * 8 = 128
        assert_eq!(l.recurrent_state.len(), 128);
    }
    // Full-attention layers (1, 3) carry empty sentinels.
    for &i in &[1_usize, 3] {
        let l = &cache.layers[i];
        assert!(
            l.is_empty(),
            "full-attention layer {i} should be empty sentinel"
        );
    }
}

/// DeltaNetCache::reset zeros all buffers.
#[test]
fn deltanet_cache_reset_zeros_state() {
    // First param is the fused-QKV width (32 = 2·ssm_inner for
    // this legacy-shaped fixture), not ssm_inner.
    let mut state = DeltaNetLayerState::new_for_ssm(32, 2, 8, 8, 4);
    state.conv_state[0] = 1.0;
    state.recurrent_state[5] = 2.5;
    state.reset();
    assert!(state.conv_state.iter().all(|&v| v == 0.0));
    assert!(state.recurrent_state.iter().all(|&v| v == 0.0));
}

/// DeltaNetCache::new_for_hybrid errors on a non-hybrid cfg. The
/// "non-hybrid → no cache" guard now lives in the engine's
/// `build_delta_net_cache` (it checks the weights first and never
/// calls this on a dense model), so reaching here without a [hybrid]
/// config block is an inconsistency — surfaced as a clean error
/// instead of a panic.
#[test]
fn deltanet_cache_errors_for_non_hybrid() {
    let cfg = synth_cfg(2, 2, 8);
    assert!(cfg.hybrid.is_none());
    let cache = DeltaNetCache::new_for_hybrid(&cfg, &[]);
    assert!(cache.is_err());
}

/// Misshapen embed buffer panics (debug build) or errors clearly.
/// Pins the contract that the splice output length must be a
/// multiple of d_model.
#[test]
#[should_panic(expected = "not a multiple of d_model")]
fn forward_prefill_from_embeds_rejects_misshapen_buffer() {
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let max_ctx = 8;
    let mut kv = KvCache::new(&cfg, max_ctx);
    // d_model = 16; 17 floats is not a multiple of 16.
    let bad = vec![0.0f32; 17];
    let _ = model.forward_prefill_from_embeds(&bad, 0, &mut kv);
}

/// Empty embedding buffer yields a zero-logit vector — same as
/// calling forward_prefill with an empty token slice. Lets the
/// caller unconditionally route through the embed path for the
/// empty-prompt edge case without special-casing.
#[test]
fn forward_prefill_from_embeds_empty_input_returns_zero_logits() {
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let max_ctx = 8;
    let mut kv = KvCache::new(&cfg, max_ctx);
    let logits = model.forward_prefill_from_embeds(&[], 0, &mut kv);
    assert_eq!(logits.len(), cfg.vocab_size);
    assert!(logits.iter().all(|&x| x == 0.0));
}

/// Pin the contract for [`LlamaModel::forward_speculation`]:
/// per-position logits must be **bit-identical** to a manual
/// loop of `forward_one` calls. The serial-verify path's
/// implementation IS the loop, so this is mostly a smoke test
/// for the API shape — but the bit-identical assertion pins
/// the baseline so future batched implementations (the actual
/// perf win for D1) have a strict parity gate.
#[test]
fn forward_speculation_matches_forward_one_loop() {
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let tokens: Vec<i32> = vec![5, 7, 2, 14, 3];
    let max_ctx = 16;
    let vocab = cfg.vocab_size;

    // Baseline: drive forward_one in a manual loop.
    let mut kv_ref = KvCache::new(&cfg, max_ctx);
    let mut ref_logits: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());
    for (i, &t) in tokens.iter().enumerate() {
        let mut l = vec![0f32; vocab];
        model.forward_one(t, i as u32, &mut kv_ref, &mut l);
        ref_logits.push(l);
    }

    // forward_speculation should produce the concatenation.
    let mut kv_spec = KvCache::new(&cfg, max_ctx);
    let mut spec_logits = vec![0f32; tokens.len() * vocab];
    model.forward_speculation(&tokens, 0, &mut kv_spec, &mut spec_logits);

    for (i, ref_row) in ref_logits.iter().enumerate() {
        let lo = i * vocab;
        let hi = lo + vocab;
        for (j, (&a, &b)) in spec_logits[lo..hi].iter().zip(ref_row.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "forward_speculation logit[{i},{j}] diverges from forward_one loop: \
                     spec={a} ref={b}"
            );
        }
    }
    // KV state should also match (both wrote `tokens.len()` positions
    // sequentially, in the same order).
    assert_eq!(kv_ref.seq_len, kv_spec.seq_len);
}

/// After `forward_speculation` populates K+1 positions, dropping
/// `kv.seq_len` back (the rewind primitive that `KvBackend::
/// set_seq_len` exposes in the engine crate) must leave the
/// cache in a state where a subsequent `forward_one` at the
/// rewound position produces logits identical to walking the
/// same prefix on a fresh cache — i.e., the K/V rows past the
/// watermark are correctly invisible to attention.
/// E1: per-position logits from the batched F32-KV speculation
/// method must match the serial-loop primitive within numerical
/// noise. They are NOT bit-identical because the serial path
/// computes attention per-position via the regular GQA kernel
/// while the batched path uses `gqa_attention_flash_prefill`
/// which has a different floating-point reduction order — same
/// math, different accumulation. The end-to-end sampler is
/// invariant to ≤ 1-ULP logit differences, so the tolerance
/// (1e-3 abs, conservative) is well above the worst-case
/// reduction-order drift on a 2-layer synthetic model.
#[test]
fn forward_speculation_batched_f32_matches_serial_for_same_input() {
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let tokens: Vec<i32> = vec![5, 7, 2, 14, 3];
    let max_ctx = 16;
    let vocab = cfg.vocab_size;

    // Baseline: serial `forward_speculation` (driven by
    // forward_one per position).
    let mut kv_ref = KvCache::new(&cfg, max_ctx);
    let mut ref_logits = vec![0f32; tokens.len() * vocab];
    model.forward_speculation(&tokens, 0, &mut kv_ref, &mut ref_logits);

    // New: batched F32-KV speculation. Shares the per-layer
    // body with forward_prefill_batched_f32.
    let mut kv_batched = KvCache::new(&cfg, max_ctx);
    let mut batched_logits = vec![0f32; tokens.len() * vocab];
    model.forward_speculation_batched_f32(&tokens, 0, &mut kv_batched, &mut batched_logits);

    let mut max_diff = 0f32;
    for (i, (&a, &b)) in batched_logits.iter().zip(ref_logits.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
        }
        assert!(
            d < 1e-3,
            "forward_speculation_batched_f32 logit[{i}] drift {d}: \
                 batched={a} serial={b}"
        );
    }
    // Document the observed drift so a future tightening of
    // the tolerance has a baseline. On the synthetic 2-layer
    // model the drift is < 1e-5 in practice.
    let _ = max_diff;
    // KV state matches: same writes, same seq_len.
    assert_eq!(kv_ref.seq_len, kv_batched.seq_len);
}

/// Feature 4 degenerate-case guard: a CHAIN-shaped tree (`depths[i]==i`,
/// `parent[i]==i-1`) run through `forward_tree_verify_f32` must match the
/// linear `forward_speculation_batched_f32` — same per-node logits (within
/// the attention-reduction-order tolerance the linear-vs-serial test uses) —
/// at both `start_pos == 0` and past a committed prefix. Proves the tree
/// depth-RoPE + ancestor-mask forward reduces exactly to the pinned linear
/// verify, so only genuine branching diverges.
#[test]
fn forward_tree_verify_f32_chain_matches_speculation() {
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let vocab = cfg.vocab_size;
    let max_ctx = 16;

    // Part A: no committed prefix.
    let tokens: Vec<i32> = vec![5, 7, 2, 14, 3];
    let mut kv_lin = KvCache::new(&cfg, max_ctx);
    let mut lin_logits = vec![0f32; tokens.len() * vocab];
    model.forward_speculation_batched_f32(&tokens, 0, &mut kv_lin, &mut lin_logits);

    let mut kv_tree = KvCache::new(&cfg, max_ctx);
    let mut tree_logits = vec![0f32; tokens.len() * vocab];
    let depths: Vec<u32> = (0..tokens.len() as u32).collect();
    let parent: Vec<i32> = (0..tokens.len() as i32).map(|i| i - 1).collect();
    model.forward_tree_verify_f32(&tokens, 0, &depths, &parent, &mut kv_tree, &mut tree_logits);
    for (i, (&a, &b)) in tree_logits.iter().zip(lin_logits.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "tree-chain logit[{i}]: tree={a} lin={b}"
        );
    }
    assert_eq!(kv_lin.seq_len, kv_tree.seq_len);

    // Part B: past a committed 2-token prefix (exercises prefix attention +
    // the `start_pos + depth` RoPE position).
    let prefix = vec![9i32, 4];
    let rest = vec![5i32, 7, 2];
    let mut kv_lin2 = KvCache::new(&cfg, max_ctx);
    let _ = model.forward_prefill_batched_f32(&prefix, 0, &mut kv_lin2);
    let mut lin2 = vec![0f32; rest.len() * vocab];
    model.forward_speculation_batched_f32(&rest, 2, &mut kv_lin2, &mut lin2);

    let mut kv_tree2 = KvCache::new(&cfg, max_ctx);
    let _ = model.forward_prefill_batched_f32(&prefix, 0, &mut kv_tree2);
    let mut tree2 = vec![0f32; rest.len() * vocab];
    let depths2: Vec<u32> = (0..rest.len() as u32).collect();
    let parent2: Vec<i32> = (0..rest.len() as i32).map(|i| i - 1).collect();
    model.forward_tree_verify_f32(&rest, 2, &depths2, &parent2, &mut kv_tree2, &mut tree2);
    for (i, (&a, &b)) in tree2.iter().zip(lin2.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "tree-chain(prefix) logit[{i}]: tree={a} lin={b}"
        );
    }
}

/// F7 degenerate-case anchor (paged twin of
/// `forward_tree_verify_f32_chain_matches_speculation`): a CHAIN-shaped tree
/// (`depths[i]==i`, `parent[i]==i-1`) run through
/// [`LlamaModel::forward_tree_verify_paged_f32`] must match `n` sequential
/// paged decodes of the same tokens at the same positions — within the
/// attention-reduction tolerance the linear-vs-tree kernels differ by. This
/// pins that the paged depth-RoPE + ancestor-mask forward reduces exactly to
/// linear paged decode, so only genuine branching diverges. Model-blind (no
/// MTP head needed — it scores arbitrary tokens as a chain).
#[test]
fn forward_tree_verify_paged_f32_chain_matches_linear() {
    use crate::shared_paged_kv::SharedPagedKv;
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let vocab = cfg.vocab_size;
    let tokens: Vec<i32> = vec![5, 7, 2, 14, 3];
    let n = tokens.len();
    let page_size = 4u32;
    let max_ctx = 32u32;
    let total_pages = max_ctx.div_ceil(page_size);

    // Linear reference: n sequential single-slot paged decodes feeding
    // tokens[i] at pos i, capturing each step's logits. (Fresh store so the
    // reference KV is independent of the tree run.)
    let lin_store = crate::paged_kv_store::PagedKvStore::new(
        total_pages,
        cfg.n_layers as u32,
        cfg.n_kv_heads as u32,
        page_size,
        cfg.head_dim as u32,
    )
    .expect("lin store");
    let lin_table = crate::page_table::PageTable::new(total_pages, page_size);
    let lin_shared = SharedPagedKv::new(lin_store, lin_table);
    let mut lin_cache = crate::paged_kv_cache::PagedKvCache::new_with_page_size(page_size);
    lin_cache
        .ensure_capacity_shared(&lin_shared, n as u32)
        .expect("lin cap");
    let mut lin_logits = vec![0f32; n * vocab];
    for (i, &tok) in tokens.iter().enumerate() {
        let mut slot = DecodeSlot {
            token_id: tok,
            pos: i as u32,
            cache: &mut lin_cache,
            logits_out: &mut lin_logits[i * vocab..(i + 1) * vocab],
        };
        model.forward_decode_paged_batched_f32(std::slice::from_mut(&mut slot), &lin_shared);
    }

    // Tree run: a chain tree over the same tokens at start_pos 0.
    let tree_store = crate::paged_kv_store::PagedKvStore::new(
        total_pages,
        cfg.n_layers as u32,
        cfg.n_kv_heads as u32,
        page_size,
        cfg.head_dim as u32,
    )
    .expect("tree store");
    let tree_table = crate::page_table::PageTable::new(total_pages, page_size);
    let tree_shared = SharedPagedKv::new(tree_store, tree_table);
    let mut tree_cache = crate::paged_kv_cache::PagedKvCache::new_with_page_size(page_size);
    tree_cache
        .ensure_capacity_shared(&tree_shared, n as u32)
        .expect("tree cap");
    let depths: Vec<u32> = (0..n as u32).collect();
    let parent: Vec<i32> = (0..n as i32).map(|i| i - 1).collect();
    let mut tree_logits = vec![0f32; n * vocab];
    model.forward_tree_verify_paged_f32(
        &tokens,
        0,
        &depths,
        &parent,
        &mut tree_cache,
        &tree_shared,
        &mut tree_logits,
    );

    let mut max_diff = 0f32;
    for (i, (&a, &b)) in tree_logits.iter().zip(lin_logits.iter()).enumerate() {
        let diff = (a - b).abs();
        max_diff = max_diff.max(diff);
        assert!(
            diff < 1e-3,
            "paged tree-chain logit[{i}]: tree={a} linear={b} (diff {diff})"
        );
    }
    let _ = max_diff;
}

/// F7 commit anchor (paged twin of `kvcache_compact_tokens_moves_scattered_slots`):
/// [`crate::paged_kv_store::PagedKvStore::compact_positions`] moves a scattered
/// accepted path's K/V (positions base+{0,2,4}) into the contiguous committed
/// positions base+{0,1,2}, for every layer + head, leaving the data intact —
/// across a page boundary (page_size 4, base 3 ⇒ positions span pages 0/1/2).
#[test]
fn paged_compact_positions_moves_scattered_slots() {
    let page_size = 4u32;
    let n_layers = 2u32;
    let n_kv_heads = 2u32;
    let head_dim = 8u32;
    let total_pages = 4u32; // covers positions [0, 16)
    let mut store = crate::paged_kv_store::PagedKvStore::new(
        total_pages,
        n_layers,
        n_kv_heads,
        page_size,
        head_dim,
    )
    .expect("store");
    let pages: Vec<crate::page_table::PageId> =
        (0..total_pages).map(crate::page_table::PageId).collect();
    let base = 3usize;
    let n_nodes = 5usize;
    let stamp = |pos: usize, h: u32, dd: u32| (pos as f32) * 100.0 + (h as f32) * 10.0 + dd as f32;
    // Stamp tree slots base+{0..5} with distinct values on every layer+head.
    for s in 0..n_nodes {
        let pos = (base + s) as u32;
        let k_row: Vec<f32> = (0..n_kv_heads * head_dim)
            .map(|i| stamp(base + s, i / head_dim, i % head_dim))
            .collect();
        let v_row: Vec<f32> = k_row.iter().map(|x| x + 0.5).collect();
        for layer in 0..n_layers {
            store
                .write_token(&pages, layer, pos, &k_row, &v_row)
                .expect("write");
        }
    }
    // Accept scattered path nodes {0, 2, 4} → committed positions {0, 1, 2}.
    let accepted = [0usize, 2, 4];
    store.compact_positions(&pages, base, &accepted);
    // After compaction, committed position base+j holds node accepted[j]'s data.
    for (j, &src) in accepted.iter().enumerate() {
        let dst_pos = base + j;
        let page = pages[dst_pos / page_size as usize];
        let in_page = (dst_pos % page_size as usize) as u32;
        for h in 0..n_kv_heads {
            let k = store.k_row(page, 0, h, in_page).expect("k row");
            for (dd, &val) in k.iter().enumerate() {
                let expect = stamp(base + src, h, dd as u32);
                assert_eq!(
                    val.to_bits(),
                    expect.to_bits(),
                    "compacted K[j={j} src={src} h={h} d={dd}]: got {val} want {expect}"
                );
            }
            let v = store.v_row(page, 1, h, in_page).expect("v row");
            for (dd, &val) in v.iter().enumerate() {
                let expect = stamp(base + src, h, dd as u32) + 0.5;
                assert_eq!(
                    val.to_bits(),
                    expect.to_bits(),
                    "compacted V[j={j} src={src}]"
                );
            }
        }
    }
}

/// F6-B anchor (paged VLM ≡ contiguous VLM at the forward level): a spliced
/// INPUT-embedding buffer run through [`LlamaModel::forward_prefill_paged_from_embeds_f32`]
/// must match the contiguous [`LlamaModel::forward_prefill_from_embeds`] on
/// the SAME buffer, within the batched-vs-serial reduction tolerance. The
/// embed rows stand in for a vision splice (text rows + projected image
/// rows); the real mmproj-driven splice is exercised end-to-end in the
/// engine's `vlm_prefill_paged_matches_contiguous_with_image`.
#[test]
fn forward_prefill_paged_from_embeds_f32_matches_contiguous_from_embeds() {
    use crate::shared_paged_kv::SharedPagedKv;
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let d = cfg.d_model;
    let vocab = cfg.vocab_size;
    let n = 6usize;
    // Arbitrary "spliced" embeddings (a VLM splice is just a row buffer).
    let embeds: Vec<f32> = (0..n * d).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();

    let mut contig_kv = KvCache::new(&cfg, 32);
    let logits_c = model.forward_prefill_from_embeds(&embeds, 0, &mut contig_kv);

    let page_size = 4u32;
    let total_pages = 32u32.div_ceil(page_size);
    let store = crate::paged_kv_store::PagedKvStore::new(
        total_pages,
        cfg.n_layers as u32,
        cfg.n_kv_heads as u32,
        page_size,
        cfg.head_dim as u32,
    )
    .expect("store");
    let table = crate::page_table::PageTable::new(total_pages, page_size);
    let shared = SharedPagedKv::new(store, table);
    let mut cache = crate::paged_kv_cache::PagedKvCache::new_with_page_size(page_size);
    cache
        .ensure_capacity_shared(&shared, n as u32)
        .expect("cap");
    let logits_p = model.forward_prefill_paged_from_embeds_f32(&embeds, 0, &mut cache, &shared);

    assert_eq!(logits_c.len(), logits_p.len());
    for (i, (&a, &b)) in logits_c.iter().zip(logits_p.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "paged-from-embeds logit[{i}]: paged={b} contiguous={a}"
        );
    }
    let _ = vocab;
}

/// F6-B text-only-identity: feeding `embed_tokens(tokens)` through the paged
/// embed-row prefill must match a paged TOKEN prefill of the same tokens,
/// within the batched reduction tolerance — i.e. the embed-row source is a
/// faithful substitute for the token-id lookup on the text path (the VLM
/// model's text-only requests stay correct).
#[test]
fn forward_prefill_paged_from_embeds_f32_text_matches_token_prefill() {
    use crate::shared_paged_kv::SharedPagedKv;
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let tokens: Vec<i32> = vec![3, 1, 4, 1, 5, 9];
    let n = tokens.len();
    let embeds = model.embed_tokens(&tokens);

    let page_size = 4u32;
    let total_pages = 32u32.div_ceil(page_size);
    // Token prefill reference (direct-store path).
    let mut tok_store = crate::paged_kv_store::PagedKvStore::new(
        total_pages,
        cfg.n_layers as u32,
        cfg.n_kv_heads as u32,
        page_size,
        cfg.head_dim as u32,
    )
    .expect("tok store");
    let mut tok_table = crate::page_table::PageTable::new(total_pages, page_size);
    let mut tok_cache = crate::paged_kv_cache::PagedKvCache::new_for(&tok_store);
    tok_cache
        .ensure_capacity(&mut tok_table, n as u32)
        .expect("tok cap");
    let logits_tok = model.forward_prefill_paged_f32(&tokens, 0, &mut tok_cache, &mut tok_store);

    // Embed-row prefill (ragged/shared path).
    let store = crate::paged_kv_store::PagedKvStore::new(
        total_pages,
        cfg.n_layers as u32,
        cfg.n_kv_heads as u32,
        page_size,
        cfg.head_dim as u32,
    )
    .expect("store");
    let table = crate::page_table::PageTable::new(total_pages, page_size);
    let shared = SharedPagedKv::new(store, table);
    let mut cache = crate::paged_kv_cache::PagedKvCache::new_with_page_size(page_size);
    cache
        .ensure_capacity_shared(&shared, n as u32)
        .expect("cap");
    let logits_emb = model.forward_prefill_paged_from_embeds_f32(&embeds, 0, &mut cache, &shared);

    assert_eq!(logits_tok.len(), logits_emb.len());
    for (i, (&a, &b)) in logits_tok.iter().zip(logits_emb.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-3,
            "paged embed-row vs token prefill logit[{i}]: embed={b} token={a}"
        );
    }
}

// ---- Feature 4 (hybrid path): recurrent-state-under-tree walk ----

/// Toy recurrent driver standing in for a DeltaNet/SSM layer's sequential
/// state update, so the SHARED [`tree_recurrent_walk`] logic (the novel,
/// write-blind-risky part) can be pinned bit-exactly without a real
/// MTP-hybrid GGUF. `fold` has the shape of a real recurrence — the new
/// state mixes the previous state with its neighbor and folds in the token —
/// which is ORDER-SENSITIVE: a mis-ordered walk, or one that leaks state
/// across sibling branches, produces a different bit pattern.
struct ToyRecurrentDriver {
    state: Vec<f32>,
    tokens: Vec<i32>,
    /// Per-node post-step state (what each node would hand its children).
    out: Vec<Vec<f32>>,
}

impl ToyRecurrentDriver {
    fn new(dim: usize, tokens: Vec<i32>) -> Self {
        let n = tokens.len();
        Self {
            state: vec![0.0; dim],
            tokens,
            out: vec![Vec::new(); n],
        }
    }
    /// `s'[i] = 0.5·s[i] + 0.25·s[(i+1) mod S] + sin(0.1·(x + i))`.
    fn fold(state: &mut [f32], x: i32) {
        let s = state.len();
        let xf = x as f32;
        let prev: Vec<f32> = state.to_vec();
        for i in 0..s {
            let nb = prev[(i + 1) % s];
            state[i] = 0.5 * prev[i] + 0.25 * nb + (0.1 * (xf + i as f32)).sin();
        }
    }
}

impl RecurrentTreeDriver for ToyRecurrentDriver {
    type Snapshot = Vec<f32>;
    fn snapshot(&self) -> Vec<f32> {
        self.state.clone()
    }
    fn restore(&mut self, snap: &Vec<f32>) {
        self.state.copy_from_slice(snap);
    }
    fn step(&mut self, j: usize) {
        Self::fold(&mut self.state, self.tokens[j]);
        self.out[j] = self.state.clone();
    }
}

/// THE write-blind correctness anchor: a CHAIN-shaped tree (`parent[j] ==
/// j-1`) driven through `tree_recurrent_walk` must reproduce the plain
/// SEQUENTIAL recurrent scan BIT-FOR-BIT. This exercises the exact shared
/// walk code the real hybrid verify uses — every snapshot/restore round-trips
/// the state we just produced, so the degenerate chain is identity-equal to
/// folding the tokens one at a time. The analogue of
/// `forward_tree_verify_f32_chain_matches_speculation` for the recurrent path.
#[test]
fn toy_recurrent_tree_walk_chain_matches_serial() {
    let tokens = vec![3i32, 9, 1, 7, 4, 2];
    let dim = 5;
    let parent: Vec<i32> = (0..tokens.len() as i32).map(|i| i - 1).collect();
    let mut drv = ToyRecurrentDriver::new(dim, tokens.clone());
    tree_recurrent_walk(&mut drv, &parent);

    // Reference: a plain sequential scan from the zero state.
    let mut s = vec![0.0f32; dim];
    let mut serial: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());
    for &t in &tokens {
        ToyRecurrentDriver::fold(&mut s, t);
        serial.push(s.clone());
    }

    assert_eq!(drv.out.len(), serial.len());
    for (j, (a, b)) in drv.out.iter().zip(serial.iter()).enumerate() {
        assert_eq!(a.len(), b.len(), "node {j} state dim");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "chain node {j} dim {i}: walk={x} serial={y}"
            );
        }
    }
}

/// Branching guard: a caterpillar tree (the shape `build_mtp_tree` emits)
/// must restore each node from its PARENT's post-step state — sibling
/// branches restart from the shared parent and never leak state into one
/// another. Hand-computes every node's expected state from explicit folds
/// and asserts bit-exact, isolating the restore-from-parent semantics that
/// make recurrent tree verification correct.
#[test]
fn toy_recurrent_tree_walk_branches_restore_from_parent() {
    let dim = 4;
    // Caterpillar: level 0 = {node0 (main), node1 (leaf)}; level 1 =
    // {node2 (main, child of 0), node3 (leaf, child of 0)}; level 2 =
    // {node4 (child of 2)}. Exactly `build_mtp_tree(.., [2, 2, 1])` shape.
    let tokens = vec![5i32, 8, 2, 9, 6];
    let parent = vec![-1i32, -1, 0, 0, 2];
    let mut drv = ToyRecurrentDriver::new(dim, tokens.clone());
    tree_recurrent_walk(&mut drv, &parent);

    let root = vec![0.0f32; dim];
    let fold = |base: &[f32], x: i32| {
        let mut s = base.to_vec();
        ToyRecurrentDriver::fold(&mut s, x);
        s
    };
    let n0 = fold(&root, tokens[0]); // from root
    let n1 = fold(&root, tokens[1]); // sibling of n0 → ALSO from root
    let n2 = fold(&n0, tokens[2]); // child of n0
    let n3 = fold(&n0, tokens[3]); // sibling of n2 → from n0, not n2
    let n4 = fold(&n2, tokens[4]); // child of n2
    let expect = [n0, n1, n2, n3, n4];

    for (j, exp) in expect.iter().enumerate() {
        assert_eq!(drv.out[j].len(), exp.len(), "node {j} state dim");
        for (i, (&x, &y)) in drv.out[j].iter().zip(exp.iter()).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "branch node {j} dim {i}: walk={x} expected={y}"
            );
        }
    }
}

/// Real-weights validation hook (mirrors
/// `hybrid_from_embeds_matches_token_prefill_on_real_model`): on a REAL
/// hybrid GGUF named by `RUSTLLAMA_REAL_HYBRID_GGUF`, a CHAIN-shaped tree
/// run through [`LlamaModel::forward_tree_verify_hybrid_f32`] must reproduce
/// `n` sequential [`LlamaModel::forward_one_hybrid`] calls BIT-FOR-BIT
/// (identical forward calls at identical positions — the degenerate chain is
/// the linear decode). `#[ignore]`d so it never runs in CI (no fixture
/// on-box); the user runs it on real hardware with a hybrid model.
#[test]
#[ignore]
fn forward_tree_verify_hybrid_chain_matches_serial_on_real_model() {
    let Ok(path) = std::env::var("RUSTLLAMA_REAL_HYBRID_GGUF") else {
        eprintln!("RUSTLLAMA_REAL_HYBRID_GGUF not set — skipping");
        return;
    };
    let gguf = Gguf::open(&path).expect("open gguf");
    let cfg = crate::llama_config::LlamaConfig::from_gguf(&gguf).expect("config");
    assert!(cfg.hybrid.is_some(), "fixture must be a hybrid model");
    let weights = LlamaWeights::from_gguf(&gguf, &cfg).expect("weights");
    let model = LlamaModel::from_parts(cfg, weights);
    let hybrid = model.weights.hybrid_layers.as_ref().expect("hybrid layers");
    let vocab = model.cfg.vocab_size;
    let tokens: Vec<i32> = vec![3, 1000, 25, 4096, 77];
    let n = tokens.len();
    let max_ctx = 64usize;

    // Reference: n sequential single-token hybrid forwards.
    let mut kv_ref = KvCache::new_with_dtype(&model.cfg, max_ctx, KvDtype::F32);
    let mut dn_ref = DeltaNetCache::new_for_hybrid(&model.cfg, hybrid).expect("dn");
    let mut ref_logits = vec![0f32; n * vocab];
    for (i, &t) in tokens.iter().enumerate() {
        model.forward_one_hybrid(
            t,
            i as u32,
            &mut kv_ref,
            &mut dn_ref,
            &mut ref_logits[i * vocab..(i + 1) * vocab],
        );
    }

    // Tree verify over a CHAIN (parent[j]=j-1, depth[j]=j), start_pos 0.
    let mut kv_tree = KvCache::new_with_dtype(&model.cfg, max_ctx, KvDtype::F32);
    let mut dn_tree = DeltaNetCache::new_for_hybrid(&model.cfg, hybrid).expect("dn");
    let depths: Vec<u32> = (0..n as u32).collect();
    let parent: Vec<i32> = (0..n as i32).map(|i| i - 1).collect();
    let mut tree_logits = vec![0f32; n * vocab];
    model.forward_tree_verify_hybrid_f32(
        &tokens,
        0,
        &depths,
        &parent,
        &mut kv_tree,
        &mut dn_tree,
        &mut tree_logits,
    );

    for (i, (&a, &b)) in tree_logits.iter().zip(ref_logits.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "hybrid tree-chain logit[{i}]: tree={a} serial={b}"
        );
    }
    eprintln!("hybrid tree-verify chain==serial OK ({n} nodes × {vocab} vocab)");
}

/// Tree-spec commit guard: `compact_tokens` moves a scattered accepted
/// path's K/V (slots base+{0,2,4}) into contiguous committed slots
/// base+{0,1,2}, for every F32 layer + head, leaving the data intact.
#[test]
fn kvcache_compact_tokens_moves_scattered_slots() {
    let cfg = synth_cfg(2, 2, 8);
    let max_ctx = 16usize;
    let n_kv_heads = cfg.n_kv_heads;
    let head_dim = cfg.head_dim;
    let base = 3usize;
    let n_nodes = 5usize;
    let mut kv = KvCache::new(&cfg, max_ctx);
    // Stamp slot s with value `s*100 + h*10 + d` (k) / +0.5 (v).
    let stamp = |s: usize, h: usize, d: usize| (s * 100 + h * 10 + d) as f32;
    for layer in kv.layers.iter_mut() {
        if let KvLayer::F32 { k, v } = layer {
            for s in 0..n_nodes {
                for h in 0..n_kv_heads {
                    for d in 0..head_dim {
                        let off = (h * max_ctx + base + s) * head_dim + d;
                        k[off] = stamp(s, h, d);
                        v[off] = stamp(s, h, d) + 0.5;
                    }
                }
            }
        }
    }
    let src = vec![0usize, 2, 4]; // accepted path at scattered nodes
    kv.compact_tokens(base, &src);
    for layer in kv.layers.iter() {
        if let KvLayer::F32 { k, v } = layer {
            for (j, &s) in src.iter().enumerate() {
                for h in 0..n_kv_heads {
                    for d in 0..head_dim {
                        let off = (h * max_ctx + base + j) * head_dim + d;
                        assert_eq!(k[off], stamp(s, h, d), "k j={j} src={s} h={h} d={d}");
                        assert_eq!(v[off], stamp(s, h, d) + 0.5, "v j={j} src={s}");
                    }
                }
            }
        }
    }
}

/// SWA per-layer selection: no window off SWA; all layers windowed with no
/// pattern (Mistral); every `p`-th layer global with a pattern (Gemma2).
#[test]
fn swa_is_swa_layer_and_window() {
    let mut cfg = synth_cfg(6, 2, 8);
    assert!(!cfg.is_swa_layer(0));
    assert_eq!(cfg.swa_window(0), 0);
    // All-SWA (no pattern → Mistral-style).
    cfg.sliding_window = Some(128);
    for i in 0..6 {
        assert!(cfg.is_swa_layer(i), "all-SWA layer {i}");
        assert_eq!(cfg.swa_window(i), 128);
    }
    // Gemma2-style: every 3rd layer ((idx+1)%3==0) is a GLOBAL layer.
    cfg.sliding_window_pattern = Some(3);
    assert!(cfg.is_swa_layer(0)); // (1)%3=1 → SWA
    assert!(cfg.is_swa_layer(1)); // (2)%3=2 → SWA
    assert!(!cfg.is_swa_layer(2)); // (3)%3=0 → global
    assert_eq!(cfg.swa_window(2), 0);
    assert!(!cfg.is_swa_layer(5)); // (6)%3=0 → global
}

/// `finalize_logits` is a no-op without metadata (byte-identical), applies
/// Cohere `logit_scale` as a multiply, and Gemma2 `final_logit_softcapping`
/// as `c·tanh(l/c)`.
#[test]
fn finalize_logits_scale_and_softcap() {
    // No metadata → unchanged.
    let model = synth_llama_model(&synth_cfg(1, 1, 4));
    let orig = vec![1.0f32, -2.0, 3.0, 0.0];
    let mut l = orig.clone();
    model.finalize_logits(&mut l);
    assert_eq!(l, orig, "no-op without scale/softcap");
    // logit_scale multiplies.
    let mut cfg = synth_cfg(1, 1, 4);
    cfg.logit_scale = Some(2.0);
    let model = synth_llama_model(&cfg);
    let mut l = vec![1.0f32, -2.0, 3.0];
    model.finalize_logits(&mut l);
    assert_eq!(l, vec![2.0, -4.0, 6.0]);
    // final softcap: c·tanh(l/c).
    let mut cfg = synth_cfg(1, 1, 4);
    cfg.final_logit_softcapping = Some(5.0);
    let model = synth_llama_model(&cfg);
    let mut l = vec![10.0f32, -10.0];
    model.finalize_logits(&mut l);
    let want0 = 5.0 * (10.0f32 / 5.0).tanh();
    assert!((l[0] - want0).abs() < 1e-6 && (l[1] + want0).abs() < 1e-6);
}

// ---- CacheBlend ----

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0f32;
    let mut na = 0f32;
    let mut nb = 0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

/// Build a chunk's isolated KV snapshot the way the engine will: a
/// standalone F32-KV prefill of the chunk at position 0.
fn chunk_snapshot(model: &LlamaModel, cfg: &LlamaConfig, chunk: &[i32]) -> KvSnapshot {
    // P2: CacheBlend snapshots a full prefix — force a non-ring (full-slab)
    // cache so an SWA cfg doesn't bound the slab below the chunk length.
    let mut kv = KvCache::new_with_dtype_no_ring(cfg, 32, KvDtype::F32);
    let _ = model.forward_prefill_batched_f32(chunk, 0, &mut kv);
    kv.snapshot_prefix(chunk.len())
}

/// The correctness anchor: at `recompute_ratio = 1.0` every token is
/// recomputed, so CacheBlend is numerically a full prefill of the assembled
/// context — the cached chunk KV is injected but fully overwritten. The two
/// differ only in attention-kernel reduction order (per-query `one_step`
/// vs the batched flash-prefill), so we gate on cosine, not bit-equality.
#[test]
fn cacheblend_exact_matches_full_prefill() {
    use crate::cacheblend::{CacheBlendConfig, ChunkPlacement};
    use std::sync::Arc;
    let cfg = synth_cfg(4, 2, 8);
    let model = synth_llama_model(&cfg);
    let chunk1: Vec<i32> = vec![5, 7, 2, 14];
    let chunk2: Vec<i32> = vec![3, 9, 1, 6];
    let suffix: Vec<i32> = vec![11, 4];
    let assembled: Vec<i32> = chunk1
        .iter()
        .chain(&chunk2)
        .chain(&suffix)
        .copied()
        .collect();
    let suffix_start = chunk1.len() + chunk2.len();
    let max_ctx = 32;

    // Reference: a full batched prefill of the assembled context.
    let mut kv_ref = KvCache::new(&cfg, max_ctx);
    let ref_logits = model.forward_prefill_batched_f32(&assembled, 0, &mut kv_ref);

    // CacheBlend at ratio 1.0 (exact).
    let placements = vec![
        ChunkPlacement {
            offset: 0,
            snapshot: Arc::new(chunk_snapshot(&model, &cfg, &chunk1)),
        },
        ChunkPlacement {
            offset: chunk1.len(),
            snapshot: Arc::new(chunk_snapshot(&model, &cfg, &chunk2)),
        },
    ];
    let mut kv_cb = KvCache::new(&cfg, max_ctx);
    let cb_logits = model.forward_prefill_cacheblend(
        &assembled,
        &placements,
        suffix_start,
        CacheBlendConfig {
            recompute_ratio: 1.0,
            selection_layer: 1,
        },
        &mut kv_cb,
    );

    let cos = cosine(&cb_logits, &ref_logits);
    assert!(
        cos > 0.9995,
        "CacheBlend(exact) vs full prefill cosine {cos} too low"
    );
    assert_eq!(kv_cb.seq_len, assembled.len());
}

/// D2b: CacheBlend at `recompute_ratio=1.0` on a sliding-window + Gemma2
/// soft-cap model ≡ a plain windowed/capped full prefill. Mirrors
/// `cacheblend_exact_matches_full_prefill` with the window + soft-cap set, so
/// it pins that `forward_prefill_cacheblend` now threads per-layer
/// window/softcap through both attention branches (the D2b fix) — before it,
/// the un-windowed CacheBlend attention would diverge from the windowed
/// reference. Gated on cosine (reduction-order difference, same as the anchor).
#[test]
fn cacheblend_exact_matches_full_prefill_swa_softcap() {
    use crate::cacheblend::{CacheBlendConfig, ChunkPlacement};
    use std::sync::Arc;
    let mut cfg = synth_cfg(4, 2, 8);
    cfg.sliding_window = Some(3);
    cfg.sliding_window_pattern = Some(2);
    cfg.attn_logit_softcapping = Some(50.0);
    let model = synth_llama_model(&cfg);
    let chunk1: Vec<i32> = vec![5, 7, 2, 14];
    let chunk2: Vec<i32> = vec![3, 9, 1, 6];
    let suffix: Vec<i32> = vec![11, 4];
    let assembled: Vec<i32> = chunk1
        .iter()
        .chain(&chunk2)
        .chain(&suffix)
        .copied()
        .collect();
    let suffix_start = chunk1.len() + chunk2.len();
    let max_ctx = 32;

    // P2: CacheBlend operates on a full-slab (absolute-offset injection),
    // so force non-ring caches for the SWA cfg. The windowed/capped kernels
    // still mask correctly on the full slab.
    let mut kv_ref = KvCache::new_with_dtype_no_ring(&cfg, max_ctx, KvDtype::F32);
    let ref_logits = model.forward_prefill_batched_f32(&assembled, 0, &mut kv_ref);

    let placements = vec![
        ChunkPlacement {
            offset: 0,
            snapshot: Arc::new(chunk_snapshot(&model, &cfg, &chunk1)),
        },
        ChunkPlacement {
            offset: chunk1.len(),
            snapshot: Arc::new(chunk_snapshot(&model, &cfg, &chunk2)),
        },
    ];
    let mut kv_cb = KvCache::new_with_dtype_no_ring(&cfg, max_ctx, KvDtype::F32);
    let cb_logits = model.forward_prefill_cacheblend(
        &assembled,
        &placements,
        suffix_start,
        CacheBlendConfig {
            recompute_ratio: 1.0,
            selection_layer: 1,
        },
        &mut kv_cb,
    );

    let cos = cosine(&cb_logits, &ref_logits);
    assert!(
        cos > 0.9995,
        "CacheBlend(exact, SWA+softcap) vs full prefill cosine {cos} too low"
    );
    assert_eq!(kv_cb.seq_len, assembled.len());
}

/// Re-rotation correctness: injecting a chunk at a non-zero offset must
/// reproduce the LAYER-0 K a fresh prefill would write at those absolute
/// positions. Layer 0's K is `proj(norm(embed))` + RoPE — context-free — so
/// the only thing that can go wrong is the positional re-rotation. (Layers
/// ≥1 deliberately differ: that's the cross-attention the HKVD set recovers.)
#[test]
fn cacheblend_inject_reroped_matches_fresh_layer0_k() {
    use crate::cacheblend::ChunkPlacement;
    use std::sync::Arc;
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let head_dim = cfg.head_dim;
    let n_kv_heads = cfg.n_kv_heads;
    let max_ctx = 32;
    let off = 3;
    let chunk: Vec<i32> = vec![5, 7, 2, 14];
    let filler: Vec<i32> = vec![1, 8, 2]; // occupies [0, off)
    let assembled: Vec<i32> = filler.iter().chain(&chunk).copied().collect();

    // Fresh full prefill → the gold layer-0 K at positions off..off+len.
    let mut kv_full = KvCache::new(&cfg, max_ctx);
    let _ = model.forward_prefill_batched_f32(&assembled, 0, &mut kv_full);

    // Inject the chunk's isolated snapshot at `off` into an empty cache.
    let snap = chunk_snapshot(&model, &cfg, &chunk);
    let mut kv_inj = KvCache::new(&cfg, max_ctx);
    model.inject_chunk_reroped(
        &mut kv_inj,
        &ChunkPlacement {
            offset: off,
            snapshot: Arc::new(snap),
        },
    );

    let (kf, ki) = match (&kv_full.layers[0], &kv_inj.layers[0]) {
        (KvLayer::F32 { k: kf, .. }, KvLayer::F32 { k: ki, .. }) => (kf, ki),
        _ => unreachable!(),
    };
    let mut max_diff = 0f32;
    for j in 0..chunk.len() {
        let pos = off + j;
        for h in 0..n_kv_heads {
            let base = (h * max_ctx + pos) * head_dim;
            for e in 0..head_dim {
                max_diff = max_diff.max((kf[base + e] - ki[base + e]).abs());
            }
        }
    }
    assert!(
        max_diff < 1e-4,
        "injected re-roped layer-0 K drifts {max_diff} from fresh prefill"
    );
}

/// Smoke: the approximate path (ratio < 1) selects an HKVD subset, shrinks
/// the recompute set, and still returns finite logits for the final
/// position with the right seq_len. Approximation quality on a random
/// synthetic model isn't meaningful, so this only pins "it runs & stays
/// finite", not a similarity bound.
#[test]
fn cacheblend_approx_runs_and_is_finite() {
    use crate::cacheblend::{CacheBlendConfig, ChunkPlacement};
    use std::sync::Arc;
    let cfg = synth_cfg(4, 2, 8);
    let model = synth_llama_model(&cfg);
    let chunk1: Vec<i32> = vec![5, 7, 2, 14];
    let chunk2: Vec<i32> = vec![3, 9, 1, 6];
    let suffix: Vec<i32> = vec![11, 4];
    let assembled: Vec<i32> = chunk1
        .iter()
        .chain(&chunk2)
        .chain(&suffix)
        .copied()
        .collect();
    let suffix_start = chunk1.len() + chunk2.len();
    let placements = vec![
        ChunkPlacement {
            offset: 0,
            snapshot: Arc::new(chunk_snapshot(&model, &cfg, &chunk1)),
        },
        ChunkPlacement {
            offset: chunk1.len(),
            snapshot: Arc::new(chunk_snapshot(&model, &cfg, &chunk2)),
        },
    ];
    let mut kv_cb = KvCache::new(&cfg, 32);
    let logits = model.forward_prefill_cacheblend(
        &assembled,
        &placements,
        suffix_start,
        CacheBlendConfig {
            recompute_ratio: 0.25,
            selection_layer: 1,
        },
        &mut kv_cb,
    );
    assert_eq!(logits.len(), cfg.vocab_size);
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "approx logits not finite"
    );
    assert_eq!(kv_cb.seq_len, assembled.len());
}

/// E1 Q8_0: per-position logits from the batched Q8_0-KV
/// speculation method match the serial-loop primitive within
/// numerical noise. Same tolerance rationale as the F32 parity
/// test — the two paths differ in attention kernel reduction
/// order. The KV append + inline-dequant in the batched path
/// also accumulates Q8_0 rows in a slightly different order
/// than the per-token forward_one loop.
#[test]
fn forward_speculation_batched_q8_0_matches_serial_for_same_input() {
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let tokens: Vec<i32> = vec![5, 7, 2, 14, 3];
    let max_ctx = 16;
    let vocab = cfg.vocab_size;

    // Baseline: serial path on a Q8_0 KV cache.
    let mut kv_ref = KvCache::new_with_dtype(&cfg, max_ctx, KvDtype::Q8_0);
    let mut ref_logits = vec![0f32; tokens.len() * vocab];
    model.forward_speculation(&tokens, 0, &mut kv_ref, &mut ref_logits);

    // Batched Q8_0 path.
    let mut kv_batched = KvCache::new_with_dtype(&cfg, max_ctx, KvDtype::Q8_0);
    let mut batched_logits = vec![0f32; tokens.len() * vocab];
    model.forward_speculation_batched_q8_0(&tokens, 0, &mut kv_batched, &mut batched_logits);

    let mut max_diff = 0f32;
    for (i, (&a, &b)) in batched_logits.iter().zip(ref_logits.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
        }
        assert!(
            d < 1e-2,
            "forward_speculation_batched_q8_0 logit[{i}] drift {d}: \
                 batched={a} serial={b}"
        );
    }
    let _ = max_diff;
    assert_eq!(kv_ref.seq_len, kv_batched.seq_len);
}

// ---- M6: f16 / bf16 KV ----

/// M6: `KvDtype::parse` accepts the half spellings.
#[test]
fn kv_dtype_parse_half_spellings() {
    assert_eq!(KvDtype::parse("f16"), Some(KvDtype::F16));
    assert_eq!(KvDtype::parse("fp16"), Some(KvDtype::F16));
    assert_eq!(KvDtype::parse("float16"), Some(KvDtype::F16));
    assert_eq!(KvDtype::parse("F16"), Some(KvDtype::F16));
    assert_eq!(KvDtype::parse("bf16"), Some(KvDtype::Bf16));
    assert_eq!(KvDtype::parse("bfloat16"), Some(KvDtype::Bf16));
    assert_eq!(KvDtype::parse("BF16"), Some(KvDtype::Bf16));
    assert_eq!(KvDtype::parse("nope"), None);
}

/// M6: f32 → half → f32 round-trips within the format's precision.
#[test]
fn half_kv_row_roundtrip_within_precision() {
    let src: Vec<f32> = (0..64).map(|i| ((i as f32) - 32.0) * 0.37).collect();
    let mut enc = vec![0u16; src.len()];
    let mut dec = vec![0f32; src.len()];
    // f16: ~10-bit mantissa ⇒ ~2^-10 relative error.
    half_encode_row(false, &src, &mut enc);
    half_decode_row(false, &enc, &mut dec);
    for (&a, &b) in src.iter().zip(dec.iter()) {
        let tol = a.abs() * 1e-3 + 1e-3;
        assert!((a - b).abs() <= tol, "f16 roundtrip {a} -> {b}");
    }
    // bf16: 7-bit mantissa ⇒ ~2^-7 relative error.
    half_encode_row(true, &src, &mut enc);
    half_decode_row(true, &enc, &mut dec);
    for (&a, &b) in src.iter().zip(dec.iter()) {
        let tol = a.abs() * 1e-2 + 1e-2;
        assert!((a - b).abs() <= tol, "bf16 roundtrip {a} -> {b}");
    }
}

fn serial_last_logits(
    model: &LlamaModel,
    cfg: &LlamaConfig,
    tokens: &[i32],
    dt: KvDtype,
    max_ctx: usize,
) -> Vec<f32> {
    let mut kv = KvCache::new_with_dtype(cfg, max_ctx, dt);
    let mut logits = vec![0f32; cfg.vocab_size];
    for (i, &tok) in tokens.iter().enumerate() {
        model.forward_one(tok, i as u32, &mut kv, &mut logits);
    }
    logits
}

/// M6: an f16/bf16-KV forward stays close (cosine > 0.999) to the
/// f32-KV forward on a synth model — both the serial `forward_one`
/// decode path and the batched prefill path.
#[test]
fn half_kv_decode_parity_vs_f32() {
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let tokens: Vec<i32> = vec![5, 7, 2, 14, 3, 9, 1, 11];
    let max_ctx = 16;

    // Serial decode path.
    let f32_serial = serial_last_logits(&model, &cfg, &tokens, KvDtype::F32, max_ctx);
    for dt in [KvDtype::F16, KvDtype::Bf16] {
        let half = serial_last_logits(&model, &cfg, &tokens, dt, max_ctx);
        let cos = cosine(&f32_serial, &half);
        assert!(cos > 0.999, "{dt:?} serial decode cosine {cos} <= 0.999");
    }

    // Batched prefill path.
    let mut kv_f32 = KvCache::new_with_dtype(&cfg, max_ctx, KvDtype::F32);
    let f32_pf = model.forward_prefill_batched_f32(&tokens, 0, &mut kv_f32);
    for bf16 in [false, true] {
        let dt = if bf16 { KvDtype::Bf16 } else { KvDtype::F16 };
        let mut kv_h = KvCache::new_with_dtype(&cfg, max_ctx, dt);
        let half_pf = model.forward_prefill_batched_half(bf16, &tokens, 0, &mut kv_h);
        let cos = cosine(&f32_pf, &half_pf);
        assert!(cos > 0.999, "{dt:?} batched prefill cosine {cos} <= 0.999");
    }
}

// ---- F4: per-layer KV dtype ----

/// F4 REGRESSION GUARD: a cache built with every layer set to dtype X
/// via `new_with_layer_dtypes` is byte-for-byte the historic
/// single-dtype cache — same per-layer variants, same `uniform_dtype`,
/// and bit-exact forward logits.
#[test]
fn kv_all_same_layer_dtypes_byte_identical() {
    let cfg = synth_cfg(3, 2, 8);
    let model = synth_llama_model(&cfg);
    let tokens: Vec<i32> = vec![5, 7, 2, 14, 3];
    let max_ctx = 16;
    // Head_dim 8 ⇒ use dtypes that don't require 32-elem alignment.
    for dt in [KvDtype::F32, KvDtype::F16, KvDtype::Bf16, KvDtype::Q8_0] {
        let single = KvCache::new_with_dtype(&cfg, max_ctx, dt);
        let per_layer = KvCache::new_with_layer_dtypes(&cfg, max_ctx, &vec![dt; cfg.n_layers]);
        assert_eq!(per_layer.layer_dtypes, vec![dt; cfg.n_layers]);
        assert_eq!(per_layer.uniform_dtype(), Some(dt));
        assert_eq!(single.dtype(), per_layer.dtype());
        assert_eq!(
            std::mem::discriminant(&single.layers[0]),
            std::mem::discriminant(&per_layer.layers[0]),
        );

        let mut kv_a = KvCache::new_with_dtype(&cfg, max_ctx, dt);
        let mut kv_b = KvCache::new_with_layer_dtypes(&cfg, max_ctx, &vec![dt; cfg.n_layers]);
        let la = model.forward_prefill(&tokens, 0, &mut kv_a);
        let lb = model.forward_prefill(&tokens, 0, &mut kv_b);
        assert_eq!(
            la, lb,
            "{dt:?} all-same per-layer != single-dtype (not byte-identical)"
        );
    }
}

/// F4: a genuinely MIXED per-layer cache (layer0=f32, layer1=f16,
/// layer2=q8_0) stores each layer at its own dtype, reports
/// `uniform_dtype()==None` (so it uses the serial per-layer forward),
/// and produces coherent finite logits.
#[test]
fn kv_mixed_layer_dtypes_runs_coherently() {
    let cfg = synth_cfg(3, 2, 8);
    let model = synth_llama_model(&cfg);
    let tokens: Vec<i32> = vec![5, 7, 2, 14, 3];
    let max_ctx = 16;
    let spec = [KvDtype::F32, KvDtype::F16, KvDtype::Q8_0];
    let mut kv = KvCache::new_with_layer_dtypes(&cfg, max_ctx, &spec);
    assert_eq!(kv.layer_dtypes, spec.to_vec());
    assert_eq!(
        kv.uniform_dtype(),
        None,
        "mixed cache must not read as uniform"
    );
    // Each layer stored at its own dtype (variant check).
    assert!(matches!(kv.layers[0], KvLayer::F32 { .. }));
    assert!(matches!(kv.layers[1], KvLayer::Half { bf16: false, .. }));
    assert!(matches!(kv.layers[2], KvLayer::Q8_0 { .. }));

    let logits = model.forward_prefill(&tokens, 0, &mut kv);
    assert_eq!(logits.len(), cfg.vocab_size);
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "mixed-dtype logits must be finite"
    );
    assert_eq!(kv.seq_len, tokens.len());

    // Snapshot/restore round-trips on a mixed cache (per-layer safe).
    let snap = kv.snapshot_prefix(tokens.len());
    let mut kv2 = KvCache::new_with_layer_dtypes(&cfg, max_ctx, &spec);
    kv2.restore_prefix(&snap);
    assert_eq!(kv2.seq_len, tokens.len());
}

/// E1 TurboQuant: per-position logits from the batched TQ-KV
/// speculation method match the serial-loop primitive within
/// numerical noise. TQ has a 4-bit quantize-on-write path that
/// accumulates differently from the per-token forward_one loop.
#[test]
fn forward_speculation_batched_tq_matches_serial_for_same_input() {
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let tokens: Vec<i32> = vec![5, 7, 2, 14, 3];
    let max_ctx = 16;
    let vocab = cfg.vocab_size;

    let mut kv_ref = KvCache::new_with_dtype(&cfg, max_ctx, KvDtype::Tq(4));
    let mut ref_logits = vec![0f32; tokens.len() * vocab];
    model.forward_speculation(&tokens, 0, &mut kv_ref, &mut ref_logits);

    let mut kv_batched = KvCache::new_with_dtype(&cfg, max_ctx, KvDtype::Tq(4));
    let mut batched_logits = vec![0f32; tokens.len() * vocab];
    model.forward_speculation_batched_tq(&tokens, 0, &mut kv_batched, &mut batched_logits);

    // TQ tolerance: 4-bit quantization injects more noise per
    // row than Q8_0 (8-bit); allow 5e-2 abs to cover the
    // worst-case attention-reduction-order drift on a
    // 4-bit-quantized cache.
    let mut max_diff = 0f32;
    for (i, (&a, &b)) in batched_logits.iter().zip(ref_logits.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
        }
        assert!(
            d < 5e-2,
            "forward_speculation_batched_tq logit[{i}] drift {d}: \
                 batched={a} serial={b}"
        );
    }
    let _ = max_diff;
    assert_eq!(kv_ref.seq_len, kv_batched.seq_len);
}

/// E1 NVFP4: per-position logits from the batched NVFP4-KV
/// speculation method match the serial-loop primitive within
/// numerical noise. NVFP4 is the second FP-quant variant
/// (alongside F32) — 4-bit FP with FP8 scales per 16-element
/// block; the batched path uses a per-block dequant
/// concurrent with the attention compute.
#[test]
fn forward_speculation_batched_nvfp4_matches_serial_for_same_input() {
    // NVFP4 requires head_dim % 16 == 0 (block size).
    let cfg = synth_cfg(2, 2, 16);
    let model = synth_llama_model(&cfg);
    let tokens: Vec<i32> = vec![5, 7, 2, 14, 3];
    let max_ctx = 16;
    let vocab = cfg.vocab_size;

    let mut kv_ref = KvCache::new_with_dtype(&cfg, max_ctx, KvDtype::Nvfp4);
    let mut ref_logits = vec![0f32; tokens.len() * vocab];
    model.forward_speculation(&tokens, 0, &mut kv_ref, &mut ref_logits);

    let mut kv_batched = KvCache::new_with_dtype(&cfg, max_ctx, KvDtype::Nvfp4);
    let mut batched_logits = vec![0f32; tokens.len() * vocab];
    model.forward_speculation_batched_nvfp4(&tokens, 0, &mut kv_batched, &mut batched_logits);

    let mut max_diff = 0f32;
    for (i, (&a, &b)) in batched_logits.iter().zip(ref_logits.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_diff {
            max_diff = d;
        }
        assert!(
            d < 5e-2,
            "forward_speculation_batched_nvfp4 logit[{i}] drift {d}: \
                 batched={a} serial={b}"
        );
    }
    let _ = max_diff;
    assert_eq!(kv_ref.seq_len, kv_batched.seq_len);
}

#[test]
fn forward_speculation_rewind_then_forward_one_matches_fresh_path() {
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let prefix: Vec<i32> = vec![3, 1, 4];
    let drafts: Vec<i32> = vec![9, 9, 9, 9]; // rejected drafts
    let replacement: i32 = 5; // the real next token (chosen post-rewind)
    let vocab = cfg.vocab_size;
    let max_ctx = 16;

    // Path A: walk prefix + drafts via forward_speculation,
    // then rewind to end-of-prefix, then forward_one(replacement).
    let mut full: Vec<i32> = prefix.clone();
    full.extend_from_slice(&drafts);
    let mut kv_a = KvCache::new(&cfg, max_ctx);
    let mut logits = vec![0f32; full.len() * vocab];
    model.forward_speculation(&full, 0, &mut kv_a, &mut logits);
    // Rewind: drop seq_len back to end-of-prefix. The K/V cells
    // at positions [prefix.len() .. full.len()) keep their data
    // but become invisible to attention because the kernels
    // walk only `seq_len` positions.
    kv_a.seq_len = prefix.len();
    let mut a_logits = vec![0f32; vocab];
    model.forward_one(replacement, prefix.len() as u32, &mut kv_a, &mut a_logits);

    // Path B: walk prefix + replacement directly (no draft
    // detour, no rewind).
    let mut kv_b = KvCache::new(&cfg, max_ctx);
    for (i, &t) in prefix.iter().enumerate() {
        let mut l = vec![0f32; vocab];
        model.forward_one(t, i as u32, &mut kv_b, &mut l);
    }
    let mut b_logits = vec![0f32; vocab];
    model.forward_one(replacement, prefix.len() as u32, &mut kv_b, &mut b_logits);

    // Bit-identical: the rewind drops rejected drafts' K/V
    // rows from attention's view (seq_len gate); the
    // forward_one at position `prefix.len()` overwrites
    // whatever was there from `drafts[0]` with the
    // replacement's actual K/V, so subsequent attention reads
    // are correct.
    for (i, (&a, &b)) in a_logits.iter().zip(b_logits.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "logit[{i}]: rewind+overwrite path = {a}, fresh path = {b}"
        );
    }
}

/// E4 phase 4b parity gate: `forward_one_with_mtp_logits` on a
/// model with `n_mtp_heads == 0` produces bit-identical main
/// logits to `forward_one`. The MTP path is purely additive —
/// for non-MTP models, the new entry point is a no-overhead
/// passthrough that emits only the main logits.
#[test]
fn forward_one_with_mtp_logits_matches_forward_one_when_no_heads() {
    let cfg = synth_cfg(2, 2, 8);
    let model = synth_llama_model(&cfg);
    let max_ctx = 16;
    let prefix: Vec<i32> = vec![3, 1, 4, 1];
    let probe: i32 = 5;

    let mut kv_a = KvCache::new(&cfg, max_ctx);
    for (i, &t) in prefix.iter().enumerate() {
        let mut tmp = vec![0f32; cfg.vocab_size];
        model.forward_one(t, i as u32, &mut kv_a, &mut tmp);
    }
    let mut logits_a = vec![0f32; cfg.vocab_size];
    model.forward_one(probe, prefix.len() as u32, &mut kv_a, &mut logits_a);

    let mut kv_b = KvCache::new(&cfg, max_ctx);
    for (i, &t) in prefix.iter().enumerate() {
        let mut tmp = vec![0f32; cfg.vocab_size];
        model.forward_one(t, i as u32, &mut kv_b, &mut tmp);
    }
    let mut logits_b = vec![0f32; cfg.vocab_size];
    let mut mtp_logits: Vec<Vec<f32>> = Vec::new();
    model.forward_one_with_mtp_logits(
        probe,
        prefix.len() as u32,
        &mut kv_b,
        &mut logits_b,
        &mut mtp_logits,
    );
    assert!(
        mtp_logits.is_empty(),
        "non-MTP model must leave mtp_logits empty"
    );
    for (i, (&a, &b)) in logits_a.iter().zip(logits_b.iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "logit[{i}]: forward_one = {a}, forward_one_with_mtp_logits = {b} \
                 — non-MTP path must be bit-identical"
        );
    }
}

/// E4 phase 4a parity gate: a model whose config does not declare
/// MTP heads loads with `mtp_heads == None`. This is the
/// no-behavior-change pin — every non-MTP model in the wild
/// continues to flow through the unchanged single-head path.
#[test]
fn synth_model_without_mtp_metadata_has_no_mtp_heads() {
    let cfg = synth_cfg(2, 2, 8);
    assert_eq!(
        cfg.n_mtp_heads, 0,
        "synth_cfg sets the no-MTP default; if this regresses, every \
             synthetic model in the test suite would start trying to \
             materialize phantom MTP heads"
    );
    let model = synth_llama_model(&cfg);
    assert!(
        model.weights.mtp_heads.is_none(),
        "models without MTP metadata must load with mtp_heads = None"
    );
}
