//! P11 — LoRA / multi-LoRA anchors.
//!
//! - P1 (merge-at-load): the library merge `W += s·B·A` is bit-exact to an
//!   independent hand-applied reference on an F32 base; a merged model's
//!   logits differ from the un-adapted base.
//! - P2 (delta-apply at dispatch): the delta-applied logits match the Phase-1
//!   merged logits (bit-tolerant) and differ from the base.
//! - P3 (multi-LoRA): per-request selection picks the right adapter; the base
//!   (no adapter) selection is byte-identical to the un-adapted forward; two
//!   concurrent threads running different adapters each match their
//!   single-adapter reference.

use std::sync::Arc;

use rustllama_gguf::synth::{write_synthetic_lora_adapter_gguf, SynthLlama, SynthLoraAdapter};
use rustllama_gguf::Gguf;
use rustllama_models::llama_arch::{KvCache, LlamaModel};
use rustllama_models::lora::{self, LoraAdapter, LoraFactors, MergeOutcome};
use rustllama_tensor::{DimOrder, Tensor};

fn tmp(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("rustllama-lora-{name}-{}.gguf", std::process::id()))
}

/// Write the default synth base + an adapter with `seed`, returning both paths.
/// The base path is seed-qualified so concurrently-running tests (cargo runs
/// test fns in parallel threads) never clobber one another's fixture.
fn write_base_and_adapter(seed: u64) -> (std::path::PathBuf, std::path::PathBuf) {
    let base_path = tmp(&format!("base-{seed:x}"));
    rustllama_gguf::synth::write_synthetic_llama_gguf(&base_path, &SynthLlama::default());
    let adapter_path = tmp(&format!("adapter-{seed:x}"));
    write_synthetic_lora_adapter_gguf(
        &adapter_path,
        &SynthLoraAdapter {
            base: SynthLlama::default(),
            rank: 8,
            alpha: 16.0,
            seed,
        },
    );
    (base_path, adapter_path)
}

fn logits_decode(model: &LlamaModel, token: i32) -> Vec<f32> {
    let mut kv = KvCache::new(&model.cfg, 16);
    let mut logits = vec![0f32; model.cfg.vocab_size];
    model.forward_one(token, 0, &mut kv, &mut logits);
    logits
}

fn logits_decode_lora(
    model: &LlamaModel,
    token: i32,
    adapter: Option<&Arc<LoraAdapter>>,
) -> Vec<f32> {
    let mut kv = KvCache::new(&model.cfg, 16);
    let mut logits = vec![0f32; model.cfg.vocab_size];
    model.forward_one_with_lora(token, 0, &mut kv, &mut logits, adapter);
    logits
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

// ----------------------------------------------------------------------
// P1 — merge math is bit-exact to a hand-applied reference (F32 base)
// ----------------------------------------------------------------------

#[test]
fn p1_merge_math_bit_exact_f32_base() {
    // Craft an F32 base `W [out,in]` + known factors, merge with the library,
    // and compare to an INDEPENDENT hand loop computing `W + s·B·A`.
    let (out, inp, rank) = (5usize, 4usize, 3usize);
    let scale = 2.0f32;
    // Deterministic small weights/factors.
    let w: Vec<f32> = (0..out * inp).map(|i| 0.1 * i as f32 - 0.3).collect();
    let a: Vec<f32> = (0..rank * inp).map(|i| 0.05 * i as f32 - 0.2).collect();
    let b: Vec<f32> = (0..out * rank).map(|i| -0.03 * i as f32 + 0.1).collect();

    // Independent hand reference: W'[r][c] = W[r][c] + s·Σ_k B[r][k]·A[k][c].
    let mut expected = w.clone();
    for r in 0..out {
        for c in 0..inp {
            let mut acc = 0.0f32;
            for k in 0..rank {
                acc += b[r * rank + k] * a[k * inp + c];
            }
            expected[r * inp + c] += scale * acc;
        }
    }

    // Library path: a Tensor named to match a single-entry adapter.
    let mut t = Tensor::from_vec_f32("blk.0.attn_q.weight", vec![out as u64, inp as u64], w);
    let mut factors = std::collections::HashMap::new();
    factors.insert(
        "blk.0.attn_q.weight".to_string(),
        LoraFactors {
            rank,
            in_dim: inp,
            out_dim: out,
            scale,
            a,
            b,
        },
    );
    let adapter = LoraAdapter {
        name: "t".into(),
        arch: None,
        alpha: Some(scale * rank as f32),
        factors,
    };
    assert_eq!(
        lora::merge_tensor_in_place(&adapter, &mut t),
        MergeOutcome::Merged
    );
    let got = rustllama_tensor::as_slice_f32(&t);
    assert_eq!(
        got,
        &expected[..],
        "merged weight must be bit-exact to hand reference"
    );
    // Merged tensor is a row-major F32 `[out, in]` weight.
    assert_eq!(t.dim_order, DimOrder::RowMajor);
    assert_eq!(t.out_dim(), out);
    assert_eq!(t.in_dim(), inp);
}

// ----------------------------------------------------------------------
// P1 — adapter parse: dims, rank, scale = alpha / rank
// ----------------------------------------------------------------------

#[test]
fn p1_adapter_parses_factors_and_scale() {
    let (_base, adapter_path) = write_base_and_adapter(0x1111);
    let g = Gguf::open(&adapter_path).expect("open adapter");
    let adapter = LoraAdapter::from_gguf(&g, "a").expect("parse adapter");
    assert_eq!(adapter.alpha, Some(16.0));
    // 7 projections × 2 layers.
    assert_eq!(adapter.len(), 14);
    let base = SynthLlama::default();
    let f = adapter
        .factors
        .get("blk.0.attn_q.weight")
        .expect("attn_q factors");
    assert_eq!(f.rank, 8);
    assert_eq!(f.in_dim, base.d_model as usize);
    assert_eq!(f.out_dim, (base.n_heads * base.head_dim) as usize);
    // scale = alpha / rank = 16 / 8 = 2.0.
    assert!((f.scale - 2.0).abs() < 1e-6);
    assert_eq!(f.a.len(), f.rank * f.in_dim);
    assert_eq!(f.b.len(), f.out_dim * f.rank);
    let _ = std::fs::remove_file(&adapter_path);
}

// ----------------------------------------------------------------------
// P1 — merged model logits ≡ hand-applied reference, and ≠ base
// ----------------------------------------------------------------------

/// Hand-merge an adapter into a model's weights via an independent loop over
/// the parsed factors (the test's own merge, cross-checking the library's).
fn hand_merge(model: &mut LlamaModel, adapter: &LoraAdapter) {
    for b in &mut model.weights.blocks {
        for t in [
            &mut b.w_q,
            &mut b.w_k,
            &mut b.w_v,
            &mut b.w_o,
            &mut b.w_gate,
            &mut b.w_up,
            &mut b.w_down,
        ] {
            let Some(f) = adapter.factors.get(&t.name) else {
                continue;
            };
            let (out, inp) = (t.out_dim(), t.in_dim());
            // Dequant base to f32 (synth weights are F16 → lossless widen).
            let mut base: Vec<f32> = rustllama_tensor::as_slice_f16(t)
                .iter()
                .map(|h| h.to_f32())
                .collect();
            for r in 0..out {
                for c in 0..inp {
                    let mut acc = 0.0f32;
                    for k in 0..f.rank {
                        acc += f.b[r * f.rank + k] * f.a[k * f.in_dim + c];
                    }
                    base[r * inp + c] += f.scale * acc;
                }
            }
            *t = Tensor::from_vec_f32(t.name.clone(), vec![out as u64, inp as u64], base);
        }
        b.w_qkv_fused = None;
    }
}

#[test]
fn p1_merged_equals_hand_and_differs_from_base() {
    let (base_path, adapter_path) = write_base_and_adapter(0x2222);
    let base_gguf = Gguf::open(&base_path).expect("base");
    let adapter_gguf = Gguf::open(&adapter_path).expect("adapter");
    let adapter = LoraAdapter::from_gguf(&adapter_gguf, "a").expect("parse");

    let base_model = LlamaModel::load(&base_gguf).expect("load base");
    let logits_base = logits_decode(&base_model, 3);

    // Library merge.
    let mut merged = LlamaModel::load(&base_gguf).expect("load for merge");
    let stats = merged.weights.merge_lora(&adapter);
    assert_eq!(
        stats.merged, 14,
        "all 7 dense projections × 2 layers merged"
    );
    assert_eq!(stats.skipped_quant, 0);
    assert_eq!(stats.shape_mismatch, 0);
    let logits_merged = logits_decode(&merged, 3);

    // Independent hand reference.
    let mut hand = LlamaModel::load(&base_gguf).expect("load for hand");
    hand_merge(&mut hand, &adapter);
    let logits_hand = logits_decode(&hand, 3);

    assert_eq!(
        logits_merged, logits_hand,
        "library merge must match the hand-applied reference bit-for-bit"
    );
    assert!(
        max_abs_diff(&logits_merged, &logits_base) > 1e-4,
        "a nonzero adapter must change the logits"
    );

    let _ = std::fs::remove_file(&base_path);
    let _ = std::fs::remove_file(&adapter_path);
}

// ----------------------------------------------------------------------
// P2 — delta-applied logits ≡ merged (tolerance), and ≠ base
// ----------------------------------------------------------------------

#[test]
fn p2_delta_equals_merged_and_differs_from_base() {
    let (base_path, adapter_path) = write_base_and_adapter(0x3333);
    let base_gguf = Gguf::open(&base_path).expect("base");
    let adapter_gguf = Gguf::open(&adapter_path).expect("adapter");
    let adapter = Arc::new(LoraAdapter::from_gguf(&adapter_gguf, "a").expect("parse"));

    // Delta path: un-merged base + resident adapter, applied at dispatch.
    let base_model = LlamaModel::load(&base_gguf).expect("load base");
    let logits_base = logits_decode_lora(&base_model, 7, None);
    let logits_delta = logits_decode_lora(&base_model, 7, Some(&adapter));

    // Merged path: Phase-1 merge, no active adapter.
    let mut merged = LlamaModel::load(&base_gguf).expect("load merge");
    merged.weights.merge_lora(&adapter);
    let logits_merged = logits_decode(&merged, 7);

    let tol = 1e-2 * (1.0 + logits_merged.iter().map(|v| v.abs()).fold(0.0, f32::max));
    assert!(
        max_abs_diff(&logits_delta, &logits_merged) <= tol,
        "delta-applied logits must match merged within tolerance (diff {}, tol {})",
        max_abs_diff(&logits_delta, &logits_merged),
        tol
    );
    assert!(
        max_abs_diff(&logits_delta, &logits_base) > 1e-4,
        "the delta must actually change the base output"
    );

    let _ = std::fs::remove_file(&base_path);
    let _ = std::fs::remove_file(&adapter_path);
}

#[test]
fn p2_prefill_delta_differs_from_base() {
    let (base_path, adapter_path) = write_base_and_adapter(0x4444);
    let base_gguf = Gguf::open(&base_path).expect("base");
    let adapter_gguf = Gguf::open(&adapter_path).expect("adapter");
    let adapter = Arc::new(LoraAdapter::from_gguf(&adapter_gguf, "a").expect("parse"));
    let model = LlamaModel::load(&base_gguf).expect("load");

    let toks = [3i32, 5, 7, 11];
    let mut kv_b = KvCache::new(&model.cfg, 16);
    let base = model.forward_prefill_with_lora(&toks, 0, &mut kv_b, None);
    let mut kv_l = KvCache::new(&model.cfg, 16);
    let lora = model.forward_prefill_with_lora(&toks, 0, &mut kv_l, Some(&adapter));
    assert!(
        max_abs_diff(&base, &lora) > 1e-4,
        "prefill must apply the adapter delta too"
    );

    let _ = std::fs::remove_file(&base_path);
    let _ = std::fs::remove_file(&adapter_path);
}

// ----------------------------------------------------------------------
// P3 — multi-LoRA: per-request select, base-identical, concurrency
// ----------------------------------------------------------------------

#[test]
fn p3_base_selection_is_byte_identical() {
    let (base_path, adapter_path) = write_base_and_adapter(0x5555);
    let base_gguf = Gguf::open(&base_path).expect("base");
    let model = LlamaModel::load(&base_gguf).expect("load");

    let plain = logits_decode(&model, 9);
    let none = logits_decode_lora(&model, 9, None);
    assert_eq!(
        plain, none,
        "selecting no adapter must be byte-identical to the un-adapted forward"
    );

    let _ = std::fs::remove_file(&base_path);
    let _ = std::fs::remove_file(&adapter_path);
}

#[test]
fn p3_registry_and_per_request_selection() {
    // Two distinct adapters (different seeds) held resident in a registry.
    let base_path = tmp("base-reg");
    rustllama_gguf::synth::write_synthetic_llama_gguf(&base_path, &SynthLlama::default());
    let pa = tmp("reg-a");
    let pb = tmp("reg-b");
    write_synthetic_lora_adapter_gguf(
        &pa,
        &SynthLoraAdapter {
            base: SynthLlama::default(),
            rank: 8,
            alpha: 16.0,
            seed: 0xAAAA,
        },
    );
    write_synthetic_lora_adapter_gguf(
        &pb,
        &SynthLoraAdapter {
            base: SynthLlama::default(),
            rank: 8,
            alpha: 16.0,
            seed: 0xBBBB,
        },
    );
    let base_gguf = Gguf::open(&base_path).expect("base");
    let model = Arc::new(LlamaModel::load(&base_gguf).expect("load"));

    let mut reg = lora::LoraRegistry::new();
    reg.insert(LoraAdapter::from_gguf(&Gguf::open(&pa).unwrap(), "lora-a").unwrap());
    reg.insert(LoraAdapter::from_gguf(&Gguf::open(&pb).unwrap(), "lora-b").unwrap());
    assert_eq!(reg.len(), 2);
    let ad_a = reg.get("lora-a").expect("lora-a resident");
    let ad_b = reg.get("lora-b").expect("lora-b resident");
    assert!(reg.get("missing").is_none());

    // Single-adapter references.
    let ref_a = logits_decode_lora(&model, 4, Some(&ad_a));
    let ref_b = logits_decode_lora(&model, 4, Some(&ad_b));
    // Distinct adapters ⇒ distinct outputs.
    assert!(
        max_abs_diff(&ref_a, &ref_b) > 1e-4,
        "two different adapters must produce different logits"
    );

    // Selecting by name reproduces the single-adapter result.
    let sel_a = logits_decode_lora(&model, 4, reg.get("lora-a").as_ref());
    assert_eq!(
        sel_a, ref_a,
        "selecting adapter X ≡ the single-adapter X result"
    );

    // Two concurrent requests, different adapters, each matches its reference.
    let m2 = Arc::clone(&model);
    let a2 = Arc::clone(&ad_a);
    let b2 = Arc::clone(&ad_b);
    let ha = std::thread::spawn(move || logits_decode_lora(&m2, 4, Some(&a2)));
    let m3 = Arc::clone(&model);
    let hb = std::thread::spawn(move || logits_decode_lora(&m3, 4, Some(&b2)));
    let got_a = ha.join().expect("thread a");
    let got_b = hb.join().expect("thread b");
    assert_eq!(
        got_a, ref_a,
        "concurrent adapter A ≡ its single-adapter reference"
    );
    assert_eq!(
        got_b, ref_b,
        "concurrent adapter B ≡ its single-adapter reference"
    );

    for p in [&base_path, &pa, &pb] {
        let _ = std::fs::remove_file(p);
    }
}
