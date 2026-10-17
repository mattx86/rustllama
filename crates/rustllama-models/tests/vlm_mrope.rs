//! M5-tail — multimodal RoPE (Qwen2-VL / Qwen2.5-VL) wired into the VLM
//! forward.
//!
//! Contracts pinned here (the forward-path side of the M5 follow-up; the
//! position-math + grid_thw derivation side is pinned in
//! `vision_arch.rs`'s unit tests):
//!
//!   1. **Byte-identical default.** On a model WITHOUT
//!      `rope.dimension_sections` (every text-only / non-VLM model),
//!      `forward_prefill_from_embeds_mrope` is bit-for-bit
//!      `forward_prefill_from_embeds` — the 3-D positions are ignored,
//!      nothing regresses.
//!   2. **Sectioned RoPE fires & is position-dependent.** With sections
//!      present, feeding DISTINCT per-axis `(t,h,w)` positions produces
//!      different logits than feeding the degenerate `(p,p,p)` text
//!      positions — i.e. the real 3-D grid actually drives the attention
//!      rotation in the forward (not a flat/placeholder assumption).
//!   3. **Text positions stay ~1-D.** With sections present, the
//!      degenerate `(p,p,p)` path matches the plain scalar forward (the
//!      sectioned kernel is bit-identical to 1-D RoPE at equal positions).

use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};
use rustllama_gguf::Gguf;
use rustllama_models::llama_arch::{KvCache, KvDtype, LlamaModel};

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn mrope_forward_wires_sectioned_rope_and_keeps_text_byte_identical() {
    let path = std::env::temp_dir().join("rustllama-vlm-mrope-fwd.gguf");
    write_synthetic_llama_gguf(&path, &SynthLlama::default());
    let gguf = Gguf::open(&path).expect("open synth llama gguf");
    let mut model = LlamaModel::load(&gguf).expect("load synth llama model");

    let d = model.cfg.d_model;
    let head_dim = model.cfg.head_dim; // 32 → head_dim/2 = 16 pairs
    assert_eq!(head_dim, 32, "fixture head_dim assumption");

    // A short spliced sequence (4 "tokens"); the embeds are just the text
    // embedding rows — the forward doesn't care whether they came from the
    // vision splice or the embed table.
    let n = 4usize;
    let ids: Vec<i32> = (0..n as i32).map(|i| i % model.cfg.vocab_size as i32).collect();
    let embeds = model.embed_tokens(&ids);
    assert_eq!(embeds.len(), n * d);

    // Build a fresh KV cache per run. Kept as short-lived borrows (not a
    // closure that captures `&model.cfg`) so `model.cfg` can be mutated
    // between runs to toggle the M-RoPE sections.
    let ctx = model.cfg.ctx_train;

    // Degenerate (text) positions (p,p,p), and a REAL image-grid layout: a
    // 1×2×2 patch grid at st=0 (what `mrope_positions(&[Image{1,2,2}], 0)`
    // produces). The grid coords vary h and w INDEPENDENTLY across tokens, so
    // the per-section relative positions genuinely differ from the text case
    // — unlike a uniform per-section offset, which RoPE (being relative)
    // would cancel.
    assert_eq!(n, 4, "the 1x2x2 image grid below assumes 4 tokens");
    let all_equal: Vec<[u32; 3]> = (0..n as u32).map(|p| [p, p, p]).collect();
    let distinct: Vec<[u32; 3]> = vec![[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]];

    // ---- Contract 1: sections ABSENT ⇒ byte-identical to the scalar entry.
    assert!(
        model.cfg.rope_dimension_sections.is_none(),
        "synth fixture must be a plain (non-sectioned) model"
    );
    let mut kv_a = KvCache::new_with_dtype(&model.cfg, ctx, KvDtype::F32);
    let scalar = model.forward_prefill_from_embeds(&embeds, 0, &mut kv_a);
    let mut kv_b = KvCache::new_with_dtype(&model.cfg, ctx, KvDtype::F32);
    let mrope_nosec = model.forward_prefill_from_embeds_mrope(&embeds, 0, &all_equal, &mut kv_b);
    assert_eq!(
        scalar, mrope_nosec,
        "without rope.dimension_sections the 3-D positions must be ignored \
         (bit-identical to forward_prefill_from_embeds)"
    );

    // Enable M-RoPE sections: [t,h,w] pair-widths summing to head_dim/2 = 16.
    model.cfg.rope_dimension_sections = Some(vec![8, 4, 4]);

    // ---- Contract 3: sections present + (p,p,p) ⇒ ~ the scalar forward.
    let mut kv_c = KvCache::new_with_dtype(&model.cfg, ctx, KvDtype::F32);
    let mrope_equal = model.forward_prefill_from_embeds_mrope(&embeds, 0, &all_equal, &mut kv_c);
    assert!(
        mrope_equal.iter().all(|v| v.is_finite()),
        "sectioned forward must be finite"
    );
    let text_vs_scalar = max_abs_diff(&scalar, &mrope_equal);
    assert!(
        text_vs_scalar < 1e-2,
        "text (p,p,p) positions through the sectioned kernel must match 1-D \
         RoPE (max|Δlogit| = {text_vs_scalar})"
    );

    // ---- Contract 2: the per-token 3-D GRID drives the rotation.
    // Compare two sectioned runs (both use the CPU sectioned kernel) that
    // differ ONLY in the 3-D positions:
    //   * `distinct` = a real 1×2×2 image grid — h and w vary independently
    //     across tokens, so the per-section RELATIVE positions genuinely
    //     change ⇒ a real effect on the attention/logits.
    //   * `uniform` = a uniform per-section offset of `all_equal`. RoPE is
    //     relative, so a constant per-section shift cancels in q·k and the
    //     attention output is unchanged ⇒ only fp-rounding noise.
    // The grid effect must dominate the uniform-offset floor — that is the
    // signal that the forward actually consumes the 3-D position (not just
    // "different numbers"), self-calibrated to this untrained model's scale.
    let uniform: Vec<[u32; 3]> = (0..n as u32).map(|p| [p, p + 5, p + 9]).collect();
    let mut kv_u = KvCache::new_with_dtype(&model.cfg, ctx, KvDtype::F32);
    let mrope_uniform = model.forward_prefill_from_embeds_mrope(&embeds, 0, &uniform, &mut kv_u);
    let uniform_floor = max_abs_diff(&mrope_equal, &mrope_uniform);

    let mut kv_d = KvCache::new_with_dtype(&model.cfg, ctx, KvDtype::F32);
    let mrope_distinct = model.forward_prefill_from_embeds_mrope(&embeds, 0, &distinct, &mut kv_d);
    let grid_effect = max_abs_diff(&mrope_equal, &mrope_distinct);

    assert!(
        grid_effect > 1e-7 && grid_effect > uniform_floor * 10.0,
        "a real image grid must change the forward output far more than a \
         (RoPE-cancelling) uniform per-section offset — grid max|Δ| = \
         {grid_effect}, uniform-offset floor = {uniform_floor}"
    );

    let _ = std::fs::remove_file(&path);
}
