//! Translate HuggingFace `transformers` tensor names to rustllama's
//! GGUF naming convention.
//!
//! AWQ and GPTQ models ship with the HF layout, e.g.
//! `model.layers.0.self_attn.q_proj.qweight`. The rustllama
//! `LlamaModel::load` path reads GGUF names like `blk.0.attn_q.weight`.
//! This module is the lookup table the A-2b grouping step calls when
//! it builds the dequanted tensor set.
//!
//! The mapping is pure-function and based on the Llama-family
//! convention shared by Llama 2/3, Qwen2/2.5, Mistral, DeepSeek-Coder,
//! and Phi-3 (and their respective coder variants). MoE,
//! vision-tower, and reranker layouts are out of scope here; A-2 v1
//! targets dense Llama-family models, matching the v1 plan's coding-
//! model focus.

/// What kind of tensor the HF name refers to, plus the GGUF name we
/// emit for it. The `kind` discriminator drives the A-2b grouping
/// step: per-linear quant trios (`Qweight + Scales + Qzeros` / `+GIdx`)
/// share a target GGUF name and get dequantized together; non-quant
/// tensors (`Plain`) carry through as-is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfTensor {
    /// The rustllama / GGUF name this HF tensor (or trio) targets.
    pub gguf_name: String,
    /// Role within the safetensors-side group.
    pub kind: HfTensorKind,
}

/// Role of a safetensors tensor within its target GGUF tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HfTensorKind {
    /// Unquantized tensor (norm weights, embeddings on some
    /// checkpoints, etc.). The HF tensor's bytes map 1:1 to a GGUF
    /// tensor with the same shape (modulo dtype conversion).
    Plain,
    /// Packed 4-bit weight matrix for a quantized linear. One of a
    /// `(Qweight, Scales, Qzeros [, GIdx])` trio; A-2b joins them
    /// up by `gguf_name` and runs the AWQ / GPTQ dequant.
    Qweight,
    /// Per-group fp16 scales for the matching `Qweight`.
    Scales,
    /// Per-group packed int4 zero points.
    Qzeros,
    /// GPTQ actorder group-index vector. AWQ files never carry one.
    GIdx,
}

/// Suffix-driven kind detection. Returns the role of the trailing
/// segment (`.qweight`, `.scales`, …) or `Plain` for a `.weight`
/// suffix. Returns `None` if the suffix isn't one we recognize at
/// all — let A-2b decide whether to ignore it or fail.
fn kind_from_suffix(name: &str) -> Option<(HfTensorKind, &str)> {
    // Order matters: `g_idx` is checked before `.weight` so a
    // pathological name like `foo.g_idx.weight` (which shouldn't
    // exist) would be routed Plain.
    for (suffix, kind) in [
        (".qweight", HfTensorKind::Qweight),
        (".scales", HfTensorKind::Scales),
        (".qzeros", HfTensorKind::Qzeros),
        (".g_idx", HfTensorKind::GIdx),
        (".weight", HfTensorKind::Plain),
        (".bias", HfTensorKind::Plain),
    ] {
        if let Some(stem) = name.strip_suffix(suffix) {
            return Some((kind, stem));
        }
    }
    None
}

/// Translate a HuggingFace tensor name to its rustllama / GGUF
/// counterpart for the dense Llama-family layout.
///
/// Returns `None` for tensors the loader should skip (e.g. RoPE
/// inverse-frequency buffers that GGUF derives from config rather
/// than carrying as a tensor), or for names that don't fit any
/// known pattern. A-2b treats `None` as "ignore + log at debug"
/// rather than an error so a future HF tensor we don't yet
/// understand doesn't break the load path entirely.
pub fn map_hf_to_gguf(hf_name: &str) -> Option<HfTensor> {
    let (kind, stem) = kind_from_suffix(hf_name)?;

    // Top-level (non-layer) tensors.
    match stem {
        "model.embed_tokens" => {
            return Some(HfTensor {
                gguf_name: "token_embd.weight".into(),
                kind,
            });
        }
        "model.norm" => {
            return Some(HfTensor {
                gguf_name: "output_norm.weight".into(),
                kind,
            });
        }
        "lm_head" => {
            return Some(HfTensor {
                gguf_name: "output.weight".into(),
                kind,
            });
        }
        // `rotary_emb.inv_freq` — derived from rope_theta + head_dim,
        // never read from the file. Skip explicitly so we don't log
        // a warning on every load.
        _ if stem.ends_with(".rotary_emb.inv_freq") => return None,
        _ => {}
    }

    // Per-layer tensors: `model.layers.<idx>.<rest>`.
    let after_layers = stem.strip_prefix("model.layers.")?;
    let dot = after_layers.find('.')?;
    let layer_idx: u32 = after_layers[..dot].parse().ok()?;
    let rest = &after_layers[dot + 1..];

    let target_suffix = match rest {
        "input_layernorm" => "attn_norm",
        "post_attention_layernorm" => "ffn_norm",
        "self_attn.q_proj" => "attn_q",
        "self_attn.k_proj" => "attn_k",
        "self_attn.v_proj" => "attn_v",
        "self_attn.o_proj" => "attn_output",
        // Qwen3 / Qwen3-MoE per-head Q/K RMSNorm. 1-D `[head_dim]`
        // norm weights the builder loads into the block's q_norm /
        // k_norm slots (Plain tensors — not quantized).
        "self_attn.q_norm" => "attn_q_norm",
        "self_attn.k_norm" => "attn_k_norm",
        "mlp.gate_proj" => "ffn_gate",
        "mlp.up_proj" => "ffn_up",
        "mlp.down_proj" => "ffn_down",
        // Recognized-but-skipped per-layer keys (RoPE inv_freq is
        // already filtered above at the stem level; future variants
        // can be added here).
        _ => return None,
    };

    // Bias tensors share the trailing suffix in GGUF too — both
    // `.weight` and `.bias` are valid suffixes on these target
    // names. Q/K/V biases exist on Qwen2 / Qwen2.5-Coder. The
    // unified GGUF naming is `<target>.weight` and `<target>.bias`.
    //
    // For the quant trio (qweight/scales/qzeros/g_idx) we always
    // emit the `.weight` suffix because the dequant step's output
    // is the dense weight — the bias path is a `Plain` tensor
    // ending in `.bias`.
    let final_suffix = match kind {
        HfTensorKind::Plain => {
            // Disambiguate weight vs bias by re-inspecting the HF
            // tail. Default to ".weight" since most norms are
            // weights only.
            if hf_name.ends_with(".bias") {
                "bias"
            } else {
                "weight"
            }
        }
        _ => "weight",
    };
    Some(HfTensor {
        gguf_name: format!("blk.{layer_idx}.{target_suffix}.{final_suffix}"),
        kind,
    })
}

/// Translate a rustllama / GGUF tensor name **back** to its HuggingFace
/// `transformers` (mlx-lm) counterpart — the inverse of
/// [`map_hf_to_gguf`] over the dense Llama/Qwen2 tensor set.
///
/// Used by `quantize --to-mlx` so the written checkpoint carries the
/// module names real `mlx-community` models use
/// (`model.layers.0.self_attn.q_proj.weight` rather than
/// `blk.0.attn_q.weight`) and is a drop-in for upstream mlx-lm tooling.
/// The trailing `.weight` / `.bias` suffix is preserved (Qwen2/2.5 ships
/// Q/K/V biases).
///
/// Returns `None` for a GGUF name outside the standard set (fused / MoE /
/// vision tensors have no clean 1:1 HF inverse) or without a `.weight` /
/// `.bias` suffix, so the caller can bail or pass it through under its
/// original name rather than emit a wrong one.
pub fn map_gguf_to_hf(gguf_name: &str) -> Option<String> {
    // Split the trailing `.weight` / `.bias` (preserved across the map).
    let (stem, suffix) = if let Some(s) = gguf_name.strip_suffix(".weight") {
        (s, "weight")
    } else if let Some(s) = gguf_name.strip_suffix(".bias") {
        (s, "bias")
    } else {
        return None;
    };

    // Top-level (non-layer) tensors — the inverse of the stem matches in
    // `map_hf_to_gguf`.
    let top = match stem {
        "token_embd" => Some("model.embed_tokens"),
        "output_norm" => Some("model.norm"),
        "output" => Some("lm_head"),
        _ => None,
    };
    if let Some(hf_stem) = top {
        return Some(format!("{hf_stem}.{suffix}"));
    }

    // Per-layer tensors: `blk.<idx>.<rest>`.
    let after_blk = stem.strip_prefix("blk.")?;
    let dot = after_blk.find('.')?;
    let layer_idx: u32 = after_blk[..dot].parse().ok()?;
    let rest = &after_blk[dot + 1..];

    let hf_rest = match rest {
        "attn_norm" => "input_layernorm",
        "ffn_norm" => "post_attention_layernorm",
        "attn_q" => "self_attn.q_proj",
        "attn_k" => "self_attn.k_proj",
        "attn_v" => "self_attn.v_proj",
        "attn_output" => "self_attn.o_proj",
        "attn_q_norm" => "self_attn.q_norm",
        "attn_k_norm" => "self_attn.k_norm",
        "ffn_gate" => "mlp.gate_proj",
        "ffn_up" => "mlp.up_proj",
        "ffn_down" => "mlp.down_proj",
        _ => return None,
    };
    Some(format!("model.layers.{layer_idx}.{hf_rest}.{suffix}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embed_tokens_maps_to_token_embd() {
        assert_eq!(
            map_hf_to_gguf("model.embed_tokens.weight").unwrap(),
            HfTensor {
                gguf_name: "token_embd.weight".into(),
                kind: HfTensorKind::Plain,
            }
        );
    }

    #[test]
    fn model_norm_maps_to_output_norm() {
        assert_eq!(
            map_hf_to_gguf("model.norm.weight").unwrap().gguf_name,
            "output_norm.weight"
        );
    }

    #[test]
    fn lm_head_maps_to_output() {
        assert_eq!(
            map_hf_to_gguf("lm_head.weight").unwrap().gguf_name,
            "output.weight"
        );
    }

    #[test]
    fn attention_proj_names_round_trip() {
        for (hf, gguf) in [
            ("model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"),
            ("model.layers.0.self_attn.k_proj.weight", "blk.0.attn_k.weight"),
            ("model.layers.0.self_attn.v_proj.weight", "blk.0.attn_v.weight"),
            (
                "model.layers.0.self_attn.o_proj.weight",
                "blk.0.attn_output.weight",
            ),
        ] {
            let got = map_hf_to_gguf(hf).unwrap();
            assert_eq!(got.gguf_name, gguf, "hf={hf}");
            assert_eq!(got.kind, HfTensorKind::Plain);
        }
    }

    #[test]
    fn mlp_proj_names_map() {
        for (hf, gguf) in [
            ("model.layers.5.mlp.gate_proj.weight", "blk.5.ffn_gate.weight"),
            ("model.layers.5.mlp.up_proj.weight", "blk.5.ffn_up.weight"),
            ("model.layers.5.mlp.down_proj.weight", "blk.5.ffn_down.weight"),
        ] {
            assert_eq!(map_hf_to_gguf(hf).unwrap().gguf_name, gguf, "hf={hf}");
        }
    }

    #[test]
    fn layer_norm_names_map() {
        assert_eq!(
            map_hf_to_gguf("model.layers.3.input_layernorm.weight")
                .unwrap()
                .gguf_name,
            "blk.3.attn_norm.weight"
        );
        assert_eq!(
            map_hf_to_gguf("model.layers.3.post_attention_layernorm.weight")
                .unwrap()
                .gguf_name,
            "blk.3.ffn_norm.weight"
        );
    }

    #[test]
    fn quant_trio_shares_target_gguf_name() {
        // The whole point of the kind discriminator: A-2b groups by
        // gguf_name across the trio so qweight + scales + qzeros all
        // funnel into the same dense weight tensor.
        let q = map_hf_to_gguf("model.layers.2.mlp.gate_proj.qweight").unwrap();
        let s = map_hf_to_gguf("model.layers.2.mlp.gate_proj.scales").unwrap();
        let z = map_hf_to_gguf("model.layers.2.mlp.gate_proj.qzeros").unwrap();
        let g = map_hf_to_gguf("model.layers.2.mlp.gate_proj.g_idx").unwrap();
        assert_eq!(q.gguf_name, "blk.2.ffn_gate.weight");
        assert_eq!(s.gguf_name, "blk.2.ffn_gate.weight");
        assert_eq!(z.gguf_name, "blk.2.ffn_gate.weight");
        assert_eq!(g.gguf_name, "blk.2.ffn_gate.weight");
        assert_eq!(q.kind, HfTensorKind::Qweight);
        assert_eq!(s.kind, HfTensorKind::Scales);
        assert_eq!(z.kind, HfTensorKind::Qzeros);
        assert_eq!(g.kind, HfTensorKind::GIdx);
    }

    #[test]
    fn bias_suffix_routes_to_bias_target() {
        // Qwen2 / Qwen2.5-Coder carries Q/K/V biases.
        let b = map_hf_to_gguf("model.layers.7.self_attn.q_proj.bias").unwrap();
        assert_eq!(b.gguf_name, "blk.7.attn_q.bias");
        assert_eq!(b.kind, HfTensorKind::Plain);
    }

    #[test]
    fn rotary_emb_inv_freq_is_skipped() {
        // RoPE inv_freq is derived from rope_theta + head_dim at
        // engine init; we don't read or convert it.
        assert!(map_hf_to_gguf(
            "model.layers.0.self_attn.rotary_emb.inv_freq"
        )
        .is_none());
    }

    #[test]
    fn unrecognized_layer_subkey_returns_none() {
        // A future HF tensor we don't know how to map shouldn't
        // crash the loader — return None so A-2b can log + skip.
        assert!(map_hf_to_gguf("model.layers.0.future_subkey.weight").is_none());
    }

    #[test]
    fn name_without_known_suffix_returns_none() {
        // No `.weight` / `.bias` / quant suffix → reject.
        assert!(map_hf_to_gguf("model.embed_tokens").is_none());
        assert!(map_hf_to_gguf("model.layers.0").is_none());
    }

    #[test]
    fn high_layer_indices_parse_correctly() {
        // Real-world: Llama-3-70B has 80 layers, DeepSeek-V2 has 60.
        // Pin that layer-index parsing isn't capped at single digits.
        assert_eq!(
            map_hf_to_gguf("model.layers.79.self_attn.q_proj.qweight")
                .unwrap()
                .gguf_name,
            "blk.79.attn_q.weight"
        );
    }

    #[test]
    fn malformed_layer_index_is_rejected() {
        // `model.layers.abc.foo` shouldn't accidentally pass as
        // layer "abc". `parse::<u32>` returns Err → None.
        assert!(map_hf_to_gguf("model.layers.abc.self_attn.q_proj.weight").is_none());
    }

    #[test]
    fn names_with_unrelated_prefix_return_none() {
        // Tensor from some other framework's checkpoint that
        // accidentally got merged — must not be silently accepted.
        assert!(map_hf_to_gguf("vision_tower.layers.0.weight").is_none());
        assert!(map_hf_to_gguf("decoder.layers.0.self_attn.q_proj.weight").is_none());
    }

    // --- gguf -> hf inverse (quantize --to-mlx names) --------------------

    #[test]
    fn gguf_to_hf_maps_the_standard_set() {
        for (gguf, hf) in [
            ("token_embd.weight", "model.embed_tokens.weight"),
            ("output_norm.weight", "model.norm.weight"),
            ("output.weight", "lm_head.weight"),
            ("blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight"),
            ("blk.0.attn_k.weight", "model.layers.0.self_attn.k_proj.weight"),
            ("blk.0.attn_v.weight", "model.layers.0.self_attn.v_proj.weight"),
            ("blk.0.attn_output.weight", "model.layers.0.self_attn.o_proj.weight"),
            ("blk.3.ffn_gate.weight", "model.layers.3.mlp.gate_proj.weight"),
            ("blk.3.ffn_up.weight", "model.layers.3.mlp.up_proj.weight"),
            ("blk.3.ffn_down.weight", "model.layers.3.mlp.down_proj.weight"),
            ("blk.7.attn_norm.weight", "model.layers.7.input_layernorm.weight"),
            ("blk.7.ffn_norm.weight", "model.layers.7.post_attention_layernorm.weight"),
            // Qwen2 Q/K/V biases keep the `.bias` suffix.
            ("blk.2.attn_q.bias", "model.layers.2.self_attn.q_proj.bias"),
            ("blk.2.attn_k.bias", "model.layers.2.self_attn.k_proj.bias"),
            ("blk.2.attn_v.bias", "model.layers.2.self_attn.v_proj.bias"),
        ] {
            assert_eq!(map_gguf_to_hf(gguf).as_deref(), Some(hf), "gguf={gguf}");
        }
    }

    #[test]
    fn gguf_to_hf_round_trips_map_hf_to_gguf() {
        // For the standard dense set, gguf->hf should be the exact inverse
        // of hf->gguf: start from an HF name, map to GGUF, map back, and
        // land on the original HF name.
        for hf in [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.11.self_attn.o_proj.weight",
            "model.layers.5.mlp.gate_proj.weight",
            "model.layers.5.mlp.down_proj.weight",
            "model.layers.3.input_layernorm.weight",
            "model.layers.3.post_attention_layernorm.weight",
            "model.layers.7.self_attn.q_proj.bias",
        ] {
            let gguf = map_hf_to_gguf(hf).unwrap().gguf_name;
            assert_eq!(map_gguf_to_hf(&gguf).as_deref(), Some(hf), "round-trip hf={hf}");
        }
    }

    #[test]
    fn gguf_to_hf_rejects_unmapped_and_suffixless() {
        // No clean HF inverse (fused / unknown) or no weight/bias suffix.
        assert!(map_gguf_to_hf("blk.0.future_subkey.weight").is_none());
        assert!(map_gguf_to_hf("blk.abc.attn_q.weight").is_none());
        assert!(map_gguf_to_hf("token_embd").is_none());
        assert!(map_gguf_to_hf("blk.0.attn_q").is_none());
        assert!(map_gguf_to_hf("rope_freqs.weight").is_none());
    }
}
