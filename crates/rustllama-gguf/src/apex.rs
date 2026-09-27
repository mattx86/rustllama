//! APEX (Adaptive Precision for EXpert models) profile recipes.
//!
//! Mirrors mudler/apex-quant's mixed-precision strategy: tensors
//! are classified by role (routed expert, shared expert, attention)
//! and layer position (edge layers higher precision, middle layers
//! lower) and assigned a per-tensor target dtype that minimizes
//! perplexity at the desired size budget.
//!
//! ## Tier overview (from the upstream README)
//!
//! - **I-Quality** (~21 GB on a 175B MoE): routed Q3_K–Q6_K layer-
//!   gradient, shared Q8_0, attention Q6_K. Beats Q8_0 at half size.
//! - **Quality** (~17 GB): routed Q3_K–Q5_K, shared Q8_0, attn Q6_K.
//! - **Balanced** (~14 GB): routed Q3_K–Q4_K, shared Q6_K, attn Q5_K.
//! - **Mini** (~12 GB): routed Q2_K–Q4_K, shared Q5_K, attn Q5_K.
//! - **Nano** (~10 GB, aggressive): routed Q2_K, shared Q4_K, attn Q4_K.
//!
//! Each tier emits a [`crate::recipe::RecipeRule`] list ordered
//! from most-specific to most-general. The CLI installs these
//! rules into the [`crate::quantize::QuantizePlan`] alongside any
//! user-provided recipe.
//!
//! ## Layer-position gradient
//!
//! The "edge layers" heuristic (boost precision for the first/last
//! 5 layers based on sensitivity analysis) is applied per-tier by
//! upgrading the affected layers' routed-expert tensors to Q6_K.
//! The function takes the model's total layer count so it can
//! generate edge-vs-middle rules dynamically — APEX recipes are
//! model-size-aware, not fixed text files.
//!
//! Detection of MoE vs dense: if the source GGUF has no
//! `*_exps.weight` tensors, the "routed expert" rules are no-ops
//! (the glob simply doesn't match anything) and the file falls back
//! to the attention + dense-FFN rules. APEX itself targets MoE
//! models but the rule shape degrades gracefully on dense.
//!
//! ## MTP / NextN floor
//!
//! Every tier — Nano included — pins the speculative-decoding MTP
//! (multi-token-prediction / NextN) head's linear weights at Q8_0.
//! Empirical finding (Colibrì): an MTP head quantized below 8 bits
//! per weight collapses draft acceptance to 0–4% (vs 39–59% at
//! int8+), erasing the entire speculative speedup for a few hundred
//! MB of savings. The floor rules are emitted FIRST so the walker's
//! first-match semantics keep them authoritative over any generic
//! rule added later.

use crate::recipe::RecipeRule;
use crate::GgmlType;

/// APEX tier — listed in the same order as the upstream README.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApexTier {
    /// I-Quality: largest, best perplexity. ~6 bpw effective.
    IQuality,
    /// Quality: smaller than I-Quality, still strong.
    Quality,
    /// Balanced: middle of the road. ~5 bpw effective.
    Balanced,
    /// Mini: aggressive size reduction with imatrix-supported quality.
    Mini,
    /// Nano: most aggressive — 2-bit routed experts.
    Nano,
}

impl ApexTier {
    /// Parse a tier name from the CLI flag value. Case-insensitive.
    pub fn parse(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "i-quality" | "iquality" | "i_quality" | "i" => Some(Self::IQuality),
            "quality" | "q" => Some(Self::Quality),
            "balanced" | "b" => Some(Self::Balanced),
            "mini" | "m" => Some(Self::Mini),
            "nano" | "n" => Some(Self::Nano),
            _ => None,
        }
    }

    /// Human-readable label for stats / log output.
    pub fn label(self) -> &'static str {
        match self {
            Self::IQuality => "APEX I-Quality",
            Self::Quality => "APEX Quality",
            Self::Balanced => "APEX Balanced",
            Self::Mini => "APEX Mini",
            Self::Nano => "APEX Nano",
        }
    }
}

/// Generate the concrete rule list for an APEX tier given the
/// model's transformer block count. Rules are emitted in
/// most-specific → least-specific order so the recipe walker's
/// first-match semantics select the right precision per tensor.
///
/// `n_layers` is `LlamaConfig.n_layers` from the source model. For
/// the layer-position gradient APEX uses the first 5 and last 5
/// layers as "edges" (or all layers if `n_layers ≤ 10`).
pub fn build_apex_rules(tier: ApexTier, n_layers: usize) -> Vec<RecipeRule> {
    let mut rules = Vec::new();

    // MTP / NextN head floor — first in the list so first-match-wins
    // pins these before any other rule can claim them. A draft head
    // below 8 bits per weight collapses speculative-decoding
    // acceptance to 0–4% (see the module docs), so every tier floors
    // the head's linear weights at Q8_0.
    //
    // Tensor names (from rustllama-models' loaders + llama.cpp GGUF
    // exports): the qwen35moe NextN head is
    // `blk.{N}.nextn.eh_proj.weight` (some exporters also emit
    // `embed_tokens` / `shared_head.head` under `nextn`); the
    // DeepSeek-style heads are `mtp.{i}.*` transformer-block linears
    // plus an optional per-head `mtp.{i}.output.weight`. Norm vectors
    // (`enorm`, `hnorm`, `shared_head_norm`, `attn_norm`, biases) are
    // deliberately NOT matched — like every other norm they fall
    // through to the pipeline's default high-precision handling.
    for pattern in [
        "blk.*.nextn.eh_proj.weight",
        "blk.*.nextn.embed_tokens.weight",
        "blk.*.nextn.shared_head.head.weight",
        "mtp.*.attn_q.weight",
        "mtp.*.attn_k.weight",
        "mtp.*.attn_v.weight",
        "mtp.*.attn_output.weight",
        "mtp.*.ffn_gate.weight",
        "mtp.*.ffn_up.weight",
        "mtp.*.ffn_down.weight",
        "mtp.*.output.weight",
    ] {
        rules.push(RecipeRule::new(pattern, GgmlType::Q8_0));
    }

    let edge_lo: usize = 5.min(n_layers);
    let edge_hi: usize = if n_layers >= 5 { n_layers - 5 } else { 0 };

    // Per-tier dtype assignments. Tuples are
    // (attention, shared_expert, routed_edge, routed_middle).
    let (attn, shared, routed_edge, routed_mid) = match tier {
        ApexTier::IQuality => (
            GgmlType::Q6_K,
            GgmlType::Q8_0,
            GgmlType::Q6_K,
            GgmlType::Q4_K,
        ),
        ApexTier::Quality => (
            GgmlType::Q6_K,
            GgmlType::Q8_0,
            GgmlType::Q5_K,
            GgmlType::Q3_K,
        ),
        ApexTier::Balanced => (
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::Q4_K,
            GgmlType::Q3_K,
        ),
        ApexTier::Mini => (
            GgmlType::Q5_K,
            GgmlType::Q5_K,
            GgmlType::Q4_K,
            GgmlType::Q2_K,
        ),
        ApexTier::Nano => (
            GgmlType::Q4_K,
            GgmlType::Q4_K,
            GgmlType::Q2_K,
            GgmlType::Q2_K,
        ),
    };

    // LM head: always at the highest precision used by the tier.
    // Mirrors llama.cpp's _M convention. Glob matches both
    // `output.weight` and architectures using `lm_head.weight`.
    rules.push(RecipeRule::new("output.weight", attn));
    rules.push(RecipeRule::new("lm_head.weight", attn));

    // Token embedding: same precision as the head.
    rules.push(RecipeRule::new("token_embd.weight", attn));

    // Per-layer routed-expert rules. APEX's edge layers (first
    // `edge_lo`, last 5) get routed_edge precision; middle layers
    // get routed_mid. Emit specific per-layer-index rules so the
    // first-match walker selects them before the generic catch-all.
    //
    // Routed-expert tensor names in GGUF (from rustllama-models'
    // MoE loader): `blk.{i}.ffn_gate_exps.weight`,
    // `blk.{i}.ffn_up_exps.weight`, `blk.{i}.ffn_down_exps.weight`.
    for i in 0..edge_lo {
        rules.push(RecipeRule::new(
            format!("blk.{i}.ffn_gate_exps.weight"),
            routed_edge,
        ));
        rules.push(RecipeRule::new(
            format!("blk.{i}.ffn_up_exps.weight"),
            routed_edge,
        ));
        rules.push(RecipeRule::new(
            format!("blk.{i}.ffn_down_exps.weight"),
            routed_edge,
        ));
    }
    for i in edge_hi..n_layers {
        if i < edge_lo {
            continue; // Already emitted above for very small models.
        }
        rules.push(RecipeRule::new(
            format!("blk.{i}.ffn_gate_exps.weight"),
            routed_edge,
        ));
        rules.push(RecipeRule::new(
            format!("blk.{i}.ffn_up_exps.weight"),
            routed_edge,
        ));
        rules.push(RecipeRule::new(
            format!("blk.{i}.ffn_down_exps.weight"),
            routed_edge,
        ));
    }
    // Middle-layer routed experts — generic glob covering everything
    // we haven't already pinned with specific rules. The walker
    // takes the first match, so per-layer rules above win.
    rules.push(RecipeRule::new(
        "blk.*.ffn_gate_exps.weight",
        routed_mid,
    ));
    rules.push(RecipeRule::new(
        "blk.*.ffn_up_exps.weight",
        routed_mid,
    ));
    rules.push(RecipeRule::new(
        "blk.*.ffn_down_exps.weight",
        routed_mid,
    ));

    // Shared-expert FFN tensors (DeepSeek-V3 family). Always-active
    // so APEX gives them the higher per-tier shared precision.
    rules.push(RecipeRule::new(
        "blk.*.ffn_gate_shexp.weight",
        shared,
    ));
    rules.push(RecipeRule::new(
        "blk.*.ffn_up_shexp.weight",
        shared,
    ));
    rules.push(RecipeRule::new(
        "blk.*.ffn_down_shexp.weight",
        shared,
    ));

    // Attention projections. APEX assigns them uniformly across
    // layers (no edge gradient) per the README's "Q6_K uniformly"
    // strategy for I-Quality / Quality tiers.
    rules.push(RecipeRule::new("blk.*.attn_q.weight", attn));
    rules.push(RecipeRule::new("blk.*.attn_k.weight", attn));
    rules.push(RecipeRule::new("blk.*.attn_v.weight", attn));
    rules.push(RecipeRule::new("blk.*.attn_output.weight", attn));

    // Dense-FFN fallback (for non-MoE source files where the
    // `*_exps.weight` patterns above match nothing). Same precision
    // as routed_mid since dense models have only one FFN per layer.
    rules.push(RecipeRule::new("blk.*.ffn_gate.weight", routed_mid));
    rules.push(RecipeRule::new("blk.*.ffn_up.weight", routed_mid));
    rules.push(RecipeRule::new("blk.*.ffn_down.weight", routed_mid));

    rules
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipe::resolve_first_match;

    /// Pin the tier name parser against the documented variants.
    #[test]
    fn parse_apex_tier_accepts_canonical_names() {
        assert_eq!(ApexTier::parse("I-Quality"), Some(ApexTier::IQuality));
        assert_eq!(ApexTier::parse("iquality"), Some(ApexTier::IQuality));
        assert_eq!(ApexTier::parse("i"), Some(ApexTier::IQuality));
        assert_eq!(ApexTier::parse("Quality"), Some(ApexTier::Quality));
        assert_eq!(ApexTier::parse("Balanced"), Some(ApexTier::Balanced));
        assert_eq!(ApexTier::parse("MINI"), Some(ApexTier::Mini));
        assert_eq!(ApexTier::parse("nano"), Some(ApexTier::Nano));
        assert_eq!(ApexTier::parse("nonsense"), None);
    }

    /// I-Quality on a 30-layer model: edge layers (0..5 and 25..30)
    /// get Q6_K routed experts, middle layers get Q4_K, shared get
    /// Q8_0, attention gets Q6_K, LM head gets Q6_K.
    #[test]
    fn i_quality_assigns_expected_targets() {
        let rules = build_apex_rules(ApexTier::IQuality, 30);
        // Edge layers — routed experts at Q6_K.
        assert_eq!(
            resolve_first_match("blk.0.ffn_gate_exps.weight", &rules),
            Some(GgmlType::Q6_K)
        );
        assert_eq!(
            resolve_first_match("blk.29.ffn_down_exps.weight", &rules),
            Some(GgmlType::Q6_K)
        );
        // Middle layer — routed experts at Q4_K.
        assert_eq!(
            resolve_first_match("blk.15.ffn_gate_exps.weight", &rules),
            Some(GgmlType::Q4_K)
        );
        // Shared expert — Q8_0.
        assert_eq!(
            resolve_first_match("blk.15.ffn_gate_shexp.weight", &rules),
            Some(GgmlType::Q8_0)
        );
        // Attention — Q6_K.
        assert_eq!(
            resolve_first_match("blk.5.attn_q.weight", &rules),
            Some(GgmlType::Q6_K)
        );
        // LM head — Q6_K.
        assert_eq!(
            resolve_first_match("output.weight", &rules),
            Some(GgmlType::Q6_K)
        );
        // Random non-MoE tensor: caller falls through to plan
        // default (no rule matches `blk.0.attn_norm.weight`).
        assert_eq!(
            resolve_first_match("blk.0.attn_norm.weight", &rules),
            None
        );
    }

    /// Nano tier on a small (4-layer) model: every layer is an
    /// "edge" so all routed experts get Q2_K (the tier's routed_edge).
    #[test]
    fn nano_small_model_treats_all_layers_as_edge() {
        let rules = build_apex_rules(ApexTier::Nano, 4);
        for i in 0..4 {
            assert_eq!(
                resolve_first_match(&format!("blk.{i}.ffn_gate_exps.weight"), &rules),
                Some(GgmlType::Q2_K)
            );
        }
        // Even with no middle layers, the generic glob is still in
        // the list; it just doesn't match any of the 4 specific
        // layer indices above (the specific rules win first).
    }

    /// Balanced tier on a 32-layer model. Pin the layer-position
    /// gradient: layer 6 is middle (Q3_K), layer 26 is edge (Q4_K).
    #[test]
    fn balanced_layer_position_gradient() {
        let rules = build_apex_rules(ApexTier::Balanced, 32);
        // Layer 4 is edge (< edge_lo=5) → Q4_K.
        assert_eq!(
            resolve_first_match("blk.4.ffn_gate_exps.weight", &rules),
            Some(GgmlType::Q4_K)
        );
        // Layer 6 is middle → Q3_K.
        assert_eq!(
            resolve_first_match("blk.6.ffn_gate_exps.weight", &rules),
            Some(GgmlType::Q3_K)
        );
        // Layer 27 is edge (>= edge_hi=27 → in range 27..32) → Q4_K.
        assert_eq!(
            resolve_first_match("blk.27.ffn_gate_exps.weight", &rules),
            Some(GgmlType::Q4_K)
        );
        // Shared expert: Q6_K for Balanced.
        assert_eq!(
            resolve_first_match("blk.10.ffn_up_shexp.weight", &rules),
            Some(GgmlType::Q6_K)
        );
    }

    /// Dense (non-MoE) source: APEX rules don't match the
    /// `*_exps.weight` patterns but DO match the dense FFN tensors
    /// via the generic `blk.*.ffn_*.weight` rule.
    #[test]
    fn dense_model_falls_through_to_ffn_generic_rule() {
        let rules = build_apex_rules(ApexTier::Mini, 16);
        // No routed-expert rule matches a dense `ffn_gate.weight`
        // (it doesn't end in `_exps.weight`). Generic dense rule
        // assigns Q2_K (the routed_mid for Mini).
        assert_eq!(
            resolve_first_match("blk.8.ffn_gate.weight", &rules),
            Some(GgmlType::Q2_K)
        );
        // Attention still hits its specific rule.
        assert_eq!(
            resolve_first_match("blk.8.attn_v.weight", &rules),
            Some(GgmlType::Q5_K)
        );
    }

    /// Every tier floors the MTP / NextN head's linear weights at
    /// Q8_0 (≥ 8 bits per weight). Below int8 the draft head's
    /// acceptance collapses to 0–4%, killing the speculative-decoding
    /// speedup — even Nano must not touch it.
    #[test]
    fn every_tier_floors_mtp_head_at_q8_0() {
        let tiers = [
            ApexTier::IQuality,
            ApexTier::Quality,
            ApexTier::Balanced,
            ApexTier::Mini,
            ApexTier::Nano,
        ];
        // qwen35moe NextN head (last-layer `blk.{N}.nextn.*`) plus
        // the DeepSeek-style `mtp.{i}.*` block linears.
        let head_linears = [
            "blk.40.nextn.eh_proj.weight",
            "blk.40.nextn.embed_tokens.weight",
            "blk.40.nextn.shared_head.head.weight",
            "mtp.0.attn_q.weight",
            "mtp.0.attn_output.weight",
            "mtp.0.ffn_down.weight",
            "mtp.1.output.weight",
        ];
        for tier in tiers {
            let rules = build_apex_rules(tier, 41);
            // The floor is emitted first so first-match-wins keeps it
            // authoritative over any generic rule.
            assert!(
                rules[0].pattern.contains("nextn"),
                "{tier:?}: MTP floor must lead the rule list"
            );
            for name in head_linears {
                let target = resolve_first_match(name, &rules)
                    .unwrap_or_else(|| panic!("{tier:?}: no rule matched {name}"));
                assert_eq!(target, GgmlType::Q8_0, "{tier:?}: {name}");
                assert!(
                    crate::quantize::dtype_at_least_8bpw(target),
                    "{tier:?}: {name} resolved below 8 bpw"
                );
            }
            // Head norms are NOT matched — like every other norm they
            // fall through to the pipeline's default handling.
            assert_eq!(
                resolve_first_match("blk.40.nextn.enorm.weight", &rules),
                None
            );
            assert_eq!(
                resolve_first_match("blk.40.nextn.shared_head_norm.weight", &rules),
                None
            );
            assert_eq!(
                resolve_first_match("mtp.0.attn_norm.weight", &rules),
                None
            );
        }
    }
}
