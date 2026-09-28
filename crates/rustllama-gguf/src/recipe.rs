//! Per-tensor target-dtype recipes for the [`crate::quantize`] pipeline.
//!
//! Two surfaces:
//!
//! 1. **Recipe files** — line-oriented text format mirroring
//!    llama.cpp's `--tensor-type-file`:
//!    ```text
//!    # Comments and blank lines OK.
//!    output.weight                Q6_K
//!    blk.*.attn_*.weight          Q6_K
//!    blk.*.ffn_*_exps.weight      Q4_K
//!    *                            Q4_K_M  (default fallback — written explicitly)
//!    ```
//!    Glob patterns support a single `*` wildcard each (matching any
//!    run of characters); exact strings match exactly. The pipeline
//!    walks rules **in order** and uses the **first match** per
//!    tensor — list specific rules before general ones.
//!
//! 2. **APEX profiles** (see [`crate::apex`]) — built-in named
//!    recipes that encode mudler/apex-quant's mixed-precision
//!    strategy. Produce the same `Vec<(String, GgmlType)>` shape as
//!    a recipe file but with concrete per-layer assignments derived
//!    from the model's actual layer count.

use std::fs;
use std::path::Path;

use crate::GgmlType;

#[derive(Debug, thiserror::Error)]
pub enum RecipeError {
    #[error("io reading recipe {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("recipe {path} line {line}: expected `pattern dtype`, got {content:?}")]
    BadFormat {
        path: String,
        line: usize,
        content: String,
    },
    #[error(
        "recipe {path} line {line}: unknown target dtype {dtype:?}. \
         Supported targets: f32, f16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0, q8_1, \
         q2_k, q3_k, q4_k, q5_k, q6_k, q8_k, tq1_0, tq2_0, iq4_nl, iq4_xs, \
         iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq1_s, iq1_m."
    )]
    UnknownDtype {
        path: String,
        line: usize,
        dtype: String,
    },
}

/// One rule: a glob pattern (with at most one `*` wildcard) paired
/// with the target dtype tensors matching it should be re-encoded to.
#[derive(Debug, Clone)]
pub struct RecipeRule {
    /// Pattern. May contain a single `*` wildcard at any position;
    /// exact strings (no `*`) match the tensor name literally.
    pub pattern: String,
    pub target: GgmlType,
}

impl RecipeRule {
    pub fn new(pattern: impl Into<String>, target: GgmlType) -> Self {
        Self {
            pattern: pattern.into(),
            target,
        }
    }

    /// Match `name` against this rule's pattern. Supports any
    /// number of `*` wildcards; each wildcard matches any (possibly
    /// empty) run of characters. Patterns with no `*` require an
    /// exact match.
    pub fn matches(&self, name: &str) -> bool {
        glob_match(&self.pattern, name)
    }
}

/// Glob match supporting any number of `*` wildcards. The matcher
/// is O(|pattern| × |name|) in the worst case (greedy search per
/// star), but in practice ggml tensor names + APEX patterns have
/// ≤ 2 stars and ≤ 64 characters, so it's effectively O(|pattern|).
pub fn glob_match(pattern: &str, name: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        // No wildcards — exact match required.
        return parts[0] == name;
    }
    // First segment must be a prefix of name.
    if !name.starts_with(parts[0]) {
        return false;
    }
    let mut cursor = parts[0].len();
    // Middle segments (`parts[1..len-1]`) must appear in order in
    // the remaining input, each AFTER the previous match position.
    for mid in &parts[1..parts.len() - 1] {
        if mid.is_empty() {
            // Adjacent stars (`**`) — collapse to a single wildcard.
            continue;
        }
        match name[cursor..].find(*mid) {
            Some(pos) => cursor += pos + mid.len(),
            None => return false,
        }
    }
    // Last segment must be a suffix of the remainder.
    let last = parts[parts.len() - 1];
    name[cursor..].ends_with(last)
}

/// Parse a recipe file. Returns the list of rules in file order;
/// the caller walks them in order and applies the first match per
/// tensor.
pub fn parse_recipe_file(path: impl AsRef<Path>) -> Result<Vec<RecipeRule>, RecipeError> {
    let path_ref = path.as_ref();
    let path_str = path_ref.display().to_string();
    let content = fs::read_to_string(path_ref).map_err(|source| RecipeError::Io {
        path: path_str.clone(),
        source,
    })?;
    parse_recipe_text(&content, &path_str)
}

/// Parse a recipe from a string. Same format as [`parse_recipe_file`];
/// exposed for tests + in-memory construction.
pub fn parse_recipe_text(content: &str, path_label: &str) -> Result<Vec<RecipeRule>, RecipeError> {
    let mut rules = Vec::new();
    for (idx, raw_line) in content.lines().enumerate() {
        let line_no = idx + 1;
        // Strip comment + trim.
        let line = match raw_line.find('#') {
            Some(p) => &raw_line[..p],
            None => raw_line,
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Split on any whitespace into exactly two tokens (pattern + dtype).
        let mut iter = line.split_whitespace();
        let pattern = iter.next().ok_or_else(|| RecipeError::BadFormat {
            path: path_label.to_string(),
            line: line_no,
            content: raw_line.to_string(),
        })?;
        let dtype_name = iter.next().ok_or_else(|| RecipeError::BadFormat {
            path: path_label.to_string(),
            line: line_no,
            content: raw_line.to_string(),
        })?;
        if iter.next().is_some() {
            return Err(RecipeError::BadFormat {
                path: path_label.to_string(),
                line: line_no,
                content: raw_line.to_string(),
            });
        }
        let target = parse_dtype_name(dtype_name).ok_or_else(|| RecipeError::UnknownDtype {
            path: path_label.to_string(),
            line: line_no,
            dtype: dtype_name.to_string(),
        })?;
        rules.push(RecipeRule::new(pattern, target));
    }
    Ok(rules)
}

/// Look up a [`GgmlType`] by its canonical lowercase name. Returns
/// `None` for unknown names; the caller wraps that as
/// `RecipeError::UnknownDtype`. Mirrors the CLI's
/// `parse_target_dtype` table — kept in sync via the alias test
/// below.
pub fn parse_dtype_name(name: &str) -> Option<GgmlType> {
    match name.to_ascii_lowercase().as_str() {
        "f32" => Some(GgmlType::F32),
        "f16" => Some(GgmlType::F16),
        "bf16" => Some(GgmlType::Bf16),
        "q4_0" => Some(GgmlType::Q4_0),
        "q4_1" => Some(GgmlType::Q4_1),
        "q5_0" => Some(GgmlType::Q5_0),
        "q5_1" => Some(GgmlType::Q5_1),
        "q8_0" => Some(GgmlType::Q8_0),
        "q8_1" => Some(GgmlType::Q8_1),
        "q2_k" => Some(GgmlType::Q2_K),
        "q3_k" => Some(GgmlType::Q3_K),
        "q4_k" => Some(GgmlType::Q4_K),
        "q5_k" => Some(GgmlType::Q5_K),
        "q6_k" => Some(GgmlType::Q6_K),
        "q8_k" => Some(GgmlType::Q8_K),
        "tq1_0" => Some(GgmlType::TQ1_0),
        "tq2_0" => Some(GgmlType::TQ2_0),
        "iq4_nl" => Some(GgmlType::IQ4_NL),
        "iq4_xs" => Some(GgmlType::IQ4_XS),
        "iq2_xxs" => Some(GgmlType::IQ2_XXS),
        "iq2_xs" => Some(GgmlType::IQ2_XS),
        "iq2_s" => Some(GgmlType::IQ2_S),
        "iq3_xxs" => Some(GgmlType::IQ3_XXS),
        "iq3_s" => Some(GgmlType::IQ3_S),
        "iq1_s" => Some(GgmlType::IQ1_S),
        "iq1_m" => Some(GgmlType::IQ1_M),
        _ => None,
    }
}

/// Resolve a tensor name to a target dtype by walking the rule list
/// in order, returning the first match. Returns `None` when no rule
/// matches — caller falls back to the plan's `default_target` or
/// passthrough.
pub fn resolve_first_match<'a>(
    name: &str,
    rules: impl IntoIterator<Item = &'a RecipeRule>,
) -> Option<GgmlType> {
    for rule in rules {
        if rule.matches(name) {
            return Some(rule.target);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_matches_exact_strings_without_star() {
        assert!(glob_match("output.weight", "output.weight"));
        assert!(!glob_match("output.weight", "output.bias"));
        assert!(!glob_match("output.weight", "OUTPUT.weight"));
    }

    #[test]
    fn glob_matches_prefix_star_suffix_patterns() {
        // Wildcard in the middle.
        assert!(glob_match("blk.*.attn_q.weight", "blk.0.attn_q.weight"));
        assert!(glob_match("blk.*.attn_q.weight", "blk.31.attn_q.weight"));
        assert!(!glob_match("blk.*.attn_q.weight", "blk.0.attn_k.weight"));
        // Wildcard at the start.
        assert!(glob_match("*.attn_q.weight", "blk.5.attn_q.weight"));
        assert!(!glob_match("*.attn_q.weight", "blk.5.attn_q.bias"));
        // Wildcard at the end.
        assert!(glob_match("blk.0.*", "blk.0.attn_q.weight"));
        assert!(!glob_match("blk.0.*", "blk.10.attn_q.weight"));
        // Empty wildcard expansion.
        assert!(glob_match("a*b", "ab"));
    }

    #[test]
    fn glob_rejects_when_prefix_or_suffix_too_long() {
        // Prefix longer than the name.
        assert!(!glob_match("blk.0.*", "blk"));
        // Suffix longer than the name remainder.
        assert!(!glob_match("*.weight", "x"));
    }

    #[test]
    fn parse_recipe_handles_comments_blanks_and_trailing_inline() {
        let content = r#"
# This is a comment
output.weight        Q6_K
# Another comment

blk.*.attn_q.weight  Q4_K  # inline comment
"#;
        let rules = parse_recipe_text(content, "<test>").unwrap();
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[0].pattern, "output.weight");
        assert_eq!(rules[0].target, GgmlType::Q6_K);
        assert_eq!(rules[1].pattern, "blk.*.attn_q.weight");
        assert_eq!(rules[1].target, GgmlType::Q4_K);
    }

    #[test]
    fn parse_recipe_rejects_malformed_lines() {
        // Single token, no dtype.
        let err = parse_recipe_text("output.weight\n", "<x>").unwrap_err();
        assert!(matches!(err, RecipeError::BadFormat { .. }));
        // Three tokens.
        let err = parse_recipe_text("a b c\n", "<x>").unwrap_err();
        assert!(matches!(err, RecipeError::BadFormat { .. }));
    }

    #[test]
    fn parse_recipe_rejects_unknown_dtype() {
        let err = parse_recipe_text("output.weight q42_k\n", "<x>").unwrap_err();
        match err {
            RecipeError::UnknownDtype { dtype, .. } => assert_eq!(dtype, "q42_k"),
            other => panic!("expected UnknownDtype, got {other:?}"),
        }
    }

    #[test]
    fn resolve_first_match_walks_rules_in_order() {
        let rules = vec![
            RecipeRule::new("output.weight", GgmlType::Q6_K),
            RecipeRule::new("blk.*.attn_*.weight", GgmlType::Q5_K),
            RecipeRule::new("*", GgmlType::Q4_K),
        ];
        assert_eq!(
            resolve_first_match("output.weight", &rules),
            Some(GgmlType::Q6_K)
        );
        assert_eq!(
            resolve_first_match("blk.5.attn_q.weight", &rules),
            Some(GgmlType::Q5_K)
        );
        assert_eq!(
            resolve_first_match("blk.5.ffn_gate.weight", &rules),
            Some(GgmlType::Q4_K)
        );
    }
}
