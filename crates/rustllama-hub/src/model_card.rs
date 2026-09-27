//! HuggingFace model-card metadata: fetch, parse, persist as a JSON
//! sidecar next to the cached GGUF.
//!
//! Model cards on HF live at
//! `https://huggingface.co/<owner>/<repo>/raw/main/README.md` and start
//! with a YAML frontmatter block:
//!
//! ```yaml
//! ---
//! license: apache-2.0
//! tags:
//!   - code
//!   - llama
//! base_model: Qwen/Qwen2.5-Coder-7B-Instruct
//! ---
//! ```
//!
//! We support the subset of YAML that HF model cards actually use: flat
//! `key: value` pairs (optionally quoted) and `key:` followed by `- item`
//! list entries. Comments (`#`) and blank lines are ignored. Nested
//! mappings would need real YAML; we don't see those on model cards.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::HubRef;

/// Parsed HF model-card metadata. All fields optional — model cards in
/// the wild are inconsistent about which keys they set, and an absent
/// field is genuinely informative (not "we failed to parse").
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelCard {
    pub license: Option<String>,
    pub library_name: Option<String>,
    pub tags: Vec<String>,
    pub base_model: Option<String>,
    pub language: Vec<String>,
    pub model_creator: Option<String>,
    pub quantized_by: Option<String>,
    /// First non-empty paragraph of the README body. Markdown is kept
    /// verbatim — terminals render it fine and the GUI can re-parse it
    /// later. Capped at ~500 chars so a sidecar never balloons.
    pub description: Option<String>,
    /// `https://huggingface.co/<owner>/<repo>` — stable link back to
    /// the source. Always populated when fetched via this crate.
    pub source_url: Option<String>,
}

/// Sidecar filename inside `cache_dir/<owner>__<repo>/`. One card per
/// repo; multiple GGUF files in the same repo (quant variants) share it.
pub const SIDECAR_FILENAME: &str = "_card.json";

/// Local path of the sidecar for a given hub ref.
pub fn sidecar_path(hub_ref: &HubRef, cache_dir: &Path) -> PathBuf {
    cache_dir
        .join(format!("{}__{}", hub_ref.owner, hub_ref.repo))
        .join(SIDECAR_FILENAME)
}

/// Locate the sidecar JSON next to a cached GGUF file. Returns `None`
/// if the file's parent directory doesn't carry a `_card.json` (the
/// GGUF may have been added by hand or downloaded by another tool).
pub fn load_for_gguf(gguf_path: &Path) -> Option<ModelCard> {
    let card_path = gguf_path.parent()?.join(SIDECAR_FILENAME);
    let bytes = std::fs::read(&card_path).ok()?;
    serde_json::from_slice(&bytes).ok()
}

/// Save a model card next to the cached GGUF for later inspection.
pub fn save(card: &ModelCard, hub_ref: &HubRef, cache_dir: &Path) -> std::io::Result<PathBuf> {
    let path = sidecar_path(hub_ref, cache_dir);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(card)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(&path, json)?;
    Ok(path)
}

/// Fetch `<owner>/<repo>/raw/main/README.md` from HuggingFace and parse
/// it. Best-effort: returns `Ok(None)` if the repo doesn't have a
/// README or the fetch fails non-network-fatally (404, parse error). A
/// genuine network/IO failure surfaces as `Err`.
pub async fn fetch(hub_ref: &HubRef) -> crate::Result<Option<ModelCard>> {
    let url = format!(
        "https://huggingface.co/{}/{}/raw/main/README.md",
        hub_ref.owner, hub_ref.repo,
    );
    let client = reqwest::Client::builder()
        .user_agent(concat!("rustllama/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| crate::HubError::HfHub(e.to_string()))?;
    let resp = match client.get(&url).send().await {
        Ok(r) => r,
        Err(e) => return Err(crate::HubError::HfHub(e.to_string())),
    };
    if resp.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(None);
    }
    if !resp.status().is_success() {
        return Err(crate::HubError::HfHub(format!(
            "GET {} → {}",
            url,
            resp.status()
        )));
    }
    let text = resp
        .text()
        .await
        .map_err(|e| crate::HubError::HfHub(e.to_string()))?;
    let mut card = parse_readme(&text);
    card.source_url = Some(format!(
        "https://huggingface.co/{}/{}",
        hub_ref.owner, hub_ref.repo
    ));
    Ok(Some(card))
}

/// Parse a README's YAML frontmatter + first body paragraph into a
/// ModelCard. Public for testing.
pub fn parse_readme(text: &str) -> ModelCard {
    let mut card = ModelCard::default();
    let (frontmatter, body) = split_frontmatter(text);
    if let Some(fm) = frontmatter {
        apply_frontmatter(&mut card, fm);
    }
    if !body.is_empty() {
        card.description = first_paragraph(body);
    }
    card
}

/// Split `text` into `(Some(frontmatter_body), body)` if it starts with
/// `---\n...\n---\n` (or `...` as the closer). Returns `(None, text)`
/// when no frontmatter is present.
fn split_frontmatter(text: &str) -> (Option<&str>, &str) {
    let trimmed = text.trim_start_matches(['\u{feff}', ' ', '\t']);
    let Some(after_open) = trimmed.strip_prefix("---") else {
        return (None, text);
    };
    // Require a newline right after the opener so we don't false-match
    // `---title-like-thing`.
    let after_open = match after_open.strip_prefix('\n') {
        Some(s) => s,
        None => match after_open.strip_prefix("\r\n") {
            Some(s) => s,
            None => return (None, text),
        },
    };
    // Find the closer (`---` or `...`) on its own line.
    let mut search_from = 0;
    while search_from < after_open.len() {
        let line_end = after_open[search_from..]
            .find('\n')
            .map(|i| search_from + i)
            .unwrap_or(after_open.len());
        let line = after_open[search_from..line_end].trim_end_matches('\r');
        if line.trim() == "---" || line.trim() == "..." {
            let body = after_open
                .get(line_end + 1..)
                .unwrap_or("")
                .trim_start_matches('\n')
                .trim_start_matches("\r\n");
            return (Some(&after_open[..search_from]), body);
        }
        search_from = line_end + 1;
    }
    // Unterminated frontmatter: treat the whole document as body.
    (None, text)
}

fn apply_frontmatter(card: &mut ModelCard, fm: &str) {
    // State: when we see `key:` with no value, we expect subsequent
    // `- item` lines to form a list under that key.
    let mut pending_list_key: Option<String> = None;
    let mut pending_list: Vec<String> = Vec::new();

    let flush_list = |key: &str, items: Vec<String>, card: &mut ModelCard| {
        match key {
            "tags" => card.tags = items,
            "language" | "languages" => card.language = items,
            _ => {}
        }
    };

    for raw_line in fm.lines() {
        let line = raw_line.trim_end_matches('\r');
        let trimmed = line.trim_start();

        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // List continuation: `  - item` or `- item`.
        if let Some(rest) = trimmed.strip_prefix('-') {
            if pending_list_key.is_some() {
                let item = unquote(rest.trim());
                if !item.is_empty() {
                    pending_list.push(item);
                }
                continue;
            }
            // A `-` outside a list context — skip silently.
            continue;
        }

        // Flush the previous list (if any) before starting a new key.
        if let Some(k) = pending_list_key.take() {
            flush_list(&k, std::mem::take(&mut pending_list), card);
        }

        // Top-level `key: value` or `key:`.
        // We only honor keys that aren't indented (HF frontmatter is flat).
        if !line.starts_with(|c: char| c.is_whitespace()) {
            let (key, value) = match trimmed.split_once(':') {
                Some(kv) => kv,
                None => continue,
            };
            let key = key.trim();
            let value = value.trim();
            if value.is_empty() {
                // Start a list for this key.
                pending_list_key = Some(key.to_string());
                pending_list = Vec::new();
                continue;
            }
            let value = unquote(value);
            match key {
                "license" => card.license = Some(value),
                "library_name" => card.library_name = Some(value),
                "base_model" => card.base_model = Some(value),
                "model_creator" => card.model_creator = Some(value),
                "quantized_by" => card.quantized_by = Some(value),
                "tags" => {
                    // Inline form: `tags: [a, b, c]`.
                    if let Some(items) = parse_inline_seq(&value) {
                        card.tags = items;
                    }
                }
                "language" | "languages" => {
                    if let Some(items) = parse_inline_seq(&value) {
                        card.language = items;
                    }
                }
                _ => {}
            }
        }
    }

    // Flush trailing list.
    if let Some(k) = pending_list_key {
        flush_list(&k, pending_list, card);
    }
}

/// Strip a single layer of matching quotes from a YAML scalar.
fn unquote(s: &str) -> String {
    let s = s.trim();
    if s.len() >= 2 {
        let first = s.chars().next().unwrap();
        let last = s.chars().last().unwrap();
        if (first == '"' && last == '"') || (first == '\'' && last == '\'') {
            return s[1..s.len() - 1].to_string();
        }
    }
    s.to_string()
}

/// Parse an inline flow sequence `[a, b, "c d"]` into a list. Returns
/// `None` if the input isn't a `[...]` block.
fn parse_inline_seq(s: &str) -> Option<Vec<String>> {
    let s = s.trim();
    let inner = s.strip_prefix('[')?.strip_suffix(']')?;
    let out: Vec<String> = inner
        .split(',')
        .map(|p| unquote(p.trim()))
        .filter(|p| !p.is_empty())
        .collect();
    Some(out)
}

/// First non-empty paragraph of `body`, capped at 500 chars to keep
/// sidecars small.
fn first_paragraph(body: &str) -> Option<String> {
    let mut paragraph = String::new();
    for line in body.lines() {
        let l = line.trim_end_matches('\r').trim();
        if l.is_empty() {
            if !paragraph.is_empty() {
                break;
            }
            continue;
        }
        // Skip leading markdown headers / image refs — they're typically
        // not "the description" and clutter terminal output.
        if l.starts_with('#') || l.starts_with("![") || l.starts_with("---") {
            if paragraph.is_empty() {
                continue;
            } else {
                break;
            }
        }
        if !paragraph.is_empty() {
            paragraph.push(' ');
        }
        paragraph.push_str(l);
        if paragraph.len() > 500 {
            paragraph.truncate(500);
            paragraph.push('…');
            break;
        }
    }
    if paragraph.is_empty() {
        None
    } else {
        Some(paragraph)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_readme_extracts_basic_frontmatter() {
        let readme = "---\n\
license: apache-2.0\n\
library_name: transformers\n\
base_model: Qwen/Qwen2.5-Coder-7B-Instruct\n\
tags:\n\
- code\n\
- llama\n\
- gguf\n\
language:\n\
- en\n\
---\n\
\n\
# Title\n\
\n\
This model is a quantized version of Qwen2.5-Coder.\n\
";
        let card = parse_readme(readme);
        assert_eq!(card.license.as_deref(), Some("apache-2.0"));
        assert_eq!(card.library_name.as_deref(), Some("transformers"));
        assert_eq!(
            card.base_model.as_deref(),
            Some("Qwen/Qwen2.5-Coder-7B-Instruct")
        );
        assert_eq!(card.tags, vec!["code", "llama", "gguf"]);
        assert_eq!(card.language, vec!["en"]);
        assert!(card
            .description
            .as_deref()
            .map(|s| s.contains("quantized version"))
            .unwrap_or(false));
    }

    #[test]
    fn parse_readme_quoted_scalars() {
        let readme = "---\n\
license: \"mit\"\n\
model_creator: 'TheBloke'\n\
---\n\
Body.\n";
        let card = parse_readme(readme);
        assert_eq!(card.license.as_deref(), Some("mit"));
        assert_eq!(card.model_creator.as_deref(), Some("TheBloke"));
    }

    #[test]
    fn parse_readme_inline_sequence() {
        let readme = "---\n\
tags: [code, gguf, \"q4_k_m\"]\n\
---\n\
Body.\n";
        let card = parse_readme(readme);
        assert_eq!(card.tags, vec!["code", "gguf", "q4_k_m"]);
    }

    #[test]
    fn parse_readme_no_frontmatter() {
        let readme = "# Just a readme\n\nWith body text.\n";
        let card = parse_readme(readme);
        assert!(card.license.is_none());
        assert!(card.description.is_some());
    }

    #[test]
    fn parse_readme_unterminated_frontmatter_yields_no_metadata() {
        let readme = "---\nlicense: mit\n\nBody but the closer is missing.";
        let card = parse_readme(readme);
        assert!(card.license.is_none());
    }

    #[test]
    fn parse_readme_skips_comments_and_blanks() {
        let readme = "---\n\
# a comment\n\
license: apache-2.0\n\
\n\
# another comment\n\
tags:\n\
# inside the list\n\
- a\n\
- b\n\
---\n";
        let card = parse_readme(readme);
        assert_eq!(card.license.as_deref(), Some("apache-2.0"));
        assert_eq!(card.tags, vec!["a", "b"]);
    }

    #[test]
    fn first_paragraph_caps_at_500_with_ellipsis() {
        let body = "a ".repeat(400);
        let para = first_paragraph(&body).unwrap();
        assert!(para.ends_with('…'));
        // The paragraph itself (without the ellipsis) is at most 500.
        assert!(para.chars().count() <= 502);
    }

    #[test]
    fn first_paragraph_skips_headers_and_images() {
        let body = "# Title\n\n![alt](img.png)\n\nReal description here.\n";
        let para = first_paragraph(body).unwrap();
        assert_eq!(para, "Real description here.");
    }

    #[test]
    fn save_then_load_roundtrip() {
        let card = ModelCard {
            license: Some("apache-2.0".into()),
            tags: vec!["code".into(), "gguf".into()],
            ..Default::default()
        };
        let tmp = std::env::temp_dir().join("rustllama-modelcard-test");
        let hub_ref = HubRef::parse("owner/repo:file.gguf").unwrap();
        let saved = save(&card, &hub_ref, &tmp).expect("save");
        assert!(saved.exists());
        // Locate via the GGUF path neighbor lookup.
        let gguf_path = hub_ref.local_path(&tmp);
        let loaded = load_for_gguf(&gguf_path).expect("load");
        assert_eq!(loaded.license, card.license);
        assert_eq!(loaded.tags, card.tags);
        // Cleanup.
        let _ = std::fs::remove_dir_all(&tmp);
    }
}
