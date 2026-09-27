//! On-disk session storage for the chat REPL.
//!
//! Sessions are stored as one JSON file per session under the user's
//! config dir (`%APPDATA%\rustllama\sessions\<name>.json` on Windows,
//! `~/.config/rustllama/sessions/<name>.json` on Linux). No database —
//! the v1 working set is small (tens of sessions × tens of messages)
//! and plain files keep them easy to back up, sync, and grep.

use std::fs;
use std::path::PathBuf;

use rustllama_client::ChatMessage;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Session {
    /// Stable session identifier — also the filename stem on disk.
    pub name: String,
    /// Model active when the session was saved (informational; the
    /// reloaded session uses whatever the REPL has selected now).
    pub model: String,
    pub messages: Vec<ChatMessage>,
    /// Epoch-seconds timestamps. Plain ints so the on-disk JSON is
    /// trivially diffable.
    pub created_at: u64,
    pub updated_at: u64,
}

#[derive(Debug, Serialize)]
pub struct SessionListEntry {
    pub name: String,
    pub model: String,
    pub message_count: usize,
    pub updated_at: u64,
}

fn sessions_dir() -> Option<PathBuf> {
    Some(rustllama_runtime::paths().sessions_dir.clone())
}

fn ensure_dir() -> Result<PathBuf, String> {
    let dir = sessions_dir().ok_or_else(|| "could not resolve session dir".to_string())?;
    fs::create_dir_all(&dir).map_err(|e| format!("mkdir {}: {e}", dir.display()))?;
    Ok(dir)
}

fn validate_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("session name must not be empty".into());
    }
    // Reject path separators / `..` / NUL so a hostile name can't escape
    // the sessions dir.
    if name
        .chars()
        .any(|c| c == '/' || c == '\\' || c == '\0' || c.is_control())
        || name == ".."
        || name.contains("..")
    {
        return Err(format!("illegal session name: {name:?}"));
    }
    Ok(())
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Save the given conversation to a JSON file under the sessions dir.
/// Returns the full path written. Overwrites any existing file with the
/// same name (after preserving its `created_at`, so updated_at moves
/// forward but the original timestamp stays accurate).
pub fn save(name: &str, model: &str, messages: &[ChatMessage]) -> Result<PathBuf, String> {
    validate_name(name)?;
    let dir = ensure_dir()?;
    let path = dir.join(format!("{name}.json"));

    let created_at = if path.exists() {
        load_from_path(&path)
            .map(|s| s.created_at)
            .unwrap_or_else(|_| now_secs())
    } else {
        now_secs()
    };

    let session = Session {
        name: name.to_string(),
        model: model.to_string(),
        messages: messages.to_vec(),
        created_at,
        updated_at: now_secs(),
    };
    let json = serde_json::to_string_pretty(&session).map_err(|e| format!("serialize: {e}"))?;
    fs::write(&path, json).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(path)
}

pub fn load(name: &str) -> Result<Session, String> {
    validate_name(name)?;
    let dir = sessions_dir().ok_or_else(|| "no session dir".to_string())?;
    let path = dir.join(format!("{name}.json"));
    load_from_path(&path)
}

fn load_from_path(path: &std::path::Path) -> Result<Session, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("parse {}: {e}", path.display()))
}

pub fn delete(name: &str) -> Result<(), String> {
    validate_name(name)?;
    let dir = sessions_dir().ok_or_else(|| "no session dir".to_string())?;
    let path = dir.join(format!("{name}.json"));
    if !path.exists() {
        return Err(format!("no such session: {name}"));
    }
    fs::remove_file(&path).map_err(|e| format!("remove {}: {e}", path.display()))
}

/// One hit from [`search`]: which session, which message index, and the
/// message's role + text. Returned in session-updated-newest-first order
/// so the most recent matches surface first.
#[derive(Debug, Serialize)]
pub struct SearchHit {
    pub session_name: String,
    pub message_index: usize,
    pub role: String,
    /// The matching message's content, truncated to ~200 chars with a
    /// trailing "…" if longer, so a search across long conversations
    /// stays readable in the terminal.
    pub snippet: String,
    pub session_updated_at: u64,
}

/// Search all sessions for `query`. When `case_sensitive` is false, both
/// the query and the message content are lowercased. When `name_filter`
/// is set, only sessions whose name contains that substring are
/// considered (also subject to the case-sensitivity flag).
pub fn search(
    query: &str,
    case_sensitive: bool,
    name_filter: Option<&str>,
) -> Result<Vec<SearchHit>, String> {
    let dir = match sessions_dir() {
        Some(d) => d,
        None => return Ok(Vec::new()),
    };
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let needle = if case_sensitive {
        query.to_string()
    } else {
        query.to_lowercase()
    };
    let name_needle = name_filter.map(|n| {
        if case_sensitive {
            n.to_string()
        } else {
            n.to_lowercase()
        }
    });

    let mut sessions: Vec<Session> = Vec::new();
    for entry in fs::read_dir(&dir).map_err(|e| format!("read_dir: {e}"))? {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let Ok(s) = load_from_path(&path) else {
            continue;
        };
        if let Some(ref nfilter) = name_needle {
            let haystack_name = if case_sensitive {
                s.name.clone()
            } else {
                s.name.to_lowercase()
            };
            if !haystack_name.contains(nfilter) {
                continue;
            }
        }
        sessions.push(s);
    }
    sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));

    let mut hits = Vec::new();
    for s in sessions {
        for (i, m) in s.messages.iter().enumerate() {
            let haystack = if case_sensitive {
                m.content.clone()
            } else {
                m.content.to_lowercase()
            };
            if haystack.contains(&needle) {
                hits.push(SearchHit {
                    session_name: s.name.clone(),
                    message_index: i,
                    role: m.role.clone(),
                    snippet: snippet_around(&m.content, &needle, case_sensitive),
                    session_updated_at: s.updated_at,
                });
            }
        }
    }
    Ok(hits)
}

/// Build a short context snippet around the first match of `needle` in
/// `content`. Centers the match in ~200 chars when possible, prepending
/// "…" if we cut the start and appending "…" if we cut the end.
fn snippet_around(content: &str, needle: &str, case_sensitive: bool) -> String {
    const RADIUS: usize = 100;
    let haystack = if case_sensitive {
        content.to_string()
    } else {
        content.to_lowercase()
    };
    let Some(idx) = haystack.find(needle) else {
        // Shouldn't happen — caller already matched — but be tolerant.
        return truncate(content, RADIUS * 2);
    };
    // Operate on chars (not bytes) so we don't bisect a multi-byte glyph.
    let char_indices: Vec<(usize, char)> = content.char_indices().collect();
    let target_byte = idx;
    let target_char_pos = char_indices
        .iter()
        .position(|(b, _)| *b >= target_byte)
        .unwrap_or(char_indices.len().saturating_sub(1));
    let start_char = target_char_pos.saturating_sub(RADIUS);
    let end_char = (target_char_pos + RADIUS).min(char_indices.len());
    let start_byte = char_indices.get(start_char).map(|(b, _)| *b).unwrap_or(0);
    let end_byte = char_indices
        .get(end_char)
        .map(|(b, _)| *b)
        .unwrap_or(content.len());
    let mut out = String::new();
    if start_char > 0 {
        out.push_str("…");
    }
    out.push_str(&content[start_byte..end_byte]);
    if end_char < char_indices.len() {
        out.push_str("…");
    }
    out
}

fn truncate(s: &str, max_chars: usize) -> String {
    let mut count = 0;
    let mut end_byte = s.len();
    for (b, _) in s.char_indices() {
        if count >= max_chars {
            end_byte = b;
            break;
        }
        count += 1;
    }
    if end_byte < s.len() {
        format!("{}…", &s[..end_byte])
    } else {
        s.to_string()
    }
}

/// Render a session as a portable markdown transcript: each message is
/// `## role` followed by a fenced code block when the content contains
/// triple backticks, otherwise a plain paragraph. Suitable for copying
/// into bug reports, GitHub issues, or a coding diary.
pub fn render_markdown(session: &Session) -> String {
    let mut out = String::new();
    out.push_str(&format!("# Session: {}\n\n", session.name));
    out.push_str(&format!("- **Model:** `{}`\n", session.model));
    out.push_str(&format!("- **Messages:** {}\n", session.messages.len()));
    out.push_str(&format!("- **Created:** {}\n", session.created_at));
    out.push_str(&format!("- **Updated:** {}\n\n", session.updated_at));
    out.push_str("---\n\n");
    for m in &session.messages {
        out.push_str(&format!("## {}\n\n", m.role));
        // If the message already contains triple-backtick fences, surround
        // it with quadruple-backtick fences so we don't break the user's
        // intended code blocks. Otherwise emit a plain paragraph.
        if m.content.contains("```") {
            out.push_str("````\n");
            out.push_str(&m.content);
            if !m.content.ends_with('\n') {
                out.push('\n');
            }
            out.push_str("````\n\n");
        } else {
            out.push_str(&m.content);
            if !m.content.ends_with('\n') {
                out.push('\n');
            }
            out.push('\n');
        }
    }
    out
}

/// Enumerate all sessions. Returns entries sorted newest-updated first.
/// Sessions that fail to parse are skipped (so a corrupt file doesn't
/// block listing the rest).
pub fn list() -> Result<Vec<SessionListEntry>, String> {
    let dir = match sessions_dir() {
        Some(d) => d,
        None => return Ok(Vec::new()),
    };
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in fs::read_dir(&dir).map_err(|e| format!("read_dir: {e}"))? {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        match load_from_path(&path) {
            Ok(s) => out.push(SessionListEntry {
                name: s.name,
                model: s.model,
                message_count: s.messages.len(),
                updated_at: s.updated_at,
            }),
            Err(_) => continue,
        }
    }
    out.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustllama_client::ChatMessage as CM;

    #[test]
    fn validate_name_rejects_path_traversal() {
        assert!(validate_name("ok-name_42").is_ok());
        assert!(validate_name("../etc").is_err());
        assert!(validate_name("a/b").is_err());
        assert!(validate_name("a\\b").is_err());
        assert!(validate_name("").is_err());
        assert!(validate_name("with\0null").is_err());
        // Plain `..` is rejected.
        assert!(validate_name("..").is_err());
    }

    fn sample(name: &str, updated_at: u64, msgs: Vec<(&str, &str)>) -> Session {
        Session {
            name: name.into(),
            model: "test-model".into(),
            messages: msgs
                .into_iter()
                .map(|(r, c)| CM {
                    role: r.into(),
                    content: c.into(),
                })
                .collect(),
            created_at: updated_at,
            updated_at,
        }
    }

    #[test]
    fn snippet_centers_match_and_marks_truncation() {
        let long = "a".repeat(300) + "needle" + &"b".repeat(300);
        let snip = super::snippet_around(&long, "needle", false);
        assert!(snip.starts_with("…"));
        assert!(snip.ends_with("…"));
        assert!(snip.contains("needle"));
        // Snippet should be much shorter than the full content but still
        // contain meaningful surrounding context (~200 chars).
        assert!(snip.chars().count() < 300);
        assert!(snip.chars().count() > 100);
    }

    #[test]
    fn snippet_handles_multibyte_chars_without_panicking() {
        let s = format!("{}日本語{}", "x".repeat(20), "y".repeat(20));
        let snip = super::snippet_around(&s, "日本語", true);
        assert!(snip.contains("日本語"));
    }

    #[test]
    fn snippet_short_input_returns_whole_string_no_ellipsis() {
        let snip = super::snippet_around("foo bar baz", "bar", false);
        assert_eq!(snip, "foo bar baz");
    }

    #[test]
    fn render_markdown_includes_header_and_messages() {
        let s = sample(
            "demo",
            1700_000_000,
            vec![("user", "What is 2+2?"), ("assistant", "4")],
        );
        let md = render_markdown(&s);
        assert!(md.contains("# Session: demo"));
        assert!(md.contains("- **Model:** `test-model`"));
        assert!(md.contains("## user"));
        assert!(md.contains("## assistant"));
        assert!(md.contains("What is 2+2?"));
        assert!(md.contains("\n4\n"));
    }

    #[test]
    fn render_markdown_wraps_code_blocks_with_quad_fences() {
        let s = sample(
            "code",
            1700_000_001,
            vec![("assistant", "Here is code:\n```rust\nfn main() {}\n```")],
        );
        let md = render_markdown(&s);
        // Quadruple fences so the user's ```rust block remains intact.
        assert!(md.contains("````\n"));
        assert!(md.contains("```rust"));
        // The wrapping `````` should appear before AND after the message.
        let count = md.matches("````").count();
        assert!(
            count >= 2,
            "expected at least 2 quad-fences, got {count}:\n{md}"
        );
    }

    /// Roundtrip a session through `save` + `search` to confirm the
    /// matching logic finds the right message and respects
    /// case-sensitivity. Uses a unique session name + cleans up after
    /// itself so the test doesn't leak files into the user's real
    /// sessions dir.
    #[test]
    fn search_finds_substring_across_sessions_case_insensitive() {
        let name = format!("rustllama-test-search-{}", now_secs());
        let saved = save(
            &name,
            "test-model",
            &[
                CM {
                    role: "user".into(),
                    content: "How do I use Tokio's mpsc channel?".into(),
                },
                CM {
                    role: "assistant".into(),
                    content: "Call tokio::sync::mpsc::channel(N) and split into (tx, rx).".into(),
                },
            ],
        );
        // Skip if we couldn't write (e.g., locked-down test env).
        let Ok(path) = saved else { return };

        let hits = search("mpsc", false, Some(&name)).expect("search");
        // Both messages contain "mpsc" — assert both surface.
        assert_eq!(hits.len(), 2, "expected 2 hits, got {hits:?}");
        assert!(hits.iter().any(|h| h.role == "user"));
        assert!(hits.iter().any(|h| h.role == "assistant"));
        assert!(hits.iter().all(|h| h.session_name == name));

        // Case-sensitive search for "MPSC" finds nothing.
        let hits_cs = search("MPSC", true, Some(&name)).expect("search");
        assert!(
            hits_cs.is_empty(),
            "case-sensitive should miss: {hits_cs:?}"
        );

        // Cleanup.
        let _ = std::fs::remove_file(&path);
    }
}
