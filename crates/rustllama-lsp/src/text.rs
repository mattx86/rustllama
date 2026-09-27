//! Text-coordinate helpers used by the LSP session.
//!
//! Two subtleties the LSP spec imposes:
//!
//!   1. **Position is line + UTF-16 code unit** (not byte, not char).
//!      Astral-plane codepoints (emoji, some CJK) take 2 UTF-16 units
//!      but 4 UTF-8 bytes. We translate both ways carefully.
//!
//!   2. **Incremental edits** (`{ range, text }`) splice one range with
//!      arbitrary text. Multiple edits per `didChange` apply in order,
//!      each relative to the document state AFTER the previous one.

/// LSP position: 0-based line, 0-based character (UTF-16 code units).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Position {
    pub line: u32,
    pub character: u32,
}

/// Convert an LSP `Position` into a byte offset into `text`. Returns
/// `None` if the position is past the end (we clamp instead so a
/// position-on-empty-document still resolves to byte 0, which is what
/// editors expect when asking for completion in a fresh buffer).
pub fn position_to_byte_offset(text: &str, pos: Position) -> Option<usize> {
    let mut line_start = 0usize;
    let mut current_line: u32 = 0;
    let bytes = text.as_bytes();

    // Walk to the start of `pos.line`.
    while current_line < pos.line {
        match bytes[line_start..].iter().position(|&b| b == b'\n') {
            Some(rel) => line_start += rel + 1,
            None => {
                // Past-end-of-text: clamp to text length.
                return Some(text.len());
            }
        }
        current_line += 1;
    }

    // Now walk `pos.character` UTF-16 code units forward from line_start.
    let line_text = &text[line_start..];
    let mut byte = 0usize;
    let mut u16_units: u32 = 0;
    for ch in line_text.chars() {
        if u16_units >= pos.character {
            break;
        }
        // Stop at a hard newline — characters beyond the line end are
        // clamped to the line end (matches VS Code / rust-analyzer).
        if ch == '\n' {
            break;
        }
        u16_units += ch.len_utf16() as u32;
        byte += ch.len_utf8();
    }
    Some(line_start + byte)
}

/// Apply one incremental edit: replace `text[start..end]` with `replace`.
/// Silently clamps positions that overshoot the document — same forgiving
/// posture VS Code's client takes.
pub fn apply_incremental_edit(text: &mut String, start: Position, end: Position, replace: &str) {
    let start_byte = position_to_byte_offset(text, start).unwrap_or(text.len());
    let end_byte = position_to_byte_offset(text, end).unwrap_or(text.len());
    let (lo, hi) = if start_byte <= end_byte {
        (start_byte, end_byte)
    } else {
        (end_byte, start_byte)
    };
    text.replace_range(lo..hi, replace);
}

/// Truncate the FIM prefix to at most `max_bytes`, keeping the SUFFIX
/// portion of the input (the bytes nearest the cursor). Snaps the cut
/// point to the next char boundary so the truncated slice stays valid
/// UTF-8.
pub fn truncate_prefix(prefix: &str, max_bytes: usize) -> &str {
    if prefix.len() <= max_bytes {
        return prefix;
    }
    let target = prefix.len() - max_bytes;
    let mut cut = target;
    while cut < prefix.len() && !prefix.is_char_boundary(cut) {
        cut += 1;
    }
    &prefix[cut..]
}

/// Truncate the FIM suffix to at most `max_bytes`, keeping the PREFIX
/// portion (the bytes nearest the cursor). Snaps to a char boundary.
pub fn truncate_suffix(suffix: &str, max_bytes: usize) -> &str {
    if suffix.len() <= max_bytes {
        return suffix;
    }
    let mut cut = max_bytes;
    while cut > 0 && !suffix.is_char_boundary(cut) {
        cut -= 1;
    }
    &suffix[..cut]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pos(line: u32, character: u32) -> Position {
        Position { line, character }
    }

    #[test]
    fn position_to_byte_offset_simple_ascii() {
        let text = "fn main() {\n    let x = 1;\n}\n";
        assert_eq!(position_to_byte_offset(text, pos(0, 0)), Some(0));
        assert_eq!(position_to_byte_offset(text, pos(0, 2)), Some(2));
        // End of line 0: "fn main() {" is 11 chars / 11 UTF-16 units.
        assert_eq!(position_to_byte_offset(text, pos(0, 11)), Some(11));
        // Start of line 1.
        assert_eq!(position_to_byte_offset(text, pos(1, 0)), Some(12));
        // After "    let " on line 1: 8 chars in.
        assert_eq!(position_to_byte_offset(text, pos(1, 8)), Some(20));
        // Past-end-of-line clamps to line end (newline byte not consumed).
        assert_eq!(position_to_byte_offset(text, pos(1, 999)), Some(26));
        // Past-end-of-file clamps to text len.
        assert_eq!(position_to_byte_offset(text, pos(99, 0)), Some(text.len()));
    }

    #[test]
    fn position_to_byte_offset_handles_astral_plane() {
        // "ab😀cd" — the emoji 😀 is U+1F600, encoded as 2 UTF-16 units
        // and 4 UTF-8 bytes. Cursor at character=4 (after the emoji)
        // should land at byte 6 ('c').
        let text = "ab😀cd";
        assert_eq!(position_to_byte_offset(text, pos(0, 0)), Some(0));
        assert_eq!(position_to_byte_offset(text, pos(0, 2)), Some(2)); // after "ab"
        // The emoji counts as 2 UTF-16 units; cursor at char 4 is after it.
        assert_eq!(position_to_byte_offset(text, pos(0, 4)), Some(6)); // after "ab😀"
        assert_eq!(position_to_byte_offset(text, pos(0, 5)), Some(7));
    }

    #[test]
    fn apply_incremental_edit_replaces_range() {
        let mut t = String::from("hello world");
        // Replace "world" with "everyone": chars 6..11.
        apply_incremental_edit(&mut t, pos(0, 6), pos(0, 11), "everyone");
        assert_eq!(t, "hello everyone");
    }

    #[test]
    fn apply_incremental_edit_insert_at_cursor() {
        let mut t = String::from("hello");
        // Insertion: zero-width range at cursor + non-empty text.
        apply_incremental_edit(&mut t, pos(0, 5), pos(0, 5), ", world");
        assert_eq!(t, "hello, world");
    }

    #[test]
    fn apply_incremental_edit_multiline() {
        let mut t = String::from("line1\nline2\nline3");
        // Replace "line2\nline" → "TWO\nNEW".
        // Start at (1, 0), end at (2, 4): byte 6..15 (excluding the
        // newline since (2, 4) lands at line3's character 4 = byte 15).
        apply_incremental_edit(&mut t, pos(1, 0), pos(2, 4), "TWO\nNEW");
        assert_eq!(t, "line1\nTWO\nNEW3");
    }

    #[test]
    fn truncate_prefix_keeps_suffix_portion() {
        let p = "aaaaaaaaaaaa";
        let truncated = truncate_prefix(p, 5);
        assert_eq!(truncated, "aaaaa");
    }

    #[test]
    fn truncate_prefix_snaps_to_char_boundary() {
        // Cutting in the middle of a 4-byte codepoint must skip forward
        // to the next valid char start.
        let p = "abc😀def"; // a,b,c, [4-byte], d,e,f — 10 bytes
        let truncated = truncate_prefix(p, 4);
        // truncated should be valid UTF-8 with the last 4 bytes that
        // fit on a boundary: "def" (3 bytes — 4 bytes would land
        // inside the emoji).
        assert!(truncated.is_char_boundary(0));
        assert!(truncated.is_char_boundary(truncated.len()));
        assert!(truncated.ends_with("def"));
    }

    #[test]
    fn truncate_suffix_keeps_prefix_portion_at_boundary() {
        let s = "abc😀def";
        let truncated = truncate_suffix(s, 4);
        // Cut at byte 4 would land inside the emoji; snap back to 3.
        assert!(truncated.is_char_boundary(0));
        assert!(truncated.is_char_boundary(truncated.len()));
        assert_eq!(truncated, "abc");
    }

    #[test]
    fn truncate_short_input_returns_as_is() {
        assert_eq!(truncate_prefix("hi", 100), "hi");
        assert_eq!(truncate_suffix("hi", 100), "hi");
    }
}
