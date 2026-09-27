//! Line-based sliding-window chunker for source files.
//!
//! Coding-LLM RAG benefits from chunks that preserve enough
//! context (multiple lines of a function) without being so large
//! that retrieval pulls in irrelevant material. Defaults match the
//! "30-line window, 5-line overlap" convention used by Aider and
//! Continue.dev — the overlap keeps function boundaries quoteable
//! when they fall on chunk boundaries.

use std::path::PathBuf;

/// One chunk of text from a source file, sized for embedding +
/// retrieval. `line_start..=line_end` are 1-indexed inclusive line
/// numbers so quoted results can be cited as "src/foo.rs:42-71".
#[derive(Debug, Clone)]
pub struct ChunkSpec {
    /// Path the chunk came from (relative to the walker's root).
    pub source_path: PathBuf,
    /// First line in this chunk (1-indexed, inclusive).
    pub line_start: usize,
    /// Last line in this chunk (1-indexed, inclusive).
    pub line_end: usize,
    /// The chunk's text content, joined with `\n`. Does NOT include
    /// a trailing newline.
    pub text: String,
}

/// Default chunk size (lines). Matches Aider's `--rag-chunk-size`
/// default and Continue.dev's text-chunker default.
pub const DEFAULT_CHUNK_LINES: usize = 30;

/// Default overlap between consecutive chunks (lines). 5 lines is
/// enough to keep most function signatures visible at both
/// chunk-tail and next-chunk-head.
pub const DEFAULT_OVERLAP_LINES: usize = 5;

/// Chunk a single file's contents using a sliding window. Returns
/// one [`ChunkSpec`] per window. Empty files yield no chunks.
///
/// `chunk_lines` must be > 0 and `> overlap_lines`. Invalid args
/// fall back to the defaults rather than panicking — RAG is a
/// best-effort feature and a bad config shouldn't tear down a
/// running indexer.
pub fn chunk_text(
    source_path: PathBuf,
    contents: &str,
    chunk_lines: usize,
    overlap_lines: usize,
) -> Vec<ChunkSpec> {
    let chunk_lines = if chunk_lines == 0 {
        DEFAULT_CHUNK_LINES
    } else {
        chunk_lines
    };
    let overlap_lines = if overlap_lines >= chunk_lines {
        // Overlap >= chunk would make stride zero and loop forever.
        // Clamp to half the chunk size as a sensible fallback.
        chunk_lines / 2
    } else {
        overlap_lines
    };
    // Stride between consecutive window starts. `chunk_lines -
    // overlap_lines` keeps each line visible in at most two chunks.
    let stride = chunk_lines - overlap_lines;

    let lines: Vec<&str> = contents.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let mut out: Vec<ChunkSpec> = Vec::new();
    let mut start_idx = 0usize;
    while start_idx < lines.len() {
        let end_idx = (start_idx + chunk_lines).min(lines.len());
        let text = lines[start_idx..end_idx].join("\n");
        out.push(ChunkSpec {
            source_path: source_path.clone(),
            line_start: start_idx + 1,
            line_end: end_idx,
            text,
        });
        // Last window already covered the tail — stop to avoid an
        // empty repeat.
        if end_idx == lines.len() {
            break;
        }
        start_idx += stride;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_lines(n: usize) -> String {
        (1..=n).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n")
    }

    #[test]
    fn empty_file_produces_no_chunks() {
        let chunks = chunk_text("foo.rs".into(), "", 30, 5);
        assert!(chunks.is_empty());
    }

    #[test]
    fn shorter_than_chunk_size_yields_one_chunk() {
        let text = make_lines(10); // 10 lines, chunk=30
        let chunks = chunk_text("foo.rs".into(), &text, 30, 5);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].line_start, 1);
        assert_eq!(chunks[0].line_end, 10);
    }

    #[test]
    fn sliding_window_with_overlap_produces_consecutive_chunks() {
        // 70 lines, chunk=30, overlap=5 → stride=25
        // Chunks: 1..30, 26..55, 51..70 (3 chunks)
        let text = make_lines(70);
        let chunks = chunk_text("foo.rs".into(), &text, 30, 5);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].line_start, 1);
        assert_eq!(chunks[0].line_end, 30);
        assert_eq!(chunks[1].line_start, 26);
        assert_eq!(chunks[1].line_end, 55);
        assert_eq!(chunks[2].line_start, 51);
        assert_eq!(chunks[2].line_end, 70);
    }

    #[test]
    fn chunk_text_field_contains_joined_lines() {
        let text = make_lines(5);
        let chunks = chunk_text("foo.rs".into(), &text, 30, 5);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "line 1\nline 2\nline 3\nline 4\nline 5");
    }

    #[test]
    fn zero_chunk_lines_falls_back_to_default() {
        let text = make_lines(50);
        let chunks = chunk_text("foo.rs".into(), &text, 0, 5);
        // 50 lines / default 30 / overlap 5 / stride 25 → 2 chunks
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn overlap_geq_chunk_size_falls_back_to_half_chunk() {
        let text = make_lines(20);
        // chunk=10, overlap=10 → would yield stride=0 (infinite loop).
        // Clamp to overlap=5 → stride=5.
        let chunks = chunk_text("foo.rs".into(), &text, 10, 10);
        // 20 lines / chunk=10 / stride=5 → starts at 1, 6, 11 →
        // chunks 1..10, 6..15, 11..20 (3 chunks)
        assert_eq!(chunks.len(), 3);
    }

    #[test]
    fn no_overlap_produces_disjoint_chunks() {
        let text = make_lines(60);
        let chunks = chunk_text("foo.rs".into(), &text, 20, 0);
        // stride = 20 → chunks 1..20, 21..40, 41..60 (3 chunks)
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].line_end, 20);
        assert_eq!(chunks[1].line_start, 21);
        assert_eq!(chunks[1].line_end, 40);
        assert_eq!(chunks[2].line_start, 41);
        assert_eq!(chunks[2].line_end, 60);
    }
}
