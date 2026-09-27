//! Workspace-aware RAG primitives: file walker, chunker, vector store.
//!
//! Three building blocks the engine layer composes into the full
//! retrieval-augmented-generation flow:
//!
//! 1. [`WorkspaceWalker`] — gitignore-aware directory walker. Skips
//!    `.git/`, `target/`, `node_modules/`, and anything `.gitignore`
//!    excludes by default. Yields one [`SourceFile`] per indexable
//!    file. File-extension allow-list (`.rs`, `.py`, `.ts`, …) drives
//!    what counts as "indexable" — binaries and assets are skipped.
//!
//! 2. [`chunker::chunk_text`] — sliding-window line-based chunker.
//!    Each chunk carries the source file + line range so retrieved
//!    results can be quoted back with line numbers. Defaults match
//!    Aider/Continue conventions: 30-line chunks with 5-line overlap.
//!
//! 3. [`RagIndex`] — in-memory vector store with cosine-similarity
//!    search. Linear scan; fine for ≤100K chunks (the typical
//!    coding-LLM workspace). HNSW or similar is a follow-up for
//!    huge workspaces.
//!
//! Embedding generation is NOT in this crate — the engine layer's
//! `/v1/embeddings` already does that. RAG callers embed query +
//! chunks separately and hand the vectors here.

pub mod chunker;
pub mod index;
#[cfg(feature = "tree-sitter")]
pub mod ts_chunker;
pub mod walker;
#[cfg(feature = "watcher")]
pub mod watcher;

pub use chunker::{chunk_text, ChunkSpec};
pub use index::{IndexError, IndexedChunk, RagIndex, SearchResult};
pub use walker::{SourceFile, WorkspaceWalker, WorkspaceWalkerError};
#[cfg(feature = "watcher")]
pub use watcher::{RagWatcher, WatcherError};

/// Chunking strategy. Use [`chunk_with_mode`] to dispatch from a value
/// of this enum so callers don't have to feature-gate at every call
/// site — `TreeSitter` silently falls back to `Lines` when either the
/// `tree-sitter` cargo feature is off or the file's language has no
/// registered grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkingMode {
    /// Pure line-based sliding window. Cheap and always available;
    /// the right default for non-code text or for builds that don't
    /// pull the tree-sitter grammar crates.
    Lines,
    /// AST-aware: one chunk per top-level definition (fn / struct /
    /// class / impl / …), with line-based fallback for unsupported
    /// languages and for oversized definitions. Requires the
    /// `tree-sitter` feature on `rustllama-rag` and a grammar for the
    /// file's extension; falls back to `Lines` cleanly otherwise.
    TreeSitter,
    /// Auto-pick: try `TreeSitter` first, fall back to `Lines` if no
    /// grammar matches. The right default for general-purpose
    /// indexing of a mixed workspace.
    Auto,
}

/// Mode-dispatched chunking entry point. `chunk_lines` + `overlap_lines`
/// only matter for the Lines path (and for tree-sitter's oversized-def
/// re-chunking fallback); TreeSitter uses `chunk_lines` as the
/// "definition too big" gate.
pub fn chunk_with_mode(
    source_path: std::path::PathBuf,
    contents: &str,
    chunk_lines: usize,
    overlap_lines: usize,
    mode: ChunkingMode,
) -> Vec<ChunkSpec> {
    match mode {
        ChunkingMode::Lines => {
            chunk_text(source_path, contents, chunk_lines, overlap_lines)
        }
        ChunkingMode::TreeSitter | ChunkingMode::Auto => {
            #[cfg(feature = "tree-sitter")]
            {
                let ts = ts_chunker::chunk_with_tree_sitter(
                    source_path.clone(),
                    contents,
                    chunk_lines.max(1),
                );
                if !ts.is_empty() {
                    return ts;
                }
            }
            chunk_text(source_path, contents, chunk_lines, overlap_lines)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_surface_exposes_three_primitives() {
        // Cheap pin so renames trip a CI failure rather than a silent
        // breakage in the engine wiring.
        let _walker = std::path::PathBuf::from(".");
        // The three top-level types should all be reachable via the
        // crate root.
        fn _typecheck_walker(_p: WorkspaceWalker) {}
        fn _typecheck_chunk(_c: ChunkSpec) {}
        fn _typecheck_index(_i: RagIndex) {}
    }
}
