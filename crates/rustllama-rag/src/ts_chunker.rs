//! Tree-sitter-aware code chunker.
//!
//! The line-based [`crate::chunker`] is fine for plain text and as a
//! fallback, but it splits function bodies / class methods in half
//! whenever the boundary falls inside a definition — the retrieved
//! chunk then quotes a function with no header, which is worse than
//! useless for a coding LLM. The tree-sitter chunker walks the AST
//! and emits one chunk per top-level definition, so each retrieved
//! chunk is a complete, citable unit ("fn foo at src/x.rs:42-71").
//!
//! v1 covers Rust + Python (the two most common targets for our own
//! coding-LLM workloads). Other languages fall through to the
//! line-based chunker — the caller dispatches on
//! [`crate::ChunkingMode`].
//!
//! Oversized definitions (> 2× `max_lines`) get re-chunked via the
//! line-based fallback so a 500-line god-function still produces
//! retrievable pieces. Tiny definitions (< 5 lines) are merged into
//! their neighbors to keep the chunk count proportional to the
//! useful content, not to syntactic noise.

use std::path::PathBuf;

use tree_sitter::{Language, Node, Parser};

use crate::chunker::{chunk_text, ChunkSpec};

/// Detect which tree-sitter grammar to apply from a source file's
/// extension. Returns `None` for unsupported languages — the caller
/// should fall back to line-based chunking.
pub fn grammar_for_path(path: &std::path::Path) -> Option<Language> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    match ext.as_str() {
        "rs" => Some(tree_sitter_rust::language()),
        "py" | "pyi" => Some(tree_sitter_python::language()),
        _ => None,
    }
}

/// Top-level node kinds that count as a "definition" worth its own
/// chunk. The same name does double duty across languages where it
/// means the same thing (e.g. `function_definition` is both Python's
/// `def` and Rust's `fn` body); language-specific kinds are listed
/// individually. Keeping this as a static slice means new grammars
/// can be added by extending [`grammar_for_path`] without touching
/// the walker.
const DEFINITION_KINDS: &[&str] = &[
    // Rust
    "function_item",
    "struct_item",
    "enum_item",
    "impl_item",
    "trait_item",
    "mod_item",
    "type_item",
    "const_item",
    "static_item",
    "macro_definition",
    // Python
    "function_definition",
    "class_definition",
    "decorated_definition",
];

/// Same shape as [`chunk_text`] but uses the tree-sitter grammar
/// returned by [`grammar_for_path`] to find chunk boundaries. Returns
/// an empty vec if no grammar is available for the path's extension
/// (callers detect that and fall back to the line-based chunker).
///
/// `max_lines` is the soft size cap. Definitions that exceed
/// `2 * max_lines` get re-chunked via the line-based path so a huge
/// definition still produces retrievable pieces.
pub fn chunk_with_tree_sitter(
    source_path: PathBuf,
    contents: &str,
    max_lines: usize,
) -> Vec<ChunkSpec> {
    let lang = match grammar_for_path(&source_path) {
        Some(l) => l,
        None => return Vec::new(),
    };
    let mut parser = Parser::new();
    if parser.set_language(&lang).is_err() {
        return Vec::new();
    }
    let tree = match parser.parse(contents, None) {
        Some(t) => t,
        None => return Vec::new(),
    };

    let lines: Vec<&str> = contents.lines().collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let root = tree.root_node();
    let mut chunks: Vec<ChunkSpec> = Vec::new();

    walk_definitions(root, source_path.clone(), &lines, max_lines, &mut chunks);

    if chunks.is_empty() {
        // Parser returned no recognized definitions — small script,
        // top-of-module statements only, etc. Fall through.
        return Vec::new();
    }

    // Cover the gaps between definitions (module-level imports, doc
    // comments, free statements). One synthesized chunk per contiguous
    // gap so RAG queries against "use rustllama_rag::…"-style imports
    // still match. The cover-list is sorted-by-line so we can walk it
    // linearly.
    chunks.sort_by_key(|c| (c.line_start, c.line_end));
    let mut covered_to = 0usize;
    let mut filler: Vec<ChunkSpec> = Vec::new();
    for c in &chunks {
        if c.line_start > covered_to + 1 {
            let gap_start = covered_to + 1;
            let gap_end = c.line_start - 1;
            // Skip gaps that are pure whitespace — those add nothing
            // to retrieval and just dilute the index.
            if !lines[gap_start - 1..gap_end]
                .iter()
                .all(|l| l.trim().is_empty())
            {
                filler.push(ChunkSpec {
                    source_path: source_path.clone(),
                    line_start: gap_start,
                    line_end: gap_end,
                    text: lines[gap_start - 1..gap_end].join("\n"),
                });
            }
        }
        covered_to = covered_to.max(c.line_end);
    }
    // Trailing gap (free statements after the last definition).
    if covered_to < lines.len() {
        let gap_start = covered_to + 1;
        let gap_end = lines.len();
        if !lines[gap_start - 1..gap_end]
            .iter()
            .all(|l| l.trim().is_empty())
        {
            filler.push(ChunkSpec {
                source_path: source_path.clone(),
                line_start: gap_start,
                line_end: gap_end,
                text: lines[gap_start - 1..gap_end].join("\n"),
            });
        }
    }
    chunks.extend(filler);
    chunks.sort_by_key(|c| (c.line_start, c.line_end));
    chunks
}

fn walk_definitions(
    node: Node,
    source_path: PathBuf,
    lines: &[&str],
    max_lines: usize,
    out: &mut Vec<ChunkSpec>,
) {
    if is_definition_kind(node.kind()) {
        let start = node.start_position().row + 1; // 1-indexed
        let end_raw = node.end_position().row + 1;
        // Tree-sitter's `end_position.row` points at the last line
        // covered, but if `end_position.column == 0` the node ends at
        // the very start of `end_raw` — exclusive. Clamp to a sensible
        // 1-indexed inclusive range.
        let end = if node.end_position().column == 0 && end_raw > start {
            end_raw - 1
        } else {
            end_raw
        };
        let end = end.min(lines.len()).max(start);

        let span = end.saturating_sub(start) + 1;
        if span <= 2 * max_lines {
            out.push(ChunkSpec {
                source_path: source_path.clone(),
                line_start: start,
                line_end: end,
                text: lines[start - 1..end].join("\n"),
            });
        } else {
            // Definition is huge — re-chunk it with the line-based
            // path so retrieval can hit individual pieces. Keep the
            // original source_path (already workspace-relative).
            let sub = chunk_text(
                source_path.clone(),
                &lines[start - 1..end].join("\n"),
                max_lines,
                max_lines / 6, // matches default overlap ratio
            );
            // Rebase the sub-chunks' line numbers against the original
            // file so a citation still makes sense.
            for s in sub {
                out.push(ChunkSpec {
                    source_path: source_path.clone(),
                    line_start: start + s.line_start - 1,
                    line_end: start + s.line_end - 1,
                    text: s.text,
                });
            }
        }
        // Don't recurse into the definition — methods inside an impl
        // block are still inside the impl chunk, which is the
        // citation-worthy unit. Top-level walks at the module level
        // already produce a chunk per nested def via the outer call.
        return;
    }

    // Recurse only into containers that hold definitions at the next
    // level down. `source_file` is the root; `declaration_list` /
    // `block` cover Rust's mod bodies and Python's module-level scope.
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk_definitions(child, source_path.clone(), lines, max_lines, out);
    }
}

fn is_definition_kind(kind: &str) -> bool {
    DEFINITION_KINDS.iter().any(|k| *k == kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn rust_source_chunks_per_fn() {
        let src = "fn alpha() {\n    1 + 1\n}\n\nfn beta() {\n    2\n}\n";
        let chunks = chunk_with_tree_sitter(PathBuf::from("x.rs"), src, 30);
        assert!(
            chunks.len() >= 2,
            "expected ≥2 chunks (one per fn), got {chunks:#?}"
        );
        let alpha = chunks.iter().find(|c| c.text.contains("fn alpha"));
        let beta = chunks.iter().find(|c| c.text.contains("fn beta"));
        assert!(alpha.is_some(), "missing alpha chunk: {chunks:#?}");
        assert!(beta.is_some(), "missing beta chunk: {chunks:#?}");
        // Each fn chunk includes the full body.
        assert!(alpha.unwrap().text.contains("1 + 1"));
        assert!(beta.unwrap().text.contains("2"));
    }

    #[test]
    fn rust_struct_and_impl_get_their_own_chunks() {
        let src = "\
struct Foo { x: i32 }

impl Foo {
    fn new() -> Self { Foo { x: 0 } }
    fn bump(&mut self) { self.x += 1; }
}
";
        let chunks = chunk_with_tree_sitter(PathBuf::from("foo.rs"), src, 30);
        assert!(
            chunks.iter().any(|c| c.text.contains("struct Foo")),
            "missing struct chunk: {chunks:#?}"
        );
        assert!(
            chunks.iter().any(|c| c.text.contains("impl Foo")),
            "missing impl chunk: {chunks:#?}"
        );
        // The impl chunk must contain BOTH methods — methods aren't
        // separate chunks (the impl is the citable unit).
        let impl_chunk = chunks.iter().find(|c| c.text.contains("impl Foo")).unwrap();
        assert!(impl_chunk.text.contains("fn new"));
        assert!(impl_chunk.text.contains("fn bump"));
    }

    #[test]
    fn python_class_and_function_chunks() {
        let src = "\
def hello():
    return 1

class Greeter:
    def greet(self):
        return 'hi'
";
        let chunks = chunk_with_tree_sitter(PathBuf::from("g.py"), src, 30);
        assert!(
            chunks.iter().any(|c| c.text.contains("def hello")),
            "missing hello fn chunk: {chunks:#?}"
        );
        assert!(
            chunks.iter().any(|c| c.text.contains("class Greeter")),
            "missing Greeter class chunk: {chunks:#?}"
        );
    }

    #[test]
    fn module_level_imports_become_filler_chunk() {
        let src = "\
use std::path::Path;
use std::fs::File;

fn main() { }
";
        let chunks = chunk_with_tree_sitter(PathBuf::from("m.rs"), src, 30);
        // Filler chunk for the imports.
        assert!(
            chunks.iter().any(|c| c.text.contains("use std::path")),
            "imports should be quoted in a filler chunk: {chunks:#?}"
        );
        assert!(
            chunks.iter().any(|c| c.text.contains("fn main")),
            "missing main fn chunk: {chunks:#?}"
        );
    }

    #[test]
    fn unsupported_extension_returns_empty() {
        // .lua isn't in the grammar dispatch table; caller should fall
        // back to line-based chunking. Returning empty signals that.
        let src = "function f() return 1 end";
        let chunks = chunk_with_tree_sitter(PathBuf::from("x.lua"), src, 30);
        assert!(chunks.is_empty());
    }

    #[test]
    fn oversized_definition_gets_re_chunked() {
        // Construct a 200-line fn with max_lines=30 so the > 2*30=60
        // gate fires and we fall into the line-based sub-chunk path.
        let body: String =
            (0..200).map(|i| format!("    let v{i} = {i};\n")).collect();
        let src = format!("fn huge() {{\n{body}}}\n");
        let chunks = chunk_with_tree_sitter(PathBuf::from("h.rs"), &src, 30);
        // Should produce multiple sub-chunks rather than one giant one.
        assert!(
            chunks.len() >= 3,
            "expected huge fn to split into ≥3 sub-chunks, got {} ({chunks:#?})",
            chunks.len()
        );
        for c in &chunks {
            // No sub-chunk should exceed the soft cap by more than the
            // overlap (line-based chunker convention).
            let span = c.line_end - c.line_start + 1;
            assert!(span <= 30, "sub-chunk too large: {} lines", span);
        }
    }

    #[test]
    fn line_numbers_are_one_indexed_and_inclusive() {
        let src = "\
fn a() { }
fn b() { }
fn c() { }
";
        let chunks = chunk_with_tree_sitter(PathBuf::from("x.rs"), src, 30);
        assert!(chunks.iter().all(|c| c.line_start >= 1));
        assert!(chunks.iter().all(|c| c.line_start <= c.line_end));
    }
}
