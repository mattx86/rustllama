//! Gitignore-aware directory walker. Wraps the `ignore` crate
//! (ripgrep's walker) with rustllama-specific defaults: a built-in
//! file-extension allow-list for "indexable code/text", and an
//! always-skip list for build artifacts that `.gitignore` doesn't
//! always cover (`target/`, `node_modules/`, `dist/`, `.next/`,
//! `__pycache__/`, etc.).
//!
//! Returns one [`SourceFile`] per file (path + UTF-8 contents). The
//! caller hands these to [`crate::chunker`] for windowing.

use std::path::{Path, PathBuf};

use ignore::WalkBuilder;

/// One file's text content, ready for chunking. `path` is relative
/// to the walker's root; `contents` is UTF-8 (binary files are
/// filtered upstream).
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub path: PathBuf,
    pub contents: String,
}

#[derive(Debug, thiserror::Error)]
pub enum WorkspaceWalkerError {
    #[error("root path does not exist: {0}")]
    RootMissing(PathBuf),
    #[error("root path is not a directory: {0}")]
    RootNotADirectory(PathBuf),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

/// File-extension allow-list. Conservatively scoped to the
/// programming-languages-and-text-formats coding-LLM workloads
/// actually care about. Binaries (`.png`, `.gguf`, `.exe`), media
/// (`.mp4`, `.wav`), and database dumps (`.sqlite`) are excluded.
///
/// The allow-list is a tuple of `(suffix, language_hint)` so callers
/// that want per-language behavior (e.g. picking a CommentStyle for
/// the code-grammar constraint) get the hint for free.
pub const INDEXABLE_EXTENSIONS: &[(&str, &str)] = &[
    // Programming languages
    (".rs", "rust"),
    (".py", "python"),
    (".js", "javascript"),
    (".jsx", "javascript"),
    (".ts", "typescript"),
    (".tsx", "typescript"),
    (".go", "go"),
    (".java", "java"),
    (".kt", "kotlin"),
    (".scala", "scala"),
    (".cs", "csharp"),
    (".swift", "swift"),
    (".c", "c"),
    (".cc", "cpp"),
    (".cpp", "cpp"),
    (".cxx", "cpp"),
    (".h", "c"),
    (".hpp", "cpp"),
    (".hh", "cpp"),
    (".rb", "ruby"),
    (".php", "php"),
    (".pl", "perl"),
    (".sh", "bash"),
    (".bash", "bash"),
    (".zsh", "bash"),
    (".fish", "bash"),
    (".lua", "lua"),
    (".r", "r"),
    (".jl", "julia"),
    (".ex", "elixir"),
    (".exs", "elixir"),
    (".dart", "dart"),
    (".m", "objective-c"),
    (".mm", "objective-c"),
    (".nim", "nim"),
    (".zig", "zig"),
    (".d", "d"),
    (".v", "verilog"),
    (".sv", "systemverilog"),
    // Markup + data formats
    (".md", "markdown"),
    (".mdx", "markdown"),
    (".rst", "rst"),
    (".txt", "text"),
    (".json", "json"),
    (".jsonc", "json"),
    (".yaml", "yaml"),
    (".yml", "yaml"),
    (".toml", "toml"),
    (".xml", "xml"),
    (".html", "html"),
    (".css", "css"),
    (".scss", "css"),
    (".sass", "css"),
    (".sql", "sql"),
    // Build / config (no extension would be handled separately;
    // these are common .* variants)
    (".dockerfile", "dockerfile"),
    (".cmake", "cmake"),
    (".gradle", "gradle"),
    (".ini", "ini"),
    (".conf", "ini"),
    (".env", "ini"),
];

/// Filenames worth indexing even without a recognized extension.
/// `Dockerfile`, `Makefile`, `Rakefile`, etc. are conventional code
/// artifacts that no extension matches.
pub const INDEXABLE_FILENAMES: &[(&str, &str)] = &[
    ("Dockerfile", "dockerfile"),
    ("Makefile", "make"),
    ("Rakefile", "ruby"),
    ("Gemfile", "ruby"),
    ("Pipfile", "toml"),
    ("README", "markdown"),
    ("LICENSE", "text"),
    ("CHANGELOG", "markdown"),
    ("AUTHORS", "text"),
];

/// Build-artifact directory names that some workspaces don't
/// `.gitignore` but that we never want to index. Saves us a redundant
/// `du -sh` of e.g. `target/debug/deps/`.
const ALWAYS_SKIP_DIRS: &[&str] = &[
    ".git",
    "target",
    "node_modules",
    "dist",
    ".next",
    ".nuxt",
    "build",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cargo",
    "venv",
    ".venv",
    "env",
    ".env-cache",
    ".tox",
    ".gradle",
    ".idea",
    ".vscode",
    "vendor",
    "Pods",
    "DerivedData",
];

/// Walker over a workspace root. Build via [`new`]; iterate via
/// [`files`] to get the lazy stream of [`SourceFile`].
///
/// Honors `.gitignore` + `.git/info/exclude` + global `$GIT_DIR/info/exclude`
/// out of the box (the `ignore` crate's default).
#[derive(Debug)]
pub struct WorkspaceWalker {
    root: PathBuf,
    /// Max bytes per file. Files above this are skipped — coding-LLM
    /// context windows mean indexing a 50MB CSV is wasted work; the
    /// useful chunks live in source code that's much smaller.
    max_file_bytes: u64,
}

impl WorkspaceWalker {
    /// Build a walker over `root`. Validates that the directory
    /// exists and is actually a directory.
    pub fn new(root: impl AsRef<Path>) -> Result<Self, WorkspaceWalkerError> {
        let root = root.as_ref().to_path_buf();
        if !root.exists() {
            return Err(WorkspaceWalkerError::RootMissing(root));
        }
        if !root.is_dir() {
            return Err(WorkspaceWalkerError::RootNotADirectory(root));
        }
        Ok(Self {
            root,
            max_file_bytes: 1_048_576, // 1 MB default
        })
    }

    /// Override the max-file-size limit. Default 1 MB.
    pub fn with_max_file_bytes(mut self, max: u64) -> Self {
        self.max_file_bytes = max;
        self
    }

    /// Lazily produce one [`SourceFile`] per indexable file. Files
    /// failing to decode as UTF-8 are skipped silently (most code
    /// is UTF-8; the few binaries that slip past the extension
    /// allow-list show up here).
    pub fn files(self) -> impl Iterator<Item = SourceFile> {
        let root = self.root.clone();
        let max = self.max_file_bytes;
        let walker = WalkBuilder::new(&self.root)
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .hidden(false) // walk hidden files — .env, .config etc. are
            // sometimes relevant for coding context
            .filter_entry(|entry| {
                // Custom predicate: reject ALWAYS_SKIP_DIRS by basename.
                let name = entry.file_name().to_string_lossy();
                if entry.file_type().map(|t| t.is_dir()).unwrap_or(false)
                    && ALWAYS_SKIP_DIRS.contains(&name.as_ref())
                {
                    return false;
                }
                true
            })
            .build();
        walker.filter_map(move |result| {
            let entry = result.ok()?;
            if !entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
                return None;
            }
            let path = entry.path();
            // Size cap.
            let meta = entry.metadata().ok()?;
            if meta.len() > max {
                return None;
            }
            if !is_indexable(path) {
                return None;
            }
            let contents = std::fs::read_to_string(path).ok()?;
            // Path relative to the walker's root for portability.
            let rel = path.strip_prefix(&root).unwrap_or(path).to_path_buf();
            Some(SourceFile {
                path: rel,
                contents,
            })
        })
    }
}

/// True if `path`'s extension or filename matches the allow-list.
fn is_indexable(path: &Path) -> bool {
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        let with_dot = format!(".{}", ext.to_ascii_lowercase());
        if INDEXABLE_EXTENSIONS
            .iter()
            .any(|(e, _)| *e == with_dot.as_str())
        {
            return true;
        }
    }
    if let Some(fname) = path.file_name().and_then(|s| s.to_str()) {
        if INDEXABLE_FILENAMES.iter().any(|(n, _)| *n == fname) {
            return true;
        }
    }
    false
}

/// Look up the language hint associated with a path's extension or
/// filename, for callers that want to drive per-language behavior
/// (e.g. the code-grammar comment-style picker).
pub fn language_hint(path: &Path) -> Option<&'static str> {
    if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
        let with_dot = format!(".{}", ext.to_ascii_lowercase());
        if let Some((_, lang)) = INDEXABLE_EXTENSIONS
            .iter()
            .find(|(e, _)| *e == with_dot.as_str())
        {
            return Some(lang);
        }
    }
    if let Some(fname) = path.file_name().and_then(|s| s.to_str()) {
        if let Some((_, lang)) = INDEXABLE_FILENAMES.iter().find(|(n, _)| *n == fname) {
            return Some(lang);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_nonexistent_root() {
        let p = std::env::temp_dir().join("rustllama-rag-test-nope");
        let _ = std::fs::remove_dir_all(&p);
        match WorkspaceWalker::new(&p) {
            Err(WorkspaceWalkerError::RootMissing(_)) => {}
            other => panic!("expected RootMissing, got {other:?}"),
        }
    }

    #[test]
    fn rejects_root_that_is_a_file() {
        let dir = tempfile::TempDir::new().unwrap();
        let file = dir.path().join("x.txt");
        std::fs::write(&file, "hi").unwrap();
        match WorkspaceWalker::new(&file) {
            Err(WorkspaceWalkerError::RootNotADirectory(_)) => {}
            other => panic!("expected RootNotADirectory, got {other:?}"),
        }
    }

    #[test]
    fn walks_indexable_files_and_skips_unrecognized_extensions() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("foo.rs"), "fn main() {}").unwrap();
        std::fs::write(dir.path().join("bar.py"), "print(1)").unwrap();
        std::fs::write(dir.path().join("baz.bin"), [0xff, 0xfe, 0xfd]).unwrap();
        let walker = WorkspaceWalker::new(dir.path()).unwrap();
        let files: Vec<SourceFile> = walker.files().collect();
        let names: Vec<String> = files
            .iter()
            .map(|f| f.path.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert!(names.contains(&"foo.rs".to_string()));
        assert!(names.contains(&"bar.py".to_string()));
        assert!(
            !names.contains(&"baz.bin".to_string()),
            "binary file should not be indexed: {names:?}"
        );
    }

    #[test]
    fn walks_indexable_filename_without_extension() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("Dockerfile"), "FROM scratch\n").unwrap();
        std::fs::write(dir.path().join("Makefile"), "all:\n\ttrue\n").unwrap();
        std::fs::write(dir.path().join("randomfile"), "nope\n").unwrap();
        let files: Vec<SourceFile> =
            WorkspaceWalker::new(dir.path()).unwrap().files().collect();
        let names: Vec<String> = files
            .iter()
            .map(|f| f.path.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert!(names.contains(&"Dockerfile".to_string()));
        assert!(names.contains(&"Makefile".to_string()));
        assert!(!names.contains(&"randomfile".to_string()));
    }

    #[test]
    fn skips_always_skip_dirs() {
        let dir = tempfile::TempDir::new().unwrap();
        // Real code file at root.
        std::fs::write(dir.path().join("foo.rs"), "fn x() {}").unwrap();
        // node_modules subdir with a file that WOULD otherwise be
        // indexable. The directory is on the always-skip list.
        let nm = dir.path().join("node_modules").join("pkg");
        std::fs::create_dir_all(&nm).unwrap();
        std::fs::write(nm.join("index.js"), "module.exports = 1;").unwrap();
        // target/ subdir likewise.
        let tgt = dir.path().join("target").join("debug");
        std::fs::create_dir_all(&tgt).unwrap();
        std::fs::write(tgt.join("foo.rs"), "// build artifact").unwrap();
        let files: Vec<SourceFile> =
            WorkspaceWalker::new(dir.path()).unwrap().files().collect();
        // Only `foo.rs` at root should appear.
        assert_eq!(files.len(), 1);
        assert_eq!(
            files[0]
                .path
                .file_name()
                .unwrap()
                .to_string_lossy(),
            "foo.rs"
        );
    }

    #[test]
    fn respects_gitignore() {
        let dir = tempfile::TempDir::new().unwrap();
        // Marker for `ignore` to treat this as a git repo root.
        std::fs::create_dir_all(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".gitignore"), "secret.rs\n").unwrap();
        std::fs::write(dir.path().join("public.rs"), "fn pub_() {}").unwrap();
        std::fs::write(dir.path().join("secret.rs"), "fn secret() {}").unwrap();
        let files: Vec<SourceFile> =
            WorkspaceWalker::new(dir.path()).unwrap().files().collect();
        let names: Vec<String> = files
            .iter()
            .map(|f| f.path.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert!(
            names.contains(&"public.rs".to_string()),
            "public.rs missing from index: {names:?}"
        );
        assert!(
            !names.contains(&"secret.rs".to_string()),
            "secret.rs should be gitignored: {names:?}"
        );
    }

    #[test]
    fn enforces_max_file_size_cap() {
        let dir = tempfile::TempDir::new().unwrap();
        std::fs::write(dir.path().join("small.rs"), "fn x() {}").unwrap();
        // 1.5 KB file with a 1 KB cap.
        let big_text = "// padding\n".repeat(200);
        std::fs::write(dir.path().join("big.rs"), &big_text).unwrap();
        let walker = WorkspaceWalker::new(dir.path())
            .unwrap()
            .with_max_file_bytes(1024);
        let files: Vec<SourceFile> = walker.files().collect();
        let names: Vec<String> = files
            .iter()
            .map(|f| f.path.file_name().unwrap().to_string_lossy().to_string())
            .collect();
        assert!(names.contains(&"small.rs".to_string()));
        assert!(
            !names.contains(&"big.rs".to_string()),
            "1.5 KB file should be skipped under 1 KB cap"
        );
    }

    #[test]
    fn language_hint_routes_extensions_correctly() {
        let cases = [
            ("foo.rs", Some("rust")),
            ("bar.py", Some("python")),
            ("baz.ts", Some("typescript")),
            ("Dockerfile", Some("dockerfile")),
            ("Makefile", Some("make")),
            ("randomfile", None),
            ("foo.unknown", None),
        ];
        for (name, expected) in cases {
            assert_eq!(
                language_hint(std::path::Path::new(name)),
                expected,
                "language_hint({name})"
            );
        }
    }
}
