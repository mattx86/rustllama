//! Optional file-system watcher for incremental RAG indexing.
//!
//! Wraps `notify-debouncer-full` (which itself wraps `notify`) so the
//! caller gets a coalesced stream of "these files changed" events
//! instead of the raw bursty write/modify/rename storm editors emit
//! when saving. The default 1-second debounce window matches what
//! Aider, Continue.dev, and similar coding-assistant indexers settle
//! on — long enough to catch a vim swap-file dance, short enough that
//! the user doesn't notice the delay.
//!
//! Off by default; opt in via this crate's `watcher` feature. Callers
//! that don't want the `notify` + `notify-debouncer-full` deps in
//! their build (e.g., the v1 server that only exposes manual `/v1/rag/index`
//! + `/v1/rag/update`) can leave the feature off.
//!
//! The watcher API is intentionally minimal: build it pointed at a
//! workspace root, get a `Receiver<HashSet<PathBuf>>` back, and read
//! the receiver in a background task. The caller decides what to do
//! per event (re-walk + re-embed via the existing engine bundle).
//! The watcher itself doesn't know about embeddings or indexes — it
//! just emits paths.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use notify_debouncer_full::{new_debouncer, DebounceEventResult, Debouncer, FileIdMap};

/// Default debounce window. Matches editor save-cycle granularity —
/// vim's `:w` typically produces 3-5 raw events within ~50ms; an IDE
/// rename emits write+remove+rename in ~200ms. One second flattens
/// both into a single delivery.
pub const DEFAULT_DEBOUNCE_MS: u64 = 1_000;

/// Owns the underlying `notify-debouncer-full` debouncer + the
/// receiver channel. Drop the `RagWatcher` to stop watching — both
/// the debouncer and its background thread shut down together.
///
/// Held by the server alongside the index slot; the server's
/// background tokio task pulls events from `recv` and dispatches them
/// to the incremental-update path.
pub struct RagWatcher {
    _debouncer: Debouncer<RecommendedWatcher, FileIdMap>,
    recv: mpsc::Receiver<HashSet<PathBuf>>,
    root: PathBuf,
}

impl RagWatcher {
    /// Build a watcher rooted at `root`. Uses the default debounce
    /// window; for custom values use [`Self::with_debounce`].
    pub fn new(root: impl AsRef<Path>) -> Result<Self, WatcherError> {
        Self::with_debounce(root, Duration::from_millis(DEFAULT_DEBOUNCE_MS))
    }

    /// Build a watcher with a custom debounce window. Very short
    /// windows (<100ms) defeat the purpose; very long windows (>5s)
    /// make the watcher feel unresponsive. The default 1s is right
    /// for editor save cycles.
    pub fn with_debounce(
        root: impl AsRef<Path>,
        debounce: Duration,
    ) -> Result<Self, WatcherError> {
        let root_path = root.as_ref().to_path_buf();
        if !root_path.exists() {
            return Err(WatcherError::RootMissing(root_path));
        }
        if !root_path.is_dir() {
            return Err(WatcherError::RootNotADirectory(root_path));
        }

        let (tx, rx) = mpsc::channel::<HashSet<PathBuf>>();

        // The debouncer hands us a `DebounceEventResult` per coalesced
        // batch. Project each batch down to "the set of file paths
        // touched" — the caller doesn't care about the kind of change
        // (write vs rename vs remove), just that the file needs re-
        // indexing. Removed files arrive here too; the caller should
        // probe the path's existence after dequeue.
        let mut debouncer = new_debouncer(
            debounce,
            None,
            move |result: DebounceEventResult| {
                if let Ok(events) = result {
                    let mut paths: HashSet<PathBuf> = HashSet::new();
                    for ev in events {
                        for p in &ev.event.paths {
                            paths.insert(p.clone());
                        }
                    }
                    if !paths.is_empty() {
                        // Sender dropped means the consumer side exited;
                        // ignore the send error and let the debouncer
                        // continue running until its `Drop` cleans up.
                        let _ = tx.send(paths);
                    }
                }
                // notify errors get dropped on the floor here — the
                // caller can re-create the watcher if it sees no events
                // arriving. Adding a sidechannel for errors costs more
                // surface area than it gains for v1.
            },
        )
        .map_err(|e| WatcherError::Notify(e.to_string()))?;

        debouncer
            .watcher()
            .watch(&root_path, RecursiveMode::Recursive)
            .map_err(|e| WatcherError::Notify(e.to_string()))?;

        Ok(Self {
            _debouncer: debouncer,
            recv: rx,
            root: root_path,
        })
    }

    /// The workspace root this watcher was built against. Useful so
    /// downstream consumers can compute `path.strip_prefix(root)` when
    /// echoing into a workspace-relative form (matches the walker's
    /// output convention).
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Receiver for coalesced "these paths changed" events. Each
    /// `HashSet<PathBuf>` represents one debounce window's worth of
    /// file changes — multiple writes within the window collapse to
    /// one set with the affected paths inside.
    pub fn receiver(&self) -> &mpsc::Receiver<HashSet<PathBuf>> {
        &self.recv
    }

    /// Convenience: blocking next-event read with a timeout. Useful
    /// for a tokio background task that polls — set a short timeout
    /// so the task can also check for shutdown signals in between.
    pub fn recv_timeout(
        &self,
        timeout: Duration,
    ) -> Result<HashSet<PathBuf>, mpsc::RecvTimeoutError> {
        self.recv.recv_timeout(timeout)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum WatcherError {
    #[error("watch root does not exist: {0}")]
    RootMissing(PathBuf),
    #[error("watch root is not a directory: {0}")]
    RootNotADirectory(PathBuf),
    #[error("notify error: {0}")]
    Notify(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn new_rejects_missing_root() {
        let path = std::env::temp_dir().join("rustllama-watcher-does-not-exist-xyz");
        // Ensure no leftover from a prior test.
        let _ = std::fs::remove_dir_all(&path);
        // `RagWatcher` deliberately doesn't derive Debug (Debouncer
        // doesn't), so unwrap the variant by hand.
        match RagWatcher::new(&path) {
            Ok(_) => panic!("expected RootMissing"),
            Err(WatcherError::RootMissing(_)) => {}
            Err(other) => panic!("expected RootMissing, got {other:?}"),
        }
    }

    #[test]
    fn new_rejects_file_root() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("not-a-dir.txt");
        std::fs::write(&file_path, b"x").unwrap();
        match RagWatcher::new(&file_path) {
            Ok(_) => panic!("expected RootNotADirectory"),
            Err(WatcherError::RootNotADirectory(_)) => {}
            Err(other) => panic!("expected RootNotADirectory, got {other:?}"),
        }
    }

    #[test]
    fn watches_changes_under_root() {
        // End-to-end: spin up a watcher, write a file, see the path
        // come back through the receiver within the debounce window.
        // Uses a very short debounce so the test runs quickly.
        let dir = tempfile::tempdir().unwrap();
        let watcher = RagWatcher::with_debounce(
            dir.path(),
            Duration::from_millis(100),
        )
        .unwrap();

        let target = dir.path().join("hello.rs");
        std::fs::write(&target, b"fn main() {}").unwrap();

        // Allow up to 2 seconds for the event to round-trip — on a
        // slow CI runner with a cold notify backend the first event
        // sometimes takes a beat to show up.
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut saw_target = false;
        while Instant::now() < deadline {
            match watcher.recv_timeout(Duration::from_millis(200)) {
                Ok(paths) => {
                    if paths.iter().any(|p| {
                        p.file_name().and_then(|s| s.to_str()) == Some("hello.rs")
                    }) {
                        saw_target = true;
                        break;
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        assert!(saw_target, "expected to see hello.rs in a watcher event");
    }
}
