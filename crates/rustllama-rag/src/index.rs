//! In-memory vector store with cosine-similarity search.
//!
//! Stores `(embedding, ChunkSpec, metadata)` triples and supports
//! top-K nearest-neighbor lookup against a query embedding. v1
//! uses a linear scan — works well up to ~100K chunks (a typical
//! coding workspace). For larger corpora swap in HNSW or similar
//! later without changing the public API.
//!
//! The vectors are stored normalized (`||v||₂ = 1`) so cosine
//! similarity reduces to a dot product. Normalization happens on
//! insert; queries are normalized at search time. Re-normalizing
//! the queries inside `search` rather than asking the caller to
//! pre-normalize means a misuse can't produce wildly-wrong rankings.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::chunker::ChunkSpec;

/// File format magic. Eight bytes so the wire shape is recognizable
/// at a glance in a hex dump. Distinct from the engine's own GGUF
/// + safetensors magics so a stray `file` call can't confuse the two.
const RAG_INDEX_MAGIC: &[u8; 8] = b"RLLMRAG\x00";

/// On-disk format version. Bump when the field layout changes — load
/// rejects unknown versions rather than corrupting an in-memory index.
const RAG_INDEX_VERSION: u32 = 1;

/// One indexed chunk: the source spec + its normalized embedding.
/// Returned by [`RagIndex::search`] alongside its similarity score.
#[derive(Debug, Clone)]
pub struct IndexedChunk {
    pub source_path: PathBuf,
    pub line_start: usize,
    pub line_end: usize,
    pub text: String,
    /// L2-normalized embedding vector.
    pub embedding: Vec<f32>,
}

/// One search hit: an indexed chunk + its cosine similarity to the
/// query. `score` is in `[-1.0, 1.0]` (cosine); higher = more similar.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: IndexedChunk,
    pub score: f32,
}

/// In-memory vector store. Build via [`RagIndex::new`], insert via
/// [`RagIndex::add`] / [`RagIndex::add_chunk`], query via
/// [`RagIndex::search`].
///
/// Not thread-safe — wrap in `RwLock` or use a sharded design for
/// concurrent access. Single-writer single-reader is the typical
/// pattern (indexer builds, server queries).
#[derive(Debug, Default)]
pub struct RagIndex {
    chunks: Vec<IndexedChunk>,
    embedding_dim: usize,
}

impl RagIndex {
    /// Build an empty index. `embedding_dim` is the expected
    /// dimension of every embedding inserted — mismatches surface
    /// at `add`-time so a bad embedding doesn't corrupt the store.
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            chunks: Vec::new(),
            embedding_dim,
        }
    }

    /// Number of indexed chunks.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Embedding dimension this index expects on inserts and queries.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Insert a chunk spec + its (un-normalized) embedding.
    /// Normalizes the embedding in place to L2 unit length so
    /// search reduces to a dot product. Returns
    /// `Err(WrongDimension)` if `embedding.len() != embedding_dim`.
    pub fn add(
        &mut self,
        chunk: ChunkSpec,
        embedding: Vec<f32>,
    ) -> Result<(), IndexError> {
        if embedding.len() != self.embedding_dim {
            return Err(IndexError::WrongDimension {
                expected: self.embedding_dim,
                got: embedding.len(),
            });
        }
        let normalized = l2_normalize(embedding);
        self.chunks.push(IndexedChunk {
            source_path: chunk.source_path,
            line_start: chunk.line_start,
            line_end: chunk.line_end,
            text: chunk.text,
            embedding: normalized,
        });
        Ok(())
    }

    /// Drop every chunk whose `source_path` equals `path`. Returns the
    /// count removed. O(n) over current chunks (linear scan); fine for
    /// the in-memory store's target size. Pair with [`Self::add`] to
    /// implement incremental "file changed" updates.
    pub fn remove_by_path(&mut self, path: &Path) -> usize {
        let before = self.chunks.len();
        self.chunks.retain(|c| c.source_path != path);
        before - self.chunks.len()
    }

    /// True if any chunk in the index was sourced from `path`.
    pub fn contains_path(&self, path: &Path) -> bool {
        self.chunks.iter().any(|c| c.source_path == path)
    }

    /// Distinct source paths currently in the index, in insertion order
    /// (first occurrence wins). Used by the server to surface a
    /// "what's indexed" view in the GUI Status page.
    pub fn source_paths(&self) -> Vec<PathBuf> {
        let mut seen: std::collections::HashSet<&Path> = std::collections::HashSet::new();
        let mut out: Vec<PathBuf> = Vec::new();
        for c in &self.chunks {
            if seen.insert(&c.source_path) {
                out.push(c.source_path.clone());
            }
        }
        out
    }

    /// Convenience: insert when the chunk + embedding are already
    /// packaged separately. Same semantics as [`Self::add`].
    pub fn add_chunk(
        &mut self,
        source_path: PathBuf,
        line_start: usize,
        line_end: usize,
        text: String,
        embedding: Vec<f32>,
    ) -> Result<(), IndexError> {
        self.add(
            ChunkSpec {
                source_path,
                line_start,
                line_end,
                text,
            },
            embedding,
        )
    }

    /// Serialize the index to `path`. Format:
    ///
    /// ```text
    /// magic        : 8 bytes  "RLLMRAG\0"
    /// version      : u32 LE   (1)
    /// embedding_dim: u32 LE
    /// chunk_count  : u64 LE
    /// per chunk:
    ///   path_len : u32 LE  (UTF-8 byte length)
    ///   path     : [u8; path_len]
    ///   line_start: u32 LE
    ///   line_end  : u32 LE
    ///   text_len  : u32 LE
    ///   text      : [u8; text_len]
    ///   embedding : [f32 LE; embedding_dim]
    /// ```
    ///
    /// Embeddings are written L2-normalized (the in-memory form). On
    /// load the rest of the engine treats them as already-normalized,
    /// matching `add`'s semantics.
    pub fn save(&self, path: &Path) -> Result<(), IndexError> {
        let f = std::fs::File::create(path).map_err(IndexError::Io)?;
        let mut w = std::io::BufWriter::new(f);
        w.write_all(RAG_INDEX_MAGIC).map_err(IndexError::Io)?;
        w.write_all(&RAG_INDEX_VERSION.to_le_bytes()).map_err(IndexError::Io)?;
        w.write_all(&(self.embedding_dim as u32).to_le_bytes()).map_err(IndexError::Io)?;
        w.write_all(&(self.chunks.len() as u64).to_le_bytes()).map_err(IndexError::Io)?;
        for c in &self.chunks {
            let path_str = c.source_path.to_string_lossy();
            let path_bytes = path_str.as_bytes();
            w.write_all(&(path_bytes.len() as u32).to_le_bytes()).map_err(IndexError::Io)?;
            w.write_all(path_bytes).map_err(IndexError::Io)?;
            w.write_all(&(c.line_start as u32).to_le_bytes()).map_err(IndexError::Io)?;
            w.write_all(&(c.line_end as u32).to_le_bytes()).map_err(IndexError::Io)?;
            let text_bytes = c.text.as_bytes();
            w.write_all(&(text_bytes.len() as u32).to_le_bytes()).map_err(IndexError::Io)?;
            w.write_all(text_bytes).map_err(IndexError::Io)?;
            // f32 LE — bytemuck would do the cast for free but a small
            // loop avoids the dep here. Embedding vector length is
            // already validated to equal embedding_dim at insert time.
            for &v in &c.embedding {
                w.write_all(&v.to_le_bytes()).map_err(IndexError::Io)?;
            }
        }
        w.flush().map_err(IndexError::Io)?;
        Ok(())
    }

    /// Read an index file previously produced by [`Self::save`].
    /// Rejects on magic / version / corruption with a typed error so
    /// callers can surface a sensible HTTP status. Embedding vectors
    /// are taken as-is (the saved form is already normalized).
    pub fn load(path: &Path) -> Result<Self, IndexError> {
        let f = std::fs::File::open(path).map_err(IndexError::Io)?;
        let mut r = std::io::BufReader::new(f);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic).map_err(IndexError::Io)?;
        if &magic != RAG_INDEX_MAGIC {
            return Err(IndexError::BadMagic);
        }
        let version = read_u32_le(&mut r)?;
        if version != RAG_INDEX_VERSION {
            return Err(IndexError::UnsupportedVersion {
                got: version,
                supported: RAG_INDEX_VERSION,
            });
        }
        let embedding_dim = read_u32_le(&mut r)? as usize;
        if embedding_dim == 0 {
            return Err(IndexError::CorruptHeader("embedding_dim == 0"));
        }
        let chunk_count = read_u64_le(&mut r)? as usize;
        let mut chunks = Vec::with_capacity(chunk_count);
        for _ in 0..chunk_count {
            let path_len = read_u32_le(&mut r)? as usize;
            let mut path_buf = vec![0u8; path_len];
            r.read_exact(&mut path_buf).map_err(IndexError::Io)?;
            let path_str = String::from_utf8(path_buf).map_err(|_| {
                IndexError::CorruptHeader("chunk source_path is not valid UTF-8")
            })?;
            let line_start = read_u32_le(&mut r)? as usize;
            let line_end = read_u32_le(&mut r)? as usize;
            let text_len = read_u32_le(&mut r)? as usize;
            let mut text_buf = vec![0u8; text_len];
            r.read_exact(&mut text_buf).map_err(IndexError::Io)?;
            let text = String::from_utf8(text_buf).map_err(|_| {
                IndexError::CorruptHeader("chunk text is not valid UTF-8")
            })?;
            let mut embedding = vec![0f32; embedding_dim];
            let mut emb_bytes = vec![0u8; embedding_dim * 4];
            r.read_exact(&mut emb_bytes).map_err(IndexError::Io)?;
            for (i, dst) in embedding.iter_mut().enumerate() {
                let b = &emb_bytes[i * 4..i * 4 + 4];
                *dst = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
            }
            chunks.push(IndexedChunk {
                source_path: PathBuf::from(path_str),
                line_start,
                line_end,
                text,
                embedding,
            });
        }
        Ok(Self {
            chunks,
            embedding_dim,
        })
    }

    /// Top-`k` nearest-neighbor search. Returns at most `k` hits
    /// ordered by descending cosine similarity. Empty index returns
    /// an empty Vec.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
    ) -> Result<Vec<SearchResult>, IndexError> {
        if query.len() != self.embedding_dim {
            return Err(IndexError::WrongDimension {
                expected: self.embedding_dim,
                got: query.len(),
            });
        }
        if self.chunks.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let q_norm = l2_normalize(query.to_vec());
        // Linear scan + partial sort. For top-k against N chunks
        // with `k << N`, a binary heap is faster than a full sort,
        // but for the v1 corpus sizes we're targeting (<100K) the
        // wall-clock difference is sub-millisecond.
        let mut scored: Vec<(f32, usize)> = self
            .chunks
            .iter()
            .enumerate()
            .map(|(i, c)| (dot(&q_norm, &c.embedding), i))
            .collect();
        // Sort descending by score.
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(scored
            .into_iter()
            .take(k)
            .map(|(score, idx)| SearchResult {
                chunk: self.chunks[idx].clone(),
                score,
            })
            .collect())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error(
        "embedding dimension mismatch: index expects {expected} but got {got}"
    )]
    WrongDimension { expected: usize, got: usize },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("not a rustllama RAG index file (bad magic)")]
    BadMagic,
    #[error("unsupported index file version {got}; this build supports {supported}")]
    UnsupportedVersion { got: u32, supported: u32 },
    #[error("corrupt index header: {0}")]
    CorruptHeader(&'static str),
}

fn read_u32_le(r: &mut impl std::io::Read) -> Result<u32, IndexError> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(IndexError::Io)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64_le(r: &mut impl std::io::Read) -> Result<u64, IndexError> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(IndexError::Io)?;
    Ok(u64::from_le_bytes(buf))
}

/// L2-normalize a vector to unit length. Zero vectors stay
/// zero (avoids NaN); the cosine similarity for a zero vector is
/// always 0, which is the correct "no signal" answer.
fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
    v
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(path: &str, text: &str) -> ChunkSpec {
        ChunkSpec {
            source_path: path.into(),
            line_start: 1,
            line_end: 1,
            text: text.into(),
        }
    }

    #[test]
    fn empty_index_returns_no_results() {
        let idx = RagIndex::new(4);
        assert_eq!(idx.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap().len(), 0);
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn add_rejects_wrong_dimension() {
        let mut idx = RagIndex::new(4);
        let err = idx
            .add(chunk("a.rs", "hi"), vec![1.0, 0.0])
            .expect_err("should reject");
        match err {
            IndexError::WrongDimension { expected: 4, got: 2 } => {}
            other => panic!("expected WrongDimension, got {other:?}"),
        }
    }

    #[test]
    fn search_rejects_wrong_dimension() {
        let idx = RagIndex::new(4);
        let err = idx.search(&[1.0, 0.0], 1).expect_err("should reject");
        match err {
            IndexError::WrongDimension { expected: 4, got: 2 } => {}
            other => panic!("expected WrongDimension, got {other:?}"),
        }
    }

    #[test]
    fn search_orders_by_cosine_descending() {
        let mut idx = RagIndex::new(3);
        // Three chunks with very different embeddings.
        idx.add(chunk("a.rs", "axis-x"), vec![1.0, 0.0, 0.0]).unwrap();
        idx.add(chunk("b.rs", "axis-y"), vec![0.0, 1.0, 0.0]).unwrap();
        idx.add(chunk("c.rs", "axis-z"), vec![0.0, 0.0, 1.0]).unwrap();
        // Query points toward x — closest is a.rs.
        let hits = idx.search(&[1.0, 0.1, 0.0], 3).unwrap();
        assert_eq!(hits.len(), 3);
        assert_eq!(
            hits[0].chunk.source_path.file_name().unwrap(),
            std::ffi::OsStr::new("a.rs")
        );
        // Scores must be descending.
        assert!(hits[0].score >= hits[1].score);
        assert!(hits[1].score >= hits[2].score);
    }

    #[test]
    fn search_truncates_to_k() {
        let mut idx = RagIndex::new(2);
        for i in 0..10 {
            let v = vec![i as f32, 1.0];
            idx.add(chunk(&format!("f{i}.rs"), "x"), v).unwrap();
        }
        let hits = idx.search(&[5.0, 1.0], 3).unwrap();
        assert_eq!(hits.len(), 3);
    }

    #[test]
    fn search_k_zero_returns_empty() {
        let mut idx = RagIndex::new(2);
        idx.add(chunk("a.rs", "x"), vec![1.0, 0.0]).unwrap();
        assert_eq!(idx.search(&[1.0, 0.0], 0).unwrap().len(), 0);
    }

    #[test]
    fn identical_vectors_score_1_0() {
        let mut idx = RagIndex::new(3);
        idx.add(chunk("a.rs", "x"), vec![3.0, 4.0, 0.0]).unwrap();
        let hits = idx.search(&[6.0, 8.0, 0.0], 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(
            (hits[0].score - 1.0).abs() < 1e-5,
            "parallel vectors should give cosine 1.0; got {}",
            hits[0].score
        );
    }

    #[test]
    fn orthogonal_vectors_score_0() {
        let mut idx = RagIndex::new(3);
        idx.add(chunk("a.rs", "x"), vec![1.0, 0.0, 0.0]).unwrap();
        let hits = idx.search(&[0.0, 1.0, 0.0], 1).unwrap();
        assert!(hits[0].score.abs() < 1e-5, "orthogonal cosine should be 0");
    }

    #[test]
    fn antiparallel_vectors_score_negative_1_0() {
        let mut idx = RagIndex::new(2);
        idx.add(chunk("a.rs", "x"), vec![1.0, 0.0]).unwrap();
        let hits = idx.search(&[-1.0, 0.0], 1).unwrap();
        assert!(
            (hits[0].score + 1.0).abs() < 1e-5,
            "anti-parallel cosine should be -1.0; got {}",
            hits[0].score
        );
    }

    #[test]
    fn zero_query_returns_zero_scores_without_nan() {
        let mut idx = RagIndex::new(3);
        idx.add(chunk("a.rs", "x"), vec![1.0, 2.0, 3.0]).unwrap();
        let hits = idx.search(&[0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].score, 0.0, "zero query should not produce NaN");
    }

    #[test]
    fn chunk_metadata_survives_round_trip() {
        let mut idx = RagIndex::new(2);
        let c = ChunkSpec {
            source_path: "src/foo.rs".into(),
            line_start: 42,
            line_end: 71,
            text: "fn foo() { /* body */ }".into(),
        };
        idx.add(c, vec![0.6, 0.8]).unwrap();
        let hits = idx.search(&[0.6, 0.8], 1).unwrap();
        assert_eq!(hits[0].chunk.source_path, PathBuf::from("src/foo.rs"));
        assert_eq!(hits[0].chunk.line_start, 42);
        assert_eq!(hits[0].chunk.line_end, 71);
        assert!(hits[0].chunk.text.contains("fn foo"));
    }

    #[test]
    fn save_load_round_trip_preserves_index() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rag.idx");

        let mut idx = RagIndex::new(3);
        idx.add(chunk("src/a.rs", "alpha"), vec![1.0, 0.0, 0.0]).unwrap();
        idx.add(chunk("src/b.rs", "beta"), vec![0.0, 1.0, 0.0]).unwrap();
        idx.add(chunk("src/c.rs", "gamma"), vec![0.0, 0.0, 1.0]).unwrap();
        idx.save(&path).expect("save should succeed");

        let loaded = RagIndex::load(&path).expect("load should succeed");
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.embedding_dim(), 3);

        // Query semantics must be identical — same input vector returns
        // the same ordering and the same scores within fp32 tolerance.
        let hits_orig = idx.search(&[1.0, 0.0, 0.0], 3).unwrap();
        let hits_load = loaded.search(&[1.0, 0.0, 0.0], 3).unwrap();
        assert_eq!(hits_orig.len(), hits_load.len());
        for (a, b) in hits_orig.iter().zip(hits_load.iter()) {
            assert_eq!(a.chunk.source_path, b.chunk.source_path);
            assert_eq!(a.chunk.text, b.chunk.text);
            assert!((a.score - b.score).abs() < 1e-6);
        }
    }

    #[test]
    fn load_rejects_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bogus.bin");
        std::fs::write(&path, b"not-a-rag-index-file").unwrap();
        match RagIndex::load(&path).expect_err("should reject") {
            IndexError::BadMagic => {}
            other => panic!("expected BadMagic, got {other:?}"),
        }
    }

    #[test]
    fn load_rejects_unsupported_version() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v999.idx");
        // Hand-craft a file with the correct magic but a future version.
        let mut bytes: Vec<u8> = Vec::new();
        bytes.extend_from_slice(b"RLLMRAG\x00");
        bytes.extend_from_slice(&999u32.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes()); // embedding_dim
        bytes.extend_from_slice(&0u64.to_le_bytes()); // chunk_count
        std::fs::write(&path, &bytes).unwrap();
        match RagIndex::load(&path).expect_err("should reject") {
            IndexError::UnsupportedVersion { got: 999, supported: 1 } => {}
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[test]
    fn save_load_round_trip_preserves_empty_index() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.idx");
        let idx = RagIndex::new(7);
        idx.save(&path).unwrap();
        let loaded = RagIndex::load(&path).unwrap();
        assert!(loaded.is_empty());
        assert_eq!(loaded.embedding_dim(), 7);
    }

    #[test]
    fn l2_normalize_preserves_zero_vector() {
        let v = l2_normalize(vec![0.0; 5]);
        for x in &v {
            assert_eq!(*x, 0.0);
        }
    }
}
