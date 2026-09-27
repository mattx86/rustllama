//! Workspace-aware RAG: `POST /v1/rag/index` + `POST /v1/rag/query`.
//!
//! v1 keeps the index entirely in process memory — a single
//! `Arc<RwLock<Option<RagIndex>>>` slot on `AppState`. The first
//! `/v1/rag/index` call walks the workspace root, chunks the files,
//! embeds every chunk through the same BERT bundle that backs
//! `/v1/embeddings`, and replaces (or initializes) the slot. The
//! `/v1/rag/query` handler then embeds the query and runs a
//! cosine-similarity top-k search over the same store.
//!
//! Persistent storage (sqlite) and incremental updates / file
//! watching are deliberately deferred — the in-memory store costs
//! ~1 KB per chunk so a 100K-chunk workspace fits in ~100 MB, which
//! is the typical coding-LLM use case. Anything larger calls the
//! follow-up sqlite path that lands in `rustllama-rag` itself.
//!
//! Both handlers require `[embeddings]` to be configured — without
//! an embedding model there's nothing to vectorize against. Missing
//! config returns 501 with the same diagnostic that
//! `/v1/embeddings` emits, so the failure mode is consistent.

use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use rustllama_rag::{
    chunk_with_mode, ChunkSpec, ChunkingMode, RagIndex, SearchResult, WorkspaceWalker,
};
use serde::{Deserialize, Serialize};

/// Parse a `mode` field from the request into a [`ChunkingMode`].
/// Unknown strings fall back to `Auto` with a warning rather than a
/// 400 — chunking mode is a hint, not a hard requirement.
fn parse_chunking_mode(s: Option<&str>) -> ChunkingMode {
    match s {
        None | Some("auto") => ChunkingMode::Auto,
        Some("lines") => ChunkingMode::Lines,
        Some("tree_sitter") | Some("tree-sitter") | Some("ts") => ChunkingMode::TreeSitter,
        Some(other) => {
            tracing::warn!(mode = %other, "unknown rag chunking mode, falling back to auto");
            ChunkingMode::Auto
        }
    }
}

use crate::embeddings::{embed_batches, get_embedding_bundle, tokenize_one};
use crate::AppState;

/// `POST /v1/rag/index` body.
#[derive(Deserialize)]
pub struct RagIndexRequest {
    /// Absolute or relative path to the workspace root to index.
    /// Resolved against the server's CWD when relative.
    pub root: PathBuf,
    /// Per-chunk size in lines. Defaults to
    /// `rustllama_rag::chunker::DEFAULT_CHUNK_LINES` (30).
    #[serde(default)]
    pub chunk_lines: Option<usize>,
    /// Overlap between consecutive chunks (lines). Defaults to
    /// `DEFAULT_OVERLAP_LINES` (5). Bad values fall back inside
    /// the chunker — see [`rustllama_rag::chunk_text`].
    #[serde(default)]
    pub overlap_lines: Option<usize>,
    /// Max bytes per file. Defaults to the walker's 1 MB cap. Files
    /// above this are skipped (a 50 MB CSV produces no useful chunks
    /// for a coding LLM, and embedding it is wasted compute).
    #[serde(default)]
    pub max_file_bytes: Option<u64>,
    /// Chunking strategy. `"auto"` (default) picks tree-sitter when a
    /// grammar is available for the file's extension and falls back
    /// to line-based otherwise. `"tree_sitter"` forces AST-aware
    /// chunking; `"lines"` forces the sliding-window form. Unknown
    /// values silently degrade to auto.
    #[serde(default)]
    pub mode: Option<String>,
}

#[derive(Serialize)]
pub struct RagIndexResponse {
    pub object: &'static str,
    pub files_indexed: u32,
    pub chunks_indexed: u32,
    pub embedding_dim: u32,
    /// Chunks that were skipped because the embedding forward failed
    /// (e.g. a single chunk too long for the embedder's context).
    /// Indexing continues across skips — this is a count, not a fatal
    /// error.
    pub chunks_skipped: u32,
}

/// `POST /v1/rag/query` body.
#[derive(Deserialize)]
pub struct RagQueryRequest {
    pub query: String,
    /// Top-K results to return. Defaults to 5; capped at 50 to keep
    /// the wire response from getting silly. 0 is rejected with 400.
    #[serde(default)]
    pub k: Option<u32>,
}

#[derive(Serialize)]
pub struct RagQueryResponse {
    pub object: &'static str,
    pub hits: Vec<RagHit>,
}

#[derive(Serialize)]
pub struct RagHit {
    /// Workspace-relative path of the chunk's source file.
    pub source_path: String,
    pub line_start: u32,
    pub line_end: u32,
    pub score: f32,
    /// The chunk's text content. Clients quote this directly back to
    /// the model as RAG context.
    pub text: String,
}

const DEFAULT_K: u32 = 5;
const MAX_K: u32 = 50;

pub async fn index(
    State(state): State<AppState>,
    Json(req): Json<RagIndexRequest>,
) -> Response {
    // Embedding model must be configured; the index has no semantics
    // without one.
    let bundle = match get_embedding_bundle(&state) {
        Ok(b) => b,
        Err(resp) => return resp,
    };
    let embedding_dim = bundle.model.cfg.d_model;

    // Walk + chunk on a blocking worker — filesystem IO + UTF-8
    // decode shouldn't block the tokio executor.
    let mode = parse_chunking_mode(req.mode.as_deref());
    let chunks_result = {
        let root = req.root.clone();
        let max_file_bytes = req.max_file_bytes;
        let chunk_lines = req.chunk_lines.unwrap_or(0);
        let overlap_lines = req.overlap_lines.unwrap_or(5);
        tokio::task::spawn_blocking(move || -> Result<(u32, Vec<ChunkSpec>), String> {
            let walker = WorkspaceWalker::new(&root).map_err(|e| e.to_string())?;
            let walker = match max_file_bytes {
                Some(n) => walker.with_max_file_bytes(n),
                None => walker,
            };
            let mut files = 0u32;
            let mut out: Vec<ChunkSpec> = Vec::new();
            for file in walker.files() {
                files += 1;
                let specs = chunk_with_mode(
                    file.path.clone(),
                    &file.contents,
                    chunk_lines,
                    overlap_lines,
                    mode,
                );
                out.extend(specs);
            }
            Ok((files, out))
        })
        .await
    };
    let (files_indexed, chunks) = match chunks_result {
        Ok(Ok(v)) => v,
        Ok(Err(msg)) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "could not walk workspace root",
                    "detail": msg,
                })),
            )
                .into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "rag walker worker panicked",
                    "detail": e.to_string(),
                })),
            )
                .into_response();
        }
    };
    if chunks.is_empty() {
        // Initialize an empty index so subsequent /query returns
        // empty hits rather than a 409 "no index built".
        let mut slot = state.rag_index.write().await;
        *slot = Some(Arc::new(tokio::sync::RwLock::new(RagIndex::new(embedding_dim))));
        return Json(RagIndexResponse {
            object: "rag.index",
            files_indexed,
            chunks_indexed: 0,
            embedding_dim: embedding_dim as u32,
            chunks_skipped: 0,
        })
        .into_response();
    }

    // Tokenize every chunk on the blocking worker that runs the
    // embed forward pass. The tokenizer is the same WordPiece
    // bundle the embeddings handler uses.
    let token_batches: Vec<Vec<i32>> = {
        let mut out: Vec<Vec<i32>> = Vec::with_capacity(chunks.len());
        for (i, c) in chunks.iter().enumerate() {
            match tokenize_one(&bundle.tokenizer, &c.text) {
                Ok(ids) => out.push(ids),
                Err(_resp) => {
                    // Tokenize failures are extremely rare on valid
                    // UTF-8 source — log and skip; the chunk simply
                    // won't make it into the index.
                    tracing::warn!(chunk_index = i, "rag index: tokenize failed");
                    out.push(Vec::new());
                }
            }
        }
        out
    };

    // Skip empty token batches so the embedder doesn't choke on a
    // zero-length input. Track the kept-vs-skipped split.
    let mut kept_indices: Vec<usize> = Vec::with_capacity(chunks.len());
    let mut batches_to_embed: Vec<Vec<i32>> = Vec::with_capacity(chunks.len());
    for (i, b) in token_batches.into_iter().enumerate() {
        if !b.is_empty() {
            kept_indices.push(i);
            batches_to_embed.push(b);
        }
    }
    let mut chunks_skipped = (chunks.len() - kept_indices.len()) as u32;

    let vectors = match embed_batches(bundle.model.clone(), batches_to_embed).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    // Build a fresh index (full reindex semantics in v1 — incremental
    // updates are a follow-up). Replace the slot atomically.
    let mut new_index = RagIndex::new(embedding_dim);
    for (vec, chunk_idx) in vectors.into_iter().zip(kept_indices.iter().copied()) {
        let c = &chunks[chunk_idx];
        if let Err(e) = new_index.add(c.clone(), vec) {
            // Dimension mismatch shouldn't happen because both halves
            // come from the same model; log loudly and skip if it does.
            tracing::error!(error = %e, "rag index: rejected chunk on add");
            chunks_skipped += 1;
        }
    }
    let chunks_indexed = new_index.len() as u32;
    {
        let mut slot = state.rag_index.write().await;
        *slot = Some(Arc::new(tokio::sync::RwLock::new(new_index)));
    }

    Json(RagIndexResponse {
        object: "rag.index",
        files_indexed,
        chunks_indexed,
        embedding_dim: embedding_dim as u32,
        chunks_skipped,
    })
    .into_response()
}

pub async fn query(
    State(state): State<AppState>,
    Json(req): Json<RagQueryRequest>,
) -> Response {
    let k = req.k.unwrap_or(DEFAULT_K);
    if k == 0 {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "k must be at least 1",
            })),
        )
            .into_response();
    }
    let k = k.min(MAX_K) as usize;

    if req.query.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "query must be a non-empty string",
            })),
        )
            .into_response();
    }

    // Need a built index; otherwise 409 with the hint that /index
    // must be called first. Distinguishes "no embedding model" (501)
    // from "no index yet" (409).
    let index_handle = {
        let slot = state.rag_index.read().await;
        match slot.as_ref() {
            Some(h) => h.clone(),
            None => {
                return (
                    StatusCode::CONFLICT,
                    Json(serde_json::json!({
                        "error": "no RAG index built",
                        "detail": "POST /v1/rag/index { \"root\": \"...\" } first to populate the in-memory index.",
                    })),
                )
                    .into_response();
            }
        }
    };

    let bundle = match get_embedding_bundle(&state) {
        Ok(b) => b,
        Err(resp) => return resp,
    };

    // Tokenize + embed the query through the same path as /index.
    let query_tokens = match tokenize_one(&bundle.tokenizer, &req.query) {
        Ok(t) => t,
        Err(resp) => return resp,
    };
    let mut vectors = match embed_batches(bundle.model.clone(), vec![query_tokens]).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let q_vec = vectors.pop().unwrap_or_default();

    // Search; map IndexedChunks back to the wire shape.
    let hits = {
        let idx = index_handle.read().await;
        match idx.search(&q_vec, k) {
            Ok(h) => h,
            Err(e) => {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": "rag search failed",
                        "detail": e.to_string(),
                    })),
                )
                    .into_response();
            }
        }
    };

    let out: Vec<RagHit> = hits.into_iter().map(to_wire_hit).collect();
    Json(RagQueryResponse {
        object: "rag.query",
        hits: out,
    })
    .into_response()
}

pub async fn update(
    State(state): State<AppState>,
    Json(req): Json<RagUpdateRequest>,
) -> Response {
    if req.paths.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "paths must be a non-empty list",
            })),
        )
            .into_response();
    }

    // Need both an embedding bundle and a live index.
    let bundle = match get_embedding_bundle(&state) {
        Ok(b) => b,
        Err(resp) => return resp,
    };
    let index_handle = {
        let slot = state.rag_index.read().await;
        match slot.as_ref() {
            Some(h) => h.clone(),
            None => {
                return (
                    StatusCode::CONFLICT,
                    Json(serde_json::json!({
                        "error": "no RAG index built",
                        "detail": "POST /v1/rag/index { \"root\": \"...\" } first; /v1/rag/update is for incremental refresh of an existing index.",
                    })),
                )
                    .into_response();
            }
        }
    };

    // Phase 1: classify each path. Surviving files get walked + chunked
    // on the blocking worker; missing files just get removed.
    let root = req.root.clone();
    let chunk_lines = req.chunk_lines.unwrap_or(0);
    let overlap_lines = req.overlap_lines.unwrap_or(5);
    let max_file_bytes = req.max_file_bytes.unwrap_or(1_048_576);
    let req_paths = req.paths.clone();
    let mode = parse_chunking_mode(req.mode.as_deref());

    let walk_result = tokio::task::spawn_blocking(
        move || -> (Vec<PathBuf>, Vec<(PathBuf, Vec<rustllama_rag::ChunkSpec>)>) {
            let mut removed_only: Vec<PathBuf> = Vec::new();
            let mut reindexed: Vec<(PathBuf, Vec<rustllama_rag::ChunkSpec>)> = Vec::new();
            for p in &req_paths {
                // Resolve absolute path on disk; chunk paths land
                // workspace-relative to match the initial /index call.
                let abs = if p.is_absolute() {
                    p.clone()
                } else {
                    root.join(p)
                };
                let rel = abs
                    .strip_prefix(&root)
                    .map(|r| r.to_path_buf())
                    .unwrap_or_else(|_| p.clone());

                if !abs.exists() || !abs.is_file() {
                    removed_only.push(rel);
                    continue;
                }
                let meta = match std::fs::metadata(&abs) {
                    Ok(m) => m,
                    Err(_) => {
                        removed_only.push(rel);
                        continue;
                    }
                };
                if meta.len() > max_file_bytes {
                    removed_only.push(rel);
                    continue;
                }
                let contents = match std::fs::read_to_string(&abs) {
                    Ok(s) => s,
                    Err(_) => {
                        removed_only.push(rel);
                        continue;
                    }
                };
                let specs = rustllama_rag::chunk_with_mode(
                    rel.clone(),
                    &contents,
                    chunk_lines,
                    overlap_lines,
                    mode,
                );
                reindexed.push((rel, specs));
            }
            (removed_only, reindexed)
        },
    )
    .await;

    let (removed_only, reindexed) = match walk_result {
        Ok(v) => v,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "rag update worker panicked",
                    "detail": e.to_string(),
                })),
            )
                .into_response();
        }
    };

    // Phase 2: tokenize + embed all surviving chunks. Track which
    // (file, chunk-position) each vector belongs to so the final
    // batched embed call can be undone if a single tokenize fails.
    let mut all_specs: Vec<rustllama_rag::ChunkSpec> = Vec::new();
    let mut spec_owner: Vec<usize> = Vec::new(); // index into `reindexed`
    for (owner_idx, (_, specs)) in reindexed.iter().enumerate() {
        for s in specs {
            all_specs.push(s.clone());
            spec_owner.push(owner_idx);
        }
    }

    let mut token_batches: Vec<Vec<i32>> = Vec::with_capacity(all_specs.len());
    for s in &all_specs {
        match tokenize_one(&bundle.tokenizer, &s.text) {
            Ok(ids) => token_batches.push(ids),
            Err(_resp) => token_batches.push(Vec::new()),
        }
    }
    let mut keep_idx: Vec<usize> = Vec::with_capacity(token_batches.len());
    let mut batches_to_embed: Vec<Vec<i32>> = Vec::with_capacity(token_batches.len());
    for (i, b) in token_batches.into_iter().enumerate() {
        if !b.is_empty() {
            keep_idx.push(i);
            batches_to_embed.push(b);
        }
    }
    let mut chunks_skipped = (all_specs.len() - keep_idx.len()) as u32;

    let vectors = if batches_to_embed.is_empty() {
        Vec::new()
    } else {
        match embed_batches(bundle.model.clone(), batches_to_embed).await {
            Ok(v) => v,
            Err(resp) => return resp,
        }
    };

    // Phase 3: mutate the index under the write lock. Remove every
    // path we're updating (existing or missing) before re-adding the
    // surviving chunks. Single critical section so /query never sees a
    // half-updated index.
    let files_removed_count;
    let chunks_added;
    {
        let mut idx = index_handle.write().await;
        let mut removed_paths: std::collections::HashSet<PathBuf> =
            std::collections::HashSet::new();
        for p in &removed_only {
            idx.remove_by_path(p);
            removed_paths.insert(p.clone());
        }
        for (p, _) in &reindexed {
            idx.remove_by_path(p);
            removed_paths.insert(p.clone());
        }
        let mut added = 0u32;
        for (vec, kept_i) in vectors.into_iter().zip(keep_idx.iter().copied()) {
            let spec = all_specs[kept_i].clone();
            if let Err(e) = idx.add(spec, vec) {
                tracing::error!(error = %e, "rag update: rejected chunk on add");
                chunks_skipped += 1;
            } else {
                added += 1;
            }
        }
        files_removed_count = removed_paths.len() as u32;
        chunks_added = added;
    }

    Json(RagUpdateResponse {
        object: "rag.update",
        files_removed: files_removed_count,
        files_reindexed: reindexed.len() as u32,
        chunks_added,
        chunks_skipped,
    })
    .into_response()
}

fn to_wire_hit(r: SearchResult) -> RagHit {
    RagHit {
        source_path: r.chunk.source_path.to_string_lossy().into_owned(),
        line_start: r.chunk.line_start as u32,
        line_end: r.chunk.line_end as u32,
        score: r.score,
        text: r.chunk.text,
    }
}

/// `POST /v1/rag/update` body.
#[derive(Deserialize)]
pub struct RagUpdateRequest {
    /// Workspace root the `paths` are interpreted against. Used to
    /// strip the prefix so chunks land with the same relative paths
    /// the initial `/v1/rag/index` produced.
    pub root: PathBuf,
    /// Files to re-index. Each path is interpreted relative to `root`
    /// (and is allowed to be absolute — the server resolves on disk
    /// either way). Files that no longer exist are removed from the
    /// index without re-adding.
    pub paths: Vec<PathBuf>,
    /// Optional chunk size + overlap overrides. Default to the
    /// chunker's `DEFAULT_CHUNK_LINES` / `DEFAULT_OVERLAP_LINES` when
    /// omitted, matching the `/v1/rag/index` defaults.
    #[serde(default)]
    pub chunk_lines: Option<usize>,
    #[serde(default)]
    pub overlap_lines: Option<usize>,
    /// Max bytes per file. Same default as the walker (1 MB).
    #[serde(default)]
    pub max_file_bytes: Option<u64>,
    /// Chunking strategy. Same semantics as `RagIndexRequest::mode`;
    /// see [`parse_chunking_mode`] for the accepted values.
    #[serde(default)]
    pub mode: Option<String>,
}

#[derive(Serialize)]
pub struct RagUpdateResponse {
    pub object: &'static str,
    /// Distinct paths whose chunks were removed before re-adding (or
    /// removed without re-adding if the file no longer exists).
    pub files_removed: u32,
    /// Distinct paths whose chunks were re-added after walking.
    pub files_reindexed: u32,
    /// Total chunks added after the update.
    pub chunks_added: u32,
    /// Chunks dropped from the embed step (tokenize / dimension
    /// failures); diagnostic — surfaces silent skips.
    pub chunks_skipped: u32,
}

/// `POST /v1/rag/save` body.
#[derive(Deserialize)]
pub struct RagSaveRequest {
    /// Absolute or relative path to write the index file to. Parent
    /// directory must exist; missing directories are caller-side
    /// fixable so we don't auto-mkdir.
    pub path: PathBuf,
}

#[derive(Serialize)]
pub struct RagSaveResponse {
    pub object: &'static str,
    pub path: String,
    pub chunks_saved: u32,
    pub embedding_dim: u32,
    pub bytes_written: u64,
}

/// `POST /v1/rag/load` body.
#[derive(Deserialize)]
pub struct RagLoadRequest {
    /// Index file written by a previous `/v1/rag/save` call. Replaces
    /// the in-memory store atomically on success; existing index is
    /// left untouched on failure.
    pub path: PathBuf,
}

#[derive(Serialize)]
pub struct RagLoadResponse {
    pub object: &'static str,
    pub path: String,
    pub chunks_loaded: u32,
    pub embedding_dim: u32,
}

pub async fn save(
    State(state): State<AppState>,
    Json(req): Json<RagSaveRequest>,
) -> Response {
    // Need a built index to save; 409 mirrors /query's "no index" path.
    let index_handle = {
        let slot = state.rag_index.read().await;
        match slot.as_ref() {
            Some(h) => h.clone(),
            None => {
                return (
                    StatusCode::CONFLICT,
                    Json(serde_json::json!({
                        "error": "no RAG index built",
                        "detail": "POST /v1/rag/index first to populate the in-memory index.",
                    })),
                )
                    .into_response();
            }
        }
    };

    let target = req.path.clone();
    let target_for_resp = req.path.clone();
    let result = tokio::task::spawn_blocking(move || -> Result<(u32, u32, u64), String> {
        // Block on the inner read lock from the blocking worker — the
        // index is owned by the server process so no async wait needed.
        // Use a runtime handle to do the lock cleanly.
        let rt = tokio::runtime::Handle::current();
        let idx = rt.block_on(async { index_handle.read().await });
        idx.save(&target).map_err(|e| e.to_string())?;
        let meta = std::fs::metadata(&target).map_err(|e| e.to_string())?;
        Ok((idx.len() as u32, idx.embedding_dim() as u32, meta.len()))
    })
    .await;

    match result {
        Ok(Ok((chunks_saved, embedding_dim, bytes_written))) => Json(RagSaveResponse {
            object: "rag.save",
            path: target_for_resp.to_string_lossy().into_owned(),
            chunks_saved,
            embedding_dim,
            bytes_written,
        })
        .into_response(),
        Ok(Err(msg)) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "could not save RAG index",
                "detail": msg,
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": "rag save worker panicked",
                "detail": e.to_string(),
            })),
        )
            .into_response(),
    }
}

pub async fn load(
    State(state): State<AppState>,
    Json(req): Json<RagLoadRequest>,
) -> Response {
    let target = req.path.clone();
    let target_for_resp = req.path.clone();

    // Read + parse the index file on a blocking worker — the file can
    // be tens of MB and we don't want to block the tokio runtime.
    let loaded = tokio::task::spawn_blocking(move || RagIndex::load(&target)).await;
    let new_index = match loaded {
        Ok(Ok(idx)) => idx,
        Ok(Err(e)) => {
            // Map index errors to sensible HTTP statuses. Bad-magic /
            // version / corruption are caller-side problems (4xx);
            // pure IO is 4xx too because the caller passed the path.
            let status = match &e {
                rustllama_rag::IndexError::Io(_)
                | rustllama_rag::IndexError::BadMagic
                | rustllama_rag::IndexError::UnsupportedVersion { .. }
                | rustllama_rag::IndexError::CorruptHeader(_) => StatusCode::BAD_REQUEST,
                rustllama_rag::IndexError::WrongDimension { .. } => {
                    StatusCode::BAD_REQUEST
                }
            };
            return (
                status,
                Json(serde_json::json!({
                    "error": "could not load RAG index",
                    "detail": e.to_string(),
                })),
            )
                .into_response();
        }
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "rag load worker panicked",
                    "detail": e.to_string(),
                })),
            )
                .into_response();
        }
    };

    // If the embedding bundle is also configured, sanity-check its
    // d_model matches the loaded index. Mismatch means /query would
    // return WrongDimension on every call — reject up front so the
    // user knows immediately. When `[embeddings]` isn't configured,
    // we accept the index anyway (the user can configure later).
    if let Some(slot) = state.embedding_model.as_ref() {
        if let Some(Ok(bundle)) = slot.get() {
            if bundle.model.cfg.d_model != new_index.embedding_dim() {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": "embedding dimension mismatch",
                        "detail": format!(
                            "loaded index has embedding_dim={} but configured embedding model has d_model={}",
                            new_index.embedding_dim(),
                            bundle.model.cfg.d_model,
                        ),
                    })),
                )
                    .into_response();
            }
        }
    }

    let chunks_loaded = new_index.len() as u32;
    let embedding_dim = new_index.embedding_dim() as u32;
    {
        let mut slot = state.rag_index.write().await;
        *slot = Some(Arc::new(tokio::sync::RwLock::new(new_index)));
    }
    Json(RagLoadResponse {
        object: "rag.load",
        path: target_for_resp.to_string_lossy().into_owned(),
        chunks_loaded,
        embedding_dim,
    })
    .into_response()
}
