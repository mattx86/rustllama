//! `POST /v1/rerank` — Cohere/Jina-shape reranker endpoint.
//!
//! Backed by a BGE-reranker-style cross-encoder GGUF (a BERT body
//! with a `cls.weight` classifier head) loaded via
//! [`AppState::reranker_model`]. The handler tokenizes
//! `(query, document)` pairs, runs the body, takes the CLS token,
//! and projects through the classifier to a relevance scalar.
//!
//! Wire shape (Cohere v2 / Jina):
//!
//!   Request  `{ query, documents: [string], top_n?, return_documents? }`
//!   Response `{ id, results: [{index, relevance_score, document?}],
//!               model, usage: {total_tokens} }`
//!
//! Results are sorted descending by `relevance_score` and truncated
//! to `top_n` (default: all). Empty `documents` returns 400.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use rustllama_models::bert_arch::{BertForwardError, BertModel};
use serde::{Deserialize, Serialize};

use crate::{AppState, LoadedRerankerModel};

#[derive(Deserialize)]
pub struct RerankRequest {
    /// Optional model id. Echoed in the response for clients that
    /// pin model id in their RAG metadata. Today only the
    /// configured `[reranker]` model is served.
    #[serde(default)]
    #[allow(dead_code)]
    pub model: Option<String>,
    pub query: String,
    pub documents: Vec<String>,
    /// Return only the top-N results. `None` returns all.
    #[serde(default)]
    pub top_n: Option<usize>,
    /// When true, each result entry includes the original document
    /// text. Default false to keep the response small.
    #[serde(default)]
    pub return_documents: bool,
    /// Accepted for shape compatibility (Cohere's `max_chunks_per_doc`
    /// + Jina's `truncate`) but ignored — we don't chunk.
    #[serde(default)]
    #[allow(dead_code)]
    pub max_chunks_per_doc: Option<u32>,
}

#[derive(Serialize)]
pub struct RerankResponse {
    pub id: String,
    pub results: Vec<RerankResult>,
    pub model: String,
    pub usage: RerankUsage,
}

#[derive(Serialize)]
pub struct RerankResult {
    pub index: u32,
    pub relevance_score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub document: Option<RerankDocument>,
}

#[derive(Serialize)]
pub struct RerankDocument {
    pub text: String,
}

#[derive(Serialize)]
pub struct RerankUsage {
    pub total_tokens: u32,
}

pub async fn rerank(
    State(state): State<AppState>,
    Json(req): Json<RerankRequest>,
) -> Response {
    if req.query.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": "query must be non-empty" })),
        )
            .into_response();
    }
    if req.documents.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "documents must be a non-empty array",
            })),
        )
            .into_response();
    }
    for (i, d) in req.documents.iter().enumerate() {
        if d.is_empty() {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "documents contains an empty string",
                    "input_index": i,
                })),
            )
                .into_response();
        }
    }

    let bundle = match get_reranker_bundle(&state) {
        Ok(b) => b,
        Err(resp) => return resp,
    };

    // Tokenize each (query, document) pair as a BERT sequence pair —
    // the WordPiece tokenizer's pair template inserts the
    // `[CLS] query [SEP] document [SEP]` layout with the right
    // token-type ids.
    let mut batches: Vec<Vec<i32>> = Vec::with_capacity(req.documents.len());
    for (i, doc) in req.documents.iter().enumerate() {
        let u32_ids = match bundle.tokenizer.encode_pair(&req.query, doc, true) {
            Ok(ids) => ids,
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({
                        "error": "rerank pair tokenize failed",
                        "input_index": i,
                        "detail": e.to_string(),
                    })),
                )
                    .into_response();
            }
        };
        let mut ids: Vec<i32> = Vec::with_capacity(u32_ids.len());
        for id in u32_ids {
            if id > i32::MAX as u32 {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({
                        "error": "tokenizer produced id outside i32 range",
                    })),
                )
                    .into_response();
            }
            ids.push(id as i32);
        }
        batches.push(ids);
    }
    let total_tokens: u32 = batches.iter().map(|b| b.len() as u32).sum();

    let scores = match classify_batches(bundle.model, batches).await {
        Ok(s) => s,
        Err(resp) => return resp,
    };

    // Sort by score desc, preserving original index for the result
    // entries. `top_n` truncates after sorting.
    let mut ranked: Vec<(usize, f32)> =
        scores.iter().enumerate().map(|(i, s)| (i, *s)).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let cap = req.top_n.unwrap_or(ranked.len()).min(ranked.len());
    let results: Vec<RerankResult> = ranked
        .into_iter()
        .take(cap)
        .map(|(idx, score)| RerankResult {
            index: idx as u32,
            relevance_score: score,
            document: if req.return_documents {
                Some(RerankDocument {
                    text: req.documents[idx].clone(),
                })
            } else {
                None
            },
        })
        .collect();

    Json(RerankResponse {
        id: format!("rerank-{}", std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)),
        results,
        model: req.model.unwrap_or_else(|| "rustllama-reranker".to_string()),
        usage: RerankUsage { total_tokens },
    })
    .into_response()
}

/// Lazy-resolve the reranker bundle, matching the embedding
/// equivalent in shape: 501 when unconfigured, 500 on load error.
pub(crate) fn get_reranker_bundle(
    state: &AppState,
) -> std::result::Result<LoadedRerankerModel, Response> {
    let Some(slot) = state.reranker_model.as_ref() else {
        return Err((
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({
                "error": "no reranker model configured",
                "detail": "Set [reranker].path = \"…\" or [reranker].hub = \"owner/repo:filename\" \
                           in config.toml + restart the server. Point at a BGE-reranker (or \
                           any BERT GGUF carrying a `cls.weight` classifier head).",
            })),
        )
            .into_response());
    };
    let load_result = slot.get_or_init(|| load_reranker_bundle_blocking(state));
    match load_result {
        Ok(b) => Ok(b.clone()),
        Err(msg) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": "failed to load reranker model",
                "detail": msg,
            })),
        )
            .into_response()),
    }
}

/// Run `forward_classify` per batch on a blocking worker. Returns
/// a per-input scalar score (the first logit of the classifier
/// output — BGE-reranker is single-label, so it's the relevance
/// score directly).
async fn classify_batches(
    model: Arc<BertModel>,
    batches: Vec<Vec<i32>>,
) -> std::result::Result<Vec<f32>, Response> {
    let n_inputs = batches.len();
    let join = tokio::task::spawn_blocking(move || {
        let mut scores: Vec<f32> = Vec::with_capacity(n_inputs);
        for (i, tokens) in batches.iter().enumerate() {
            let v = model.forward_classify(tokens).map_err(|e| (i, e))?;
            // For multi-label heads we take the first logit. Real
            // BGE-reranker ships `n_labels = 1` so this is the only
            // logit; for multi-label heads the convention is that
            // position 0 is the "relevance" class.
            let score = v.first().copied().unwrap_or(0.0);
            scores.push(score);
        }
        Ok::<_, (usize, BertForwardError)>(scores)
    })
    .await;

    match join {
        Ok(Ok(v)) => Ok(v),
        Ok(Err((i, err))) => {
            // `NoClassifierHead` is the only 5xx path here — that
            // means the loaded GGUF is an embedding-only model, a
            // server-config bug that the user can fix by switching
            // to a reranker GGUF. Everything else is caller input.
            let status = match err {
                BertForwardError::NoClassifierHead => StatusCode::INTERNAL_SERVER_ERROR,
                _ => StatusCode::BAD_REQUEST,
            };
            Err((
                status,
                Json(serde_json::json!({
                    "error": "rerank forward pass failed",
                    "input_index": i,
                    "detail": err.to_string(),
                })),
            )
                .into_response())
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": "rerank worker panicked",
                "detail": e.to_string(),
            })),
        )
            .into_response()),
    }
}

/// Load the configured reranker model + tokenizer. Rejects pure
/// embedding GGUFs (no classifier head) with a clear error rather
/// than silently giving 500s at first request time.
fn load_reranker_bundle_blocking(
    state: &AppState,
) -> std::result::Result<LoadedRerankerModel, String> {
    let cfg = state
        .config_path
        .as_ref()
        .and_then(|p| rustllama_config::load(p.as_ref()).ok())
        .ok_or_else(|| "could not load config.toml to resolve [reranker]".to_string())?;
    let path = resolve_reranker_path(&cfg)?;
    let gguf = rustllama_gguf::Gguf::open(&path)
        .map_err(|e| format!("failed to open reranker GGUF at {}: {e}", path.display()))?;
    let model = BertModel::load(&gguf)
        .map_err(|e| format!("failed to parse reranker GGUF as BERT: {e}"))?;
    if model.classifier_head.is_none() {
        return Err(format!(
            "GGUF at {} has no `cls.weight` / `classifier.weight` tensor — \
             this is an embedding model, not a reranker. Use [embeddings] \
             instead, or point [reranker] at a BGE-reranker GGUF.",
            path.display()
        ));
    }
    let tokenizer = rustllama_tokenizer::Tokenizer::from_gguf(&gguf)
        .map_err(|e| format!("failed to load reranker tokenizer from GGUF: {e}"))?;
    tracing::info!(
        path = %path.display(),
        arch = %model.cfg.arch,
        d_model = model.cfg.d_model,
        n_layers = model.cfg.n_layers,
        n_labels = model.classifier_head.as_ref().map(|h| h.n_labels).unwrap_or(0),
        vocab = tokenizer.vocab_size(),
        "reranker model + tokenizer loaded"
    );
    Ok(LoadedRerankerModel {
        model: Arc::new(model),
        tokenizer: Arc::new(tokenizer),
    })
}

fn resolve_reranker_path(
    cfg: &rustllama_config::Config,
) -> std::result::Result<std::path::PathBuf, String> {
    if let Some(p) = cfg.reranker.path.as_ref() {
        if !p.exists() {
            return Err(format!("[reranker].path does not exist: {}", p.display()));
        }
        return Ok(p.clone());
    }
    if let Some(hub) = cfg.reranker.hub.as_ref() {
        let href = rustllama_hub::HubRef::parse(hub)
            .map_err(|e| format!("invalid [reranker].hub `{hub}`: {e}"))?;
        let cache = rustllama_hub::default_cache_dir()
            .ok_or_else(|| "no hub cache dir resolvable".to_string())?;
        let p = href.local_path(&cache);
        if !p.exists() {
            return Err(format!(
                "[reranker].hub `{hub}` not in cache at {}. Run `rustllama pull {hub}` first.",
                p.display()
            ));
        }
        return Ok(p);
    }
    Err("neither [reranker].path nor [reranker].hub is set".to_string())
}
