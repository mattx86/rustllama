//! OpenAI-compatible `POST /v1/embeddings`.
//!
//! Workflow tiers:
//!   - `[embeddings]` unconfigured → 501 with "no embedding model
//!     configured" hint pointing at the config field
//!   - configured + text input → tokenize via the WordPiece tokenizer
//!     loaded from the same GGUF (lazy, cached) then forward through
//!     `BertModel::forward_embed`
//!   - configured + pre-tokenized input (int arrays per OpenAI spec
//!     extension) → forward directly, skipping tokenization
//!
//! Both the model and the tokenizer are lazy-loaded together via the
//! `AppState::embedding_model` `OnceLock` slot — first request pays
//! the load cost (~ms for a small BGE GGUF), subsequent requests
//! hit the cached `Arc`s.

use std::sync::Arc;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use rustllama_models::bert_arch::{BertForwardError, BertModel};
use serde::{Deserialize, Serialize};

use crate::{AppState, LoadedEmbeddingModel};

#[derive(Deserialize)]
pub struct EmbeddingsRequest {
    /// Embedding model id. Today only the configured `[embeddings]`
    /// model is served — the field is accepted for OpenAI-shape
    /// compatibility but the per-request model override is a
    /// follow-up (would need a multi-model embedding registry).
    #[serde(default)]
    pub model: Option<String>,
    pub input: EmbeddingsInput,
    /// `"float"` (default) returns plain JSON arrays of f32.
    /// `"base64"` returns each vector as a single LE-f32 base64
    /// blob — saves wire bytes for big batches (~33% smaller than
    /// the JSON-array form). Other values → 400.
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// MRL / Matryoshka truncated-dim support. When set, each
    /// embedding vector is truncated to the first `dimensions`
    /// entries and L2-renormalized (the standard MRL convention).
    /// The model itself must have been MRL-trained for the
    /// truncated vector to be semantically meaningful; we don't
    /// gate on that — clients that pass `dimensions` know what
    /// model they're talking to. Must be ≤ the model's d_model
    /// or 400.
    #[serde(default)]
    pub dimensions: Option<u32>,
}

/// OpenAI accepts `input` as either a single string or array. We
/// also accept pre-tokenized int arrays (a documented OpenAI
/// extension used by power users + clients that want to skip our
/// tokenizer). Untagged ordering matters for serde — string forms
/// must come first so they win on ambiguous shapes.
#[derive(Deserialize)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    Single(String),
    Batch(Vec<String>),
    /// Pre-tokenized single sequence (e.g. `[101, 7592, 2088, 102]`).
    SingleTokens(Vec<i32>),
    /// Pre-tokenized batch (`[[101, 7592, 102], [101, 2088, 102]]`).
    BatchTokens(Vec<Vec<i32>>),
}

impl EmbeddingsInput {
    fn len(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::SingleTokens(_) => 1,
            Self::Batch(v) => v.len(),
            Self::BatchTokens(v) => v.len(),
        }
    }
}

#[derive(Serialize)]
pub struct EmbeddingsResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingItem>,
    pub model: String,
    pub usage: EmbeddingsUsage,
}

#[derive(Serialize)]
pub struct EmbeddingItem {
    pub object: &'static str,
    pub index: u32,
    pub embedding: EmbeddingValue,
}

/// Either a JSON array of floats (`encoding_format: "float"`,
/// default) or a base64-encoded LE-f32 blob
/// (`encoding_format: "base64"`). `serde(untagged)` selects the
/// JSON shape automatically — clients get an array or a string
/// depending on which variant we return.
#[derive(Serialize)]
#[serde(untagged)]
pub enum EmbeddingValue {
    Floats(Vec<f32>),
    Base64(String),
}

#[derive(Serialize)]
pub struct EmbeddingsUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

pub async fn embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingsRequest>,
) -> Response {
    if req.input.len() == 0 {
        return (
            StatusCode::BAD_REQUEST,
            "input must be a non-empty string or array",
        )
            .into_response();
    }

    // Validate `encoding_format` up-front. OpenAI accepts only
    // "float" and "base64"; anything else is a client error.
    let encode_as_base64 = match req.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "invalid encoding_format",
                    "detail": format!(
                        "encoding_format must be \"float\" or \"base64\"; got {other:?}"
                    ),
                })),
            )
                .into_response();
        }
    };

    let bundle = match get_embedding_bundle(&state) {
        Ok(b) => b,
        Err(resp) => return resp,
    };

    // Validate `dimensions` against the loaded model's d_model.
    // OpenAI's MRL convention: clients pass `dimensions` ≤ d_model;
    // anything larger is a client error (we can't fabricate extra
    // dimensions). 0 is meaningless and gets rejected too.
    let truncate_to = if let Some(d) = req.dimensions {
        let d_model = bundle.model.cfg.d_model as u32;
        if d == 0 || d > d_model {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": "invalid dimensions",
                    "detail": format!(
                        "dimensions must be in 1..={d_model} (model d_model); got {d}"
                    ),
                })),
            )
                .into_response();
        }
        Some(d as usize)
    } else {
        None
    };

    // Tokenize text inputs (using the bundled tokenizer) into the
    // same i32 batch shape the pre-tokenized fast-path produces.
    // Pre-tokenized inputs skip the tokenize step entirely.
    let batches: Vec<Vec<i32>> = match req.input {
        EmbeddingsInput::SingleTokens(t) => vec![t],
        EmbeddingsInput::BatchTokens(b) => b,
        EmbeddingsInput::Single(s) => match tokenize_one(&bundle.tokenizer, &s) {
            Ok(ids) => vec![ids],
            Err(resp) => return resp,
        },
        EmbeddingsInput::Batch(strs) => {
            let mut out = Vec::with_capacity(strs.len());
            for (i, s) in strs.iter().enumerate() {
                match tokenize_one(&bundle.tokenizer, s) {
                    Ok(ids) => out.push(ids),
                    Err(resp) => {
                        tracing::warn!(input_index = i, "embeddings tokenize failed");
                        return resp;
                    }
                }
            }
            out
        }
    };

    let total_tokens: u32 = batches.iter().map(|b| b.len() as u32).sum();
    let vectors = match embed_batches(bundle.model, batches).await {
        Ok(v) => v,
        Err(resp) => return resp,
    };

    let data: Vec<EmbeddingItem> = vectors
        .into_iter()
        .enumerate()
        .map(|(i, mut v)| {
            // Matryoshka truncation: keep the first `truncate_to`
            // entries and L2-renormalize. The model has to be
            // MRL-trained for the truncated vector to retain its
            // semantic structure; we don't gate on that, but the
            // renorm is the standard convention so cosine
            // similarity stays well-defined.
            if let Some(d) = truncate_to {
                v.truncate(d);
                l2_normalize_inplace(&mut v);
            }
            let embedding = if encode_as_base64 {
                EmbeddingValue::Base64(encode_le_f32_base64(&v))
            } else {
                EmbeddingValue::Floats(v)
            };
            EmbeddingItem {
                object: "embedding",
                index: i as u32,
                embedding,
            }
        })
        .collect();

    Json(EmbeddingsResponse {
        object: "list",
        data,
        // No per-request model field today (we only serve the
        // configured one); echo a stable label so clients pinning
        // model id in their RAG metadata get a consistent value.
        model: req.model.unwrap_or_else(|| "rustllama-embeddings".to_string()),
        usage: EmbeddingsUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    })
    .into_response()
}

/// Resolve the lazy-loaded embedding bundle for the given state.
/// Returns the bundle on success or an HTTP response describing the
/// failure (501 when not configured, 500 on load error). Used by
/// both the OpenAI handler and the Ollama-shape adapters.
pub(crate) fn get_embedding_bundle(
    state: &AppState,
) -> std::result::Result<LoadedEmbeddingModel, Response> {
    let Some(slot) = state.embedding_model.as_ref() else {
        return Err((
            StatusCode::NOT_IMPLEMENTED,
            Json(serde_json::json!({
                "error": "no embedding model configured",
                "detail": "Set [embeddings].path = \"…\" or [embeddings].hub = \"owner/repo:filename\" \
                           in config.toml + restart the server. Both text and pre-tokenized \
                           int-array inputs are accepted at this endpoint.",
            })),
        )
            .into_response());
    };
    let load_result = slot.get_or_init(|| load_embedding_bundle_blocking(state));
    match load_result {
        Ok(b) => Ok(b.clone()),
        Err(msg) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": "failed to load embedding model",
                "detail": msg,
            })),
        )
            .into_response()),
    }
}

/// Run `forward_embed` per pre-tokenized input batch on a blocking
/// worker. Maps `BertForwardError` to the appropriate 400/500
/// response (with `input_index` so the client can fix the offending
/// entry without re-running the whole batch).
pub(crate) async fn embed_batches(
    model: Arc<BertModel>,
    batches: Vec<Vec<i32>>,
) -> std::result::Result<Vec<Vec<f32>>, Response> {
    let n_inputs = batches.len();
    let join = tokio::task::spawn_blocking(move || {
        let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(n_inputs);
        for (i, tokens) in batches.iter().enumerate() {
            let v = model.forward_embed(tokens).map_err(|e| (i, e))?;
            vectors.push(v);
        }
        Ok::<_, (usize, BertForwardError)>(vectors)
    })
    .await;

    match join {
        Ok(Ok(v)) => Ok(v),
        Ok(Err((i, err))) => {
            // 4xx for caller-side problems (input shape / sizing); 5xx
            // for the impossible-here `NoClassifierHead` (the embedding
            // path never invokes the classifier head). `detail` carries
            // the thiserror `Display` form which already includes the
            // concrete sizes / token ids — clients no longer have to
            // guess "how much to trim" from a generic prefix.
            let status = match err {
                BertForwardError::NoClassifierHead => StatusCode::INTERNAL_SERVER_ERROR,
                _ => StatusCode::BAD_REQUEST,
            };
            Err((
                status,
                Json(serde_json::json!({
                    "error": "embedding forward pass failed",
                    "input_index": i,
                    "detail": err.to_string(),
                })),
            )
                .into_response())
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": "embedding worker panicked",
                "detail": e.to_string(),
            })),
        )
            .into_response()),
    }
}

/// Load the configured embedding model + tokenizer from disk in a
/// single open. Runs on the `get_or_init` path inside `OnceLock`, so
/// the cost is paid once per server lifetime. Returns the
/// load-error string on failure so the cache holds the failure too
/// (subsequent requests don't retry the doomed load).
fn load_embedding_bundle_blocking(
    state: &AppState,
) -> std::result::Result<LoadedEmbeddingModel, String> {
    let cfg = state
        .config_path
        .as_ref()
        .and_then(|p| rustllama_config::load(p.as_ref()).ok())
        .ok_or_else(|| "could not load config.toml to resolve [embeddings]".to_string())?;
    let path = resolve_embedding_path(&cfg)?;
    let gguf = rustllama_gguf::Gguf::open(&path)
        .map_err(|e| format!("failed to open embedding GGUF at {}: {e}", path.display()))?;
    let model = BertModel::load(&gguf)
        .map_err(|e| format!("failed to parse embedding GGUF as BERT: {e}"))?;
    let tokenizer = rustllama_tokenizer::Tokenizer::from_gguf(&gguf)
        .map_err(|e| format!("failed to load BERT tokenizer from GGUF: {e}"))?;
    tracing::info!(
        path = %path.display(),
        arch = %model.cfg.arch,
        d_model = model.cfg.d_model,
        n_layers = model.cfg.n_layers,
        vocab = tokenizer.vocab_size(),
        "embedding model + tokenizer loaded"
    );
    Ok(LoadedEmbeddingModel {
        model: Arc::new(model),
        tokenizer: Arc::new(tokenizer),
    })
}

/// Tokenize one string for the BERT embedding path. Wraps the
/// `[CLS] … [SEP]` markers via `add_special_tokens=true` (the
/// post-processor in the WordPiece tokenizer handles the
/// wrapping). Returns the ids as i32 because that's what
/// `BertModel::forward_embed` consumes — WordPiece ids are
/// always within u16 range for real-world vocabs, so the cast
/// is safe in practice; we guard explicitly anyway.
pub(crate) fn tokenize_one(
    tokenizer: &rustllama_tokenizer::Tokenizer,
    text: &str,
) -> std::result::Result<Vec<i32>, Response> {
    let u32_ids = tokenizer.encode(text, true).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "tokenize failed",
                "detail": e.to_string(),
            })),
        )
            .into_response()
    })?;
    // i32::MAX is well above any real WordPiece vocab — bail
    // loudly if a model ever exceeds it rather than truncating.
    let mut out = Vec::with_capacity(u32_ids.len());
    for id in u32_ids {
        if id > i32::MAX as u32 {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": "tokenizer produced id outside i32 range",
                    "detail": format!("id = {id}"),
                })),
            )
                .into_response());
        }
        out.push(id as i32);
    }
    Ok(out)
}

/// Encode `vec` as a single base64 string of its little-endian f32
/// bytes — matches OpenAI's `encoding_format: "base64"` wire shape.
/// Inline encoder (~20 lines) avoids pulling a base64 crate for one
/// use site. Uses the standard alphabet with `=` padding.
fn encode_le_f32_base64(vec: &[f32]) -> String {
    // Materialize the LE-f32 byte slab. `bytemuck` would do the
    // cast for free but is overkill — a small loop is clearer here.
    let mut bytes = Vec::with_capacity(vec.len() * 4);
    for f in vec {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    base64_standard_encode(&bytes)
}

/// Standard-alphabet base64 encode (`A-Za-z0-9+/`) with `=` padding.
/// Returns ASCII so a `String` is safe to construct directly.
fn base64_standard_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = Vec::with_capacity((bytes.len() + 2) / 3 * 4);
    let mut i = 0;
    while i + 3 <= bytes.len() {
        let n = (bytes[i] as u32) << 16 | (bytes[i + 1] as u32) << 8 | (bytes[i + 2] as u32);
        out.push(ALPHABET[((n >> 18) & 0x3f) as usize]);
        out.push(ALPHABET[((n >> 12) & 0x3f) as usize]);
        out.push(ALPHABET[((n >> 6) & 0x3f) as usize]);
        out.push(ALPHABET[(n & 0x3f) as usize]);
        i += 3;
    }
    let rem = bytes.len() - i;
    if rem == 1 {
        let n = (bytes[i] as u32) << 16;
        out.push(ALPHABET[((n >> 18) & 0x3f) as usize]);
        out.push(ALPHABET[((n >> 12) & 0x3f) as usize]);
        out.push(b'=');
        out.push(b'=');
    } else if rem == 2 {
        let n = (bytes[i] as u32) << 16 | (bytes[i + 1] as u32) << 8;
        out.push(ALPHABET[((n >> 18) & 0x3f) as usize]);
        out.push(ALPHABET[((n >> 12) & 0x3f) as usize]);
        out.push(ALPHABET[((n >> 6) & 0x3f) as usize]);
        out.push(b'=');
    }
    // ASCII-only by construction.
    String::from_utf8(out).expect("base64 alphabet is ASCII")
}

/// L2-normalize `vec` in place. Used after MRL truncation so cosine
/// similarity over the truncated vectors stays well-defined. No-ops
/// on the zero vector (which would otherwise produce NaN).
fn l2_normalize_inplace(vec: &mut [f32]) {
    let sum_sq: f32 = vec.iter().map(|v| v * v).sum();
    let norm = sum_sq.sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for v in vec.iter_mut() {
            *v *= inv;
        }
    }
}

fn resolve_embedding_path(cfg: &rustllama_config::Config) -> std::result::Result<std::path::PathBuf, String> {
    if let Some(p) = cfg.embeddings.path.as_ref() {
        if !p.exists() {
            return Err(format!(
                "[embeddings].path does not exist: {}",
                p.display()
            ));
        }
        return Ok(p.clone());
    }
    if let Some(hub) = cfg.embeddings.hub.as_ref() {
        let href = rustllama_hub::HubRef::parse(hub)
            .map_err(|e| format!("invalid [embeddings].hub `{hub}`: {e}"))?;
        let cache = rustllama_hub::default_cache_dir()
            .ok_or_else(|| "no hub cache dir resolvable".to_string())?;
        let p = href.local_path(&cache);
        if !p.exists() {
            return Err(format!(
                "[embeddings].hub `{hub}` not in cache at {}. Run `rustllama pull {hub}` first.",
                p.display()
            ));
        }
        return Ok(p);
    }
    Err("neither [embeddings].path nor [embeddings].hub is set".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_encode_known_fixtures() {
        // Standard RFC 4648 vectors.
        assert_eq!(base64_standard_encode(b""), "");
        assert_eq!(base64_standard_encode(b"f"), "Zg==");
        assert_eq!(base64_standard_encode(b"fo"), "Zm8=");
        assert_eq!(base64_standard_encode(b"foo"), "Zm9v");
        assert_eq!(base64_standard_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_standard_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_standard_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn encode_le_f32_base64_roundtrips_through_bytes() {
        // Pick a few f32s with distinctive bit patterns and verify
        // the encoded blob matches the standard-base64 of their LE
        // byte interleave.
        let vec = vec![1.0f32, -1.0, 0.0, std::f32::consts::PI];
        let mut bytes = Vec::with_capacity(vec.len() * 4);
        for f in &vec {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        assert_eq!(encode_le_f32_base64(&vec), base64_standard_encode(&bytes));
    }

    #[test]
    fn l2_normalize_zero_vector_is_noop() {
        let mut v = vec![0.0f32; 4];
        l2_normalize_inplace(&mut v);
        assert!(v.iter().all(|x| *x == 0.0), "zero vec must stay zero");
    }

    #[test]
    fn l2_normalize_unit_vector_stays_unit() {
        let mut v = vec![1.0f32, 0.0, 0.0, 0.0];
        l2_normalize_inplace(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm should be 1, got {norm}");
    }

    #[test]
    fn l2_normalize_arbitrary_vector_produces_unit_norm() {
        let mut v = vec![3.0f32, 4.0]; // norm = 5
        l2_normalize_inplace(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }
}
