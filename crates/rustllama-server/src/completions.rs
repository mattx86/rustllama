//! `POST /v1/completions` — OpenAI-compatible legacy completion endpoint.
//!
//! Supports two modes:
//!   - **Plain completion**: `{ prompt }` — tokenize the prompt, generate.
//!   - **FIM (Fill-In-Middle)**: `{ prompt, suffix }` — assemble the prompt
//!     with the model's FIM special tokens so the LLM "fills the gap"
//!     between prefix and suffix. This is what editor inline-completion
//!     tools (Continue.dev, Tabby, llama-vscode, …) use.
//!
//! FIM token detection auto-selects Qwen2.5-Coder, DeepSeek-Coder, and
//! StarCoder/CodeLlama templates. If the loaded model doesn't carry FIM
//! special tokens, supplying `suffix` returns 400.

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use rustllama_engine::{SamplingParams, Token};
use serde::{Deserialize, Serialize};

use crate::{lora_not_found, model_not_found, AppState, ReqSchedule, Usage};

#[derive(Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: String,
    /// FIM suffix — when set, prompt is wrapped in the model's FIM special tokens.
    pub suffix: Option<String>,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    /// Locally-typical sampling threshold (Meister et al. 2022).
    pub typical_p: Option<f32>,
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stop: Vec<String>,
    pub seed: Option<u64>,
    pub repeat_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    /// Mirostat sampler mode: `0` disabled (default), `1` = v1, `2` = v2.
    /// Non-OpenAI extension field — included for parity with the chat
    /// endpoint and editor integrations.
    pub mirostat: Option<u32>,
    pub mirostat_tau: Option<f32>,
    pub mirostat_eta: Option<f32>,
    /// min-p sampling: keep only tokens with `p >= min_p * p_max`. `0`
    /// disables.
    pub min_p: Option<f32>,
    /// OpenAI `logit_bias`: `{ "<token_id>": bias }` added to logits
    /// before sampling.
    pub logit_bias: Option<std::collections::HashMap<String, f32>>,
    /// DRY sequence-repetition penalty. `0` (default) disables.
    pub dry_multiplier: Option<f32>,
    pub dry_base: Option<f32>,
    pub dry_allowed_length: Option<u32>,
    /// DRY sequence breakers as TOKEN IDS.
    pub dry_sequence_breakers: Option<Vec<u32>>,
    /// XTC (eXclude Top Choices) sampler. `0` probability disables.
    pub xtc_probability: Option<f32>,
    pub xtc_threshold: Option<f32>,
    /// Dynamic-temperature (dynatemp). `0` range disables.
    pub dynatemp_range: Option<f32>,
    pub dynatemp_exponent: Option<f32>,
    #[serde(default)]
    pub echo: bool,
    /// OpenAI streaming knob. `include_usage: true` adds a final usage
    /// chunk before `[DONE]`.
    pub stream_options: Option<crate::StreamOptions>,
    /// Legacy-completions `logprobs`: an integer 0..=5 giving the number of
    /// top alternative logprobs to return per token. `None` disables.
    pub logprobs: Option<u32>,
    /// P12a — OpenAI `n`: how many independent completions to return for the
    /// prompt. `None` / `1` (default) is the single-completion path, which is
    /// byte-identical to before this field existed. `> 1` returns `n`
    /// `choices`, each a distinct seeded sample that shares the prompt prefix.
    /// Not supported with `stream: true`.
    pub n: Option<u32>,
    /// P12a — OpenAI `best_of`: generate `best_of` completions server-side and
    /// return the `n` best by length-normalized cumulative logprob. Defaults
    /// to `n`; must be `>= n`. `best_of == n == 1` is the untouched single
    /// path. Not supported with `stream: true`.
    pub best_of: Option<u32>,
    /// P12b — beam search (rustllama extension; OpenAI has no beam API).
    /// `None` / `1` (default) is the ordinary sample-and-decode path. `> 1`
    /// switches to DETERMINISTIC beam search with this many beams, returning
    /// the top `n` (default 1) hypotheses ranked by length-penalty-adjusted
    /// cumulative log-prob. Beam search ignores the stochastic sampler
    /// (temperature/top-k/top-p) and the `best_of` sample-ranking path — it is
    /// its own decode mode. Not supported with `stream: true`.
    pub num_beams: Option<u32>,
    /// P12b — length penalty (GNMT/HuggingFace exponent) applied when ranking
    /// finished beams: `score = cum_logprob / len^length_penalty`. Default
    /// `1.0`. `> 1` favors longer sequences, `< 1` favors shorter. Only
    /// consulted when `num_beams > 1`.
    pub length_penalty: Option<f32>,
    /// P12b — stop beam search as soon as `num_beams` finished hypotheses
    /// exist. Default `false` (keep expanding until the best finished beam is
    /// provably unbeatable). Only consulted when `num_beams > 1`.
    pub early_stopping: Option<bool>,
    /// CacheBlend (rustllama extension): retrieved RAG context chunks, each a
    /// document whose KV is precomputed in isolation and blended into the
    /// prompt. They are prepended to `prompt` as context; when CacheBlend is
    /// enabled (`[inference].cacheblend_recompute_ratio > 0`) the non-streaming
    /// path reuses each chunk's cached KV instead of a full prefill. Ignored
    /// (treated as plain prepended context) when empty or CacheBlend is off.
    #[serde(default)]
    pub documents: Option<Vec<String>>,
    /// P15 — GBNF grammar (llama.cpp-server-style field). When set to a
    /// non-empty GBNF grammar text, the output is constrained to the grammar
    /// (the sampler masks out any token that would break it). A malformed
    /// grammar is rejected with a 400 before generation. See
    /// [`rustllama_engine::GrammarKind::Gbnf`].
    #[serde(default)]
    pub grammar: Option<String>,
    /// P11 (multi-LoRA) — name of a resident LoRA adapter to apply for this
    /// request (rustllama extension; the OpenAI `model: "<base>:<adapter>"`
    /// convention is also accepted and resolved to the same selection). `None`
    /// = the un-adapted base. An unknown name is a 4xx before generation. See
    /// [`rustllama_engine::SamplingParams::lora_adapter`].
    #[serde(default)]
    pub lora: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// OpenAI-compat `system_fingerprint`: stable hash of
    /// `(server_version, model_id, kv_dtype)`. Editor clients
    /// (Aider, Continue) compare it across consecutive requests to
    /// detect mid-conversation backend swaps. Same wire format as
    /// `/v1/chat/completions`.
    pub system_fingerprint: String,
}

#[derive(Debug, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<LogprobsOut>,
    pub finish_reason: &'static str,
}

/// Per-OpenAI legacy completions logprobs shape:
///   { tokens: [...], token_logprobs: [...], top_logprobs: [{...}, ...],
///     text_offset: [...] }
///
/// `token_logprobs` / `top_logprobs` are `Option` so the FIRST prompt token on
/// the `echo: true` + `logprobs` path serializes as JSON `null` (it has no
/// preceding context, so its conditional logprob is undefined — OpenAI's
/// contract). On the completion-only path every entry is `Some`, which
/// serializes byte-identically to the pre-echo `Vec<f32>` / `Vec<Map>` shape.
#[derive(Debug, Serialize)]
pub struct LogprobsOut {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<Option<f32>>,
    pub top_logprobs: Vec<Option<serde_json::Map<String, serde_json::Value>>>,
    pub text_offset: Vec<usize>,
}

pub async fn completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CompletionRequest>,
) -> Response {
    // P14-http: the completions endpoint is ContiguousEngine-only (it 503s on the
    // paged engine and drives generation directly off `shared_cpu`), so the
    // paged driver's deadline/heap don't apply here; carry only the admission
    // priority into the server scheduler slot. Default (no header) is
    // byte-identical to before.
    let sched = ReqSchedule::from_headers(&headers);
    // P11: resolve the model honoring the `model: "<base>:<adapter>"` LoRA
    // convention; `model_adapter` is the adapter peeled from the model string
    // (if any). The explicit `lora` field (set in `build_sampling`) takes
    // precedence over it.
    let Some((serving, model_adapter)) = state.resolve_with_lora(req.model.as_deref()).await else {
        return model_not_found(req.model.as_deref());
    };
    // P15: validate a supplied GBNF `grammar` up front so a malformed grammar
    // is a clean 400 instead of being silently dropped at the engine dispatch.
    if let Some(g) = req.grammar.as_ref().filter(|g| !g.trim().is_empty()) {
        if let Err(e) = rustllama_engine::gbnf::validate_gbnf(g) {
            return (
                StatusCode::BAD_REQUEST,
                axum::Json(serde_json::json!({
                    "error": "invalid GBNF grammar",
                    "detail": e.to_string(),
                })),
            )
                .into_response();
        }
    }
    // P12a: resolve n / best_of. `best_of` defaults to `n` and must be >= n;
    // `n_candidates = max(n, best_of) = best_of` completions are generated,
    // and the top `n_choices` are returned. `n_candidates == 1` is the
    // untouched single-completion path. Streaming + multi is rejected (OpenAI
    // disallows `best_of` with stream, and we don't multiplex `n` choices
    // over one SSE channel).
    let n_choices = req.n.unwrap_or(1).max(1);
    let best_of = req.best_of.unwrap_or(n_choices);
    if best_of < n_choices {
        return (StatusCode::BAD_REQUEST, "best_of must be >= n".to_string()).into_response();
    }
    let n_candidates = n_choices.max(best_of);
    if n_candidates > 1 && req.stream {
        return (
            StatusCode::BAD_REQUEST,
            "n > 1 / best_of > 1 is not supported with stream=true".to_string(),
        )
            .into_response();
    }
    // P12b — beam search. `num_beams > 1` is a distinct, deterministic decode
    // mode that takes precedence over the n/best_of sample-ranking path. Like
    // the other multi-output modes it can't be streamed (beams reorder as they
    // expand), so reject stream up front with a clean 400.
    let num_beams = req.num_beams.unwrap_or(1).max(1);
    if num_beams > 1 && req.stream {
        return (
            StatusCode::BAD_REQUEST,
            "beam search (num_beams > 1) is not supported with stream=true".to_string(),
        )
            .into_response();
    }
    // Two-phase admission: tokenizer access + FIM prompt assembly +
    // prompt-token count all happen against the Arc-shared CPU engine
    // BEFORE the gate wait, so the work overlaps any in-flight
    // request's decode.
    let pregate_permit = match serving.try_admit_with_priority(sched.priority()) {
        Ok(p) => p,
        Err(be) => return be.into_response(),
    };
    let mut sampling = build_sampling_with_base(
        &req,
        serving.engine.recommended_sampling().unwrap_or_default(),
    );
    // P11: explicit `lora` field (already set by `build_sampling`) wins; else
    // fall back to the `model:adapter` convention. Then validate the final
    // selection against the resolved engine — an unknown adapter is a clean 404,
    // never a silent base serve.
    if sampling.lora_adapter.is_none() {
        sampling.lora_adapter = model_adapter;
    }
    if let Some(name) = sampling.lora_adapter.as_deref() {
        if !serving.engine.has_lora_adapter(name) {
            return lora_not_found(name);
        }
    }
    let model = req
        .model
        .clone()
        .unwrap_or_else(|| serving.model_id.clone());

    let shared_cpu = match serving.cpu_engine.as_ref() {
        Some(e) => e.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "completions endpoint requires a real (non-mock) engine",
            )
                .into_response();
        }
    };
    let tokenizer = match shared_cpu.tokenizer() {
        Some(t) => t,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "completions endpoint requires a model with a tokenizer",
            )
                .into_response();
        }
    };

    let query_ids = match build_prompt_ids(tokenizer, &req.prompt, req.suffix.as_deref()) {
        Ok(ids) => ids,
        Err(e) => return (StatusCode::BAD_REQUEST, e).into_response(),
    };

    // CacheBlend RAG chunks: tokenize each document independently with NO
    // special tokens (they appear mid-context, not at a sequence boundary) —
    // this is what the isolated-at-position-0 KV precompute expects. Done here,
    // pre-gate, so it overlaps any in-flight decode like the prompt tokenize.
    let doc_chunks: Vec<Vec<i32>> = req
        .documents
        .iter()
        .flatten()
        .filter_map(|d| {
            tokenizer
                .encode(d, false)
                .ok()
                .filter(|ids| !ids.is_empty())
                .map(|ids| ids.iter().map(|&v| v as i32).collect())
        })
        .collect();

    // The full context = documents prepended to the query. Every existing
    // generation arm uses this unchanged (documents become plain context); only
    // the CacheBlend arm splits it back into (chunks, query) to reuse chunk KV.
    let prompt_ids: Vec<u32> = if doc_chunks.is_empty() {
        query_ids.clone()
    } else {
        let mut v: Vec<u32> = doc_chunks.iter().flatten().map(|&t| t as u32).collect();
        v.extend_from_slice(&query_ids);
        v
    };

    // Prompt token count = length of the full assembled context (already
    // includes any FIM special tokens + prepended documents). Cheap and exact.
    let prompt_tokens = prompt_ids.len() as u32;
    // E2-ingress: clamp the generation budget to the context window (OpenAI
    // semantics) now that the prompt length + engine max_ctx are both known.
    if let Some((code, msg)) =
        crate::clamp_max_tokens_to_ctx(&mut sampling, prompt_ids.len(), shared_cpu.max_ctx())
    {
        return (code, msg).into_response();
    }
    // Now wait on the gate; tokenization is already done.
    let permit = match pregate_permit.acquire_gate().await {
        Ok(h) => h,
        Err(be) => return be.into_response(),
    };
    // After the gate, use the per-request fork for `generate_*` and
    // `last_request_stats` — those live on the fork's state.
    let cpu = match permit.cpu_engine.as_ref() {
        Some(e) => e.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "completions endpoint requires a real (non-mock) engine",
            )
                .into_response();
        }
    };
    let include_usage = req
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);

    // For streaming, we still want to return as a stream — drive on a
    // blocking task that emits tokens via mpsc, build SSE around it.
    if req.stream {
        let id = request_id();
        let cancel_guard = state.register_cancel(&id);
        let fingerprint = crate::system_fingerprint(&state.version, &model, kv_dtype_label(&cpu));
        return completions_stream(
            cpu,
            prompt_ids,
            sampling,
            model,
            req.echo,
            req.prompt,
            prompt_tokens,
            include_usage,
            req.logprobs,
            permit,
            cancel_guard,
            id,
            fingerprint,
        )
        .await;
    }
    // Non-streaming path: hold the permit for the duration of generation.
    let _permit = permit;

    // P12b — beam search path. `num_beams > 1` runs the deterministic beam
    // driver (its own decode mode — ignores the stochastic sampler and the
    // n/best_of sample-ranking path) and returns the top `n` hypotheses ranked
    // by length-penalty-adjusted cumulative log-prob. Each returned choice's
    // index follows the ranking (0 = best beam).
    if num_beams > 1 {
        let beam_params = rustllama_engine::BeamParams {
            num_beams: num_beams as usize,
            length_penalty: req.length_penalty.unwrap_or(1.0),
            early_stopping: req.early_stopping.unwrap_or(false),
            num_return: n_choices as usize,
        };
        let prompt_ids_b = prompt_ids.clone();
        let max_new = sampling.max_tokens;
        let cpu_b = cpu.clone();
        let gen = tokio::task::spawn_blocking(move || {
            generate_beam_completions(&cpu_b, &prompt_ids_b, max_new, &beam_params)
        })
        .await;
        let beams = match gen {
            Ok(Ok(b)) => b,
            Ok(Err(e)) => return (StatusCode::BAD_REQUEST, e).into_response(),
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        };
        let total_completion_tokens: u32 = beams.iter().map(|b| b.completion_tokens).sum();
        let prompt_for_echo = if req.echo {
            req.prompt.clone()
        } else {
            String::new()
        };
        let choices: Vec<CompletionChoice> = beams
            .iter()
            .enumerate()
            .map(|(idx, b)| CompletionChoice {
                text: format!("{}{}", prompt_for_echo, b.text),
                index: idx as u32,
                logprobs: None,
                finish_reason: if b.completion_tokens >= sampling.max_tokens {
                    "length"
                } else {
                    "stop"
                },
            })
            .collect();
        let stats = cpu.last_request_stats();
        let fingerprint = crate::system_fingerprint(&state.version, &model, kv_dtype_label(&cpu));
        return Json(CompletionResponse {
            id: request_id(),
            object: "text_completion",
            created: unix_ts(),
            model,
            choices,
            usage: Some(Usage::new(prompt_tokens, total_completion_tokens).with_stats(&stats)),
            system_fingerprint: fingerprint,
        })
        .into_response();
    }

    // P12a — multi-completion (`n` / `best_of`) path. Generates
    // `n_candidates` independent samples that share the prompt prefix (the
    // engine's prefix cache makes candidates 1.. an LCP no-op), each seeded
    // from `seed + i` so candidate 0 reproduces the single-completion result
    // and the set is deterministic by seed. When `best_of > n` the
    // candidates are ranked by length-normalized cumulative logprob and the
    // top `n` are returned; otherwise generation order is kept. Runs on the
    // classic sampler (no spec/logprobs) — consistent with the non-streaming
    // completions path, which already doesn't stop-truncate or spec on the
    // logprobs arm.
    if n_candidates > 1 {
        let prompt_ids_m = prompt_ids.clone();
        let sampling_m = sampling.clone();
        let cpu_m = cpu.clone();
        let gen = tokio::task::spawn_blocking(move || {
            generate_ranked_candidates(&cpu_m, &prompt_ids_m, &sampling_m, n_candidates)
        })
        .await;
        // (`&cpu_m` deref-coerces `&Arc<ContiguousEngine>` → `&ContiguousEngine`.)
        let mut cands = match gen {
            Ok(Ok(c)) => c,
            Ok(Err(e)) => return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response(),
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        };
        // Total generated tokens across ALL candidates — OpenAI bills every
        // best_of completion, not just the returned ones.
        let total_completion_tokens: u32 = cands.iter().map(|c| c.completion_tokens).sum();
        // best_of culling: rank by score (descending) and keep the top n.
        if best_of > n_choices {
            cands.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        cands.truncate(n_choices as usize);
        let prompt_for_echo = if req.echo {
            req.prompt.clone()
        } else {
            String::new()
        };
        let choices: Vec<CompletionChoice> = cands
            .iter()
            .enumerate()
            .map(|(idx, c)| CompletionChoice {
                text: format!("{}{}", prompt_for_echo, c.text),
                index: idx as u32,
                logprobs: None,
                finish_reason: if c.completion_tokens >= sampling.max_tokens {
                    "length"
                } else {
                    "stop"
                },
            })
            .collect();
        let stats = cpu.last_request_stats();
        let fingerprint = crate::system_fingerprint(&state.version, &model, kv_dtype_label(&cpu));
        return Json(CompletionResponse {
            id: request_id(),
            object: "text_completion",
            created: unix_ts(),
            model,
            choices,
            usage: Some(Usage::new(prompt_tokens, total_completion_tokens).with_stats(&stats)),
            system_fingerprint: fingerprint,
        })
        .into_response();
    }

    // Non-streaming: run generation, return JSON.
    let cpu_for_blocking = cpu.clone();
    let prompt_for_echo = if req.echo {
        req.prompt.clone()
    } else {
        String::new()
    };
    let want_logprobs = req.logprobs;
    // P15 (echo + logprobs): when BOTH are set, the logprobs arm prepends the
    // prompt tokens + their logprobs to the completion's, and rebuilds the echo
    // prefix from the DETOKENIZED prompt tokens so `text_offset` lines up with
    // the returned `text`. It stores that prefix here; the final assembly uses
    // it instead of the raw `prompt_for_echo`. `None` keeps today's behavior.
    let mut prompt_echo_override: Option<String> = None;

    // Speculative decoding for completions, routed at the ID level so this
    // endpoint's own tokenization (no BOS; FIM specials for `suffix`
    // requests) is preserved — re-encoding text would change the prompt.
    // P10a: draft-model spec now runs here too (via the ID-level
    // `speculate_draft_stream_from_ids`), with the same precedence as
    // `Engine::generate`: draft > MTP > n-gram. A grammar or a logprobs
    // request disables spec (the verify runs on raw softmax, yields no
    // per-token logprobs) — encoded in `pick_completion_spec_mode`.
    let spec_ngram = cpu.ngram_speculative();
    let spec_draft = cpu.draft_speculative();
    let spec_mode = pick_completion_spec_mode(
        sampling.grammar.is_some(),
        want_logprobs.is_some(),
        spec_draft.is_some(),
        cpu.mtp_speculative(),
        spec_ngram.is_some(),
    );

    let (text, logprobs_out, completion_tokens) = if !doc_chunks.is_empty() {
        // CacheBlend arm (RAG `documents` present): reuse each chunk's isolated
        // KV and blend it into the context, recomputing only the high-deviation
        // tokens. `generate_text_cacheblend` self-falls-back to a normal prefill
        // over the assembled prompt when CacheBlend is disabled or inapplicable,
        // so this arm is always correct. Classic decode only (no logprobs/spec
        // on the CacheBlend path in v1).
        let chunks = doc_chunks.clone();
        let suffix_i32: Vec<i32> = query_ids.iter().map(|&v| v as i32).collect();
        let sampling_inner = sampling.clone();
        let cpu_inner = cpu_for_blocking.clone();
        let res = tokio::task::spawn_blocking(move || {
            cpu_inner.generate_text_cacheblend(&chunks, &suffix_i32, &sampling_inner)
        })
        .await;
        let text = match res {
            Ok(Ok(t)) => t,
            Ok(Err(e)) => {
                return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
            }
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        };
        let completion_tokens = cpu
            .tokenizer()
            .and_then(|t| t.encode(&text, false).ok())
            .map(|ids| ids.len() as u32)
            .unwrap_or(0);
        (text, None, completion_tokens)
    } else if let Some(k) = want_logprobs {
        // Logprobs path: drives `generate_token_ids_with_logprobs`, then
        // detokenizes each token id individually so we can build the
        // OpenAI shape's `tokens` / `token_logprobs` / `top_logprobs`
        // arrays in lockstep.
        let prompt_ids_i32: Vec<i32> = prompt_ids.iter().map(|&v| v as i32).collect();
        let sampling_inner = sampling.clone();
        let cpu_inner = cpu_for_blocking.clone();
        let res = tokio::task::spawn_blocking(move || {
            cpu_inner.generate_token_ids_with_logprobs(
                &prompt_ids_i32,
                sampling_inner.max_tokens,
                &sampling_inner,
                k as usize,
            )
        })
        .await;
        let pairs = match res {
            Ok(Ok(p)) => p,
            Ok(Err(e)) => {
                return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
            }
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        };

        let tokenizer = match cpu.tokenizer() {
            Some(t) => t,
            None => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "logprobs path requires a tokenizer",
                )
                    .into_response();
            }
        };

        // Detokenize one TopLogprob set into the OpenAI `{token: logprob}` map.
        let build_top_map = |top: &[rustllama_engine::TopLogprob]| {
            let mut m: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
            for alt in top {
                let alt_str = tokenizer.decode_single(alt.id, true).unwrap_or_default();
                m.insert(alt_str, serde_json::json!(alt.logprob));
            }
            m
        };

        let mut tokens_out: Vec<String> = Vec::new();
        let mut token_logprobs: Vec<Option<f32>> = Vec::new();
        let mut top_logprobs: Vec<Option<serde_json::Map<String, serde_json::Value>>> = Vec::new();
        let mut text_offset: Vec<usize> = Vec::new();
        // `text_acc` accumulates the FULL returned `text` so byte offsets are
        // exact. On echo it opens with the detokenized prompt; otherwise it is
        // the completion alone (byte-identical to the pre-P15 path).
        let mut text_acc = String::new();

        // P15 echo branch: prepend each prompt token + its logprob. The first
        // prompt token's logprob/top are `null` (no preceding context).
        if req.echo {
            let prompt_ids_i32: Vec<i32> = prompt_ids.iter().map(|&v| v as i32).collect();
            let cpu_p = cpu_for_blocking.clone();
            let pids = prompt_ids_i32.clone();
            let plps = match tokio::task::spawn_blocking(move || {
                cpu_p.prompt_token_logprobs(&pids, k as usize)
            })
            .await
            {
                Ok(Ok(v)) => v,
                Ok(Err(e)) => {
                    return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
                }
                Err(e) => {
                    return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
                }
            };
            for (pid, lp) in prompt_ids.iter().zip(plps.iter()) {
                let tok_str = tokenizer.decode_single(*pid, true).unwrap_or_default();
                text_offset.push(text_acc.len());
                text_acc.push_str(&tok_str);
                tokens_out.push(tok_str);
                match lp {
                    None => {
                        token_logprobs.push(None);
                        top_logprobs.push(None);
                    }
                    Some(lp) => {
                        token_logprobs.push(Some(lp.logprob));
                        top_logprobs.push(Some(build_top_map(&lp.top)));
                    }
                }
            }
            // The echo prefix IS the detokenized prompt we just laid down, so
            // the final assembly must not prepend `prompt_for_echo` again.
            prompt_echo_override = Some(text_acc.clone());
        }

        let mut tok_count = 0u32;
        for (id, lp) in pairs.iter() {
            let tok_str = tokenizer.decode_single(*id, true).unwrap_or_default();
            text_offset.push(text_acc.len());
            text_acc.push_str(&tok_str);
            tokens_out.push(tok_str);
            token_logprobs.push(Some(lp.logprob));
            top_logprobs.push(Some(build_top_map(&lp.top)));
            tok_count += 1;
        }

        let logprobs_payload = LogprobsOut {
            tokens: tokens_out,
            token_logprobs,
            top_logprobs,
            text_offset,
        };
        // Return the COMPLETION-ONLY text; the prompt prefix (if any) is carried
        // by `prompt_echo_override` and prepended at final assembly so the
        // non-echo path stays byte-identical.
        let completion_text: String = pairs
            .iter()
            .map(|(id, _)| tokenizer.decode_single(*id, true).unwrap_or_default())
            .collect();
        (completion_text, Some(logprobs_payload), tok_count)
    } else {
        match spec_mode {
            CompletionSpecMode::Draft => {
                // P10a: draft-model speculative decoding at the ID level, so
                // the endpoint's exact tokenization (no BOS; FIM specials) is
                // what the target verifies. Takes precedence over MTP/n-gram.
                let (draft, k) = spec_draft.expect("spec_mode==Draft implies spec_draft");
                let prompt_ids_i32: Vec<i32> = prompt_ids.iter().map(|&v| v as i32).collect();
                let mut st = match cpu.speculate_draft_stream_from_ids(
                    prompt_ids_i32,
                    draft,
                    k,
                    sampling.clone(),
                ) {
                    Ok(st) => st,
                    Err(e) => {
                        return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
                    }
                };
                use futures::StreamExt;
                let mut text = String::new();
                while let Some(item) = st.next().await {
                    match item {
                        Ok(tok) => text.push_str(&tok.text),
                        Err(e) => {
                            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
                                .into_response()
                        }
                    }
                }
                let completion_tokens = cpu
                    .tokenizer()
                    .and_then(|t| t.encode(&text, false).ok())
                    .map(|ids| ids.len() as u32)
                    .unwrap_or(0);
                (text, None, completion_tokens)
            }
            CompletionSpecMode::Mtp => {
                // MTP / NextN self-speculative path — same draining shape as
                // the n-gram arm, but drafts come from the model's NextN head.
                let prompt_ids_i32: Vec<i32> = prompt_ids.iter().map(|&v| v as i32).collect();
                let mut st =
                    match cpu.speculate_mtp_stream_from_ids(prompt_ids_i32, sampling.clone(), None)
                    {
                        Ok(st) => st,
                        Err(e) => {
                            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
                                .into_response()
                        }
                    };
                use futures::StreamExt;
                let mut text = String::new();
                while let Some(item) = st.next().await {
                    match item {
                        Ok(tok) => text.push_str(&tok.text),
                        Err(e) => {
                            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
                                .into_response()
                        }
                    }
                }
                let completion_tokens = cpu
                    .tokenizer()
                    .and_then(|t| t.encode(&text, false).ok())
                    .map(|ids| ids.len() as u32)
                    .unwrap_or(0);
                (text, None, completion_tokens)
            }
            CompletionSpecMode::Ngram => {
                // Speculative path: drain the ID-level TokenStream here in
                // the async handler — its verify rounds run on
                // `spawn_blocking` internally, so this does not block the
                // runtime. Stop strings are honored inside the stream
                // (engine-side), same as the classic `generate_*` methods.
                let ng = spec_ngram.expect("spec_mode==Ngram implies spec_ngram");
                let prompt_ids_i32: Vec<i32> = prompt_ids.iter().map(|&v| v as i32).collect();
                let mut st =
                    match cpu.speculate_ngram_stream_from_ids(prompt_ids_i32, ng, sampling.clone())
                    {
                        Ok(st) => st,
                        Err(e) => {
                            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
                                .into_response()
                        }
                    };
                use futures::StreamExt;
                let mut text = String::new();
                while let Some(item) = st.next().await {
                    match item {
                        Ok(tok) => text.push_str(&tok.text),
                        Err(e) => {
                            return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
                                .into_response()
                        }
                    }
                }
                let completion_tokens = cpu
                    .tokenizer()
                    .and_then(|t| t.encode(&text, false).ok())
                    .map(|ids| ids.len() as u32)
                    .unwrap_or(0);
                (text, None, completion_tokens)
            }
            CompletionSpecMode::None => {
                // Clone the sampling config before moving it into the blocking
                // closure — the outer fn still needs `sampling.max_tokens` for
                // the finish_reason decision below.
                let sampling_inner = sampling.clone();
                let result = tokio::task::spawn_blocking(move || {
                    cpu_for_blocking.generate_text_from_ids(&prompt_ids, &sampling_inner)
                })
                .await;
                let text = match result {
                    Ok(Ok(t)) => t,
                    Ok(Err(e)) => {
                        return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
                    }
                    Err(e) => {
                        return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
                    }
                };

                // Completion tokens: re-tokenize the generated text.
                // Approximate but matches what OpenAI reports.
                let completion_tokens = cpu
                    .tokenizer()
                    .and_then(|t| t.encode(&text, false).ok())
                    .map(|ids| ids.len() as u32)
                    .unwrap_or(0);
                (text, None, completion_tokens)
            }
        }
    };

    // P15: on echo+logprobs the prompt prefix is the DETOKENIZED prompt
    // (`prompt_echo_override`), so `text_offset` aligns with the returned text;
    // every other path uses the raw `prompt_for_echo` (empty when `echo` is off
    // — byte-identical to before).
    let echo_prefix = prompt_echo_override.unwrap_or(prompt_for_echo);
    let full_text = format!("{}{}", echo_prefix, text);
    let stats = cpu.last_request_stats();
    let fingerprint = crate::system_fingerprint(&state.version, &model, kv_dtype_label(&cpu));
    Json(CompletionResponse {
        id: request_id(),
        object: "text_completion",
        created: unix_ts(),
        model,
        choices: vec![CompletionChoice {
            text: full_text,
            index: 0,
            logprobs: logprobs_out,
            // OpenAI legacy-completions spec: `"length"` when the
            // response was truncated by `max_tokens`. Was hardcoded
            // `"stop"` until this audit pass.
            finish_reason: if completion_tokens >= sampling.max_tokens {
                "length"
            } else {
                "stop"
            },
        }],
        usage: Some(Usage::new(prompt_tokens, completion_tokens).with_stats(&stats)),
        system_fingerprint: fingerprint,
    })
    .into_response()
}

fn build_prompt_ids(
    tokenizer: &rustllama_tokenizer::Tokenizer,
    prompt: &str,
    suffix: Option<&str>,
) -> Result<Vec<u32>, String> {
    match suffix {
        None => tokenizer.encode(prompt, false).map_err(|e| e.to_string()),
        Some(suffix_text) => {
            let fim = tokenizer.fim_tokens().ok_or_else(|| {
                "model tokenizer has no Fill-In-Middle special tokens; cannot honor `suffix`. \
                 Use a coding-specific model (Qwen2.5-Coder, DeepSeek-Coder, StarCoder, ...)"
                    .to_string()
            })?;
            let prefix_ids = tokenizer.encode(prompt, false).map_err(|e| e.to_string())?;
            let suffix_ids = tokenizer
                .encode(suffix_text, false)
                .map_err(|e| e.to_string())?;
            let mut ids = Vec::with_capacity(prefix_ids.len() + suffix_ids.len() + 3);
            ids.push(fim.prefix);
            ids.extend_from_slice(&prefix_ids);
            ids.push(fim.suffix);
            ids.extend_from_slice(&suffix_ids);
            ids.push(fim.middle);
            Ok(ids)
        }
    }
}

/// The generic-default sampling builder (base = [`SamplingParams::default`]).
/// Production calls [`build_sampling_with_base`] with the model's recommended
/// profile; this wrapper is retained for the sampling / seed-determinism tests.
#[cfg(test)]
fn build_sampling(req: &CompletionRequest) -> SamplingParams {
    build_sampling_with_base(req, SamplingParams::default())
}

/// Like [`build_sampling`] but starts from a per-MODEL `base` profile (the
/// loaded model's [`rustllama_engine::Engine::recommended_sampling`], or the
/// generic [`SamplingParams::default`]) instead of the bare default. Request
/// fields overlay on top, so an explicit request field always wins. See the
/// `chat` twin for the rationale.
fn build_sampling_with_base(req: &CompletionRequest, base: SamplingParams) -> SamplingParams {
    let mut s = base;
    if let Some(t) = req.temperature {
        s.temperature = t;
    }
    if let Some(p) = req.top_p {
        s.top_p = p;
    }
    if let Some(k) = req.top_k {
        s.top_k = k;
    }
    if let Some(tp) = req.typical_p {
        s.typical_p = tp;
    }
    if let Some(m) = req.max_tokens {
        s.max_tokens = m;
    }
    if let Some(rp) = req.repeat_penalty {
        s.repeat_penalty = rp;
    }
    if let Some(fp) = req.frequency_penalty {
        s.frequency_penalty = fp;
    }
    if let Some(pp) = req.presence_penalty {
        s.presence_penalty = pp;
    }
    s.stop = req.stop.clone();
    if let Some(seed) = req.seed {
        s.seed = seed;
    }
    if let Some(m) = req.mirostat {
        s.mirostat = m;
    }
    if let Some(t) = req.mirostat_tau {
        s.mirostat_tau = t;
    }
    if let Some(e) = req.mirostat_eta {
        s.mirostat_eta = e;
    }
    if let Some(mp) = req.min_p {
        s.min_p = mp;
    }
    if let Some(lb) = req.logit_bias.as_ref() {
        s.logit_bias = crate::chat::parse_openai_logit_bias(lb);
    }
    if let Some(dm) = req.dry_multiplier {
        s.dry_multiplier = dm;
    }
    if let Some(db) = req.dry_base {
        s.dry_base = db;
    }
    if let Some(dal) = req.dry_allowed_length {
        s.dry_allowed_length = dal;
    }
    if let Some(dsb) = req.dry_sequence_breakers.as_ref() {
        s.dry_sequence_breakers = dsb.clone();
    }
    if let Some(xp) = req.xtc_probability {
        s.xtc_probability = xp;
    }
    if let Some(xt) = req.xtc_threshold {
        s.xtc_threshold = xt;
    }
    if let Some(dr) = req.dynatemp_range {
        s.dynatemp_range = dr;
    }
    if let Some(de) = req.dynatemp_exponent {
        s.dynatemp_exponent = de;
    }
    // P15: a non-empty `grammar` field engages the GBNF constraint. The grammar
    // text was already validated (clean 400) at the request boundary; here it is
    // compiled by the engine when the sampler builds its mask.
    if let Some(g) = req.grammar.as_ref().filter(|g| !g.trim().is_empty()) {
        s.grammar = Some(rustllama_engine::GrammarKind::Gbnf { grammar: g.clone() });
    }
    // P11: explicit `lora` field selects a resident adapter for this request.
    // The `model: "<base>:<adapter>"` convention is layered on top by the
    // handler (which sets this when the field is absent). Empty ⇒ base.
    if let Some(a) = req.lora.as_ref().filter(|a| !a.trim().is_empty()) {
        s.lora_adapter = Some(a.clone());
    }
    s
}

/// Which speculative-decode driver the non-streaming `/v1/completions`
/// handler routes a request through. P10a adds `Draft` (draft-model spec,
/// previously chat-only) at the top of the precedence order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompletionSpecMode {
    /// Classic single-token decode (`generate_text_from_ids`).
    None,
    /// Draft-model speculation (`speculate_draft_stream_from_ids`).
    Draft,
    /// MTP / NextN self-speculation.
    Mtp,
    /// N-gram (prompt-lookup) speculation.
    Ngram,
}

/// Resolve the completions speculative-decode mode. A grammar-constrained OR
/// a logprobs request disables spec entirely (the speculative verify runs on
/// raw softmax and yields no per-token logprobs, so those take the classic
/// arms). Otherwise precedence is draft > MTP > n-gram, matching
/// `Engine::generate` on the chat path. Pure so the routing policy is unit
/// testable without a live engine.
pub(crate) fn pick_completion_spec_mode(
    grammar: bool,
    want_logprobs: bool,
    has_draft: bool,
    has_mtp: bool,
    has_ngram: bool,
) -> CompletionSpecMode {
    if grammar || want_logprobs {
        return CompletionSpecMode::None;
    }
    if has_draft {
        CompletionSpecMode::Draft
    } else if has_mtp {
        CompletionSpecMode::Mtp
    } else if has_ngram {
        CompletionSpecMode::Ngram
    } else {
        CompletionSpecMode::None
    }
}

/// One generated completion in the P12a `n` / `best_of` path, with the
/// length-normalized cumulative chosen-token logprob used for `best_of`
/// ranking.
struct RankedCandidate {
    text: String,
    /// Mean chosen-token logprob (cumulative logprob / tokens). Higher is
    /// "better" per OpenAI `best_of`. `NEG_INFINITY` for an empty completion
    /// so it never wins a ranking.
    score: f32,
    completion_tokens: u32,
}

/// Generate `g` independent completions sharing the prompt prefix. Candidate
/// `i` samples from `seed + i`, so candidate 0 reproduces the single
/// completion byte-for-byte and the whole set is deterministic by seed. The
/// engine's prefix cache makes candidates `1..` an LCP no-op on the prompt.
/// Runs on the classic logprobs sampler (no spec) so every candidate carries
/// its cumulative chosen-token logprob for `best_of` ranking.
fn generate_ranked_candidates(
    cpu: &rustllama_engine::ContiguousEngine,
    prompt_ids: &[u32],
    sampling: &SamplingParams,
    g: u32,
) -> Result<Vec<RankedCandidate>, String> {
    let tokenizer = cpu
        .tokenizer()
        .ok_or_else(|| "n/best_of path requires a tokenizer".to_string())?;
    let prompt_i32: Vec<i32> = prompt_ids.iter().map(|&v| v as i32).collect();
    let base_seed = sampling.seed;
    let mut out = Vec::with_capacity(g as usize);
    for i in 0..g {
        let mut s_i = sampling.clone();
        s_i.seed = base_seed.wrapping_add(i as u64);
        let pairs = cpu
            .generate_token_ids_with_logprobs(&prompt_i32, s_i.max_tokens, &s_i, 0)
            .map_err(|e| e.to_string())?;
        let ids: Vec<u32> = pairs.iter().map(|(id, _)| *id).collect();
        let text = tokenizer.decode(&ids, true).map_err(|e| e.to_string())?;
        let cum: f32 = pairs.iter().map(|(_, lp)| lp.logprob).sum();
        let n_tok = pairs.len() as u32;
        // Length-normalized (per-token mean) logprob — OpenAI's best_of
        // criterion. Empty completions score -inf so they never win.
        let score = if n_tok > 0 {
            cum / n_tok as f32
        } else {
            f32::NEG_INFINITY
        };
        out.push(RankedCandidate {
            text,
            score,
            completion_tokens: n_tok,
        });
    }
    Ok(out)
}

/// One returned beam-search hypothesis: its decoded text and token count.
struct BeamCompletion {
    text: String,
    completion_tokens: u32,
}

/// Run deterministic beam search over `prompt_ids` and decode each returned
/// hypothesis to text. Beams come back already ranked (best first) by the
/// length-penalty-adjusted score. Maps the engine's "beam unsupported for this
/// backend/model" error to a user-facing string the caller returns as a 400.
fn generate_beam_completions(
    cpu: &rustllama_engine::ContiguousEngine,
    prompt_ids: &[u32],
    max_new: u32,
    params: &rustllama_engine::BeamParams,
) -> Result<Vec<BeamCompletion>, String> {
    let tokenizer = cpu
        .tokenizer()
        .ok_or_else(|| "beam search requires a tokenizer".to_string())?;
    let prompt_i32: Vec<i32> = prompt_ids.iter().map(|&v| v as i32).collect();
    let beams = cpu
        .generate_beam_token_ids(&prompt_i32, max_new, params)
        .map_err(|e| e.to_string())?;
    let mut out = Vec::with_capacity(beams.len());
    for b in &beams {
        let text = tokenizer
            .decode(&b.tokens, true)
            .map_err(|e| e.to_string())?;
        out.push(BeamCompletion {
            text,
            completion_tokens: b.tokens.len() as u32,
        });
    }
    Ok(out)
}

fn unix_ts() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn request_id() -> String {
    // E1: process-unique (ns + counter). See `crate::new_request_id`.
    crate::new_request_id("cmpl")
}

/// Short stable label for the engine's active KV dtype — mirror of
/// `chat::kv_dtype_label`, kept local so completions doesn't depend
/// on `chat`'s private helpers. Used as one of the inputs to
/// `system_fingerprint`.
fn kv_dtype_label(cpu: &rustllama_engine::ContiguousEngine) -> &'static str {
    match cpu.kv_dtype() {
        rustllama_engine::KvDtype::F32 => "f32",
        rustllama_engine::KvDtype::F16 => "f16",
        rustllama_engine::KvDtype::Bf16 => "bf16",
        rustllama_engine::KvDtype::Q8_0 => "q8_0",
        rustllama_engine::KvDtype::Tq(_) => "tq",
        rustllama_engine::KvDtype::Nvfp4 => "nvfp4",
        rustllama_engine::KvDtype::Q4_0 => "q4_0",
        rustllama_engine::KvDtype::Mxfp4 => "mxfp4",
        rustllama_engine::KvDtype::Mxfp6 => "mxfp6",
        rustllama_engine::KvDtype::Mxfp8 => "mxfp8",
    }
}

#[allow(clippy::too_many_arguments)]
async fn completions_stream(
    cpu: std::sync::Arc<rustllama_engine::ContiguousEngine>,
    prompt_ids: Vec<u32>,
    sampling: SamplingParams,
    model: String,
    echo: bool,
    prompt_text: String,
    prompt_tokens: u32,
    include_usage: bool,
    logprobs_k: Option<u32>,
    permit: crate::PermitGuard,
    cancel_guard: crate::CancelGuard,
    id: String,
    fingerprint: String,
) -> Response {
    use axum::response::sse::{Event, KeepAlive, Sse};
    use std::convert::Infallible;
    use tokio::sync::mpsc;

    let created = unix_ts();
    let cancel_flag = cancel_guard.flag.clone();
    let id_for_header = id.clone();

    // Drive on a blocking task; stream tokens through mpsc.
    // Buffer is intentionally small so the engine stays in lockstep
    // with the SSE consumer — see ContiguousEngine::generate_token_ids_streaming.
    let (tx, mut rx) = mpsc::channel::<Result<Token, String>>(4);
    let cpu_inner = cpu.clone();
    let sampling_inner = sampling.clone();
    let logprobs_k_inner = logprobs_k.map(|k| k as usize);
    // N-gram speculation, mirroring the non-streaming handler's
    // eligibility gate: grammar-free, no logprobs (the speculative
    // path yields none). ID-level so the endpoint's tokenization
    // (no BOS; FIM specials) is preserved. Stop strings are honored
    // inside the engine stream.
    let spec_ngram = if sampling.grammar.is_none() && logprobs_k.is_none() {
        cpu.ngram_speculative()
    } else {
        None
    };
    // MTP / NextN self-speculation, precedence over n-gram; same gate.
    // `mtp_speculative()` is `false` on models without a NextN head.
    let spec_mtp = sampling.grammar.is_none() && logprobs_k.is_none() && cpu.mtp_speculative();
    // P10a: draft-model spec, highest precedence (same gate: grammar-free,
    // no logprobs). ID-level so the endpoint's tokenization is preserved.
    let spec_draft = if sampling.grammar.is_none() && logprobs_k.is_none() {
        cpu.draft_speculative()
    } else {
        None
    };
    if let Some((draft, k)) = spec_draft {
        // Draft-model self-speculative stream — same forwarding shape as the
        // MTP / n-gram arms; drafts come from the paired draft model.
        tokio::spawn(async move {
            use futures::StreamExt;
            let prompt_i32: Vec<i32> = prompt_ids.iter().map(|&v| v as i32).collect();
            match cpu_inner.speculate_draft_stream_from_ids(prompt_i32, draft, k, sampling_inner) {
                Ok(mut st) => {
                    while let Some(item) = st.next().await {
                        let mapped = item.map_err(|e| e.to_string());
                        if tx.send(mapped).await.is_err() {
                            break;
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e.to_string())).await;
                }
            }
        });
    } else if spec_mtp {
        // MTP self-speculative stream — same forwarding shape as the
        // n-gram arm below; drafts come from the model's NextN head.
        tokio::spawn(async move {
            use futures::StreamExt;
            let prompt_i32: Vec<i32> = prompt_ids.iter().map(|&v| v as i32).collect();
            match cpu_inner.speculate_mtp_stream_from_ids(prompt_i32, sampling_inner, None) {
                Ok(mut st) => {
                    while let Some(item) = st.next().await {
                        let mapped = item.map_err(|e| e.to_string());
                        if tx.send(mapped).await.is_err() {
                            break;
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e.to_string())).await;
                }
            }
        });
    } else if let Some(ng) = spec_ngram {
        // The speculative TokenStream is async (verify rounds run on
        // spawn_blocking internally) — forward it into the same mpsc
        // the SSE loop drains, from a plain tokio task.
        tokio::spawn(async move {
            use futures::StreamExt;
            let prompt_i32: Vec<i32> = prompt_ids.iter().map(|&v| v as i32).collect();
            match cpu_inner.speculate_ngram_stream_from_ids(prompt_i32, ng, sampling_inner) {
                Ok(mut st) => {
                    while let Some(item) = st.next().await {
                        let mapped = item.map_err(|e| e.to_string());
                        if tx.send(mapped).await.is_err() {
                            // SSE consumer dropped (cancel/disconnect):
                            // dropping `st` ends generation at the next
                            // round boundary.
                            break;
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(e.to_string())).await;
                }
            }
        });
    } else {
        tokio::task::spawn_blocking(move || {
            let prompt_i32: Vec<i32> = prompt_ids.iter().map(|&v| v as i32).collect();
            // Single true-streaming path: tokens flow to the SSE consumer
            // as the engine generates them. `logprobs_k_inner = None` skips
            // the per-token raw-logits clone (zero-overhead fast path).
            if let Err(e) = cpu_inner.generate_token_ids_streaming(
                &prompt_i32,
                &sampling_inner,
                logprobs_k_inner,
                &tx,
            ) {
                let _ = tx.blocking_send(Err(e.to_string()));
            }
        });
    }

    let echo_text = if echo { prompt_text } else { String::new() };
    let cpu_for_stream = cpu.clone();
    // Capture for the max_tokens-hit termination-reason check below.
    let max_tokens = sampling.max_tokens;
    let event_stream = async_stream::stream! {
        let mut sent_echo = false;
        let mut completion_tokens = 0u32;
        let mut text_cursor = 0usize; // running byte offset into emitted text
        let mut finish_reason: &'static str = "stop";
        // H7: reusable serialization buffer across stream chunks.
        // `serde_json::to_writer` writes into this `Vec<u8>` instead
        // of allocating a fresh String per chunk; we drain into a
        // String for `Event::data()` and `.clear()` for next iter.
        // Pre-sized to a typical chunk (~512 B) so the first chunk
        // doesn't trigger growth either.
        let mut json_buf: Vec<u8> = Vec::with_capacity(512);
        loop {
            if cancel_flag.load(std::sync::atomic::Ordering::Acquire) {
                finish_reason = "cancelled";
                break;
            }
            let item = rx.recv().await;
            let (token_text, token_lp) = match item {
                Some(Ok(t)) => {
                    completion_tokens += 1;
                    (t.text, t.logprobs)
                },
                Some(Err(e)) => {
                    let err = serde_json::json!({ "error": { "message": e } });
                    yield Ok::<_, Infallible>(Event::default().data(err.to_string()));
                    finish_reason = "error";
                    break;
                }
                None => break,
            };

            let mut payload_text = token_text.clone();
            if !sent_echo && !echo_text.is_empty() {
                payload_text = format!("{echo_text}{payload_text}");
                sent_echo = true;
            }

            // Build the per-chunk logprobs payload when requested.
            // Shape: { tokens:[t], token_logprobs:[lp], top_logprobs:[{...}],
            //          text_offset:[off] } — arrays of length 1 per chunk.
            let logprobs_value: serde_json::Value = match (&token_lp, cpu_for_stream.tokenizer()) {
                (Some(lp), Some(tokenizer)) => {
                    let mut top_map = serde_json::Map::new();
                    for alt in &lp.top {
                        let alt_str = tokenizer.decode_single(alt.id, true).unwrap_or_default();
                        top_map.insert(alt_str, serde_json::json!(alt.logprob));
                    }
                    let chunk_offset = text_cursor;
                    text_cursor += token_text.len();
                    serde_json::json!({
                        "tokens": [token_text.clone()],
                        "token_logprobs": [lp.logprob],
                        "top_logprobs": [top_map],
                        "text_offset": [chunk_offset],
                    })
                }
                _ => serde_json::Value::Null,
            };

            let chunk = serde_json::json!({
                "id": &id,
                "object": "text_completion",
                "created": created,
                "model": &model,
                "system_fingerprint": &fingerprint,
                "choices": [{
                    "text": payload_text,
                    "index": 0,
                    "logprobs": logprobs_value,
                    "finish_reason": serde_json::Value::Null,
                }]
            });
            // H7: write into the reused buffer, then move to a fresh
            // String owned by the Event (axum's API takes ownership).
            // Net result: only the Event's owned String is freshly
            // allocated per chunk; the intermediate serialization
            // buffer is recycled. Cuts ~200-500 B of per-chunk
            // allocator churn vs the prior `chunk.to_string()` path.
            json_buf.clear();
            serde_json::to_writer(&mut json_buf, &chunk).expect("serialize chunk");
            let chunk_str = String::from_utf8(json_buf.split_off(0))
                .expect("serde_json emits valid UTF-8");
            yield Ok(Event::default().data(chunk_str));
        }
        // Final chunk + DONE. Apply the length-cap label when the
        // stream ended on a natural "stop" but `completion_tokens`
        // reached `max_tokens` — same precedence rule as the chat
        // path: cancelled / error take priority.
        if finish_reason == "stop" && completion_tokens >= max_tokens {
            finish_reason = "length";
        }
        let chunk = serde_json::json!({
            "id": &id,
            "object": "text_completion",
            "created": created,
            "model": &model,
            "system_fingerprint": &fingerprint,
            "choices": [{
                "text": "",
                "index": 0,
                "logprobs": serde_json::Value::Null,
                "finish_reason": finish_reason,
            }]
        });
        yield Ok(Event::default().data(chunk.to_string()));
        if include_usage {
            let stats = cpu_for_stream.last_request_stats();
            let usage_chunk = serde_json::json!({
                "id": &id,
                "object": "text_completion",
                "created": created,
                "model": &model,
                "system_fingerprint": &fingerprint,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "prefill_ms": stats.prefill_ms,
                    "decode_ms": stats.decode_ms,
                    "tokens_prefilled": stats.tokens_prefilled,
                    "cache_hit_tokens": stats.cache_hit_tokens,
                }
            });
            yield Ok(Event::default().data(usage_chunk.to_string()));
        }
        yield Ok(Event::default().data("[DONE]"));
        drop(permit);
        drop(cancel_guard);
    };

    (
        [("x-rustllama-request-id", id_for_header.as_str())],
        Sse::new(event_stream).keep_alive(KeepAlive::default()),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::{
        build_prompt_ids, build_sampling, pick_completion_spec_mode, CompletionRequest,
        CompletionSpecMode,
    };
    use rustllama_gguf::synth::{write_synthetic_llama_gguf, SynthLlama};
    use rustllama_gguf::Gguf;
    use rustllama_tokenizer::Tokenizer;

    // ----- seed determinism pinning -----------------------------------------

    fn completion_request_with_seed(seed: Option<u64>) -> CompletionRequest {
        let mut value = serde_json::json!({
            "model": "rustllama-mock",
            "prompt": "hi",
            "max_tokens": 4,
        });
        if let Some(s) = seed {
            value["seed"] = serde_json::json!(s);
        }
        serde_json::from_value(value).expect("CompletionRequest parses")
    }

    #[test]
    fn seed_field_threads_into_sampling_params_for_completions() {
        let req = completion_request_with_seed(Some(99));
        let sampling = build_sampling(&req);
        assert_eq!(
            sampling.seed, 99,
            "OpenAI completions seed must reach SamplingParams"
        );
    }

    #[test]
    fn seed_absent_keeps_default_sampling_seed_for_completions() {
        let req = completion_request_with_seed(None);
        let default_seed = rustllama_engine::SamplingParams::default().seed;
        assert_eq!(build_sampling(&req).seed, default_seed);
    }

    #[test]
    fn typical_p_threads_into_sampling_params_for_completions() {
        let value = serde_json::json!({
            "prompt": "hi",
            "max_tokens": 4,
            "typical_p": 0.85,
        });
        let req: CompletionRequest =
            serde_json::from_value(value).expect("CompletionRequest parses");
        let sampling = build_sampling(&req);
        assert!((sampling.typical_p - 0.85).abs() < 1e-6);
    }

    #[test]
    fn mirostat_fields_thread_into_sampling_params_for_completions() {
        let value = serde_json::json!({
            "model": "rustllama-mock",
            "prompt": "hi",
            "max_tokens": 4,
            "mirostat": 1,
            "mirostat_tau": 3.0,
            "mirostat_eta": 0.05,
        });
        let req: CompletionRequest =
            serde_json::from_value(value).expect("CompletionRequest parses");
        let sampling = build_sampling(&req);
        assert_eq!(sampling.mirostat, 1);
        assert!((sampling.mirostat_tau - 3.0).abs() < 1e-6);
        assert!((sampling.mirostat_eta - 0.05).abs() < 1e-6);
    }

    #[test]
    fn p13_samplers_thread_into_sampling_params_for_completions() {
        let value = serde_json::json!({
            "prompt": "hi",
            "max_tokens": 4,
            "min_p": 0.03,
            "logit_bias": { "7": -100.0 },
            "dry_multiplier": 1.0,
            "dry_base": 1.5,
            "dry_allowed_length": 2,
            "dry_sequence_breakers": [5],
            "xtc_probability": 0.3,
            "xtc_threshold": 0.2,
            "dynatemp_range": 0.25,
            "dynatemp_exponent": 0.9,
        });
        let req: CompletionRequest =
            serde_json::from_value(value).expect("CompletionRequest parses");
        let s = build_sampling(&req);
        assert!((s.min_p - 0.03).abs() < 1e-6);
        assert_eq!(s.logit_bias, vec![(7, -100.0)]);
        assert!((s.dry_multiplier - 1.0).abs() < 1e-6);
        assert!((s.dry_base - 1.5).abs() < 1e-6);
        assert_eq!(s.dry_allowed_length, 2);
        assert_eq!(s.dry_sequence_breakers, vec![5]);
        assert!((s.xtc_probability - 0.3).abs() < 1e-6);
        assert!((s.xtc_threshold - 0.2).abs() < 1e-6);
        assert!((s.dynatemp_range - 0.25).abs() < 1e-6);
        assert!((s.dynatemp_exponent - 0.9).abs() < 1e-6);
    }

    fn synth_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("rustllama-fim-{tag}.gguf"))
    }

    fn fim_tokenizer(tag: &str) -> (Tokenizer, std::path::PathBuf) {
        let path = synth_path(tag);
        let params = SynthLlama {
            include_fim_tokens: true,
            ..SynthLlama::default()
        };
        write_synthetic_llama_gguf(&path, &params);
        let gguf = Gguf::open(&path).expect("open synth gguf");
        let tok = Tokenizer::from_gguf(&gguf).expect("load tokenizer");
        (tok, path)
    }

    #[test]
    fn build_prompt_ids_without_suffix_is_plain_encode() {
        let (tok, path) = fim_tokenizer("plain");
        let prompt = "<tok_5> <tok_6>";
        let got = build_prompt_ids(&tok, prompt, None).expect("build");
        let expected = tok.encode(prompt, false).expect("encode");
        assert_eq!(got, expected);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn build_prompt_ids_with_suffix_wraps_in_fim_markers() {
        let (tok, path) = fim_tokenizer("suffix");
        let prefix = "<tok_5>";
        let suffix = "<tok_6>";
        let got = build_prompt_ids(&tok, prefix, Some(suffix)).expect("build");

        let fim = tok.fim_tokens().expect("synth carries FIM markers");
        let prefix_ids = tok.encode(prefix, false).expect("enc prefix");
        let suffix_ids = tok.encode(suffix, false).expect("enc suffix");

        // Expected: [<|fim_prefix|>] + prefix_ids + [<|fim_suffix|>] + suffix_ids + [<|fim_middle|>]
        assert_eq!(got[0], fim.prefix);
        assert_eq!(&got[1..1 + prefix_ids.len()], prefix_ids.as_slice());
        let off = 1 + prefix_ids.len();
        assert_eq!(got[off], fim.suffix);
        assert_eq!(
            &got[off + 1..off + 1 + suffix_ids.len()],
            suffix_ids.as_slice()
        );
        assert_eq!(got[off + 1 + suffix_ids.len()], fim.middle);
        assert_eq!(got.len(), prefix_ids.len() + suffix_ids.len() + 3);
        let _ = std::fs::remove_file(&path);
    }

    // ----- P12a: n / best_of parsing --------------------------------------

    #[test]
    fn n_and_best_of_parse_from_request() {
        let value = serde_json::json!({
            "prompt": "hi",
            "max_tokens": 4,
            "n": 3,
            "best_of": 5,
        });
        let req: CompletionRequest =
            serde_json::from_value(value).expect("CompletionRequest parses");
        assert_eq!(req.n, Some(3));
        assert_eq!(req.best_of, Some(5));
    }

    #[test]
    fn n_and_best_of_absent_are_none() {
        let value = serde_json::json!({ "prompt": "hi", "max_tokens": 4 });
        let req: CompletionRequest =
            serde_json::from_value(value).expect("CompletionRequest parses");
        assert_eq!(req.n, None, "absent n must be None (handler defaults to 1)");
        assert_eq!(
            req.best_of, None,
            "absent best_of must be None (defaults to n)"
        );
    }

    #[test]
    fn beam_params_parse_from_request() {
        let value = serde_json::json!({
            "prompt": "hi",
            "max_tokens": 4,
            "num_beams": 4,
            "length_penalty": 0.8,
            "early_stopping": true,
            "n": 2,
        });
        let req: CompletionRequest =
            serde_json::from_value(value).expect("CompletionRequest parses");
        assert_eq!(req.num_beams, Some(4));
        assert_eq!(req.length_penalty, Some(0.8));
        assert_eq!(req.early_stopping, Some(true));
        assert_eq!(req.n, Some(2));
    }

    #[test]
    fn beam_params_absent_are_none() {
        let value = serde_json::json!({ "prompt": "hi", "max_tokens": 4 });
        let req: CompletionRequest =
            serde_json::from_value(value).expect("CompletionRequest parses");
        assert_eq!(
            req.num_beams, None,
            "absent num_beams must be None (defaults to 1)"
        );
        assert_eq!(req.length_penalty, None);
        assert_eq!(req.early_stopping, None);
    }

    // ----- P10a: completions speculative-mode routing ---------------------

    #[test]
    fn spec_mode_draft_takes_precedence() {
        // With a draft model present (and no grammar/logprobs), the draft
        // arm engages even when MTP / n-gram are also available.
        assert_eq!(
            pick_completion_spec_mode(false, false, true, true, true),
            CompletionSpecMode::Draft
        );
    }

    #[test]
    fn spec_mode_grammar_disables_spec() {
        // A grammar request must fall back to the classic sampler even with
        // a draft model attached (the verify can't honor a grammar mask).
        assert_eq!(
            pick_completion_spec_mode(true, false, true, true, true),
            CompletionSpecMode::None
        );
    }

    #[test]
    fn spec_mode_logprobs_disables_spec() {
        // A logprobs request takes the classic logprobs arm (spec yields no
        // per-token logprobs).
        assert_eq!(
            pick_completion_spec_mode(false, true, true, false, false),
            CompletionSpecMode::None
        );
    }

    #[test]
    fn spec_mode_mtp_then_ngram_then_none() {
        assert_eq!(
            pick_completion_spec_mode(false, false, false, true, true),
            CompletionSpecMode::Mtp
        );
        assert_eq!(
            pick_completion_spec_mode(false, false, false, false, true),
            CompletionSpecMode::Ngram
        );
        assert_eq!(
            pick_completion_spec_mode(false, false, false, false, false),
            CompletionSpecMode::None
        );
    }

    #[test]
    fn build_prompt_ids_with_suffix_errors_when_model_lacks_fim_tokens() {
        // Standard synth GGUF (no FIM markers).
        let path = synth_path("no-fim");
        let params = SynthLlama::default();
        write_synthetic_llama_gguf(&path, &params);
        let gguf = Gguf::open(&path).expect("open");
        let tok = Tokenizer::from_gguf(&gguf).expect("load");
        let err = build_prompt_ids(&tok, "hi", Some("bye")).expect_err("must fail");
        assert!(
            err.contains("Fill-In-Middle") || err.contains("FIM"),
            "error mentions FIM: {err}"
        );
        let _ = std::fs::remove_file(&path);
    }
}
