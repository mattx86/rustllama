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
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use rustllama_engine::{SamplingParams, Token};
use serde::{Deserialize, Serialize};

use crate::{model_not_found, AppState, Usage};

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
    #[serde(default)]
    pub echo: bool,
    /// OpenAI streaming knob. `include_usage: true` adds a final usage
    /// chunk before `[DONE]`.
    pub stream_options: Option<crate::StreamOptions>,
    /// Legacy-completions `logprobs`: an integer 0..=5 giving the number of
    /// top alternative logprobs to return per token. `None` disables.
    pub logprobs: Option<u32>,
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
#[derive(Debug, Serialize)]
pub struct LogprobsOut {
    pub tokens: Vec<String>,
    pub token_logprobs: Vec<f32>,
    pub top_logprobs: Vec<serde_json::Map<String, serde_json::Value>>,
    pub text_offset: Vec<usize>,
}

pub async fn completions(
    State(state): State<AppState>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let Some(serving) = state.resolve(req.model.as_deref()).await else {
        return model_not_found(req.model.as_deref());
    };
    // Two-phase admission: tokenizer access + FIM prompt assembly +
    // prompt-token count all happen against the Arc-shared CPU engine
    // BEFORE the gate wait, so the work overlaps any in-flight
    // request's decode.
    let pregate_permit = match serving.try_admit() {
        Ok(p) => p,
        Err(be) => return be.into_response(),
    };
    let sampling = build_sampling(&req);
    let model = req.model.clone().unwrap_or_else(|| serving.model_id.clone());

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

    let prompt_ids = match build_prompt_ids(tokenizer, &req.prompt, req.suffix.as_deref()) {
        Ok(ids) => ids,
        Err(e) => return (StatusCode::BAD_REQUEST, e).into_response(),
    };

    // Prompt token count = length of the prompt_ids list we just assembled
    // (already includes any FIM special tokens). Cheap and exact.
    let prompt_tokens = prompt_ids.len() as u32;
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
        let fingerprint =
            crate::system_fingerprint(&state.version, &model, kv_dtype_label(&cpu));
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

    // Non-streaming: run generation, return JSON.
    let cpu_for_blocking = cpu.clone();
    let prompt_for_echo = if req.echo {
        req.prompt.clone()
    } else {
        String::new()
    };
    let want_logprobs = req.logprobs;

    // N-gram speculation for completions, routed at the ID level so
    // this endpoint's own tokenization (no BOS; FIM specials for
    // `suffix` requests) is preserved — re-encoding text through
    // `Engine::generate` would change the prompt. Gated like chat
    // (grammar-free only); logprobs requests take the classic arm
    // above this one (the speculative path yields no logprobs).
    // Draft-model speculation stays chat-only: its drafter consumes
    // text and would re-encode the prompt differently.
    let spec_ngram = if sampling.grammar.is_none() {
        cpu.ngram_speculative()
    } else {
        None
    };

    let (text, logprobs_out, completion_tokens) = if let Some(k) = want_logprobs {
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
            Ok(Err(e)) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
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

        let mut tokens_out = Vec::with_capacity(pairs.len());
        let mut token_logprobs = Vec::with_capacity(pairs.len());
        let mut top_logprobs = Vec::with_capacity(pairs.len());
        let mut text_offset = Vec::with_capacity(pairs.len());
        let mut text_acc = String::new();
        let mut tok_count = 0u32;

        for (id, lp) in pairs.iter() {
            let tok_str = tokenizer.decode_single(*id, true).unwrap_or_default();
            text_offset.push(text_acc.len());
            text_acc.push_str(&tok_str);
            tokens_out.push(tok_str);
            token_logprobs.push(lp.logprob);

            let mut top_map: serde_json::Map<String, serde_json::Value> =
                serde_json::Map::new();
            for alt in &lp.top {
                let alt_str = tokenizer.decode_single(alt.id, true).unwrap_or_default();
                top_map.insert(alt_str, serde_json::json!(alt.logprob));
            }
            top_logprobs.push(top_map);
            tok_count += 1;
        }

        let logprobs_payload = LogprobsOut {
            tokens: tokens_out,
            token_logprobs,
            top_logprobs,
            text_offset,
        };
        (text_acc, Some(logprobs_payload), tok_count)
    } else if let Some(ng) = spec_ngram {
        // Speculative path: drain the ID-level TokenStream here in
        // the async handler — its verify rounds run on
        // `spawn_blocking` internally, so this does not block the
        // runtime. Stop strings are honored inside the stream
        // (engine-side), same as the classic `generate_*` methods.
        let prompt_ids_i32: Vec<i32> = prompt_ids.iter().map(|&v| v as i32).collect();
        let mut st = match cpu.speculate_ngram_stream_from_ids(prompt_ids_i32, ng, sampling.clone())
        {
            Ok(st) => st,
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        };
        use futures::StreamExt;
        let mut text = String::new();
        while let Some(item) = st.next().await {
            match item {
                Ok(tok) => text.push_str(&tok.text),
                Err(e) => {
                    return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response()
                }
            }
        }
        // Completion tokens: re-tokenize the generated text, matching
        // the classic arm below.
        let completion_tokens = cpu
            .tokenizer()
            .and_then(|t| t.encode(&text, false).ok())
            .map(|ids| ids.len() as u32)
            .unwrap_or(0);
        (text, None, completion_tokens)
    } else {
        // Clone the sampling config before moving it into the blocking
        // closure — the outer fn still needs `sampling.max_tokens` for
        // the finish_reason decision below.
        let sampling_inner = sampling.clone();
        let result =
            tokio::task::spawn_blocking(move || cpu_for_blocking.generate_text_from_ids(&prompt_ids, &sampling_inner))
                .await;
        let text = match result {
            Ok(Ok(t)) => t,
            Ok(Err(e)) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        };

        // Completion tokens: re-tokenize the generated text. Approximate but
        // matches what OpenAI reports.
        let completion_tokens = cpu
            .tokenizer()
            .and_then(|t| t.encode(&text, false).ok())
            .map(|ids| ids.len() as u32)
            .unwrap_or(0);
        (text, None, completion_tokens)
    };

    let full_text = format!("{}{}", prompt_for_echo, text);
    let stats = cpu.last_request_stats();
    let fingerprint =
        crate::system_fingerprint(&state.version, &model, kv_dtype_label(&cpu));
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
            let prefix_ids = tokenizer
                .encode(prompt, false)
                .map_err(|e| e.to_string())?;
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

fn build_sampling(req: &CompletionRequest) -> SamplingParams {
    let mut s = SamplingParams::default();
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
    s
}

fn unix_ts() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn request_id() -> String {
    format!("cmpl-{:032x}", unix_ts() as u128)
}

/// Short stable label for the engine's active KV dtype — mirror of
/// `chat::kv_dtype_label`, kept local so completions doesn't depend
/// on `chat`'s private helpers. Used as one of the inputs to
/// `system_fingerprint`.
fn kv_dtype_label(cpu: &rustllama_engine::CpuEngine) -> &'static str {
    match cpu.kv_dtype() {
        rustllama_engine::KvDtype::F32 => "f32",
        rustllama_engine::KvDtype::Q8_0 => "q8_0",
        rustllama_engine::KvDtype::Tq(_) => "tq",
        rustllama_engine::KvDtype::Nvfp4 => "nvfp4",
        rustllama_engine::KvDtype::Q4_0 => "q4_0",
    }
}

#[allow(clippy::too_many_arguments)]
async fn completions_stream(
    cpu: std::sync::Arc<rustllama_engine::CpuEngine>,
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
    // with the SSE consumer — see CpuEngine::generate_token_ids_streaming.
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
    if let Some(ng) = spec_ngram {
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
    use super::{build_prompt_ids, build_sampling, CompletionRequest};
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
        assert_eq!(&got[off + 1..off + 1 + suffix_ids.len()], suffix_ids.as_slice());
        assert_eq!(got[off + 1 + suffix_ids.len()], fim.middle);
        assert_eq!(got.len(), prefix_ids.len() + suffix_ids.len() + 3);
        let _ = std::fs::remove_file(&path);
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
