//! Minimal OpenAI-compatible HTTP client used by `rustllama chat` and the
//! Tauri GUI.

use std::pin::Pin;

use futures::stream::Stream;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("server returned {status}: {body}")]
    Server { status: u16, body: String },
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("stream: {0}")]
    Stream(String),
}

pub type Result<T> = std::result::Result<T, ClientError>;

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    /// OpenAI `stream_options` — currently honors only
    /// `include_usage` (server emits a final SSE chunk with
    /// `choices: []` + a populated `usage` block). When set,
    /// streaming consumers receive a [`ChatEvent::Usage`] event
    /// just before [`ChatEvent::Finish`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream_options: Option<StreamOptions>,
    /// CLARIFY opt-in (rustllama extension). When `Some(true)`, the server
    /// routes the request through the tools path advertising only the
    /// reserved `ask_user` tool, so the model may pause and ask the user a
    /// clarifying question (delivered as [`ChatEvent::AskUser`]) instead of
    /// guessing. `None` omits the field entirely, so a request that leaves
    /// it unset is wire-identical to the pre-CLARIFY client. The chat
    /// frontends (REPL/TUI/GUI) set this to `Some(true)` by default and let
    /// the user toggle it off.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_clarify: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct StreamOptions {
    pub include_usage: bool,
}

#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    /// OpenAI-compat stable backend identifier (`fp_<12hex>`). Same
    /// inputs (server version + model id + KV dtype) always yield
    /// the same fingerprint within a process — clients compare it
    /// across consecutive requests to detect a backend swap.
    /// Absent from a hypothetical pre-fingerprint server response;
    /// `#[serde(default)]` keeps deserialization lenient.
    #[serde(default)]
    pub system_fingerprint: Option<String>,
    /// Token-count + per-request timing block. Editor clients that
    /// display "32 tok/s" in a status bar read `usage.decode_ms` /
    /// `usage.completion_tokens` here. `None` on errors / minimal
    /// servers that don't emit usage.
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessageOut,
    pub finish_reason: Option<String>,
}

/// OpenAI-compat usage block. The base three fields are the
/// spec'd ones; everything else is a rustllama extension (the
/// server emits them under `#[serde(skip_serializing_if = "is_none")]`
/// so spec-strict clients can ignore them).
#[derive(Debug, Clone, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    /// Wall-clock prefill ms (rustllama extension).
    #[serde(default)]
    pub prefill_ms: Option<f64>,
    /// Wall-clock decode ms (rustllama extension).
    #[serde(default)]
    pub decode_ms: Option<f64>,
    /// Prompt tokens actually re-run through prefill after the
    /// prefix-cache hit pre-empted the matching prefix
    /// (rustllama extension).
    #[serde(default)]
    pub tokens_prefilled: Option<u32>,
    /// Prompt tokens served from the prefix cache without
    /// re-running prefill (rustllama extension).
    #[serde(default)]
    pub cache_hit_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessageOut {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    /// FIM suffix. When set, the server wraps the prompt+suffix in the
    /// model's Fill-In-Middle special tokens so the LLM fills the gap.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k sampling. `0` = full vocab, no truncation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    /// Repeat penalty. `1.0` = no penalty.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub stop: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
pub struct CompletionResponse {
    pub id: String,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    /// See [`ChatResponse::system_fingerprint`].
    #[serde(default)]
    pub system_fingerprint: Option<String>,
    /// See [`ChatResponse::usage`].
    #[serde(default)]
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: Option<String>,
}

/// `POST /v1/embeddings` request shape. Accepts a single string,
/// an array of strings, or pre-tokenized int arrays (the OpenAI
/// power-user extension). Defaults to "float" encoding format.
#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingsRequest {
    pub model: Option<String>,
    pub input: EmbeddingsInput,
    /// `"float"` (default) returns JSON arrays of f32. `"base64"`
    /// returns each vector as an LE-f32 base64 blob. Other values
    /// → 400 from the server.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    /// MRL / Matryoshka truncated-dim. Requires an MRL-trained
    /// model for the truncated vector to be semantically valid.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
}

/// Input shape — server-side parses as untagged enum so all four
/// variants below are valid wire payloads.
#[derive(Debug, Clone, Serialize)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    Single(String),
    Batch(Vec<String>),
    SingleTokens(Vec<i32>),
    BatchTokens(Vec<Vec<i32>>),
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<EmbeddingItem>,
    pub model: String,
    pub usage: EmbeddingsUsage,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingItem {
    pub object: String,
    pub index: u32,
    /// Either an array of f32 (encoding_format = "float") or a
    /// base64-encoded LE-f32 blob (encoding_format = "base64").
    /// `serde(untagged)` resolves both shapes automatically.
    pub embedding: EmbeddingValue,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingValue {
    Floats(Vec<f32>),
    Base64(String),
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingsUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

/// Body shape for `POST /v1/models/load`. Either `path` or `hub` is
/// required; the others are optional knobs that get applied to the
/// freshly-loaded model.
#[derive(Debug, Clone, Default, Serialize)]
pub struct LoadModelParams {
    /// Absolute path to a `.gguf` file. Mutually exclusive with `hub`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<std::path::PathBuf>,
    /// HuggingFace ref `owner/repo:filename`. Server resolves against
    /// its local cache; does not trigger a download.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hub: Option<String>,
    /// Context length override (defaults to 8192 server-side).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ctx_size: Option<usize>,
    /// Prefill chunk size in tokens.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub batch_size: Option<u32>,
    /// KV-cache dtype: `"f32"` or `"q8_0"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_dtype: Option<String>,
}

/// One tool call the model emitted, reassembled from the streaming
/// `tool_calls` deltas (header + argument fragments). Surfaced whole via
/// [`ChatEvent::ToolCalls`] so a frontend can apply a confirm / auto-run
/// policy without reimplementing delta reassembly.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ToolCall {
    /// Server-assigned call id (`call_…`).
    pub id: String,
    /// Function name.
    pub name: String,
    /// Raw argument JSON string (accumulated across arg-delta chunks).
    pub arguments: String,
}

/// One streamed event delivered to the REPL / GUI from `chat_stream`.
#[derive(Debug, Clone)]
pub enum ChatEvent {
    /// Initial chunk announcing role=assistant.
    Start,
    /// Delta of generated content.
    Content(String),
    /// The model proposed one or more tool calls. Emitted once, just
    /// before [`ChatEvent::Finish`] (`"tool_calls"`), with the calls fully
    /// reassembled. Frontends that don't execute tools use this to apply a
    /// confirm / auto-run policy (approve, deny, or accept silently).
    ToolCalls(Vec<ToolCall>),
    /// CLARIFY: the model called the reserved `ask_user` tool. Carries the
    /// question `prompt` + selectable `options` for the frontend to present.
    /// Emitted just before [`ChatEvent::Finish`] (`"ask_user"`).
    AskUser {
        id: String,
        prompt: String,
        options: Vec<String>,
    },
    /// Stream finished cleanly with the given reason ("stop", "length", ...).
    Finish(String),
    /// Server-reported error.
    Error(String),
    /// Final usage chunk — appears when the client requested
    /// `stream_options.include_usage = true`. Carries the per-request
    /// token counts + the rustllama timing extensions so editor SDKs
    /// can display `"32 tok/s"` in their status bar without re-running
    /// the response through a non-streaming round-trip.
    Usage(Usage),
}

pub struct Client {
    base: reqwest::Url,
    http: reqwest::Client,
}

impl Client {
    pub fn new(base: impl reqwest::IntoUrl) -> Result<Self> {
        Ok(Self {
            base: base.into_url()?,
            http: reqwest::Client::new(),
        })
    }

    pub async fn chat(&self, req: &ChatRequest) -> Result<ChatResponse> {
        let url = join(&self.base, "/v1/chat/completions")?;
        let resp = self.http.post(url).json(req).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(ClientError::Server {
                status: status.as_u16(),
                body,
            });
        }
        Ok(resp.json().await?)
    }

    pub async fn healthz(&self) -> Result<serde_json::Value> {
        let url = join(&self.base, "/healthz")?;
        Ok(self.http.get(url).send().await?.json().await?)
    }

    /// `POST /v1/completions` — legacy completions / FIM endpoint. When
    /// `suffix` is set, the server wraps `prompt` in the model's
    /// Fill-In-Middle special tokens for inline-completion use cases
    /// (LSP `textDocument/inlineCompletion`, editor tab-completion, …).
    pub async fn completions(&self, req: &CompletionRequest) -> Result<CompletionResponse> {
        let url = join(&self.base, "/v1/completions")?;
        let resp = self.http.post(url).json(req).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(ClientError::Server {
                status: status.as_u16(),
                body,
            });
        }
        Ok(resp.json().await?)
    }

    /// Streaming `/v1/completions`. Mirrors [`Self::chat_stream`]
    /// but for the legacy completions shape — chunks carry
    /// `choices[0].text` deltas instead of `choices[0].delta.content`.
    /// `--stream` mode of the `rustllama generate` CLI dispatches here.
    pub async fn completions_stream(
        &self,
        mut req: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<CompletionEvent>> + Send>>> {
        req.stream = true;
        let url = join(&self.base, "/v1/completions")?;
        let resp = self.http.post(url).json(&req).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(ClientError::Server {
                status: status.as_u16(),
                body,
            });
        }
        let byte_stream = resp.bytes_stream();
        Ok(Box::pin(parse_sse_completions(byte_stream)))
    }

    /// `GET /v1/models` — returns the raw JSON payload. The chat REPL
    /// surfaces this via `/models` and uses it to validate `/model <id>`
    /// switches.
    pub async fn list_models(&self) -> Result<serde_json::Value> {
        let url = join(&self.base, "/v1/models")?;
        Ok(self.http.get(url).send().await?.json().await?)
    }

    /// `POST /v1/embeddings` — single-shot embedding request. Server
    /// returns one vector per input element (single string → one
    /// vector; batch → `data.len() == input.len()`). The configured
    /// `[embeddings]` slot must be enabled or the server 501s.
    pub async fn embeddings(&self, req: &EmbeddingsRequest) -> Result<EmbeddingsResponse> {
        let url = join(&self.base, "/v1/embeddings")?;
        let resp = self.http.post(url).json(req).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(ClientError::Server {
                status: status.as_u16(),
                body,
            });
        }
        Ok(resp.json().await?)
    }

    /// `POST /v1/models/load` — load a GGUF into the server's registry.
    /// Pass exactly one of `path` or `hub`. The returned JSON includes
    /// `{ model_id, previous_model_id, is_default, loaded_at }`.
    pub async fn load_model(
        &self,
        params: &LoadModelParams,
    ) -> Result<serde_json::Value> {
        let url = join(&self.base, "/v1/models/load")?;
        let resp = self.http.post(url).json(params).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(ClientError::Server {
                status: status.as_u16(),
                body,
            });
        }
        Ok(resp.json().await?)
    }

    /// `POST /v1/models/unload` — remove a model from the registry.
    pub async fn unload_model(&self, model_id: &str) -> Result<serde_json::Value> {
        let url = join(&self.base, "/v1/models/unload")?;
        let resp = self
            .http
            .post(url)
            .json(&serde_json::json!({ "model": model_id }))
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(ClientError::Server {
                status: status.as_u16(),
                body,
            });
        }
        Ok(resp.json().await?)
    }

    /// `POST /v1/models/default` — promote a loaded model to the default.
    pub async fn set_default_model(&self, model_id: &str) -> Result<serde_json::Value> {
        let url = join(&self.base, "/v1/models/default")?;
        let resp = self
            .http
            .post(url)
            .json(&serde_json::json!({ "model": model_id }))
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(ClientError::Server {
                status: status.as_u16(),
                body,
            });
        }
        Ok(resp.json().await?)
    }

    /// Issue a streaming chat completion (`stream=true`) and parse the SSE
    /// response into [`ChatEvent`]s. The returned stream yields events until
    /// `Finish` or `Error`, then completes.
    pub async fn chat_stream(
        &self,
        mut req: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<ChatEvent>> + Send>>> {
        req.stream = true;
        let url = join(&self.base, "/v1/chat/completions")?;
        let resp = self.http.post(url).json(&req).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(ClientError::Server {
                status: status.as_u16(),
                body,
            });
        }

        let byte_stream = resp.bytes_stream();
        Ok(Box::pin(parse_sse(byte_stream)))
    }
}

impl From<reqwest::Url> for Client {
    fn from(base: reqwest::Url) -> Self {
        Self {
            base,
            http: reqwest::Client::new(),
        }
    }
}

fn join(base: &reqwest::Url, path: &str) -> Result<reqwest::Url> {
    base.join(path).map_err(|e| ClientError::Server {
        status: 0,
        body: format!("url join failed: {e}"),
    })
}

/// Drive the raw byte stream into [`ChatEvent`]s. Buffers across chunk
/// boundaries; events are split on blank lines per the SSE spec.
fn parse_sse<S>(byte_stream: S) -> impl Stream<Item = Result<ChatEvent>> + Send
where
    S: Stream<Item = reqwest::Result<bytes::Bytes>> + Send + Unpin + 'static,
{
    async_stream::stream! {
        let mut buf = Vec::<u8>::new();
        let mut sent_start = false;
        let mut byte_stream = byte_stream;
        // Reassembles the streaming `tool_calls` deltas (header + arg
        // fragments) into whole calls across chunk boundaries. Drained into
        // a `ChatEvent::ToolCalls` at the finish chunk.
        let mut tool_acc: Vec<ToolCall> = Vec::new();

        while let Some(chunk) = byte_stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    yield Err(ClientError::Http(e));
                    return;
                }
            };
            buf.extend_from_slice(&chunk);

            // Split on `\n\n` event boundaries.
            loop {
                let Some(pos) = find_event_boundary(&buf) else { break; };
                let event_bytes = buf.drain(..pos).collect::<Vec<u8>>();
                // Drop the boundary marker (\n\n or \r\n\r\n).
                drain_boundary(&mut buf);
                for event in decode_event(&event_bytes, &mut tool_acc) {
                    if !sent_start {
                        sent_start = true;
                        yield Ok(ChatEvent::Start);
                    }
                    yield event;
                    // If the parsed event was Finish or Error, the server is
                    // about to send `data: [DONE]` and close. We can keep
                    // looping to consume it cleanly.
                }
            }
        }
    }
}

fn find_event_boundary(buf: &[u8]) -> Option<usize> {
    // Look for `\n\n` first; if present, return position.
    buf.windows(2).position(|w| w == b"\n\n")
}

fn drain_boundary(buf: &mut Vec<u8>) {
    if buf.starts_with(b"\n\n") {
        buf.drain(..2);
    } else if buf.starts_with(b"\r\n\r\n") {
        buf.drain(..4);
    }
}

/// Decode one SSE event into zero or more [`ChatEvent`]s. `tool_acc` is
/// caller-owned state (see [`parse_sse`]): the streaming `tool_calls` deltas
/// arrive across several events, so this accumulates them there and drains a
/// whole [`ChatEvent::ToolCalls`] at the finish chunk.
fn decode_event(bytes: &[u8], tool_acc: &mut Vec<ToolCall>) -> Vec<Result<ChatEvent>> {
    let mut out = Vec::new();
    let text = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => {
            out.push(Err(ClientError::Stream(format!("utf-8 in SSE: {e}"))));
            return out;
        }
    };

    for line in text.lines() {
        let Some(payload) = line.strip_prefix("data: ").or_else(|| line.strip_prefix("data:")) else {
            continue;
        };
        let payload = payload.trim();
        if payload == "[DONE]" {
            // No event — Finish is emitted earlier via the final-chunk
            // `finish_reason`. Continue so we keep parsing.
            continue;
        }
        match serde_json::from_str::<serde_json::Value>(payload) {
            Ok(v) => {
                if let Some(err) = v.get("error").and_then(|e| e.get("message")).and_then(|m| m.as_str()) {
                    out.push(Ok(ChatEvent::Error(err.to_string())));
                    continue;
                }
                let choice = v.pointer("/choices/0");
                if let Some(delta) = choice.and_then(|c| c.get("delta")) {
                    if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
                        if !content.is_empty() {
                            out.push(Ok(ChatEvent::Content(content.to_string())));
                        }
                    }
                    // Reassemble streaming tool-call deltas into `tool_acc`.
                    if let Some(calls) = delta.get("tool_calls").and_then(|c| c.as_array()) {
                        merge_tool_call_deltas(tool_acc, calls);
                    }
                    // CLARIFY: the reserved `ask_user` tool rides the terminal
                    // chunk as `delta.ask_user` (finish_reason "ask_user").
                    if let Some(ask) = delta.get("ask_user") {
                        out.push(Ok(ChatEvent::AskUser {
                            id: ask.get("id").and_then(|s| s.as_str()).unwrap_or("").to_string(),
                            prompt: ask
                                .get("prompt")
                                .and_then(|s| s.as_str())
                                .unwrap_or("")
                                .to_string(),
                            options: ask
                                .get("options")
                                .and_then(|o| o.as_array())
                                .map(|arr| {
                                    arr.iter()
                                        .filter_map(|x| x.as_str().map(|s| s.to_string()))
                                        .collect()
                                })
                                .unwrap_or_default(),
                        }));
                    }
                }
                if let Some(reason) = choice.and_then(|c| c.get("finish_reason")).and_then(|r| r.as_str()) {
                    // Surface any reassembled tool calls once, right before
                    // Finish, so a frontend can gate them (confirm / auto-run).
                    if !tool_acc.is_empty() {
                        out.push(Ok(ChatEvent::ToolCalls(std::mem::take(tool_acc))));
                    }
                    out.push(Ok(ChatEvent::Finish(reason.to_string())));
                }
                // OpenAI `stream_options.include_usage` emits a final
                // chunk with `choices: []` + a populated `usage`
                // block. Deserialize via the same Usage struct the
                // non-streaming response uses so the timing
                // extensions (`prefill_ms`, `decode_ms`, etc.) survive.
                if let Some(usage_val) = v.get("usage") {
                    if let Ok(usage) = serde_json::from_value::<Usage>(usage_val.clone()) {
                        out.push(Ok(ChatEvent::Usage(usage)));
                    }
                }
            }
            Err(e) => out.push(Err(ClientError::Stream(format!("bad SSE JSON: {e}")))),
        }
    }
    out
}

/// Merge a chunk's `tool_calls` delta array into the accumulator, keyed by
/// the delta's `index`. The server emits a header delta (index + id + name)
/// then argument delta(s) (index + `function.arguments`) per call; this
/// tolerates them arriving in any number of chunks by appending argument
/// fragments and only overwriting id / name when non-empty.
fn merge_tool_call_deltas(acc: &mut Vec<ToolCall>, calls: &[serde_json::Value]) {
    for c in calls {
        let index = c.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
        if acc.len() <= index {
            acc.resize(index + 1, ToolCall::default());
        }
        let slot = &mut acc[index];
        if let Some(id) = c.get("id").and_then(|x| x.as_str()) {
            if !id.is_empty() {
                slot.id = id.to_string();
            }
        }
        if let Some(f) = c.get("function") {
            if let Some(name) = f.get("name").and_then(|x| x.as_str()) {
                if !name.is_empty() {
                    slot.name = name.to_string();
                }
            }
            if let Some(args) = f.get("arguments").and_then(|x| x.as_str()) {
                slot.arguments.push_str(args);
            }
        }
    }
}

/// One streamed event from `completions_stream`. Mirror of [`ChatEvent`]
/// for the legacy `/v1/completions` SSE shape — chunks carry
/// `choices[0].text` deltas (no `delta.content` wrapper).
#[derive(Debug, Clone)]
pub enum CompletionEvent {
    /// Text delta from the model.
    Content(String),
    /// Stream finished with the given reason (`"stop"`, `"length"`, …).
    Finish(String),
    /// Server-reported error.
    Error(String),
    /// Final usage chunk when the client opted into
    /// `stream_options.include_usage`. Same shape as the chat-side variant.
    Usage(Usage),
}

fn parse_sse_completions<S>(byte_stream: S) -> impl Stream<Item = Result<CompletionEvent>> + Send
where
    S: Stream<Item = reqwest::Result<bytes::Bytes>> + Send + Unpin + 'static,
{
    async_stream::stream! {
        let mut buf = Vec::<u8>::new();
        let mut byte_stream = byte_stream;

        while let Some(chunk) = byte_stream.next().await {
            let chunk = match chunk {
                Ok(c) => c,
                Err(e) => {
                    yield Err(ClientError::Http(e));
                    return;
                }
            };
            buf.extend_from_slice(&chunk);
            loop {
                let Some(pos) = find_event_boundary(&buf) else { break; };
                let event_bytes = buf.drain(..pos).collect::<Vec<u8>>();
                drain_boundary(&mut buf);
                for event in decode_completion_event(&event_bytes) {
                    yield event;
                }
            }
        }
    }
}

fn decode_completion_event(bytes: &[u8]) -> Vec<Result<CompletionEvent>> {
    let mut out = Vec::new();
    let text = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(e) => {
            out.push(Err(ClientError::Stream(format!("utf-8 in SSE: {e}"))));
            return out;
        }
    };
    for line in text.lines() {
        let Some(payload) = line.strip_prefix("data: ").or_else(|| line.strip_prefix("data:")) else {
            continue;
        };
        let payload = payload.trim();
        if payload == "[DONE]" {
            continue;
        }
        match serde_json::from_str::<serde_json::Value>(payload) {
            Ok(v) => {
                if let Some(err) = v.get("error").and_then(|e| e.get("message")).and_then(|m| m.as_str()) {
                    out.push(Ok(CompletionEvent::Error(err.to_string())));
                    continue;
                }
                let choice = v.pointer("/choices/0");
                if let Some(text_delta) = choice.and_then(|c| c.get("text")).and_then(|t| t.as_str()) {
                    if !text_delta.is_empty() {
                        out.push(Ok(CompletionEvent::Content(text_delta.to_string())));
                    }
                }
                if let Some(reason) = choice.and_then(|c| c.get("finish_reason")).and_then(|r| r.as_str()) {
                    out.push(Ok(CompletionEvent::Finish(reason.to_string())));
                }
                if let Some(usage_val) = v.get("usage") {
                    if let Ok(usage) = serde_json::from_value::<Usage>(usage_val.clone()) {
                        out.push(Ok(CompletionEvent::Usage(usage)));
                    }
                }
            }
            Err(e) => out.push(Err(ClientError::Stream(format!("bad SSE JSON: {e}")))),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_role_chunk_does_not_emit_content() {
        let s = "data: {\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},\"finish_reason\":null}]}";
        let evts = decode_event(s.as_bytes(), &mut Vec::new());
        let count = evts.into_iter().filter_map(|e| e.ok()).count();
        assert_eq!(count, 0);
    }

    #[test]
    fn decode_content_chunk() {
        let s = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}";
        let evts: Vec<_> = decode_event(s.as_bytes(), &mut Vec::new())
            .into_iter()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(evts.len(), 1);
        assert!(matches!(evts[0], ChatEvent::Content(ref c) if c == "hi"));
    }

    #[test]
    fn decode_finish_chunk() {
        let s = "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}";
        let evts: Vec<_> = decode_event(s.as_bytes(), &mut Vec::new())
            .into_iter()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(evts.len(), 1);
        assert!(matches!(evts[0], ChatEvent::Finish(ref r) if r == "stop"));
    }

    #[test]
    fn decode_done_marker_is_silent() {
        let evts = decode_event(b"data: [DONE]", &mut Vec::new());
        assert!(evts.is_empty());
    }

    /// Streaming tool-call deltas (header chunk, then an args chunk, then
    /// the finish chunk) reassemble into one `ChatEvent::ToolCalls`, emitted
    /// just before `Finish("tool_calls")`. The accumulator persists across
    /// `decode_event` calls exactly as `parse_sse` drives it.
    #[test]
    fn decode_reassembles_streaming_tool_calls_before_finish() {
        let mut acc: Vec<ToolCall> = Vec::new();
        let header = "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\
                      \"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\
                      \"arguments\":\"\"}}]}}]}";
        let args = "data: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\
                    \"function\":{\"arguments\":\"{\\\"city\\\":\\\"NYC\\\"}\"}}]}}]}";
        let fin = "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}";

        // Header + args chunks accumulate but emit no ChatEvent.
        assert!(decode_event(header.as_bytes(), &mut acc)
            .into_iter()
            .filter_map(|e| e.ok())
            .next()
            .is_none());
        assert!(decode_event(args.as_bytes(), &mut acc)
            .into_iter()
            .filter_map(|e| e.ok())
            .next()
            .is_none());
        // Finish chunk drains the calls, then Finish.
        let evts: Vec<_> = decode_event(fin.as_bytes(), &mut acc)
            .into_iter()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(evts.len(), 2);
        match &evts[0] {
            ChatEvent::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, "call_1");
                assert_eq!(calls[0].name, "get_weather");
                assert_eq!(calls[0].arguments, "{\"city\":\"NYC\"}");
            }
            other => panic!("expected ToolCalls first, got {other:?}"),
        }
        assert!(matches!(evts[1], ChatEvent::Finish(ref r) if r == "tool_calls"));
        assert!(acc.is_empty(), "accumulator must be drained");
    }

    /// CLARIFY: the terminal `ask_user` chunk decodes to `ChatEvent::AskUser`
    /// (prompt + options), then `Finish("ask_user")`.
    #[test]
    fn decode_ask_user_terminal_chunk() {
        let s = "data: {\"choices\":[{\"index\":0,\"delta\":{\"ask_user\":{\"id\":\"call_9\",\
                 \"kind\":\"clarify\",\"prompt\":\"Which file?\",\"options\":[\"a.rs\",\"b.rs\"]}},\
                 \"finish_reason\":\"ask_user\"}]}";
        let evts: Vec<_> = decode_event(s.as_bytes(), &mut Vec::new())
            .into_iter()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(evts.len(), 2);
        match &evts[0] {
            ChatEvent::AskUser { id, prompt, options } => {
                assert_eq!(id, "call_9");
                assert_eq!(prompt, "Which file?");
                assert_eq!(options, &["a.rs".to_string(), "b.rs".to_string()]);
            }
            other => panic!("expected AskUser first, got {other:?}"),
        }
        assert!(matches!(evts[1], ChatEvent::Finish(ref r) if r == "ask_user"));
    }

    /// Pinning the non-streaming `ChatResponse` deserializes the new
    /// fields the server emits (`system_fingerprint`, `usage`).
    /// Before this turn, the client dropped both — editor SDKs
    /// built on top would silently miss them.
    #[test]
    fn chat_response_deserializes_system_fingerprint_and_usage() {
        let body = serde_json::json!({
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "created": 1700000000u64,
            "model": "qwen2.5-coder-7b",
            "system_fingerprint": "fp_44709d6fcb12",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "ok" },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 17,
                "completion_tokens": 1,
                "total_tokens": 18,
                "prefill_ms": 12.5,
                "decode_ms": 8.0,
                "tokens_prefilled": 17,
                "cache_hit_tokens": 0
            }
        });
        let resp: ChatResponse = serde_json::from_value(body).unwrap();
        assert_eq!(resp.system_fingerprint.as_deref(), Some("fp_44709d6fcb12"));
        let usage = resp.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 17);
        assert_eq!(usage.completion_tokens, 1);
        assert_eq!(usage.total_tokens, 18);
        assert_eq!(usage.prefill_ms, Some(12.5));
        assert_eq!(usage.cache_hit_tokens, Some(0));
    }

    /// Backwards-compat: a minimal server (or a future mocking server)
    /// that omits `system_fingerprint` + `usage` must still
    /// deserialize cleanly. `#[serde(default)]` enforces this.
    #[test]
    fn chat_response_deserializes_minimal_shape_without_new_fields() {
        let body = serde_json::json!({
            "id": "chatcmpl-min",
            "model": "mock",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "x" },
                "finish_reason": null
            }]
        });
        let resp: ChatResponse = serde_json::from_value(body).unwrap();
        assert!(resp.system_fingerprint.is_none());
        assert!(resp.usage.is_none());
    }

    /// Streaming usage chunk → `ChatEvent::Usage`. Editor SDKs
    /// reading the stream see the per-request stats without
    /// re-running through the non-streaming endpoint.
    #[test]
    fn decode_usage_chunk_emits_usage_event() {
        let s = "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":17,\
                 \"completion_tokens\":3,\"total_tokens\":20,\
                 \"prefill_ms\":12.5,\"decode_ms\":30.0,\
                 \"tokens_prefilled\":17,\"cache_hit_tokens\":0}}";
        let evts: Vec<_> = decode_event(s.as_bytes(), &mut Vec::new())
            .into_iter()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(evts.len(), 1);
        let usage = match &evts[0] {
            ChatEvent::Usage(u) => u,
            other => panic!("expected Usage, got {other:?}"),
        };
        assert_eq!(usage.prompt_tokens, 17);
        assert_eq!(usage.completion_tokens, 3);
        assert_eq!(usage.decode_ms, Some(30.0));
        assert_eq!(usage.cache_hit_tokens, Some(0));
    }

    /// Malformed usage block: missing required `total_tokens`.
    /// Decode falls through silently rather than emitting an
    /// Error event — the rest of the chunk still parses (a
    /// future-Ollama-quirk surface that flags `choices: []` +
    /// some non-OpenAI `usage` shape shouldn't bring the stream
    /// down).
    #[test]
    fn decode_usage_chunk_silently_skips_malformed_usage() {
        let s = "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":17}}";
        let evts: Vec<_> = decode_event(s.as_bytes(), &mut Vec::new())
            .into_iter()
            .filter_map(|e| e.ok())
            .collect();
        // No Usage event — Usage deserialization failed silently.
        assert!(
            !evts.iter().any(|e| matches!(e, ChatEvent::Usage(_))),
            "malformed usage must not emit a Usage event: {evts:?}"
        );
    }

    /// Completions streaming: `choices[0].text` (not `delta.content`)
    /// is where the per-chunk delta lives. Pin the wire mapping so a
    /// future refactor doesn't accidentally read `delta.content`
    /// (which would silently produce empty Content events).
    #[test]
    fn decode_completion_event_emits_text_delta() {
        let s = "data: {\"choices\":[{\"text\":\"hello\",\"index\":0,\"finish_reason\":null}]}";
        let evts: Vec<_> = decode_completion_event(s.as_bytes())
            .into_iter()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(evts.len(), 1);
        assert!(matches!(&evts[0], CompletionEvent::Content(c) if c == "hello"));
    }

    /// Empty `text` delta produces no event — same convention as
    /// chat-side, lets the server emit a heartbeat-style empty
    /// chunk without spamming downstream consumers.
    #[test]
    fn decode_completion_event_skips_empty_text() {
        let s = "data: {\"choices\":[{\"text\":\"\",\"index\":0,\"finish_reason\":null}]}";
        let evts: Vec<_> = decode_completion_event(s.as_bytes())
            .into_iter()
            .filter_map(|e| e.ok())
            .collect();
        assert!(evts.is_empty());
    }

    /// `finish_reason` on the final chunk surfaces as a Finish event.
    /// Drives the streaming `generate --stream` CLI to break its loop.
    #[test]
    fn decode_completion_event_emits_finish_on_terminal_chunk() {
        let s = "data: {\"choices\":[{\"text\":\"\",\"index\":0,\"finish_reason\":\"stop\"}]}";
        let evts: Vec<_> = decode_completion_event(s.as_bytes())
            .into_iter()
            .filter_map(|e| e.ok())
            .collect();
        // text=""  → no Content; finish_reason → Finish. Exactly one event.
        assert_eq!(evts.len(), 1);
        assert!(matches!(&evts[0], CompletionEvent::Finish(r) if r == "stop"));
    }

    /// Usage chunk in the completions stream → CompletionEvent::Usage,
    /// same shape the chat-side stream uses.
    #[test]
    fn decode_completion_event_emits_usage_chunk() {
        let s = "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\
                 \"completion_tokens\":2,\"total_tokens\":5}}";
        let evts: Vec<_> = decode_completion_event(s.as_bytes())
            .into_iter()
            .filter_map(|e| e.ok())
            .collect();
        let usage = match &evts[0] {
            CompletionEvent::Usage(u) => u,
            other => panic!("expected Usage, got {other:?}"),
        };
        assert_eq!(usage.total_tokens, 5);
    }

    /// Embeddings response with the `"float"` encoding shape: each
    /// item carries an array-of-f32 vector. Pin the deserialization
    /// so a server-side renaming doesn't silently break clients.
    #[test]
    fn embeddings_response_deserializes_float_format() {
        let body = serde_json::json!({
            "object": "list",
            "data": [{
                "object": "embedding",
                "index": 0,
                "embedding": [0.1, 0.2, -0.3, 0.4]
            }],
            "model": "bge-small-en",
            "usage": {
                "prompt_tokens": 4,
                "total_tokens": 4
            }
        });
        let resp: EmbeddingsResponse = serde_json::from_value(body).unwrap();
        assert_eq!(resp.data.len(), 1);
        match &resp.data[0].embedding {
            EmbeddingValue::Floats(v) => {
                assert_eq!(v.len(), 4);
                assert!((v[0] - 0.1).abs() < 1e-6);
            }
            other => panic!("expected Floats, got {other:?}"),
        }
        assert_eq!(resp.usage.total_tokens, 4);
    }

    /// `"base64"` format: the `embedding` value is a string.
    #[test]
    fn embeddings_response_deserializes_base64_format() {
        let body = serde_json::json!({
            "object": "list",
            "data": [{
                "object": "embedding",
                "index": 0,
                "embedding": "AAAAvX//Pz+amZk9"
            }],
            "model": "bge",
            "usage": { "prompt_tokens": 1, "total_tokens": 1 }
        });
        let resp: EmbeddingsResponse = serde_json::from_value(body).unwrap();
        match &resp.data[0].embedding {
            EmbeddingValue::Base64(s) => {
                assert_eq!(s, "AAAAvX//Pz+amZk9");
            }
            other => panic!("expected Base64, got {other:?}"),
        }
    }

    /// `CompletionResponse` mirror: legacy `/v1/completions` shape
    /// also gets the two new fields.
    #[test]
    fn completion_response_deserializes_system_fingerprint_and_usage() {
        let body = serde_json::json!({
            "id": "cmpl-xyz",
            "object": "text_completion",
            "model": "qwen-coder-fim",
            "system_fingerprint": "fp_deadbeefcafe",
            "choices": [{
                "text": " return 42;",
                "index": 0,
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 4,
                "total_tokens": 9
            }
        });
        let resp: CompletionResponse = serde_json::from_value(body).unwrap();
        assert_eq!(resp.system_fingerprint.as_deref(), Some("fp_deadbeefcafe"));
        assert_eq!(resp.usage.unwrap().total_tokens, 9);
    }
}
