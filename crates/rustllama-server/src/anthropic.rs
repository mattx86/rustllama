//! Anthropic Messages API compatibility layer (`POST /v1/messages`).
//!
//! Implements the request/response shape and the event-typed SSE stream
//! protocol that Anthropic SDK clients (Claude Code, `@anthropic-ai/sdk`,
//! `anthropic-sdk-python`, ...) speak.
//!
//! v1 scope:
//!   - text-only content (string or `[{"type":"text","text":"..."}]`)
//!   - non-streaming + streaming (full event sequence: `message_start` →
//!     `content_block_start` → `content_block_delta` → `content_block_stop`
//!     → `message_delta` → `message_stop`)
//!   - `system` field (top-level, per spec; accepts plain string or
//!     `[{"type":"text","text":...}]` blocks)
//!   - `stop_sequences`, `temperature`, `top_p`, `top_k`, `max_tokens`
//!   - `usage` block with `input_tokens` / `output_tokens`
//!   - `tools` + `tool_use` content blocks (non-stream + stream)
//!   - cancellation: surfaces `stop_reason: "stop_sequence"` +
//!     `stop_sequence: "__cancelled__"` so SDK consumers can detect a
//!     server-initiated abort without crashing on an unknown enum value.
//!
//! Audit gaps (deferred to v1.x):
//!   - Image content blocks: parsed but rendered as a
//!     `[image: <media_type>]` placeholder rather than fed to a VLM.
//!     Lets multimodal SDK callers (Claude Code with image attachments)
//!     hit our text-only models without crashing. Real VLM support is
//!     a v1.x item.
//!   - `cache_control` annotations are accepted on text blocks and
//!     surfaced via `cache_read_input_tokens` in the usage block,
//!     but the per-block `ephemeral` marker is informational — the
//!     engine uses implicit LCP prefix caching for every prefill,
//!     not per-marker hard boundaries. `cache_creation_input_tokens`
//!     always reports 0 until per-marker caching lands.
//!   - Extended thinking annotations.
//!   - `tool_result` content blocks in the request: parsed as plain text;
//!     Anthropic's full tool-result loop requires a richer message shape
//!     than we currently render through the chat template.
//!   - Tool-call iteration cap (engine field `tool_call_limit_hit`): the
//!     Anthropic enum has no dedicated reason for this, so we keep
//!     `stop_reason: "tool_use"` (SDKs strict about the enum still get a
//!     valid value) and populate `stop_sequence:
//!     "__tool_call_iteration_limit__"` as the discriminator. This
//!     mirrors the `__cancelled__` sentinel pattern used for
//!     cancellation, so clients can pattern-match on `stop_sequence` to
//!     tell a normal tool-use stop from a capped one.
//!   - `usage` extension fields (`prefill_ms`, `decode_ms`,
//!     `cache_hit_tokens`, `tokens_prefilled`): surfaced on both the
//!     non-stream response and the `message_delta` streaming event
//!     when a `CpuEngine` is loaded. Marked `skip_serializing_if`
//!     so they stay absent under MockEngine and don't break strict
//!     SDK consumers that haven't seen them before.

use std::convert::Infallible;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::Stream;
use futures::StreamExt;
use rustllama_engine::{ChatMessage, SamplingParams};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{model_not_found, AppState, ServingModel};

// ----- request --------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    pub model: Option<String>,
    pub messages: Vec<MessageWire>,
    /// Anthropic puts the system prompt at the top level, not in `messages`.
    /// Accept either a plain string or an array of text blocks.
    pub system: Option<SystemPrompt>,
    pub max_tokens: u32,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    /// Accepted but ignored (telemetry hook).
    #[allow(dead_code)]
    pub metadata: Option<Value>,
    /// Anthropic-shaped tools array: `[{name, description, input_schema}]`.
    /// Converted to OpenAI shape before being passed into the chat
    /// template (templates were trained against the OpenAI shape).
    pub tools: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum SystemPrompt {
    Plain(String),
    Blocks(Vec<TextBlock>),
}

impl SystemPrompt {
    fn into_text(self) -> String {
        match self {
            SystemPrompt::Plain(s) => s,
            SystemPrompt::Blocks(blocks) => blocks
                .into_iter()
                .filter(|b| b.kind == "text")
                .map(|b| b.text)
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct TextBlock {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: String,
    /// For `kind: "image"` (Anthropic) — `{type, media_type, data | url}`.
    #[serde(default)]
    pub source: Option<ImageSource>,
    /// For `kind: "image_url"` (OpenAI-style multimodal payload that
    /// some clients send through Anthropic-shaped APIs anyway).
    /// Accepted from the wire so deserialization doesn't fail; the
    /// stub placeholder is derived from `kind` alone, not from this
    /// field's url/detail.
    #[allow(dead_code)]
    #[serde(default)]
    pub image_url: Option<ImageUrl>,
    /// Anthropic prefix-caching annotation (`{"type": "ephemeral"}`).
    /// Claude Code / Claude Desktop / langchain-anthropic auto-annotate
    /// system prompts + long context blocks with this to mark them as
    /// cacheable. We accept the field so the request parses cleanly
    /// AND surface a corresponding `cache_read_input_tokens` count in
    /// [`AnthropicUsage`] when our prefix cache actually fires.
    ///
    /// Honoring the per-block `ephemeral` marker as a hard cache
    /// boundary (vs implicit longest-common-prefix) is a follow-up;
    /// today's behavior caches prefixes implicitly and the marker is
    /// informational. The `cache_read_input_tokens` we report is the
    /// real engine measurement, so clients get accurate feedback
    /// about whether their cache strategy is winning.
    #[allow(dead_code)]
    #[serde(default)]
    pub cache_control: Option<CacheControl>,
}

/// Anthropic prefix-cache annotation. Today only `{"type": "ephemeral"}`
/// is defined upstream — we accept any string in the `type` field
/// without validation so a future Anthropic-side extension (e.g.
/// `"persistent"`) doesn't 400 our requests. Accepted on text blocks
/// (Anthropic message content + system text blocks) and on tool
/// definitions (top-level `tools[].cache_control`).
#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
pub struct CacheControl {
    #[serde(default, rename = "type")]
    pub kind: String,
}

/// `source` payload on an Anthropic-shape `image` content block. The
/// `media_type` lets us emit a descriptive placeholder; `data` / `url`
/// / `kind` are accepted from the wire but not surfaced to the
/// text-only model — they'd flow once VLM support lands.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct ImageSource {
    #[serde(default, rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub media_type: String,
    #[serde(default)]
    pub data: String,
    #[serde(default)]
    pub url: String,
}

/// `image_url` payload on an OpenAI-shape `image_url` content block.
/// Same v1 status as [`ImageSource`]: accepted but not surfaced.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct ImageUrl {
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub detail: String,
}

#[derive(Debug, Deserialize)]
pub struct MessageWire {
    pub role: String,
    pub content: ContentField,
}

/// Anthropic's `content` accepts either a string OR an array of blocks.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ContentField {
    Plain(String),
    Blocks(Vec<TextBlock>),
}

impl ContentField {
    fn into_text(self) -> String {
        match self {
            ContentField::Plain(s) => s,
            ContentField::Blocks(blocks) => blocks
                .into_iter()
                .map(content_block_to_text)
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

/// Render one content block as plain text. Text blocks pass through;
/// image blocks become a `[image: <media_type or "unknown">]`
/// placeholder so the text-only model knows an attachment was
/// referenced. Unknown kinds with non-empty `text` get their text
/// surfaced as a fallback — matches real-Anthropic's lenient
/// forward-compatibility behavior; unknown kinds without text become
/// a no-op so the request doesn't 400.
///
/// v1 explicitly does NOT route image data into the prompt — we don't
/// have a VLM. Once VLM support lands, this is the seam where the
/// `data` / `url` payload starts flowing through.
fn content_block_to_text(b: TextBlock) -> String {
    match b.kind.as_str() {
        "text" => b.text,
        "image" => {
            let media = if !b.source.as_ref().map(|s| s.media_type.is_empty()).unwrap_or(true) {
                b.source.as_ref().unwrap().media_type.clone()
            } else {
                "unknown".to_string()
            };
            format!("[image: {media}]")
        }
        "image_url" => {
            // OpenAI-shape multimodal payload that some clients send
            // through Anthropic-shaped APIs anyway. Mirror the OpenAI
            // adapter: preserve the URL in the placeholder so
            // non-vision coding models can reason about file
            // extension / domain.
            match b.image_url.as_ref() {
                Some(iu) if !iu.url.is_empty() => format!("[image: {}]", iu.url),
                _ => "[image: url]".to_string(),
            }
        }
        _ => b.text,
    }
}

// ----- response (non-streaming) ---------------------------------------------

#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<ResponseBlock>,
    pub stop_reason: &'static str,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

/// One content block in a non-streaming `messages` response. Either a
/// `{"type":"text", "text":"..."}` block or a
/// `{"type":"tool_use", "id":..., "name":..., "input":{...}}` block.
/// Serialized with `#[serde(untagged)]` so each variant produces only the
/// fields Anthropic clients expect.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum ResponseBlock {
    Text {
        #[serde(rename = "type")]
        kind: &'static str,
        text: String,
    },
    ToolUse {
        #[serde(rename = "type")]
        kind: &'static str,
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Serialize, Clone, Copy, Default)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// **Anthropic-native**: prompt tokens served from a prior
    /// cached prefix. Mirrors `cache_hit_tokens` under the field name
    /// Anthropic SDKs read (Claude Desktop / Code / langchain-anthropic
    /// look here to verify their `cache_control: {type: "ephemeral"}`
    /// annotations actually saved tokens). Our engine's prefix cache
    /// is implicit-LCP rather than per-marker-controlled, but the
    /// measurement is accurate: this is the real number of tokens
    /// skipped during prefill.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u32>,
    /// **Anthropic-native**: prompt tokens that populated the cache
    /// during this request. Reported as `0` today — our engine caches
    /// prefixes implicitly on every successful generation, so
    /// attributing a specific number to "creation due to the client's
    /// cache_control marker" would be misleading. Honoring the
    /// per-block `ephemeral` marker as a hard cache boundary is a
    /// follow-up; once that lands this field will report the real
    /// per-marker creation count.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u32>,
    /// **Extension** (rustllama-only): wall-clock ms the engine spent
    /// running prefill forward passes. Useful for editor UIs that
    /// surface TTFT. Anthropic SDK consumers see this as an unknown
    /// field and ignore it; strictness-on-unknown SDKs would need to
    /// opt out, but every shipping SDK we tested tolerates it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_ms: Option<f64>,
    /// **Extension**: wall-clock ms spent running decode forwards.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_ms: Option<f64>,
    /// **Extension**: number of prompt positions whose K/V state was
    /// reused from the prefix cache (live LCP + pool restore + extended
    /// LCP after restore). Higher = more of the prefill was skipped.
    /// Same number as [`Self::cache_read_input_tokens`] but kept
    /// under both names for backwards compatibility with rustllama
    /// clients that came in before the Anthropic-native field
    /// landed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_hit_tokens: Option<u32>,
    /// **Extension**: number of prompt positions that actually ran
    /// through prefill forward passes after subtracting `cache_hit_tokens`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_prefilled: Option<u32>,
}

impl AnthropicUsage {
    /// Populate the rustllama extension fields from a `RequestStats`.
    /// Mirrors the OpenAI `Usage::with_stats` helper so both surfaces
    /// emit the same metrics under the same names. Also populates
    /// the Anthropic-native `cache_read_input_tokens` /
    /// `cache_creation_input_tokens` so Anthropic SDK consumers can
    /// verify their cache_control annotations are paying off.
    pub fn with_stats(mut self, s: &rustllama_engine::RequestStats) -> Self {
        self.prefill_ms = Some(s.prefill_ms);
        self.decode_ms = Some(s.decode_ms);
        self.cache_hit_tokens = Some(s.cache_hit_tokens);
        self.tokens_prefilled = Some(s.tokens_prefilled);
        // Anthropic-native mirror of the engine's cache_hit_tokens.
        // Same number, different name — gives Anthropic SDK
        // consumers the field they're looking for.
        self.cache_read_input_tokens = Some(s.cache_hit_tokens);
        // Always 0 until per-marker cache boundaries land.
        self.cache_creation_input_tokens = Some(0);
        self
    }
}

// ----- handler --------------------------------------------------------------

pub async fn messages(
    State(state): State<AppState>,
    Json(req): Json<MessagesRequest>,
) -> Response {
    // Destructure once so the borrow checker doesn't have to reason about
    // partial moves vs. borrows across the `system` / `messages` /
    // `build_sampling` boundary.
    let MessagesRequest {
        model: req_model,
        messages: req_messages,
        system: req_system,
        max_tokens,
        stream,
        temperature,
        top_p,
        top_k,
        stop_sequences,
        tools: req_tools,
        ..
    } = req;

    let sampling = build_sampling(SamplingInputs {
        temperature,
        top_p,
        top_k,
        max_tokens,
        stops: stop_sequences,
    });

    // Build the ChatMessage list. Anthropic's `system` is top-level; the
    // existing chat-template renderer expects it inline as a system
    // message (which most Jinja templates handle correctly).
    let mut msgs: Vec<ChatMessage> = Vec::with_capacity(req_messages.len() + 1);
    if let Some(sys) = req_system {
        let text = sys.into_text();
        if !text.is_empty() {
            msgs.push(ChatMessage::text("system", text));
        }
    }
    for m in req_messages {
        // Anthropic adapter still surfaces images as placeholder text
        // (the Anthropic `source.data` shape is base64 already; future
        // work can attach those bytes to ChatMessage.images alongside
        // the OpenAI image_url path landing in chat.rs).
        msgs.push(ChatMessage::text(m.role, m.content.into_text()));
    }

    let Some(serving) = state.resolve(req_model.as_deref()).await else {
        return model_not_found(req_model.as_deref());
    };
    let model = req_model.unwrap_or_else(|| serving.model_id.clone());
    let id = format!("msg_{:032x}", unix_ts() as u128);

    let input_tokens = serving
        .cpu_engine
        .as_ref()
        .and_then(|e| e.count_chat_prompt(&msgs).ok())
        .unwrap_or(0);

    // Tools: convert Anthropic shape → OpenAI shape (templates were
    // trained against the OpenAI form). When present, we pre-render the
    // prompt with the tools branch and drive the engine via `generate`
    // rather than `chat`, mirroring the OpenAI tools path.
    let tools_openai = req_tools.map(anthropic_tools_to_openai);
    let has_tools = tools_openai.is_some();

    let pre_rendered_prompt = if has_tools {
        match serving.cpu_engine.as_ref().and_then(|e| e.tokenizer()) {
            Some(tokenizer) => {
                let tok_msgs: Vec<rustllama_tokenizer::ChatMessage<'_>> = msgs
                    .iter()
                    .map(|m| rustllama_tokenizer::ChatMessage {
                        role: &m.role,
                        content: &m.content,
                    })
                    .collect();
                match tokenizer.render_chat_with_tools(
                    &tok_msgs,
                    true,
                    tools_openai.as_ref(),
                ) {
                    Ok(p) => Some(p),
                    Err(e) => {
                        return (
                            StatusCode::INTERNAL_SERVER_ERROR,
                            format!("render chat template with tools: {e}"),
                        )
                            .into_response();
                    }
                }
            }
            None => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    "tools require a model with a tokenizer",
                )
                    .into_response();
            }
        }
    } else {
        None
    };

    if stream {
        let cancel_guard = state.register_cancel(&id);
        messages_stream(
            serving,
            msgs,
            sampling,
            model,
            id,
            input_tokens,
            pre_rendered_prompt,
            cancel_guard,
        )
        .await
    } else {
        messages_blocking(
            serving,
            msgs,
            sampling,
            model,
            id,
            input_tokens,
            pre_rendered_prompt,
        )
        .await
    }
}

/// Anthropic's tools array element looks like
/// `{ name, description, input_schema }`. Most chat templates we ship
/// were trained against the OpenAI shape
/// `{ type: "function", function: { name, description, parameters } }`.
/// Convert in place so the template branch fires correctly.
fn anthropic_tools_to_openai(tools: Value) -> Value {
    let Value::Array(items) = tools else {
        return tools;
    };
    let out: Vec<Value> = items
        .into_iter()
        .map(|t| {
            let mut function = serde_json::Map::new();
            if let Some(name) = t.get("name").cloned() {
                function.insert("name".into(), name);
            }
            if let Some(desc) = t.get("description").cloned() {
                function.insert("description".into(), desc);
            }
            // Anthropic calls it `input_schema`; OpenAI calls it `parameters`.
            if let Some(schema) = t.get("input_schema").cloned() {
                function.insert("parameters".into(), schema);
            } else if let Some(params) = t.get("parameters").cloned() {
                function.insert("parameters".into(), params);
            }
            json!({ "type": "function", "function": Value::Object(function) })
        })
        .collect();
    Value::Array(out)
}

/// Sentinel `stop_sequence` value used to surface a tool-call iteration
/// cap (engine `RequestStats.tool_call_limit_hit`) over the Anthropic
/// Messages wire. `stop_reason` stays `"tool_use"` so SDK consumers
/// strict about the enum don't crash on an unknown value; clients that
/// need to distinguish capped from normal stops match on this sentinel.
pub(crate) const ANTHROPIC_TOOL_CALL_LIMIT_SENTINEL: &str = "__tool_call_iteration_limit__";

/// Sentinel `stop_sequence` value for server-initiated cancellation.
/// Same rationale as `ANTHROPIC_TOOL_CALL_LIMIT_SENTINEL`.
pub(crate) const ANTHROPIC_CANCELLED_SENTINEL: &str = "__cancelled__";

/// Decide the `(stop_reason, stop_sequence)` pair for a Messages
/// response, given the terminal flags observed during generation.
///
/// Precedence (highest first):
///   1. `hit_cancel` — operator aborted via `/v1/cancel`.
///   2. `tool_call_limit_hit` — engine's tool-call iteration cap fired.
///      Beats vanilla `tool_use` because the cap-fired case is the more
///      specific story; clients distinguish via the sentinel.
///   3. `hit_tool_use` — the response contains tool_use blocks.
///   4. `hit_stop_sequence` — a configured stop string matched.
///   5. `output_tokens >= max_tokens` — hit the generation cap.
///   6. otherwise — natural end-of-turn.
pub(crate) fn decide_anthropic_stop(
    hit_cancel: bool,
    tool_call_limit_hit: bool,
    hit_tool_use: bool,
    hit_stop_sequence: bool,
    output_tokens: u32,
    max_tokens: u32,
    stops: &[String],
    content: &str,
) -> (&'static str, Option<String>) {
    if hit_cancel {
        return ("stop_sequence", Some(ANTHROPIC_CANCELLED_SENTINEL.to_string()));
    }
    if tool_call_limit_hit {
        return (
            "tool_use",
            Some(ANTHROPIC_TOOL_CALL_LIMIT_SENTINEL.to_string()),
        );
    }
    if hit_tool_use {
        return ("tool_use", None);
    }
    if hit_stop_sequence {
        let matched = stops
            .iter()
            .find(|s| !s.is_empty() && content.contains(s.as_str()))
            .cloned();
        return ("stop_sequence", matched);
    }
    if output_tokens >= max_tokens {
        return ("max_tokens", None);
    }
    ("end_turn", None)
}

async fn messages_blocking(
    serving: ServingModel,
    msgs: Vec<ChatMessage>,
    sampling: SamplingParams,
    model: String,
    id: String,
    input_tokens: u32,
    pre_rendered_prompt: Option<String>,
) -> Response {
    let handle = match serving.try_acquire().await {
        Ok(p) => p,
        Err(be) => return be.into_response(),
    };

    let has_tools = pre_rendered_prompt.is_some();
    let stream_result = match pre_rendered_prompt {
        Some(prompt) => handle.engine.generate(&prompt, &sampling),
        None => handle.engine.chat(&msgs, &sampling),
    };
    let mut stream = match stream_result {
        Ok(s) => s,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };

    let mut content = String::new();
    let mut output_tokens = 0u32;
    let mut hit_stop_sequence = false;
    while let Some(tok) = stream.next().await {
        match tok {
            Ok(t) => {
                content.push_str(&t.text);
                output_tokens += 1;
                // Stop-sequence detection happens inside the engine via the
                // sampler's `stop` field, but the engine stops emission
                // before reporting which sequence hit. We approximate.
                for s in sampling.stop.iter() {
                    if !s.is_empty() && content.contains(s.as_str()) {
                        hit_stop_sequence = true;
                        break;
                    }
                }
            }
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        }
    }

    // Parse `<tool_call>` blocks out of the raw output if tools were
    // requested. Anything outside the blocks becomes a {type:"text"} block;
    // each parsed block becomes a {type:"tool_use", id, name, input}.
    let (content_blocks, hit_tool_use) = if has_tools {
        let (text_remainder, tool_calls) = crate::chat::parse_tool_calls(&content);
        let mut blocks: Vec<ResponseBlock> = Vec::new();
        if let Some(t) = text_remainder {
            if !t.is_empty() {
                blocks.push(ResponseBlock::Text {
                    kind: "text",
                    text: t,
                });
            }
        }
        let mut any_tool = false;
        if let Some(calls) = tool_calls {
            for call in calls {
                any_tool = true;
                let input: Value = serde_json::from_str(&call.function.arguments)
                    .unwrap_or_else(|_| Value::String(call.function.arguments.clone()));
                blocks.push(ResponseBlock::ToolUse {
                    kind: "tool_use",
                    id: call.id,
                    name: call.function.name,
                    input,
                });
            }
        }
        if blocks.is_empty() {
            blocks.push(ResponseBlock::Text {
                kind: "text",
                text: content.clone(),
            });
        }
        (blocks, any_tool)
    } else {
        (
            vec![ResponseBlock::Text {
                kind: "text",
                text: content.clone(),
            }],
            false,
        )
    };

    // Tool-call iteration cap: if the grammar blocked an extra
    // `<tool_call>` opener mid-generation, surface the cap via the
    // `__tool_call_iteration_limit__` sentinel on `stop_sequence`. Same
    // pattern as the OpenAI path uses with `finish_reason: "tool_call_iteration_limit"`.
    let tool_call_limit_hit = handle
        .cpu_engine
        .as_ref()
        .map(|c| c.last_request_stats().tool_call_limit_hit)
        .unwrap_or(false);

    let (stop_reason, stop_sequence_out) = decide_anthropic_stop(
        false, // non-stream path doesn't see cancellation here
        tool_call_limit_hit,
        hit_tool_use,
        hit_stop_sequence,
        output_tokens,
        sampling.max_tokens,
        &sampling.stop,
        &content,
    );

    // Populate the rustllama extension fields on `usage` when a real
    // engine is present. With only the MockEngine (e.g. tests + clients
    // hitting the harness without loading a model), the extension
    // fields stay `None` and the response shape stays minimal.
    let usage = AnthropicUsage {
        input_tokens,
        output_tokens,
        ..Default::default()
    };
    let usage = match handle.cpu_engine.as_ref() {
        Some(cpu) => usage.with_stats(&cpu.last_request_stats()),
        None => usage,
    };

    Json(MessagesResponse {
        id,
        kind: "message",
        role: "assistant",
        model,
        content: content_blocks,
        stop_reason,
        stop_sequence: stop_sequence_out,
        usage,
    })
    .into_response()
}

#[allow(clippy::too_many_arguments)]
async fn messages_stream(
    serving: ServingModel,
    msgs: Vec<ChatMessage>,
    sampling: SamplingParams,
    model: String,
    id: String,
    input_tokens: u32,
    pre_rendered_prompt: Option<String>,
    cancel_guard: crate::CancelGuard,
) -> Response {
    let handle = match serving.try_acquire().await {
        Ok(p) => p,
        Err(be) => return be.into_response(),
    };

    let has_tools = pre_rendered_prompt.is_some();
    let tok_stream = match pre_rendered_prompt {
        Some(prompt) => handle.engine.generate(&prompt, &sampling),
        None => handle.engine.chat(&msgs, &sampling),
    };
    let tok_stream = match tok_stream {
        Ok(s) => s,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };

    // Keep a clone of the per-request fork so the SSE builder can
    // query `last_request_stats().tool_call_limit_hit` at the end of
    // the stream. Must be the handle's cpu_engine, not `serving`'s —
    // in multi-flight, `serving.cpu_engine` is a different fork.
    let cpu_engine_for_stats = handle.cpu_engine.clone();
    let id_for_header = id.clone();
    let event_stream = build_anthropic_sse(
        id,
        model,
        tok_stream,
        handle,
        input_tokens,
        sampling.max_tokens,
        sampling.stop.clone(),
        has_tools,
        cancel_guard,
        cpu_engine_for_stats,
    );
    (
        [("x-rustllama-request-id", id_for_header.as_str())],
        Sse::new(event_stream).keep_alive(KeepAlive::default()),
    )
        .into_response()
}

/// Stream events per Anthropic spec:
///   message_start
///     → content_block_start(text|tool_use, idx)
///     → content_block_delta(text_delta|input_json_delta, idx)*
///     → content_block_stop(idx)
///     [repeat per block]
///   message_delta(stop_reason, usage)
///   message_stop
///
/// When `has_tools` is true, the engine's raw text is parsed via
/// [`StreamingToolCallParser`] and `<tool_call>...</tool_call>` blocks
/// surface as separate `tool_use` content blocks with `input_json_delta`
/// deltas — matching what `anthropic-sdk-*` and Claude Code consume.
#[allow(clippy::too_many_arguments)]
fn build_anthropic_sse(
    id: String,
    model: String,
    mut tok_stream: rustllama_engine::TokenStream,
    permit: crate::PermitGuard,
    input_tokens: u32,
    max_tokens: u32,
    stops: Vec<String>,
    has_tools: bool,
    cancel_guard: crate::CancelGuard,
    cpu_engine: Option<std::sync::Arc<rustllama_engine::CpuEngine>>,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let cancel_flag = cancel_guard.flag.clone();
        // message_start
        let msg_start = json!({
            "type": "message_start",
            "message": {
                "id": id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": Value::Null,
                "stop_sequence": Value::Null,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": 0,
                },
            }
        });
        yield Ok(Event::default().event("message_start").data(msg_start.to_string()));

        let mut emitter = BlockEmitter::new();
        let mut parser = if has_tools { Some(crate::chat::StreamingToolCallParser::new()) } else { None };
        let mut output_tokens = 0u32;
        let mut so_far = String::new();
        let mut hit_stop = false;
        let mut hit_tool_use = false;
        let mut hit_cancel = false;

        while let Some(tok) = tok_stream.next().await {
            if cancel_flag.load(std::sync::atomic::Ordering::Acquire) {
                hit_cancel = true;
                break;
            }
            match tok {
                Ok(t) => {
                    output_tokens += 1;
                    so_far.push_str(&t.text);
                    if let Some(p) = parser.as_mut() {
                        for ev in p.feed(&t.text) {
                            for sse in emitter.handle(ev, &mut hit_tool_use) {
                                yield Ok(sse);
                            }
                        }
                    } else {
                        for sse in emitter.emit_text(&t.text) {
                            yield Ok(sse);
                        }
                    }
                    for s in stops.iter() {
                        if !s.is_empty() && so_far.contains(s.as_str()) {
                            hit_stop = true;
                            break;
                        }
                    }
                    if hit_stop { break; }
                }
                Err(e) => {
                    let err = json!({
                        "type": "error",
                        "error": {"type": "api_error", "message": e.to_string()}
                    });
                    yield Ok(Event::default().event("error").data(err.to_string()));
                    drop(permit);
                    drop(cancel_guard);
                    return;
                }
            }
        }

        // Flush trailing parser state (e.g. unterminated tool_call falls
        // back to text per StreamingToolCallParser::finish). Skip on cancel:
        // we want to terminate as fast as possible.
        if !hit_cancel {
            if let Some(p) = parser {
                for ev in p.finish() {
                    for sse in emitter.handle(ev, &mut hit_tool_use) {
                        yield Ok(sse);
                    }
                }
            }
        }

        // Close the currently-open block.
        for sse in emitter.close_current() {
            yield Ok(sse);
        }

        // message_delta — final stop_reason + cumulative usage.
        // Routes through `decide_anthropic_stop` so cancellation and the
        // tool-call iteration cap surface via their sentinel
        // `stop_sequence` values without crashing strict-enum SDK
        // consumers.
        // Fetch the request stats ONCE so the limit-hit decision and
        // the usage extension fields both read a consistent snapshot.
        let stats = cpu_engine
            .as_ref()
            .map(|c| c.last_request_stats());
        let tool_call_limit_hit = stats
            .as_ref()
            .map(|s| s.tool_call_limit_hit)
            .unwrap_or(false);
        let (stop_reason, stop_sequence_str) = decide_anthropic_stop(
            hit_cancel,
            tool_call_limit_hit,
            hit_tool_use,
            hit_stop,
            output_tokens,
            max_tokens,
            &stops,
            &so_far,
        );
        let stop_sequence = stop_sequence_str
            .map(Value::String)
            .unwrap_or(Value::Null);
        // Build the usage payload, optionally extended with the
        // rustllama performance metrics when a CpuEngine is present.
        // Anthropic-strict SDKs see the extension fields as unknowns
        // and tolerate them; tests in `tests/anthropic_messages.rs`
        // pin both shapes.
        let mut usage_payload = serde_json::Map::new();
        usage_payload.insert("output_tokens".into(), json!(output_tokens));
        if let Some(s) = stats.as_ref() {
            usage_payload.insert("prefill_ms".into(), json!(s.prefill_ms));
            usage_payload.insert("decode_ms".into(), json!(s.decode_ms));
            usage_payload.insert("cache_hit_tokens".into(), json!(s.cache_hit_tokens));
            usage_payload.insert("tokens_prefilled".into(), json!(s.tokens_prefilled));
        }
        let msg_delta = json!({
            "type": "message_delta",
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": stop_sequence,
            },
            "usage": Value::Object(usage_payload),
        });
        yield Ok(Event::default().event("message_delta").data(msg_delta.to_string()));

        // message_stop
        yield Ok(Event::default()
            .event("message_stop")
            .data(json!({"type": "message_stop"}).to_string()));

        drop(permit);
        drop(cancel_guard);
    }
}

/// Tracks which Anthropic content block is currently open and emits the
/// content_block_start / content_block_delta / content_block_stop SSE
/// events as the underlying parser yields its `StreamEvent`s.
struct BlockEmitter {
    state: BlockState,
    next_index: usize,
    /// Header buffered until the matching `ToolCallArgs` arrives — the
    /// parser delivers them as a paired emission, but we want to open
    /// the tool_use block, emit one input_json_delta, and close it in a
    /// single `emit_tool_use` call.
    pending_header: Option<(String, String)>,
}

#[derive(Clone, Copy)]
enum BlockState {
    None,
    Text(usize),
    /// We use the same `index` to close the tool_use block right after
    /// we open it (since our parser delivers Header+Args as a paired
    /// emission). The variant exists for completeness; in practice we
    /// open and close inside the same `handle()` call.
    #[allow(dead_code)]
    ToolUse(usize),
}

impl BlockEmitter {
    fn new() -> Self {
        Self {
            state: BlockState::None,
            next_index: 0,
            pending_header: None,
        }
    }

    /// Ensure a text content block is open; emit its content_block_start
    /// if needed. Then emit a text_delta.
    fn emit_text(&mut self, text: &str) -> Vec<Event> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut evts = Vec::new();
        let idx = match self.state {
            BlockState::Text(idx) => idx,
            BlockState::None => {
                let idx = self.next_index;
                self.next_index += 1;
                self.state = BlockState::Text(idx);
                evts.push(
                    Event::default().event("content_block_start").data(
                        json!({
                            "type": "content_block_start",
                            "index": idx,
                            "content_block": {"type": "text", "text": ""},
                        })
                        .to_string(),
                    ),
                );
                idx
            }
            BlockState::ToolUse(idx) => {
                // Close the tool_use, open a fresh text block.
                evts.push(
                    Event::default()
                        .event("content_block_stop")
                        .data(json!({"type": "content_block_stop", "index": idx}).to_string()),
                );
                let new_idx = self.next_index;
                self.next_index += 1;
                self.state = BlockState::Text(new_idx);
                evts.push(
                    Event::default().event("content_block_start").data(
                        json!({
                            "type": "content_block_start",
                            "index": new_idx,
                            "content_block": {"type": "text", "text": ""},
                        })
                        .to_string(),
                    ),
                );
                new_idx
            }
        };
        evts.push(
            Event::default().event("content_block_delta").data(
                json!({
                    "type": "content_block_delta",
                    "index": idx,
                    "delta": {"type": "text_delta", "text": text},
                })
                .to_string(),
            ),
        );
        evts
    }

    /// Open a tool_use block, emit `input_json_delta` deltas that carry
    /// the args JSON in ~32-byte shards (matching what the real
    /// Anthropic streaming API does — clients concatenate `partial_json`
    /// fragments to assemble the final input). Then close the block.
    /// Closes any open text block first.
    fn emit_tool_use(&mut self, call_id: String, name: String, args: String) -> Vec<Event> {
        let mut evts = Vec::new();
        if let BlockState::Text(idx) | BlockState::ToolUse(idx) = self.state {
            evts.push(
                Event::default()
                    .event("content_block_stop")
                    .data(json!({"type": "content_block_stop", "index": idx}).to_string()),
            );
        }
        let idx = self.next_index;
        self.next_index += 1;
        // content_block_start(tool_use)
        evts.push(
            Event::default().event("content_block_start").data(
                json!({
                    "type": "content_block_start",
                    "index": idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": call_id,
                        "name": name,
                        "input": {},
                    },
                })
                .to_string(),
            ),
        );
        // Shard the args into ~32-byte chunks on UTF-8 char boundaries
        // so split offsets never land mid-codepoint. SDK clients
        // concatenate `partial_json` values, so the split point is
        // semantically arbitrary; we just need each individual delta to
        // be valid UTF-8 on its own.
        for shard in shard_partial_json(&args, 32) {
            evts.push(
                Event::default().event("content_block_delta").data(
                    json!({
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "input_json_delta", "partial_json": shard},
                    })
                    .to_string(),
                ),
            );
        }
        // content_block_stop
        evts.push(
            Event::default()
                .event("content_block_stop")
                .data(json!({"type": "content_block_stop", "index": idx}).to_string()),
        );
        self.state = BlockState::None;
        evts
    }

    fn close_current(&mut self) -> Vec<Event> {
        let mut evts = Vec::new();
        if let BlockState::Text(idx) | BlockState::ToolUse(idx) = self.state {
            evts.push(
                Event::default()
                    .event("content_block_stop")
                    .data(json!({"type": "content_block_stop", "index": idx}).to_string()),
            );
            self.state = BlockState::None;
        }
        evts
    }

    /// Dispatch a [`crate::chat::StreamEvent`]. Pairs of `ToolCallHeader`
    /// + `ToolCallArgs` are buffered into a single `emit_tool_use` call.
    fn handle(
        &mut self,
        ev: crate::chat::StreamEvent,
        hit_tool_use: &mut bool,
    ) -> Vec<Event> {
        use crate::chat::StreamEvent as SE;
        match ev {
            SE::Content(s) => self.emit_text(&s),
            SE::ToolCallHeader { id, name, .. } => {
                // Stash for pairing with ToolCallArgs.
                self.pending_header = Some((id, name));
                Vec::new()
            }
            SE::ToolCallArgs { args, .. } => {
                if let Some((cid, name)) = self.pending_header.take() {
                    *hit_tool_use = true;
                    self.emit_tool_use(cid, name, args)
                } else {
                    Vec::new()
                }
            }
        }
    }
}

// ----- helpers --------------------------------------------------------------

struct SamplingInputs {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    max_tokens: u32,
    stops: Vec<String>,
}

fn build_sampling(inputs: SamplingInputs) -> SamplingParams {
    let mut s = SamplingParams::default();
    if let Some(t) = inputs.temperature {
        s.temperature = t;
    }
    if let Some(p) = inputs.top_p {
        s.top_p = p;
    }
    if let Some(k) = inputs.top_k {
        s.top_k = k;
    }
    s.max_tokens = inputs.max_tokens;
    s.stop = inputs.stops;
    s
}

/// Split `s` into chunks of at most `target` bytes, falling on UTF-8 char
/// boundaries. Returns `[s]` when `s.len() <= target`. Returns `[]` when
/// `s` is empty (no delta needed; Anthropic clients infer empty input).
fn shard_partial_json(s: &str, target: usize) -> Vec<String> {
    if s.is_empty() {
        return Vec::new();
    }
    if s.len() <= target {
        return vec![s.to_string()];
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / target + 1);
    let mut start = 0usize;
    while start < bytes.len() {
        let mut end = (start + target).min(bytes.len());
        // Walk back to the previous char boundary so we never split a
        // codepoint. `is_char_boundary` is O(1).
        while end < bytes.len() && !s.is_char_boundary(end) {
            end -= 1;
        }
        // Safety: `start..end` is guaranteed to span complete codepoints.
        out.push(s[start..end].to_string());
        start = end;
    }
    out
}

fn unix_ts() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Extract just the `event: ...` line of each Event so the test
    /// asserts the SEQUENCE of event types without coupling to JSON
    /// formatting / field ordering. (axum's `Event` doesn't expose the
    /// event field directly; we serialize via Display on the Bytes side
    /// — but since BlockEmitter constructs Events with `.event(name)`,
    /// we can just count them by name in the test by stringifying.)
    fn event_names(events: &[Event]) -> Vec<String> {
        events
            .iter()
            .map(|e| {
                // Event's Debug impl renders newlines as escape sequences
                // (`\n` → literal backslash + 'n'), so we split on the
                // first backslash to truncate the event name cleanly.
                let dbg = format!("{e:?}");
                if let Some(start) = dbg.find("event: ") {
                    let rest = &dbg[start + 7..];
                    let end = rest.find('\\').unwrap_or(rest.len());
                    rest[..end].to_string()
                } else {
                    "(no event)".into()
                }
            })
            .collect()
    }

    #[test]
    fn block_emitter_text_only_opens_one_block() {
        let mut em = BlockEmitter::new();
        let mut events = Vec::new();
        events.extend(em.emit_text("hello "));
        events.extend(em.emit_text("world"));
        events.extend(em.close_current());

        // Expected: block_start, delta, delta, block_stop.
        let names = event_names(&events);
        assert_eq!(
            names,
            vec![
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
            ]
        );
    }

    #[test]
    fn block_emitter_tool_use_after_text_closes_text_first() {
        let mut em = BlockEmitter::new();
        let mut events = Vec::new();
        events.extend(em.emit_text("Let me check. "));
        events.extend(em.emit_tool_use(
            "call_abc".into(),
            "get_weather".into(),
            r#"{"city":"Paris"}"#.into(),
        ));

        let names = event_names(&events);
        // text block_start, text delta, text block_stop, tool block_start,
        // tool delta (input_json_delta), tool block_stop.
        assert_eq!(
            names,
            vec![
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
            ]
        );
    }

    #[test]
    fn block_emitter_handle_pairs_header_and_args() {
        use crate::chat::StreamEvent as SE;
        let mut em = BlockEmitter::new();
        let mut hit = false;
        let mut events = Vec::new();
        // Header alone yields nothing (we wait for paired Args).
        events.extend(em.handle(
            SE::ToolCallHeader {
                index: 0,
                id: "call_xyz".into(),
                name: "get_weather".into(),
            },
            &mut hit,
        ));
        assert!(events.is_empty(), "header alone shouldn't emit");
        assert!(!hit);
        // Args triggers the full tool_use emission.
        events.extend(em.handle(
            SE::ToolCallArgs {
                index: 0,
                args: r#"{"city":"Paris"}"#.into(),
            },
            &mut hit,
        ));
        assert!(hit, "hit_tool_use should be set after Args");
        let names = event_names(&events);
        assert_eq!(
            names,
            vec![
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
            ]
        );
    }

    #[test]
    fn block_emitter_text_after_tool_use_opens_new_text_block() {
        let mut em = BlockEmitter::new();
        let mut events = Vec::new();
        events.extend(em.emit_tool_use(
            "call_1".into(),
            "f".into(),
            "{}".into(),
        ));
        events.extend(em.emit_text("more text"));
        events.extend(em.close_current());

        let names = event_names(&events);
        // tool_start, tool_delta, tool_stop, text_start, text_delta, text_stop.
        assert_eq!(
            names,
            vec![
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
            ]
        );
    }

    #[test]
    fn shard_partial_json_splits_on_char_boundaries() {
        // ASCII: simple length-based split into chunks of ≤4 bytes.
        let chunks = super::shard_partial_json("abcdefghij", 4);
        assert_eq!(chunks, vec!["abcd", "efgh", "ij"]);

        // Empty input → zero chunks (no delta to emit).
        assert!(super::shard_partial_json("", 4).is_empty());

        // Short input fits in one shard.
        assert_eq!(super::shard_partial_json("abc", 4), vec!["abc"]);

        // Multi-byte codepoint must not be split. `é` is 2 bytes, `中`
        // is 3 bytes. With target=3, a naive byte split would land
        // mid-codepoint on the 中; we must walk back to the boundary.
        let s = "ab中cd";
        let chunks = super::shard_partial_json(s, 3);
        for c in &chunks {
            assert!(c.is_char_boundary(0));
            assert!(c.is_char_boundary(c.len()));
        }
        assert_eq!(chunks.concat(), s);
    }

    #[test]
    fn block_emitter_shards_long_tool_use_args() {
        // 100-byte ASCII args → at least 3 deltas with target=32.
        let long_args = format!(
            r#"{{"city":"{}","unit":"celsius"}}"#,
            "A".repeat(70)
        );
        assert!(long_args.len() > 96);
        let mut em = BlockEmitter::new();
        let events = em.emit_tool_use(
            "call_long".into(),
            "get_weather".into(),
            long_args.clone(),
        );
        let names = event_names(&events);
        // Expected: 1× content_block_start, N× content_block_delta,
        // 1× content_block_stop.
        let n_deltas = names
            .iter()
            .filter(|n| *n == "content_block_delta")
            .count();
        assert!(
            n_deltas >= 3,
            "expected ≥3 input_json_delta shards for {}-byte args, got {n_deltas}: {names:?}",
            long_args.len()
        );
        assert_eq!(names.first().map(String::as_str), Some("content_block_start"));
        assert_eq!(names.last().map(String::as_str), Some("content_block_stop"));
    }

    #[test]
    fn anthropic_tools_to_openai_converts_input_schema() {
        let input = json!([{
            "name": "get_weather",
            "description": "Fetch weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }]);
        let out = anthropic_tools_to_openai(input);
        let arr = out.as_array().expect("array");
        assert_eq!(arr.len(), 1);
        let entry = &arr[0];
        assert_eq!(entry["type"], "function");
        let func = &entry["function"];
        assert_eq!(func["name"], "get_weather");
        assert_eq!(func["description"], "Fetch weather");
        // `input_schema` → `parameters` rename.
        assert_eq!(func["parameters"]["type"], "object");
        assert!(func["parameters"]["properties"]["city"].is_object());
    }

    // ----- decide_anthropic_stop ---------------------------------------------

    // ----- content_block_to_text --------------------------------------------

    #[test]
    fn text_block_passes_through() {
        let b = TextBlock {
            kind: "text".into(),
            text: "hello".into(),
            source: None,
            image_url: None,
            cache_control: None,
        };
        assert_eq!(content_block_to_text(b), "hello");
    }

    #[test]
    fn anthropic_image_block_renders_media_type_placeholder() {
        let b = TextBlock {
            kind: "image".into(),
            text: String::new(),
            source: Some(ImageSource {
                kind: "base64".into(),
                media_type: "image/png".into(),
                data: "fake".into(),
                url: String::new(),
            }),
            image_url: None,
            cache_control: None,
        };
        assert_eq!(content_block_to_text(b), "[image: image/png]");
    }

    #[test]
    fn anthropic_image_block_without_media_type_falls_back_to_unknown() {
        let b = TextBlock {
            kind: "image".into(),
            text: String::new(),
            source: None,
            image_url: None,
            cache_control: None,
        };
        assert_eq!(content_block_to_text(b), "[image: unknown]");
    }

    #[test]
    fn openai_image_url_block_renders_placeholder_with_url() {
        let b = TextBlock {
            kind: "image_url".into(),
            text: String::new(),
            source: None,
            image_url: Some(ImageUrl {
                url: "https://example.com/x.png".into(),
                detail: "auto".into(),
            }),
            cache_control: None,
        };
        // Mirror the OpenAI adapter's fix: include the URL so the
        // model can at least reason about file extension / domain.
        assert_eq!(
            content_block_to_text(b),
            "[image: https://example.com/x.png]"
        );
    }

    #[test]
    fn openai_image_url_block_with_empty_url_falls_back_to_generic() {
        let b = TextBlock {
            kind: "image_url".into(),
            text: String::new(),
            source: None,
            image_url: Some(ImageUrl {
                url: String::new(),
                detail: "auto".into(),
            }),
            cache_control: None,
        };
        assert_eq!(content_block_to_text(b), "[image: url]");
    }

    #[test]
    fn openai_image_url_block_without_image_url_field_uses_generic() {
        let b = TextBlock {
            kind: "image_url".into(),
            text: String::new(),
            source: None,
            image_url: None,
            cache_control: None,
        };
        assert_eq!(content_block_to_text(b), "[image: url]");
    }

    // ----- seed determinism: anthropic intentionally lacks the field -------

    #[test]
    fn anthropic_messages_request_silently_ignores_seed_field() {
        // The official Anthropic Messages API doesn't expose a `seed`
        // field. Some SDKs ship "compat shims" that pass seed through
        // anyway — serde should ignore the unknown field rather than
        // 400. Pinning behavior to catch a future drift if we ever
        // add `#[serde(deny_unknown_fields)]`.
        let input = serde_json::json!({
            "model": "rustllama-mock",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "seed": 42,
        });
        let req: super::MessagesRequest =
            serde_json::from_value(input).expect("must accept (and ignore) unknown `seed`");
        // Sanity: the rest of the request still landed correctly.
        assert_eq!(req.max_tokens, 8);
    }

    #[test]
    fn unknown_block_with_text_field_falls_back_to_text() {
        // Forward-compat: if a future client sends an unknown block
        // kind that still carries a `text` field, surface that as a
        // text fallback rather than erroring. Matches real-Anthropic's
        // lenient behavior.
        let b = TextBlock {
            kind: "tool_result".into(),
            text: "result string".into(),
            source: None,
            image_url: None,
            cache_control: None,
        };
        assert_eq!(content_block_to_text(b), "result string");
    }

    // ----- decide_anthropic_stop --------------------------------------------

    #[test]
    fn stop_decision_end_turn_when_nothing_special_happened() {
        let (reason, seq) =
            decide_anthropic_stop(false, false, false, false, 5, 32, &[], "");
        assert_eq!(reason, "end_turn");
        assert!(seq.is_none(), "no stop_sequence on natural end_turn");
    }

    #[test]
    fn stop_decision_max_tokens_when_output_hits_cap() {
        let (reason, seq) =
            decide_anthropic_stop(false, false, false, false, 32, 32, &[], "");
        assert_eq!(reason, "max_tokens");
        assert!(seq.is_none());
    }

    #[test]
    fn stop_decision_stop_sequence_returns_matched_string() {
        // The model's content ends with the configured stop sequence
        // "<|END|>"; the helper should surface that sequence.
        let stops = vec!["<|END|>".to_string()];
        let (reason, seq) = decide_anthropic_stop(
            false,
            false,
            false,
            true,
            10,
            32,
            &stops,
            "hello world<|END|>",
        );
        assert_eq!(reason, "stop_sequence");
        assert_eq!(seq.as_deref(), Some("<|END|>"));
    }

    #[test]
    fn stop_decision_tool_use_no_sentinel_for_normal_tool_call() {
        let (reason, seq) =
            decide_anthropic_stop(false, false, true, false, 5, 32, &[], "");
        assert_eq!(reason, "tool_use");
        assert!(
            seq.is_none(),
            "normal tool_use must NOT carry a stop_sequence sentinel"
        );
    }

    #[test]
    fn stop_decision_tool_call_limit_keeps_tool_use_reason_with_sentinel() {
        // This is the regression-target case for this work: the engine
        // signals tool_call_limit_hit; the adapter keeps stop_reason as
        // the standard "tool_use" (Anthropic enum has no dedicated
        // value) but surfaces __tool_call_iteration_limit__ on
        // stop_sequence so callers can tell capped from normal stops.
        let (reason, seq) = decide_anthropic_stop(
            false, true, true, false, 5, 32, &[], "",
        );
        assert_eq!(reason, "tool_use");
        assert_eq!(
            seq.as_deref(),
            Some(ANTHROPIC_TOOL_CALL_LIMIT_SENTINEL),
            "limit-cap stop_sequence sentinel must surface"
        );
    }

    #[test]
    fn stop_decision_tool_call_limit_fires_even_without_explicit_tool_use_event() {
        // The cap can fire on the FIRST attempted tool open (no
        // hit_tool_use event observed yet). The flag alone is enough.
        let (reason, seq) = decide_anthropic_stop(
            false, true, false, false, 5, 32, &[], "",
        );
        assert_eq!(reason, "tool_use");
        assert_eq!(seq.as_deref(), Some(ANTHROPIC_TOOL_CALL_LIMIT_SENTINEL));
    }

    #[test]
    fn stop_decision_cancel_wins_over_everything_else() {
        // Cancel beats tool_call_limit_hit, hit_tool_use, hit_stop,
        // and max_tokens — the operator aborted, so the response should
        // reflect that, not whatever else was happening.
        let stops = vec!["<|END|>".to_string()];
        let (reason, seq) = decide_anthropic_stop(
            true, true, true, true, 32, 32, &stops, "hello<|END|>",
        );
        assert_eq!(reason, "stop_sequence");
        assert_eq!(seq.as_deref(), Some(ANTHROPIC_CANCELLED_SENTINEL));
    }

    #[test]
    fn stop_decision_limit_beats_plain_tool_use_and_stop_sequence() {
        // When tool_call_limit_hit is true, the limit-cap branch wins
        // over a coincidental stop-sequence match in the buffered text.
        let stops = vec!["</s>".to_string()];
        let (reason, seq) = decide_anthropic_stop(
            false,
            true,
            true,
            true,
            5,
            32,
            &stops,
            "hi</s>",
        );
        assert_eq!(reason, "tool_use");
        assert_eq!(seq.as_deref(), Some(ANTHROPIC_TOOL_CALL_LIMIT_SENTINEL));
    }
}
