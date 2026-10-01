//! `POST /v1/chat/completions` — OpenAI-compatible chat endpoint.
//!
//! Supports both modes:
//!   - non-streaming (`stream=false`, default): single JSON response.
//!   - streaming (`stream=true`): SSE response with `chat.completion.chunk`
//!     events terminated by `data: [DONE]`.

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

use crate::{model_not_found, AppState, StreamOptions, Usage};

#[derive(Debug, Deserialize)]
pub struct ChatRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessageWire>,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    /// Locally-typical sampling threshold (Meister et al. 2022). Range
    /// 0..1; `1.0` (default) disables. Non-OpenAI extension field —
    /// recognized because Ollama and llama.cpp expose it under the
    /// same name.
    pub typical_p: Option<f32>,
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub stop: Vec<String>,
    pub seed: Option<u64>,
    pub repeat_penalty: Option<f32>,
    /// Mirostat sampler mode: `0` disabled (default), `1` = v1, `2` = v2.
    /// Non-OpenAI extension field — supported because Ollama and many
    /// editor integrations expose it.
    pub mirostat: Option<u32>,
    /// Mirostat target surprise (nats). Ignored when `mirostat == 0`.
    pub mirostat_tau: Option<f32>,
    /// Mirostat learning rate. Ignored when `mirostat == 0`.
    pub mirostat_eta: Option<f32>,
    /// Optional OpenAI-style tool definitions. When present, we render the
    /// chat template with `tools` exposed (which Qwen / DeepSeek / Llama-3
    /// chat templates branch on to inject function-calling instructions)
    /// and parse `<tool_call>` blocks from the response.
    pub tools: Option<Value>,
    /// Legacy alias for `tools` (OpenAI's older API).
    pub functions: Option<Value>,
    /// OpenAI `tool_choice`. Controls whether / which tool the model may
    /// call: `null`/absent or `"auto"` → model decides; `"none"` →
    /// tools are shown in the prompt but no call is forced or parsed;
    /// `"required"` → the grammar blocks EOS until ≥1 call is emitted;
    /// `{"type":"function","function":{"name":"X"}}` → forces a call to
    /// exactly `X` (tools + grammar are filtered to it). See
    /// [`parse_tool_choice`].
    pub tool_choice: Option<Value>,
    /// OpenAI-style additive penalties (range −2..2).
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    /// `{"type": "json_object"}` puts the request in JSON mode:
    /// `apply_response_format` prepends a system message instructing
    /// the model to emit JSON only, AND `build_sampling` engages a
    /// `GrammarKind::Json` mask so the bytes are validated at
    /// sampling time. `{"type": "json_schema", "json_schema": {...}}`
    /// additionally enforces the supplied schema's structure (types,
    /// required keys, enum literals). `{"type": "code", "json_schema":
    /// {"language": "..."}}` engages the bracket-balance code
    /// grammar. The prompt nudge + grammar are layered: nudge tells
    /// the model what to put inside; grammar guarantees the bytes
    /// parse + match the structure.
    pub response_format: Option<ResponseFormat>,
    /// Include logprob info per generated token.
    #[serde(default)]
    pub logprobs: bool,
    /// Number of top alternative logprobs to include alongside each
    /// chosen token (0..=20). Implies `logprobs: true`.
    pub top_logprobs: Option<u32>,
    /// OpenAI streaming knob. With `include_usage: true` we emit a final
    /// chunk containing the `usage` block before `[DONE]`.
    pub stream_options: Option<StreamOptions>,
    /// rustllama extension (CLARIFY opt-in). When `Some(true)`, route the
    /// request through the tools path even if the caller sent no
    /// `tools`/`functions`, advertising ONLY the reserved `ask_user` tool
    /// so the model can pause and ask the human a clarifying question with
    /// selectable options instead of guessing. `tool_choice` stays at its
    /// default (`auto`), so the model asks only when it wants to. `None` /
    /// `Some(false)` leaves the plain path byte-identical to before. The
    /// chat frontends (REPL/TUI/GUI) default this on so their UIs can
    /// surface clarifying questions; it's a toggle because it engages the
    /// tool grammar.
    #[serde(default)]
    pub allow_clarify: Option<bool>,
}

#[derive(Debug, Deserialize)]
pub struct ResponseFormat {
    #[serde(rename = "type")]
    pub kind: String,
    /// Optional JSON Schema (OpenAI's `json_schema` extension).
    /// Used by both the prompt nudge AND the
    /// `GrammarKind::JsonSchema` mask — bytes are parser-validated
    /// against this at sampling time. For `kind = "code"` the field
    /// doubles as an options bag (e.g. `{"language": "rust"}`).
    pub json_schema: Option<Value>,
    /// Pattern body for `kind = "regex"`. Anchored to the start of
    /// output via the compiled DFA (see
    /// [`rustllama_engine::GrammarKind::Regex`]); the model must
    /// emit bytes that, taken together from position 0, match this
    /// pattern. Useful for shaped outputs (phone numbers, ids,
    /// fixed enumerations) where JSON shape is overkill.
    #[serde(default)]
    pub pattern: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct ToolCallOut {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: ToolCallFunction,
}

#[derive(Debug, Serialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// JSON-encoded string of the arguments object, per OpenAI spec.
    pub arguments: String,
}

#[derive(Debug, Deserialize, Default)]
pub struct ChatMessageWire {
    pub role: String,
    /// Lenient: absent OR explicit `null` both deserialize to empty
    /// content. OpenAI assistant messages that carry `tool_calls` send
    /// `content: null`, so a strict `OpenAiContent` here would 400 the
    /// whole multi-turn round-trip at deserialization.
    #[serde(default, deserialize_with = "deserialize_content_lenient")]
    pub content: OpenAiContent,
    /// Assistant turn's prior tool calls, echoed back by the client on a
    /// multi-turn tool round-trip. Forwarded verbatim to the chat
    /// template (via the raw-JSON render path) so the template can
    /// re-render the assistant's tool invocation. OpenAI shape:
    /// `[{"id","type":"function","function":{"name","arguments"}}]`.
    #[serde(default)]
    pub tool_calls: Option<Value>,
    /// Links a `role:"tool"` result message back to the `tool_calls[].id`
    /// it answers. Forwarded to the template verbatim.
    #[serde(default)]
    pub tool_call_id: Option<String>,
    /// Tool/function name on a `role:"tool"` result message (some chat
    /// templates key on it). Forwarded to the template verbatim.
    #[serde(default)]
    pub name: Option<String>,
}

/// OpenAI `messages[].content` accepts either a plain string OR an
/// array of typed content blocks (the multimodal payload — GPT-4-Vision
/// shape, also sent by Cursor / Continue / clients that target OpenAI
/// but happen to attach images). Untagged so serde tries both shapes.
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum OpenAiContent {
    Plain(String),
    Blocks(Vec<OpenAiContentBlock>),
}

/// One element of the OpenAI multimodal content array. The discriminator
/// is the `type` field; `text` blocks pass through, `image_url` blocks
/// render as a `[image: url]` placeholder so a text-only model still
/// gets a coherent prompt. Future-compat: unknown kinds with a `text`
/// field fall through to that text (matches the Anthropic adapter).
#[derive(Debug, Deserialize)]
pub struct OpenAiContentBlock {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: String,
    #[allow(dead_code)]
    #[serde(default)]
    pub image_url: Option<OpenAiImageUrl>,
}

/// `image_url` payload. Accepted from the wire so deserialization
/// doesn't fail; v1 doesn't surface the URL to a text-only model.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct OpenAiImageUrl {
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub detail: String,
}

impl OpenAiContent {
    pub(crate) fn into_text(self) -> String {
        match self {
            Self::Plain(s) => s,
            Self::Blocks(blocks) => blocks
                .into_iter()
                .map(openai_content_block_to_text)
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Borrowing accessor for callers that need the rendered text
    /// without consuming the value (e.g. log formatting and tests).
    /// Allocates only when the content is a block array.
    pub(crate) fn text(&self) -> std::borrow::Cow<'_, str> {
        match self {
            Self::Plain(s) => std::borrow::Cow::Borrowed(s.as_str()),
            Self::Blocks(_) => std::borrow::Cow::Owned(self.clone_text()),
        }
    }

    fn clone_text(&self) -> String {
        match self {
            Self::Plain(s) => s.clone(),
            Self::Blocks(blocks) => blocks
                .iter()
                .map(|b| match b.kind.as_str() {
                    "text" => b.text.clone(),
                    "image_url" => image_url_placeholder(b.image_url.as_ref()),
                    _ => b.text.clone(),
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

impl Default for OpenAiContent {
    /// Empty plain content — the fallback when a message omits `content`
    /// or sends `content: null` (assistant-with-tool_calls turns do).
    fn default() -> Self {
        Self::Plain(String::new())
    }
}

/// Deserialize `messages[].content` leniently: an absent field OR an
/// explicit `null` both yield empty [`OpenAiContent`], while a string or
/// block-array deserializes normally. Needed so an OpenAI assistant
/// message that carries `tool_calls` (and sets `content: null`) parses
/// instead of 400-ing the entire multi-turn tool round-trip.
fn deserialize_content_lenient<'de, D>(d: D) -> std::result::Result<OpenAiContent, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt = Option::<OpenAiContent>::deserialize(d)?;
    Ok(opt.unwrap_or_default())
}

impl From<&str> for OpenAiContent {
    fn from(s: &str) -> Self {
        Self::Plain(s.to_string())
    }
}

impl From<String> for OpenAiContent {
    fn from(s: String) -> Self {
        Self::Plain(s)
    }
}

/// Render one OpenAI content block as plain text. Mirrors the Anthropic
/// adapter's `content_block_to_text` so the two surfaces behave
/// symmetrically when a multimodal-aware client targets either API.
pub(crate) fn openai_content_block_to_text(b: OpenAiContentBlock) -> String {
    match b.kind.as_str() {
        "text" => b.text,
        "image_url" => image_url_placeholder(b.image_url.as_ref()),
        // Forward-compat: unknown block kind with a text field surfaces
        // that text rather than erroring.
        _ => b.text,
    }
}

/// Render an `image_url` content block as a placeholder string when
/// the model is non-vision. Previously this returned the literal text
/// `"[image: url]"` — useless to the model, since the actual URL was
/// dropped. Now we include the URL in the placeholder
/// (`"[image: <url>]"`) so coding models that reason about file
/// extensions / domains (e.g. "the screenshot at github.com/...
/// pixel art") get useful signal. The placeholder text alone never
/// causes the model to actually fetch the URL; that's gated on the
/// future VLM support landing.
fn image_url_placeholder(image_url: Option<&OpenAiImageUrl>) -> String {
    match image_url {
        Some(iu) if !iu.url.is_empty() => format!("[image: {}]", iu.url),
        _ => "[image: url]".to_string(),
    }
}

impl From<ChatMessageWire> for ChatMessage {
    /// Lossy conversion: image_url blocks become placeholder text and
    /// the bytes are dropped. Used by call sites that don't yet route
    /// through the V-4 decoder (anthropic adapter, ollama adapter,
    /// any test that constructs a ChatMessageWire and immediately
    /// flattens). The OpenAI `/v1/chat/completions` path uses the
    /// fallible [`ChatMessageWire::try_into_with_image_bytes`] instead
    /// so the decoded bytes reach `ChatMessage::images`.
    fn from(m: ChatMessageWire) -> Self {
        Self {
            role: m.role,
            content: m.content.into_text(),
            images: Vec::new(),
        }
    }
}

/// Walk a vector of wire chat messages and produce engine
/// [`ChatMessage`]s with any embedded `image_url` blocks decoded via
/// [`crate::image_url::decode_image_url`]. Returns a ready-to-emit
/// HTTP 400 response on decode failure so the caller doesn't have to
/// hand-roll the error envelope.
fn wire_messages_to_engine(
    msgs: Vec<ChatMessageWire>,
) -> std::result::Result<Vec<ChatMessage>, Response> {
    let mut out = Vec::with_capacity(msgs.len());
    for (idx, m) in msgs.into_iter().enumerate() {
        match m.try_into_with_image_bytes() {
            Ok(cm) => out.push(cm),
            Err(e) => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    axum::Json(serde_json::json!({
                        "error": {
                            "message": format!(
                                "messages[{idx}].content.image_url: {e}"
                            ),
                            "type": "invalid_request_error",
                            "param": format!("messages[{idx}].content"),
                            "code": "invalid_image_url",
                        }
                    })),
                )
                    .into_response())
            }
        }
    }
    Ok(out)
}

impl ChatMessageWire {
    /// Convert to engine [`ChatMessage`], decoding any embedded
    /// `image_url` content blocks as raw bytes attached to
    /// [`ChatMessage::images`]. Text content (including the
    /// placeholder strings for image positions) is preserved so a
    /// vision-aware engine can scan the prompt for placeholder tokens
    /// and splice the projected features at those positions.
    ///
    /// Returns `Err(ImageUrlError)` for any malformed `image_url`
    /// block — the chat handler maps that to HTTP 400 with the
    /// error message in the OpenAI error envelope.
    pub(crate) fn try_into_with_image_bytes(
        self,
    ) -> std::result::Result<ChatMessage, crate::image_url::ImageUrlError> {
        let role = self.role;
        match self.content {
            OpenAiContent::Plain(s) => Ok(ChatMessage {
                role,
                content: s,
                images: Vec::new(),
            }),
            OpenAiContent::Blocks(blocks) => {
                let mut text_parts: Vec<String> = Vec::with_capacity(blocks.len());
                let mut images: Vec<Vec<u8>> = Vec::new();
                for b in blocks {
                    match b.kind.as_str() {
                        "text" => text_parts.push(b.text),
                        "image_url" => {
                            // Decode the URL via V-4 and attach bytes;
                            // also keep the placeholder marker in the
                            // text so the splice step can find it.
                            match b.image_url.as_ref() {
                                Some(iu) if !iu.url.is_empty() => {
                                    let bytes = crate::image_url::decode_image_url(&iu.url)?;
                                    images.push(bytes);
                                    text_parts.push(format!("[image: {}]", iu.url));
                                }
                                _ => {
                                    // Pathological: image_url block
                                    // without a populated url. Render
                                    // the generic placeholder; no
                                    // bytes attached.
                                    text_parts.push("[image: url]".to_string());
                                }
                            }
                        }
                        // Forward-compat for unknown block kinds with
                        // a text field — surface it through.
                        _ => text_parts.push(b.text),
                    }
                }
                Ok(ChatMessage {
                    role,
                    content: text_parts.join(""),
                    images,
                })
            }
        }
    }
}

#[derive(Debug, Serialize)]
pub struct ChatResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    /// OpenAI-compat: stable identifier for the backend config.
    /// Changes when the server version, the active model id, or the
    /// KV dtype changes — so editor clients (Aider, Continue) can
    /// detect "the server I'm talking to silently swapped under me"
    /// mid-conversation. Format: `fp_<12-hex-chars>`.
    pub system_fingerprint: String,
}


#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessageOut,
    pub finish_reason: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

#[derive(Debug, Serialize)]
pub struct ChatMessageOut {
    pub role: &'static str,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallOut>>,
    /// rustllama extension: parsed SEARCH/REPLACE blocks emitted by the
    /// model when `response_format: {"type":"diff"}` was set. Each entry
    /// is one apply-edits hunk. The raw text remains in `content`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diffs: Option<Vec<DiffBlock>>,
}

/// One SEARCH/REPLACE hunk parsed out of a diff-mode response. Matches
/// the aider de-facto format that most editor agents consume.
#[derive(Debug, Serialize, Clone, PartialEq, Eq)]
pub struct DiffBlock {
    /// File path declared on the line immediately above `<<<<<<< SEARCH`.
    /// `None` when the model omits the filename (a malformed hunk we
    /// emit anyway so the client can decide how strict to be).
    pub file: Option<String>,
    pub search: String,
    pub replace: String,
}

/// Per-OpenAI chat.completions logprobs shape:
///   {"content": [{ token, logprob, bytes, top_logprobs: [...] }, ...]}
#[derive(Debug, Serialize)]
pub struct ChoiceLogprobs {
    pub content: Vec<TokenLogprobOut>,
}

#[derive(Debug, Serialize)]
pub struct TokenLogprobOut {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
    pub top_logprobs: Vec<TopLogprobOut>,
}

#[derive(Debug, Serialize)]
pub struct TopLogprobOut {
    pub token: String,
    pub logprob: f32,
    pub bytes: Vec<u8>,
}

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(mut req): Json<ChatRequest>,
) -> Response {
    // Validate response_format up front. Currently this only gates
    // on `type=regex` patterns (other shapes are accepted as-is and
    // unsupported keywords just pass through to the grammar
    // dispatch where they're handled or ignored).
    if let Err(resp) = validate_response_format(req.response_format.as_ref()) {
        return resp;
    }
    // response_format: prepend instructions to the messages before any
    // template rendering. This works for plain chat, streaming, and tools.
    req.messages = apply_response_format(req.messages, &req.response_format);

    // Tools / functions request: needs explicit prompt building (via the
    // tokenizer chat template with `tools` exposed) and incremental parsing
    // of `<tool_call>...</tool_call>` blocks.
    //
    // `allow_clarify` (CLARIFY opt-in) ALSO takes the tools path even with
    // no caller-supplied tools: the tools normalizer below turns the empty
    // set into just `[ask_user]` (see `with_reserved_ask_user_tool`), so the
    // model can ask a clarifying question. A clarify-only request skips the
    // shell env hint (it advertises no shell/command tool the hint would
    // help). `tool_choice` stays at its default (`auto`) — the model asks
    // only when it wants to.
    let has_tools = req.tools.is_some() || req.functions.is_some();
    let clarify_only = !has_tools && req.allow_clarify == Some(true);
    if has_tools || clarify_only {
        if has_tools {
            // Inject the server host-environment context so the model picks
            // the right shell dialect on a shell/command tool call. Gated by
            // `[server].tool_environment_hint` (default on); appends to an
            // existing system message, else prepends a new one.
            req.messages = maybe_inject_env_hint(req.messages);
        }
        if req.stream {
            return chat_stream_with_tools(state, req).await;
        }
        return chat_with_tools(state, req).await;
    }

    if req.stream {
        chat_stream(state, req).await
    } else {
        chat_blocking(state, req).await
    }
}

/// Resolved OpenAI `tool_choice` directive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ToolChoice {
    /// Model decides whether to call a tool (OpenAI default). The tool
    /// grammar is engaged so any call is well-formed, but EOS is never
    /// blocked — the model may answer in prose instead.
    Auto,
    /// Tools are rendered into the prompt so the model knows they exist,
    /// but the grammar is NOT engaged and the output is NOT parsed for
    /// tool calls — a faithful OpenAI `"none"`.
    None,
    /// The model must emit at least one tool call: the grammar blocks
    /// EOS (`min_completed = 1`) until one completes.
    Required,
    /// The model must call exactly this function: tools + grammar are
    /// filtered to it and `min_completed = 1`.
    Function(String),
}

/// Map an OpenAI `tool_choice` value to a [`ToolChoice`]. Unknown /
/// unparseable shapes fall back to [`ToolChoice::Auto`] (the OpenAI
/// default) rather than erroring — a lenient surface for the many
/// clients that send slightly-off shapes.
pub(crate) fn parse_tool_choice(v: &Option<Value>) -> ToolChoice {
    match v {
        None | Some(Value::Null) => ToolChoice::Auto,
        Some(Value::String(s)) => match s.as_str() {
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            "auto" => ToolChoice::Auto,
            _ => ToolChoice::Auto,
        },
        Some(Value::Object(o)) => {
            // `{"type":"function","function":{"name":"X"}}`.
            let name = o
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str());
            match name {
                Some(n) if !n.is_empty() => ToolChoice::Function(n.to_string()),
                _ => ToolChoice::Auto,
            }
        }
        _ => ToolChoice::Auto,
    }
}

/// Filter a normalized `tools` array down to the single function named
/// `name` (used to force a specific tool). Non-array values pass
/// through unchanged.
fn filter_tools_to_name(tools: &Value, name: &str) -> Value {
    match tools {
        Value::Array(arr) => Value::Array(
            arr.iter()
                .filter(|t| {
                    t.get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(|n| n.as_str())
                        == Some(name)
                })
                .cloned()
                .collect(),
        ),
        other => other.clone(),
    }
}

/// Name of the reserved "ask the user" tool (CLARIFY). Injected alongside
/// the caller's tools on every tools request so the model can pause and ask
/// the human a question with selectable options instead of guessing. It's a
/// perfectly ordinary function schema — so [`parse_tool_choice`], the
/// tool-call grammar, and [`parse_tool_calls`] treat it like any other tool.
/// The streaming terminus in [`build_sse_stream_with_tools`] is the only
/// place that special-cases it: a completed `ask_user` call is re-emitted as
/// an `ask_user` SSE delta (finish_reason `"ask_user"`) rather than a
/// `tool_calls` one, so clients render a chooser instead of trying to run it.
const ASK_USER_TOOL_NAME: &str = "ask_user";

/// The OpenAI-shape tool schema for [`ASK_USER_TOOL_NAME`].
fn ask_user_tool_schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": ASK_USER_TOOL_NAME,
            "description": "Ask the user a clarifying question and let them choose from a list of \
                            options. Call this instead of guessing when you're missing a decision \
                            or detail you need to continue.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The question to put to the user."
                    },
                    "options": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "The selectable answers to offer the user."
                    }
                },
                "required": ["prompt", "options"]
            }
        }
    })
}

/// Append the reserved [`ask_user`](ask_user_tool_schema) tool to a
/// normalized `tools` array. A non-array value (e.g. `Value::Null` from a
/// caller that sent `functions: []`) becomes a one-element array so the
/// reserved tool is always advertised on the tools path.
fn with_reserved_ask_user_tool(tools: Value) -> Value {
    match tools {
        Value::Array(mut arr) => {
            arr.push(ask_user_tool_schema());
            Value::Array(arr)
        }
        _ => Value::Array(vec![ask_user_tool_schema()]),
    }
}

/// Pull `(prompt, options)` out of a completed `ask_user` call's argument
/// JSON string for the `ask_user` SSE delta. Tolerant of missing / mistyped
/// fields — a missing `prompt` yields `""` and missing / non-array `options`
/// yields `[]` — so the terminal chunk is always well-formed even if the
/// model drifted from the schema.
fn parse_ask_user_args(args: &str) -> (String, Vec<String>) {
    let v: Value = serde_json::from_str(args).unwrap_or(Value::Null);
    let prompt = v
        .get("prompt")
        .and_then(|p| p.as_str())
        .unwrap_or("")
        .to_string();
    let options = v
        .get("options")
        .and_then(|o| o.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    (prompt, options)
}

/// Build the raw-JSON message array the tools render path feeds to the
/// chat template. Each element carries `role` + flattened `content`,
/// plus the multi-turn round-trip fields (`tool_calls`, `tool_call_id`,
/// `name`) when the wire message set them — so the template can
/// re-render prior tool turns verbatim.
fn wire_messages_to_json(messages: &[ChatMessageWire]) -> Vec<Value> {
    messages
        .iter()
        .map(|m| {
            let mut obj = serde_json::Map::new();
            obj.insert("role".into(), Value::String(m.role.clone()));
            obj.insert(
                "content".into(),
                Value::String(m.content.text().into_owned()),
            );
            if let Some(tc) = m.tool_calls.as_ref() {
                obj.insert("tool_calls".into(), tc.clone());
            }
            if let Some(id) = m.tool_call_id.as_ref() {
                obj.insert("tool_call_id".into(), Value::String(id.clone()));
            }
            if let Some(name) = m.name.as_ref() {
                obj.insert("name".into(), Value::String(name.clone()));
            }
            Value::Object(obj)
        })
        .collect()
}

async fn chat_with_tools(state: AppState, req: ChatRequest) -> Response {
    let Some(serving) = state.resolve(req.model.as_deref()).await else {
        return model_not_found(req.model.as_deref());
    };
    // Two-phase admission: tokenizer access + chat-template render +
    // prompt-token count all happen against the Arc-shared CPU engine
    // BEFORE the gate wait, so the work overlaps the current
    // request's decode on queued arrivals.
    let permit = match serving.try_admit() {
        Ok(p) => p,
        Err(be) => return be.into_response(),
    };
    let shared_cpu = match serving.cpu_engine.as_ref() {
        Some(e) => e.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "tools/functions requires a real engine (not the MockEngine)",
            )
                .into_response();
        }
    };
    let tokenizer = match shared_cpu.tokenizer() {
        Some(t) => t,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "tools/functions requires a model with a tokenizer",
            )
                .into_response();
        }
    };

    let mut sampling = build_sampling(&req);
    let model_id = req.model.clone().unwrap_or_else(|| serving.model_id.clone());
    let tool_choice = parse_tool_choice(&req.tool_choice);

    // Normalize tools array — accept either `tools: [...]` (new) or
    // `functions: [...]` (legacy).
    let tools_value = match (&req.tools, &req.functions) {
        (Some(t), _) => t.clone(),
        (_, Some(funcs)) => {
            // Convert legacy functions array → tools array shape.
            let arr = funcs.as_array().cloned().unwrap_or_default();
            let wrapped: Vec<Value> = arr
                .into_iter()
                .map(|f| json!({ "type": "function", "function": f }))
                .collect();
            Value::Array(wrapped)
        }
        _ => Value::Null,
    };
    // Advertise the reserved `ask_user` tool (CLARIFY) alongside the
    // caller's tools. Injected before the `tool_choice: {function}` filter
    // so forcing a specific function still drops it (the model must call the
    // one the caller pinned, not ask instead).
    let tools_value = with_reserved_ask_user_tool(tools_value);
    // `tool_choice: {function}` forces one tool: filter the rendered
    // tools (and, below, the grammar) down to it so the prompt only
    // advertises that function.
    let tools_value = match &tool_choice {
        ToolChoice::Function(name) => filter_tools_to_name(&tools_value, name),
        _ => tools_value,
    };

    // Engage the tool-call stream grammar (mirrors the streaming path).
    // Skipped for `tool_choice: "none"` (the model answers in prose,
    // still aware of the tools) and when the user already pinned a
    // `response_format` grammar (more specific intent wins). Required /
    // a forced function set `min_completed = 1` so EOS is blocked until
    // at least one call is emitted.
    if sampling.grammar.is_none() && !matches!(tool_choice, ToolChoice::None) {
        let schemas = extract_tool_schemas(&tools_value);
        if !schemas.is_empty() {
            let min_completed = match tool_choice {
                ToolChoice::Required | ToolChoice::Function(_) => 1,
                _ => 0,
            };
            sampling.grammar = Some(rustllama_engine::GrammarKind::ToolCallStream {
                schemas_by_name: schemas,
                min_completed,
            });
        }
    }

    // Render the prompt using the tokenizer's chat template, with `tools`
    // exposed to the template. Qwen / DeepSeek / Llama-3 templates branch
    // on this to insert function-calling system instructions
    // automatically. The raw-JSON message path forwards any prior-turn
    // `tool_calls` / `tool_call_id` / `name` fields so multi-turn tool
    // round-trips re-render correctly. Image blocks are flattened to
    // `[image: url]` placeholders inside `content` so text-only models
    // still receive a coherent prompt (real VLM support is a v1.x item).
    let json_msgs = wire_messages_to_json(&req.messages);
    let prompt = match tokenizer.render_chat_messages_json(&json_msgs, true, Some(&tools_value)) {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("render chat template: {e}"),
            )
                .into_response();
        }
    };

    // Count prompt tokens up-front — the rendered prompt is already in scope.
    let prompt_tokens = tokenizer
        .encode(&prompt, tokenizer.add_bos_token())
        .map(|ids| ids.len() as u32)
        .unwrap_or(0);

    // Drive the streaming engine so we can collect per-token logprobs
    // alongside the text. This used to call `cpu.generate_text(...)`
    // — a one-shot convenience that returns only the joined string —
    // which is why `logprobs: None` was hardcoded. Walking the token
    // stream costs nothing extra (the underlying engine streams either
    // way) and unlocks the OpenAI `logprobs` field for the tools path.
    let want_logprobs = sampling.logprobs.is_some();
    // Now wait on the per-model gate — all pre-engine work above
    // already happened. The handle's fork is the per-request engine
    // for state-mutating ops (generate, last_request_stats).
    let handle = match permit.acquire_gate().await {
        Ok(h) => h,
        Err(be) => return be.into_response(),
    };
    let cpu = match handle.cpu_engine.as_ref() {
        Some(e) => e.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "tools/functions requires a real engine (not the MockEngine)",
            )
                .into_response();
        }
    };
    let mut tok_stream = match handle.engine.generate(&prompt, &sampling) {
        Ok(s) => s,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };
    let mut raw = String::new();
    let mut logprob_tokens: Vec<TokenLogprobOut> = Vec::new();
    let mut completion_tokens = 0u32;
    while let Some(tok) = tok_stream.next().await {
        match tok {
            Ok(t) => {
                if want_logprobs {
                    if let Some(lp) = t.logprobs.as_ref() {
                        logprob_tokens.push(make_token_logprob_out(&t.text, lp, Some(&cpu)));
                    }
                }
                raw.push_str(&t.text);
                completion_tokens += 1;
            }
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        }
    }

    // Capture per-request stats while the gate permit is still held.
    let stats = cpu.last_request_stats();

    // Parse model output: anything inside `<tool_call>{...}</tool_call>`
    // (plus the Llama-3 `<|python_tag|>`, Mistral `[TOOL_CALLS]`, and
    // bare-JSON fallbacks) becomes a structured tool_call; the rest
    // becomes content. For `tool_choice: "none"` we skip parsing
    // entirely — the whole output is returned as plain content.
    let (content, tool_calls) = if matches!(tool_choice, ToolChoice::None) {
        let trimmed = raw.trim();
        (
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            },
            None,
        )
    } else {
        parse_tool_calls(&raw)
    };
    // Tool-call iteration cap: when the grammar blocked an extra
    // `<tool_call>` opener mid-generation, surface that as the finish
    // reason so clients can spot capped runs.
    let finish_reason: &'static str = if stats.tool_call_limit_hit {
        "tool_call_iteration_limit"
    } else if tool_calls.is_some() {
        "tool_calls"
    } else {
        "stop"
    };

    let logprobs_out = if want_logprobs {
        Some(ChoiceLogprobs {
            content: logprob_tokens,
        })
    } else {
        None
    };

    let fingerprint = crate::system_fingerprint(
        &state.version,
        &model_id,
        kv_dtype_label(&cpu),
    );
    Json(ChatResponse {
        id: request_id(),
        object: "chat.completion",
        created: unix_ts(),
        model: model_id,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageOut {
                role: "assistant",
                content,
                tool_calls,
                diffs: None,
            },
            finish_reason,
            logprobs: logprobs_out,
        }],
        usage: Some(Usage::new(prompt_tokens, completion_tokens).with_stats(&stats)),
        system_fingerprint: fingerprint,
    })
    .into_response()
}

/// Streaming variant of `chat_with_tools` — pre-renders the prompt with the
/// chat template's `tools` branch, drives a token stream, and emits OpenAI-
/// compatible `tool_calls` deltas as `<tool_call>` blocks complete.
async fn chat_stream_with_tools(state: AppState, req: ChatRequest) -> Response {
    let Some(serving) = state.resolve(req.model.as_deref()).await else {
        return model_not_found(req.model.as_deref());
    };
    // Two-phase admission — tokenization + chat-template render with
    // tools all happen pre-gate against the Arc-shared CPU engine.
    let permit = match serving.try_admit() {
        Ok(p) => p,
        Err(be) => return be.into_response(),
    };
    let shared_cpu = match serving.cpu_engine.as_ref() {
        Some(e) => e.clone(),
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "tools/functions requires a real engine (not the MockEngine)",
            )
                .into_response();
        }
    };
    let tokenizer = match shared_cpu.tokenizer() {
        Some(t) => t,
        None => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                "tools/functions requires a model with a tokenizer",
            )
                .into_response();
        }
    };

    // Normalize tools array (same as non-streaming path).
    let tools_value = match (&req.tools, &req.functions) {
        (Some(t), _) => t.clone(),
        (_, Some(funcs)) => {
            let arr = funcs.as_array().cloned().unwrap_or_default();
            let wrapped: Vec<Value> = arr
                .into_iter()
                .map(|f| json!({ "type": "function", "function": f }))
                .collect();
            Value::Array(wrapped)
        }
        _ => Value::Null,
    };
    let tool_choice = parse_tool_choice(&req.tool_choice);
    // Advertise the reserved `ask_user` tool (CLARIFY) — see the
    // non-streaming path for the rationale on ordering vs. the filter.
    let tools_value = with_reserved_ask_user_tool(tools_value);
    // `tool_choice: {function}` forces one tool — filter the rendered
    // tools (and the grammar below) down to it.
    let tools_value = match &tool_choice {
        ToolChoice::Function(name) => filter_tools_to_name(&tools_value, name),
        _ => tools_value,
    };

    // Render the prompt via the raw-JSON message path so prior-turn
    // `tool_calls` / `tool_call_id` / `name` fields reach the template
    // (multi-turn tool round-trips). Image blocks are flattened to
    // `[image: url]` placeholders inside `content` so text-only models
    // still receive a coherent prompt (real VLM support is a v1.x item).
    let json_msgs = wire_messages_to_json(&req.messages);
    let prompt = match tokenizer.render_chat_messages_json(&json_msgs, true, Some(&tools_value)) {
        Ok(p) => p,
        Err(e) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("render chat template: {e}"),
            )
                .into_response();
        }
    };

    let mut sampling = build_sampling(&req);
    // Engage the tool-call stream grammar when concrete per-tool
    // schemas are available. The grammar guarantees that any
    // `<tool_call>...</tool_call>` block contains well-formed JSON
    // matching the called function's parameter schema. Skipped for
    // `tool_choice: "none"` (prose answer) and when the user already
    // pinned a `response_format` grammar (more specific intent wins).
    // Required / a forced function set `min_completed = 1` so EOS is
    // blocked until at least one call is emitted.
    if sampling.grammar.is_none() && !matches!(tool_choice, ToolChoice::None) {
        let schemas = extract_tool_schemas(&tools_value);
        if !schemas.is_empty() {
            let min_completed = match tool_choice {
                ToolChoice::Required | ToolChoice::Function(_) => 1,
                _ => 0,
            };
            sampling.grammar = Some(rustllama_engine::GrammarKind::ToolCallStream {
                schemas_by_name: schemas,
                min_completed,
            });
        }
    }
    let model = req.model.unwrap_or_else(|| serving.model_id.clone());
    let id = request_id();
    let cancel_guard = state.register_cancel(&id);
    let created = unix_ts();
    let include_usage = req
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);
    let prompt_tokens = tokenizer
        .encode(&prompt, tokenizer.add_bos_token())
        .map(|ids| ids.len() as u32)
        .unwrap_or(0);

    // Wait on the gate now that all pre-engine work is done.
    let handle = match permit.acquire_gate().await {
        Ok(h) => h,
        Err(be) => return be.into_response(),
    };

    let tok_stream = match handle.engine.generate(&prompt, &sampling) {
        Ok(s) => s,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };

    // The cpu_engine passed downstream is used for `last_request_stats`
    // — that lives on the per-request fork in multi-flight, so it
    // must come from the handle (not from serving).
    let cpu_for_stats = handle.cpu_engine.clone();
    let id_for_header = id.clone();
    let fingerprint = crate::system_fingerprint(
        &state.version,
        &model,
        cpu_for_stats.as_deref().map(kv_dtype_label).unwrap_or(""),
    );
    // `tool_choice: "none"`: tools were rendered into the prompt so the
    // model knows they exist, but we neither engaged the grammar nor
    // parse `<tool_call>` blocks — stream plain content exactly like the
    // non-tools path so no tool_calls deltas are ever emitted.
    if matches!(tool_choice, ToolChoice::None) {
        let event_stream = build_sse_stream(
            id,
            created,
            model,
            tok_stream,
            handle,
            cpu_for_stats,
            prompt_tokens,
            include_usage,
            false, // diff_mode
            cancel_guard,
            sampling.max_tokens,
            fingerprint,
        );
        return (
            [("x-rustllama-request-id", id_for_header.as_str())],
            Sse::new(event_stream).keep_alive(KeepAlive::default()),
        )
            .into_response();
    }
    let event_stream = build_sse_stream_with_tools(
        id,
        created,
        model,
        tok_stream,
        handle,
        prompt_tokens,
        include_usage,
        cancel_guard,
        cpu_for_stats,
        sampling.max_tokens,
        fingerprint,
    );
    (
        [("x-rustllama-request-id", id_for_header.as_str())],
        Sse::new(event_stream).keep_alive(KeepAlive::default()),
    )
        .into_response()
}

#[allow(clippy::too_many_arguments)]
fn build_sse_stream_with_tools(
    id: String,
    created: u64,
    model: String,
    mut tok_stream: rustllama_engine::TokenStream,
    permit: crate::PermitGuard,
    prompt_tokens: u32,
    include_usage: bool,
    cancel_guard: crate::CancelGuard,
    cpu: Option<std::sync::Arc<rustllama_engine::CpuEngine>>,
    max_tokens: u32,
    fingerprint: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let cancel_flag = cancel_guard.flag.clone();
        let fp = fingerprint.as_str();
        // Initial role chunk.
        yield Ok(Event::default().data(
            chunk_json_with_fp(&id, created, &model, json!({"role": "assistant"}), None, None, fp).to_string()
        ));

        let mut parser = StreamingToolCallParser::new();
        let mut any_tool_call = false;
        let mut finish_reason: &'static str = "stop";
        let mut completion_tokens = 0u32;
        // CLARIFY: a completed `ask_user` call is swallowed here (its
        // header/args are NOT streamed as a normal tool_calls delta) and
        // re-emitted at the terminus as a single `ask_user` delta. The
        // header and its args arrive as consecutive parser events, so we
        // latch the call index off the header and buffer the args that
        // follow it.
        let mut ask_user_seen = false;
        let mut ask_user_id: Option<String> = None;
        let mut ask_user_index: Option<usize> = None;
        let mut ask_user_args = String::new();

        while let Some(item) = tok_stream.next().await {
            if cancel_flag.load(std::sync::atomic::Ordering::Acquire) {
                finish_reason = "cancelled";
                break;
            }
            match item {
                Ok(tok) => {
                    completion_tokens += 1;
                    let events = parser.feed(&tok.text);
                    // Per-token logprobs are attached to the FIRST
                    // Content event the parser produces from this
                    // token, then cleared. Multiple content events
                    // from one token (rare) only carry logprobs once,
                    // matching OpenAI's "one logprobs entry per
                    // generated token" semantics. Tool-call deltas
                    // never carry logprobs (the OpenAI spec is fuzzy
                    // on this — tool_call internals are part of the
                    // grammar, not user-visible token choices).
                    let mut pending_lp = tok
                        .logprobs
                        .as_ref()
                        .map(|lp| token_logprob_chunk_json(&tok.text, lp, cpu.as_deref()));
                    for ev in events {
                        match ev {
                            StreamEvent::Content(s) if !s.is_empty() => {
                                let lp = pending_lp.take();
                                yield Ok(Event::default().data(
                                    chunk_json_with_fp(&id, created, &model,
                                               json!({"content": s}), None, lp, fp).to_string()
                                ));
                            }
                            StreamEvent::Content(_) => {}
                            StreamEvent::ToolCallHeader { index, id: cid, name } => {
                                if name == ASK_USER_TOOL_NAME {
                                    // CLARIFY: swallow the header; the call is
                                    // re-emitted as an `ask_user` delta at the
                                    // terminus (see below).
                                    ask_user_seen = true;
                                    ask_user_index = Some(index);
                                    ask_user_id = Some(cid);
                                } else {
                                    any_tool_call = true;
                                    let delta = json!({
                                        "tool_calls": [{
                                            "index": index,
                                            "id": cid,
                                            "type": "function",
                                            "function": { "name": name, "arguments": "" },
                                        }]
                                    });
                                    yield Ok(Event::default().data(
                                        chunk_json_with_fp(&id, created, &model, delta, None, None, fp).to_string()
                                    ));
                                }
                            }
                            StreamEvent::ToolCallArgs { index, args } => {
                                if ask_user_index == Some(index) {
                                    // Buffer the `ask_user` args for the terminus.
                                    ask_user_args.push_str(&args);
                                } else {
                                    let delta = json!({
                                        "tool_calls": [{
                                            "index": index,
                                            "function": { "arguments": args },
                                        }]
                                    });
                                    yield Ok(Event::default().data(
                                        chunk_json_with_fp(&id, created, &model, delta, None, None, fp).to_string()
                                    ));
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    let err = json!({"error": {"message": e.to_string()}});
                    yield Ok(Event::default().data(err.to_string()));
                    finish_reason = "error";
                    break;
                }
            }
        }

        // Flush any trailing pending content / unterminated tool_call.
        if finish_reason != "error" && finish_reason != "cancelled" {
            for ev in parser.finish() {
                match ev {
                    StreamEvent::Content(s) if !s.is_empty() => {
                        yield Ok(Event::default().data(
                            chunk_json_with_fp(&id, created, &model,
                                       json!({"content": s}), None, None, fp).to_string()
                        ));
                    }
                    StreamEvent::Content(_) => {}
                    StreamEvent::ToolCallHeader { index, id: cid, name } => {
                        if name == ASK_USER_TOOL_NAME {
                            // CLARIFY: swallow the header (see the streaming
                            // loop above); re-emitted at the terminus.
                            ask_user_seen = true;
                            ask_user_index = Some(index);
                            ask_user_id = Some(cid);
                        } else {
                            any_tool_call = true;
                            let delta = json!({
                                "tool_calls": [{
                                    "index": index,
                                    "id": cid,
                                    "type": "function",
                                    "function": { "name": name, "arguments": "" },
                                }]
                            });
                            yield Ok(Event::default().data(
                                chunk_json_with_fp(&id, created, &model, delta, None, None, fp).to_string()
                            ));
                        }
                    }
                    StreamEvent::ToolCallArgs { index, args } => {
                        if ask_user_index == Some(index) {
                            ask_user_args.push_str(&args);
                        } else {
                            let delta = json!({
                                "tool_calls": [{
                                    "index": index,
                                    "function": { "arguments": args },
                                }]
                            });
                            yield Ok(Event::default().data(
                                chunk_json_with_fp(&id, created, &model, delta, None, None, fp).to_string()
                            ));
                        }
                    }
                }
            }

            if ask_user_seen {
                // CLARIFY takes precedence over tool_calls: the model paused
                // to ask the user, so this isn't a normal tool-call terminus.
                finish_reason = "ask_user";
            } else if any_tool_call {
                finish_reason = "tool_calls";
            }
        }
        // Termination-reason precedence (highest first):
        //   1. cancelled    — operator aborted (set in the loop above)
        //   2. error        — engine returned an error
        //   3. ask_user     — the model called the reserved CLARIFY tool
        //   4. tool_calls   — the response contains tool_use blocks
        //   5. tool_call_iteration_limit — the grammar capped recursion
        //   6. length       — output_tokens reached max_tokens cap
        //   7. stop         — natural end of stream
        //
        // 1 and 2 are already latched in `finish_reason` before this
        // point; 3 and 4 were set during the parser-flush pass. Apply 5 and
        // 6 here on top of the default "stop" — but never over `ask_user`,
        // which is a clean, user-actionable terminus.
        let stats = cpu.as_deref().map(|c| c.last_request_stats());
        if !ask_user_seen && stats.as_ref().map(|s| s.tool_call_limit_hit).unwrap_or(false) {
            finish_reason = "tool_call_iteration_limit";
        } else if finish_reason == "stop" && completion_tokens >= max_tokens {
            // Only apply the length-cap label when we'd otherwise be
            // reporting "stop" — tool_calls / cancel / error take
            // precedence even when the token count happened to match.
            finish_reason = "length";
        }

        // Terminal chunk. For a CLARIFY it carries the parsed question +
        // options on `delta.ask_user` (a ride-along on the existing chunk
        // shape, mirroring how tool_calls attach); otherwise the usual
        // empty delta.
        let final_delta = if ask_user_seen {
            let (prompt, options) = parse_ask_user_args(&ask_user_args);
            json!({
                "ask_user": {
                    "id": ask_user_id.clone().unwrap_or_default(),
                    "kind": "clarify",
                    "prompt": prompt,
                    "options": options,
                }
            })
        } else {
            json!({})
        };
        yield Ok(Event::default().data(
            chunk_json_with_fp(&id, created, &model, final_delta, Some(finish_reason), None, fp).to_string()
        ));
        if include_usage {
            yield Ok(Event::default().data(
                usage_chunk_json_with_stats(
                    &id, created, &model, prompt_tokens, completion_tokens, stats.as_ref()
                ).to_string()
            ));
        }
        yield Ok(Event::default().data("[DONE]"));
        drop(permit);
        drop(cancel_guard);
    }
}

/// Final SSE chunk used when `stream_options.include_usage` is set. The
/// chunk has an empty `choices` array (per spec) and a populated `usage`.
fn usage_chunk_json_with_stats(
    id: &str,
    created: u64,
    model: &str,
    prompt_tokens: u32,
    completion_tokens: u32,
    stats: Option<&rustllama_engine::RequestStats>,
) -> Value {
    let mut usage = json!({
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    });
    if let Some(s) = stats {
        let map = usage.as_object_mut().unwrap();
        map.insert("prefill_ms".into(), json!(s.prefill_ms));
        map.insert("decode_ms".into(), json!(s.decode_ms));
        map.insert("tokens_prefilled".into(), json!(s.tokens_prefilled));
        map.insert("cache_hit_tokens".into(), json!(s.cache_hit_tokens));
    }
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": usage,
    })
}

/// Events produced by [`StreamingToolCallParser`] as text arrives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum StreamEvent {
    Content(String),
    ToolCallHeader {
        index: usize,
        id: String,
        name: String,
    },
    ToolCallArgs {
        index: usize,
        args: String,
    },
}

/// Incrementally parses a stream of text tokens, splitting it into plaintext
/// content and tool-call blocks. Token boundaries are not aligned with the
/// markers, so the parser holds back trailing characters that could be the
/// start of a marker.
///
/// Recognizes the same tool-call encodings as the non-streaming
/// [`parse_tool_calls`], so the streaming and blocking paths agree:
///   - the canonical Qwen / DeepSeek `<tool_call>...</tool_call>` block
///     (fully incremental — content around it streams live);
///   - the Llama-3.1 `<|python_tag|>` and Mistral `[TOOL_CALLS]` markers
///     (content BEFORE the marker streams live; everything after is buffered
///     and recovered at [`finish`](Self::finish) via `parse_tool_calls`, since
///     the JSON payload only resolves once the whole tail is in hand);
///   - a bare-JSON / ```json-fenced tool call that IS the whole message
///     (detected when the stream starts with `{` / `[` / a code fence; the
///     output is buffered and recovered at `finish`).
///
/// The marker/bare-JSON tail forms defer to `finish` on purpose — their
/// framing can't be validated mid-stream — but the shared recovery keeps
/// their result identical to the blocking path. Since this parser is only
/// used on tool-enabled requests, buffering a JSON-leading response until
/// `finish` is safe: a tools request emitting bare JSON is almost always a
/// tool call.
pub(crate) struct StreamingToolCallParser {
    state: ParserState,
    /// Outside a block: text pending content emission. Inside `<tool_call>`:
    /// JSON body so far. In the alt / maybe-JSON tail states: the accumulated
    /// text parsed at `finish`.
    buf: String,
    next_call_index: usize,
    /// Set once the first non-whitespace char is seen, so the "stream starts
    /// with JSON / a fence" detection only fires at the very start.
    started: bool,
    /// The alt-format marker (`<|python_tag|>` / `[TOOL_CALLS]`) that put the
    /// parser into `AltToEnd`, so `finish` can rebuild the original text.
    alt_marker: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ParserState {
    Outside,
    /// Inside a `<tool_call>...</tool_call>` block.
    Inside,
    /// After a `<|python_tag|>` / `[TOOL_CALLS]` marker — accumulate the rest
    /// of the stream; parse it at `finish` via the shared recovery.
    AltToEnd,
    /// The stream began with `{` / `[` / a ``` fence — accumulate everything
    /// and, at `finish`, run `parse_tool_calls`: emit tool calls if it
    /// recovers any, else flush the buffer as plain content.
    MaybeJson,
}

impl StreamingToolCallParser {
    const OPEN: &'static str = "<tool_call>";
    const CLOSE: &'static str = "</tool_call>";
    const PYTAG: &'static str = "<|python_tag|>";
    const MISTRAL: &'static str = "[TOOL_CALLS]";

    pub(crate) fn new() -> Self {
        Self {
            state: ParserState::Outside,
            buf: String::new(),
            next_call_index: 0,
            started: false,
            alt_marker: "",
        }
    }

    pub(crate) fn feed(&mut self, text: &str) -> Vec<StreamEvent> {
        self.buf.push_str(text);
        let mut events = Vec::new();
        loop {
            match self.state {
                ParserState::Outside => {
                    // At the very start, route a JSON- / fence-leading stream
                    // into `MaybeJson` so a bare-JSON or fenced tool call
                    // (which carries no `<tool_call>` framing) is recovered
                    // whole at `finish` rather than streamed out as content.
                    if !self.started {
                        match first_non_ws(&self.buf) {
                            // Only whitespace so far — wait for a real char.
                            None => break,
                            Some(c) => {
                                self.started = true;
                                if c == '{' || c == '[' || c == '`' {
                                    self.state = ParserState::MaybeJson;
                                    continue;
                                }
                            }
                        }
                    }
                    // Find the earliest recognized open marker.
                    let hit = [Self::OPEN, Self::PYTAG, Self::MISTRAL]
                        .into_iter()
                        .filter_map(|m| self.buf.find(m).map(|p| (p, m)))
                        .min_by_key(|(p, _)| *p);
                    if let Some((p, marker)) = hit {
                        let before = self.buf[..p].to_string();
                        if !before.is_empty() {
                            events.push(StreamEvent::Content(before));
                        }
                        let rest = self.buf[p + marker.len()..].to_string();
                        self.buf = rest;
                        if marker == Self::OPEN {
                            self.state = ParserState::Inside;
                        } else {
                            // `<|python_tag|>` / `[TOOL_CALLS]`: the JSON tail
                            // is recovered at `finish`.
                            self.alt_marker = marker;
                            self.state = ParserState::AltToEnd;
                        }
                        continue;
                    }
                    // No complete marker — flush content up to a safe boundary,
                    // holding back any suffix that could still become one of
                    // the markers once more tokens arrive.
                    let holdback = longest_suffix_that_is_prefix_of_any(
                        &self.buf,
                        &[Self::OPEN, Self::PYTAG, Self::MISTRAL],
                    );
                    if holdback < self.buf.len() {
                        let cut = self.buf.len() - holdback;
                        let emit = self.buf[..cut].to_string();
                        if !emit.is_empty() {
                            events.push(StreamEvent::Content(emit));
                        }
                        self.buf = self.buf[cut..].to_string();
                    }
                    break;
                }
                ParserState::Inside => {
                    if let Some(p) = self.buf.find(Self::CLOSE) {
                        let body = self.buf[..p].trim().to_string();
                        let rest = self.buf[p + Self::CLOSE.len()..].to_string();
                        let index = self.next_call_index;
                        match parse_streaming_tool_call_body(&body, index) {
                            Some((id, name, args)) => {
                                events.push(StreamEvent::ToolCallHeader {
                                    index,
                                    id,
                                    name,
                                });
                                events.push(StreamEvent::ToolCallArgs { index, args });
                                self.next_call_index += 1;
                            }
                            None => {
                                // Couldn't parse — re-emit the raw block as
                                // content so the user at least sees it.
                                let fallback = format!("{}{}{}", Self::OPEN, body, Self::CLOSE);
                                events.push(StreamEvent::Content(fallback));
                            }
                        }
                        self.buf = rest;
                        self.state = ParserState::Outside;
                        continue;
                    }
                    // No CLOSE yet — wait for more tokens.
                    break;
                }
                // Alt-marker tail + maybe-JSON: accumulate to end of stream;
                // both are resolved in `finish`.
                ParserState::AltToEnd | ParserState::MaybeJson => break,
            }
        }
        events
    }

    pub(crate) fn finish(mut self) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        match self.state {
            ParserState::Outside => {
                if !self.buf.is_empty() {
                    events.push(StreamEvent::Content(std::mem::take(&mut self.buf)));
                }
            }
            ParserState::Inside => {
                // Unterminated — emit the open tag + accumulated body as
                // plain content so the caller at least sees the partial.
                let partial = format!("{}{}", Self::OPEN, self.buf);
                events.push(StreamEvent::Content(partial));
            }
            ParserState::AltToEnd => {
                // Rebuild the original alt-format text (marker + tail) and
                // reuse the non-streaming recovery so `<|python_tag|>` /
                // `[TOOL_CALLS]` parse identically to the blocking path.
                let full = format!("{}{}", self.alt_marker, self.buf);
                events.extend(recover_streaming_tool_calls(&full, &mut self.next_call_index));
            }
            ParserState::MaybeJson => {
                let full = std::mem::take(&mut self.buf);
                events.extend(recover_streaming_tool_calls(&full, &mut self.next_call_index));
            }
        }
        events
    }
}

/// The first non-whitespace `char` of `s`, or `None` when `s` is empty or all
/// whitespace. Used to decide whether a stream begins with a JSON value / code
/// fence (→ [`ParserState::MaybeJson`]).
fn first_non_ws(s: &str) -> Option<char> {
    s.chars().find(|c| !c.is_whitespace())
}

/// Run the shared non-streaming [`parse_tool_calls`] over a buffered tail and
/// turn the result into streaming events: any leading content first, then a
/// `ToolCallHeader` + `ToolCallArgs` pair per recovered call. When no call is
/// recovered, the whole text is flushed as content so nothing is lost.
fn recover_streaming_tool_calls(raw: &str, next_index: &mut usize) -> Vec<StreamEvent> {
    let (content, calls) = parse_tool_calls(raw);
    let mut events = Vec::new();
    match calls {
        Some(calls) if !calls.is_empty() => {
            if let Some(c) = content {
                if !c.is_empty() {
                    events.push(StreamEvent::Content(c));
                }
            }
            for call in calls {
                let index = *next_index;
                *next_index += 1;
                events.push(StreamEvent::ToolCallHeader {
                    index,
                    id: call.id,
                    name: call.function.name,
                });
                events.push(StreamEvent::ToolCallArgs {
                    index,
                    args: call.function.arguments,
                });
            }
        }
        _ => {
            if !raw.is_empty() {
                events.push(StreamEvent::Content(raw.to_string()));
            }
        }
    }
    events
}

fn parse_streaming_tool_call_body(body: &str, index: usize) -> Option<(String, String, String)> {
    let v: Value = serde_json::from_str(body).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    let arguments = v.get("arguments").cloned().unwrap_or(Value::Null);
    let args_str = match arguments {
        Value::String(s) => s,
        other => other.to_string(),
    };
    let id = format!("call_{:x}_{index}", unix_ts());
    Some((id, name, args_str))
}

/// Length (in bytes) of the longest suffix of `s` that is a (strict) prefix
/// of `marker`. Used to hold back trailing characters that could turn into a
/// complete marker once more tokens arrive.
fn longest_suffix_that_is_prefix_of(s: &str, marker: &str) -> usize {
    let max = s.len().min(marker.len().saturating_sub(1));
    for n in (1..=max).rev() {
        let cut = s.len() - n;
        if s.is_char_boundary(cut) && marker.starts_with(&s[cut..]) {
            return n;
        }
    }
    0
}

/// Longest suffix of `s` that is a prefix of ANY of `markers` — the multi-
/// marker generalization used by [`StreamingToolCallParser`] to hold back a
/// tail that could still become `<tool_call>`, `<|python_tag|>`, or
/// `[TOOL_CALLS]`.
fn longest_suffix_that_is_prefix_of_any(s: &str, markers: &[&str]) -> usize {
    markers
        .iter()
        .map(|m| longest_suffix_that_is_prefix_of(s, m))
        .max()
        .unwrap_or(0)
}

/// Pull tool calls out of the model's text output and return
/// `(remaining_text, tool_calls)`. The remaining text is the pieces of
/// output that were not part of any tool call (trimmed).
///
/// The primary, canonical shape is the Qwen / DeepSeek
/// `<tool_call>{...}</tool_call>` block. When NO such block yields a
/// call, this ALSO tries — in order — the alternate encodings other
/// model families emit (see [`parse_alt_tool_call_formats`]):
///   (a) Llama-3.1 `<|python_tag|>` + JSON object(s),
///   (b) Mistral `[TOOL_CALLS]` + a JSON array,
///   (c) the entire trimmed output as a JSON object / array of
///       tool-call objects (optionally inside a ```json fence).
/// If none match, the output is returned unchanged as plain content.
pub(crate) fn parse_tool_calls(raw: &str) -> (Option<String>, Option<Vec<ToolCallOut>>) {
    const OPEN: &str = "<tool_call>";
    const CLOSE: &str = "</tool_call>";

    let mut calls: Vec<ToolCallOut> = Vec::new();
    let mut remaining = String::with_capacity(raw.len());
    let mut cursor = 0;
    while let Some(rel) = raw[cursor..].find(OPEN) {
        let open = cursor + rel;
        remaining.push_str(&raw[cursor..open]);
        let body_start = open + OPEN.len();
        let close = match raw[body_start..].find(CLOSE) {
            Some(r) => body_start + r,
            None => {
                // Unterminated — keep the open tag and everything after it
                // as content; the caller can surface it as a partial.
                remaining.push_str(&raw[open..]);
                cursor = raw.len();
                break;
            }
        };
        let body = raw[body_start..close].trim();
        if let Some(call) = parse_tool_call_body(body) {
            calls.push(call);
        } else {
            // Couldn't parse — include the raw block as content so the user
            // at least sees what came out.
            remaining.push_str(&raw[open..close + CLOSE.len()]);
        }
        cursor = close + CLOSE.len();
    }
    remaining.push_str(&raw[cursor..]);

    let trimmed = remaining.trim();
    // Primary path found at least one `<tool_call>` block: return it.
    if !calls.is_empty() {
        let content = if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        };
        return (content, Some(calls));
    }

    // No `<tool_call>` block yielded a call — try the alternate formats.
    if let Some((content, alt_calls)) = parse_alt_tool_call_formats(raw) {
        if !alt_calls.is_empty() {
            return (content, Some(alt_calls));
        }
    }

    // Nothing matched — the whole output is plain content.
    let content = if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    };
    (content, None)
}

/// Fallback tool-call extraction for model families that DON'T use the
/// Qwen `<tool_call>` markers. Tried (in order) only when the primary
/// `<tool_call>` scan found nothing. Returns `(leading_content, calls)`
/// on the first shape that yields ≥1 call, else `None`.
fn parse_alt_tool_call_formats(raw: &str) -> Option<(Option<String>, Vec<ToolCallOut>)> {
    // (a) Llama-3.1 `<|python_tag|>` — one or more JSON objects follow,
    //     whitespace / comma / newline separated, to end of string.
    const PYTAG: &str = "<|python_tag|>";
    if let Some(p) = raw.find(PYTAG) {
        let before = raw[..p].trim();
        let after = raw[p + PYTAG.len()..].trim();
        let calls = parse_json_object_sequence(after);
        if !calls.is_empty() {
            let content = (!before.is_empty()).then(|| before.to_string());
            return Some((content, calls));
        }
    }

    // (b) Mistral `[TOOL_CALLS]` — a JSON array of tool-call objects.
    const MISTRAL: &str = "[TOOL_CALLS]";
    if let Some(p) = raw.find(MISTRAL) {
        let before = raw[..p].trim();
        let after = strip_json_fence(raw[p + MISTRAL.len()..].trim());
        if let Ok(v) = serde_json::from_str::<Value>(after) {
            let calls = tool_calls_from_json_value(&v, false);
            if !calls.is_empty() {
                let content = (!before.is_empty()).then(|| before.to_string());
                return Some((content, calls));
            }
        }
    }

    // (c) The entire trimmed output as a JSON object or array of
    //     tool-call objects, optionally inside a ```json fence. Strict
    //     here (each object must carry an args field) so a plain data
    //     object that merely happens to have a `name` isn't misread as
    //     a call.
    let body = strip_json_fence(raw.trim());
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        let calls = tool_calls_from_json_value(&v, true);
        if !calls.is_empty() {
            return Some((None, calls));
        }
    }

    None
}

/// Convert a parsed JSON value (object or array) into tool calls. When
/// `strict`, each object must look like a tool call (name + an
/// `arguments`/`parameters` field, or a `function` wrapper) — used for
/// the "whole output is JSON" heuristic to avoid false positives on
/// ordinary data. When not strict, any object with a usable `name` is
/// accepted (the surrounding marker already signalled intent).
fn tool_calls_from_json_value(v: &Value, strict: bool) -> Vec<ToolCallOut> {
    let conv = |item: &Value| -> Option<ToolCallOut> {
        if strict && !looks_like_tool_call(item) {
            return None;
        }
        tool_call_from_value(item)
    };
    match v {
        Value::Array(arr) => arr.iter().filter_map(conv).collect(),
        Value::Object(_) => conv(v).into_iter().collect(),
        _ => Vec::new(),
    }
}

/// Heuristic: does this JSON object have the shape of a tool call —
/// a `name` plus an args field (`arguments` / `parameters`), possibly
/// under a `function` wrapper?
fn looks_like_tool_call(v: &Value) -> bool {
    let inner = v.get("function").filter(|f| f.is_object()).unwrap_or(v);
    inner.get("name").and_then(|n| n.as_str()).is_some()
        && (inner.get("arguments").is_some() || inner.get("parameters").is_some())
}

/// Parse one or more JSON objects from `s`, tolerating whitespace /
/// newline separation, a bracketed array, or comma-separated objects.
/// Returns the tool calls recovered (may be empty).
fn parse_json_object_sequence(s: &str) -> Vec<ToolCallOut> {
    let s = strip_json_fence(s.trim());
    if s.is_empty() {
        return Vec::new();
    }
    // A bracketed JSON array.
    if s.starts_with('[') {
        if let Ok(v) = serde_json::from_str::<Value>(s) {
            return tool_calls_from_json_value(&v, false);
        }
    }
    // Comma-separated objects (`{...}, {...}`) — wrapping in brackets
    // yields a valid array. Also covers the single-object case
    // (`[{...}]`). Whitespace / newline separated objects (no commas)
    // make an invalid array, so those fall through to the stream parser.
    if let Ok(v) = serde_json::from_str::<Value>(&format!("[{s}]")) {
        let calls = tool_calls_from_json_value(&v, false);
        if !calls.is_empty() {
            return calls;
        }
    }
    // Whitespace / newline separated object stream (`{...}\n{...}`).
    let mut out = Vec::new();
    let stream = serde_json::Deserializer::from_str(s).into_iter::<Value>();
    for item in stream {
        match item {
            Ok(v) => {
                if let Some(c) = tool_call_from_value(&v) {
                    out.push(c);
                }
            }
            Err(_) => break,
        }
    }
    out
}

/// Strip a surrounding ```json ... ``` (or bare ``` ... ```) fence if
/// present, returning the inner body; otherwise returns `s` unchanged.
fn strip_json_fence(s: &str) -> &str {
    let s = s.trim();
    let Some(rest) = s.strip_prefix("```") else {
        return s;
    };
    // Drop an optional language tag (`json`) right after the fence.
    let rest = rest.strip_prefix("json").unwrap_or(rest);
    let rest = rest.trim();
    let rest = rest.strip_suffix("```").unwrap_or(rest);
    rest.trim()
}

/// Extract every aider-style SEARCH/REPLACE block from a model response.
/// Tolerates extra prose between blocks — we just scan for the markers.
///
/// Format:
/// ```text
///     path/to/file.ext     ← optional, on the line preceding `<<<<<<< SEARCH`
///     <<<<<<< SEARCH
///     ...old text...
///     =======
///     ...new text...
///     >>>>>>> REPLACE
/// ```
///
/// Returns `None` if no complete block was found. Malformed blocks
/// (e.g. missing `=======` or `>>>>>>> REPLACE`) are skipped silently.
pub(crate) fn parse_diff_blocks(raw: &str) -> Option<Vec<DiffBlock>> {
    // Find the next marker that starts with `prefix` (7 angle-brackets)
    // then matches `tag` (SEARCH/REPLACE) with any whitespace between.
    // Returns `(start, end_exclusive)` of the marker within `raw[from..]`,
    // translated to absolute offsets. Small-model outputs frequently drop
    // the space (e.g. `<<<<<<<SEARCH`) so we tolerate any number of
    // intermediate whitespace chars.
    fn find_marker(raw: &str, from: usize, prefix: &str, tag: &str) -> Option<(usize, usize)> {
        let bytes = raw.as_bytes();
        let mut search_from = from;
        loop {
            let rel = raw[search_from..].find(prefix)?;
            let start = search_from + rel;
            // Skip whitespace (NOT newlines — markers are single-line).
            let mut p = start + prefix.len();
            while p < bytes.len() && (bytes[p] == b' ' || bytes[p] == b'\t') {
                p += 1;
            }
            if raw[p..].starts_with(tag) {
                return Some((start, p + tag.len()));
            }
            // No match here — keep scanning past this prefix occurrence.
            search_from = start + prefix.len();
        }
    }

    let mut out: Vec<DiffBlock> = Vec::new();
    let mut cursor = 0usize;
    while let Some((open_at, after_open_marker)) = find_marker(raw, cursor, "<<<<<<<", "SEARCH") {
        // Filename: the last non-empty line strictly before OPEN. Tolerate
        // surrounding markdown fences (```py, etc.) and blank lines by
        // walking backwards.
        let file = raw[..open_at]
            .lines()
            .rev()
            .map(str::trim)
            .find(|line| {
                !line.is_empty()
                    && !line.starts_with("```")
                    && !line.starts_with("//")
                    && !line.starts_with("# ")
            })
            .map(str::to_string);

        // Body starts after the newline that follows the OPEN marker.
        let after_open = match raw[after_open_marker..].find('\n') {
            Some(n) => after_open_marker + n + 1,
            None => break,
        };

        // SEP marker: at least 7 `=` followed by newline. We accept any
        // ≥ 3 run, since small models sometimes shorten it; ≥ 3 still
        // unambiguously identifies the separator in practice.
        let sep_at = match find_sep_line(raw, after_open) {
            Some(p) => p,
            None => break,
        };
        let after_sep = match raw[sep_at..].find('\n') {
            Some(n) => sep_at + n + 1,
            None => break,
        };

        // CLOSE marker — same tolerance as OPEN.
        let (close_at, _) = match find_marker(raw, after_sep, ">>>>>>>", "REPLACE") {
            Some(t) => t,
            None => break,
        };

        let search = raw[after_open..sep_at]
            .trim_end_matches('\n')
            .to_string();
        let replace = raw[after_sep..close_at]
            .trim_end_matches('\n')
            .to_string();

        out.push(DiffBlock {
            file,
            search,
            replace,
        });
        // Skip past the REPLACE keyword + any trailing whitespace on
        // the marker line so the next iteration doesn't re-detect this
        // block.
        let next = raw[close_at..]
            .find('\n')
            .map(|n| close_at + n + 1)
            .unwrap_or(raw.len());
        cursor = next;
    }

    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Locate a line consisting of 3+ `=` characters (possibly with trailing
/// whitespace). Returns the byte offset of the first `=` on that line.
fn find_sep_line(raw: &str, from: usize) -> Option<usize> {
    let mut line_start = from;
    loop {
        let line_end = raw[line_start..]
            .find('\n')
            .map(|n| line_start + n)
            .unwrap_or(raw.len());
        let line = &raw[line_start..line_end];
        let trimmed = line.trim();
        if trimmed.len() >= 3 && trimmed.bytes().all(|b| b == b'=') {
            // Return offset of the first `=` (skip leading whitespace).
            let leading_ws = line.len() - line.trim_start().len();
            return Some(line_start + leading_ws);
        }
        if line_end >= raw.len() {
            return None;
        }
        line_start = line_end + 1;
    }
}

/// Parse the JSON body of a `<tool_call>` block into a `ToolCallOut`.
/// Delegates to [`tool_call_from_value`] so it accepts the same shape
/// variants (`arguments` OR `parameters`, and a `{"function":{...}}`
/// wrapper).
fn parse_tool_call_body(body: &str) -> Option<ToolCallOut> {
    let v: Value = serde_json::from_str(body).ok()?;
    tool_call_from_value(&v)
}

/// Convert one parsed tool-call JSON object into a [`ToolCallOut`].
/// Tolerant of the shapes different model families emit:
///   - `{"name": ..., "arguments": {...}}` — Qwen / OpenAI native,
///   - `{"name": ..., "parameters": {...}}` — Llama-3.1 `python_tag`,
///   - `{"function": {"name": ..., "arguments"/"parameters": ...}}` —
///     the OpenAI tool-call wrapper, unwrapped one level.
/// `arguments` may be a JSON object OR a pre-stringified JSON string;
/// both round-trip to the OpenAI string form.
fn tool_call_from_value(v: &Value) -> Option<ToolCallOut> {
    // Unwrap a `{"function": {...}}` wrapper when present.
    let inner = v.get("function").filter(|f| f.is_object()).unwrap_or(v);
    let name = inner.get("name")?.as_str()?.to_string();
    let arguments = inner
        .get("arguments")
        .or_else(|| inner.get("parameters"))
        .cloned()
        .unwrap_or(Value::Null);
    let arguments_str = match arguments {
        Value::String(s) => s,
        other => other.to_string(),
    };
    Some(ToolCallOut {
        id: format!("call_{:x}", unix_ts()),
        kind: "function",
        function: ToolCallFunction {
            name,
            arguments: arguments_str,
        },
    })
}

/// Walk a normalized OpenAI `tools` array
/// (`[{"type":"function","function":{"name":"...","parameters":{...}}}]`)
/// and pull out the `(name, parameters)` pairs as a per-name schema
/// map. Schemas with no `parameters` field fall back to
/// [`Schema::Any`] — the grammar still validates JSON shape, just
/// without per-argument constraints.
pub(crate) fn extract_tool_schemas(
    tools: &Value,
) -> std::collections::BTreeMap<String, rustllama_engine::grammar::Schema> {
    let mut out = std::collections::BTreeMap::new();
    let Value::Array(arr) = tools else {
        return out;
    };
    for t in arr {
        // `t` is `{"type":"function","function":{"name":..., "parameters":...}}`.
        let function = match t.get("function") {
            Some(f) => f,
            None => continue,
        };
        let Some(name) = function.get("name").and_then(|n| n.as_str()) else {
            continue;
        };
        let schema = function
            .get("parameters")
            .map(rustllama_engine::grammar::Schema::from_json_value)
            .unwrap_or(rustllama_engine::grammar::Schema::Any);
        out.insert(name.to_string(), schema);
    }
    out
}

fn build_sampling(req: &ChatRequest) -> SamplingParams {
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
    // logprobs: `top_logprobs` implies `logprobs: true` per OpenAI spec.
    if req.logprobs || req.top_logprobs.is_some() {
        s.logprobs = Some(req.top_logprobs.unwrap_or(0));
    }
    // response_format: `json_object` engages the plain JSON grammar.
    // `json_schema` additionally consults the supplied schema (under
    // `response_format.json_schema.schema`, per the OpenAI shape) so
    // the bytes are guaranteed schema-valid, not just well-formed JSON.
    // The system-message nudge from `apply_response_format` still
    // applies — it tells the model what to PUT in the JSON; the grammar
    // makes sure the bytes parse + match the structure.
    if let Some(fmt) = &req.response_format {
        match fmt.kind.as_str() {
            "json_object" => {
                s.grammar = Some(rustllama_engine::GrammarKind::Json);
            }
            "json_schema" => {
                let schema_value = fmt
                    .json_schema
                    .as_ref()
                    .and_then(|js| js.get("schema").cloned());
                if let Some(sv) = schema_value {
                    let schema = rustllama_engine::grammar::Schema::from_json_value(&sv);
                    s.grammar = Some(rustllama_engine::GrammarKind::JsonSchema { schema });
                } else {
                    // No schema body — fall back to plain JSON validity.
                    s.grammar = Some(rustllama_engine::GrammarKind::Json);
                }
            }
            "code" => {
                // Token-level code-syntax constraint. The `language`
                // hint rides on `response_format.json_schema` (we
                // reuse the field as a generic options bag rather
                // than introduce a new one; clients have set
                // precedent for this with the OpenAI extension
                // pattern).
                let language = fmt
                    .json_schema
                    .as_ref()
                    .and_then(|js| js.get("language"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                s.grammar = Some(rustllama_engine::GrammarKind::Code { language });
            }
            "regex" => {
                // Regex-constrained output. Pattern must be
                // non-empty; an empty / missing pattern silently
                // falls back to "no grammar" (treated as a client
                // omission rather than a hard error so legacy
                // clients that send `{"type": "regex"}` without
                // the body still get a usable response). The
                // request-boundary validation in
                // `validate_response_format` rejects malformed
                // patterns with a 400 before we reach here.
                if let Some(p) = fmt.pattern.as_ref().filter(|p| !p.is_empty()) {
                    s.grammar = Some(rustllama_engine::GrammarKind::Regex {
                        pattern: p.clone(),
                    });
                }
            }
            _ => {}
        }
    }
    s
}

/// Validate the request's `response_format` at the boundary. Returns
/// an HTTP response on a malformed regex pattern (400 with the
/// underlying error); `Ok(())` for every other shape (json_object /
/// json_schema / code / unrecognized).
///
/// Called before the sampler is built so a bad pattern doesn't reach
/// the engine — defense in depth alongside the warn-and-drop path
/// inside `build_grammar_mask_from_kind`.
pub(crate) fn validate_response_format(
    fmt: Option<&ResponseFormat>,
) -> std::result::Result<(), axum::response::Response> {
    use axum::http::StatusCode;
    use axum::response::IntoResponse;
    let Some(rf) = fmt else { return Ok(()) };
    if rf.kind != "regex" {
        return Ok(());
    }
    let Some(p) = rf.pattern.as_ref().filter(|p| !p.is_empty()) else {
        return Err((
            StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({
                "error": "response_format.type=regex requires a non-empty pattern",
                "detail": "Set `response_format.pattern` to the regex body (anchored at start of output).",
            })),
        )
            .into_response());
    };
    if let Err(e) = rustllama_engine::grammar::RegexGrammarParser::new(p) {
        return Err((
            StatusCode::BAD_REQUEST,
            axum::Json(serde_json::json!({
                "error": "invalid regex pattern",
                "detail": e.to_string(),
            })),
        )
            .into_response());
    }
    Ok(())
}

/// Apply `response_format` instructions to the message list before rendering
/// the chat template. Returns a (possibly modified) list — for JSON mode we
/// prepend a system message that instructs the model to emit only JSON.
fn apply_response_format(
    messages: Vec<ChatMessageWire>,
    fmt: &Option<ResponseFormat>,
) -> Vec<ChatMessageWire> {
    let Some(rf) = fmt else { return messages };
    match rf.kind.as_str() {
        "json_object" => prepend_system(
            messages,
            "You must respond with a single valid JSON object. \
             Do not include any text, prose, or markdown fences outside of the JSON. \
             Do not wrap the JSON in code blocks.",
        ),
        "json_schema" => {
            let mut nudge = String::from(
                "You must respond with a single valid JSON object that matches the following JSON Schema. \
                 Do not include any text outside of the JSON.",
            );
            if let Some(schema) = &rf.json_schema {
                nudge.push_str("\n\nSchema:\n");
                if let Ok(s) = serde_json::to_string_pretty(schema) {
                    nudge.push_str(&s);
                }
            }
            prepend_system(messages, &nudge)
        }
        // rustllama extension. Instructs the model to emit apply-edits
        // hunks in the aider-style SEARCH/REPLACE format that VS Code
        // / Continue.dev / Cursor / Cline all consume:
        //
        //     path/to/file.ext
        //     <<<<<<< SEARCH
        //     <verbatim existing text>
        //     =======
        //     <new text>
        //     >>>>>>> REPLACE
        //
        // The server parses these out and surfaces them in
        // `message.diffs` alongside the raw `content`.
        "diff" => prepend_system(
            messages,
            "You are emitting code edits. Reply ONLY with one or more \
             SEARCH/REPLACE blocks, each preceded by a path on its own \
             line. Use this exact format and nothing else:\n\
             \n\
             path/to/file.ext\n\
             <<<<<<< SEARCH\n\
             <verbatim existing text>\n\
             =======\n\
             <new text>\n\
             >>>>>>> REPLACE\n\
             \n\
             Rules: the SEARCH section must match the file byte-for-byte; \
             include surrounding context lines if the change isn't unique. \
             No prose outside the blocks. Each block edits exactly one \
             file. Emit multiple blocks for edits in different files.",
        ),
        // Regex-constrained output. The sampler grammar enforces
        // byte-by-byte compliance with the pattern, but giving the
        // model an explicit instruction + the pattern itself helps
        // it stop drifting off-pattern at the start of generation
        // (where most regex constraints fail loudly because the
        // first wrong byte gets masked out and the model wastes
        // tokens hunting for the start state).
        "regex" => {
            if let Some(p) = rf.pattern.as_ref().filter(|p| !p.is_empty()) {
                prepend_system(
                    messages,
                    &format!(
                        "Your entire response MUST match this regular expression \
                         (anchored at the start): {p}\n\
                         Emit only characters that fit the pattern. Do not include \
                         prose, code fences, or commentary."
                    ),
                )
            } else {
                messages
            }
        }
        _ => messages,
    }
}

/// Inject the SERVER host-environment `[Host environment]` block into a
/// tools request's messages when `[server].tool_environment_hint` is on.
/// Appends the block (blank-line separated) to a leading system message,
/// else prepends a fresh system message — the same
/// append-vs-prepend rule `apply_response_format` uses via
/// [`prepend_system`]. The block is clearly labeled so the model treats
/// it as context, not a user instruction.
fn maybe_inject_env_hint(messages: Vec<ChatMessageWire>) -> Vec<ChatMessageWire> {
    if !crate::env_hint::tool_environment_hint_enabled() {
        return messages;
    }
    prepend_system(messages, crate::env_hint::host_environment_hint())
}

fn prepend_system(mut messages: Vec<ChatMessageWire>, sys_text: &str) -> Vec<ChatMessageWire> {
    // If an existing system message is present, append our nudge to it so
    // both stay in effect. Otherwise insert a fresh system message first.
    if matches!(messages.first(), Some(m) if m.role == "system") {
        let existing = messages.remove(0).content.into_text();
        let merged = format!("{existing}\n\n{sys_text}");
        messages.insert(
            0,
            ChatMessageWire {
                role: "system".into(),
                content: OpenAiContent::Plain(merged),
                ..Default::default()
            },
        );
    } else {
        messages.insert(
            0,
            ChatMessageWire {
                role: "system".into(),
                content: OpenAiContent::Plain(sys_text.to_string()),
                ..Default::default()
            },
        );
    }
    messages
}

fn unix_ts() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Short stable label for the engine's active KV dtype, used as
/// one of the inputs to `system_fingerprint`. Matches the
/// `/v1/metrics` `kv_dtype` field so consecutive responses + the
/// metrics endpoint agree on the same string.
fn kv_dtype_label(cpu: &rustllama_engine::CpuEngine) -> &'static str {
    match cpu.kv_dtype() {
        rustllama_engine::KvDtype::F32 => "f32",
        rustllama_engine::KvDtype::Q8_0 => "q8_0",
        rustllama_engine::KvDtype::Tq(_) => "tq",
        rustllama_engine::KvDtype::Nvfp4 => "nvfp4",
        rustllama_engine::KvDtype::Q4_0 => "q4_0",
        rustllama_engine::KvDtype::Mxfp4 => "mxfp4",
        rustllama_engine::KvDtype::Mxfp6 => "mxfp6",
        rustllama_engine::KvDtype::Mxfp8 => "mxfp8",
    }
}

fn request_id() -> String {
    format!("chatcmpl-{:032x}", unix_ts() as u128)
}

async fn chat_blocking(state: AppState, req: ChatRequest) -> Response {
    let Some(serving) = state.resolve(req.model.as_deref()).await else {
        return model_not_found(req.model.as_deref());
    };
    // Two-phase admission — see `chat_stream` for the rationale.
    // The prompt-token count runs against the Arc-shared CPU engine
    // BEFORE the gate wait so it overlaps the current request's
    // decode on queued arrivals.
    let permit = match serving.try_admit() {
        Ok(p) => p,
        Err(be) => return be.into_response(),
    };
    let sampling = build_sampling(&req);
    let model = req.model.unwrap_or_else(|| serving.model_id.clone());
    let want_logprobs = sampling.logprobs.is_some();
    let msgs: Vec<ChatMessage> = match wire_messages_to_engine(req.messages) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    let shared_cpu = serving.cpu_engine.clone();
    let prompt_tokens = shared_cpu
        .as_ref()
        .and_then(|e| e.count_chat_prompt(&msgs).ok())
        .unwrap_or(0);
    let handle = match permit.acquire_gate().await {
        Ok(h) => h,
        Err(be) => return be.into_response(),
    };
    // Use the handle's fork for both the chat() call and for the
    // `last_request_stats()` read further down — both touch
    // per-request state.
    let cpu = handle.cpu_engine.clone();
    let mut stream = match handle.engine.chat(&msgs, &sampling) {
        Ok(s) => s,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };

    let mut content = String::new();
    let mut logprob_tokens: Vec<TokenLogprobOut> = Vec::new();
    let mut completion_tokens = 0u32;
    while let Some(tok) = stream.next().await {
        match tok {
            Ok(t) => {
                if want_logprobs {
                    if let Some(lp) = t.logprobs.as_ref() {
                        logprob_tokens.push(make_token_logprob_out(&t.text, lp, cpu.as_deref()));
                    }
                }
                content.push_str(&t.text);
                completion_tokens += 1;
            }
            Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        }
    }

    let logprobs_out = if want_logprobs {
        Some(ChoiceLogprobs {
            content: logprob_tokens,
        })
    } else {
        None
    };

    // Diff mode: when the caller set response_format.type = "diff", parse
    // SEARCH/REPLACE hunks out of the raw text and surface them in
    // `message.diffs`. The raw content stays in `message.content` so
    // clients can keep their own diff parser if they want.
    let diffs = if matches!(
        req.response_format.as_ref().map(|r| r.kind.as_str()),
        Some("diff")
    ) {
        parse_diff_blocks(&content)
    } else {
        None
    };

    // Capture per-request stats while the gate permit is still held.
    let usage_base = Usage::new(prompt_tokens, completion_tokens);
    let usage = match cpu.as_deref() {
        Some(cpu) => usage_base.with_stats(&cpu.last_request_stats()),
        None => usage_base,
    };

    let kv_label = cpu.as_deref().map(kv_dtype_label).unwrap_or("");
    let fingerprint = crate::system_fingerprint(&state.version, &model, kv_label);
    Json(ChatResponse {
        id: request_id(),
        object: "chat.completion",
        created: unix_ts(),
        model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageOut {
                role: "assistant",
                content: Some(content),
                tool_calls: None,
                diffs,
            },
            // OpenAI spec: `"length"` when truncated by `max_tokens`,
            // `"stop"` for a natural end-of-stream. Until this commit
            // we always reported `"stop"` — clients that act on the
            // distinction (e.g. retrying with a larger budget) were
            // mis-served. Pin the spec-correct mapping.
            finish_reason: if completion_tokens >= sampling.max_tokens {
                "length"
            } else {
                "stop"
            },
            logprobs: logprobs_out,
        }],
        usage: Some(usage),
        system_fingerprint: fingerprint,
    })
    .into_response()
}

/// Build the OpenAI `logprobs.content[i]` entry. The chosen token's text is
/// passed in directly (already decoded). Alternatives are detokenized via
/// the engine if available; if not, the id is rendered as `"<N>"`.
fn make_token_logprob_out(
    chosen_text: &str,
    lp: &rustllama_engine::TokenLogprobs,
    cpu: Option<&rustllama_engine::CpuEngine>,
) -> TokenLogprobOut {
    let top: Vec<TopLogprobOut> = lp
        .top
        .iter()
        .map(|alt| {
            let alt_text = cpu
                .and_then(|e| e.tokenizer())
                .and_then(|t| t.decode_single(alt.id, true).ok())
                .unwrap_or_else(|| format!("<{}>", alt.id));
            let bytes = alt_text.as_bytes().to_vec();
            TopLogprobOut {
                token: alt_text,
                logprob: alt.logprob,
                bytes,
            }
        })
        .collect();
    TokenLogprobOut {
        token: chosen_text.to_string(),
        logprob: lp.logprob,
        bytes: chosen_text.as_bytes().to_vec(),
        top_logprobs: top,
    }
}

async fn chat_stream(state: AppState, req: ChatRequest) -> Response {
    let Some(serving) = state.resolve(req.model.as_deref()).await else {
        return model_not_found(req.model.as_deref());
    };
    // Two-phase admission: try_admit() runs the backpressure check
    // + scheduler slot reservation but does NOT block on the gate.
    // We then do tokenization (wire→engine message conversion +
    // chat-template render + prompt-token count) using the Arc-shared
    // CPU engine — that work overlaps the gate wait for queued
    // requests, reducing observed TTFT by ~tokenize_ms when a prior
    // request is still decoding.
    let permit = match serving.try_admit() {
        Ok(p) => p,
        Err(be) => return be.into_response(),
    };
    let sampling = build_sampling(&req);
    let model = req.model.unwrap_or_else(|| serving.model_id.clone());
    let id = request_id();
    let cancel_guard = state.register_cancel(&id);
    let created = unix_ts();
    let include_usage = req
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);
    let diff_mode = matches!(
        req.response_format.as_ref().map(|r| r.kind.as_str()),
        Some("diff")
    );
    let msgs: Vec<ChatMessage> = match wire_messages_to_engine(req.messages) {
        Ok(v) => v,
        Err(resp) => return resp,
    };
    // Pre-gate prompt-token count via the Arc-shared CPU engine.
    // Tokenizer state is shared across the multi-flight pool, so
    // this is safe to call without holding the gate.
    let shared_cpu = serving.cpu_engine.clone();
    let prompt_tokens = shared_cpu
        .as_ref()
        .and_then(|e| e.count_chat_prompt(&msgs).ok())
        .unwrap_or(0);

    // Wait on the per-model gate. Earlier admission + tokenization
    // already happened; this is the only step that actually
    // serializes against the current in-flight request.
    let handle = match permit.acquire_gate().await {
        Ok(h) => h,
        Err(be) => return be.into_response(),
    };
    let cpu = handle.cpu_engine.clone();

    // Kick off the engine BEFORE moving into the stream — that way we can
    // return an HTTP error synchronously if the engine refuses (e.g., no
    // tokenizer loaded) instead of producing a half-empty SSE stream.
    let tok_stream = match handle.engine.chat(&msgs, &sampling) {
        Ok(s) => s,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };

    let id_for_header = id.clone();
    let fingerprint = crate::system_fingerprint(
        &state.version,
        &model,
        cpu.as_deref().map(kv_dtype_label).unwrap_or(""),
    );
    let event_stream = build_sse_stream(
        id,
        created,
        model,
        tok_stream,
        handle,
        cpu,
        prompt_tokens,
        include_usage,
        diff_mode,
        cancel_guard,
        sampling.max_tokens,
        fingerprint,
    );
    (
        [("x-rustllama-request-id", id_for_header.as_str())],
        Sse::new(event_stream).keep_alive(KeepAlive::default()),
    )
        .into_response()
}

#[allow(clippy::too_many_arguments)]
fn build_sse_stream(
    id: String,
    created: u64,
    model: String,
    mut tok_stream: rustllama_engine::TokenStream,
    permit: crate::PermitGuard,
    cpu: Option<std::sync::Arc<rustllama_engine::CpuEngine>>,
    prompt_tokens: u32,
    include_usage: bool,
    diff_mode: bool,
    cancel_guard: crate::CancelGuard,
    max_tokens: u32,
    fingerprint: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let fp = fingerprint.as_str();
        // Initial role chunk.
        yield Ok(Event::default().data(
            chunk_json_with_fp(&id, created, &model, json!({"role": "assistant"}), None, None, fp).to_string()
        ));

        // Per-token deltas.
        let mut finish_reason: &'static str = "stop";
        let mut completion_tokens = 0u32;
        let mut diff_accum = if diff_mode { Some(String::new()) } else { None };
        let cancel_flag = cancel_guard.flag.clone();
        // Reusable per-token chunk buffer. 1 KB covers a typical
        // text-only chunk envelope (~200 B) with headroom for a
        // multi-byte UTF-8 token without re-allocating.
        let mut chunk_buf = String::with_capacity(1024);
        while let Some(item) = tok_stream.next().await {
            // /v1/cancel polled between tokens. The check before emit
            // means a fired cancel halts further deltas immediately;
            // dropping `tok_stream` after the loop closes rx so the
            // engine sees `tx.is_closed()` and exits within ~one
            // forward pass.
            if cancel_flag.load(std::sync::atomic::Ordering::Acquire) {
                finish_reason = "cancelled";
                break;
            }
            match item {
                Ok(tok) => {
                    completion_tokens += 1;
                    if let Some(buf) = diff_accum.as_mut() {
                        buf.push_str(&tok.text);
                    }
                    // Fast path: text-only delta, no logprobs. Builds
                    // the JSON directly into `chunk_buf` — no
                    // serde_json `Value` tree, no per-token String
                    // allocation. Falls through to the serde_json
                    // path when the caller asked for logprobs (an
                    // OpenAI option used by eval frameworks, not the
                    // typical streaming chat client).
                    if tok.logprobs.is_none() {
                        chunk_text_delta_into(
                            &mut chunk_buf, &id, created, &model, &tok.text, fp,
                        );
                        yield Ok(Event::default().data(chunk_buf.clone()));
                    } else {
                        let lp_json = tok.logprobs.as_ref().map(|lp| {
                            token_logprob_chunk_json(&tok.text, lp, cpu.as_deref())
                        });
                        yield Ok(Event::default().data(
                            chunk_json_with_fp(
                                &id,
                                created,
                                &model,
                                json!({"content": tok.text}),
                                None,
                                lp_json,
                                fp,
                            )
                            .to_string(),
                        ));
                    }
                }
                Err(e) => {
                    // Emit an OpenAI-style error event then stop. Most clients
                    // tolerate this; some treat it as the end of the stream.
                    let err = json!({"error": {"message": e.to_string()}});
                    yield Ok(Event::default().data(err.to_string()));
                    finish_reason = "error";
                    break;
                }
            }
        }

        // Final delta. In diff mode, attach the parsed hunks to the
        // choice as `diffs: [...]`. Clients walking the stream see this
        // on the same chunk that carries `finish_reason: "stop"`.
        let diffs_json: Option<Value> = diff_accum
            .as_deref()
            .and_then(parse_diff_blocks)
            .and_then(|v| serde_json::to_value(v).ok());
        let stats = cpu.as_deref().map(|c| c.last_request_stats());
        if stats.as_ref().map(|s| s.tool_call_limit_hit).unwrap_or(false) {
            finish_reason = "tool_call_iteration_limit";
        } else if finish_reason == "stop" && completion_tokens >= max_tokens {
            // Spec-correct finish_reason for max_tokens-hit. Only
            // applies when we'd otherwise be reporting `"stop"` —
            // cancel / error / tool_call_iteration_limit take
            // precedence.
            finish_reason = "length";
        }
        let mut final_chunk = chunk_json_with_fp(&id, created, &model, json!({}), Some(finish_reason), None, fp);
        if let Some(d) = diffs_json {
            if let Some(choices) = final_chunk.get_mut("choices").and_then(Value::as_array_mut) {
                if let Some(choice) = choices.get_mut(0).and_then(Value::as_object_mut) {
                    choice.insert("diffs".into(), d);
                }
            }
        }
        yield Ok(Event::default().data(final_chunk.to_string()));
        if include_usage {
            yield Ok(Event::default().data(
                usage_chunk_json_with_stats(
                    &id, created, &model, prompt_tokens, completion_tokens, stats.as_ref()
                ).to_string()
            ));
        }
        yield Ok(Event::default().data("[DONE]"));
        drop(permit);
        drop(cancel_guard); // removes the entry from AppState.cancellations
    }
}

#[cfg(test)]
fn chunk_json(
    id: &str,
    created: u64,
    model: &str,
    delta: Value,
    finish_reason: Option<&'static str>,
    logprobs: Option<Value>,
) -> Value {
    chunk_json_with_fp(id, created, model, delta, finish_reason, logprobs, "")
}

/// Same as [`chunk_json`] but threads the OpenAI `system_fingerprint`
/// into the chunk. Production streaming sites compute the fingerprint
/// once per request and pass it on every chunk so editor clients
/// (Aider, Continue) can compare it across chunks and detect mid-
/// conversation backend swaps. Empty fingerprint omits the field —
/// preserved for the test harness which doesn't compute one.
fn chunk_json_with_fp(
    id: &str,
    created: u64,
    model: &str,
    delta: Value,
    finish_reason: Option<&'static str>,
    logprobs: Option<Value>,
    system_fingerprint: &str,
) -> Value {
    let mut choice = json!({
        "index": 0,
        "delta": delta,
        "finish_reason": finish_reason,
    });
    if let Some(lp) = logprobs {
        choice["logprobs"] = lp;
    }
    let mut out = json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice]
    });
    if !system_fingerprint.is_empty() {
        if let Some(obj) = out.as_object_mut() {
            obj.insert(
                "system_fingerprint".to_string(),
                Value::String(system_fingerprint.to_string()),
            );
        }
    }
    out
}

/// Fast-path per-token chunk serialization for the common case
/// (text-content delta, no logprobs, no finish_reason). Builds the
/// JSON directly into a caller-owned buffer — no serde_json `Value`
/// trees, no per-call `String` allocation for the result. Caller
/// passes a `&mut String` that's reused across all tokens in the
/// stream (pre-allocated to ~1 KB at stream start).
///
/// Equivalent output to
/// `chunk_json_with_fp(id, created, model, json!({"content": text}), None, None, fp).to_string()`,
/// modulo serde_json's whitespace policy (we emit no whitespace).
/// Properly escapes the token text per RFC 8259 §7 — `"`, `\`, and
/// control chars `[0x00, 0x1F]` are emitted as escape sequences.
fn chunk_text_delta_into(
    buf: &mut String,
    id: &str,
    created: u64,
    model: &str,
    text: &str,
    fp: &str,
) {
    buf.clear();
    buf.push_str(r#"{"id":""#);
    append_json_escaped(buf, id);
    buf.push_str(r#"","object":"chat.completion.chunk","created":"#);
    use std::fmt::Write as _;
    let _ = write!(buf, "{created}");
    buf.push_str(r#","model":""#);
    append_json_escaped(buf, model);
    buf.push_str(r#"","choices":[{"index":0,"delta":{"content":""#);
    append_json_escaped(buf, text);
    buf.push_str(r#""},"finish_reason":null}]"#);
    if !fp.is_empty() {
        buf.push_str(r#","system_fingerprint":""#);
        append_json_escaped(buf, fp);
        buf.push('"');
    }
    buf.push('}');
}

/// Append `s` to `buf` with JSON string escaping. Hot enough on the
/// per-token streaming path that we avoid `String::escape_debug` /
/// `serde_json` here — the spec only mandates escaping for `"`, `\`,
/// and control chars in `[0x00, 0x1F]`; other Unicode passes through
/// verbatim as UTF-8, which is exactly what `Event::data` expects.
fn append_json_escaped(buf: &mut String, s: &str) {
    for c in s.chars() {
        match c {
            '"' => buf.push_str(r#"\""#),
            '\\' => buf.push_str(r"\\"),
            '\n' => buf.push_str(r"\n"),
            '\r' => buf.push_str(r"\r"),
            '\t' => buf.push_str(r"\t"),
            '\x08' => buf.push_str(r"\b"),
            '\x0c' => buf.push_str(r"\f"),
            c if (c as u32) < 0x20 => {
                use std::fmt::Write as _;
                let _ = write!(buf, "\\u{:04x}", c as u32);
            }
            c => buf.push(c),
        }
    }
}

/// Build the per-chunk `logprobs.content` value for one streamed token.
/// Mirrors `make_token_logprob_out` but yields raw JSON since the SSE
/// path stitches Values together rather than building typed Outs.
fn token_logprob_chunk_json(
    chosen_text: &str,
    lp: &rustllama_engine::TokenLogprobs,
    cpu: Option<&rustllama_engine::CpuEngine>,
) -> Value {
    let top: Vec<Value> = lp
        .top
        .iter()
        .map(|alt| {
            let alt_text = cpu
                .and_then(|e| e.tokenizer())
                .and_then(|t| t.decode_single(alt.id, true).ok())
                .unwrap_or_else(|| format!("<{}>", alt.id));
            let bytes: Vec<u8> = alt_text.as_bytes().to_vec();
            json!({
                "token": alt_text,
                "logprob": alt.logprob,
                "bytes": bytes,
            })
        })
        .collect();
    json!({
        "content": [{
            "token": chosen_text,
            "logprob": lp.logprob,
            "bytes": chosen_text.as_bytes(),
            "top_logprobs": top,
        }]
    })
}

#[cfg(test)]
mod tests {
    use super::{
        apply_response_format, build_sampling, chunk_json, chunk_json_with_fp,
        parse_ask_user_args, with_reserved_ask_user_tool, ASK_USER_TOOL_NAME,
        chunk_text_delta_into, extract_tool_schemas, longest_suffix_that_is_prefix_of,
        make_token_logprob_out, openai_content_block_to_text, parse_tool_call_body,
        parse_tool_calls, parse_tool_choice, token_logprob_chunk_json, wire_messages_to_engine,
        wire_messages_to_json, ChatMessageWire, ChatRequest, OpenAiContent, OpenAiContentBlock,
        ResponseFormat, StreamEvent, StreamingToolCallParser, ToolChoice,
    };
    use axum::http::StatusCode;
    use rustllama_engine::{TokenLogprobs, TopLogprob};
    use serde_json::{json, Value};

    // ----- seed determinism pinning ----------------------------------------
    //
    // The engine's seed honor is proven by
    // `rustllama-engine/tests/cpu_engine.rs::seeded_sampling_is_reproducible`.
    // What we pin here is that the API surface threads the request's
    // `seed` field into the SamplingParams the engine consumes — i.e.,
    // that two API requests with the same `seed` reach the engine with
    // the same sampling config. (Anthropic's seed audit lives in
    // `anthropic::tests` since its build_sampling has a different
    // shape; the official Anthropic API doesn't expose seed.)

    fn chat_request_with_seed(seed: Option<u64>) -> ChatRequest {
        // Construct a minimal valid ChatRequest. serde-derive is the
        // public deserializer path; using it here keeps the test honest
        // about how the wire shape parses.
        let mut value = serde_json::json!({
            "model": "rustllama-mock",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
        });
        if let Some(s) = seed {
            value["seed"] = serde_json::json!(s);
        }
        serde_json::from_value(value).expect("ChatRequest parses")
    }

    #[test]
    fn seed_field_threads_into_sampling_params_for_chat() {
        let req = chat_request_with_seed(Some(42));
        let sampling = build_sampling(&req);
        assert_eq!(sampling.seed, 42, "OpenAI chat seed must reach SamplingParams");
    }

    #[test]
    fn seed_absent_keeps_default_sampling_seed() {
        // Negative case: no `seed` field on the request keeps the
        // SamplingParams default (currently 0). This is what lets
        // callers opt out of determinism explicitly.
        let req = chat_request_with_seed(None);
        let default_seed = rustllama_engine::SamplingParams::default().seed;
        let sampling = build_sampling(&req);
        assert_eq!(
            sampling.seed, default_seed,
            "absent seed must not override the SamplingParams default"
        );
    }

    #[test]
    fn code_response_format_engages_code_grammar() {
        // `response_format: {"type":"code", "json_schema":{"language":"python"}}`
        // should produce `SamplingParams.grammar = GrammarKind::Code { language: "python" }`.
        // Pins the wire→engine plumbing for the new constraint.
        let value = serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "response_format": {
                "type": "code",
                "json_schema": {"language": "python"}
            }
        });
        let req: ChatRequest = serde_json::from_value(value).expect("parses");
        let sampling = build_sampling(&req);
        match sampling.grammar {
            Some(rustllama_engine::GrammarKind::Code { language }) => {
                assert_eq!(language, "python");
            }
            other => panic!("expected GrammarKind::Code, got {other:?}"),
        }
    }

    #[test]
    fn code_response_format_without_language_still_engages_code_grammar() {
        // The `language` hint is optional. Without it, the constraint
        // still engages with an empty language string. Pin the
        // forward-compat shape — clients can omit `language` and get
        // the generic bracket-balance behavior.
        let value = serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "response_format": { "type": "code" }
        });
        let req: ChatRequest = serde_json::from_value(value).expect("parses");
        let sampling = build_sampling(&req);
        match sampling.grammar {
            Some(rustllama_engine::GrammarKind::Code { language }) => {
                assert_eq!(language, "");
            }
            other => panic!("expected GrammarKind::Code, got {other:?}"),
        }
    }

    #[test]
    fn typical_p_threads_into_sampling_params_for_chat() {
        let value = serde_json::json!({
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "typical_p": 0.6,
        });
        let req: ChatRequest = serde_json::from_value(value).expect("ChatRequest parses");
        let sampling = build_sampling(&req);
        assert!((sampling.typical_p - 0.6).abs() < 1e-6);
    }

    #[test]
    fn typical_p_absent_keeps_default_for_chat() {
        let req = chat_request_with_seed(None);
        let sampling = build_sampling(&req);
        let defaults = rustllama_engine::SamplingParams::default();
        assert!((sampling.typical_p - defaults.typical_p).abs() < 1e-6);
    }

    #[test]
    fn mirostat_fields_thread_into_sampling_params_for_chat() {
        // Same wiring contract as `seed`: setting `mirostat: 2`,
        // `mirostat_tau`, and `mirostat_eta` on the OpenAI chat
        // request lands the values on the SamplingParams the engine
        // consumes.
        let value = serde_json::json!({
            "model": "rustllama-mock",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            "mirostat": 2,
            "mirostat_tau": 7.0,
            "mirostat_eta": 0.2,
        });
        let req: ChatRequest = serde_json::from_value(value).expect("ChatRequest parses");
        let sampling = build_sampling(&req);
        assert_eq!(sampling.mirostat, 2);
        assert!((sampling.mirostat_tau - 7.0).abs() < 1e-6);
        assert!((sampling.mirostat_eta - 0.2).abs() < 1e-6);
    }

    #[test]
    fn mirostat_absent_leaves_sampling_defaults_for_chat() {
        // Absent fields keep the SamplingParams defaults (mirostat=0,
        // i.e., disabled). Catches a regression where the deserializer
        // accidentally forced a non-zero default.
        let req = chat_request_with_seed(None);
        let sampling = build_sampling(&req);
        let defaults = rustllama_engine::SamplingParams::default();
        assert_eq!(sampling.mirostat, defaults.mirostat);
        assert!((sampling.mirostat_tau - defaults.mirostat_tau).abs() < 1e-6);
        assert!((sampling.mirostat_eta - defaults.mirostat_eta).abs() < 1e-6);
    }

    #[test]
    fn seed_round_trips_through_two_identical_chat_requests() {
        // Two requests with the same seed produce the same
        // SamplingParams. Strictly weaker than "two identical token
        // streams" (which is the engine's job), but pins the wire→
        // engine plumbing.
        let s1 = build_sampling(&chat_request_with_seed(Some(0xDEAD_BEEF)));
        let s2 = build_sampling(&chat_request_with_seed(Some(0xDEAD_BEEF)));
        assert_eq!(s1.seed, s2.seed);
    }

    // ----- multimodal content blocks ----------------------------------------

    #[test]
    fn openai_content_block_text_passes_through() {
        let b = OpenAiContentBlock {
            kind: "text".into(),
            text: "hello".into(),
            image_url: None,
        };
        assert_eq!(openai_content_block_to_text(b), "hello");
    }

    #[test]
    fn openai_content_block_image_url_renders_placeholder_with_url() {
        // Placeholder includes the actual URL so coding models can
        // reason about file extension / domain even on non-vision
        // backends. Previously the literal string "[image: url]"
        // was emitted regardless of the actual URL — useless signal.
        let b = OpenAiContentBlock {
            kind: "image_url".into(),
            text: String::new(),
            image_url: Some(super::OpenAiImageUrl {
                url: "https://example.com/x.png".into(),
                detail: "auto".into(),
            }),
        };
        assert_eq!(
            openai_content_block_to_text(b),
            "[image: https://example.com/x.png]"
        );
    }

    #[test]
    fn openai_content_block_image_url_with_empty_url_falls_back_to_generic() {
        // Edge case: the wire-protocol allows an `image_url` block
        // without a populated URL field. We don't want to emit
        // "[image: ]" — fall back to the original literal.
        let b = OpenAiContentBlock {
            kind: "image_url".into(),
            text: String::new(),
            image_url: Some(super::OpenAiImageUrl {
                url: String::new(),
                detail: "auto".into(),
            }),
        };
        assert_eq!(openai_content_block_to_text(b), "[image: url]");
    }

    #[test]
    fn openai_content_block_image_url_without_image_url_field_uses_generic() {
        // Pathological: `kind = "image_url"` but no `image_url` field
        // populated (image_url: None). Fall back to the generic
        // literal — clients that send malformed blocks still get
        // something parseable.
        let b = OpenAiContentBlock {
            kind: "image_url".into(),
            text: String::new(),
            image_url: None,
        };
        assert_eq!(openai_content_block_to_text(b), "[image: url]");
    }

    #[test]
    fn openai_content_plain_string_deserializes_unchanged() {
        let json_in = json!({"role": "user", "content": "plain"});
        let msg: ChatMessageWire = serde_json::from_value(json_in).expect("parse");
        assert_eq!(msg.content.text().as_ref(), "plain");
    }

    #[test]
    fn openai_content_block_array_renders_mixed_text_and_image_placeholders() {
        // GPT-4-Vision-shape: alternating text + image_url blocks.
        // URL is preserved in the placeholder so the model sees it.
        let json_in = json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "describe "},
                {"type": "image_url", "image_url": {"url": "https://e.com/x.png"}},
                {"type": "text", "text": " in detail"}
            ]
        });
        let msg: ChatMessageWire = serde_json::from_value(json_in).expect("parse");
        assert_eq!(
            msg.content.text().as_ref(),
            "describe [image: https://e.com/x.png] in detail"
        );
    }

    #[test]
    fn openai_content_unknown_block_kind_falls_back_to_text_field() {
        // Forward-compat: a future block kind with a `text` field
        // should surface that text rather than dropping silently.
        let b = OpenAiContentBlock {
            kind: "audio".into(),
            text: "transcript here".into(),
            image_url: None,
        };
        assert_eq!(openai_content_block_to_text(b), "transcript here");
    }

    #[test]
    fn openai_content_from_str_constructs_plain_variant() {
        // The `From<&str>` helper keeps the existing
        // `content: "...".into()` ergonomics for test fixtures alive
        // after the field type changed from String → OpenAiContent.
        let c: OpenAiContent = "hi".into();
        assert_eq!(c.text().as_ref(), "hi");
    }

    // ----- V-6a: wire -> engine image plumbing -----------------------------

    /// Helper: build a `data:image/png;base64,<...>` URI from raw bytes
    /// so the V-6a tests can mint syntactically valid image_url blocks
    /// without depending on the V-4 module's test helpers.
    fn make_test_png_data_uri(bytes: &[u8]) -> String {
        use base64::Engine as _;
        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
        format!("data:image/png;base64,{encoded}")
    }

    #[test]
    fn try_into_with_image_bytes_plain_string_has_no_images() {
        let json_in = json!({"role": "user", "content": "plain"});
        let wire: ChatMessageWire = serde_json::from_value(json_in).unwrap();
        let m = wire.try_into_with_image_bytes().unwrap();
        assert_eq!(m.role, "user");
        assert_eq!(m.content, "plain");
        assert!(m.images.is_empty());
    }

    #[test]
    fn try_into_with_image_bytes_decodes_image_url_block_to_bytes() {
        let raw = b"\x89PNG\r\n\x1a\n_fake_payload";
        let url = make_test_png_data_uri(raw);
        let json_in = json!({
            "role": "user",
            "content": [
                {"type": "text", "text": "describe: "},
                {"type": "image_url", "image_url": {"url": url}}
            ]
        });
        let wire: ChatMessageWire = serde_json::from_value(json_in).unwrap();
        let m = wire.try_into_with_image_bytes().unwrap();
        assert_eq!(m.images.len(), 1);
        assert_eq!(m.images[0], raw);
        // Placeholder still in content so a vision-aware engine can find
        // the splice position.
        assert!(m.content.starts_with("describe: [image: data:image/png;base64,"));
    }

    #[test]
    fn try_into_with_image_bytes_collects_multiple_images_in_order() {
        let raw0 = b"\x89PNG\r\n\x1a\n_first";
        let raw1 = b"\xff\xd8\xff_second_jpeg";
        let url0 = make_test_png_data_uri(raw0);
        // Use the JPEG media-type for the second one to pin that the
        // decoder routes both PNG and JPEG block-kinds.
        let url1 = {
            use base64::Engine as _;
            let enc = base64::engine::general_purpose::STANDARD.encode(raw1);
            format!("data:image/jpeg;base64,{enc}")
        };
        let json_in = json!({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url0}},
                {"type": "text", "text": " vs "},
                {"type": "image_url", "image_url": {"url": url1}}
            ]
        });
        let wire: ChatMessageWire = serde_json::from_value(json_in).unwrap();
        let m = wire.try_into_with_image_bytes().unwrap();
        assert_eq!(m.images.len(), 2);
        assert_eq!(m.images[0], raw0);
        assert_eq!(m.images[1], raw1);
    }

    #[test]
    fn try_into_with_image_bytes_errors_on_https_url_v1_ssrf_gate() {
        let json_in = json!({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}
            ]
        });
        let wire: ChatMessageWire = serde_json::from_value(json_in).unwrap();
        match wire.try_into_with_image_bytes() {
            Err(crate::image_url::ImageUrlError::RemoteUrlNotSupported(ref s)) if s == "https" => {}
            other => panic!("expected RemoteUrlNotSupported, got {other:?}"),
        }
    }

    #[test]
    fn try_into_with_image_bytes_empty_url_field_emits_placeholder_no_bytes() {
        // Pathological wire shape — accept gracefully (no error, no
        // attached bytes), so a malformed client doesn't bring down
        // an entire chat session.
        let json_in = json!({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": ""}}
            ]
        });
        let wire: ChatMessageWire = serde_json::from_value(json_in).unwrap();
        let m = wire.try_into_with_image_bytes().unwrap();
        assert!(m.images.is_empty());
        assert_eq!(m.content, "[image: url]");
    }

    #[test]
    fn wire_messages_to_engine_returns_bad_request_on_decode_failure() {
        use axum::body::to_bytes;
        let raw = b"\x89PNG\r\n\x1a\n_ok";
        let good_url = make_test_png_data_uri(raw);
        let msgs = vec![
            ChatMessageWire {
                role: "user".into(),
                content: OpenAiContent::Plain("hi".into()),
                ..Default::default()
            },
            ChatMessageWire {
                role: "user".into(),
                content: OpenAiContent::Blocks(vec![
                    OpenAiContentBlock {
                        kind: "image_url".into(),
                        text: String::new(),
                        image_url: Some(super::OpenAiImageUrl {
                            url: good_url,
                            detail: "auto".into(),
                        }),
                    },
                    OpenAiContentBlock {
                        kind: "image_url".into(),
                        text: String::new(),
                        // Bad URL — must surface as 400.
                        image_url: Some(super::OpenAiImageUrl {
                            url: "https://attacker.example/x.png".into(),
                            detail: "auto".into(),
                        }),
                    },
                ]),
                ..Default::default()
            },
        ];
        let err_resp = wire_messages_to_engine(msgs).expect_err("should fail on second message");
        assert_eq!(err_resp.status(), StatusCode::BAD_REQUEST);
        // Body should mention the offending message index so clients
        // can pinpoint which content block was rejected.
        let body = futures::executor::block_on(async {
            to_bytes(err_resp.into_body(), 1_000_000).await.unwrap()
        });
        let body_str = std::str::from_utf8(&body).unwrap();
        assert!(body_str.contains("messages[1]"), "body: {body_str}");
        assert!(body_str.contains("invalid_image_url"), "body: {body_str}");
    }

    // ----- logprobs surface --------------------------------------------------

    #[test]
    fn make_token_logprob_out_emits_openai_shape() {
        let lp = TokenLogprobs {
            logprob: -0.1234,
            top: vec![
                TopLogprob { id: 7, logprob: -0.1234 },
                TopLogprob { id: 11, logprob: -2.5 },
            ],
        };
        // No tokenizer wired (cpu = None) → alt tokens render as `"<7>"` etc.
        let out = make_token_logprob_out("hello", &lp, None);
        assert_eq!(out.token, "hello");
        assert!((out.logprob - (-0.1234)).abs() < 1e-6);
        assert_eq!(out.bytes, b"hello");
        assert_eq!(out.top_logprobs.len(), 2);
        assert_eq!(out.top_logprobs[0].token, "<7>");
        assert!((out.top_logprobs[0].logprob - (-0.1234)).abs() < 1e-6);
        assert_eq!(out.top_logprobs[1].token, "<11>");
    }

    #[test]
    fn token_logprob_chunk_json_attaches_to_first_content_delta() {
        // The chunk_json helper appends `logprobs` to the choice only
        // when the caller passes Some. This is the wiring path the
        // streaming-with-tools handler uses to attach logprobs to the
        // first Content event per token.
        let lp = TokenLogprobs {
            logprob: -0.5,
            top: vec![TopLogprob { id: 42, logprob: -0.5 }],
        };
        let lp_json = token_logprob_chunk_json(" world", &lp, None);
        let chunk = chunk_json(
            "chatcmpl-x",
            12345,
            "rustllama-test",
            json!({"content": " world"}),
            None,
            Some(lp_json.clone()),
        );
        let choice = &chunk["choices"][0];
        assert_eq!(choice["delta"]["content"], " world");
        // The logprobs payload should be attached and well-formed.
        let entries = choice["logprobs"]["content"]
            .as_array()
            .expect("content array");
        assert_eq!(entries.len(), 1, "one entry per token");
        assert_eq!(entries[0]["token"], " world");
        assert!((entries[0]["logprob"].as_f64().unwrap() - (-0.5)).abs() < 1e-6);
        let top = entries[0]["top_logprobs"].as_array().expect("top array");
        assert_eq!(top.len(), 1);
        assert_eq!(top[0]["token"], "<42>");
    }

    #[test]
    fn chunk_json_without_logprobs_omits_the_field() {
        // Negative: when the handler doesn't pass logprobs (e.g.,
        // tool_call deltas), the choice shouldn't carry a stray null.
        let chunk = chunk_json(
            "chatcmpl-x",
            1,
            "m",
            json!({"role": "assistant"}),
            None,
            None,
        );
        let choice = &chunk["choices"][0];
        assert!(
            choice.get("logprobs").is_none(),
            "logprobs field should be absent when not requested: {chunk}"
        );
    }

    #[test]
    fn chunk_json_with_finish_reason_carries_through() {
        // Sanity: the helper's other args still serialize correctly.
        let chunk = chunk_json("id", 1, "m", json!({}), Some("stop"), None);
        assert_eq!(chunk["choices"][0]["finish_reason"], "stop");
    }

    #[test]
    fn chunk_json_with_fp_adds_system_fingerprint_to_envelope() {
        // OpenAI-compat: streaming chunks carry `system_fingerprint`
        // at the envelope level (next to `id`, `object`, `model`),
        // NOT inside `choices`. Verify the wiring puts it in the
        // right place so editor clients (Aider, Continue) can parse
        // and compare it across consecutive chunks.
        let chunk = chunk_json_with_fp(
            "id",
            1,
            "m",
            json!({"content": "x"}),
            None,
            None,
            "fp_abc123def456",
        );
        assert_eq!(chunk["system_fingerprint"], "fp_abc123def456");
        // Choice block itself is unchanged — fingerprint is not duplicated there.
        let choice = &chunk["choices"][0];
        assert!(choice.get("system_fingerprint").is_none());
    }

    #[test]
    fn chunk_text_delta_into_matches_serde_json_baseline() {
        // The fast-path manual builder must produce JSON that
        // deserializes to the same value as the serde_json baseline.
        // We compare the parsed Value (not the raw string) because
        // serde_json's serializer doesn't guarantee key ordering
        // across versions — and the SSE consumer parses, doesn't
        // pattern-match raw bytes.
        let cases: &[(&str, &str)] = &[
            ("hello world", "fp_abc"),
            ("with \"quotes\" and \\ slashes", "fp_xyz"),
            ("multi\nline\rtoken", ""), // empty fp → omits the field
            ("\u{2028}unicode\t\u{0001}", "fp_z"), // controls + line-sep
            ("", "fp_empty_text"),
        ];
        for &(text, fp) in cases {
            let want = chunk_json_with_fp(
                "chatcmpl-test", 42, "qwen2.5-coder",
                json!({"content": text}), None, None, fp,
            );
            let mut buf = String::with_capacity(256);
            chunk_text_delta_into(
                &mut buf, "chatcmpl-test", 42, "qwen2.5-coder", text, fp,
            );
            let got: Value = serde_json::from_str(&buf)
                .unwrap_or_else(|e| panic!("manual JSON did not parse ({e}): {buf}"));
            assert_eq!(got, want, "chunk parity mismatch for text={text:?} fp={fp:?}");
        }
    }

    #[test]
    fn chunk_json_with_empty_fp_omits_the_field() {
        // The empty-string convention means "skip the field" — useful
        // so the `chunk_json` test wrapper can stay in tests without
        // forcing every test to thread a fingerprint through.
        let chunk = chunk_json_with_fp(
            "id",
            1,
            "m",
            json!({}),
            Some("stop"),
            None,
            "",
        );
        assert!(
            chunk.get("system_fingerprint").is_none(),
            "empty fp must omit the field: {chunk}"
        );
    }


    #[test]
    fn extract_tool_schemas_pulls_name_and_parameters() {
        let tools = json!([
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Fetch the weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": { "type": "string" },
                            "unit": { "type": "string" }
                        },
                        "required": ["city"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "parameters": {
                        "type": "object",
                        "properties": { "path": { "type": "string" } },
                        "required": ["path"]
                    }
                }
            }
        ]);
        let map = extract_tool_schemas(&tools);
        assert_eq!(map.len(), 2);
        assert!(map.contains_key("get_weather"));
        assert!(map.contains_key("list_files"));
    }

    #[test]
    fn extract_tool_schemas_falls_back_to_any_when_parameters_missing() {
        let tools = json!([
            { "type": "function", "function": { "name": "noop" } }
        ]);
        let map = extract_tool_schemas(&tools);
        assert_eq!(map.len(), 1);
        assert!(matches!(
            map.get("noop"),
            Some(rustllama_engine::grammar::Schema::Any)
        ));
    }

    #[test]
    fn extract_tool_schemas_returns_empty_for_non_array() {
        let map = extract_tool_schemas(&json!(null));
        assert!(map.is_empty());
        let map = extract_tool_schemas(&json!({}));
        assert!(map.is_empty());
    }

    #[test]
    fn with_reserved_ask_user_tool_appends_to_caller_tools() {
        // The caller's tools are preserved and `ask_user` is appended, so a
        // model can still call the real tools OR ask the user.
        let tools = json!([
            { "type": "function", "function": { "name": "get_weather" } }
        ]);
        let out = with_reserved_ask_user_tool(tools);
        let names: Vec<&str> = out
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|t| t.pointer("/function/name").and_then(|n| n.as_str()))
            .collect();
        assert_eq!(names, ["get_weather", ASK_USER_TOOL_NAME]);
        // The reserved tool is a real function schema, so `extract_tool_schemas`
        // picks it up for the grammar with no special-casing.
        let map = extract_tool_schemas(&out);
        assert!(map.contains_key(ASK_USER_TOOL_NAME));
    }

    #[test]
    fn with_reserved_ask_user_tool_handles_non_array() {
        // A caller that sent `functions: []` normalizes to `Value::Null`;
        // the reserved tool still gets advertised as a one-element array.
        let out = with_reserved_ask_user_tool(Value::Null);
        assert_eq!(out.as_array().map(|a| a.len()), Some(1));
        assert_eq!(
            out.pointer("/0/function/name").and_then(|n| n.as_str()),
            Some(ASK_USER_TOOL_NAME)
        );
    }

    #[test]
    fn parse_ask_user_args_extracts_prompt_and_options() {
        let (prompt, options) = parse_ask_user_args(
            r#"{"prompt":"Which file?","options":["main.rs","lib.rs"]}"#,
        );
        assert_eq!(prompt, "Which file?");
        assert_eq!(options, ["main.rs", "lib.rs"]);
    }

    #[test]
    fn parse_ask_user_args_tolerates_missing_and_mistyped_fields() {
        // Missing prompt → ""; missing/non-array options → []; non-string
        // option entries are skipped. Never panics — the terminal chunk must
        // always be well-formed even if the model drifted from the schema.
        let (prompt, options) = parse_ask_user_args(r#"{"options":42}"#);
        assert_eq!(prompt, "");
        assert!(options.is_empty());
        let (_, options) = parse_ask_user_args(r#"{"options":["a",7,"b"]}"#);
        assert_eq!(options, ["a", "b"]);
        let (prompt, options) = parse_ask_user_args("not json at all");
        assert_eq!(prompt, "");
        assert!(options.is_empty());
    }

    #[test]
    fn json_mode_prepends_system_message() {
        let msgs = vec![ChatMessageWire {
            role: "user".into(),
            content: "hi".into(),
            ..Default::default()
        }];
        let rf = Some(ResponseFormat {
            kind: "json_object".into(),
            json_schema: None,
            pattern: None,
        });
        let out = apply_response_format(msgs, &rf);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].role, "system");
        assert!(out[0].content.text().contains("JSON"));
        assert_eq!(out[1].role, "user");
    }

    #[test]
    fn json_schema_includes_schema_in_system() {
        let msgs = vec![ChatMessageWire {
            role: "user".into(),
            content: "make one".into(),
            ..Default::default()
        }];
        let schema = serde_json::json!({"type":"object","properties":{"x":{"type":"number"}}});
        let rf = Some(ResponseFormat {
            kind: "json_schema".into(),
            json_schema: Some(schema),
            pattern: None,
        });
        let out = apply_response_format(msgs, &rf);
        assert_eq!(out[0].role, "system");
        assert!(out[0].content.text().contains("Schema"));
        assert!(out[0].content.text().contains("properties"));
    }

    #[test]
    fn json_mode_merges_with_existing_system() {
        let msgs = vec![
            ChatMessageWire {
                role: "system".into(),
                content: "be terse".into(),
                ..Default::default()
            },
            ChatMessageWire {
                role: "user".into(),
                content: "hi".into(),
                ..Default::default()
            },
        ];
        let rf = Some(ResponseFormat {
            kind: "json_object".into(),
            json_schema: None,
            pattern: None,
        });
        let out = apply_response_format(msgs, &rf);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].role, "system");
        assert!(out[0].content.text().contains("be terse"));
        assert!(out[0].content.text().contains("JSON"));
    }

    #[test]
    fn no_response_format_passes_through() {
        let msgs = vec![ChatMessageWire {
            role: "user".into(),
            content: "hi".into(),
            ..Default::default()
        }];
        let out = apply_response_format(msgs, &None);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].role, "user");
    }

    #[test]
    fn unknown_response_format_passes_through() {
        let msgs = vec![ChatMessageWire {
            role: "user".into(),
            content: "hi".into(),
            ..Default::default()
        }];
        let rf = Some(ResponseFormat {
            kind: "yaml_object".into(),
            json_schema: None,
            pattern: None,
        });
        let out = apply_response_format(msgs, &rf);
        // Unknown format is silently ignored — no system msg added.
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].role, "user");
    }


    #[test]
    fn parses_single_qwen_style_tool_call() {
        let raw = r#"<tool_call>
{"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}}
</tool_call>"#;
        let (content, calls) = parse_tool_calls(raw);
        assert!(content.is_none(), "expected no remaining content");
        let calls = calls.expect("expected tool_calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("\"city\""));
        assert!(calls[0].function.arguments.contains("Paris"));
        assert_eq!(calls[0].kind, "function");
    }

    #[test]
    fn parses_multiple_tool_calls_with_text_between() {
        let raw = r#"Let me check.
<tool_call>{"name": "f1", "arguments": {"a": 1}}</tool_call>
Now the other.
<tool_call>{"name": "f2", "arguments": {"b": 2}}</tool_call>"#;
        let (content, calls) = parse_tool_calls(raw);
        let content = content.expect("text content");
        assert!(content.contains("Let me check."));
        assert!(content.contains("Now the other."));
        let calls = calls.expect("tool_calls");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "f1");
        assert_eq!(calls[1].function.name, "f2");
    }

    #[test]
    fn no_tool_calls_returns_plain_content() {
        let raw = "Just a regular reply, no tools.";
        let (content, calls) = parse_tool_calls(raw);
        assert_eq!(content.as_deref(), Some(raw));
        assert!(calls.is_none());
    }

    #[test]
    fn malformed_tool_call_falls_through_as_content() {
        let raw = "<tool_call>not even json</tool_call>";
        let (content, calls) = parse_tool_calls(raw);
        assert!(content.is_some());
        assert!(calls.is_none());
    }

    #[test]
    fn holdback_returns_longest_prefix_of_marker() {
        assert_eq!(longest_suffix_that_is_prefix_of("hello", "<tool_call>"), 0);
        assert_eq!(longest_suffix_that_is_prefix_of("hi<", "<tool_call>"), 1);
        assert_eq!(longest_suffix_that_is_prefix_of("hi<too", "<tool_call>"), 4);
        // "<tool_call>" is the full marker — but we never hold back the
        // entire marker (the caller has already found and consumed it).
        assert_eq!(longest_suffix_that_is_prefix_of("<tool_call", "<tool_call>"), 10);
    }

    #[test]
    fn parser_emits_plain_content_when_no_tool_call() {
        let mut p = StreamingToolCallParser::new();
        let evs = p.feed("hello ");
        assert_eq!(evs, vec![StreamEvent::Content("hello ".into())]);
        let evs = p.feed("world");
        assert_eq!(evs, vec![StreamEvent::Content("world".into())]);
        let evs = p.finish();
        assert!(evs.is_empty());
    }

    #[test]
    fn parser_holds_back_partial_open_marker() {
        let mut p = StreamingToolCallParser::new();
        // After "hi<" we must NOT emit the trailing "<" because it could be
        // the start of "<tool_call>".
        let evs = p.feed("hi<");
        assert_eq!(evs, vec![StreamEvent::Content("hi".into())]);
        // Now the trailing "<too" stays held back too.
        let evs = p.feed("too");
        assert!(evs.is_empty(), "got {evs:?}");
        // Following text isn't a tool_call marker — flush.
        let evs = p.feed("ls of the trade");
        assert_eq!(
            evs,
            vec![StreamEvent::Content("<tools of the trade".into())]
        );
    }

    #[test]
    fn parser_emits_tool_call_header_then_args() {
        let mut p = StreamingToolCallParser::new();
        let mut evs = Vec::new();
        evs.extend(p.feed("Let me check. "));
        evs.extend(p.feed("<tool_call>"));
        evs.extend(p.feed(r#"{"name":"get_weather","arguments":{"city":"Paris"}}"#));
        evs.extend(p.feed("</tool_call>"));
        evs.extend(p.finish());

        // Expect: Content("Let me check. "), Header, Args.
        let mut iter = evs.into_iter();
        match iter.next() {
            Some(StreamEvent::Content(s)) => assert_eq!(s, "Let me check. "),
            other => panic!("expected Content, got {other:?}"),
        }
        match iter.next() {
            Some(StreamEvent::ToolCallHeader { index, id, name }) => {
                assert_eq!(index, 0);
                assert!(id.starts_with("call_"));
                assert_eq!(name, "get_weather");
            }
            other => panic!("expected ToolCallHeader, got {other:?}"),
        }
        match iter.next() {
            Some(StreamEvent::ToolCallArgs { index, args }) => {
                assert_eq!(index, 0);
                assert!(args.contains("Paris"));
            }
            other => panic!("expected ToolCallArgs, got {other:?}"),
        }
        assert!(iter.next().is_none(), "expected exactly 3 events");
    }

    #[test]
    fn parser_handles_multiple_tool_calls_with_text_between() {
        let mut p = StreamingToolCallParser::new();
        let mut evs = Vec::new();
        evs.extend(p.feed(r#"<tool_call>{"name":"f1","arguments":{"a":1}}</tool_call>"#));
        evs.extend(p.feed(" middle "));
        evs.extend(p.feed(r#"<tool_call>{"name":"f2","arguments":{"b":2}}</tool_call>"#));
        evs.extend(p.finish());

        let names: Vec<String> = evs
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCallHeader { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(names, vec!["f1", "f2"]);
        let indices: Vec<usize> = evs
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCallHeader { index, .. } => Some(*index),
                _ => None,
            })
            .collect();
        assert_eq!(indices, vec![0, 1]);
        let content: String = evs
            .iter()
            .filter_map(|e| match e {
                StreamEvent::Content(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(content, " middle ");
    }

    #[test]
    fn parser_streams_tool_call_split_across_many_feeds() {
        // The marker, name field, and args all arrive in tiny pieces, like
        // a slow tokenizer would produce.
        let chunks = [
            "<tool",
            "_call>",
            "{\"na",
            "me\":\"",
            "get_",
            "time",
            "\",\"argu",
            "ments\":",
            "{}}",
            "</tool",
            "_call>",
        ];
        let mut p = StreamingToolCallParser::new();
        let mut evs = Vec::new();
        for c in chunks {
            evs.extend(p.feed(c));
        }
        evs.extend(p.finish());
        // Should produce no Content events (everything was inside a tool_call).
        assert!(
            !evs.iter().any(|e| matches!(e, StreamEvent::Content(_))),
            "got unexpected content event: {evs:?}",
        );
        let names: Vec<_> = evs
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCallHeader { name, .. } => Some(name.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(names, vec!["get_time"]);
    }

    #[test]
    fn parser_malformed_block_falls_back_to_content() {
        let mut p = StreamingToolCallParser::new();
        let evs = p.feed("<tool_call>not even json</tool_call>");
        assert!(
            matches!(evs.first(), Some(StreamEvent::Content(s)) if s.contains("not even json")),
            "got {evs:?}",
        );
    }

    #[test]
    fn parser_unterminated_block_flushes_on_finish() {
        let mut p = StreamingToolCallParser::new();
        let mid = p.feed("<tool_call>{\"name\":\"f\",\"argu");
        assert!(mid.is_empty(), "got {mid:?}");
        let end = p.finish();
        match end.as_slice() {
            [StreamEvent::Content(s)] => {
                assert!(s.starts_with("<tool_call>"));
                assert!(s.contains("argu"));
            }
            other => panic!("expected single Content, got {other:?}"),
        }
    }

    #[test]
    fn parser_arguments_string_form_passes_through() {
        // arguments may be a JSON-encoded string rather than an object.
        let mut p = StreamingToolCallParser::new();
        let evs =
            p.feed(r#"<tool_call>{"name":"x","arguments":"{\"a\":1}"}</tool_call>"#);
        let args: Vec<String> = evs
            .iter()
            .filter_map(|e| match e {
                StreamEvent::ToolCallArgs { args, .. } => Some(args.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(args, vec![r#"{"a":1}"#]);
    }

    #[test]
    fn arguments_string_form_passes_through() {
        // OpenAI clients sometimes send `arguments` as an escaped string.
        // We should accept either nested object or pre-stringified.
        let raw = r#"<tool_call>{"name": "x", "arguments": "{\"a\":1}"}</tool_call>"#;
        let call = parse_tool_call_body(
            r#"{"name": "x", "arguments": "{\"a\":1}"}"#,
        )
        .expect("parse");
        assert_eq!(call.function.name, "x");
        assert_eq!(call.function.arguments, "{\"a\":1}");
        let _ = raw;
    }

    // ---- multi-format post-hoc tool-call parsing ----------------------

    #[test]
    fn parse_tool_call_body_accepts_parameters_field() {
        // Llama-3.1 emits `parameters` rather than `arguments`.
        let call = parse_tool_call_body(r#"{"name": "get_weather", "parameters": {"city": "Paris"}}"#)
            .expect("parse");
        assert_eq!(call.function.name, "get_weather");
        assert!(call.function.arguments.contains("Paris"));
    }

    #[test]
    fn parse_tool_call_body_unwraps_function_wrapper() {
        // Some clients nest the call under a `function` key.
        let call = parse_tool_call_body(
            r#"{"function": {"name": "f", "arguments": {"a": 1}}}"#,
        )
        .expect("parse");
        assert_eq!(call.function.name, "f");
        assert!(call.function.arguments.contains("\"a\""));
    }

    #[test]
    fn parses_llama3_python_tag_tool_call() {
        let raw = r#"<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}"#;
        let (content, calls) = parse_tool_calls(raw);
        assert!(content.is_none(), "python_tag body should be all tool call");
        let calls = calls.expect("expected tool_calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("Paris"));
    }

    #[test]
    fn parses_multiple_python_tag_calls_comma_separated() {
        let raw = r#"<|python_tag|>{"name": "f1", "arguments": {"a": 1}}, {"name": "f2", "arguments": {"b": 2}}"#;
        let (_content, calls) = parse_tool_calls(raw);
        let calls = calls.expect("expected tool_calls");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "f1");
        assert_eq!(calls[1].function.name, "f2");
    }

    #[test]
    fn parses_mistral_tool_calls_array() {
        let raw = r#"[TOOL_CALLS][{"name": "get_weather", "arguments": {"city": "Tokyo"}}]"#;
        let (content, calls) = parse_tool_calls(raw);
        assert!(content.is_none());
        let calls = calls.expect("expected tool_calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert!(calls[0].function.arguments.contains("Tokyo"));
    }

    #[test]
    fn parses_bare_json_array_of_tool_calls() {
        // The entire output is a JSON array of tool-call objects.
        let raw = r#"[{"name": "f1", "arguments": {"a": 1}}, {"name": "f2", "arguments": {"b": 2}}]"#;
        let (content, calls) = parse_tool_calls(raw);
        assert!(content.is_none());
        let calls = calls.expect("expected tool_calls");
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "f1");
        assert_eq!(calls[1].function.name, "f2");
    }

    #[test]
    fn parses_bare_json_object_in_fenced_block() {
        let raw = "```json\n{\"name\": \"f\", \"arguments\": {\"x\": 1}}\n```";
        let (_content, calls) = parse_tool_calls(raw);
        let calls = calls.expect("expected tool_calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "f");
    }

    #[test]
    fn bare_json_data_object_without_args_is_not_a_tool_call() {
        // A plain data object that happens to carry a `name` must NOT be
        // misread as a tool call (strict whole-output heuristic).
        let raw = r#"{"name": "Alice", "age": 30}"#;
        let (content, calls) = parse_tool_calls(raw);
        assert!(calls.is_none(), "data object misdetected as tool call");
        assert!(content.is_some());
    }

    #[test]
    fn tool_call_marker_still_wins_over_fallbacks() {
        // When a real `<tool_call>` block is present it takes priority
        // and the fallbacks don't run.
        let raw = r#"<tool_call>{"name": "primary", "arguments": {}}</tool_call>"#;
        let (_content, calls) = parse_tool_calls(raw);
        let calls = calls.expect("tool_calls");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "primary");
    }

    // ---- tool_choice parsing ------------------------------------------

    #[test]
    fn parse_tool_choice_maps_all_shapes() {
        assert_eq!(parse_tool_choice(&None), ToolChoice::Auto);
        assert_eq!(parse_tool_choice(&Some(json!("auto"))), ToolChoice::Auto);
        assert_eq!(parse_tool_choice(&Some(Value::Null)), ToolChoice::Auto);
        assert_eq!(parse_tool_choice(&Some(json!("none"))), ToolChoice::None);
        assert_eq!(parse_tool_choice(&Some(json!("required"))), ToolChoice::Required);
        assert_eq!(
            parse_tool_choice(&Some(json!({"type": "function", "function": {"name": "get_weather"}}))),
            ToolChoice::Function("get_weather".to_string())
        );
        // Unknown / malformed → Auto.
        assert_eq!(parse_tool_choice(&Some(json!("banana"))), ToolChoice::Auto);
        assert_eq!(parse_tool_choice(&Some(json!({"type": "function"}))), ToolChoice::Auto);
    }

    #[test]
    fn tool_choice_field_deserializes_on_chat_request() {
        let value = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "tool_choice": "required",
        });
        let req: ChatRequest = serde_json::from_value(value).expect("parses");
        assert_eq!(parse_tool_choice(&req.tool_choice), ToolChoice::Required);
    }

    // ---- multi-turn round-trip message rendering ----------------------

    #[test]
    fn wire_messages_to_json_preserves_tool_roundtrip_fields() {
        // assistant-with-tool_calls (content: null) + a tool result.
        let value = json!([
            {"role": "user", "content": "weather in Paris?"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                }]
            },
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather", "content": "18C"}
        ]);
        let wire: Vec<ChatMessageWire> = serde_json::from_value(value).expect("parses");
        let msgs = wire_messages_to_json(&wire);
        assert_eq!(msgs.len(), 3);
        // assistant turn keeps tool_calls, content coerced to "".
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "");
        assert!(msgs[1]["tool_calls"].is_array());
        assert_eq!(msgs[1]["tool_calls"][0]["function"]["name"], "get_weather");
        // tool turn keeps tool_call_id + name + content.
        assert_eq!(msgs[2]["role"], "tool");
        assert_eq!(msgs[2]["tool_call_id"], "call_1");
        assert_eq!(msgs[2]["name"], "get_weather");
        assert_eq!(msgs[2]["content"], "18C");
        // A plain message omits the optional fields entirely.
        assert!(msgs[0].get("tool_calls").is_none());
        assert!(msgs[0].get("tool_call_id").is_none());
    }

    // ---- host-environment hint injection (change #6) ------------------

    #[test]
    fn env_hint_prepends_new_system_when_absent() {
        let block = "[Host environment]\nOS: Test (x)";
        let msgs = vec![ChatMessageWire {
            role: "user".into(),
            content: "hi".into(),
            ..Default::default()
        }];
        let out = super::prepend_system(msgs, block);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].role, "system");
        assert!(out[0].content.text().starts_with("[Host environment]"));
        assert_eq!(out[1].role, "user");
    }

    #[test]
    fn env_hint_appends_to_existing_system() {
        let block = "[Host environment]\nOS: Test (x)";
        let msgs = vec![
            ChatMessageWire {
                role: "system".into(),
                content: "be nice".into(),
                ..Default::default()
            },
            ChatMessageWire {
                role: "user".into(),
                content: "hi".into(),
                ..Default::default()
            },
        ];
        let out = super::prepend_system(msgs, block);
        assert_eq!(out.len(), 2, "must not add a second system message");
        assert_eq!(out[0].role, "system");
        let sys = out[0].content.text();
        assert!(sys.contains("be nice"), "keeps the user system prompt");
        assert!(sys.contains("[Host environment]"), "appends the block");
        assert_eq!(out[1].role, "user");
    }

    #[test]
    fn host_environment_hint_names_this_os() {
        let hint = crate::env_hint::host_environment_hint();
        if cfg!(windows) {
            assert!(hint.contains("Windows"), "hint: {hint}");
        } else {
            assert!(
                hint.contains("Linux") || hint.contains("macOS"),
                "hint: {hint}"
            );
        }
    }

    // ---- diff (SEARCH/REPLACE) parser tests ---------------------------

    #[test]
    fn parses_single_diff_block_with_filename() {
        let raw = "src/main.rs\n<<<<<<< SEARCH\nlet x = 1;\n=======\nlet x = 2;\n>>>>>>> REPLACE";
        let blocks = super::parse_diff_blocks(raw).expect("at least one block");
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].file.as_deref(), Some("src/main.rs"));
        assert_eq!(blocks[0].search, "let x = 1;");
        assert_eq!(blocks[0].replace, "let x = 2;");
    }

    #[test]
    fn parses_multi_line_search_and_replace_bodies() {
        let raw = "lib.rs\n<<<<<<< SEARCH\nfn foo() {\n    a\n    b\n}\n=======\nfn foo() {\n    new\n    bar\n}\n>>>>>>> REPLACE";
        let blocks = super::parse_diff_blocks(raw).unwrap();
        assert_eq!(blocks[0].search, "fn foo() {\n    a\n    b\n}");
        assert_eq!(blocks[0].replace, "fn foo() {\n    new\n    bar\n}");
    }

    #[test]
    fn parses_multiple_diff_blocks() {
        let raw = "a.rs\n<<<<<<< SEARCH\nfoo\n=======\nbar\n>>>>>>> REPLACE\nb.rs\n<<<<<<< SEARCH\nbaz\n=======\nqux\n>>>>>>> REPLACE";
        let blocks = super::parse_diff_blocks(raw).unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].file.as_deref(), Some("a.rs"));
        assert_eq!(blocks[0].search, "foo");
        assert_eq!(blocks[0].replace, "bar");
        assert_eq!(blocks[1].file.as_deref(), Some("b.rs"));
        assert_eq!(blocks[1].search, "baz");
        assert_eq!(blocks[1].replace, "qux");
    }

    #[test]
    fn skips_markdown_fences_when_finding_filename() {
        let raw = "Here is the edit:\n\n```rust\nsrc/lib.rs\n```\n<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE";
        let blocks = super::parse_diff_blocks(raw).unwrap();
        assert_eq!(blocks[0].file.as_deref(), Some("src/lib.rs"));
    }

    #[test]
    fn empty_diff_block_search_and_replace_are_recognized() {
        // Pure file-create: empty search, non-empty replace.
        let raw = "new.txt\n<<<<<<< SEARCH\n=======\nhello\n>>>>>>> REPLACE";
        let blocks = super::parse_diff_blocks(raw).unwrap();
        assert_eq!(blocks[0].search, "");
        assert_eq!(blocks[0].replace, "hello");
    }

    #[test]
    fn truncated_block_returns_none_or_skips() {
        // No closing marker → no blocks recognized.
        let raw = "x.rs\n<<<<<<< SEARCH\nfoo\n=======\nbar\n";
        assert!(super::parse_diff_blocks(raw).is_none());
    }

    #[test]
    fn no_diff_markers_returns_none() {
        let raw = "Just some prose. No edits here.";
        assert!(super::parse_diff_blocks(raw).is_none());
    }

    #[test]
    fn tolerates_missing_space_between_markers_and_keywords() {
        // Small models often emit `<<<<<<<SEARCH` without the space.
        // We accept any whitespace (including none) between the angle
        // brackets and the keyword.
        let raw = "src/lib.rs\n<<<<<<<SEARCH\nold\n=======\nnew\n>>>>>>>REPLACE";
        let blocks = super::parse_diff_blocks(raw).expect("space-less markers");
        assert_eq!(blocks[0].file.as_deref(), Some("src/lib.rs"));
        assert_eq!(blocks[0].search, "old");
        assert_eq!(blocks[0].replace, "new");
    }

    #[test]
    fn tolerates_separator_run_longer_or_shorter_than_seven_equals() {
        // ≥ 3 equals on a line is enough for the SEP recognizer.
        let raw_long = "f.rs\n<<<<<<< SEARCH\nold\n==========\nnew\n>>>>>>> REPLACE";
        let raw_short = "f.rs\n<<<<<<< SEARCH\nold\n===\nnew\n>>>>>>> REPLACE";
        let a = super::parse_diff_blocks(raw_long).unwrap();
        let b = super::parse_diff_blocks(raw_short).unwrap();
        assert_eq!(a[0].search, "old");
        assert_eq!(b[0].search, "old");
        assert_eq!(a[0].replace, "new");
        assert_eq!(b[0].replace, "new");
    }

    #[test]
    fn diff_response_format_appends_system_prompt() {
        let msgs = vec![ChatMessageWire {
            role: "user".into(),
            content: "fix the bug".into(),
            ..Default::default()
        }];
        let rf = Some(super::ResponseFormat {
            kind: "diff".into(),
            json_schema: None,
            pattern: None,
        });
        let out = super::apply_response_format(msgs, &rf);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].role, "system");
        assert!(out[0].content.text().contains("SEARCH/REPLACE"));
        assert!(out[0].content.text().contains(">>>>>>> REPLACE"));
    }

    /// End-to-end SSE wiring proof: feed `build_sse_stream` a synthetic
    /// `TokenStream` whose tokens spell out a complete SEARCH/REPLACE
    /// block, then assert the final chunk carries the parsed `diffs`
    /// array. Runs entirely on the mock engine — no model required.
    #[tokio::test]
    async fn streaming_chat_in_diff_mode_emits_diffs_on_final_chunk() {
        use futures::StreamExt;
        use rustllama_engine::Token;
        use std::sync::Arc;
        use tokio::sync::Semaphore;

        // Tokens that compose a complete SEARCH/REPLACE block when
        // concatenated. Whitespace and newlines are real.
        let token_strs: Vec<String> = [
            "src/lib.rs\n",
            "<<<<<<< SEARCH\n",
            "fn add(a: u32, b: u32) -> u32 {\n",
            "    a - b\n",
            "}\n",
            "=======\n",
            "fn add(a: u32, b: u32) -> u32 {\n",
            "    a + b\n",
            "}\n",
            ">>>>>>> REPLACE",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let stream = futures::stream::iter(
            token_strs
                .into_iter()
                .enumerate()
                .map(|(i, t)| {
                    Ok(Token {
                        id: i as u32,
                        text: t,
                        logprobs: None,
                    })
                })
                .collect::<Vec<_>>(),
        );
        let tok_stream: rustllama_engine::TokenStream = Box::pin(stream);

        // Synthetic permit that the diff path can drop at the end.
        let gate = Arc::new(Semaphore::new(1));
        let owned = gate.clone().try_acquire_owned().expect("permit");
        let permit = crate::EngineHandle {
            engine: Arc::new(rustllama_engine::MockEngine) as Arc<dyn rustllama_engine::Engine>,
            cpu_engine: None,
            engine_idx: 0,
            _permit: Some(owned),
            pending: Arc::new(std::sync::atomic::AtomicUsize::new(1)),
            // Synthetic guard: no scheduler bookkeeping needed.
            scheduler: None,
            request_id: None,
        };

        // Synthetic cancel guard — never fires; we just need the field
        // to satisfy the build_sse_stream signature.
        let cancel_guard = crate::CancelGuard {
            id: "test-id".into(),
            flag: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            cancellations: Arc::new(std::sync::Mutex::new(
                std::collections::HashMap::new(),
            )),
        };

        let sse = super::build_sse_stream(
            "test-id".into(),
            0,
            "test-model".into(),
            tok_stream,
            permit,
            None, // cpu
            42,   // prompt_tokens
            false, // include_usage
            true,  // diff_mode
            cancel_guard,
            256,  // max_tokens — high enough to avoid the length-cap branch
            String::new(), // system_fingerprint — empty omits the field
        );
        let events: Vec<_> = sse.collect().await;
        // Build a single string of all event payloads so we can match on
        // the final chunk regardless of formatting.
        let combined: String = events
            .iter()
            .filter_map(|r| r.as_ref().ok())
            .map(|e| format!("{e:?}"))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            combined.contains("\\\"finish_reason\\\":\\\"stop\\\""),
            "expected stop finish_reason, got: {combined}",
        );
        assert!(
            combined.contains("\\\"diffs\\\""),
            "expected diffs key in final chunk, got: {combined}",
        );
        assert!(
            combined.contains("\\\"src/lib.rs\\\""),
            "expected diff file 'src/lib.rs' in output, got: {combined}",
        );
        assert!(
            combined.contains("a + b"),
            "expected 'a + b' in the parsed replace body, got: {combined}",
        );
    }
}
