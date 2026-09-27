//! Tokenizer + chat-template support backed by HuggingFace `tokenizers`.
//!
//! Two construction paths:
//!   - [`Tokenizer::from_file`] — load a HF `tokenizer.json` from disk.
//!   - [`Tokenizer::from_gguf`] — build directly from GGUF metadata.
//!
//! Supported `tokenizer.ggml.model` values:
//!   - `gpt2` — byte-level BPE (Qwen2 / Qwen2.5 / Llama-3 / DeepSeek)
//!   - `llama` — SentencePiece / Unigram (Llama-1/2, Mistral, TinyLlama, Phi-3)
//!   - `bert` — WordPiece with BertNormalizer + BertPreTokenizer +
//!     TemplateProcessing for `[CLS] … [SEP]` wrapping. Used by BGE / E5 /
//!     sentence-transformers BERT embedding models — feeds `/v1/embeddings`
//!     text input. The wrapping is applied automatically when
//!     `add_special_tokens=true` is passed to [`Tokenizer::encode`].

use std::collections::HashMap;
use std::path::Path;

use rustllama_gguf::{Gguf, MetadataValue};
use serde_json::{json, Value};
use tokenizers::Tokenizer as HfTokenizer;

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("failed to load tokenizer: {0}")]
    Load(String),
    #[error("failed to encode: {0}")]
    Encode(String),
    #[error("failed to decode: {0}")]
    Decode(String),
    #[error("required GGUF metadata key missing: {0}")]
    MissingMeta(&'static str),
    #[error("GGUF metadata key {0} has unexpected type")]
    BadMetaType(&'static str),
    #[error("unsupported GGUF tokenizer model: {0:?} (supported: `gpt2`, `llama`, `bert`)")]
    UnsupportedModel(String),
    #[error("chat template render: {0}")]
    Template(String),
    #[error(
        "no chat template embedded in GGUF; specify one in config or use a model that carries one"
    )]
    NoChatTemplate,
}

pub type Result<T> = std::result::Result<T, TokenizerError>;

/// Chat message used by [`Tokenizer::render_chat`]. Matches the shape that
/// most embedded GGUF chat templates expect.
#[derive(Debug, Clone)]
pub struct ChatMessage<'a> {
    pub role: &'a str,
    pub content: &'a str,
}

/// Fill-In-Middle special token IDs for a model that supports FIM completions.
#[derive(Debug, Clone, Copy)]
pub struct FimTokens {
    pub prefix: u32,
    pub suffix: u32,
    pub middle: u32,
}

pub struct Tokenizer {
    inner: HfTokenizer,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    add_bos: bool,
    add_eos: bool,
    chat_template: Option<String>,
}

impl Tokenizer {
    /// Load a `tokenizer.json` file from disk. Special-token IDs and chat
    /// template are not populated by this path.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner =
            HfTokenizer::from_file(path).map_err(|e| TokenizerError::Load(e.to_string()))?;
        Ok(Self {
            inner,
            bos_token_id: None,
            eos_token_id: None,
            add_bos: false,
            add_eos: false,
            chat_template: None,
        })
    }

    /// Build a tokenizer from a GGUF file's metadata. The strategy:
    ///   1. read `tokenizer.ggml.model` to choose the model family
    ///   2. assemble a `tokenizer.json`-shaped JSON document
    ///   3. let `tokenizers::Tokenizer::from_bytes` parse it
    /// This avoids depending on internals of the `tokenizers` crate and
    /// gives us robust GPT2 byte-level behavior for free.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let model = read_string(gguf, "tokenizer.ggml.model")?;
        let tokens = read_string_array(gguf, "tokenizer.ggml.tokens")?;

        let bos = read_u32_opt(gguf, "tokenizer.ggml.bos_token_id");
        let eos = read_u32_opt(gguf, "tokenizer.ggml.eos_token_id");
        let pad = read_u32_opt(gguf, "tokenizer.ggml.padding_token_id");
        let unk = read_u32_opt(gguf, "tokenizer.ggml.unknown_token_id");
        let add_bos = read_bool_opt(gguf, "tokenizer.ggml.add_bos_token").unwrap_or(false);
        let add_eos = read_bool_opt(gguf, "tokenizer.ggml.add_eos_token").unwrap_or(false);
        let token_types = read_i32_array_opt(gguf, "tokenizer.ggml.token_type");
        // GGUF spec was ambiguous about the key for chat templates. Both
        // forms are in the wild — Qwen/DeepSeek use `tokenizer.chat_template`
        // (no ggml infix), older Llama-family conversions use
        // `tokenizer.ggml.chat_template`. Accept either.
        let chat_template = read_string_opt(gguf, "tokenizer.chat_template")
            .or_else(|| read_string_opt(gguf, "tokenizer.ggml.chat_template"));

        let inner = match model.as_str() {
            "gpt2" => build_bpe_byte_level(&tokens, gguf, &token_types, bos, eos, pad, unk)?,
            "llama" => build_unigram_spm(&tokens, gguf, &token_types, bos, eos, pad, unk)?,
            "bert" => {
                // BERT carries CLS/SEP/MASK rather than BOS/EOS. The
                // GGUF spec stores them either under the dedicated keys
                // (`tokenizer.ggml.cls_token_id`, etc.) or — for some
                // older sentence-transformers conversions — reuses the
                // `bos_token_id` slot for CLS and `eos_token_id` for
                // SEP. We accept both.
                let cls = read_u32_opt(gguf, "tokenizer.ggml.cls_token_id").or(bos);
                let sep = read_u32_opt(gguf, "tokenizer.ggml.sep_token_id").or(eos);
                let mask = read_u32_opt(gguf, "tokenizer.ggml.mask_token_id");
                build_wordpiece_bert(&tokens, &token_types, cls, sep, pad, unk, mask)?
            }
            other => return Err(TokenizerError::UnsupportedModel(other.to_string())),
        };

        Ok(Self {
            inner,
            bos_token_id: bos,
            eos_token_id: eos,
            add_bos,
            add_eos,
            chat_template,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    pub fn add_bos_token(&self) -> bool {
        self.add_bos
    }

    pub fn add_eos_token(&self) -> bool {
        self.add_eos
    }

    /// String form of the BOS token (e.g. `"<s>"` for Llama family) for
    /// passing into chat-template renderers. `None` if no BOS is set or
    /// decode fails.
    pub fn bos_token_str(&self) -> Option<String> {
        let id = self.bos_token_id?;
        self.decode_single(id, false).ok()
    }

    pub fn eos_token_str(&self) -> Option<String> {
        let id = self.eos_token_id?;
        self.decode_single(id, false).ok()
    }

    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    /// Look up Fill-In-Middle special-token IDs for the model. Tries the
    /// known templates in order; returns `None` if the tokenizer has no FIM
    /// support (e.g. base chat models, vanilla Llama).
    pub fn fim_tokens(&self) -> Option<FimTokens> {
        // Qwen2.5-Coder / Qwen3-Coder
        if let (Some(p), Some(s), Some(m)) = (
            self.token_to_id("<|fim_prefix|>"),
            self.token_to_id("<|fim_suffix|>"),
            self.token_to_id("<|fim_middle|>"),
        ) {
            return Some(FimTokens {
                prefix: p,
                suffix: s,
                middle: m,
            });
        }
        // DeepSeek-Coder
        if let (Some(p), Some(s), Some(m)) = (
            self.token_to_id("<｜fim▁begin｜>"),
            self.token_to_id("<｜fim▁hole｜>"),
            self.token_to_id("<｜fim▁end｜>"),
        ) {
            return Some(FimTokens {
                prefix: p,
                suffix: s,
                middle: m,
            });
        }
        // StarCoder / CodeLlama-Instruct
        if let (Some(p), Some(s), Some(m)) = (
            self.token_to_id("<fim_prefix>"),
            self.token_to_id("<fim_suffix>"),
            self.token_to_id("<fim_middle>"),
        ) {
            return Some(FimTokens {
                prefix: p,
                suffix: s,
                middle: m,
            });
        }
        None
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Encode a sequence pair (e.g. `query` + `document` for a BGE
    /// cross-encoder reranker). The `pair` post-processor template
    /// installed by [`build_wordpiece_bert`] wraps as
    /// `[CLS] a [SEP] b [SEP]` with proper token-type ids. Returns
    /// the concatenated id list — type ids are not surfaced
    /// because the BERT forward path inside `rustllama-models`
    /// receives them implicitly (positions in the input slice
    /// before/after the first SEP map to type 0/1).
    pub fn encode_pair(
        &self,
        a: &str,
        b: &str,
        add_special_tokens: bool,
    ) -> Result<Vec<u32>> {
        let enc = self
            .inner
            .encode((a, b), add_special_tokens)
            .map_err(|e| TokenizerError::Encode(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    /// Streaming-friendly variant of [`encode`]. Splits the input at
    /// chat-template `<|...|>` special-token boundaries and yields each
    /// segment's tokens as a separate chunk through a sync channel.
    ///
    /// Segment 0 is encoded synchronously on the caller's thread so a
    /// downstream prefill loop can start consuming tokens immediately.
    /// Remaining segments are encoded in parallel via `encode_batch`
    /// (rayon under the hood) on a background thread; each batch result
    /// is sent in order as the parallel encode completes.
    ///
    /// Concatenating all received chunks reproduces exactly what a
    /// single [`encode`] call on the full input would have returned —
    /// the split point `<|` is always immediately followed by a special
    /// token, which BPE cannot fuse with surrounding text, so splitting
    /// there is byte-safe.
    pub fn encode_streaming(
        &self,
        text: &str,
        add_special_tokens: bool,
    ) -> Result<TokenStream> {
        let segments = split_at_special_tokens(text);
        let (tx, rx) = std::sync::mpsc::channel::<Result<Vec<u32>>>();
        if segments.is_empty() {
            // Empty prompt — close the channel immediately.
            drop(tx);
            return Ok(TokenStream { rx });
        }

        // First segment: encode synchronously on the caller's thread.
        let first_ids = self.encode(segments[0], add_special_tokens)?;
        let _ = tx.send(Ok(first_ids));

        if segments.len() == 1 {
            drop(tx);
            return Ok(TokenStream { rx });
        }

        // Remaining segments: parallel encode_batch on a worker thread.
        let rest: Vec<String> = segments[1..].iter().map(|s| (*s).to_string()).collect();
        // HfTokenizer is internally Arc-shared, so this clone is cheap
        // and safe to send to another thread.
        let inner = self.inner.clone();
        std::thread::spawn(move || {
            let refs: Vec<&str> = rest.iter().map(String::as_str).collect();
            match inner.encode_batch(refs, add_special_tokens) {
                Ok(encs) => {
                    for enc in encs {
                        let ids: Vec<u32> = enc.get_ids().to_vec();
                        if tx.send(Ok(ids)).is_err() {
                            return;
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(TokenizerError::Encode(e.to_string())));
                }
            }
        });

        Ok(TokenStream { rx })
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| TokenizerError::Decode(e.to_string()))
    }

    /// Decode a *single* token to its surface form. Useful for streaming.
    pub fn decode_single(&self, id: u32, skip_special_tokens: bool) -> Result<String> {
        self.decode(&[id], skip_special_tokens)
    }

    /// Render the embedded chat template into a prompt string. Exposes
    /// `messages`, `add_generation_prompt`, and optionally `tools` to the
    /// template. Most modern coding-model templates branch on `tools` to
    /// inject function-calling system prompts (e.g. Qwen2.5-Coder).
    pub fn render_chat(
        &self,
        messages: &[ChatMessage<'_>],
        add_generation_prompt: bool,
    ) -> Result<String> {
        self.render_chat_with_tools(messages, add_generation_prompt, None)
    }

    /// Render the chat template with an optional `tools` JSON array.
    /// Pass `None` for no tools (equivalent to [`render_chat`]).
    ///
    /// The configured BOS/EOS strings are forwarded to the template context
    /// (`bos_token`, `eos_token`) — many HF chat templates reference them.
    pub fn render_chat_with_tools(
        &self,
        messages: &[ChatMessage<'_>],
        add_generation_prompt: bool,
        tools: Option<&Value>,
    ) -> Result<String> {
        let template = self
            .chat_template
            .as_deref()
            .ok_or(TokenizerError::NoChatTemplate)?;
        let bos = self.bos_token_str();
        let eos = self.eos_token_str();
        render_chat_template_with_specials(
            template,
            messages,
            add_generation_prompt,
            tools,
            bos.as_deref(),
            eos.as_deref(),
        )
    }

    /// Render the chat template from raw `serde_json::Value` messages.
    ///
    /// Unlike [`render_chat_with_tools`], each message is passed to the
    /// template exactly as supplied rather than being flattened to
    /// `{role, content}`. This preserves the extra fields multi-turn
    /// tool round-trips rely on — an assistant message's `tool_calls`
    /// and a `tool`-role message's `tool_call_id` / `name` — so the
    /// template can re-render prior tool turns. BOS/EOS are forwarded to
    /// the template context the same way as the other render paths.
    pub fn render_chat_messages_json(
        &self,
        messages: &[Value],
        add_generation_prompt: bool,
        tools: Option<&Value>,
    ) -> Result<String> {
        let template = self
            .chat_template
            .as_deref()
            .ok_or(TokenizerError::NoChatTemplate)?;
        let bos = self.bos_token_str();
        let eos = self.eos_token_str();
        render_chat_template_with_specials_json(
            template,
            messages,
            add_generation_prompt,
            tools,
            bos.as_deref(),
            eos.as_deref(),
        )
    }
}

/// Streaming receiver returned by [`Tokenizer::encode_streaming`].
/// Each `recv` yields the next ordered chunk of token ids. The channel
/// closes once all chunks have been pushed; iterate via [`Self::next`]
/// or fully drain via [`Self::collect_into_vec`].
pub struct TokenStream {
    pub(crate) rx: std::sync::mpsc::Receiver<Result<Vec<u32>>>,
}

impl TokenStream {
    /// Block until the next chunk arrives. `None` means the worker has
    /// finished and no more chunks will arrive.
    pub fn next(&self) -> Option<Result<Vec<u32>>> {
        self.rx.recv().ok()
    }

    /// Try to fetch the next chunk without blocking. Returns `Err(Empty)`
    /// if the worker hasn't produced a chunk yet, `Err(Disconnected)`
    /// once it's done. Useful when the consumer wants to interleave
    /// other work (e.g., prefill forward passes) between waits.
    pub fn try_next(&self) -> std::result::Result<Result<Vec<u32>>, std::sync::mpsc::TryRecvError> {
        self.rx.try_recv()
    }

    /// Drain the channel into a single `Vec<u32>`. The result equals
    /// the concatenation of every chunk.
    pub fn collect_into_vec(self) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        while let Ok(chunk) = self.rx.recv() {
            out.extend_from_slice(&chunk?);
        }
        Ok(out)
    }
}

/// Split a rendered prompt at chat-template special-token boundaries.
/// Each `<|` starts a new segment, except possibly a prefix before the
/// first such marker. Returns the segments in document order; their
/// concatenation equals the original text.
///
/// BPE never fuses across a special token (the tokenizer treats them
/// as atomic), so splitting here is byte-safe — encoding each segment
/// in isolation and concatenating the resulting token id streams gives
/// the same result as a single encode of the full string.
pub fn split_at_special_tokens(text: &str) -> Vec<&str> {
    let mut segments: Vec<&str> = Vec::new();
    let mut last = 0usize;
    for (idx, _) in text.match_indices("<|") {
        if idx == 0 {
            // No prefix before the first special token; let it lead the
            // first segment.
            continue;
        }
        if idx > last {
            segments.push(&text[last..idx]);
            last = idx;
        }
    }
    if last < text.len() {
        segments.push(&text[last..]);
    }
    segments
}

/// Standalone template renderer — exposed so callers with a custom template
/// (not loaded from GGUF) can use the same code path.
pub fn render_chat_template(
    template: &str,
    messages: &[ChatMessage<'_>],
    add_generation_prompt: bool,
) -> Result<String> {
    render_chat_template_with_specials(template, messages, add_generation_prompt, None, None, None)
}

pub fn render_chat_template_with_tools(
    template: &str,
    messages: &[ChatMessage<'_>],
    add_generation_prompt: bool,
    tools: Option<&Value>,
) -> Result<String> {
    render_chat_template_with_specials(template, messages, add_generation_prompt, tools, None, None)
}

/// Full-context renderer. `bos_token` / `eos_token` are exposed to the
/// template by name (HF templates frequently reference them). Pass `None`
/// for either if the tokenizer doesn't define one.
pub fn render_chat_template_with_specials(
    template: &str,
    messages: &[ChatMessage<'_>],
    add_generation_prompt: bool,
    tools: Option<&Value>,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
) -> Result<String> {
    let env = build_chat_env(template)?;
    let tmpl = env
        .get_template("chat")
        .map_err(|e| TokenizerError::Template(e.to_string()))?;

    let msgs: Vec<Value> = messages
        .iter()
        .map(|m| json!({ "role": m.role, "content": m.content }))
        .collect();

    let tools_value: Value = tools.cloned().unwrap_or(Value::Null);

    tmpl.render(minijinja::context! {
        messages => msgs,
        add_generation_prompt => add_generation_prompt,
        tools => tools_value,
        bos_token => bos_token.unwrap_or(""),
        eos_token => eos_token.unwrap_or(""),
    })
    .map_err(|e| TokenizerError::Template(e.to_string()))
}

/// Same as [`render_chat_template_with_specials`] but the message array
/// is passed as raw `serde_json::Value` objects rather than
/// `{role, content}` maps built from [`ChatMessage`]. This lets callers
/// forward richer message shapes straight to the template — an
/// assistant turn carrying `tool_calls`, or a `tool`-role message with
/// `tool_call_id` / `name` — which is what a multi-turn tool round-trip
/// needs so the template can re-render prior tool turns verbatim. The
/// caller is responsible for the object shapes (each element should at
/// minimum carry `role` + `content`).
pub fn render_chat_template_with_specials_json(
    template: &str,
    messages: &[Value],
    add_generation_prompt: bool,
    tools: Option<&Value>,
    bos_token: Option<&str>,
    eos_token: Option<&str>,
) -> Result<String> {
    let env = build_chat_env(template)?;
    let tmpl = env
        .get_template("chat")
        .map_err(|e| TokenizerError::Template(e.to_string()))?;

    let tools_value: Value = tools.cloned().unwrap_or(Value::Null);

    tmpl.render(minijinja::context! {
        messages => messages,
        add_generation_prompt => add_generation_prompt,
        tools => tools_value,
        bos_token => bos_token.unwrap_or(""),
        eos_token => eos_token.unwrap_or(""),
    })
    .map_err(|e| TokenizerError::Template(e.to_string()))
}

/// Construct the minijinja [`Environment`](minijinja::Environment)
/// configured the way HF chat templates expect and register `template`
/// under the name `"chat"`. Shared by both the string-message
/// ([`render_chat_template_with_specials`]) and raw-JSON-message
/// ([`render_chat_template_with_specials_json`]) render paths so the
/// lenient-undefined / pycompat / `tojson` setup lives in one place.
fn build_chat_env(template: &str) -> Result<minijinja::Environment<'_>> {
    let mut env = minijinja::Environment::new();

    // minijinja defaults to strict-undefined; HF chat templates were
    // written against real Jinja2 / Python semantics where missing values
    // coerce to empty strings or `None`. Without this, templates that
    // append a possibly-undefined system message (`'...' + system_message`)
    // — as TinyLlama, Llama-1/2-Chat, and many small-model templates do —
    // fail with "tried to use + operator on unsupported types string and
    // undefined" the moment the caller omits a system message.
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Lenient);

    // HF chat templates call Python string/dict/list methods that real
    // Jinja2 inherits from Python itself — `.startswith()`, `.strip()`,
    // `.split()`, dict `.get()`, `.items()`, … (the Qwen3.6 template
    // calls `startswith` at its line 81, which 500'd every chat
    // request as "unknown method"). minijinja-contrib's pycompat
    // callback implements that method surface.
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);

    // Most modern coding-model chat templates (Qwen / DeepSeek / Llama-3)
    // call `{{ tool | tojson }}` to format function-call schemas.
    // minijinja doesn't ship this filter by default; register it.
    env.add_filter(
        "tojson",
        |value: minijinja::Value,
         indent: Option<i32>|
         -> std::result::Result<String, minijinja::Error> {
            let s = if indent.is_some() {
                serde_json::to_string_pretty(&value)
            } else {
                serde_json::to_string(&value)
            };
            s.map_err(|e| {
                minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, e.to_string())
            })
        },
    );

    env.add_template("chat", template)
        .map_err(|e| TokenizerError::Template(e.to_string()))?;
    Ok(env)
}

// ---------- internal helpers ----------

/// Build a WordPiece (BERT-style) tokenizer from GGUF metadata.
/// Covers `tokenizer.ggml.model = "bert"` — BGE / E5 / sentence-transformers
/// embedding models. The resulting tokenizer applies:
///   - BertNormalizer (lowercase + accent-strip on uncased models)
///   - BertPreTokenizer (whitespace + punctuation splitting)
///   - WordPiece model with `##` continuing-subword prefix
///   - TemplateProcessing that wraps inputs in `[CLS] … [SEP]` when
///     `add_special_tokens=true` is passed to `encode`
///
/// Lowercasing is auto-detected from the vocab: a vocab containing any
/// uppercase ASCII letter in a "normal" (non-control) token is treated
/// as cased and disables lowercasing. Most production BGE/E5 embedding
/// models are uncased, so the default path lowercases.
fn build_wordpiece_bert(
    tokens: &[String],
    token_types: &Option<Vec<i32>>,
    cls: Option<u32>,
    sep: Option<u32>,
    pad: Option<u32>,
    unk: Option<u32>,
    mask: Option<u32>,
) -> Result<HfTokenizer> {
    // Resolve special-token strings, falling back to BERT defaults if
    // the GGUF didn't pin an ID and the token happens to live in the
    // vocab. Real BGE/E5 GGUFs always set the IDs; the fallback is
    // defensive against synth fixtures / oddly-converted models.
    let resolve = |id: Option<u32>, default: &str| -> Option<String> {
        if let Some(i) = id {
            return tokens.get(i as usize).cloned();
        }
        if tokens.iter().any(|t| t == default) {
            return Some(default.to_string());
        }
        None
    };
    let cls_str = resolve(cls, "[CLS]");
    let sep_str = resolve(sep, "[SEP]");
    let unk_str = resolve(unk, "[UNK]").unwrap_or_else(|| "[UNK]".to_string());
    let pad_str = resolve(pad, "[PAD]");
    let mask_str = resolve(mask, "[MASK]");

    let cls_id = cls.and_then(|i| tokens.get(i as usize).map(|_| i));
    let sep_id = sep.and_then(|i| tokens.get(i as usize).map(|_| i));

    // Detect cased vs uncased by scanning the vocab. Production
    // embedding models that are *cased* always include uppercase
    // ASCII somewhere; uncased ones never do. Skip the special
    // tokens (control types) since those carry brackets like
    // `[CLS]` regardless.
    let lowercase = !tokens.iter().enumerate().any(|(i, t)| {
        let is_control = token_types
            .as_ref()
            .and_then(|tt| tt.get(i).copied())
            .map(|ty| matches!(ty, 3 | 4))
            .unwrap_or(false);
        !is_control && t.chars().any(|c| c.is_ascii_uppercase())
    });

    // WordPiece vocab map (token → id).
    let mut vocab_map = serde_json::Map::with_capacity(tokens.len());
    for (id, tok) in tokens.iter().enumerate() {
        vocab_map.insert(tok.clone(), json!(id));
    }

    // Register control / user-defined tokens + the special-token set
    // as added tokens so the WordPiece splitter treats them as atomic.
    let mut added_tokens: Vec<Value> = Vec::new();
    if let Some(types) = token_types {
        for (id, ty) in types.iter().enumerate() {
            if matches!(*ty, 3 | 4) {
                if let Some(tok) = tokens.get(id) {
                    push_added(&mut added_tokens, id as u32, tok, *ty == 3);
                }
            }
        }
    }
    for (opt_id, default_str, is_special) in [
        (cls, "[CLS]", true),
        (sep, "[SEP]", true),
        (pad, "[PAD]", true),
        (unk, "[UNK]", true),
        (mask, "[MASK]", true),
    ] {
        let id = match opt_id {
            Some(i) => Some(i),
            None => tokens
                .iter()
                .position(|t| t == default_str)
                .map(|p| p as u32),
        };
        if let Some(id) = id {
            if let Some(tok) = tokens.get(id as usize) {
                if !added_tokens.iter().any(|v| v["id"] == json!(id)) {
                    push_added(&mut added_tokens, id, tok, is_special);
                }
            }
        }
    }

    let mut model = serde_json::Map::new();
    model.insert("type".into(), json!("WordPiece"));
    model.insert("unk_token".into(), json!(unk_str));
    model.insert("continuing_subword_prefix".into(), json!("##"));
    model.insert("max_input_chars_per_word".into(), json!(100));
    model.insert("vocab".into(), Value::Object(vocab_map));

    // Build a TemplateProcessing post-processor that wraps the input
    // in `[CLS] … [SEP]` when `add_special_tokens=true` (the default
    // for embedding clients). This is what makes the tokenizer produce
    // a sequence the BERT model can consume directly.
    let post_processor = if let (Some(cls_s), Some(sep_s), Some(cls_i), Some(sep_i)) =
        (cls_str.as_ref(), sep_str.as_ref(), cls_id, sep_id)
    {
        let mut special_tokens = serde_json::Map::new();
        special_tokens.insert(
            cls_s.clone(),
            json!({
                "id": cls_s,
                "ids": [cls_i],
                "tokens": [cls_s],
            }),
        );
        special_tokens.insert(
            sep_s.clone(),
            json!({
                "id": sep_s,
                "ids": [sep_i],
                "tokens": [sep_s],
            }),
        );
        json!({
            "type": "TemplateProcessing",
            "single": [
                { "SpecialToken": { "id": cls_s, "type_id": 0 } },
                { "Sequence":     { "id": "A",   "type_id": 0 } },
                { "SpecialToken": { "id": sep_s, "type_id": 0 } },
            ],
            "pair": [
                { "SpecialToken": { "id": cls_s, "type_id": 0 } },
                { "Sequence":     { "id": "A",   "type_id": 0 } },
                { "SpecialToken": { "id": sep_s, "type_id": 0 } },
                { "Sequence":     { "id": "B",   "type_id": 1 } },
                { "SpecialToken": { "id": sep_s, "type_id": 1 } },
            ],
            "special_tokens": special_tokens,
        })
    } else {
        Value::Null
    };

    // Silence unused-variable warnings for the special-token strings
    // that only feed the diagnostic / future-use path.
    let _ = (pad_str, mask_str);

    let tokenizer_json = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": {
            "type": "BertNormalizer",
            "clean_text": true,
            "handle_chinese_chars": true,
            "strip_accents": null,
            "lowercase": lowercase,
        },
        "pre_tokenizer": { "type": "BertPreTokenizer" },
        "post_processor": post_processor,
        "decoder": {
            "type": "WordPiece",
            "prefix": "##",
            "cleanup": true,
        },
        "model": model,
    });

    let bytes = serde_json::to_vec(&tokenizer_json)
        .map_err(|e| TokenizerError::Load(format!("json encode: {e}")))?;
    HfTokenizer::from_bytes(&bytes).map_err(|e| TokenizerError::Load(e.to_string()))
}

/// Build a SentencePiece-style Unigram tokenizer from GGUF metadata.
/// Covers `tokenizer.ggml.model = "llama"` — Llama-1/2, Mistral, TinyLlama,
/// Phi-3, Yi. The actual scores are taken from `tokenizer.ggml.scores`.
fn build_unigram_spm(
    tokens: &[String],
    gguf: &Gguf,
    token_types: &Option<Vec<i32>>,
    bos: Option<u32>,
    eos: Option<u32>,
    pad: Option<u32>,
    unk: Option<u32>,
) -> Result<HfTokenizer> {
    let scores = read_f32_array_opt(gguf, "tokenizer.ggml.scores")
        .ok_or(TokenizerError::MissingMeta("tokenizer.ggml.scores"))?;
    if scores.len() != tokens.len() {
        return Err(TokenizerError::Load(format!(
            "scores length {} != tokens length {}",
            scores.len(),
            tokens.len()
        )));
    }

    let vocab: Vec<Value> = tokens
        .iter()
        .zip(scores.iter())
        .map(|(tok, score)| json!([tok, *score as f64]))
        .collect();

    // Build added tokens for special / control / user-defined entries so
    // the Unigram model doesn't try to split them.
    let mut added_tokens: Vec<Value> = Vec::new();
    if let Some(types) = token_types {
        for (id, ty) in types.iter().enumerate() {
            // 1=normal, 2=unknown, 3=control, 4=user-defined, 5=unused, 6=byte
            if matches!(*ty, 3 | 4) {
                if let Some(tok) = tokens.get(id) {
                    push_added(&mut added_tokens, id as u32, tok, *ty == 3);
                }
            }
        }
    }
    for (opt_id, is_special) in [(bos, true), (eos, true), (pad, true), (unk, false)] {
        if let Some(id) = opt_id {
            if let Some(tok) = tokens.get(id as usize) {
                if !added_tokens.iter().any(|v| v["id"] == json!(id)) {
                    push_added(&mut added_tokens, id, tok, is_special);
                }
            }
        }
    }

    let mut model = serde_json::Map::new();
    model.insert("type".into(), json!("Unigram"));
    if let Some(unk_id) = unk {
        model.insert("unk_id".into(), json!(unk_id));
    }
    model.insert("vocab".into(), Value::Array(vocab));

    let tokenizer_json = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": {
            "type": "Metaspace",
            "replacement": "▁",
            "prepend_scheme": "always",
            "split": true,
        },
        "post_processor": null,
        "decoder": {
            "type": "Metaspace",
            "replacement": "▁",
            "prepend_scheme": "always",
        },
        "model": model,
    });

    let bytes = serde_json::to_vec(&tokenizer_json)
        .map_err(|e| TokenizerError::Load(format!("json encode: {e}")))?;
    HfTokenizer::from_bytes(&bytes).map_err(|e| TokenizerError::Load(e.to_string()))
}

fn push_added(added: &mut Vec<Value>, id: u32, content: &str, special: bool) {
    added.push(json!({
        "id": id,
        "content": content,
        "special": special,
        "single_word": false,
        "lstrip": false,
        "rstrip": false,
        "normalized": false,
    }));
}

fn build_bpe_byte_level(
    tokens: &[String],
    gguf: &Gguf,
    token_types: &Option<Vec<i32>>,
    bos: Option<u32>,
    eos: Option<u32>,
    pad: Option<u32>,
    unk: Option<u32>,
) -> Result<HfTokenizer> {
    let merges_raw = read_string_array_opt(gguf, "tokenizer.ggml.merges").unwrap_or_default();
    let merges: Vec<Value> = merges_raw
        .iter()
        .filter_map(|m| {
            let mut parts = m.splitn(2, ' ');
            let a = parts.next()?;
            let b = parts.next()?;
            Some(json!([a, b]))
        })
        .collect();

    let mut vocab_map = serde_json::Map::with_capacity(tokens.len());
    for (id, tok) in tokens.iter().enumerate() {
        vocab_map.insert(tok.clone(), json!(id));
    }

    // Token-type IDs of 3 (control) / 4 (user-defined) / 6 (byte) get
    // registered as added tokens so the BPE doesn't try to split them.
    // Type 1 (normal) and 2 (unknown) stay in the vocab.
    let mut added_tokens: Vec<Value> = Vec::new();
    if let Some(types) = token_types {
        for (id, ty) in types.iter().enumerate() {
            // 1=normal, 2=unknown, 3=control, 4=user-defined, 5=unused, 6=byte
            if matches!(*ty, 3 | 4) {
                if let Some(tok) = tokens.get(id) {
                    push_added(&mut added_tokens, id as u32, tok, *ty == 3);
                }
            }
        }
    }
    for (opt_id, is_special) in [(bos, true), (eos, true), (pad, true), (unk, false)] {
        if let Some(id) = opt_id {
            if let Some(tok) = tokens.get(id as usize) {
                if !added_tokens.iter().any(|v| v["id"] == json!(id)) {
                    push_added(&mut added_tokens, id, tok, is_special);
                }
            }
        }
    }

    let mut model = serde_json::Map::new();
    model.insert("type".into(), json!("BPE"));
    model.insert("dropout".into(), Value::Null);
    if let Some(unk_id) = unk {
        if let Some(tok) = tokens.get(unk_id as usize) {
            model.insert("unk_token".into(), json!(tok));
        }
    }
    model.insert("continuing_subword_prefix".into(), Value::Null);
    model.insert("end_of_word_suffix".into(), Value::Null);
    model.insert("fuse_unk".into(), json!(false));
    model.insert("byte_fallback".into(), json!(false));
    model.insert("ignore_merges".into(), json!(false));
    model.insert("vocab".into(), Value::Object(vocab_map));
    model.insert("merges".into(), Value::Array(merges));

    let tokenizer_json = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": added_tokens,
        "normalizer": null,
        "pre_tokenizer": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
        },
        "post_processor": null,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
        },
        "model": model
    });

    let bytes = serde_json::to_vec(&tokenizer_json)
        .map_err(|e| TokenizerError::Load(format!("json encode: {e}")))?;
    HfTokenizer::from_bytes(&bytes).map_err(|e| TokenizerError::Load(e.to_string()))
}

fn read_string(gguf: &Gguf, key: &'static str) -> Result<String> {
    match gguf.metadata_get(key) {
        Some(MetadataValue::String(s)) => Ok(s.clone()),
        Some(_) => Err(TokenizerError::BadMetaType(key)),
        None => Err(TokenizerError::MissingMeta(key)),
    }
}

fn read_string_opt(gguf: &Gguf, key: &str) -> Option<String> {
    match gguf.metadata_get(key)? {
        MetadataValue::String(s) => Some(s.clone()),
        _ => None,
    }
}

fn read_string_array(gguf: &Gguf, key: &'static str) -> Result<Vec<String>> {
    match gguf.metadata_get(key) {
        Some(MetadataValue::Array(arr)) => arr
            .iter()
            .map(|v| match v {
                MetadataValue::String(s) => Ok(s.clone()),
                _ => Err(TokenizerError::BadMetaType(key)),
            })
            .collect(),
        Some(_) => Err(TokenizerError::BadMetaType(key)),
        None => Err(TokenizerError::MissingMeta(key)),
    }
}

fn read_string_array_opt(gguf: &Gguf, key: &str) -> Option<Vec<String>> {
    match gguf.metadata_get(key)? {
        MetadataValue::Array(arr) => arr
            .iter()
            .map(|v| match v {
                MetadataValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect::<Option<Vec<_>>>(),
        _ => None,
    }
}

fn read_i32_array_opt(gguf: &Gguf, key: &str) -> Option<Vec<i32>> {
    match gguf.metadata_get(key)? {
        MetadataValue::Array(arr) => arr
            .iter()
            .map(|v| match v {
                MetadataValue::I32(x) => Some(*x),
                MetadataValue::U32(x) => Some(*x as i32),
                _ => None,
            })
            .collect::<Option<Vec<_>>>(),
        _ => None,
    }
}

fn read_f32_array_opt(gguf: &Gguf, key: &str) -> Option<Vec<f32>> {
    match gguf.metadata_get(key)? {
        MetadataValue::Array(arr) => arr
            .iter()
            .map(|v| match v {
                MetadataValue::F32(x) => Some(*x),
                MetadataValue::F64(x) => Some(*x as f32),
                _ => None,
            })
            .collect::<Option<Vec<_>>>(),
        _ => None,
    }
}

fn read_u32_opt(gguf: &Gguf, key: &str) -> Option<u32> {
    match gguf.metadata_get(key)? {
        MetadataValue::U32(v) => Some(*v),
        MetadataValue::I32(v) if *v >= 0 => Some(*v as u32),
        MetadataValue::U64(v) if *v <= u32::MAX as u64 => Some(*v as u32),
        _ => None,
    }
}

fn read_bool_opt(gguf: &Gguf, key: &str) -> Option<bool> {
    match gguf.metadata_get(key)? {
        MetadataValue::Bool(b) => Some(*b),
        _ => None,
    }
}

#[allow(dead_code)]
fn _unused_for_future_sp(_: &HashMap<String, f64>) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_template_renders_chatml() {
        let template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}";
        let out = render_chat_template(
            template,
            &[
                ChatMessage {
                    role: "system",
                    content: "you are helpful",
                },
                ChatMessage {
                    role: "user",
                    content: "hi",
                },
            ],
            true,
        )
        .unwrap();
        assert!(out.contains("<|im_start|>system\nyou are helpful<|im_end|>"));
        assert!(out.contains("<|im_start|>user\nhi<|im_end|>"));
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    /// Regression for the Qwen3.6 family (and most modern HF templates):
    /// real Jinja2 inherits Python's string/dict methods, so templates
    /// freely call `.startswith()`, `.strip()`, `.split()`, `.get()`.
    /// Without the pycompat unknown-method callback, minijinja fails
    /// with `unknown method: string has no method named startswith`
    /// — which 500'd every chat request against the Qwen3.6 GGUF's
    /// embedded template (its line 81 calls `startswith`).
    #[test]
    fn chat_template_supports_python_string_methods() {
        let template = "{% for message in messages %}\
{% if message['content'].startswith('sys:') %}\
[SYS]{{ message['content'].split(':')[1].strip() }}[/SYS]\
{% elif message['content'].endswith('!') %}\
[BANG]{{ message['content'].rstrip('!').upper() }}\
{% else %}{{ message['content'] }}{% endif %}\
{% endfor %}";
        let out = render_chat_template(
            template,
            &[
                ChatMessage {
                    role: "system",
                    content: "sys:  be helpful  ",
                },
                ChatMessage {
                    role: "user",
                    content: "hello!",
                },
                ChatMessage {
                    role: "user",
                    content: "plain",
                },
            ],
            false,
        )
        .expect("python string methods must render via pycompat");
        assert!(out.contains("[SYS]be helpful[/SYS]"), "got: {out}");
        assert!(out.contains("[BANG]HELLO"), "got: {out}");
        assert!(out.contains("plain"), "got: {out}");
    }

    /// Regression for the TinyLlama / Llama-2-Chat family: templates that
    /// reference `bos_token` and `eos_token` and that branch on a
    /// possibly-undefined `system_message`. Without the BOS/EOS context
    /// plumbing AND the lenient-undefined env setting, this template fails
    /// with "tried to use + operator on unsupported types string and
    /// undefined" the moment the caller omits a system message.
    #[test]
    fn chat_template_handles_bos_eos_and_missing_system() {
        let template = "{% if messages[0]['role'] == 'system' %}\
{% set system_message = messages[0]['content'] %}{% set loop_messages = messages[1:] %}\
{% else %}{% set system_message = false %}{% set loop_messages = messages %}{% endif %}\
{% for message in loop_messages %}\
{% if loop.index0 == 0 and system_message != false %}\
{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}\
{% else %}{% set content = message['content'] %}{% endif %}\
{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content + ' [/INST]' }}\
{% elif message['role'] == 'assistant' %}{{ ' ' + content + ' ' + eos_token }}\
{% endif %}{% endfor %}";

        // No system message + BOS/EOS supplied → must succeed and include BOS.
        let out = render_chat_template_with_specials(
            template,
            &[ChatMessage {
                role: "user",
                content: "hi",
            }],
            false,
            None,
            Some("<s>"),
            Some("</s>"),
        )
        .expect("must render without panicking on missing system");
        assert!(out.contains("<s>[INST] hi [/INST]"), "got: {out}");

        // No system message + BOS/EOS *unset* → must still render (empty
        // placeholders). Lenient-undefined makes `'' + ''` succeed.
        let out2 = render_chat_template(
            template,
            &[ChatMessage {
                role: "user",
                content: "hi",
            }],
            false,
        )
        .expect("must render without BOS/EOS too");
        assert!(out2.contains("[INST] hi [/INST]"));

        // With a system message: the system_message branch fires.
        let out3 = render_chat_template_with_specials(
            template,
            &[
                ChatMessage { role: "system", content: "be terse" },
                ChatMessage { role: "user", content: "hi" },
            ],
            false,
            None,
            Some("<s>"),
            Some("</s>"),
        )
        .unwrap();
        assert!(out3.contains("<<SYS>>"));
        assert!(out3.contains("be terse"));
        assert!(out3.contains("<s>[INST] <<SYS>>"));
    }

    #[test]
    fn split_at_special_tokens_chatml_layout() {
        let text =
            "<|im_start|>system\nyou are helpful<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n";
        let segs = split_at_special_tokens(text);
        // Four `<|` markers → four segments. Each segment carries one special token + its content.
        assert_eq!(segs.len(), 4);
        assert!(segs[0].starts_with("<|im_start|>system"));
        assert!(segs[1].starts_with("<|im_end|>"));
        assert!(segs[2].starts_with("<|im_start|>user"));
        assert!(segs[3].starts_with("<|im_end|>"));
        // Concat must reproduce the input exactly.
        assert_eq!(segs.concat(), text);
    }

    #[test]
    fn split_at_special_tokens_no_specials_returns_single_segment() {
        let text = "plain text with no special markers";
        let segs = split_at_special_tokens(text);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0], text);
    }

    #[test]
    fn split_at_special_tokens_empty_returns_empty() {
        assert!(split_at_special_tokens("").is_empty());
    }

    #[test]
    fn split_at_special_tokens_prefix_before_first_marker_is_kept() {
        let text = "BOS<|im_start|>user\nhi<|im_end|>";
        let segs = split_at_special_tokens(text);
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[0], "BOS");
        assert!(segs[1].starts_with("<|im_start|>"));
        assert!(segs[2].starts_with("<|im_end|>"));
        assert_eq!(segs.concat(), text);
    }
}
