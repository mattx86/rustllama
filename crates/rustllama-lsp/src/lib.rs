//! Minimal Language Server Protocol bridge for rustllama.
//!
//! Speaks LSP/JSON-RPC over stdio. Editors that target this binary get
//! AI inline completion via two surfaces:
//!
//!   - `textDocument/inlineCompletion` (LSP 3.18+, ghost-text style)
//!   - `textDocument/completion` (older list-style menu)
//!
//! Both translate to a single FIM call against an already-running
//! rustllama HTTP server (`POST /v1/completions` with `prompt` + `suffix`).
//! The LSP server is intentionally stateless on the inference side — it
//! never loads a model itself. Start `rustllama serve` separately and
//! point the LSP at it via `--base-url` (or rely on the runtime record
//! that `serve` writes when both run on the same machine).
//!
//! Wire format (per LSP spec):
//!
//! ```text
//! Content-Length: <N>\r\n
//! \r\n
//! { jsonrpc: "2.0", id?: N, method: "...", params: {...} }
//! ```
//!
//! No async runtime needed in the LSP loop itself — stdio is blocking.
//! For the outbound FIM call we spin a small multi-thread tokio runtime
//! so request fan-out doesn't block the LSP read loop.

use std::collections::HashMap;
use std::io::{BufRead, Read, Write};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

pub mod text;

#[derive(Debug, thiserror::Error)]
pub enum LspError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("parse: {0}")]
    Parse(String),
    #[error("client: {0}")]
    Client(#[from] rustllama_client::ClientError),
}

pub type Result<T> = std::result::Result<T, LspError>;

/// Per-LSP-session configuration. All fields are tunable on the command
/// line; sensible defaults exist for each.
#[derive(Debug, Clone)]
pub struct LspConfig {
    /// `http://host:port` URL of the rustllama HTTP server. Required.
    pub base_url: String,
    /// Model id to send in the `model` field of `/v1/completions`. If
    /// empty, the server uses its default.
    pub model: String,
    /// Max tokens for each inline completion. Editors typically want a
    /// few hundred at most; larger values waste compute on text the user
    /// will never accept.
    pub max_tokens: u32,
    /// Sampling temperature. 0.0 = greedy (deterministic), best for code.
    pub temperature: f32,
    /// Optional stop sequences that terminate completion early — e.g.
    /// "\n\n" to bound suggestions to a single logical block.
    pub stop: Vec<String>,
    /// How much of the buffer before/after the cursor to include in the
    /// FIM prefix/suffix. Long buffers can overflow `ctx_size`; we
    /// truncate from the far end. Defaults to ~4KB each side.
    pub max_prefix_bytes: usize,
    pub max_suffix_bytes: usize,
}

impl Default for LspConfig {
    fn default() -> Self {
        Self {
            base_url: "http://127.0.0.1:11434".into(),
            model: String::new(),
            max_tokens: 128,
            temperature: 0.0,
            stop: vec!["\n\n".into()],
            max_prefix_bytes: 4096,
            max_suffix_bytes: 4096,
        }
    }
}

/// One open document in the editor. Tracked across didOpen/didChange/
/// didClose. `text` holds the full latest content; LSP positions index
/// into `text` via UTF-16 code units per the spec.
#[derive(Debug, Default)]
pub struct Document {
    pub text: String,
    pub version: i64,
}

/// Stateful LSP session. Holds open documents, the FIM client, and the
/// session config. Run [`Session::run`] on a pair of stdio handles to
/// service requests until the client sends `exit`.
pub struct Session {
    docs: HashMap<String, Document>,
    config: LspConfig,
    client: Arc<rustllama_client::Client>,
    /// Tokio runtime for the outbound HTTP call. A 2-thread pool is
    /// plenty — inline completions are issued one at a time per editor.
    runtime: tokio::runtime::Runtime,
    initialized: bool,
    shutdown_requested: bool,
}

impl Session {
    pub fn new(config: LspConfig) -> Result<Self> {
        let client = rustllama_client::Client::new(&config.base_url)
            .map_err(LspError::Client)?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()?;
        Ok(Self {
            docs: HashMap::new(),
            config,
            client: Arc::new(client),
            runtime,
            initialized: false,
            shutdown_requested: false,
        })
    }

    /// Run the LSP loop until the peer sends `exit`. Reads JSON-RPC
    /// messages from `input` and writes responses to `output`. Both must
    /// be unbuffered (LSP requires \r\n framing on every byte boundary).
    pub fn run<R: Read, W: Write>(mut self, input: R, mut output: W) -> Result<()> {
        let mut reader = std::io::BufReader::new(input);
        loop {
            let message = match read_message(&mut reader)? {
                Some(m) => m,
                None => break, // peer closed stdin
            };
            tracing::trace!(method = ?message.get("method"), id = ?message.get("id"), "lsp msg");
            if let Some(response) = self.handle(&message) {
                write_message(&mut output, &response)?;
            }
            if self.shutdown_requested
                && message.get("method").and_then(|m| m.as_str()) == Some("exit")
            {
                break;
            }
        }
        Ok(())
    }

    /// Dispatch one JSON-RPC message. Returns the response to write back
    /// (for requests) or `None` for notifications.
    fn handle(&mut self, msg: &Value) -> Option<Value> {
        let method = msg.get("method").and_then(|m| m.as_str())?;
        let id = msg.get("id").cloned();
        let params = msg.get("params").cloned().unwrap_or(Value::Null);

        match method {
            "initialize" => {
                self.initialized = true;
                Some(response(id, json!({
                    "capabilities": server_capabilities(),
                    "serverInfo": {
                        "name": "rustllama-lsp",
                        "version": env!("CARGO_PKG_VERSION"),
                    },
                })))
            }
            "initialized" => None, // notification, no response
            "shutdown" => {
                self.shutdown_requested = true;
                Some(response(id, Value::Null))
            }
            "exit" => None,
            "textDocument/didOpen" => {
                self.did_open(&params);
                None
            }
            "textDocument/didChange" => {
                self.did_change(&params);
                None
            }
            "textDocument/didClose" => {
                self.did_close(&params);
                None
            }
            "textDocument/inlineCompletion" => {
                Some(self.inline_completion(id, &params))
            }
            "textDocument/completion" => {
                Some(self.completion(id, &params))
            }
            // Unhandled — return empty success for requests, ignore notifications.
            other => {
                if id.is_some() {
                    Some(error_response(
                        id,
                        -32601,
                        format!("method not supported: {other}"),
                    ))
                } else {
                    tracing::debug!(method = other, "ignoring unhandled notification");
                    None
                }
            }
        }
    }

    fn did_open(&mut self, params: &Value) {
        let Some(td) = params.get("textDocument") else { return };
        let Some(uri) = td.get("uri").and_then(|u| u.as_str()) else { return };
        let text = td
            .get("text")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();
        let version = td.get("version").and_then(|v| v.as_i64()).unwrap_or(0);
        self.docs.insert(uri.to_string(), Document { text, version });
    }

    fn did_change(&mut self, params: &Value) {
        let Some(td) = params.get("textDocument") else { return };
        let Some(uri) = td.get("uri").and_then(|u| u.as_str()) else { return };
        let new_version = td.get("version").and_then(|v| v.as_i64()).unwrap_or(0);
        let Some(doc) = self.docs.get_mut(uri) else { return };
        doc.version = new_version;
        let Some(changes) = params.get("contentChanges").and_then(|c| c.as_array()) else {
            return;
        };
        for change in changes {
            // LSP allows two kinds: full-replace `{ text }` OR incremental
            // `{ range: { start, end }, text }`. Handle both.
            match change.get("range") {
                None => {
                    if let Some(t) = change.get("text").and_then(|t| t.as_str()) {
                        doc.text = t.to_string();
                    }
                }
                Some(range) => {
                    let Some(text) = change.get("text").and_then(|t| t.as_str()) else {
                        continue;
                    };
                    let Some(start) = parse_position(range.get("start")) else { continue };
                    let Some(end) = parse_position(range.get("end")) else { continue };
                    text::apply_incremental_edit(&mut doc.text, start, end, text);
                }
            }
        }
    }

    fn did_close(&mut self, params: &Value) {
        let Some(td) = params.get("textDocument") else { return };
        let Some(uri) = td.get("uri").and_then(|u| u.as_str()) else { return };
        self.docs.remove(uri);
    }

    fn inline_completion(&mut self, id: Option<Value>, params: &Value) -> Value {
        let Some((prefix, suffix)) = self.fim_split(params) else {
            return response(id, json!({ "items": [] }));
        };
        match self.run_fim(prefix, suffix) {
            Ok(completion) => {
                // textDocument/inlineCompletion response shape (LSP 3.18):
                //   { items: [{ insertText: "...", filterText?: "..." }] }
                response(id, json!({
                    "items": [{
                        "insertText": completion,
                    }]
                }))
            }
            Err(e) => {
                tracing::warn!(error = %e, "FIM completion failed");
                response(id, json!({ "items": [] }))
            }
        }
    }

    fn completion(&mut self, id: Option<Value>, params: &Value) -> Value {
        let Some((prefix, suffix)) = self.fim_split(params) else {
            return response(id, json!({ "items": [] }));
        };
        match self.run_fim(prefix, suffix) {
            Ok(completion) => {
                // Classic completion response: an array of CompletionItem.
                // Use kind=15 ("Snippet") so editors don't mangle the text
                // by applying their own filter heuristics. The label is
                // the completion's first line (so the menu remains readable).
                let label = completion
                    .lines()
                    .next()
                    .unwrap_or(&completion)
                    .to_string();
                response(id, json!({
                    "isIncomplete": false,
                    "items": [{
                        "label": label,
                        "kind": 15,
                        "insertText": completion,
                        "detail": "rustllama",
                    }]
                }))
            }
            Err(e) => {
                tracing::warn!(error = %e, "FIM completion failed");
                response(id, json!({ "items": [] }))
            }
        }
    }

    /// Slice `(prefix, suffix)` from the current document at the LSP
    /// position. Truncates to `max_prefix_bytes` / `max_suffix_bytes`
    /// from the cursor outward (on UTF-8 char boundaries).
    fn fim_split(&self, params: &Value) -> Option<(String, String)> {
        let td = params.get("textDocument")?;
        let uri = td.get("uri")?.as_str()?;
        let position = parse_position(params.get("position"))?;
        let doc = self.docs.get(uri)?;
        let cursor = text::position_to_byte_offset(&doc.text, position)?;
        let (full_prefix, full_suffix) = doc.text.split_at(cursor);
        let prefix =
            text::truncate_prefix(full_prefix, self.config.max_prefix_bytes).to_string();
        let suffix =
            text::truncate_suffix(full_suffix, self.config.max_suffix_bytes).to_string();
        Some((prefix, suffix))
    }

    fn run_fim(&self, prefix: String, suffix: String) -> Result<String> {
        let client = self.client.clone();
        let req = rustllama_client::CompletionRequest {
            model: self.config.model.clone(),
            prompt: prefix,
            suffix: Some(suffix),
            temperature: Some(self.config.temperature),
            top_p: None,
            top_k: None,
            repeat_penalty: None,
            max_tokens: Some(self.config.max_tokens),
            stop: self.config.stop.clone(),
            seed: None,
            stream: false,
        };
        let resp = self.runtime.block_on(async move {
            client.completions(&req).await
        })?;
        let text = resp.choices.into_iter().next().map(|c| c.text).unwrap_or_default();
        Ok(text)
    }
}

fn server_capabilities() -> Value {
    json!({
        // Incremental sync (TextDocumentSyncKind.Incremental = 2).
        "textDocumentSync": 2,
        // We provide AI inline completion. Use the LSP 3.18 inlineCompletionProvider.
        "inlineCompletionProvider": {},
        // We also expose the older classic-completion surface for editors
        // that don't yet handle inlineCompletion. No trigger characters —
        // the editor decides when to ask (typically on a manual trigger
        // or after a debounce). No resolveProvider — items are final.
        "completionProvider": {
            "resolveProvider": false,
        },
    })
}

fn response(id: Option<Value>, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    })
}

fn error_response(id: Option<Value>, code: i64, message: String) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message },
    })
}

fn parse_position(value: Option<&Value>) -> Option<text::Position> {
    let v = value?;
    Some(text::Position {
        line: v.get("line")?.as_u64()? as u32,
        character: v.get("character")?.as_u64()? as u32,
    })
}

/// Read one LSP message: Content-Length header, blank line, JSON body.
/// Returns `Ok(None)` on EOF so the caller can exit cleanly when the
/// editor closes stdin.
fn read_message<R: BufRead>(reader: &mut R) -> Result<Option<Value>> {
    let mut content_length: Option<usize> = None;
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line)?;
        if n == 0 {
            return Ok(None);
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.is_empty() {
            break;
        }
        if let Some(rest) = trimmed.strip_prefix("Content-Length:") {
            content_length = rest.trim().parse::<usize>().ok();
        }
        // Other headers ("Content-Type", ...) are accepted and ignored.
    }
    let Some(len) = content_length else {
        return Err(LspError::Parse("missing Content-Length header".into()));
    };
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    let value: Value = serde_json::from_slice(&buf)
        .map_err(|e| LspError::Parse(format!("invalid JSON body: {e}")))?;
    Ok(Some(value))
}

fn write_message<W: Write>(writer: &mut W, msg: &Value) -> Result<()> {
    let body = serde_json::to_vec(msg).map_err(|e| LspError::Parse(e.to_string()))?;
    write!(writer, "Content-Length: {}\r\n\r\n", body.len())?;
    writer.write_all(&body)?;
    writer.flush()?;
    Ok(())
}

// Unused but kept so the type appears in the generated rustdoc.
#[derive(Serialize, Deserialize)]
pub struct InitializeResult {
    pub capabilities: Value,
}
