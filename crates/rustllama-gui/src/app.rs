//! The Phase-2 app: [`GuiApp`], its async bridge, and the egui layout.
//!
//! Phase 2 adds a left icon-nav sidebar, a Models management view (loaded +
//! cached lists, load / unload / set-default / delete, HuggingFace search +
//! streamed pull), and an always-on bottom status bar (RAM / per-GPU VRAM /
//! tok-s / active model, polled from `/v1/metrics`). The Phase-1 model bar +
//! transcript + composer are now the **Chat** view.

use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use egui::{Align, Align2, Color32, FontId, Layout, Margin, Rect, RichText, Rounding, Sense, Stroke};
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use futures::StreamExt;
use rustllama_client::{
    ChatEvent, ChatMessage, ChatRequest, Client, HfFile, HfModel, LoadModelParams, MetricsSnapshot,
    StreamOptions, SystemPrompt, TagsModel, ToolCall, Usage,
};

// --- Theme tokens ---------------------------------------------------------
// Approximate the app's design tokens (LM-Studio-ish dark). `Color32::from_rgb`
// is const, so these live as module constants and can be used anywhere without
// recomputing. Light theme is a later phase.
const BG: Color32 = Color32::from_rgb(0x17, 0x18, 0x1c); // window / panel bg
const ELEVATED: Color32 = Color32::from_rgb(0x1e, 0x1f, 0x25); // cards, inactive widgets
const ELEVATED2: Color32 = Color32::from_rgb(0x26, 0x28, 0x32); // hovered / strokes
const SIDEBAR: Color32 = Color32::from_rgb(0x12, 0x13, 0x17); // nav rail + status bar
const BORDER: Color32 = Color32::from_rgb(0x2a, 0x2c, 0x35); // hairline separators
const ACCENT: Color32 = Color32::from_rgb(0x61, 0x72, 0xf3); // indigo
const TEXT: Color32 = Color32::from_rgb(0xe7, 0xe8, 0xee);
const MUTED: Color32 = Color32::from_rgb(0x9a, 0x9c, 0xab);
const DISABLED: Color32 = Color32::from_rgb(0x5a, 0x5c, 0x6a); // dimmed nav placeholders
const FIELD_BG: Color32 = Color32::from_rgb(0x12, 0x13, 0x16); // TextEdit / meter troughs
const SELECT_BG: Color32 = Color32::from_rgb(0x2a, 0x30, 0x55);
const HEALTH_OK: Color32 = Color32::from_rgb(0x3f, 0xb9, 0x50);
const HEALTH_BAD: Color32 = Color32::from_rgb(0xe5, 0x48, 0x4d);
const WARN: Color32 = Color32::from_rgb(0xd2, 0x99, 0x22); // meter "yellow" band
const DANGER: Color32 = Color32::from_rgb(0xf8, 0x51, 0x49); // delete affordances

/// Keyboard-shortcut rows shown in the chat's help overlay (see
/// [`GuiApp::render_chat`]).
const SHORTCUTS: &[(&str, &str)] = &[
    ("Enter", "Send message"),
    ("Ctrl+Enter", "Send message"),
    ("Shift+Enter", "Insert a newline"),
    ("Esc", "Stop generating / close this overlay"),
    ("Ctrl+L", "Clear the transcript"),
    ("Ctrl+R", "Regenerate the last response"),
    ("Ctrl+E", "Export the conversation as Markdown"),
    ("?", "Toggle this shortcuts overlay"),
];

/// Which page the central area renders. Chat + Models are built in Phase 2;
/// Status / Settings / Decide / Quantize are stubbed in the nav for later.
#[derive(Clone, Copy, PartialEq, Eq)]
enum View {
    Chat,
    Models,
}

/// A currently-loaded model from `/v1/models` (`data[]`). `is_default` gates
/// the "default" badge vs the "Set default" button in the Models view.
#[derive(Clone)]
struct LoadedModel {
    id: String,
    is_default: bool,
}

/// The little vector glyph a nav item draws to the left of its label.
#[derive(Clone, Copy)]
enum NavIcon {
    Chat,
    Models,
    Placeholder,
}

/// One committed message in the transcript. `role` is the OpenAI role string
/// (`"user"` / `"assistant"` / `"system"`) so it maps 1:1 onto [`ChatMessage`].
struct Turn {
    role: String,
    content: String,
}

impl Turn {
    fn user(content: String) -> Self {
        Self {
            role: "user".into(),
            content,
        }
    }
    fn assistant(content: String) -> Self {
        Self {
            role: "assistant".into(),
            content,
        }
    }
    fn system(content: String) -> Self {
        Self {
            role: "system".into(),
            content,
        }
    }
}

/// Sampling controls surfaced in the chat's Sampling popover. Each field is
/// wired into [`ChatRequest`]; [`Sampling::apply`] sends a value only when it
/// differs from the neutral default, so an untouched panel keeps the request
/// wire-minimal (the server/model default wins). Mirrors Chat.tsx's Settings
/// panel (temperature / max_tokens / seed) plus the top_p / top_k /
/// repeat_penalty knobs the request already supports.
#[derive(Clone)]
struct Sampling {
    temperature: f32,
    max_tokens: u32,
    /// Free text; parsed to `u64`. Blank / unparseable = random each turn.
    seed: String,
    top_p: f32,
    top_k: u32,
    repeat_penalty: f32,
}

impl Default for Sampling {
    fn default() -> Self {
        // Neutral-ish defaults: temperature/max_tokens match the web UI's
        // starting values; top_p 1.0 / top_k 0 / repeat_penalty 1.0 are the
        // "no-op" sampler settings, so leaving them untouched means "don't
        // override the model/server default".
        Self {
            temperature: 0.7,
            max_tokens: 512,
            seed: String::new(),
            top_p: 1.0,
            top_k: 0,
            repeat_penalty: 1.0,
        }
    }
}

impl Sampling {
    /// Fold the panel values into a [`ChatRequest`], leaving each field `None`
    /// when it still sits at the default — an untouched panel adds nothing to
    /// the wire body, so the server/model defaults apply as before.
    fn apply(&self, req: &mut ChatRequest) {
        let d = Sampling::default();
        if (self.temperature - d.temperature).abs() > f32::EPSILON {
            req.temperature = Some(self.temperature);
        }
        if self.max_tokens != d.max_tokens {
            req.max_tokens = Some(self.max_tokens);
        }
        if (self.top_p - d.top_p).abs() > f32::EPSILON {
            req.top_p = Some(self.top_p);
        }
        if self.top_k != d.top_k {
            req.top_k = Some(self.top_k);
        }
        if (self.repeat_penalty - d.repeat_penalty).abs() > f32::EPSILON {
            req.repeat_penalty = Some(self.repeat_penalty);
        }
        if let Ok(s) = self.seed.trim().parse::<u64>() {
            req.seed = Some(s);
        }
    }

    /// True when every knob is still at its neutral default (drives a subtle
    /// "· custom" hint on the Sampling button).
    fn is_default(&self) -> bool {
        let d = Sampling::default();
        (self.temperature - d.temperature).abs() <= f32::EPSILON
            && self.max_tokens == d.max_tokens
            && self.seed.trim().is_empty()
            && (self.top_p - d.top_p).abs() <= f32::EPSILON
            && self.top_k == d.top_k
            && (self.repeat_penalty - d.repeat_penalty).abs() <= f32::EPSILON
    }
}

/// A pending CLARIFY (`ask_user`) question rendered as clickable option
/// buttons; picking one continues the conversation (see `choose_option`).
struct PendingAsk {
    prompt: String,
    options: Vec<String>,
}

/// Messages sent from spawned tokio tasks (and the file-dialog thread) back to
/// the UI thread over an `std::sync::mpsc` channel. Drained non-blocking each
/// frame (see [`GuiApp::update`]).
enum UiMsg {
    // --- chat / model bar ---
    /// `/v1/models` `data[]` ids (the model-bar combo).
    Models(Vec<String>),
    /// `/v1/models` `data[]` as {id, is_default} (the Models view Loaded list).
    LoadedModels(Vec<LoadedModel>),
    /// `/healthz` (or a metrics poll) says the server is reachable.
    Health(bool),
    /// The chat stream announced its server request id (`chatcmpl-…`), read
    /// off the first SSE chunk. Stored so Stop can POST it to `/v1/cancel`.
    ChatStart(String),
    /// A streamed chat content delta.
    ChatDelta(String),
    /// The chat stream drained cleanly; carries the final usage block.
    ChatDone(Option<Usage>),
    /// A chat stream failed.
    ChatError(String),
    /// CLARIFY: the model called the reserved `ask_user` tool — render the
    /// prompt + options as a chooser instead of guessing.
    AskUser {
        prompt: String,
        options: Vec<String>,
    },
    /// The model proposed tool call(s). With "Auto-run tools" off we render an
    /// Approve/Deny prompt; on, we note the proposal in the transcript.
    ToolCalls(Vec<ToolCall>),
    /// A model was promoted to the server default (model-bar Load).
    ModelLoaded(String),
    /// Generic status/error line for the model bar.
    Err(String),

    // --- Models view ---
    /// `/api/tags` cached-model list.
    Cached(Vec<TagsModel>),
    /// A model load/unload/default/delete finished — refresh both lists.
    ModelActionDone,
    /// A Models-view action failed.
    ModelsError(String),
    /// The native file dialog returned a `.gguf` path to load.
    PickedFile(std::path::PathBuf),

    // --- HuggingFace search + pull ---
    /// `/api/hf/search` results, tagged with the query they answered so a
    /// stale response (the user kept typing) can be dropped.
    HfResults { query: String, models: Vec<HfModel> },
    /// `/api/hf/files` for the expanded repo.
    HfFiles { repo: String, files: Vec<HfFile> },
    /// An HF search / files call failed.
    HfError(String),
    /// One NDJSON pull-progress frame.
    PullProgress {
        status: String,
        completed: Option<u64>,
        total: Option<u64>,
    },
    /// The pull stream drained cleanly.
    PullDone,
    /// The pull failed (transport or server `{"error":…}` frame).
    PullError(String),

    // --- status bar ---
    /// `/v1/metrics` snapshot (boxed — it's the largest UiMsg variant).
    Metrics(Box<MetricsSnapshot>),
    /// The metrics poll failed (server unreachable) → offline dot.
    MetricsError,

    // --- chat config / token counter / export ---
    /// `/v1/config` returned: the system-prompt library + a fallback ctx budget.
    Config {
        system_prompts: Vec<SystemPrompt>,
        ctx_size: u64,
    },
    /// A debounced `/v1/tokenize` count for the composer text (tagged with the
    /// text it answered so a stale response is dropped).
    DraftTokens { text: String, count: u32 },
    /// The tokenize call failed (e.g. mock engine) → hide the counter.
    DraftTokenError,
    /// The export save-dialog thread wrote the transcript to a file.
    ExportDone(String),
    /// The export failed (write error; a cancelled dialog sends nothing).
    ExportError(String),
}

/// Deferred UI action. Immediate-mode widgets push these while a panel is
/// open; they're applied *after* every panel closes so the spawn helpers get
/// clean (unaliased) access to `self` (see the tail of [`GuiApp::update`]).
enum Action {
    SwitchView(View),
    RefreshModels,
    LoadCached(String),
    PickFile,
    SetDefault(String),
    UnloadModel(String),
    AskDelete(String),
    ConfirmDelete(String),
    CancelDelete,
    HfPickRepo(String),
    HfBack,
    HfPickFile { repo: String, file: String },
    Pull(String),
}

pub struct GuiApp {
    // --- infra ---
    base_url: String,
    rt: tokio::runtime::Handle,
    client: Arc<Client>,
    /// Cloned into each task so it can `request_repaint()` the UI.
    ctx: egui::Context,
    tx: Sender<UiMsg>,
    rx: Receiver<UiMsg>,

    // --- navigation ---
    view: View,

    // --- model-bar / chat state ---
    models: Vec<String>,
    selected_model: Option<String>,
    loaded_model: Option<String>,
    transcript: Vec<Turn>,
    /// In-progress assistant text, shown live while `streaming`.
    pending: String,
    input: String,
    streaming: bool,

    // --- Models view state ---
    loaded_models: Vec<LoadedModel>,
    cached: Vec<TagsModel>,
    models_error: Option<String>,
    /// The model id/name whose row action is in flight (disables its buttons).
    busy_model: Option<String>,
    /// Cached model pending a delete confirmation (drives the confirm modal).
    confirm_delete: Option<String>,

    // --- HuggingFace search + pull ---
    hf_query: String,
    /// When the query text last changed — drives the ~300 ms search debounce.
    hf_dirty_since: Option<Instant>,
    /// The last query actually dispatched (dedupes repeat searches).
    hf_last_searched: String,
    hf_results: Vec<HfModel>,
    hf_searching: bool,
    hf_repo: Option<String>,
    hf_files: Vec<HfFile>,
    hf_files_loading: bool,
    hf_error: Option<String>,
    /// Staged pull ref (`owner/repo:file.gguf`) — the Pull field / button.
    pull_ref: String,
    pulling: bool,
    pull_status: Option<String>,
    /// Download fraction 0..1 while a pull streams byte counts, else `None`.
    pull_pct: Option<f32>,

    // --- status bar ---
    health: Option<bool>,
    metrics: Option<MetricsSnapshot>,
    /// When metrics were last polled + whether a poll is in flight — together
    /// they throttle the poll to ~2 s without piling up requests.
    metrics_last_poll: Option<Instant>,
    metrics_inflight: bool,

    // --- chat status line ---
    last_usage: Option<Usage>,
    status: Option<String>,

    // --- chat: system prompt (from /v1/config) ---
    system_prompts: Vec<SystemPrompt>,
    /// Name of the picked prompt, or "" for none. Auto-selected from
    /// `default_for_model` until the user touches the dropdown.
    selected_prompt: String,
    user_picked_prompt: bool,
    /// Fallback context budget from `[inference].ctx_size` (used when
    /// `/v1/metrics.ctx_size` is 0).
    config_ctx_size: u64,

    // --- chat: sampling ---
    sampling: Sampling,

    // --- chat: live token counter ---
    /// Last debounced token count for the composer, or `None` (counting /
    /// unsupported). Tagged internally by `tok_last_text`.
    draft_tokens: Option<u32>,
    /// When the composer text last changed — drives the ~300 ms tokenize
    /// debounce.
    tok_dirty_since: Option<Instant>,
    /// The composer text last sent to `/v1/tokenize` (dedupes repeats).
    tok_last_text: String,
    tok_inflight: bool,

    // --- chat: cancel / clarify / tools ---
    /// Server request id of the in-flight stream (for Stop → `/v1/cancel`).
    request_id: Option<String>,
    /// Handle to the streaming task so Stop can abort local SSE consumption.
    chat_task: Option<tokio::task::JoinHandle<()>>,
    /// Accept proposed tool calls without prompting (chat never executes
    /// tools — this only gates the confirm UI). Persisted via eframe storage.
    auto_tools: bool,
    /// CLARIFY opt-in: let the model pause and ask (drives `allow_clarify`).
    /// Persisted via eframe storage.
    clarify: bool,
    pending_question: Option<PendingAsk>,
    pending_tools: Option<Vec<ToolCall>>,

    // --- chat: keyboard-shortcuts overlay ---
    show_shortcuts: bool,

    md_cache: CommonMarkCache,
}

impl GuiApp {
    pub fn new(
        cc: &eframe::CreationContext<'_>,
        base_url: String,
        rt: tokio::runtime::Handle,
    ) -> anyhow::Result<Self> {
        configure_style(&cc.egui_ctx);
        // Borrow `base_url` for the client before it moves into the struct.
        let client = Client::new(base_url.as_str())
            .map_err(|e| anyhow::anyhow!("invalid base url {base_url:?}: {e}"))?;
        let (tx, rx) = std::sync::mpsc::channel();

        // Restore the header toggles from eframe storage. Without the eframe
        // `persistence` feature `cc.storage` is `None`, so this degrades to the
        // in-session defaults (auto-tools OFF, clarify ON). Mirrors the web
        // UI's localStorage keys.
        let (auto_tools, clarify) = cc
            .storage
            .map(|s| {
                let auto = s.get_string("auto_tools").as_deref() == Some("1");
                let clar = s.get_string("clarify").as_deref() != Some("0");
                (auto, clar)
            })
            .unwrap_or((false, true));

        let app = Self {
            base_url,
            rt,
            client: Arc::new(client),
            ctx: cc.egui_ctx.clone(),
            tx,
            rx,
            view: View::Chat,
            models: Vec::new(),
            selected_model: None,
            loaded_model: None,
            transcript: Vec::new(),
            pending: String::new(),
            input: String::new(),
            streaming: false,
            loaded_models: Vec::new(),
            cached: Vec::new(),
            models_error: None,
            busy_model: None,
            confirm_delete: None,
            hf_query: String::new(),
            hf_dirty_since: None,
            hf_last_searched: String::new(),
            hf_results: Vec::new(),
            hf_searching: false,
            hf_repo: None,
            hf_files: Vec::new(),
            hf_files_loading: false,
            hf_error: None,
            pull_ref: String::new(),
            pulling: false,
            pull_status: None,
            pull_pct: None,
            health: None,
            metrics: None,
            metrics_last_poll: None,
            metrics_inflight: false,
            last_usage: None,
            status: None,
            system_prompts: Vec::new(),
            selected_prompt: String::new(),
            user_picked_prompt: false,
            config_ctx_size: 0,
            sampling: Sampling::default(),
            draft_tokens: Some(0),
            tok_dirty_since: None,
            tok_last_text: String::new(),
            tok_inflight: false,
            request_id: None,
            chat_task: None,
            auto_tools,
            clarify,
            pending_question: None,
            pending_tools: None,
            show_shortcuts: false,
            md_cache: CommonMarkCache::default(),
        };

        // Kick off the initial loads (health probe + model list + cached list +
        // the config's system-prompt library / ctx budget).
        app.spawn_health();
        app.spawn_refresh_models();
        app.spawn_refresh_cached();
        app.spawn_config();
        Ok(app)
    }

    // --- async bridge ------------------------------------------------------
    // Each helper spawns a task on the tokio runtime holding cloned handles,
    // reports over `tx`, and pings `ctx.request_repaint()` so the UI wakes.

    fn spawn_health(&self) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            let ok = client.healthz().await.is_ok();
            let _ = tx.send(UiMsg::Health(ok));
            ctx.request_repaint();
        });
    }

    /// Refresh `/v1/models`: feeds BOTH the model-bar combo (ids) and the
    /// Models-view Loaded list (id + is_default) from one request.
    fn spawn_refresh_models(&self) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            match client.list_models().await {
                Ok(v) => {
                    let _ = tx.send(UiMsg::Models(parse_model_ids(&v)));
                    let _ = tx.send(UiMsg::LoadedModels(parse_loaded_models(&v)));
                }
                Err(e) => {
                    let _ = tx.send(UiMsg::Err(format!("list models: {e}")));
                }
            }
            ctx.request_repaint();
        });
    }

    /// Refresh `/api/tags` (the cached-on-disk models).
    fn spawn_refresh_cached(&self) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            match client.cached_models().await {
                Ok(v) => {
                    let _ = tx.send(UiMsg::Cached(v));
                }
                Err(e) => {
                    let _ = tx.send(UiMsg::ModelsError(format!("list cached: {e}")));
                }
            }
            ctx.request_repaint();
        });
    }

    /// Poll `/v1/metrics` for the status bar.
    fn spawn_metrics(&self) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            match client.metrics().await {
                Ok(m) => {
                    let _ = tx.send(UiMsg::Metrics(Box::new(m)));
                }
                Err(_) => {
                    let _ = tx.send(UiMsg::MetricsError);
                }
            }
            ctx.request_repaint();
        });
    }

    /// Fetch `/v1/config` once at startup: the system-prompt library (chat
    /// dropdown) + a fallback context budget. A failure is silent — the chat
    /// just hides the dropdown / falls back to metrics for the ctx budget.
    fn spawn_config(&self) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            if let Ok(env) = client.get_config().await {
                let _ = tx.send(UiMsg::Config {
                    system_prompts: env.config.system_prompts,
                    ctx_size: env.config.inference.ctx_size,
                });
                ctx.request_repaint();
            }
        });
    }

    /// Debounced `/v1/tokenize` for the composer counter. The result is tagged
    /// with `text` so a stale reply (the user kept typing) is dropped.
    fn spawn_tokenize(&self, text: String) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        let model = self.current_model();
        self.rt.spawn(async move {
            let model_opt = if model.is_empty() {
                None
            } else {
                Some(model.as_str())
            };
            match client.tokenize(&text, model_opt).await {
                Ok(r) => {
                    let _ = tx.send(UiMsg::DraftTokens {
                        text,
                        count: r.count,
                    });
                }
                Err(_) => {
                    let _ = tx.send(UiMsg::DraftTokenError);
                }
            }
            ctx.request_repaint();
        });
    }

    /// Model-bar Load: promote the selected (already-loaded) model to default.
    /// The bar's ids come from `/v1/models`, so a `hub` load is best-effort
    /// ("already loaded" is a non-fatal no-op); the set-default is what makes
    /// chat target it.
    fn spawn_load(&self, model_id: String) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            let params = LoadModelParams {
                hub: Some(model_id.clone()),
                ..Default::default()
            };
            let _ = client.load_model(&params).await;
            match client.set_default_model(&model_id).await {
                Ok(_) => {
                    let _ = tx.send(UiMsg::ModelLoaded(model_id));
                }
                Err(e) => {
                    let _ = tx.send(UiMsg::Err(format!("set default: {e}")));
                }
            }
            ctx.request_repaint();
        });
    }

    /// Models-view Load of a CACHED entry. Replicates the web Models page's
    /// `doLoad`: a HuggingFace ref (contains BOTH `/` and `:`) loads via the
    /// `hub` field; anything else is a `/api/tags` file-stem and loads via the
    /// `name` field, which the server resolves against its model cache. This
    /// is the payload that actually loads a cached model by name (the CLI
    /// `model load <name>` fails because it treats the arg as a file path).
    fn spawn_load_cached(&self, name: String) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            let is_hub = name.contains('/') && name.contains(':');
            let params = if is_hub {
                LoadModelParams {
                    hub: Some(name.clone()),
                    ..Default::default()
                }
            } else {
                LoadModelParams {
                    name: Some(name.clone()),
                    ..Default::default()
                }
            };
            match client.load_model(&params).await {
                Ok(_) => {
                    let _ = tx.send(UiMsg::ModelActionDone);
                }
                Err(e) => {
                    let _ = tx.send(UiMsg::ModelsError(format!("load {name}: {e}")));
                }
            }
            ctx.request_repaint();
        });
    }

    /// Load a GGUF straight off disk by absolute path (drag-drop equivalent /
    /// native file picker result).
    fn spawn_load_path(&self, path: std::path::PathBuf) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            let params = LoadModelParams {
                path: Some(path.clone()),
                ..Default::default()
            };
            match client.load_model(&params).await {
                Ok(_) => {
                    let _ = tx.send(UiMsg::ModelActionDone);
                }
                Err(e) => {
                    let _ = tx.send(UiMsg::ModelsError(format!(
                        "load {}: {e}",
                        path.display()
                    )));
                }
            }
            ctx.request_repaint();
        });
    }

    fn spawn_set_default(&self, model_id: String) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            match client.set_default_model(&model_id).await {
                Ok(_) => {
                    let _ = tx.send(UiMsg::ModelActionDone);
                }
                Err(e) => {
                    let _ = tx.send(UiMsg::ModelsError(format!("set default: {e}")));
                }
            }
            ctx.request_repaint();
        });
    }

    fn spawn_unload(&self, model_id: String) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            match client.unload_model(&model_id).await {
                Ok(_) => {
                    let _ = tx.send(UiMsg::ModelActionDone);
                }
                Err(e) => {
                    let _ = tx.send(UiMsg::ModelsError(format!("unload: {e}")));
                }
            }
            ctx.request_repaint();
        });
    }

    fn spawn_delete(&self, name: String) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            match client.delete_model(&name).await {
                Ok(()) => {
                    let _ = tx.send(UiMsg::ModelActionDone);
                }
                Err(e) => {
                    let _ = tx.send(UiMsg::ModelsError(format!("delete {name}: {e}")));
                }
            }
            ctx.request_repaint();
        });
    }

    fn spawn_hf_search(&self, query: String) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            match client.hf_search(&query, 20).await {
                Ok(models) => {
                    let _ = tx.send(UiMsg::HfResults { query, models });
                }
                Err(e) => {
                    let _ = tx.send(UiMsg::HfError(format!("hf search: {e}")));
                }
            }
            ctx.request_repaint();
        });
    }

    fn spawn_hf_files(&self, repo: String) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            match client.hf_files(&repo).await {
                Ok(files) => {
                    let _ = tx.send(UiMsg::HfFiles { repo, files });
                }
                Err(e) => {
                    let _ = tx.send(UiMsg::HfError(format!("hf files: {e}")));
                }
            }
            ctx.request_repaint();
        });
    }

    /// Stream `POST /api/pull`, forwarding each NDJSON frame to the UI. A
    /// server `{"error":…}` frame (or a transport failure) stops the stream
    /// with a `PullError`; a clean drain sends `PullDone` (which refreshes the
    /// cached list).
    fn spawn_pull(&self, model_ref: String) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            match client.pull_model(&model_ref).await {
                Ok(mut stream) => {
                    while let Some(item) = stream.next().await {
                        match item {
                            Ok(p) => {
                                if let Some(err) = p.error {
                                    let _ = tx.send(UiMsg::PullError(err));
                                    ctx.request_repaint();
                                    return;
                                }
                                let _ = tx.send(UiMsg::PullProgress {
                                    status: p.status,
                                    completed: p.completed,
                                    total: p.total,
                                });
                                ctx.request_repaint();
                            }
                            Err(e) => {
                                let _ = tx.send(UiMsg::PullError(e.to_string()));
                                ctx.request_repaint();
                                return;
                            }
                        }
                    }
                    let _ = tx.send(UiMsg::PullDone);
                }
                Err(e) => {
                    let _ = tx.send(UiMsg::PullError(e.to_string()));
                }
            }
            ctx.request_repaint();
        });
    }

    /// Open the native "load a .gguf" file dialog. rfd's `pick_file` blocks
    /// until the user chooses, so it runs on a dedicated OS thread (NOT the
    /// tokio pool or the UI thread) and reports the pick back over the same
    /// channel — the event loop never stalls.
    fn spawn_pick_file(&self) {
        let (tx, ctx) = (self.tx.clone(), self.ctx.clone());
        std::thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("GGUF model", &["gguf"])
                .set_title("Load a GGUF model")
                .pick_file()
            {
                let _ = tx.send(UiMsg::PickedFile(path));
                ctx.request_repaint();
            }
        });
    }

    fn spawn_chat(
        &mut self,
        model: String,
        messages: Vec<ChatMessage>,
        sampling: Sampling,
        allow_clarify: bool,
    ) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        let handle = self.rt.spawn(async move {
            let mut req = ChatRequest {
                model,
                messages,
                temperature: None,
                top_p: None,
                top_k: None,
                max_tokens: None,
                repeat_penalty: None,
                seed: None,
                stream: true,
                stream_options: Some(StreamOptions {
                    include_usage: true,
                }),
                // CLARIFY opt-in: `Some(true)` routes through the ask_user tool
                // path so the model can pause and ask; `None` keeps the request
                // wire-identical to the plain path (mirrors the web client
                // sending the flag only when the toggle is on).
                allow_clarify: if allow_clarify { Some(true) } else { None },
            };
            // Fold in the sampling panel (each field stays None at its default).
            sampling.apply(&mut req);

            // Event-ordering NOTE: with `stream_options.include_usage` the
            // server emits the Usage event in a *separate* final SSE chunk that
            // arrives AFTER Finish. So we do NOT commit on Finish — we
            // accumulate `usage` and send `ChatDone` once the stream fully
            // drains, which guarantees the usage stats ride along with the
            // completion.
            let mut usage: Option<Usage> = None;
            let mut errored = false;
            match client.chat_stream(req).await {
                Ok(mut stream) => {
                    while let Some(ev) = stream.next().await {
                        match ev {
                            // The first chunk carries the request id — forward
                            // it so a Stop can POST it to `/v1/cancel`.
                            Ok(ChatEvent::Start(id)) => {
                                let _ = tx.send(UiMsg::ChatStart(id));
                                ctx.request_repaint();
                            }
                            Ok(ChatEvent::Content(t)) => {
                                let _ = tx.send(UiMsg::ChatDelta(t));
                                ctx.request_repaint(); // wake the UI per token
                            }
                            Ok(ChatEvent::Usage(u)) => usage = Some(u),
                            // CLARIFY: surface the question so the transcript
                            // renders clickable option buttons (not inline text).
                            Ok(ChatEvent::AskUser {
                                prompt, options, ..
                            }) => {
                                let _ = tx.send(UiMsg::AskUser { prompt, options });
                                ctx.request_repaint();
                            }
                            // Tool-call proposal — the UI applies its confirm /
                            // auto-run policy (the chat has no tool executor).
                            Ok(ChatEvent::ToolCalls(calls)) => {
                                let _ = tx.send(UiMsg::ToolCalls(calls));
                                ctx.request_repaint();
                            }
                            Ok(ChatEvent::Error(e)) => {
                                errored = true;
                                let _ = tx.send(UiMsg::ChatError(e));
                                ctx.request_repaint();
                            }
                            // Finish: nothing to render (Usage rides a later chunk).
                            Ok(_) => {}
                            Err(e) => {
                                errored = true;
                                let _ = tx.send(UiMsg::ChatError(e.to_string()));
                                ctx.request_repaint();
                            }
                        }
                    }
                    if !errored {
                        let _ = tx.send(UiMsg::ChatDone(usage));
                    }
                }
                Err(e) => {
                    let _ = tx.send(UiMsg::ChatError(e.to_string()));
                }
            }
            ctx.request_repaint();
        });
        self.chat_task = Some(handle);
    }

    // --- state helpers -----------------------------------------------------

    /// The model id to target for the next chat: the explicit selection, else
    /// the last-loaded default, else the first known model.
    fn current_model(&self) -> String {
        if let Some(m) = self.selected_model.as_ref().filter(|s| !s.is_empty()) {
            return m.clone();
        }
        if let Some(m) = &self.loaded_model {
            return m.clone();
        }
        self.models.first().cloned().unwrap_or_default()
    }

    /// The active model id used for system-prompt auto-selection: prefer the
    /// server's reported `model_id` (via `/v1/metrics`), else the last-loaded,
    /// else the model-bar selection.
    fn active_model_id(&self) -> String {
        if let Some(id) = self
            .metrics
            .as_ref()
            .map(|m| m.model_id.as_str())
            .filter(|s| !s.is_empty())
        {
            return id.to_string();
        }
        self.current_model()
    }

    /// The context budget for the token counter: `/v1/metrics.ctx_size` when
    /// known, else the config's `[inference].ctx_size`, else `None`.
    fn ctx_budget(&self) -> Option<u64> {
        self.metrics
            .as_ref()
            .map(|m| m.ctx_size)
            .filter(|&c| c > 0)
            .or(if self.config_ctx_size > 0 {
                Some(self.config_ctx_size)
            } else {
                None
            })
    }

    /// The body of the currently-selected system prompt (if any / non-empty).
    fn active_system_prompt_body(&self) -> Option<String> {
        if self.selected_prompt.is_empty() {
            return None;
        }
        self.system_prompts
            .iter()
            .find(|p| p.name == self.selected_prompt)
            .map(|p| p.body.clone())
            .filter(|b| !b.is_empty())
    }

    /// Auto-select the prompt whose `default_for_model` matches the active
    /// model — but only until the user picks one from the dropdown. Mirrors
    /// Chat.tsx's auto-pick effect.
    fn maybe_autoselect_prompt(&mut self) {
        if self.user_picked_prompt || self.system_prompts.is_empty() {
            return;
        }
        let model = self.active_model_id();
        if model.is_empty() {
            return;
        }
        let matched = self
            .system_prompts
            .iter()
            .find(|p| !p.default_for_model.is_empty() && p.default_for_model == model)
            .map(|p| p.name.clone());
        self.selected_prompt = matched.unwrap_or_default();
    }

    /// Stream one assistant turn against the current transcript. Shared by
    /// send / regenerate / clarify-pick / tool-deny so each reuses the exact
    /// same request build (sampling + clarify) and streaming path.
    fn stream_current(&mut self) {
        let model = self.current_model();
        if model.is_empty() {
            self.status = Some("Select and load a model first.".into());
            self.streaming = false;
            return;
        }
        self.pending.clear();
        self.last_usage = None;
        self.status = None;
        self.request_id = None;
        self.streaming = true;
        // Send the full transcript so the server has the conversation context.
        let messages: Vec<ChatMessage> = self
            .transcript
            .iter()
            .map(|t| ChatMessage {
                role: t.role.clone(),
                content: t.content.clone(),
            })
            .collect();
        let sampling = self.sampling.clone();
        let clarify = self.clarify;
        self.spawn_chat(model, messages, sampling, clarify);
    }

    fn send_message(&mut self) {
        if self.streaming {
            return;
        }
        let text = self.input.trim().to_string();
        if text.is_empty() {
            return;
        }
        // A fresh manual send supersedes any pending clarify / tool prompt.
        self.pending_question = None;
        self.pending_tools = None;
        self.input.clear();
        self.draft_tokens = Some(0);
        self.tok_dirty_since = None;

        // On the first turn, materialize the selected system prompt as the
        // conversation's leading message. It then rides in `history` for every
        // later turn; the dropdown locks while the transcript is non-empty
        // (mirrors Chat.tsx — the system message belongs to the conversation).
        if self.transcript.is_empty() {
            if let Some(body) = self.active_system_prompt_body() {
                self.transcript.push(Turn::system(body));
            }
        }
        self.transcript.push(Turn::user(text));
        self.stream_current();
    }

    /// Re-run generation as if the last assistant turn never happened: drop the
    /// trailing assistant message (and anything after it) and resend. Mirrors
    /// Chat.tsx `regenerateLast`.
    fn regenerate_last(&mut self) {
        if self.streaming {
            return;
        }
        let Some(idx) = self.transcript.iter().rposition(|t| t.role == "assistant") else {
            return;
        };
        // Need a user turn before it — else there's nothing to regenerate from.
        if !self.transcript[..idx].iter().any(|t| t.role == "user") {
            return;
        }
        self.transcript.truncate(idx);
        self.pending_question = None;
        self.pending_tools = None;
        self.stream_current();
    }

    /// CLARIFY: the user picked an option. Record the question as an assistant
    /// turn + the answer as a user turn, then continue. Mirrors `chooseOption`.
    fn choose_option(&mut self, opt: String) {
        if self.streaming {
            return;
        }
        let Some(q) = self.pending_question.take() else {
            return;
        };
        self.transcript.push(Turn::assistant(q.prompt));
        self.transcript.push(Turn::user(opt));
        self.stream_current();
    }

    /// Approve a proposed tool call — the chat has no executor, so this just
    /// dismisses the prompt.
    fn approve_tools(&mut self) {
        self.pending_tools = None;
    }

    /// Deny a proposed tool call: append a brief "don't run that" user turn and
    /// continue so the model answers directly. Mirrors `denyTools`.
    fn deny_tools(&mut self) {
        if self.streaming {
            return;
        }
        self.pending_tools = None;
        self.transcript.push(Turn::user(
            "Please don't run that tool — answer directly instead.".into(),
        ));
        self.stream_current();
    }

    /// Clear the transcript (Ctrl+L / New). Aborts any in-flight stream first.
    fn clear_chat(&mut self) {
        self.stop_stream();
        self.transcript.clear();
        self.pending.clear();
        self.streaming = false;
        self.last_usage = None;
        self.status = None;
        self.pending_question = None;
        self.pending_tools = None;
        self.request_id = None;
    }

    /// Stop the in-flight stream: POST `/v1/cancel` so the ENGINE stops
    /// generating (not merely drop SSE chunks), abort the local streaming task,
    /// and commit any partial text so it isn't lost — the same three-part
    /// teardown the web client does on Stop.
    fn stop_stream(&mut self) {
        if !self.streaming {
            return;
        }
        if let Some(id) = self.request_id.take() {
            let (client, ctx) = (self.client.clone(), self.ctx.clone());
            self.rt.spawn(async move {
                let _ = client.cancel(&id).await;
                ctx.request_repaint();
            });
        }
        if let Some(h) = self.chat_task.take() {
            h.abort();
        }
        if !self.pending.is_empty() {
            let text = std::mem::take(&mut self.pending);
            self.transcript.push(Turn::assistant(text));
        }
        self.streaming = false;
    }

    /// Kick off an off-thread save-dialog + write for the transcript. `as_json`
    /// picks the format; the rfd dialog blocks, so it runs on a dedicated OS
    /// thread and reports back over the channel (the UI never stalls).
    fn spawn_export(&self, as_json: bool) {
        let (ext, content) = if as_json {
            ("json", self.build_export_json())
        } else {
            ("md", self.build_export_markdown())
        };
        let default_name = export_filename(ext);
        let (tx, ctx) = (self.tx.clone(), self.ctx.clone());
        let filter_name = if as_json { "JSON" } else { "Markdown" };
        std::thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new()
                .set_file_name(&default_name)
                .add_filter(filter_name, &[ext])
                .set_title("Export conversation")
                .save_file()
            {
                let msg = match std::fs::write(&path, content.as_bytes()) {
                    Ok(()) => UiMsg::ExportDone(path.display().to_string()),
                    Err(e) => UiMsg::ExportError(e.to_string()),
                };
                let _ = tx.send(msg);
                ctx.request_repaint();
            }
        });
    }

    /// Markdown rendering of the transcript (roles bold-labeled, `---` between
    /// turns). Mirrors Chat.tsx `buildExportMarkdown`.
    fn build_export_markdown(&self) -> String {
        let model = self.active_model_id();
        let head = if model.is_empty() {
            "# Conversation\n\nExported from rustllama.\n\n---\n\n".to_string()
        } else {
            format!("# Conversation\n\nExported from rustllama (model: `{model}`).\n\n---\n\n")
        };
        let body = self
            .transcript
            .iter()
            .map(|t| format!("**{}:**\n\n{}", t.role, t.content))
            .collect::<Vec<_>>()
            .join("\n\n---\n\n");
        format!("{head}{body}\n")
    }

    /// JSON dump of the transcript — an OpenAI-compatible `messages` array that
    /// round-trips back into `/v1/chat/completions`. Mirrors `buildExportJson`.
    fn build_export_json(&self) -> String {
        let model = self.active_model_id();
        let messages: Vec<serde_json::Value> = self
            .transcript
            .iter()
            .map(|t| serde_json::json!({ "role": t.role, "content": t.content }))
            .collect();
        let doc = serde_json::json!({
            "model_id": if model.is_empty() { serde_json::Value::Null } else { serde_json::Value::String(model) },
            "messages": messages,
        });
        serde_json::to_string_pretty(&doc).unwrap_or_else(|_| "{}".into())
    }

    fn handle_msg(&mut self, msg: UiMsg) {
        match msg {
            UiMsg::Models(m) => {
                // Keep the current selection if it survived the refresh; else
                // fall back to the first available model.
                if self.selected_model.as_ref().map_or(true, |s| !m.contains(s)) {
                    self.selected_model = m.first().cloned();
                }
                self.models = m;
            }
            UiMsg::LoadedModels(list) => self.loaded_models = list,
            UiMsg::Cached(list) => {
                self.cached = list;
                self.models_error = None;
            }
            UiMsg::Health(ok) => self.health = Some(ok),
            UiMsg::ChatStart(id) => self.request_id = Some(id),
            UiMsg::ChatDelta(t) => {
                // Ignore stragglers that may land after a Stop aborted the task
                // but before the channel drained.
                if self.streaming {
                    self.pending.push_str(&t);
                }
            }
            UiMsg::ChatDone(usage) => {
                if self.streaming {
                    if !self.pending.is_empty() {
                        let text = std::mem::take(&mut self.pending);
                        self.transcript.push(Turn::assistant(text));
                    }
                    self.streaming = false;
                    self.last_usage = usage;
                }
                self.request_id = None;
                self.chat_task = None;
            }
            UiMsg::ChatError(e) => {
                // Commit any partial text so it isn't lost, then surface the error.
                if !self.pending.is_empty() {
                    let text = std::mem::take(&mut self.pending);
                    self.transcript.push(Turn::assistant(text));
                }
                self.streaming = false;
                self.request_id = None;
                self.chat_task = None;
                tracing::warn!(target: "rustllama_gui", "chat error: {e}");
                self.status = Some(format!("Error: {e}"));
            }
            UiMsg::AskUser { prompt, options } => {
                self.pending_tools = None;
                self.pending_question = Some(PendingAsk { prompt, options });
            }
            UiMsg::ToolCalls(calls) => {
                if !calls.is_empty() {
                    if self.auto_tools {
                        // Auto-run accepts silently — note the proposal in the
                        // transcript (the chat has no tool executor).
                        self.transcript
                            .push(Turn::assistant(describe_tool_calls(&calls)));
                    } else {
                        self.pending_question = None;
                        self.pending_tools = Some(calls);
                    }
                }
            }
            UiMsg::ModelLoaded(id) => {
                self.loaded_model = Some(id.clone());
                self.selected_model = Some(id);
                self.status = Some("Model loaded.".into());
                // A load may have pulled in a not-yet-listed model.
                self.spawn_refresh_models();
            }
            UiMsg::ModelActionDone => {
                // A load/unload/default/delete succeeded — refresh both the
                // Models view AND the chat model bar so every surface agrees.
                self.busy_model = None;
                self.models_error = None;
                self.spawn_refresh_models();
                self.spawn_refresh_cached();
            }
            UiMsg::ModelsError(e) => {
                self.busy_model = None;
                tracing::warn!(target: "rustllama_gui", "models: {e}");
                self.models_error = Some(e);
            }
            UiMsg::PickedFile(path) => {
                self.busy_model = Some(path.display().to_string());
                self.models_error = None;
                self.spawn_load_path(path);
            }
            UiMsg::HfResults { query, models } => {
                // Drop a stale response: only accept results for the query the
                // box currently holds (the user may have typed on since).
                if query == self.hf_query.trim() {
                    self.hf_results = models;
                    self.hf_searching = false;
                    self.hf_error = None;
                }
            }
            UiMsg::HfFiles { repo, files } => {
                // Ignore files for a repo we've since navigated away from.
                if self.hf_repo.as_deref() == Some(repo.as_str()) {
                    self.hf_files = files;
                    self.hf_files_loading = false;
                }
            }
            UiMsg::HfError(e) => {
                self.hf_searching = false;
                self.hf_files_loading = false;
                self.hf_error = Some(e);
            }
            UiMsg::PullProgress {
                status,
                completed,
                total,
            } => {
                // Show a byte-percent line + bar when the server reports both
                // counts; otherwise just echo the status phase text.
                match (completed, total) {
                    (Some(c), Some(t)) if t > 0 => {
                        let frac = (c as f32 / t as f32).clamp(0.0, 1.0);
                        self.pull_pct = Some(frac);
                        self.pull_status = Some(format!(
                            "{status} — {:.0}% ({} / {})",
                            frac * 100.0,
                            human_bytes(c),
                            human_bytes(t)
                        ));
                    }
                    _ => {
                        if !status.is_empty() {
                            self.pull_status = Some(status);
                        }
                    }
                }
            }
            UiMsg::PullDone => {
                self.pulling = false;
                self.pull_pct = None;
                self.pull_status = Some("done".into());
                self.spawn_refresh_cached();
            }
            UiMsg::PullError(e) => {
                self.pulling = false;
                self.pull_pct = None;
                self.pull_status = Some(format!("error: {e}"));
            }
            UiMsg::Metrics(m) => {
                self.metrics_inflight = false;
                self.health = Some(true);
                self.metrics = Some(*m);
            }
            UiMsg::MetricsError => {
                self.metrics_inflight = false;
                self.health = Some(false);
                self.metrics = None;
            }
            UiMsg::Config {
                system_prompts,
                ctx_size,
            } => {
                self.system_prompts = system_prompts;
                self.config_ctx_size = ctx_size;
                // Auto-pick a default-for-model prompt now that we have the list.
                self.maybe_autoselect_prompt();
            }
            UiMsg::DraftTokens { text, count } => {
                self.tok_inflight = false;
                // Only accept the count if the composer still holds that text.
                if text == self.input {
                    self.draft_tokens = Some(count);
                }
            }
            UiMsg::DraftTokenError => {
                self.tok_inflight = false;
                // Server can't tokenize (mock engine) — hide the counter.
                self.draft_tokens = None;
            }
            UiMsg::ExportDone(path) => {
                self.status = Some(format!("Exported to {path}"));
            }
            UiMsg::ExportError(e) => {
                self.status = Some(format!("Export failed: {e}"));
            }
            UiMsg::Err(e) => {
                tracing::debug!(target: "rustllama_gui", "{e}");
                self.status = Some(e);
            }
        }
    }
}

impl eframe::App for GuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // 1) Drain the async channel into state. Non-blocking: `try_recv`
        //    pulls whatever arrived since the last frame. (Collect first so the
        //    receiver borrow is released before `handle_msg`, which may spawn.)
        let mut incoming = Vec::new();
        while let Ok(msg) = self.rx.try_recv() {
            incoming.push(msg);
        }
        for msg in incoming {
            self.handle_msg(msg);
        }

        // Track the active model for system-prompt auto-selection (no-op once
        // the user has picked a prompt or when nothing matches).
        self.maybe_autoselect_prompt();

        let now = Instant::now();

        // 2) Status-bar metrics poll (~2 s cadence). Gate on an in-flight flag
        //    so a slow server doesn't queue overlapping requests. The
        //    `request_repaint_after` keeps the poll alive when the UI is idle.
        let due = self
            .metrics_last_poll
            .map_or(true, |t| now.duration_since(t) >= Duration::from_secs(2));
        if due && !self.metrics_inflight {
            self.metrics_inflight = true;
            self.metrics_last_poll = Some(now);
            self.spawn_metrics();
        }
        ctx.request_repaint_after(Duration::from_secs(2));

        // 3) HuggingFace search debounce: fire ~300 ms after the last keystroke
        //    (avoids a request per character). A pending debounce schedules a
        //    near-term repaint so the timer actually elapses without input.
        if let Some(t) = self.hf_dirty_since {
            if now.duration_since(t) >= Duration::from_millis(300) {
                self.hf_dirty_since = None;
                let q = self.hf_query.trim().to_string();
                if q.len() >= 2 && q != self.hf_last_searched {
                    self.hf_last_searched = q.clone();
                    self.hf_searching = true;
                    self.spawn_hf_search(q);
                } else if q.len() < 2 {
                    self.hf_results.clear();
                    self.hf_searching = false;
                }
            } else {
                ctx.request_repaint_after(Duration::from_millis(300));
            }
        }

        // Composer token-counter debounce: ~300 ms after the last keystroke,
        // POST the draft to `/v1/tokenize`. Non-blocking (channel bridge); the
        // response is tagged with its text so a stale count is dropped.
        if let Some(t) = self.tok_dirty_since {
            if now.duration_since(t) >= Duration::from_millis(300) {
                self.tok_dirty_since = None;
                let text = self.input.clone();
                if text.trim().is_empty() {
                    self.draft_tokens = Some(0);
                } else if text != self.tok_last_text && !self.tok_inflight {
                    self.tok_last_text = text.clone();
                    self.tok_inflight = true;
                    self.spawn_tokenize(text);
                }
            } else {
                ctx.request_repaint_after(Duration::from_millis(300));
            }
        }

        // Deferred actions gathered from immediate-mode widgets, applied after
        // the panels close (see the tail of this fn).
        let mut actions: Vec<Action> = Vec::new();

        // 4) Left nav rail (all views).
        egui::SidePanel::left("nav")
            .resizable(false)
            .exact_width(170.0)
            .frame(
                egui::Frame::default()
                    .fill(SIDEBAR)
                    .inner_margin(Margin::symmetric(10.0, 12.0)),
            )
            .show(ctx, |ui| {
                // Brand: accent chip + wordmark.
                ui.horizontal(|ui| {
                    let (r, _) = ui.allocate_exact_size(egui::vec2(26.0, 26.0), Sense::hover());
                    ui.painter().rect_filled(r, Rounding::same(7.0), ACCENT);
                    ui.painter().text(
                        r.center(),
                        Align2::CENTER_CENTER,
                        "r",
                        FontId::proportional(16.0),
                        TEXT,
                    );
                    ui.add_space(2.0);
                    ui.label(RichText::new("rustllama").strong().color(TEXT));
                });
                ui.add_space(16.0);

                if nav_item(ui, NavIcon::Chat, "Chat", self.view == View::Chat, true) {
                    actions.push(Action::SwitchView(View::Chat));
                }
                ui.add_space(3.0);
                if nav_item(ui, NavIcon::Models, "Models", self.view == View::Models, true) {
                    actions.push(Action::SwitchView(View::Models));
                }
                ui.add_space(3.0);
                // Room for later phases — rendered but disabled.
                nav_item(ui, NavIcon::Placeholder, "Decide", false, false);
                ui.add_space(3.0);
                nav_item(ui, NavIcon::Placeholder, "Status", false, false);
                ui.add_space(3.0);
                nav_item(ui, NavIcon::Placeholder, "Quantize", false, false);
                ui.add_space(3.0);
                nav_item(ui, NavIcon::Placeholder, "Settings", false, false);
            });

        // 5) Status bar (all views) — added before any per-view bottom panel so
        //    it sits at the very bottom, spanning the content area.
        egui::TopBottomPanel::bottom("statusbar")
            .frame(
                egui::Frame::default()
                    .fill(SIDEBAR)
                    .inner_margin(Margin::symmetric(12.0, 6.0))
                    .stroke(Stroke::new(1.0, BORDER)),
            )
            .show(ctx, |ui| {
                render_status_bar(
                    ui,
                    self.health,
                    self.metrics.as_ref(),
                    self.loaded_model.as_deref(),
                );
            });

        // 6) Per-view central content.
        match self.view {
            View::Chat => self.render_chat(ctx),
            View::Models => self.render_models(ctx, &mut actions),
        }

        // 7) Delete-confirmation modal (Models view). A floating egui window so
        //    it overlays whatever's behind it without blocking the UI thread.
        if let Some(name) = self.confirm_delete.clone() {
            egui::Window::new("Delete cached model")
                .collapsible(false)
                .resizable(false)
                .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.set_max_width(360.0);
                    ui.label(RichText::new(display_name(&name)).strong().color(TEXT));
                    ui.add_space(4.0);
                    ui.label(
                        RichText::new("This removes the GGUF from disk. This can't be undone.")
                            .color(MUTED),
                    );
                    ui.add_space(12.0);
                    ui.horizontal(|ui| {
                        if ui.button("Cancel").clicked() {
                            actions.push(Action::CancelDelete);
                        }
                        if ui
                            .add(egui::Button::new(RichText::new("Delete").color(TEXT)).fill(DANGER))
                            .clicked()
                        {
                            actions.push(Action::ConfirmDelete(name.clone()));
                        }
                    });
                });
        }

        // 8) Apply the deferred actions, now that `self` is free of panel
        //    borrows and the spawn helpers can access it cleanly.
        for action in actions {
            match action {
                Action::SwitchView(v) => {
                    self.view = v;
                    // Freshen the lists when entering Models.
                    if v == View::Models {
                        self.spawn_refresh_models();
                        self.spawn_refresh_cached();
                    }
                }
                Action::RefreshModels => {
                    self.models_error = None;
                    self.spawn_refresh_models();
                    self.spawn_refresh_cached();
                }
                Action::LoadCached(name) => {
                    self.busy_model = Some(name.clone());
                    self.models_error = None;
                    self.spawn_load_cached(name);
                }
                Action::PickFile => self.spawn_pick_file(),
                Action::SetDefault(id) => {
                    self.busy_model = Some(id.clone());
                    self.spawn_set_default(id);
                }
                Action::UnloadModel(id) => {
                    self.busy_model = Some(id.clone());
                    self.spawn_unload(id);
                }
                Action::AskDelete(name) => self.confirm_delete = Some(name),
                Action::ConfirmDelete(name) => {
                    self.confirm_delete = None;
                    self.busy_model = Some(name.clone());
                    self.spawn_delete(name);
                }
                Action::CancelDelete => self.confirm_delete = None,
                Action::HfPickRepo(repo) => {
                    self.hf_repo = Some(repo.clone());
                    self.hf_files.clear();
                    self.hf_files_loading = true;
                    self.hf_error = None;
                    self.spawn_hf_files(repo);
                }
                Action::HfBack => self.hf_repo = None,
                Action::HfPickFile { repo, file } => {
                    // Stage the concrete ref into the Pull field, collapse search.
                    self.pull_ref = format!("{repo}:{file}");
                    self.hf_repo = None;
                    self.hf_results.clear();
                    self.hf_query.clear();
                    self.hf_last_searched.clear();
                    self.hf_dirty_since = None;
                }
                Action::Pull(r) => {
                    if !self.pulling && !r.trim().is_empty() {
                        self.pulling = true;
                        self.pull_pct = None;
                        self.pull_status = Some("starting…".into());
                        self.spawn_pull(r);
                    }
                }
            }
        }
    }

    /// Persist the header toggles across runs. Only effective when eframe is
    /// built with the `persistence` feature (this crate isn't, today), in which
    /// case it degrades to in-session state — the toggles still work, they just
    /// don't survive a restart. Mirrors the web UI's localStorage keys.
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        storage.set_string("auto_tools", if self.auto_tools { "1" } else { "0" }.into());
        storage.set_string("clarify", if self.clarify { "1" } else { "0" }.into());
    }
}

impl GuiApp {
    /// The Chat view: model bar + toolbar (top), composer (bottom), transcript
    /// (center). Widgets set local intent flags that are applied after the
    /// panels close, so the spawn helpers get clean access to `self`.
    //
    // DEFERRED (later phases, intentionally not built here):
    //   - Image attach / multimodal (OpenAI image_url content blocks) — the
    //     composer is text-only for now.
    //   - Conversation-history sidebar (the server's `/api/conversations`
    //     sqlite routes, `--features history`).
    //   - Auto context compaction (summarize older turns to fit ctx_size).
    fn render_chat(&mut self, ctx: &egui::Context) {
        // Intent flags collected during immediate-mode rendering.
        let mut do_load = false;
        let mut do_unload = false;
        let mut do_refresh = false;
        let mut do_send = false;
        let mut do_stop = false;
        let mut do_new = false;
        let mut do_regen = false;
        let mut do_export_md = false;
        let mut do_export_json = false;
        let mut open_shortcuts = false;
        let mut prompt_picked = false;
        let mut input_changed = false;
        let mut chosen_option: Option<String> = None;
        let mut do_approve = false;
        let mut do_deny = false;

        // Model bar.
        let models = self.models.clone(); // cheap; avoids nested-closure borrows
        egui::TopBottomPanel::top("model_bar")
            .frame(
                egui::Frame::default()
                    .fill(BG)
                    .inner_margin(Margin::symmetric(12.0, 8.0))
                    .stroke(Stroke::new(1.0, BORDER)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Model").color(MUTED));
                    egui::ComboBox::from_id_salt("model_combo")
                        .width(300.0)
                        .selected_text(
                            self.selected_model
                                .clone()
                                .unwrap_or_else(|| "Select a model".to_owned()),
                        )
                        .show_ui(ui, |ui| {
                            for m in &models {
                                ui.selectable_value(
                                    &mut self.selected_model,
                                    Some(m.clone()),
                                    m.as_str(),
                                );
                            }
                        });
                    if ui.button("Load").clicked() {
                        do_load = true;
                    }
                    if ui.button("Unload").clicked() {
                        do_unload = true;
                    }
                    if ui.button("Refresh").clicked() {
                        do_refresh = true;
                    }

                    // Health dot + status, pushed to the right edge.
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let (col, tip) = match self.health {
                            Some(true) => (HEALTH_OK, "server: healthy"),
                            Some(false) => (HEALTH_BAD, "server: unreachable"),
                            None => (MUTED, "server: unknown"),
                        };
                        let (rect, resp) =
                            ui.allocate_exact_size(egui::vec2(14.0, 14.0), Sense::hover());
                        ui.painter().circle_filled(rect.center(), 5.0, col);
                        resp.on_hover_text(format!("{tip}\n{}", self.base_url));
                        if let Some(s) = &self.status {
                            ui.label(RichText::new(s).color(MUTED).small());
                        }
                    });
                });
            });

        // Chat toolbar: New, Sampling popover, tool/clarify toggles, regen /
        // export, shortcuts help, and the last-turn usage line.
        let usage_line = self.last_usage.as_ref().map(format_usage);
        let transcript_empty = self.transcript.is_empty();
        let has_assistant = self.transcript.iter().any(|t| t.role == "assistant");
        let sampling_is_default = self.sampling.is_default();
        let streaming = self.streaming;
        egui::TopBottomPanel::top("chat_toolbar")
            .frame(
                egui::Frame::default()
                    .fill(BG)
                    .inner_margin(Margin::symmetric(12.0, 6.0))
                    .stroke(Stroke::new(1.0, BORDER)),
            )
            .show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .add_enabled(!transcript_empty, egui::Button::new("New"))
                        .on_hover_text("Clear the transcript (Ctrl+L)")
                        .clicked()
                    {
                        do_new = true;
                    }

                    // Sampling popover — sliders bind straight to self.sampling.
                    let sampling_label = if sampling_is_default {
                        "Sampling"
                    } else {
                        "Sampling · custom"
                    };
                    ui.menu_button(sampling_label, |ui| {
                        ui.set_min_width(260.0);
                        ui.label(RichText::new("Sampling").strong().color(TEXT));
                        ui.add_space(4.0);
                        ui.add(
                            egui::Slider::new(&mut self.sampling.temperature, 0.0..=2.0)
                                .text("temperature"),
                        );
                        ui.add(
                            egui::Slider::new(&mut self.sampling.top_p, 0.0..=1.0).text("top_p"),
                        );
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::DragValue::new(&mut self.sampling.top_k)
                                    .range(0..=1000)
                                    .speed(1.0),
                            );
                            ui.label(RichText::new("top_k (0 = off)").color(MUTED).small());
                        });
                        ui.add(
                            egui::Slider::new(&mut self.sampling.repeat_penalty, 0.8..=2.0)
                                .text("repeat_penalty"),
                        );
                        ui.horizontal(|ui| {
                            ui.add(
                                egui::DragValue::new(&mut self.sampling.max_tokens)
                                    .range(1..=32768)
                                    .speed(4.0),
                            );
                            ui.label(RichText::new("max_tokens").color(MUTED).small());
                        });
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("seed").color(MUTED).small());
                            ui.add(
                                egui::TextEdit::singleline(&mut self.sampling.seed)
                                    .hint_text("random")
                                    .desired_width(120.0),
                            );
                        });
                        ui.add_space(6.0);
                        if ui.button("Reset to defaults").clicked() {
                            self.sampling = Sampling::default();
                        }
                        ui.label(
                            RichText::new("Values at their default are left unset.")
                                .color(MUTED)
                                .small(),
                        );
                    })
                    .response
                    .on_hover_text("Temperature / top_p / top_k / repeat_penalty / max_tokens / seed");

                    ui.checkbox(&mut self.auto_tools, "Auto-run tools").on_hover_text(
                        "When on, proposed tool calls are accepted without asking. This chat \
                         never executes tools — it only gates the confirm prompt.",
                    );
                    ui.checkbox(&mut self.clarify, "Clarifying questions").on_hover_text(
                        "When on, the model can pause and ask a clarifying question with \
                         selectable options instead of guessing.",
                    );

                    if ui
                        .add_enabled(has_assistant && !streaming, egui::Button::new("Regenerate"))
                        .on_hover_text("Re-run the last user turn (Ctrl+R)")
                        .clicked()
                    {
                        do_regen = true;
                    }
                    if ui
                        .add_enabled(!transcript_empty && !streaming, egui::Button::new("Export MD"))
                        .on_hover_text("Save the transcript as Markdown (Ctrl+E)")
                        .clicked()
                    {
                        do_export_md = true;
                    }
                    if ui
                        .add_enabled(
                            !transcript_empty && !streaming,
                            egui::Button::new("Export JSON"),
                        )
                        .on_hover_text("Save the transcript as JSON")
                        .clicked()
                    {
                        do_export_json = true;
                    }

                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if ui.button("?").on_hover_text("Keyboard shortcuts").clicked() {
                            open_shortcuts = true;
                        }
                        if let Some(line) = &usage_line {
                            ui.label(RichText::new(line).color(MUTED).small());
                        }
                    });
                });
            });

        // Composer (bottom): system-prompt selector, message field, token
        // counter, and Send / Stop.
        let active_model = self.active_model_id();
        let prompts: Vec<(String, String)> = self
            .system_prompts
            .iter()
            .map(|p| {
                let mut label = p.name.clone();
                if !p.default_for_model.is_empty() && p.default_for_model == active_model {
                    label.push_str("  (default for this model)");
                }
                (p.name.clone(), label)
            })
            .collect();
        let show_prompts = !prompts.is_empty();
        let budget = self.ctx_budget();
        let draft_tokens = self.draft_tokens;
        egui::TopBottomPanel::bottom("composer")
            .frame(
                egui::Frame::default()
                    .fill(BG)
                    .inner_margin(Margin::symmetric(12.0, 10.0)),
            )
            .show(ctx, |ui| {
                // System-prompt selector (locked once the conversation starts —
                // the system message belongs to the conversation, not the live
                // dropdown; mirrors Chat.tsx).
                if show_prompts {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("System prompt").color(MUTED).small());
                        ui.add_enabled_ui(transcript_empty, |ui| {
                            egui::ComboBox::from_id_salt("sysprompt_combo")
                                .width(240.0)
                                .selected_text(if self.selected_prompt.is_empty() {
                                    "— none —".to_string()
                                } else {
                                    self.selected_prompt.clone()
                                })
                                .show_ui(ui, |ui| {
                                    if ui
                                        .selectable_value(
                                            &mut self.selected_prompt,
                                            String::new(),
                                            "— none —",
                                        )
                                        .clicked()
                                    {
                                        prompt_picked = true;
                                    }
                                    for (name, label) in &prompts {
                                        if ui
                                            .selectable_value(
                                                &mut self.selected_prompt,
                                                name.clone(),
                                                label.as_str(),
                                            )
                                            .clicked()
                                        {
                                            prompt_picked = true;
                                        }
                                    }
                                });
                        });
                        if !transcript_empty {
                            ui.label(
                                RichText::new("(locked for this conversation)")
                                    .color(DISABLED)
                                    .small(),
                            );
                        }
                    });
                    ui.add_space(6.0);
                }

                let te = ui.add_sized(
                    [ui.available_width(), 64.0],
                    egui::TextEdit::multiline(&mut self.input)
                        .hint_text("Message the model…  (Enter to send · Shift+Enter for newline)")
                        .desired_rows(3),
                );
                if te.changed() {
                    input_changed = true;
                }
                // Enter (no Shift) or Ctrl+Enter sends; Shift+Enter inserts a
                // newline. `send_message` trims, so the '\n' the field inserts
                // this same frame is dropped.
                let enter_send = te.has_focus()
                    && ui.input(|i| i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
                if enter_send && !self.streaming && !self.input.trim().is_empty() {
                    do_send = true;
                }

                // Token counter (built after the TextEdit so it reflects this
                // frame's typing).
                let counter = if self.input.trim().is_empty() {
                    RichText::new("Enter to send · Shift+Enter for a newline")
                        .color(MUTED)
                        .small()
                } else {
                    let chars = self.input.chars().count();
                    match draft_tokens {
                        Some(n) => match budget {
                            Some(b) if b > 0 => {
                                let pct = (n as f64 / b as f64) * 100.0;
                                let col = if pct >= 90.0 {
                                    HEALTH_BAD
                                } else if pct >= 50.0 {
                                    WARN
                                } else {
                                    MUTED
                                };
                                RichText::new(format!(
                                    "{n} / {b} tokens · {pct:.0}% of ctx · {chars} chars"
                                ))
                                .color(col)
                                .small()
                            }
                            _ => RichText::new(format!("{n} tokens · {chars} chars"))
                                .color(MUTED)
                                .small(),
                        },
                        None => RichText::new(format!("{chars} chars · counting…"))
                            .color(MUTED)
                            .small(),
                    }
                };

                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label(counter);
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if self.streaming {
                            if ui
                                .add(egui::Button::new(RichText::new("Stop").color(TEXT)).fill(DANGER))
                                .on_hover_text("Stop generating (Esc)")
                                .clicked()
                            {
                                do_stop = true;
                            }
                            ui.spinner();
                            ui.label(RichText::new("generating…").color(MUTED).small());
                        } else {
                            let can_send = !self.input.trim().is_empty();
                            if ui.add_enabled(can_send, egui::Button::new("Send")).clicked() {
                                do_send = true;
                            }
                        }
                    });
                });
            });

        // Transcript. Bind the fields the closure needs up front as disjoint
        // borrows (shared reads + mut `md_cache`).
        let transcript = &self.transcript;
        let pending = &self.pending;
        let streaming = self.streaming;
        let pending_question = &self.pending_question;
        let pending_tools = &self.pending_tools;
        let cache = &mut self.md_cache;
        egui::CentralPanel::default()
            .frame(
                egui::Frame::default()
                    .fill(BG)
                    .inner_margin(Margin::same(14.0)),
            )
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        if transcript.is_empty()
                            && !streaming
                            && pending_question.is_none()
                            && pending_tools.is_none()
                        {
                            ui.add_space(64.0);
                            ui.vertical_centered(|ui| {
                                ui.label(
                                    RichText::new("Start a conversation").heading().color(MUTED),
                                );
                            });
                            return;
                        }
                        for (idx, turn) in transcript.iter().enumerate() {
                            render_turn(ui, cache, turn, idx);
                        }
                        // Live streaming assistant text (think-aware card).
                        if streaming {
                            ui.add_space(6.0);
                            render_assistant_card(ui, cache, pending, usize::MAX);
                        }
                        // CLARIFY chooser: clickable option buttons.
                        if let Some(q) = pending_question {
                            ui.add_space(10.0);
                            egui::Frame::none()
                                .fill(ELEVATED)
                                .rounding(Rounding::same(8.0))
                                .stroke(Stroke::new(1.0, BORDER))
                                .inner_margin(Margin::same(12.0))
                                .show(ui, |ui| {
                                    let prompt = if q.prompt.is_empty() {
                                        "Choose an option:"
                                    } else {
                                        q.prompt.as_str()
                                    };
                                    ui.label(RichText::new(prompt).color(TEXT));
                                    ui.add_space(8.0);
                                    ui.horizontal_wrapped(|ui| {
                                        for opt in &q.options {
                                            if ui
                                                .add_enabled(
                                                    !streaming,
                                                    egui::Button::new(opt.as_str()),
                                                )
                                                .clicked()
                                            {
                                                chosen_option = Some(opt.clone());
                                            }
                                        }
                                    });
                                });
                        }
                        // Tool-confirm: Approve / Deny (auto-run OFF).
                        if let Some(calls) = pending_tools {
                            ui.add_space(10.0);
                            egui::Frame::none()
                                .fill(WARN.gamma_multiply(0.12))
                                .rounding(Rounding::same(8.0))
                                .stroke(Stroke::new(1.0, WARN))
                                .inner_margin(Margin::same(12.0))
                                .show(ui, |ui| {
                                    ui.label(
                                        RichText::new(format!(
                                            "The model proposes a tool call: {}. This chat can't \
                                             run tools — approve to acknowledge, or deny to have \
                                             it answer directly.",
                                            describe_tool_calls(calls)
                                        ))
                                        .color(TEXT),
                                    );
                                    ui.add_space(8.0);
                                    ui.horizontal(|ui| {
                                        if ui
                                            .add_enabled(!streaming, egui::Button::new("Approve"))
                                            .clicked()
                                        {
                                            do_approve = true;
                                        }
                                        if ui
                                            .add_enabled(!streaming, egui::Button::new("Deny"))
                                            .clicked()
                                        {
                                            do_deny = true;
                                        }
                                    });
                                });
                        }
                    });
            });

        // Keyboard shortcuts overlay (a floating window, like the delete modal).
        if self.show_shortcuts {
            let mut close = false;
            egui::Window::new("Keyboard shortcuts")
                .collapsible(false)
                .resizable(false)
                .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.set_max_width(420.0);
                    for (keys, what) in SHORTCUTS {
                        ui.horizontal(|ui| {
                            ui.add_sized(
                                [140.0, 18.0],
                                egui::Label::new(RichText::new(*keys).monospace().color(TEXT)),
                            );
                            ui.label(RichText::new(*what).color(MUTED));
                        });
                    }
                    ui.add_space(10.0);
                    if ui.button("Close").clicked() {
                        close = true;
                    }
                });
            if close {
                self.show_shortcuts = false;
            }
        }

        // Global chat shortcuts (Esc / Ctrl+L / Ctrl+R / Ctrl+E / ?). Enter is
        // handled in the composer above so it can gate on field focus. `?` only
        // fires when no widget is focused (so typing `?` into the composer
        // isn't swallowed) — mirrors Chat.tsx's isTypingTarget guard.
        let typing = ctx.memory(|m| m.focused().is_some());
        let (mut sc_clear, mut sc_regen, mut sc_export, mut sc_esc, mut sc_help) =
            (false, false, false, false, false);
        ctx.input(|i| {
            let cmd = i.modifiers.command; // ctrl on win/linux, ⌘ on mac
            sc_esc = i.key_pressed(egui::Key::Escape);
            sc_clear = cmd && i.key_pressed(egui::Key::L);
            sc_regen = cmd && i.key_pressed(egui::Key::R);
            sc_export = cmd && i.key_pressed(egui::Key::E);
            if !typing {
                sc_help = i
                    .events
                    .iter()
                    .any(|e| matches!(e, egui::Event::Text(t) if t == "?"));
            }
        });
        if sc_help {
            self.show_shortcuts = !self.show_shortcuts;
        }
        if sc_esc {
            // Esc closes the overlay first, else stops an in-flight stream.
            if self.show_shortcuts {
                self.show_shortcuts = false;
            } else if self.streaming {
                do_stop = true;
            }
        }
        if sc_clear {
            do_new = true;
        }
        if sc_regen {
            do_regen = true;
        }
        if sc_export {
            do_export_md = true;
        }

        // Apply intents now that the panels have closed and `self` is free.
        if open_shortcuts {
            self.show_shortcuts = true;
        }
        if prompt_picked {
            self.user_picked_prompt = true;
        }
        if input_changed {
            self.tok_dirty_since = Some(Instant::now());
        }
        if do_refresh {
            self.spawn_refresh_models();
        }
        if do_load {
            if let Some(m) = self.selected_model.clone().filter(|s| !s.is_empty()) {
                self.spawn_load(m);
            }
        }
        if do_unload {
            if let Some(m) = self.selected_model.clone().filter(|s| !s.is_empty()) {
                self.spawn_unload(m);
            }
        }
        if do_stop {
            self.stop_stream();
        }
        if do_new {
            self.clear_chat();
        }
        if do_regen {
            self.regenerate_last();
        }
        if let Some(opt) = chosen_option {
            self.choose_option(opt);
        }
        if do_approve {
            self.approve_tools();
        }
        if do_deny {
            self.deny_tools();
        }
        if do_export_md {
            self.spawn_export(false);
        }
        if do_export_json {
            self.spawn_export(true);
        }
        if do_send {
            self.send_message();
        }
    }

    /// The Models view: Loaded list, cached "My Models" list, and the
    /// HuggingFace pull card. Widgets push [`Action`]s (applied by `update`);
    /// the lists are cloned into locals up front so the closure can also take
    /// `&mut self.hf_query` / `&mut self.pull_ref` for the text fields without
    /// aliasing.
    fn render_models(&mut self, ctx: &egui::Context, actions: &mut Vec<Action>) {
        let loaded = self.loaded_models.clone();
        let cached = self.cached.clone();
        let hf_results = self.hf_results.clone();
        let hf_files = self.hf_files.clone();
        let hf_repo = self.hf_repo.clone();
        let busy = self.busy_model.clone();

        egui::CentralPanel::default()
            .frame(
                egui::Frame::default()
                    .fill(BG)
                    .inner_margin(Margin::same(16.0)),
            )
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        // Header.
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Models").heading().color(TEXT));
                            ui.add_space(6.0);
                            if ui.button("Refresh").clicked() {
                                actions.push(Action::RefreshModels);
                            }
                            if ui.button("Load from file…").clicked() {
                                actions.push(Action::PickFile);
                            }
                        });
                        if let Some(e) = &self.models_error {
                            ui.add_space(8.0);
                            error_banner(ui, e);
                        }
                        ui.add_space(14.0);

                        // --- Loaded ---
                        card(ui, |ui| {
                            ui.label(
                                RichText::new(format!("Loaded ({})", loaded.len()))
                                    .strong()
                                    .color(TEXT),
                            );
                            ui.add_space(8.0);
                            if loaded.is_empty() {
                                ui.label(
                                    RichText::new(
                                        "No models loaded. Pick one from My Models below.",
                                    )
                                    .color(MUTED),
                                );
                            }
                            for m in &loaded {
                                let is_busy = busy.as_deref() == Some(m.id.as_str());
                                ui.horizontal(|ui| {
                                    ui.label(RichText::new(&m.id).monospace().color(TEXT));
                                    ui.with_layout(
                                        Layout::right_to_left(Align::Center),
                                        |ui| {
                                            if is_busy {
                                                ui.spinner();
                                            }
                                            if ui
                                                .add_enabled(!is_busy, egui::Button::new("Unload"))
                                                .clicked()
                                            {
                                                actions.push(Action::UnloadModel(m.id.clone()));
                                            }
                                            if m.is_default {
                                                pill(ui, "default", HEALTH_OK);
                                            } else if ui
                                                .add_enabled(
                                                    !is_busy,
                                                    egui::Button::new("Set default"),
                                                )
                                                .clicked()
                                            {
                                                actions.push(Action::SetDefault(m.id.clone()));
                                            }
                                        },
                                    );
                                });
                                ui.add_space(2.0);
                            }
                        });

                        ui.add_space(16.0);

                        // --- Pull from HuggingFace ---
                        card(ui, |ui| {
                            ui.label(
                                RichText::new("Pull from HuggingFace").strong().color(TEXT),
                            );
                            ui.add_space(4.0);
                            ui.label(
                                RichText::new("Format: owner/repo:filename.gguf")
                                    .color(MUTED)
                                    .small(),
                            );
                            ui.add_space(8.0);

                            // Debounced search box.
                            let resp = ui.add(
                                egui::TextEdit::singleline(&mut self.hf_query)
                                    .desired_width(f32::INFINITY)
                                    .hint_text("Search HuggingFace for GGUF models…"),
                            );
                            if resp.changed() {
                                self.hf_dirty_since = Some(Instant::now());
                                self.hf_repo = None; // typing resets any expansion
                            }

                            // Results / files dropdown (inline panel).
                            let show_dropdown = self.hf_searching
                                || !hf_results.is_empty()
                                || hf_repo.is_some()
                                || self.hf_error.is_some();
                            if show_dropdown {
                                ui.add_space(6.0);
                                egui::Frame::none()
                                    .fill(ELEVATED)
                                    .rounding(Rounding::same(8.0))
                                    .stroke(Stroke::new(1.0, BORDER))
                                    .inner_margin(Margin::same(6.0))
                                    .show(ui, |ui| {
                                        egui::ScrollArea::vertical()
                                            .max_height(240.0)
                                            .auto_shrink([false, true])
                                            .show(ui, |ui| {
                                                if self.hf_searching {
                                                    ui.label(
                                                        RichText::new("searching…")
                                                            .color(MUTED)
                                                            .small(),
                                                    );
                                                }
                                                if let Some(e) = &self.hf_error {
                                                    ui.label(
                                                        RichText::new(e).color(DANGER).small(),
                                                    );
                                                }
                                                match &hf_repo {
                                                    None => {
                                                        for m in &hf_results {
                                                            if hf_result_row(ui, m) {
                                                                actions.push(Action::HfPickRepo(
                                                                    m.id.clone(),
                                                                ));
                                                            }
                                                        }
                                                    }
                                                    Some(repo) => {
                                                        if ui
                                                            .add(
                                                                egui::Label::new(
                                                                    RichText::new(format!(
                                                                        "← {repo} (back)"
                                                                    ))
                                                                    .color(MUTED)
                                                                    .small(),
                                                                )
                                                                .sense(Sense::click()),
                                                            )
                                                            .clicked()
                                                        {
                                                            actions.push(Action::HfBack);
                                                        }
                                                        if self.hf_files_loading {
                                                            ui.label(
                                                                RichText::new("loading files…")
                                                                    .color(MUTED)
                                                                    .small(),
                                                            );
                                                        }
                                                        for f in &hf_files {
                                                            if hf_file_row(ui, f) {
                                                                actions.push(Action::HfPickFile {
                                                                    repo: repo.clone(),
                                                                    file: f.rfilename.clone(),
                                                                });
                                                            }
                                                        }
                                                        if !self.hf_files_loading
                                                            && hf_files.is_empty()
                                                        {
                                                            ui.label(
                                                                RichText::new(
                                                                    "no .gguf files in this repo",
                                                                )
                                                                .color(MUTED)
                                                                .small(),
                                                            );
                                                        }
                                                    }
                                                }
                                            });
                                    });
                            }

                            ui.add_space(8.0);
                            // Staged ref + Pull button.
                            ui.horizontal(|ui| {
                                ui.add_enabled(
                                    !self.pulling,
                                    egui::TextEdit::singleline(&mut self.pull_ref)
                                        .desired_width(ui.available_width() - 90.0)
                                        .hint_text("owner/repo:filename.gguf"),
                                );
                                let can_pull = !self.pulling && !self.pull_ref.trim().is_empty();
                                if ui
                                    .add_enabled(
                                        can_pull,
                                        egui::Button::new(if self.pulling {
                                            "pulling…"
                                        } else {
                                            "Pull"
                                        }),
                                    )
                                    .clicked()
                                {
                                    actions.push(Action::Pull(self.pull_ref.clone()));
                                }
                            });
                            if let Some(pct) = self.pull_pct {
                                ui.add_space(6.0);
                                ui.add(
                                    egui::ProgressBar::new(pct)
                                        .desired_width(f32::INFINITY)
                                        .fill(ACCENT),
                                );
                            }
                            if let Some(s) = &self.pull_status {
                                ui.add_space(4.0);
                                let col = if s.starts_with("error") { DANGER } else { MUTED };
                                ui.label(RichText::new(s).color(col).monospace().small());
                            }
                        });

                        ui.add_space(16.0);

                        // --- My Models (cached) ---
                        card(ui, |ui| {
                            ui.label(
                                RichText::new(format!("My Models ({})", cached.len()))
                                    .strong()
                                    .color(TEXT),
                            );
                            ui.add_space(8.0);
                            if cached.is_empty() {
                                ui.label(
                                    RichText::new(
                                        "No models on disk yet — pull one from HuggingFace above.",
                                    )
                                    .color(MUTED),
                                );
                            }
                            for m in &cached {
                                let is_busy = busy.as_deref() == Some(m.name.as_str());
                                cached_row(ui, m, is_busy, actions);
                                ui.add_space(6.0);
                            }
                        });
                    });
            });
    }
}

// --- free helpers ---------------------------------------------------------

/// Extract the `data[].id` strings from a `/v1/models` payload.
fn parse_model_ids(v: &serde_json::Value) -> Vec<String> {
    v.get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| m.get("id").and_then(|i| i.as_str()).map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// Extract `{id, is_default}` per loaded model from a `/v1/models` payload
/// (the rustllama extension adds `is_default` to each `data[]` entry).
fn parse_loaded_models(v: &serde_json::Value) -> Vec<LoadedModel> {
    v.get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|m| {
                    let id = m.get("id")?.as_str()?.to_string();
                    let is_default = m
                        .get("is_default")
                        .and_then(|b| b.as_bool())
                        .unwrap_or(false);
                    Some(LoadedModel { id, is_default })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// One transcript turn. User/system turns render as plain (wrapped) text in a
/// subtle card; assistant turns render as a markdown card with `<think>`
/// reasoning collapsed (see [`render_assistant_card`]). `idx` seeds a stable id
/// for each assistant card's collapse state.
fn render_turn(ui: &mut egui::Ui, cache: &mut CommonMarkCache, turn: &Turn, idx: usize) {
    ui.add_space(8.0);
    match turn.role.as_str() {
        "user" => {
            ui.label(RichText::new("You").strong().color(MUTED));
            egui::Frame::none()
                .fill(ELEVATED2)
                .rounding(Rounding::same(8.0))
                .inner_margin(Margin::same(10.0))
                .show(ui, |ui| {
                    ui.label(RichText::new(turn.content.as_str()).color(TEXT));
                });
        }
        "system" => {
            ui.label(RichText::new("System").italics().color(MUTED));
            ui.label(RichText::new(turn.content.as_str()).color(MUTED));
        }
        _ => {
            render_assistant_card(ui, cache, turn.content.as_str(), idx);
        }
    }
    ui.add_space(6.0);
}

/// Result of pulling a `<think>…</think>` reasoning span out of assistant text.
struct ThinkSplit {
    thinking: String,
    answer: String,
    has_thinking: bool,
    thinking_done: bool,
}

/// Render an assistant message as the Phase-2 markdown card, collapsing any
/// `<think>…</think>` (or `<thinking>`) reasoning span into a "Reasoning"
/// section (collapsed by default) and rendering the remainder as markdown.
/// `id_salt` distinguishes each card's collapse state.
fn render_assistant_card<H: std::hash::Hash>(
    ui: &mut egui::Ui,
    cache: &mut CommonMarkCache,
    content: &str,
    id_salt: H,
) {
    ui.label(RichText::new("Assistant").strong().color(ACCENT));
    egui::Frame::none()
        .fill(ELEVATED)
        .rounding(Rounding::same(8.0))
        .inner_margin(Margin::same(10.0))
        .show(ui, |ui| {
            let split = split_thinking(content);
            if split.has_thinking {
                // Collapsed by default — reasoning traces are long + noisy, so
                // the user opts in (mirrors Chat.tsx's ThinkingBlock). Before
                // the closing tag arrives mid-stream the header reads
                // "Thinking…"; once closed it becomes "Reasoning".
                let title = if split.thinking_done {
                    "Reasoning"
                } else {
                    "Thinking…"
                };
                egui::CollapsingHeader::new(RichText::new(title).small().color(MUTED))
                    .id_salt(("chat_think", id_salt))
                    .default_open(false)
                    .show(ui, |ui| {
                        let t = split.thinking.trim();
                        if t.is_empty() {
                            ui.label(RichText::new("(thinking…)").color(MUTED).small());
                        } else {
                            // Partial reasoning is rarely valid markdown — show
                            // it as plain wrapped text.
                            ui.label(RichText::new(t).color(MUTED));
                        }
                    });
            }
            // The answer (think stripped), or the whole content when there was
            // no think span. Empty while the model is still inside <think>.
            let body = if split.has_thinking {
                split.answer.as_str()
            } else {
                content
            };
            if body.is_empty() {
                if !split.has_thinking {
                    ui.label(RichText::new("…").color(MUTED));
                }
            } else {
                CommonMarkViewer::new().show(ui, cache, body);
            }
        });
}

/// Split a `<think>…</think>` / `<thinking>…</thinking>` reasoning span out of
/// assistant text. Port of Chat.tsx `splitThinking`, including the
/// still-streaming case (opening tag seen, closing tag not yet): everything
/// after `<think>` is the partial reasoning and the answer is empty.
fn split_thinking(content: &str) -> ThinkSplit {
    let Some((oi, olen)) = find_tag_ci(content, &["<think>", "<thinking>"]) else {
        return ThinkSplit {
            thinking: String::new(),
            answer: content.to_string(),
            has_thinking: false,
            thinking_done: true,
        };
    };
    let rest = &content[oi + olen..];
    let Some((ci, clen)) = find_tag_ci(rest, &["</think>", "</thinking>"]) else {
        return ThinkSplit {
            thinking: rest.to_string(),
            answer: String::new(),
            has_thinking: true,
            thinking_done: false,
        };
    };
    let before = &content[..oi];
    let after = &rest[ci + clen..];
    ThinkSplit {
        thinking: rest[..ci].to_string(),
        answer: format!("{before}{after}").trim().to_string(),
        has_thinking: true,
        thinking_done: true,
    }
}

/// Earliest case-insensitive match of any tag in `tags` (all pre-lowercased
/// ASCII). Returns `(byte offset into hay, matched length)`. ASCII-lowercasing
/// the haystack preserves byte offsets (only A–Z change, never a multi-byte
/// UTF-8 lead byte), so the index maps straight back onto `hay`.
fn find_tag_ci(hay: &str, tags: &[&str]) -> Option<(usize, usize)> {
    let lower = hay.to_ascii_lowercase();
    let mut best: Option<(usize, usize)> = None;
    for t in tags {
        if let Some(idx) = lower.find(t) {
            if best.map_or(true, |(b, _)| idx < b) {
                best = Some((idx, t.len()));
            }
        }
    }
    best
}

/// One-line description of proposed tool calls (the tool-confirm prompt +
/// the auto-run transcript note). Mirrors Chat.tsx `describeToolCalls`.
fn describe_tool_calls(calls: &[ToolCall]) -> String {
    let list = calls
        .iter()
        .map(|c| format!("{}({})", c.name, c.arguments))
        .collect::<Vec<_>>()
        .join(", ");
    format!("Proposed tool call: {list}")
}

/// A timestamped export filename (`rustllama-chat-<unix>.md`). Uses unix
/// seconds rather than a formatted date to avoid a chrono dependency — enough
/// to keep successive exports distinct.
fn export_filename(ext: &str) -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("rustllama-chat-{secs}.{ext}")
}

/// Format the usage block into a one-liner: prompt/completion counts + a
/// tok/s derived from the decode-time rustllama extension.
fn format_usage(u: &Usage) -> String {
    let tok_per_s = match u.decode_ms {
        Some(ms) if ms > 0.0 => (u.completion_tokens as f64) / (ms / 1000.0),
        _ => 0.0,
    };
    format!(
        "{} prompt · {} completion · {:.1} tok/s",
        u.prompt_tokens, u.completion_tokens, tok_per_s
    )
}

/// A rounded elevated "card" container; runs `add` inside its padded body.
fn card<R>(ui: &mut egui::Ui, add: impl FnOnce(&mut egui::Ui) -> R) -> R {
    egui::Frame::none()
        .fill(ELEVATED)
        .rounding(Rounding::same(10.0))
        .stroke(Stroke::new(1.0, BORDER))
        .inner_margin(Margin::same(14.0))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add(ui)
        })
        .inner
}

/// A small colored pill (status badge), e.g. the "default" marker.
fn pill(ui: &mut egui::Ui, text: &str, color: Color32) {
    egui::Frame::none()
        .fill(color.gamma_multiply(0.18))
        .rounding(Rounding::same(6.0))
        .inner_margin(Margin::symmetric(7.0, 2.0))
        .show(ui, |ui| {
            ui.label(RichText::new(text).color(color).small());
        });
}

/// A neutral metadata badge (family / quant / size on cached rows).
fn badge(ui: &mut egui::Ui, text: &str) {
    egui::Frame::none()
        .fill(ELEVATED2)
        .rounding(Rounding::same(6.0))
        .inner_margin(Margin::symmetric(7.0, 2.0))
        .show(ui, |ui| {
            ui.label(RichText::new(text).color(MUTED).small());
        });
}

/// A red-tinted error banner for the Models view header.
fn error_banner(ui: &mut egui::Ui, text: &str) {
    egui::Frame::none()
        .fill(DANGER.gamma_multiply(0.12))
        .rounding(Rounding::same(8.0))
        .stroke(Stroke::new(1.0, DANGER))
        .inner_margin(Margin::same(10.0))
        .show(ui, |ui| {
            ui.label(RichText::new(text).color(DANGER).small());
        });
}

/// One cached-model row: name + badges on the left, Load/Delete on the right.
/// Returns nothing — button clicks push [`Action`]s.
fn cached_row(ui: &mut egui::Ui, m: &TagsModel, is_busy: bool, actions: &mut Vec<Action>) {
    egui::Frame::none()
        .fill(ELEVATED2)
        .rounding(Rounding::same(10.0))
        .stroke(Stroke::new(1.0, BORDER))
        .inner_margin(Margin::symmetric(12.0, 10.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.label(RichText::new(display_name(&m.name)).strong().color(TEXT));
                    ui.add_space(4.0);
                    ui.horizontal_wrapped(|ui| {
                        if let Some(d) = &m.details {
                            if let Some(f) = &d.family {
                                if !f.is_empty() {
                                    badge(ui, f);
                                }
                            }
                            if let Some(q) = &d.quantization_level {
                                if !q.is_empty() {
                                    badge(ui, q);
                                }
                            }
                        }
                        if m.size > 0 {
                            badge(ui, &human_bytes(m.size));
                        }
                    });
                    if is_busy {
                        ui.add_space(4.0);
                        ui.label(
                            RichText::new(
                                "Loading… large GGUFs (10+ GB) can take 30–90 s.",
                            )
                            .color(MUTED)
                            .italics()
                            .small(),
                        );
                    }
                });
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui
                        .add_enabled(
                            !is_busy,
                            egui::Button::new(RichText::new("Delete").color(DANGER)),
                        )
                        .clicked()
                    {
                        actions.push(Action::AskDelete(m.name.clone()));
                    }
                    if ui
                        .add_enabled(
                            !is_busy,
                            egui::Button::new(if is_busy { "Loading…" } else { "Load" }),
                        )
                        .clicked()
                    {
                        actions.push(Action::LoadCached(m.name.clone()));
                    }
                    if is_busy {
                        ui.spinner();
                    }
                });
            });
        });
}

/// One HF search result row (repo id + downloads/likes). Returns `true` when
/// clicked (to expand into its files).
fn hf_result_row(ui: &mut egui::Ui, m: &HfModel) -> bool {
    let resp = ui.add(
        egui::Label::new(RichText::new(&m.id).color(TEXT).small())
            .sense(Sense::click()),
    );
    ui.label(
        RichText::new(format!("↓ {} · ♥ {}", m.downloads, m.likes))
            .color(MUTED)
            .small(),
    );
    ui.separator();
    resp.clicked()
}

/// One HF file row (filename + size). Returns `true` when clicked.
fn hf_file_row(ui: &mut egui::Ui, f: &HfFile) -> bool {
    let resp = ui.add(
        egui::Label::new(RichText::new(&f.rfilename).color(TEXT).small())
            .sense(Sense::click()),
    );
    ui.label(RichText::new(human_bytes(f.size)).color(MUTED).small());
    resp.clicked()
}

/// The bottom status bar: health dot, CPU/RAM/GPU meters, tok/s, active model.
fn render_status_bar(
    ui: &mut egui::Ui,
    health: Option<bool>,
    m: Option<&MetricsSnapshot>,
    loaded: Option<&str>,
) {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 6.0;

        // Health dot + label.
        let (dot, label) = match health {
            Some(true) => (HEALTH_OK, "Ready"),
            Some(false) => (HEALTH_BAD, "Offline"),
            None => (MUTED, "Checking…"),
        };
        let (r, _) = ui.allocate_exact_size(egui::vec2(10.0, 10.0), Sense::hover());
        ui.painter().circle_filled(r.center(), 4.0, dot);
        ui.label(RichText::new(label).color(TEXT).small());

        if let Some(m) = m {
            // CPU utilization.
            if let Some(cpu) = m.cpu_utilization_pct {
                ui.add_space(12.0);
                status_meter(ui, "CPU", (cpu / 100.0) as f32, &format!("{cpu:.0}%"));
            }
            // RAM used / total.
            if m.ram_total_bytes > 0 {
                let used = m.ram_total_bytes.saturating_sub(m.ram_available_bytes);
                let frac = used as f32 / m.ram_total_bytes as f32;
                ui.add_space(12.0);
                status_meter(
                    ui,
                    "RAM",
                    frac,
                    &format!("{} / {}", fmt_gib(used), fmt_gib(m.ram_total_bytes)),
                );
            }
            // Per-GPU util + VRAM.
            if let Some(gpus) = &m.gpus {
                for g in gpus {
                    ui.add_space(12.0);
                    let util = g.utilization_pct.unwrap_or(0.0);
                    let util_txt = g
                        .utilization_pct
                        .map(|u| format!("{u:.0}%"))
                        .unwrap_or_else(|| "n/a".into());
                    status_meter(ui, &format!("GPU{}", g.index), (util / 100.0) as f32, &util_txt);
                    if let (Some(t), Some(f)) = (g.vram_total_bytes, g.vram_free_bytes) {
                        if t > 0 {
                            let used = t.saturating_sub(f);
                            status_meter(
                                ui,
                                "VRAM",
                                used as f32 / t as f32,
                                &format!("{} / {}", fmt_gib(used), fmt_gib(t)),
                            );
                        }
                    }
                }
            }
            // tok/s (EMA preferred).
            ui.add_space(12.0);
            let toks = m.ema_tok_s.or(m.last_tok_s).unwrap_or(0.0);
            ui.label(RichText::new("tok/s").color(MUTED).small());
            ui.label(RichText::new(format!("{toks:.1}")).color(TEXT).small());
        }

        // Active model id, right-aligned.
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            let model = m
                .map(|x| x.model_id.as_str())
                .filter(|s| !s.is_empty())
                .or(loaded)
                .unwrap_or("No model loaded");
            ui.label(RichText::new(model).color(MUTED).small());
        });
    });
}

/// A status-bar meter: `label [====   ] value`. `frac` in 0..1.
fn status_meter(ui: &mut egui::Ui, label: &str, frac: f32, value: &str) {
    ui.label(RichText::new(label).color(MUTED).small());
    thin_meter(ui, frac, 54.0);
    ui.label(RichText::new(value).color(TEXT).small());
}

/// A thin rounded meter bar. Color bands: >0.9 red, >0.75 yellow, else accent.
fn thin_meter(ui: &mut egui::Ui, frac: f32, width: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, 7.0), Sense::hover());
    let rounding = Rounding::same(3.5);
    ui.painter().rect_filled(rect, rounding, FIELD_BG);
    let f = frac.clamp(0.0, 1.0);
    if f > 0.0 {
        let fill = Rect::from_min_size(rect.min, egui::vec2(rect.width() * f, rect.height()));
        ui.painter().rect_filled(fill, rounding, meter_color(f));
    }
}

fn meter_color(frac: f32) -> Color32 {
    if frac > 0.9 {
        HEALTH_BAD
    } else if frac > 0.75 {
        WARN
    } else {
        ACCENT
    }
}

/// A full-width clickable nav row with a small vector glyph + label. Active
/// items get an accent side-bar + highlight; disabled ones are dimmed
/// placeholders for later-phase views. Returns `true` when clicked.
fn nav_item(ui: &mut egui::Ui, icon: NavIcon, label: &str, active: bool, enabled: bool) -> bool {
    let sense = if enabled {
        Sense::click()
    } else {
        Sense::hover()
    };
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 38.0), sense);
    let hovered = enabled && resp.hovered();
    let bg = if active {
        SELECT_BG
    } else if hovered {
        ELEVATED
    } else {
        Color32::TRANSPARENT
    };
    if bg != Color32::TRANSPARENT {
        ui.painter().rect_filled(rect, Rounding::same(8.0), bg);
    }
    if active {
        // Left accent bar.
        let bar = Rect::from_min_size(
            rect.left_top() + egui::vec2(0.0, 7.0),
            egui::vec2(3.0, rect.height() - 14.0),
        );
        ui.painter().rect_filled(bar, Rounding::same(2.0), ACCENT);
    }
    let color = if !enabled {
        DISABLED
    } else if active {
        TEXT
    } else {
        MUTED
    };
    let icon_rect =
        Rect::from_center_size(egui::pos2(rect.left() + 24.0, rect.center().y), egui::vec2(18.0, 18.0));
    draw_nav_icon(ui.painter(), icon_rect, icon, color);
    ui.painter().text(
        egui::pos2(rect.left() + 42.0, rect.center().y),
        Align2::LEFT_CENTER,
        label,
        FontId::proportional(13.5),
        color,
    );
    enabled && resp.clicked()
}

/// Draw a minimal line-icon into `rect` for a nav item.
fn draw_nav_icon(painter: &egui::Painter, rect: Rect, icon: NavIcon, color: Color32) {
    let s = Stroke::new(1.7, color);
    match icon {
        NavIcon::Chat => {
            // Speech bubble: rounded rect + a small tail.
            let body = Rect::from_min_max(
                rect.min,
                egui::pos2(rect.max.x, rect.max.y - 3.0),
            );
            painter.rect_stroke(body, Rounding::same(4.0), s);
            painter.line_segment(
                [
                    egui::pos2(body.min.x + 4.0, body.max.y),
                    egui::pos2(body.min.x + 1.5, rect.max.y),
                ],
                s,
            );
        }
        NavIcon::Models => {
            // A cube / stacked-layers hexagon.
            let c = rect.center();
            let w = rect.width() * 0.46;
            let h = rect.height() * 0.5;
            let pts = vec![
                egui::pos2(c.x, c.y - h),
                egui::pos2(c.x + w, c.y - h * 0.5),
                egui::pos2(c.x + w, c.y + h * 0.5),
                egui::pos2(c.x, c.y + h),
                egui::pos2(c.x - w, c.y + h * 0.5),
                egui::pos2(c.x - w, c.y - h * 0.5),
            ];
            painter.add(egui::Shape::closed_line(pts, s));
            painter.line_segment([egui::pos2(c.x - w, c.y - h * 0.5), c], s);
            painter.line_segment([egui::pos2(c.x + w, c.y - h * 0.5), c], s);
            painter.line_segment([c, egui::pos2(c.x, c.y + h)], s);
        }
        NavIcon::Placeholder => {
            painter.rect_stroke(rect.shrink(2.0), Rounding::same(3.0), s);
        }
    }
}

/// Human-readable byte size (mirrors the web UI's `humanBytes`).
fn human_bytes(n: u64) -> String {
    if n == 0 {
        return "—".into();
    }
    const U: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = n as f64;
    let mut i = 0;
    while v >= 1024.0 && i < U.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if v < 10.0 {
        format!("{v:.2} {}", U[i])
    } else {
        format!("{v:.1} {}", U[i])
    }
}

/// GiB with one decimal (status-bar RAM/VRAM meters).
fn fmt_gib(b: u64) -> String {
    format!("{:.1} GB", b as f64 / 1_073_741_824.0)
}

/// Strip a trailing `.gguf` (case-insensitive) for a cleaner display name.
fn display_name(name: &str) -> &str {
    name.strip_suffix(".gguf")
        .or_else(|| name.strip_suffix(".GGUF"))
        .unwrap_or(name)
}

/// Install the dark visuals once at startup. Called from [`GuiApp::new`].
fn configure_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    let mut v = egui::Visuals::dark();

    v.override_text_color = Some(TEXT);
    v.panel_fill = BG;
    v.window_fill = ELEVATED;
    v.extreme_bg_color = FIELD_BG; // TextEdit / scroll troughs
    v.faint_bg_color = ELEVATED;
    v.hyperlink_color = ACCENT;
    v.window_stroke = egui::Stroke::new(1.0, BORDER);
    v.selection.bg_fill = SELECT_BG;
    v.selection.stroke = egui::Stroke::new(1.0, ACCENT);

    let round = Rounding::same(8.0);

    v.widgets.noninteractive.bg_fill = ELEVATED;
    v.widgets.noninteractive.weak_bg_fill = ELEVATED;
    v.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, BORDER);
    v.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, MUTED);
    v.widgets.noninteractive.rounding = round;

    v.widgets.inactive.bg_fill = ELEVATED;
    v.widgets.inactive.weak_bg_fill = ELEVATED;
    v.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT);
    v.widgets.inactive.rounding = round;

    v.widgets.hovered.bg_fill = ELEVATED2;
    v.widgets.hovered.weak_bg_fill = ELEVATED2;
    v.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, ACCENT);
    v.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, TEXT);
    v.widgets.hovered.rounding = round;

    v.widgets.active.bg_fill = ACCENT;
    v.widgets.active.weak_bg_fill = ACCENT;
    v.widgets.active.fg_stroke = egui::Stroke::new(1.0, TEXT);
    v.widgets.active.rounding = round;

    v.widgets.open.bg_fill = ELEVATED2;
    v.widgets.open.rounding = round;

    v.window_rounding = round;
    v.menu_rounding = round;

    style.visuals = v;
    style.spacing.item_spacing = egui::vec2(8.0, 8.0);
    style.spacing.button_padding = egui::vec2(10.0, 6.0);

    ctx.set_style(style);
}
