//! The Phase-2 app: [`GuiApp`], its async bridge, and the egui layout.
//!
//! Phase 2 adds a left icon-nav sidebar, a Models management view (loaded +
//! cached lists, load / unload / set-default / delete, HuggingFace search +
//! streamed pull), and an always-on bottom status bar (RAM / per-GPU VRAM /
//! tok-s / active model, polled from `/v1/metrics`). The Phase-1 model bar +
//! transcript + composer are now the **Chat** view.

use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

use egui::{Align, Align2, Color32, FontId, Layout, Margin, Rect, RichText, Rounding, Sense, Stroke};
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use futures::StreamExt;
use rustllama_client::{
    Capabilities, ChatEvent, ChatMessage, ChatRequest, Client, ConversationSummary, DecideResult,
    HfFile, HfModel, LoadModelParams, MetricsSnapshot, MultimodalMessage, StoredMessage,
    StreamOptions, SystemPrompt, TagsModel, ToolCall, TuneProgress, TuningRecommendations,
    TuningSummary, Usage,
};

/// How many `/v1/metrics` samples the Status sparklines retain. At the ~1 Hz
/// Status-view poll cadence that's two minutes of rolling history — enough to
/// see a generation's shape without the ring buffer growing unbounded.
const METRICS_WINDOW: usize = 120;

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

/// Which page the central area renders. All six nav items are live as of
/// Phase 5 (Decide + Quantize completed the parity pass).
#[derive(Clone, Copy, PartialEq, Eq)]
enum View {
    Chat,
    Models,
    Decide,
    Status,
    Quantize,
    Settings,
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
    Decide,
    Status,
    Quantize,
    Settings,
    #[allow(dead_code)]
    Placeholder,
}

/// One committed message in the transcript. `role` is the OpenAI role string
/// (`"user"` / `"assistant"` / `"system"`) so it maps 1:1 onto
/// [`MultimodalMessage`]. `images` carries any attached images as base64
/// `data:` URIs (user turns only); empty is the common text-only path.
struct Turn {
    role: String,
    content: String,
    images: Vec<String>,
}

impl Turn {
    fn user(content: String) -> Self {
        Self {
            role: "user".into(),
            content,
            images: Vec::new(),
        }
    }
    /// A user turn with attached images (base64 `data:` URIs).
    fn user_with_images(content: String, images: Vec<String>) -> Self {
        Self {
            role: "user".into(),
            content,
            images,
        }
    }
    fn assistant(content: String) -> Self {
        Self {
            role: "assistant".into(),
            content,
            images: Vec::new(),
        }
    }
    fn system(content: String) -> Self {
        Self {
            role: "system".into(),
            content,
            images: Vec::new(),
        }
    }
}

/// Which Decide sub-tab is active (mirrors Decide.tsx's `Mode`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum DecideTab {
    Choice,
    Score,
    Boolean,
}

/// One image staged in the chat composer before send: its display name (for
/// the chip) + the base64 `data:` URI actually sent on the wire.
struct PendingImage {
    name: String,
    uri: String,
    bytes: u64,
}

/// The result summary of an in-process quantize run — the fields the Quantize
/// view renders (mirrors Quantize.tsx's `QuantizeResult`). Populated from
/// `rustllama_gguf::quantize::QuantizeStats` on the worker thread.
#[derive(Clone)]
struct QuantizeSummary {
    tensors_total: usize,
    tensors_requantized: usize,
    tensors_passthrough: usize,
    bytes_in: u64,
    bytes_out: u64,
    elapsed_ms: f64,
    target: String,
    n_layers: usize,
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

/// The editable subset of the on-disk config the Settings form exposes. Loaded
/// from `/v1/config` (see [`draft_from_config`]) and written back over the
/// original config JSON on Save (see [`apply_draft_to_config`]) so unmodeled
/// sections — placement overrides, `[tuning]`, `[[profiles]]`,
/// `[[system_prompts]]`, the speculative-draft knobs — round-trip untouched.
/// `PartialEq` drives the "dirty" gate (a form equal to the freshly-loaded
/// snapshot disables Save / permits Apply-profile), mirroring Settings.tsx.
#[derive(Clone, PartialEq)]
struct SettingsDraft {
    // --- server (restart) ---
    bind_addr: String,
    /// Free text; parsed to u16 on save so a mid-edit "" doesn't clamp to 0.
    port: String,
    api_key: String,
    max_loaded_models: u32,
    /// Comma-separated origins; split on save (mirrors the React text field).
    cors_origins: String,
    // --- model (reload) ---
    model_path: String,
    model_hub: String,
    chat_template: String,
    // --- inference (reload) ---
    ctx_size: u64,
    batch_size: u32,
    threads: u32,
    kv_dtype: String,
    /// "" = inherit `kv_dtype` (written back as JSON null).
    k_dtype: String,
    v_dtype: String,
    flash_attention: bool,
    speculative_ngram: bool,
    prefix_cache: bool,
    keep_quant_raw: bool,
    // --- ui (live) ---
    theme: String,
    font_size: u32,
    code_theme: String,
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

    // --- Status / Settings views (Phase 4) ---
    /// `/v1/capabilities` — server version + compute-backend snapshot (boxed;
    /// it's a large struct). Feeds the Status Backends panel + Settings
    /// Hardware panel.
    Caps(Box<Capabilities>),
    /// `/v1/tuning_summary` + `/v1/tuning/recommendations` for the Status
    /// Tuner-cache panel. `Ok((summary, recs))` on success (summary boxed — the
    /// larger of the two); `Err(msg)` when the summary fetch failed.
    Tuning(std::result::Result<(Box<TuningSummary>, TuningRecommendations), String>),
    /// A re-tune finished (`Ok` clears the busy state + refreshes the summary;
    /// `Err` surfaces the message).
    TuneDone(std::result::Result<(), String>),
    /// One `/v1/tune/progress` poll frame (stage bar + last log line).
    TuneProg(Box<TuneProgress>),
    /// `/v1/config` loaded for the Settings form: the whole config object (for
    /// round-trip preservation), its on-disk path, and the profile names.
    SettingsLoaded {
        config: Box<serde_json::Value>,
        config_path: String,
        profiles: Vec<String>,
    },
    /// The Settings config fetch failed.
    SettingsError(String),
    /// A Save (PUT) / Apply-profile succeeded — carries the server's
    /// reload/restart flags for the result banner. Triggers a config re-fetch.
    SettingsSaved { reload: bool, restart: bool },
    /// A Save / Apply-profile failed.
    SettingsSaveError(String),

    // --- Decide view (Phase 5) ---
    /// A `/v1/decide/*` call finished (`Ok` result, or `Err(message)`).
    DecideDone(std::result::Result<Box<DecideResult>, String>),

    // --- Quantize view (Phase 5) ---
    /// The in-process quantize worker finished (`Ok` stats, or `Err(message)`).
    QuantizeDone(std::result::Result<Box<QuantizeSummary>, String>),
    /// The native picker returned the source `.gguf` path to quantize.
    PickedQuantInput(std::path::PathBuf),
    /// The native picker returned the output `.gguf` path to write.
    PickedQuantOutput(std::path::PathBuf),

    // --- Chat: image attach + conversation history (Phase 5) ---
    /// The native image picker returned a chosen image (already base64
    /// `data:`-encoded on the picker thread — decoding a large image off the
    /// UI thread keeps the event loop responsive).
    PickedImage {
        name: String,
        uri: String,
        bytes: u64,
    },
    /// The history-feature probe result (`--features history` present?).
    HistoryAvailable(bool),
    /// The saved-conversation list (list / post-mutation refresh).
    Conversations(Vec<ConversationSummary>),
    /// A conversation's messages loaded for the transcript (select-to-load).
    ConversationLoaded { id: i64, messages: Vec<StoredMessage> },
    /// A save (create-if-needed + append) persisted `appended` new messages
    /// to conversation `id`.
    HistorySaved { id: i64, appended: usize },
    /// A history operation failed (kept non-fatal — chat never breaks on it).
    HistoryError(String),
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
    // --- Status / Settings (Phase 4) ---
    /// Refresh the Status dashboard's tuning summary + recommendations.
    RefreshTuning,
    /// Start a re-tune of the named cached model (full "all" sweep).
    TuneModel(String),
    /// Reload the Settings config from disk (Refresh / post-save).
    ReloadSettings,
    /// PUT the edited config.
    SaveSettings,
    /// Reset the Settings form to the last-loaded config (Discard).
    ResetSettings,
    /// Apply a named config profile (sparse override merge).
    ApplyProfile(String),
    // --- Decide / Quantize (Phase 5) ---
    /// Run the current Decide tab (choice / score / boolean).
    RunDecide,
    /// Open the source-GGUF picker for the Quantize view.
    PickQuantInput,
    /// Open the output-GGUF save picker for the Quantize view.
    PickQuantOutput,
    /// Start the in-process quantize with the current form values.
    RunQuantize,
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

    // --- Status / Settings shared: compute-backend snapshot ---
    /// `/v1/capabilities`, fetched once at startup (backends don't change
    /// mid-process — adding a GPU needs a server restart). Feeds the Status
    /// Backends panel + the Settings Hardware panel.
    capabilities: Option<Capabilities>,

    // --- Status view ---
    /// Rolling ring buffers for the sparklines, appended on each metrics poll
    /// (see [`GuiApp::push_metric_samples`]) and capped at `METRICS_WINDOW`.
    tok_series: VecDeque<f64>,
    ctx_series: VecDeque<f64>,
    pending_series: VecDeque<f64>,
    tuning_summary: Option<TuningSummary>,
    tuning_recs: Option<TuningRecommendations>,
    tuning_error: Option<String>,
    /// When the tuner summary was last polled (slow ~10 s cadence).
    tuning_last_poll: Option<Instant>,
    /// The cached model selected in the Re-tune picker.
    tune_sel_model: String,
    /// True while a re-tune subprocess runs (disables the buttons).
    tune_running: bool,
    /// Latest `/v1/tune/progress` frame (stage bar + last log line).
    tune_progress: Option<TuneProgress>,
    /// When tune progress was last polled (~1 s cadence while a re-tune runs).
    tune_prog_last_poll: Option<Instant>,
    /// Result line of the most recent re-tune ("done" / "error: …").
    tune_status: Option<String>,

    // --- Settings view ---
    /// The full config OBJECT last loaded from `/v1/config`, kept whole so a
    /// Save writes only the edited keys and preserves everything else.
    settings_config: Option<serde_json::Value>,
    settings_config_path: String,
    /// The editable form state, and a snapshot of it at load time for the
    /// dirty check.
    settings_draft: Option<SettingsDraft>,
    settings_orig: Option<SettingsDraft>,
    /// `[[profiles]]` names for the Apply dropdown.
    settings_profiles: Vec<String>,
    settings_profile_sel: String,
    settings_error: Option<String>,
    /// True while a config fetch is in flight (drives the "loading…" state).
    settings_loading: bool,
    /// True while a Save / Apply-profile PUT is in flight.
    settings_saving: bool,
    /// The Save/Apply result banner: (is_error, message).
    settings_status: Option<(bool, String)>,

    // --- Decide view (Phase 5) ---
    decide_tab: DecideTab,
    /// The context / prompt scored against the model.
    decide_context: String,
    /// Choice: the candidate option strings (dynamic add/remove list).
    decide_options: Vec<String>,
    /// Score: the ordered scale level strings (dynamic add/remove list).
    decide_levels: Vec<String>,
    /// Boolean: the yes/no question.
    decide_question: String,
    decide_busy: bool,
    /// The last result + the labels it answered (so the bars carry the right
    /// captions even if the option fields are edited afterwards).
    decide_result: Option<DecideResult>,
    decide_result_labels: Vec<String>,
    decide_result_tab: DecideTab,
    decide_error: Option<String>,

    // --- Quantize view (Phase 5) ---
    q_input: String,
    q_output: String,
    q_target: String,
    q_apex: String,
    q_recipe: String,
    q_keep_output: bool,
    q_running: bool,
    q_result: Option<QuantizeSummary>,
    q_error: Option<String>,

    // --- Chat: image attach (Phase 5) ---
    /// Images staged in the composer, sent with the next user turn.
    pending_images: Vec<PendingImage>,

    // --- Chat: conversation history (Phase 5) ---
    /// `None` while the feature probe is in flight; `Some(false)` hides the
    /// sidebar entirely (server built without `--features history`).
    history_enabled: Option<bool>,
    /// Sidebar expand/collapse (unobtrusive — a toolbar toggle flips it).
    history_open: bool,
    /// The saved-conversation list rendered in the sidebar.
    conversations: Vec<ConversationSummary>,
    /// The conversation the live transcript is bound to (for incremental
    /// save), or `None` for an unsaved fresh chat.
    current_conv: Option<i64>,
    /// How many transcript messages have already been persisted to
    /// `current_conv` — the save appends only `transcript[saved_msgs..]`.
    saved_msgs: usize,
    /// True while a create/append save task runs (dedupes concurrent saves).
    history_saving: bool,

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
            capabilities: None,
            tok_series: VecDeque::new(),
            ctx_series: VecDeque::new(),
            pending_series: VecDeque::new(),
            tuning_summary: None,
            tuning_recs: None,
            tuning_error: None,
            tuning_last_poll: None,
            tune_sel_model: String::new(),
            tune_running: false,
            tune_progress: None,
            tune_prog_last_poll: None,
            tune_status: None,
            settings_config: None,
            settings_config_path: String::new(),
            settings_draft: None,
            settings_orig: None,
            settings_profiles: Vec::new(),
            settings_profile_sel: String::new(),
            settings_error: None,
            settings_loading: false,
            settings_saving: false,
            settings_status: None,
            // Decide view: seed with Decide.tsx's sample sentiment task so the
            // page is immediately runnable.
            decide_tab: DecideTab::Choice,
            decide_context:
                "Classify the sentiment.\nReview: \"I absolutely love this, best purchase ever!\"\nSentiment:"
                    .to_string(),
            decide_options: vec!["positive".into(), "negative".into(), "neutral".into()],
            decide_levels: vec!["1".into(), "2".into(), "3".into(), "4".into(), "5".into()],
            decide_question: "Is this review positive?".to_string(),
            decide_busy: false,
            decide_result: None,
            decide_result_labels: Vec::new(),
            decide_result_tab: DecideTab::Choice,
            decide_error: None,
            // Quantize view.
            q_input: String::new(),
            q_output: String::new(),
            q_target: "q4_k".to_string(),
            q_apex: String::new(),
            q_recipe: String::new(),
            q_keep_output: true,
            q_running: false,
            q_result: None,
            q_error: None,
            // Chat image attach + history.
            pending_images: Vec::new(),
            history_enabled: None,
            history_open: true,
            conversations: Vec::new(),
            current_conv: None,
            saved_msgs: 0,
            history_saving: false,
            md_cache: CommonMarkCache::default(),
        };

        // Kick off the initial loads (health probe + model list + cached list +
        // the config's system-prompt library / ctx budget + the compute-backend
        // capabilities snapshot for Status/Settings).
        app.spawn_health();
        app.spawn_refresh_models();
        app.spawn_refresh_cached();
        app.spawn_config();
        app.spawn_capabilities();
        // Probe for the optional conversation-history feature; if present, the
        // sidebar appears and the initial list loads (see the probe helper).
        app.spawn_history_probe();
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

    /// Fetch `/v1/capabilities` once (backends are process-stable). Feeds the
    /// Status Backends panel + Settings Hardware panel; a failure is silent
    /// (those panels just show "unavailable").
    fn spawn_capabilities(&self) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            if let Ok(c) = client.capabilities().await {
                let _ = tx.send(UiMsg::Caps(Box::new(c)));
                ctx.request_repaint();
            }
        });
    }

    /// Poll the tuner-cache summary + untuned-shape recommendations for the
    /// Status Tuner panel. Recommendations are best-effort (older servers lack
    /// the endpoint), so a failure there degrades to an empty list.
    fn spawn_tuning(&self) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            let msg = match client.tuning_summary().await {
                Ok(summary) => {
                    let recs = client.tuning_recommendations().await.unwrap_or_default();
                    UiMsg::Tuning(Ok((Box::new(summary), recs)))
                }
                Err(e) => UiMsg::Tuning(Err(e.to_string())),
            };
            let _ = tx.send(msg);
            ctx.request_repaint();
        });
    }

    /// One `/v1/tune/progress` poll — the stage bar + last log line shown while
    /// a re-tune runs.
    fn spawn_tune_progress(&self) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            if let Ok(p) = client.tune_progress().await {
                let _ = tx.send(UiMsg::TuneProg(Box::new(p)));
                ctx.request_repaint();
            }
        });
    }

    /// Force a full ("all") re-tune of `model`. The POST blocks for the whole
    /// sweep; the Status view polls [`Self::spawn_tune_progress`] meanwhile.
    fn spawn_tune_model(&self, model: String) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            let res = client
                .tune_model(&model, true, "all")
                .await
                .map_err(|e| e.to_string());
            let _ = tx.send(UiMsg::TuneDone(res));
            ctx.request_repaint();
        });
    }

    /// Load the full config for the Settings form (raw JSON so unmodeled
    /// sections round-trip on Save). Extracts the profile names for the Apply
    /// dropdown.
    fn spawn_settings_config(&self) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            match client.get_config_raw().await {
                Ok(env) => {
                    let config = env.get("config").cloned().unwrap_or(serde_json::Value::Null);
                    let config_path = env
                        .get("config_path")
                        .and_then(|p| p.as_str())
                        .unwrap_or("config.toml")
                        .to_string();
                    let profiles = config
                        .get("profiles")
                        .and_then(|p| p.as_array())
                        .map(|arr| {
                            arr.iter()
                                .filter_map(|p| {
                                    p.get("name").and_then(|n| n.as_str()).map(str::to_string)
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    let _ = tx.send(UiMsg::SettingsLoaded {
                        config: Box::new(config),
                        config_path,
                        profiles,
                    });
                }
                Err(e) => {
                    let _ = tx.send(UiMsg::SettingsError(e.to_string()));
                }
            }
            ctx.request_repaint();
        });
    }

    /// PUT the edited config. `config` is the merged object (draft written over
    /// the loaded config so unmodeled keys survive). The reload/restart flags
    /// in the response drive the result banner.
    fn spawn_save_settings(&self, config: serde_json::Value) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            match client.set_config(&config).await {
                Ok(v) => {
                    let _ = tx.send(UiMsg::SettingsSaved {
                        reload: flag(&v, "requires_model_reload"),
                        restart: flag(&v, "requires_server_restart"),
                    });
                }
                Err(e) => {
                    let _ = tx.send(UiMsg::SettingsSaveError(e.to_string()));
                }
            }
            ctx.request_repaint();
        });
    }

    /// Apply a named config profile (server merges its sparse overrides). Same
    /// reload/restart result shape as a Save.
    fn spawn_apply_profile(&self, name: String) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            match client.apply_profile(&name).await {
                Ok(v) => {
                    let _ = tx.send(UiMsg::SettingsSaved {
                        reload: flag(&v, "requires_model_reload"),
                        restart: flag(&v, "requires_server_restart"),
                    });
                }
                Err(e) => {
                    let _ = tx.send(UiMsg::SettingsSaveError(e.to_string()));
                }
            }
            ctx.request_repaint();
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

    // --- Decide view -------------------------------------------------------

    /// Run the active Decide tab against the loaded model. `labels` is the
    /// caption list carried back so the result bars stay in sync even if the
    /// option fields are edited before the reply lands.
    fn spawn_decide(&self, tab: DecideTab, context: String, labels: Vec<String>) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        let question = self.decide_question.clone();
        self.rt.spawn(async move {
            let res = match tab {
                DecideTab::Choice => client.decide_choice(&context, &labels).await,
                DecideTab::Score => client.decide_score(&context, &labels).await,
                DecideTab::Boolean => client.decide_boolean(&context, &question).await,
            };
            let msg = res.map(Box::new).map_err(|e| e.to_string());
            let _ = tx.send(UiMsg::DecideDone(msg));
            ctx.request_repaint();
        });
    }

    // --- Quantize view -----------------------------------------------------

    /// Run the (CPU-bound, minutes-long) quantize pipeline on a DEDICATED OS
    /// thread — NOT the tokio pool (which the network calls share) or the UI
    /// thread. The `rustllama_gguf::quantize` encode loop is synchronous +
    /// heavy; a plain `std::thread` keeps both the async runtime and the event
    /// loop fully responsive, and the result rides back over the channel. The
    /// pipeline exposes no progress callback, so the view shows a spinner.
    fn spawn_quantize(
        &self,
        input: String,
        output: String,
        target: String,
        apex: Option<String>,
        recipe: Option<String>,
        keep_output: bool,
    ) {
        let (tx, ctx) = (self.tx.clone(), self.ctx.clone());
        std::thread::spawn(move || {
            let res = run_quantize_job(
                &input,
                &output,
                &target,
                apex.as_deref(),
                recipe.as_deref(),
                keep_output,
            )
            .map(Box::new);
            let _ = tx.send(UiMsg::QuantizeDone(res));
            ctx.request_repaint();
        });
    }

    /// Open the source-GGUF picker (blocking rfd → its own OS thread).
    fn spawn_pick_quant_input(&self) {
        let (tx, ctx) = (self.tx.clone(), self.ctx.clone());
        std::thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("GGUF model", &["gguf"])
                .set_title("Source GGUF to quantize")
                .pick_file()
            {
                let _ = tx.send(UiMsg::PickedQuantInput(path));
                ctx.request_repaint();
            }
        });
    }

    /// Open the output-GGUF save picker (blocking rfd → its own OS thread).
    fn spawn_pick_quant_output(&self) {
        let (tx, ctx) = (self.tx.clone(), self.ctx.clone());
        std::thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("GGUF model", &["gguf"])
                .set_title("Output GGUF (created or overwritten)")
                .set_file_name("quantized.gguf")
                .save_file()
            {
                let _ = tx.send(UiMsg::PickedQuantOutput(path));
                ctx.request_repaint();
            }
        });
    }

    /// Open the native image picker (blocking rfd → its own OS thread). Reads
    /// the file + base64-encodes it into a `data:` URI on that thread so a
    /// large image never blocks the UI, then reports it back for the composer.
    fn spawn_pick_image(&self) {
        let (tx, ctx) = (self.tx.clone(), self.ctx.clone());
        std::thread::spawn(move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("Image", &["png", "jpg", "jpeg", "webp", "gif"])
                .set_title("Attach an image")
                .pick_file()
            {
                match std::fs::read(&path) {
                    Ok(bytes) => {
                        let uri = format!(
                            "data:{};base64,{}",
                            mime_from_path(&path),
                            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &bytes)
                        );
                        let name = path
                            .file_name()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_else(|| "image".into());
                        let _ = tx.send(UiMsg::PickedImage {
                            name,
                            uri,
                            bytes: bytes.len() as u64,
                        });
                    }
                    Err(e) => {
                        let _ = tx.send(UiMsg::Err(format!("read image: {e}")));
                    }
                }
                ctx.request_repaint();
            }
        });
    }

    // --- Conversation history ----------------------------------------------

    /// Probe the history feature once at startup; on success also load the
    /// initial conversation list.
    fn spawn_history_probe(&self) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            let ok = client.history_available().await;
            let _ = tx.send(UiMsg::HistoryAvailable(ok));
            if ok {
                if let Ok(list) = client.list_conversations().await {
                    let _ = tx.send(UiMsg::Conversations(list));
                }
            }
            ctx.request_repaint();
        });
    }

    /// Refresh the saved-conversation list.
    fn spawn_list_conversations(&self) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            if let Ok(list) = client.list_conversations().await {
                let _ = tx.send(UiMsg::Conversations(list));
                ctx.request_repaint();
            }
        });
    }

    /// Load a conversation's messages into the transcript.
    fn spawn_get_conversation(&self, id: i64) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            match client.get_conversation(id).await {
                Ok(c) => {
                    let _ = tx.send(UiMsg::ConversationLoaded {
                        id,
                        messages: c.messages,
                    });
                }
                Err(e) => {
                    let _ = tx.send(UiMsg::HistoryError(e.to_string()));
                }
            }
            ctx.request_repaint();
        });
    }

    /// Delete a conversation, then refresh the list.
    fn spawn_delete_conversation(&self, id: i64) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            let _ = client.delete_conversation(id).await;
            if let Ok(list) = client.list_conversations().await {
                let _ = tx.send(UiMsg::Conversations(list));
            }
            ctx.request_repaint();
        });
    }

    /// Persist the not-yet-saved tail of the transcript: create the
    /// conversation on the first save (title = the first user line, trimmed),
    /// then append each pending message in order. Best-effort — a failure just
    /// leaves the messages unsaved (chat itself never breaks).
    fn spawn_save_history(&self, conv: Option<i64>, title: String, msgs: Vec<(String, String)>) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            let id = match conv {
                Some(id) => id,
                None => match client.create_conversation(&title).await {
                    Ok(id) => id,
                    Err(e) => {
                        let _ = tx.send(UiMsg::HistoryError(e.to_string()));
                        ctx.request_repaint();
                        return;
                    }
                },
            };
            let mut appended = 0usize;
            for (role, content) in &msgs {
                if client.append_message(id, role, content).await.is_ok() {
                    appended += 1;
                } else {
                    break;
                }
            }
            let _ = tx.send(UiMsg::HistorySaved { id, appended });
            ctx.request_repaint();
        });
    }

    fn spawn_chat(
        &mut self,
        model: String,
        messages: Vec<MultimodalMessage>,
        sampling: Sampling,
        allow_clarify: bool,
    ) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        let handle = self.rt.spawn(async move {
            let has_images = messages.iter().any(|m| !m.images.is_empty());
            // The typed request carries the plain-text messages; the sampling
            // panel folds in below. For the image path we serialize it and
            // splice a multimodal `messages[]` (see MultimodalMessage::to_json)
            // — the typed ChatRequest can't express per-message image blocks.
            let plain: Vec<ChatMessage> = messages
                .iter()
                .map(|m| ChatMessage {
                    role: m.role.clone(),
                    content: m.content.clone(),
                })
                .collect();
            let mut req = ChatRequest {
                model,
                messages: plain,
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

            // Pick the transport: raw-JSON body (with multimodal content blocks)
            // when any message has images, else the fully-typed path.
            let stream_res = if has_images {
                match serde_json::to_value(&req) {
                    Ok(mut body) => {
                        body["messages"] = serde_json::Value::Array(
                            messages.iter().map(|m| m.to_json()).collect(),
                        );
                        Some(client.chat_stream_value(body).await)
                    }
                    Err(e) => {
                        let _ = tx.send(UiMsg::ChatError(format!("serialize request: {e}")));
                        ctx.request_repaint();
                        None
                    }
                }
            } else {
                Some(client.chat_stream(req).await)
            };
            let Some(stream_res) = stream_res else {
                return;
            };

            // Event-ordering NOTE: with `stream_options.include_usage` the
            // server emits the Usage event in a *separate* final SSE chunk that
            // arrives AFTER Finish. So we do NOT commit on Finish — we
            // accumulate `usage` and send `ChatDone` once the stream fully
            // drains, which guarantees the usage stats ride along with the
            // completion.
            let mut usage: Option<Usage> = None;
            let mut errored = false;
            match stream_res {
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
        // Images ride the user turn(s) they were attached to (see `Turn`).
        let messages: Vec<MultimodalMessage> = self
            .transcript
            .iter()
            .map(|t| MultimodalMessage {
                role: t.role.clone(),
                content: t.content.clone(),
                images: t.images.clone(),
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
        // Allow an image-only turn (no text) as long as something is attached.
        if text.is_empty() && self.pending_images.is_empty() {
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
        // Take the staged images (if any) and attach them to this user turn.
        if self.pending_images.is_empty() {
            self.transcript.push(Turn::user(text));
        } else {
            let imgs: Vec<String> =
                std::mem::take(&mut self.pending_images).into_iter().map(|p| p.uri).collect();
            self.transcript.push(Turn::user_with_images(text, imgs));
        }
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
        self.pending_images.clear();
        self.streaming = false;
        self.last_usage = None;
        self.status = None;
        self.pending_question = None;
        self.pending_tools = None;
        self.request_id = None;
    }

    /// Start a brand-new chat: clear the transcript AND detach the history
    /// binding so the next send begins a fresh conversation (History sidebar +
    /// Ctrl+L). A no-op-safe superset of [`Self::clear_chat`].
    fn new_chat(&mut self) {
        self.clear_chat();
        self.current_conv = None;
        self.saved_msgs = 0;
    }

    /// Select-to-load: fetch a saved conversation and replace the transcript
    /// with it. Aborts any in-flight stream first.
    fn load_conversation(&mut self, id: i64) {
        self.stop_stream();
        self.spawn_get_conversation(id);
    }

    /// After a turn completes, persist the transcript tail if history is on
    /// and something new is pending. Guarded so only one save runs at a time
    /// (a partial failure just retries the remainder on the next completion).
    fn maybe_save_history(&mut self) {
        if self.history_enabled != Some(true) || self.history_saving {
            return;
        }
        if self.transcript.len() <= self.saved_msgs {
            return;
        }
        let pending: Vec<(String, String)> = self.transcript[self.saved_msgs..]
            .iter()
            .map(|t| (t.role.clone(), t.content.clone()))
            .collect();
        if pending.is_empty() {
            return;
        }
        let title = self
            .transcript
            .iter()
            .find(|t| t.role == "user")
            .map(|t| truncate_title(&t.content))
            .unwrap_or_else(|| "New chat".into());
        self.history_saving = true;
        self.spawn_save_history(self.current_conv, title, pending);
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
                // Persist the completed exchange when history is enabled.
                self.maybe_save_history();
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
                // Append the Status sparkline samples before storing (the ring
                // buffers roll at METRICS_WINDOW).
                self.push_metric_samples(&m);
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
            UiMsg::Caps(c) => self.capabilities = Some(*c),
            UiMsg::Tuning(res) => match res {
                Ok((summary, recs)) => {
                    self.tuning_summary = Some(*summary);
                    self.tuning_recs = Some(recs);
                    self.tuning_error = None;
                    // Seed the re-tune picker with the first cached model once,
                    // so the button is immediately actionable.
                    if self.tune_sel_model.is_empty() {
                        if let Some(first) = self.cached.first() {
                            self.tune_sel_model = first.name.clone();
                        }
                    }
                }
                Err(e) => self.tuning_error = Some(e),
            },
            UiMsg::TuneProg(p) => self.tune_progress = Some(*p),
            UiMsg::TuneDone(res) => {
                self.tune_running = false;
                self.tune_status = Some(match res {
                    Ok(()) => "tune complete — reload the model to apply the new winners".into(),
                    Err(e) => format!("error: {e}"),
                });
                // Refresh the summary so the new winners show.
                self.spawn_tuning();
            }
            UiMsg::SettingsLoaded {
                config,
                config_path,
                profiles,
            } => {
                let draft = draft_from_config(&config);
                self.settings_config = Some(*config);
                self.settings_config_path = config_path;
                self.settings_profiles = profiles;
                self.settings_orig = Some(draft.clone());
                self.settings_draft = Some(draft);
                self.settings_loading = false;
                self.settings_error = None;
            }
            UiMsg::SettingsError(e) => {
                self.settings_loading = false;
                self.settings_error = Some(e);
            }
            UiMsg::SettingsSaved { reload, restart } => {
                self.settings_saving = false;
                let msg = if restart {
                    "Saved. Server fields changed — restart rustllama for them to take effect."
                } else if reload {
                    "Saved. Model / inference fields changed — reload the model from the Models page."
                } else {
                    "Saved. Changes hot-applied via the watcher."
                };
                self.settings_status = Some((false, msg.into()));
                // Re-fetch so the form (and its dirty snapshot) tracks disk.
                self.spawn_settings_config();
            }
            UiMsg::SettingsSaveError(e) => {
                self.settings_saving = false;
                self.settings_status = Some((true, format!("save failed: {e}")));
            }
            UiMsg::DecideDone(res) => {
                self.decide_busy = false;
                match res {
                    Ok(r) => {
                        self.decide_result = Some(*r);
                        self.decide_result_tab = self.decide_tab;
                        self.decide_error = None;
                    }
                    Err(e) => {
                        self.decide_result = None;
                        self.decide_error = Some(e);
                    }
                }
            }
            UiMsg::QuantizeDone(res) => {
                self.q_running = false;
                match res {
                    Ok(s) => {
                        self.q_result = Some(*s);
                        self.q_error = None;
                    }
                    Err(e) => {
                        self.q_result = None;
                        self.q_error = Some(e);
                    }
                }
            }
            UiMsg::PickedQuantInput(path) => {
                self.q_input = path.display().to_string();
                // Suggest an output path beside the source if none set yet.
                if self.q_output.trim().is_empty() {
                    self.q_output = suggest_quant_output(&path, &self.q_target);
                }
            }
            UiMsg::PickedQuantOutput(path) => {
                self.q_output = path.display().to_string();
            }
            UiMsg::PickedImage { name, uri, bytes } => {
                self.pending_images.push(PendingImage { name, uri, bytes });
            }
            UiMsg::HistoryAvailable(ok) => {
                self.history_enabled = Some(ok);
            }
            UiMsg::Conversations(list) => {
                self.conversations = list;
            }
            UiMsg::ConversationLoaded { id, messages } => {
                self.stop_stream();
                self.transcript = messages
                    .into_iter()
                    .map(|m| Turn {
                        role: m.role,
                        content: m.content,
                        images: Vec::new(),
                    })
                    .collect();
                self.pending.clear();
                self.pending_images.clear();
                self.streaming = false;
                self.last_usage = None;
                self.status = None;
                self.pending_question = None;
                self.pending_tools = None;
                self.request_id = None;
                self.current_conv = Some(id);
                self.saved_msgs = self.transcript.len();
            }
            UiMsg::HistorySaved { id, appended } => {
                self.history_saving = false;
                self.current_conv = Some(id);
                self.saved_msgs += appended;
                // Refresh the list so a newly-created conversation appears.
                self.spawn_list_conversations();
            }
            UiMsg::HistoryError(e) => {
                self.history_saving = false;
                tracing::warn!(target: "rustllama_gui", "history: {e}");
            }
            UiMsg::Err(e) => {
                tracing::debug!(target: "rustllama_gui", "{e}");
                self.status = Some(e);
            }
        }
    }

    /// Append one metrics tick to the Status sparkline ring buffers, evicting
    /// the oldest sample once each exceeds `METRICS_WINDOW`. Charts the smoothed
    /// EMA tok/s (falls back to the per-request value pre-warm-up) so the line
    /// doesn't jitter between cold and warm requests.
    fn push_metric_samples(&mut self, m: &MetricsSnapshot) {
        push_capped(&mut self.tok_series, m.ema_tok_s.or(m.last_tok_s).unwrap_or(0.0));
        push_capped(&mut self.ctx_series, m.ctx_used as f64);
        push_capped(&mut self.pending_series, m.pending as f64);
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

        // 2) Metrics poll. The status bar wants ~2 s, but the Status view's
        //    sparklines want ~1 Hz, so tighten the cadence to 1 s while that
        //    view is up. Gate on an in-flight flag so a slow server doesn't
        //    queue overlapping requests; `request_repaint_after` keeps the poll
        //    alive when the UI is idle.
        let metrics_interval = if self.view == View::Status {
            Duration::from_secs(1)
        } else {
            Duration::from_secs(2)
        };
        let due = self
            .metrics_last_poll
            .map_or(true, |t| now.duration_since(t) >= metrics_interval);
        if due && !self.metrics_inflight {
            self.metrics_inflight = true;
            self.metrics_last_poll = Some(now);
            self.spawn_metrics();
        }
        ctx.request_repaint_after(metrics_interval);

        // 2b) Status view: refresh the tuner-cache summary on a slow ~10 s
        //     cadence (it only changes when the user runs a tune), and — while
        //     a re-tune runs — poll its progress ~1 Hz for the stage bar.
        if self.view == View::Status {
            let tuning_due = self
                .tuning_last_poll
                .map_or(true, |t| now.duration_since(t) >= Duration::from_secs(10));
            if tuning_due {
                self.tuning_last_poll = Some(now);
                self.spawn_tuning();
            }
        }
        if self.tune_running {
            let prog_due = self
                .tune_prog_last_poll
                .map_or(true, |t| now.duration_since(t) >= Duration::from_secs(1));
            if prog_due {
                self.tune_prog_last_poll = Some(now);
                self.spawn_tune_progress();
            }
            ctx.request_repaint_after(Duration::from_secs(1));
        }

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
                if nav_item(ui, NavIcon::Decide, "Decide", self.view == View::Decide, true) {
                    actions.push(Action::SwitchView(View::Decide));
                }
                ui.add_space(3.0);
                if nav_item(ui, NavIcon::Status, "Status", self.view == View::Status, true) {
                    actions.push(Action::SwitchView(View::Status));
                }
                ui.add_space(3.0);
                if nav_item(ui, NavIcon::Quantize, "Quantize", self.view == View::Quantize, true) {
                    actions.push(Action::SwitchView(View::Quantize));
                }
                ui.add_space(3.0);
                if nav_item(
                    ui,
                    NavIcon::Settings,
                    "Settings",
                    self.view == View::Settings,
                    true,
                ) {
                    actions.push(Action::SwitchView(View::Settings));
                }
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
            View::Decide => self.render_decide(ctx, &mut actions),
            View::Status => self.render_status(ctx, &mut actions),
            View::Quantize => self.render_quantize(ctx, &mut actions),
            View::Settings => self.render_settings(ctx, &mut actions),
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
                    match v {
                        // Freshen the lists when entering Models.
                        View::Models => {
                            self.spawn_refresh_models();
                            self.spawn_refresh_cached();
                        }
                        // Status needs the tuner summary + the cached-model
                        // list (for the re-tune picker) on entry. Stamp the
                        // poll clock so step 2b doesn't immediately re-fetch.
                        View::Status => {
                            self.tuning_last_poll = Some(Instant::now());
                            self.spawn_tuning();
                            self.spawn_refresh_cached();
                        }
                        // Settings loads (or reloads) the on-disk config on
                        // entry unless it's already loaded.
                        View::Settings => {
                            if self.settings_draft.is_none() && !self.settings_loading {
                                self.settings_loading = true;
                                self.spawn_settings_config();
                            }
                        }
                        // Chat re-syncs the history list (if enabled) so a
                        // conversation saved from another surface shows.
                        View::Chat => {
                            if self.history_enabled == Some(true) {
                                self.spawn_list_conversations();
                            }
                        }
                        // Decide / Quantize keep local form state — no fetch.
                        View::Decide | View::Quantize => {}
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
                Action::RefreshTuning => {
                    self.tuning_last_poll = Some(Instant::now());
                    self.spawn_tuning();
                    self.spawn_capabilities();
                }
                Action::TuneModel(model) => {
                    if !self.tune_running && !model.trim().is_empty() {
                        self.tune_running = true;
                        self.tune_status = None;
                        self.tune_progress = None;
                        self.tune_prog_last_poll = None;
                        self.spawn_tune_model(model);
                    }
                }
                Action::ReloadSettings => {
                    self.settings_loading = true;
                    self.settings_status = None;
                    self.spawn_settings_config();
                }
                Action::SaveSettings => {
                    // Merge the draft over the loaded config so unmodeled
                    // sections survive, then PUT.
                    if let (Some(base), Some(draft)) =
                        (self.settings_config.clone(), self.settings_draft.clone())
                    {
                        let mut merged = base;
                        apply_draft_to_config(&mut merged, &draft);
                        self.settings_saving = true;
                        self.settings_status = None;
                        self.spawn_save_settings(merged);
                    }
                }
                Action::ResetSettings => {
                    // Discard edits: restore the form to the loaded snapshot.
                    if let Some(orig) = self.settings_orig.clone() {
                        self.settings_draft = Some(orig);
                    }
                    self.settings_status = None;
                }
                Action::ApplyProfile(name) => {
                    if !name.is_empty() && !self.settings_saving {
                        self.settings_saving = true;
                        self.settings_status = None;
                        self.spawn_apply_profile(name);
                    }
                }
                Action::RunDecide => {
                    if !self.decide_busy {
                        // The labels the result bars caption: options / levels /
                        // (boolean uses yes/no, filled in at render time).
                        let labels: Vec<String> = match self.decide_tab {
                            DecideTab::Choice => self
                                .decide_options
                                .iter()
                                .map(|s| s.trim().to_string())
                                .filter(|s| !s.is_empty())
                                .collect(),
                            DecideTab::Score => self
                                .decide_levels
                                .iter()
                                .map(|s| s.trim().to_string())
                                .filter(|s| !s.is_empty())
                                .collect(),
                            DecideTab::Boolean => vec!["yes".into(), "no".into()],
                        };
                        let context = self.decide_context.clone();
                        let valid = match self.decide_tab {
                            DecideTab::Boolean => !self.decide_question.trim().is_empty(),
                            _ => labels.len() >= 2,
                        };
                        if !valid {
                            self.decide_error = Some(match self.decide_tab {
                                DecideTab::Boolean => "Enter a yes/no question.".into(),
                                _ => "Enter at least two non-empty options.".into(),
                            });
                        } else {
                            self.decide_busy = true;
                            self.decide_error = None;
                            self.decide_result = None;
                            self.decide_result_labels = labels.clone();
                            self.spawn_decide(self.decide_tab, context, labels);
                        }
                    }
                }
                Action::PickQuantInput => self.spawn_pick_quant_input(),
                Action::PickQuantOutput => self.spawn_pick_quant_output(),
                Action::RunQuantize => {
                    if !self.q_running {
                        if self.q_input.trim().is_empty() || self.q_output.trim().is_empty() {
                            self.q_error = Some("Both input and output paths are required.".into());
                        } else {
                            self.q_running = true;
                            self.q_error = None;
                            self.q_result = None;
                            let apex = {
                                let a = self.q_apex.trim();
                                if a.is_empty() {
                                    None
                                } else {
                                    Some(a.to_string())
                                }
                            };
                            let recipe = {
                                let r = self.q_recipe.trim();
                                if r.is_empty() {
                                    None
                                } else {
                                    Some(r.to_string())
                                }
                            };
                            self.spawn_quantize(
                                self.q_input.trim().to_string(),
                                self.q_output.trim().to_string(),
                                self.q_target.clone(),
                                apex,
                                recipe,
                                self.q_keep_output,
                            );
                        }
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
    // As of Phase 5 the composer supports image attach (OpenAI `image_url`
    // content blocks) and a collapsible conversation-history sidebar (the
    // server's `/api/conversations` sqlite routes, feature-probed at startup).
    //
    // DEFERRED (intentionally not built here):
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
        // Phase-5 chat intents (history sidebar + image attach).
        let mut do_toggle_history = false;
        let mut hist_new = false;
        let mut hist_load: Option<i64> = None;
        let mut hist_delete: Option<i64> = None;
        let mut do_attach_image = false;
        let mut remove_image: Option<usize> = None;

        // Conversation-history sidebar (left; only when the feature is present).
        // Added before the top/bottom panels so it claims the full-height left
        // strip beside the nav rail. Unobtrusive — a toolbar toggle hides it.
        if self.history_enabled == Some(true) && self.history_open {
            let convs = self.conversations.clone();
            let current = self.current_conv;
            egui::SidePanel::left("history")
                .resizable(true)
                .default_width(212.0)
                .width_range(170.0..=340.0)
                .frame(
                    egui::Frame::default()
                        .fill(SIDEBAR)
                        .inner_margin(Margin::symmetric(8.0, 10.0))
                        .stroke(Stroke::new(1.0, BORDER)),
                )
                .show(ctx, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("History").strong().color(TEXT));
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if ui
                                .small_button("‹")
                                .on_hover_text("Hide the history sidebar")
                                .clicked()
                            {
                                do_toggle_history = true;
                            }
                        });
                    });
                    ui.add_space(6.0);
                    if ui
                        .add_sized(
                            [ui.available_width(), 26.0],
                            egui::Button::new(RichText::new("+  New chat").color(TEXT))
                                .fill(ACCENT),
                        )
                        .clicked()
                    {
                        hist_new = true;
                    }
                    ui.add_space(8.0);
                    egui::ScrollArea::vertical()
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            if convs.is_empty() {
                                ui.label(
                                    RichText::new("No conversations yet").color(MUTED).small(),
                                );
                            }
                            for c in &convs {
                                let (load, delete) =
                                    conversation_row(ui, c, current == Some(c.id));
                                if load {
                                    hist_load = Some(c.id);
                                }
                                if delete {
                                    hist_delete = Some(c.id);
                                }
                                ui.add_space(4.0);
                            }
                        });
                });
        }

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
        // History-sidebar toggle affordance state (drawn only when the feature
        // is present so the toolbar stays clean on a default server).
        let history_present = self.history_enabled == Some(true);
        let history_shown = self.history_open;
        egui::TopBottomPanel::top("chat_toolbar")
            .frame(
                egui::Frame::default()
                    .fill(BG)
                    .inner_margin(Margin::symmetric(12.0, 6.0))
                    .stroke(Stroke::new(1.0, BORDER)),
            )
            .show(ctx, |ui| {
                ui.horizontal_wrapped(|ui| {
                    if history_present {
                        let label = if history_shown { "☰ History" } else { "☰ History ›" };
                        if ui
                            .selectable_label(history_shown, label)
                            .on_hover_text("Show / hide the conversation-history sidebar")
                            .clicked()
                        {
                            do_toggle_history = true;
                        }
                    }
                    if ui
                        .add_enabled(!transcript_empty, egui::Button::new("New"))
                        .on_hover_text("Start a fresh chat (Ctrl+L)")
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
        // Snapshot the staged images for the composer chips (name + size).
        let pending_imgs: Vec<(String, u64)> = self
            .pending_images
            .iter()
            .map(|p| (p.name.clone(), p.bytes))
            .collect();
        let has_pending_imgs = !pending_imgs.is_empty();
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

                // Attach-image affordance + staged-image chips. Sent as OpenAI
                // `image_url` content blocks with the next user turn (a
                // vision-aware model splices them; a text model sees a
                // `[image: …]` placeholder).
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .add_enabled(!streaming, egui::Button::new("＋ Image"))
                        .on_hover_text("Attach an image to the next message")
                        .clicked()
                    {
                        do_attach_image = true;
                    }
                    for (i, (name, bytes)) in pending_imgs.iter().enumerate() {
                        image_chip(ui, name, *bytes, &mut remove_image, i);
                    }
                });
                ui.add_space(6.0);

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
                if enter_send
                    && !self.streaming
                    && (!self.input.trim().is_empty() || has_pending_imgs)
                {
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
                            let can_send =
                                !self.input.trim().is_empty() || has_pending_imgs;
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
            // Ctrl+L / New: a fresh chat (also detaches the history binding).
            self.new_chat();
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
        // Phase-5 chat intents (history sidebar + image attach).
        if do_toggle_history {
            self.history_open = !self.history_open;
        }
        if hist_new {
            self.new_chat();
        }
        if let Some(id) = hist_load {
            self.load_conversation(id);
        }
        if let Some(id) = hist_delete {
            if self.current_conv == Some(id) {
                self.current_conv = None;
                self.saved_msgs = 0;
            }
            self.spawn_delete_conversation(id);
        }
        if do_attach_image {
            self.spawn_pick_image();
        }
        if let Some(i) = remove_image {
            if i < self.pending_images.len() {
                self.pending_images.remove(i);
            }
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

    /// The Status view (Phase 4): a live dashboard — server health / version /
    /// uptime, the compute-backend snapshot, a per-device compute inventory,
    /// live sparklines (tok/s · ctx · pending), and the tuner-cache panel with
    /// a per-model re-tune (progress polled ~1 Hz while it runs).
    //
    // DEFERRED (TODOs, intentionally not built here):
    //   - GPU power / energy derivation (needs the L0 Sysman energy counter +
    //     a two-sample dE/dt like Status.tsx's GPU-sensors panel).
    //   - The paged-KV pool occupancy panel.
    //   - The synthetic throughput probe ("Run probe").
    //   - The static API-surface endpoint list.
    fn render_status(&mut self, ctx: &egui::Context, actions: &mut Vec<Action>) {
        // Clone the read-only display state up front so the panel closure can
        // still take `&mut self.tune_sel_model` for the re-tune picker without
        // aliasing (mirrors render_models). The ring buffers are ≤120 f64 each,
        // so cloning them per frame is cheap.
        let caps = self.capabilities.clone();
        let metrics = self.metrics.clone();
        let health = self.health;
        let tuning = self.tuning_summary.clone();
        let recs = self.tuning_recs.clone();
        let tuning_error = self.tuning_error.clone();
        let cached = self.cached.clone();
        let tune_running = self.tune_running;
        let tune_progress = self.tune_progress.clone();
        let tune_status = self.tune_status.clone();
        let tok = self.tok_series.clone();
        let ctxs = self.ctx_series.clone();
        let pend = self.pending_series.clone();

        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(BG).inner_margin(Margin::same(16.0)))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Status").heading().color(TEXT));
                            ui.add_space(6.0);
                            if ui.button("Refresh").clicked() {
                                actions.push(Action::RefreshTuning);
                            }
                        });
                        ui.add_space(14.0);

                        // --- Server ---
                        card(ui, |ui| {
                            ui.label(RichText::new("Server").strong().color(TEXT));
                            ui.add_space(6.0);
                            kv_row(
                                ui,
                                "Status",
                                match health {
                                    Some(true) => "live",
                                    Some(false) => "offline",
                                    None => "—",
                                },
                            );
                            kv_row(
                                ui,
                                "Version",
                                caps.as_ref()
                                    .map(|c| c.server_version.as_str())
                                    .filter(|s| !s.is_empty())
                                    .unwrap_or("—"),
                            );
                            kv_row(
                                ui,
                                "Uptime",
                                &metrics
                                    .as_ref()
                                    .map(|m| fmt_uptime(m.uptime_s))
                                    .unwrap_or_else(|| "—".into()),
                            );
                            kv_row(
                                ui,
                                "Loaded model",
                                metrics
                                    .as_ref()
                                    .map(|m| m.model_id.as_str())
                                    .filter(|s| !s.is_empty())
                                    .unwrap_or("—"),
                            );
                        });
                        ui.add_space(14.0);

                        // --- Backends ---
                        card(ui, |ui| {
                            ui.label(RichText::new("Backends").strong().color(TEXT));
                            ui.label(
                                RichText::new("compute dispatch paths").color(MUTED).small(),
                            );
                            ui.add_space(8.0);
                            match &caps {
                                Some(c) => render_backends(ui, &c.backends),
                                None => {
                                    ui.label(
                                        RichText::new("capabilities unavailable").color(MUTED),
                                    );
                                }
                            }
                        });
                        ui.add_space(14.0);

                        // --- Compute inventory (every GPU + the CPU tier) ---
                        card(ui, |ui| {
                            ui.label(RichText::new("Compute inventory").strong().color(TEXT));
                            ui.label(
                                RichText::new("every GPU + the CPU tier").color(MUTED).small(),
                            );
                            ui.add_space(8.0);
                            if let Some(m) = &metrics {
                                let gpus = m.gpus.clone().unwrap_or_default();
                                for g in &gpus {
                                    let used = match (g.vram_total_bytes, g.vram_free_bytes) {
                                        (Some(t), Some(f)) => Some(t.saturating_sub(f)),
                                        _ => None,
                                    };
                                    inventory_row(
                                        ui,
                                        gpu_badge(&g.vendor),
                                        &format!("GPU {}", g.index),
                                        &g.name,
                                        used,
                                        g.vram_total_bytes,
                                        g.utilization_pct,
                                    );
                                }
                                let ram_used = (m.ram_total_bytes > 0)
                                    .then(|| m.ram_total_bytes.saturating_sub(m.ram_available_bytes));
                                inventory_row(
                                    ui,
                                    "CPU",
                                    "CPU",
                                    m.cpu_brand.as_deref().unwrap_or("Host CPU"),
                                    ram_used,
                                    (m.ram_total_bytes > 0).then_some(m.ram_total_bytes),
                                    m.cpu_utilization_pct,
                                );
                                if gpus.is_empty() {
                                    ui.label(
                                        RichText::new(
                                            "No GPU visible — inference runs on the CPU tier.",
                                        )
                                        .color(MUTED)
                                        .small(),
                                    );
                                }
                            } else {
                                ui.label(RichText::new("waiting for /v1/metrics…").color(MUTED));
                            }
                        });
                        ui.add_space(14.0);

                        // --- Live metrics (sparklines) ---
                        card(ui, |ui| {
                            ui.label(RichText::new("Live metrics").strong().color(TEXT));
                            ui.add_space(8.0);
                            if let Some(m) = &metrics {
                                ui.horizontal_wrapped(|ui| {
                                    let tok_v = m
                                        .ema_tok_s
                                        .or(m.last_tok_s)
                                        .map(|t| format!("{t:.1}"))
                                        .unwrap_or_else(|| "—".into());
                                    sparkline(
                                        ui, "spark_tok", "Tokens / second", &tok_v, &tok,
                                        HEALTH_OK, None,
                                    );
                                    ui.add_space(12.0);
                                    sparkline(
                                        ui,
                                        "spark_ctx",
                                        "KV context used",
                                        &format!("{} / {}", m.ctx_used, m.ctx_size),
                                        &ctxs,
                                        ACCENT,
                                        (m.ctx_size > 0).then_some(m.ctx_size as f64),
                                    );
                                    ui.add_space(12.0);
                                    sparkline(
                                        ui,
                                        "spark_pend",
                                        "Pending requests",
                                        &format!("{} / {}", m.pending, m.max_pending),
                                        &pend,
                                        WARN,
                                        (m.max_pending > 0).then_some(m.max_pending as f64),
                                    );
                                });
                                ui.add_space(6.0);
                                ui.label(
                                    RichText::new(format!(
                                        "KV dtype: {} · concurrency: {} · uptime {}",
                                        m.kv_dtype.as_deref().unwrap_or("—"),
                                        m.concurrency,
                                        fmt_uptime(m.uptime_s),
                                    ))
                                    .color(MUTED)
                                    .small(),
                                );
                            } else {
                                ui.label(
                                    RichText::new("Waiting for the first /v1/metrics tick…")
                                        .color(MUTED),
                                );
                            }
                        });
                        ui.add_space(14.0);

                        // --- Tuner cache ---
                        card(ui, |ui| {
                            ui.label(RichText::new("Tuner cache").strong().color(TEXT));
                            ui.add_space(8.0);
                            if let Some(e) = &tuning_error {
                                ui.label(
                                    RichText::new(format!("failed to read: {e}"))
                                        .color(DANGER)
                                        .small(),
                                );
                            }
                            match &tuning {
                                None if tuning_error.is_none() => {
                                    ui.label(RichText::new("loading…").color(MUTED));
                                }
                                None => {}
                                Some(t) => {
                                    match &t.device {
                                        None => {
                                            ui.label(
                                                RichText::new(
                                                    "No SYCL device visible — the tuner cache is \
                                                     keyed by device, so there is nothing to \
                                                     surface.",
                                                )
                                                .color(MUTED)
                                                .small(),
                                            );
                                        }
                                        Some(dev) => {
                                            kv_row(ui, "Device", &dev.name);
                                            kv_row(ui, "Driver", &dev.driver_ver);
                                            kv_row(ui, "VRAM", &format!("{} MiB", dev.vram_mb));
                                        }
                                    }
                                    kv_row(
                                        ui,
                                        "Cache present",
                                        if t.cache_present {
                                            "yes"
                                        } else {
                                            "none — run rustllama tune"
                                        },
                                    );
                                    kv_row(ui, "Last tuned", t.last_tuned.as_deref().unwrap_or("—"));
                                    kv_row(
                                        ui,
                                        "Tuned kernel shapes",
                                        &t.kernel_entry_count.to_string(),
                                    );
                                    winner_row(
                                        ui,
                                        "Batch-size winner",
                                        t.batch_size.map(|b| b.to_string()),
                                        t.auto_apply_batch_size,
                                    );
                                    winner_row(
                                        ui,
                                        "KV-dtype winner",
                                        t.kv_dtype.clone(),
                                        t.auto_apply_kv_dtype,
                                    );
                                    winner_row(
                                        ui,
                                        "Flash-attn winner",
                                        t.flash_attention
                                            .map(|b| if b { "on".into() } else { "off".into() }),
                                        t.auto_apply_flash_attention,
                                    );
                                    winner_row(
                                        ui,
                                        "KV-layout winner",
                                        t.kv_cache_layout.clone(),
                                        t.auto_apply_kv_cache_layout,
                                    );
                                    if !t.placement.is_empty() {
                                        ui.add_space(6.0);
                                        ui.label(
                                            RichText::new(format!(
                                                "Placement winners ({}){}",
                                                t.placement.len(),
                                                if t.auto_apply_placement {
                                                    " · auto-applied"
                                                } else {
                                                    " · stored"
                                                }
                                            ))
                                            .color(MUTED)
                                            .small(),
                                        );
                                        for p in &t.placement {
                                            ui.label(
                                                RichText::new(format!(
                                                    "  {} → n_gpu_layers={}",
                                                    p.model_key, p.n_gpu_layers
                                                ))
                                                .color(TEXT)
                                                .small(),
                                            );
                                        }
                                    }
                                }
                            }

                            // Untuned-shape banner.
                            if let Some(r) = &recs {
                                if r.untuned_count > 0 {
                                    ui.add_space(8.0);
                                    egui::Frame::none()
                                        .fill(WARN.gamma_multiply(0.12))
                                        .rounding(Rounding::same(8.0))
                                        .stroke(Stroke::new(1.0, WARN))
                                        .inner_margin(Margin::same(10.0))
                                        .show(ui, |ui| {
                                            ui.label(
                                                RichText::new(format!(
                                                    "{} kernel shape(s) dispatched without a \
                                                     cached LWS entry — the kernel default works \
                                                     but typically leaves 1.5–3× on the table.",
                                                    r.untuned_count
                                                ))
                                                .color(TEXT)
                                                .small(),
                                            );
                                        });
                                }
                            }

                            // --- Re-tune (per cached model) ---
                            ui.add_space(10.0);
                            ui.separator();
                            ui.label(RichText::new("Re-tune").color(MUTED).small());
                            ui.add_space(4.0);
                            if cached.is_empty() {
                                ui.label(
                                    RichText::new(
                                        "No cached models — pull one from the Models page first.",
                                    )
                                    .color(MUTED)
                                    .small(),
                                );
                            } else {
                                ui.horizontal(|ui| {
                                    egui::ComboBox::from_id_salt("tune_model_combo")
                                        .width(280.0)
                                        .selected_text(if self.tune_sel_model.is_empty() {
                                            "select a model".to_string()
                                        } else {
                                            display_name(&self.tune_sel_model).to_string()
                                        })
                                        .show_ui(ui, |ui| {
                                            for m in &cached {
                                                ui.selectable_value(
                                                    &mut self.tune_sel_model,
                                                    m.name.clone(),
                                                    display_name(&m.name),
                                                );
                                            }
                                        });
                                    let can = !tune_running && !self.tune_sel_model.is_empty();
                                    if ui
                                        .add_enabled(
                                            can,
                                            egui::Button::new(if tune_running {
                                                "tuning…"
                                            } else {
                                                "Re-tune"
                                            }),
                                        )
                                        .on_hover_text(
                                            "Full per-model sweep (KV-dtype, CPU/GPU dispatch, \
                                             kernels, batch size), persisted to the tuner cache. \
                                             Runs in a background process; reload the model \
                                             afterwards to apply.",
                                        )
                                        .clicked()
                                    {
                                        actions.push(Action::TuneModel(self.tune_sel_model.clone()));
                                    }
                                    if tune_running {
                                        ui.spinner();
                                    }
                                });
                                if tune_running {
                                    ui.add_space(6.0);
                                    if let Some(p) = &tune_progress {
                                        let frac = (p.pct / 100.0).clamp(0.0, 1.0) as f32;
                                        ui.add(
                                            egui::ProgressBar::new(frac)
                                                .desired_width(f32::INFINITY)
                                                .fill(ACCENT)
                                                .text(format!(
                                                    "{}/{} {}",
                                                    p.stage_idx, p.stage_total, p.stage_name
                                                )),
                                        );
                                        if !p.line.is_empty() {
                                            ui.label(
                                                RichText::new(&p.line)
                                                    .color(MUTED)
                                                    .monospace()
                                                    .small(),
                                            );
                                        }
                                    } else {
                                        ui.label(RichText::new("starting…").color(MUTED).small());
                                    }
                                }
                                if let Some(s) = &tune_status {
                                    ui.add_space(6.0);
                                    let col = if s.starts_with("error") { DANGER } else { HEALTH_OK };
                                    ui.label(RichText::new(s).color(col).small());
                                }
                            }
                        });
                    });
            });
    }

    /// The Settings view (Phase 4): an editable form over the on-disk config,
    /// grouped into collapsible sections + a Save (PUT /v1/config). Sections are
    /// tagged live / reload / restart per the hot-apply semantics (mirrors
    /// Settings.tsx); the form round-trips the whole config so unmodeled
    /// sections survive a save. Profiles apply sparse overrides; Hardware is
    /// read-only.
    //
    // DEFERRED (TODOs): the LAN-access QR panel (needs a QR renderer), the
    // audit-log tail panel, the crash-logs panel, and the chat-template live
    // preview. The client methods exist (`lan_info` / `audit_log_tail` /
    // `crash_logs` / `template_preview`) but those panels are omitted here.
    fn render_settings(&mut self, ctx: &egui::Context, actions: &mut Vec<Action>) {
        // Read-only display data cloned up front; the form below takes
        // `&mut self.settings_draft`, and the profile picker edits a local
        // (written back after the panel) — both keep the closure aliasing-free.
        let caps = self.capabilities.clone();
        let metrics = self.metrics.clone();
        let config_path = self.settings_config_path.clone();
        let profiles = self.settings_profiles.clone();
        let error = self.settings_error.clone();
        let status = self.settings_status.clone();
        let saving = self.settings_saving;
        let loading = self.settings_loading;
        let orig = self.settings_orig.clone();
        let has_draft = self.settings_draft.is_some();
        let mut profile_sel = self.settings_profile_sel.clone();

        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(BG).inner_margin(Margin::same(16.0)))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Settings").heading().color(TEXT));
                            ui.add_space(6.0);
                            // Re-read the on-disk config (e.g. after editing
                            // config.toml by hand). Discards unsaved form edits.
                            if ui
                                .add_enabled(!saving, egui::Button::new("Reload"))
                                .on_hover_text("Re-read config.toml from disk (discards unsaved edits)")
                                .clicked()
                            {
                                actions.push(Action::ReloadSettings);
                            }
                        });
                        ui.label(
                            RichText::new(format!(
                                "Edits write to {}.  live = hot-applied · reload = reload the \
                                 model (Models → Load) · restart = restart rustllama.",
                                if config_path.is_empty() {
                                    "config.toml"
                                } else {
                                    config_path.as_str()
                                }
                            ))
                            .color(MUTED)
                            .small(),
                        );
                        ui.add_space(10.0);

                        if let Some(e) = &error {
                            error_banner(ui, &format!("error loading config: {e}"));
                            ui.add_space(8.0);
                        }
                        if let Some((is_err, msg)) = &status {
                            let col = if *is_err { DANGER } else { HEALTH_OK };
                            egui::Frame::none()
                                .fill(col.gamma_multiply(0.12))
                                .rounding(Rounding::same(8.0))
                                .stroke(Stroke::new(1.0, col))
                                .inner_margin(Margin::same(10.0))
                                .show(ui, |ui| {
                                    ui.label(RichText::new(msg).color(col).small());
                                });
                            ui.add_space(8.0);
                        }

                        if loading && !has_draft {
                            ui.label(RichText::new("loading config…").color(MUTED));
                            return;
                        }

                        // Profiles (sparse overrides) — a local &mut so it stays
                        // disjoint from the draft borrow below.
                        settings_profiles_section(
                            ui,
                            &profiles,
                            &mut profile_sel,
                            // dirty gate computed against the draft below; pass a
                            // conservative value here then refine via the button.
                            self.settings_draft.as_ref() != orig.as_ref(),
                            saving,
                            actions,
                        );

                        if let Some(draft) = self.settings_draft.as_mut() {
                            let dirty = orig.as_ref().map_or(false, |o| *draft != *o);

                            collapsing_section(ui, "Server", "restart", |ui| {
                                form_text(ui, "Bind address",
                                    "127.0.0.1 = localhost only · 0.0.0.0 = LAN",
                                    &mut draft.bind_addr);
                                form_text(ui, "Port", "default 11434", &mut draft.port);
                                form_text(ui, "API key",
                                    "Empty = no auth. Sent as `Authorization: Bearer <key>`.",
                                    &mut draft.api_key);
                                form_num(ui, "Max loaded models",
                                    "Warm pool cap (LRU evicts non-default models).",
                                    &mut draft.max_loaded_models, 1.0..=64.0);
                                form_text(ui, "CORS origins",
                                    "Comma-separated. Empty = none. `*` allows any origin.",
                                    &mut draft.cors_origins);
                            });

                            collapsing_section(ui, "Model", "reload", |ui| {
                                form_text(ui, "GGUF path",
                                    "Absolute path. Mutually exclusive with the hub ref.",
                                    &mut draft.model_path);
                                form_text(ui, "Hub ref",
                                    "owner/repo:filename.gguf — resolved from the local cache.",
                                    &mut draft.model_hub);
                                form_text(ui, "Chat template",
                                    "auto = read from the GGUF · chatml / llama3 / inline Jinja.",
                                    &mut draft.chat_template);
                            });

                            collapsing_section(ui, "Inference", "reload", |ui| {
                                form_num(ui, "Context size",
                                    "Max tokens (prompt + completion) per request.",
                                    &mut draft.ctx_size, 512.0..=1_048_576.0);
                                form_num(ui, "Batch size",
                                    "Prefill chunk. Larger = faster but more RAM per chunk.",
                                    &mut draft.batch_size, 1.0..=65_536.0);
                                form_num(ui, "Threads", "0 = auto (physical cores).",
                                    &mut draft.threads, 0.0..=256.0);
                                form_combo(ui, "kv_dtype_combo", "KV dtype (both K and V)",
                                    "Default for both K and V; override per-channel below to split.",
                                    &mut draft.kv_dtype, &KV_DTYPES);
                                form_combo(ui, "k_dtype_combo", "K dtype (override)",
                                    "Empty = use KV dtype.",
                                    &mut draft.k_dtype, &KV_DTYPES_OPT);
                                form_combo(ui, "v_dtype_combo", "V dtype (override)",
                                    "Empty = use KV dtype. V is more precision-sensitive than K.",
                                    &mut draft.v_dtype, &KV_DTYPES_OPT);
                                form_bool(ui, "Flash attention",
                                    "Fused softmax-attention (no-op until SYCL dispatch wires up).",
                                    &mut draft.flash_attention);
                                form_bool(ui, "Speculative decoding (n-gram)",
                                    "Prompt-lookup drafting + one batched verify per round. No \
                                     second model, zero RAM.",
                                    &mut draft.speculative_ngram);
                                form_bool(ui, "Prefix cache",
                                    "Reuse KV for shared prompt prefixes (big speedup for chat / \
                                     coding flows).",
                                    &mut draft.prefix_cache);
                                form_bool(ui, "Keep quant raw",
                                    "Skip dequant-to-F16 on small tensors. Saves 1-3 GB on a \
                                     7B-24B model; recommended on ≤16 GB hosts.",
                                    &mut draft.keep_quant_raw);
                            });

                            collapsing_section(ui, "UI", "live", |ui| {
                                form_combo(ui, "theme_combo", "Theme",
                                    "`system` follows the OS dark/light setting.",
                                    &mut draft.theme, &THEMES);
                                form_num(ui, "Font size", "", &mut draft.font_size, 8.0..=32.0);
                                form_text(ui, "Code theme",
                                    "Syntax-highlight palette name for assistant code blocks.",
                                    &mut draft.code_theme);
                            });

                            // Save bar.
                            ui.add_space(12.0);
                            ui.separator();
                            ui.add_space(6.0);
                            ui.horizontal(|ui| {
                                if ui
                                    .add_enabled(
                                        dirty && !saving,
                                        egui::Button::new(RichText::new("Save changes").color(TEXT))
                                            .fill(ACCENT),
                                    )
                                    .clicked()
                                {
                                    actions.push(Action::SaveSettings);
                                }
                                if ui
                                    .add_enabled(dirty && !saving, egui::Button::new("Discard"))
                                    .clicked()
                                {
                                    actions.push(Action::ResetSettings);
                                }
                                if saving {
                                    ui.spinner();
                                    ui.label(RichText::new("saving…").color(MUTED).small());
                                } else if !dirty {
                                    ui.label(
                                        RichText::new("no unsaved changes").color(MUTED).small(),
                                    );
                                }
                            });
                        }

                        // Hardware (read-only) — a summary of the compute
                        // backends the server sees.
                        ui.add_space(12.0);
                        collapsing_section_ro(ui, "Hardware", |ui| {
                            let sycl = caps
                                .as_ref()
                                .map(|c| c.backends.sycl.device_count)
                                .unwrap_or(0);
                            kv_row(
                                ui,
                                "SYCL devices",
                                &if sycl > 0 {
                                    format!("{sycl} (oneAPI detected)")
                                } else {
                                    "0 (no Intel GPU / oneAPI runtime)".into()
                                },
                            );
                            if let Some(c) = &caps {
                                if let Some(cuda) = &c.backends.cuda {
                                    kv_row(
                                        ui,
                                        "CUDA devices",
                                        &format!(
                                            "{} ({} kernel-ready)",
                                            cuda.device_count, cuda.compute_ready
                                        ),
                                    );
                                }
                                let simd = if c.backends.cpu.simd_features.is_empty() {
                                    "scalar (non-x86)".to_string()
                                } else {
                                    c.backends.cpu.simd_features.join(", ")
                                };
                                kv_row(ui, "CPU SIMD", &simd);
                            }
                            if let Some(m) = &metrics {
                                if let Some(gpus) = &m.gpus {
                                    for g in gpus {
                                        kv_row(ui, &format!("GPU {}", g.index), &g.name);
                                    }
                                }
                            }
                        });
                        ui.add_space(60.0);
                    });
            });

        // Write the profile picker's edit back to state.
        self.settings_profile_sel = profile_sel;
    }

    /// The Decide view (Phase 5): tabs Choice / Score / Boolean over the
    /// `/v1/decide/*` endpoints. Each scores candidate options against the
    /// loaded model and returns a typed value + per-option probabilities (no
    /// prose). Mirrors Decide.tsx — a context box, a per-tab input (dynamic
    /// option/level list, or a yes/no question), a Decide button, and a result
    /// with a probability bar per option + a "calibrated" badge.
    fn render_decide(&mut self, ctx: &egui::Context, actions: &mut Vec<Action>) {
        let busy = self.decide_busy;
        let result = self.decide_result.clone();
        let result_tab = self.decide_result_tab;
        let result_labels = self.decide_result_labels.clone();
        let error = self.decide_error.clone();

        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(BG).inner_margin(Margin::same(16.0)))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.set_max_width(760.0);
                        ui.label(RichText::new("Decide").heading().color(TEXT));
                        ui.label(
                            RichText::new(
                                "Score candidate options against the loaded model and return a \
                                 typed value + probabilities — no prose generation. Choice picks \
                                 one, Score rates an ordered scale, Boolean is a calibrated yes/no.",
                            )
                            .color(MUTED)
                            .small(),
                        );
                        ui.add_space(12.0);

                        // Tabs.
                        ui.horizontal(|ui| {
                            for (tab, label) in [
                                (DecideTab::Choice, "Choice"),
                                (DecideTab::Score, "Score"),
                                (DecideTab::Boolean, "Boolean"),
                            ] {
                                if ui
                                    .selectable_label(self.decide_tab == tab, label)
                                    .clicked()
                                {
                                    self.decide_tab = tab;
                                }
                            }
                        });
                        ui.add_space(10.0);

                        card(ui, |ui| {
                            ui.label(RichText::new("Context").color(MUTED).small());
                            ui.add(
                                egui::TextEdit::multiline(&mut self.decide_context)
                                    .desired_width(f32::INFINITY)
                                    .desired_rows(5)
                                    .font(egui::TextStyle::Monospace),
                            );
                            ui.add_space(8.0);

                            match self.decide_tab {
                                DecideTab::Boolean => {
                                    ui.label(
                                        RichText::new("Question (yes/no)").color(MUTED).small(),
                                    );
                                    ui.add(
                                        egui::TextEdit::singleline(&mut self.decide_question)
                                            .desired_width(f32::INFINITY),
                                    );
                                }
                                DecideTab::Choice => {
                                    ui.label(RichText::new("Options").color(MUTED).small());
                                    string_list_editor(ui, &mut self.decide_options, "option");
                                }
                                DecideTab::Score => {
                                    ui.label(
                                        RichText::new("Levels (ordered scale)")
                                            .color(MUTED)
                                            .small(),
                                    );
                                    string_list_editor(ui, &mut self.decide_levels, "level");
                                }
                            }

                            ui.add_space(10.0);
                            ui.horizontal(|ui| {
                                if ui
                                    .add_enabled(
                                        !busy,
                                        egui::Button::new(
                                            RichText::new(if busy {
                                                "Deciding…"
                                            } else {
                                                "Decide"
                                            })
                                            .color(TEXT),
                                        )
                                        .fill(ACCENT),
                                    )
                                    .clicked()
                                {
                                    actions.push(Action::RunDecide);
                                }
                                if busy {
                                    ui.spinner();
                                }
                            });
                        });

                        if let Some(e) = &error {
                            ui.add_space(10.0);
                            error_banner(ui, e);
                        }

                        if let Some(r) = &result {
                            ui.add_space(14.0);
                            card(ui, |ui| {
                                render_decide_result(ui, r, result_tab, &result_labels);
                            });
                        }
                    });
            });
    }

    /// The Quantize view (Phase 5): re-encode a GGUF to a smaller target dtype
    /// in-process, on a worker thread. Mirrors Quantize.tsx — source/output
    /// pickers, a target-dtype dropdown, an optional APEX mixed-precision tier,
    /// an optional recipe file, a keep-LM-head checkbox, and a result panel.
    /// Runs the SAME `rustllama_gguf::quantize::quantize_gguf_to_path` pipeline
    /// the Tauri `quantize_model` command drives (see [`run_quantize_job`]).
    fn render_quantize(&mut self, ctx: &egui::Context, actions: &mut Vec<Action>) {
        let running = self.q_running;
        let result = self.q_result.clone();
        let error = self.q_error.clone();

        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(BG).inner_margin(Margin::same(16.0)))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.set_max_width(820.0);
                        ui.label(RichText::new("Quantize").heading().color(TEXT));
                        ui.label(
                            RichText::new(
                                "Re-encode a GGUF model to a smaller target dtype. Source can be \
                                 any supported quant (F32/F16/BF16, Q4/Q5/Q8, Q2–Q8_K, TQ1/2, \
                                 IQ1–IQ4); norms and biases pass through automatically. Runs on a \
                                 worker thread — a real model can take minutes.",
                            )
                            .color(MUTED)
                            .small(),
                        );
                        ui.add_space(12.0);

                        // --- Paths ---
                        card(ui, |ui| {
                            ui.label(RichText::new("Paths").strong().color(TEXT));
                            ui.add_space(8.0);
                            ui.label(RichText::new("Source GGUF").color(MUTED).small());
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::TextEdit::singleline(&mut self.q_input)
                                        .desired_width(ui.available_width() - 96.0)
                                        .hint_text("path to the source .gguf")
                                        .font(egui::TextStyle::Monospace),
                                );
                                if ui.button("Browse…").clicked() {
                                    actions.push(Action::PickQuantInput);
                                }
                            });
                            ui.add_space(8.0);
                            ui.label(
                                RichText::new("Output GGUF (created or overwritten)")
                                    .color(MUTED)
                                    .small(),
                            );
                            ui.horizontal(|ui| {
                                ui.add(
                                    egui::TextEdit::singleline(&mut self.q_output)
                                        .desired_width(ui.available_width() - 96.0)
                                        .hint_text("path to write the quantized .gguf")
                                        .font(egui::TextStyle::Monospace),
                                );
                                if ui.button("Save as…").clicked() {
                                    actions.push(Action::PickQuantOutput);
                                }
                            });
                        });
                        ui.add_space(14.0);

                        // --- Target ---
                        card(ui, |ui| {
                            ui.label(RichText::new("Target").strong().color(TEXT));
                            ui.add_space(8.0);
                            ui.label(
                                RichText::new(
                                    "Default dtype (any tensor not matched by APEX or a recipe)",
                                )
                                .color(MUTED)
                                .small(),
                            );
                            egui::ComboBox::from_id_salt("quant_target_combo")
                                .width(300.0)
                                .selected_text(quant_target_label(&self.q_target))
                                .show_ui(ui, |ui| {
                                    for (name, label) in QUANT_TARGETS {
                                        ui.selectable_value(
                                            &mut self.q_target,
                                            (*name).to_string(),
                                            *label,
                                        );
                                    }
                                });

                            ui.add_space(8.0);
                            ui.label(
                                RichText::new("APEX profile (mixed-precision for MoE models)")
                                    .color(MUTED)
                                    .small(),
                            );
                            egui::ComboBox::from_id_salt("quant_apex_combo")
                                .width(300.0)
                                .selected_text(apex_tier_label(&self.q_apex))
                                .show_ui(ui, |ui| {
                                    for (name, label) in APEX_TIERS {
                                        ui.selectable_value(
                                            &mut self.q_apex,
                                            (*name).to_string(),
                                            *label,
                                        );
                                    }
                                });

                            ui.add_space(8.0);
                            ui.label(
                                RichText::new("Recipe file (optional; `<glob> <dtype>` per line)")
                                    .color(MUTED)
                                    .small(),
                            );
                            ui.add(
                                egui::TextEdit::singleline(&mut self.q_recipe)
                                    .desired_width(f32::INFINITY)
                                    .hint_text("(leave blank to skip)")
                                    .font(egui::TextStyle::Monospace),
                            );

                            ui.add_space(8.0);
                            ui.checkbox(
                                &mut self.q_keep_output,
                                RichText::new(
                                    "Keep LM head at source precision (recommended for <4 bpw)",
                                )
                                .color(TEXT)
                                .small(),
                            );
                        });
                        ui.add_space(14.0);

                        ui.horizontal(|ui| {
                            let can_run = !running
                                && !self.q_input.trim().is_empty()
                                && !self.q_output.trim().is_empty();
                            if ui
                                .add_enabled(
                                    can_run,
                                    egui::Button::new(
                                        RichText::new(if running {
                                            "Running…"
                                        } else {
                                            "Run quantize"
                                        })
                                        .color(TEXT),
                                    )
                                    .fill(if running { ELEVATED2 } else { HEALTH_OK }),
                                )
                                .clicked()
                            {
                                actions.push(Action::RunQuantize);
                            }
                            if running {
                                ui.spinner();
                                ui.label(
                                    RichText::new("re-encoding — this can take minutes…")
                                        .color(MUTED)
                                        .small(),
                                );
                            }
                        });

                        if let Some(e) = &error {
                            ui.add_space(14.0);
                            error_banner(ui, e);
                        }

                        if let Some(s) = &result {
                            ui.add_space(14.0);
                            card(ui, |ui| {
                                render_quantize_result(ui, s);
                            });
                        }
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
                    if !turn.images.is_empty() {
                        ui.label(
                            RichText::new(format!(
                                "🖼 {} image{} attached",
                                turn.images.len(),
                                if turn.images.len() == 1 { "" } else { "s" }
                            ))
                            .color(MUTED)
                            .small(),
                        );
                    }
                    if !turn.content.is_empty() {
                        ui.label(RichText::new(turn.content.as_str()).color(TEXT));
                    }
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
        NavIcon::Status => {
            // A three-bar mini bar-chart (dashboard vibe).
            let base = rect.max.y - 1.0;
            let heights = [0.45f32, 0.85, 0.62];
            let n = heights.len();
            let slot = rect.width() / (n as f32);
            for (i, h) in heights.iter().enumerate() {
                let x = rect.left() + slot * (i as f32) + slot * 0.5;
                let top = base - rect.height() * h;
                painter.line_segment([egui::pos2(x, base), egui::pos2(x, top)], s);
            }
            painter.line_segment(
                [egui::pos2(rect.left(), base), egui::pos2(rect.right(), base)],
                s,
            );
        }
        NavIcon::Settings => {
            // A gear approximation: a ring + four radial ticks.
            let c = rect.center();
            let r = rect.width() * 0.28;
            painter.circle_stroke(c, r, s);
            let tick = rect.width() * 0.16;
            for (dx, dy) in [(0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0)] {
                let a = egui::pos2(c.x + dx * r, c.y + dy * r);
                let b = egui::pos2(c.x + dx * (r + tick), c.y + dy * (r + tick));
                painter.line_segment([a, b], s);
            }
        }
        NavIcon::Decide => {
            // A decision diamond (rotated square) with a small check inside.
            let c = rect.center();
            let r = rect.width() * 0.42;
            let pts = vec![
                egui::pos2(c.x, c.y - r),
                egui::pos2(c.x + r, c.y),
                egui::pos2(c.x, c.y + r),
                egui::pos2(c.x - r, c.y),
            ];
            painter.add(egui::Shape::closed_line(pts, s));
            painter.line_segment(
                [
                    egui::pos2(c.x - r * 0.4, c.y),
                    egui::pos2(c.x - r * 0.05, c.y + r * 0.35),
                ],
                s,
            );
            painter.line_segment(
                [
                    egui::pos2(c.x - r * 0.05, c.y + r * 0.35),
                    egui::pos2(c.x + r * 0.45, c.y - r * 0.35),
                ],
                s,
            );
        }
        NavIcon::Quantize => {
            // A "compress" glyph: two arrows pointing toward a middle bar.
            let c = rect.center();
            let w = rect.width() * 0.34;
            painter.line_segment(
                [egui::pos2(c.x - w, c.y), egui::pos2(c.x + w, c.y)],
                s,
            );
            // Top arrow pointing down to the bar.
            let ty = rect.top() + 1.0;
            painter.line_segment([egui::pos2(c.x, ty), egui::pos2(c.x, c.y - 3.0)], s);
            painter.line_segment(
                [egui::pos2(c.x - 3.0, c.y - 6.0), egui::pos2(c.x, c.y - 3.0)],
                s,
            );
            painter.line_segment(
                [egui::pos2(c.x + 3.0, c.y - 6.0), egui::pos2(c.x, c.y - 3.0)],
                s,
            );
            // Bottom arrow pointing up to the bar.
            let by = rect.bottom() - 1.0;
            painter.line_segment([egui::pos2(c.x, by), egui::pos2(c.x, c.y + 3.0)], s);
            painter.line_segment(
                [egui::pos2(c.x - 3.0, c.y + 6.0), egui::pos2(c.x, c.y + 3.0)],
                s,
            );
            painter.line_segment(
                [egui::pos2(c.x + 3.0, c.y + 6.0), egui::pos2(c.x, c.y + 3.0)],
                s,
            );
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

// --- Phase-4 Status / Settings helpers ------------------------------------

/// The KV-dtype option lists (mirror Settings.tsx). `_OPT` prepends `""`
/// (rendered "— inherit —") for the optional per-channel K/V overrides.
const KV_DTYPES: [&str; 8] = ["f32", "q8_0", "q4_0", "tq1", "tq2", "tq4", "tq8", "nvfp4"];
const KV_DTYPES_OPT: [&str; 9] = [
    "", "f32", "q8_0", "q4_0", "tq1", "tq2", "tq4", "tq8", "nvfp4",
];
/// UI theme options.
const THEMES: [&str; 3] = ["system", "dark", "light"];

/// Push `v` onto a sparkline ring buffer, evicting the oldest sample once it
/// exceeds `METRICS_WINDOW`. `VecDeque` keeps both ends O(1), so the buffer
/// never grows past the window regardless of how long a session runs.
fn push_capped(buf: &mut VecDeque<f64>, v: f64) {
    buf.push_back(v);
    while buf.len() > METRICS_WINDOW {
        buf.pop_front();
    }
}

/// Read a boolean field off a JSON object (the `set_config` / `apply_profile`
/// change report), defaulting to false.
fn flag(v: &serde_json::Value, key: &str) -> bool {
    v.get(key).and_then(|b| b.as_bool()).unwrap_or(false)
}

/// Human-readable server uptime (mirrors Status.tsx `formatUptime`).
fn fmt_uptime(s: u64) -> String {
    if s < 60 {
        return format!("{s}s");
    }
    let m = s / 60;
    if m < 60 {
        return format!("{}m {}s", m, s % 60);
    }
    let h = m / 60;
    format!("{}h {}m", h, m % 60)
}

/// GPU vendor → dispatch-backend badge (Intel/AMD ⇒ SYCL, NVIDIA ⇒ CUDA).
fn gpu_badge(vendor: &str) -> &'static str {
    match vendor {
        "nvidia" => "CUDA",
        "intel" | "amd" | "gpu" => "SYCL",
        _ => "GPU",
    }
}

// --- config <-> draft round-trip ------------------------------------------

/// Read a string field `config[section][key]`, treating a missing value or
/// JSON null as `default` (so a null `model.path` reads as "").
fn cfg_str(cfg: &serde_json::Value, section: &str, key: &str, default: &str) -> String {
    cfg.get(section)
        .and_then(|s| s.get(key))
        .and_then(|v| v.as_str())
        .unwrap_or(default)
        .to_string()
}

fn cfg_u64(cfg: &serde_json::Value, section: &str, key: &str, default: u64) -> u64 {
    cfg.get(section)
        .and_then(|s| s.get(key))
        .and_then(|v| v.as_u64())
        .unwrap_or(default)
}

fn cfg_bool(cfg: &serde_json::Value, section: &str, key: &str, default: bool) -> bool {
    cfg.get(section)
        .and_then(|s| s.get(key))
        .and_then(|v| v.as_bool())
        .unwrap_or(default)
}

/// Extract the editable form fields from a loaded config object. Defaults
/// mirror the server-side config defaults so a slim config still populates a
/// sensible form.
fn draft_from_config(cfg: &serde_json::Value) -> SettingsDraft {
    let cors = cfg
        .get("server")
        .and_then(|s| s.get("cors_origins"))
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        })
        .unwrap_or_default();
    SettingsDraft {
        bind_addr: cfg_str(cfg, "server", "bind_addr", "127.0.0.1"),
        port: cfg_u64(cfg, "server", "port", 11434).to_string(),
        api_key: cfg_str(cfg, "server", "api_key", ""),
        max_loaded_models: cfg_u64(cfg, "server", "max_loaded_models", 1) as u32,
        cors_origins: cors,
        model_path: cfg_str(cfg, "model", "path", ""),
        model_hub: cfg_str(cfg, "model", "hub", ""),
        chat_template: cfg_str(cfg, "model", "chat_template", "auto"),
        ctx_size: cfg_u64(cfg, "inference", "ctx_size", 8192),
        batch_size: cfg_u64(cfg, "inference", "batch_size", 512) as u32,
        threads: cfg_u64(cfg, "inference", "threads", 0) as u32,
        kv_dtype: cfg_str(cfg, "inference", "kv_dtype", "f32"),
        k_dtype: cfg_str(cfg, "inference", "k_dtype", ""),
        v_dtype: cfg_str(cfg, "inference", "v_dtype", ""),
        flash_attention: cfg_bool(cfg, "inference", "flash_attention", false),
        speculative_ngram: cfg_bool(cfg, "inference", "speculative_ngram", false),
        prefix_cache: cfg_bool(cfg, "inference", "prefix_cache", false),
        keep_quant_raw: cfg_bool(cfg, "inference", "keep_quant_raw", false),
        theme: cfg_str(cfg, "ui", "theme", "system"),
        font_size: cfg_u64(cfg, "ui", "font_size", 14) as u32,
        code_theme: cfg_str(cfg, "ui", "code_theme", ""),
    }
}

/// Write the edited form fields back over a config object, in place, leaving
/// every unmodeled key untouched. An empty path/hub/K/V override is written as
/// JSON null (matching how Settings.tsx clears those optional fields). The
/// `port` field parses to a u16; a mid-edit unparseable value is left as-is on
/// disk rather than clobbered to 0.
fn apply_draft_to_config(cfg: &mut serde_json::Value, d: &SettingsDraft) {
    use serde_json::Value;
    let str_or_null = |s: &str| {
        if s.trim().is_empty() {
            Value::Null
        } else {
            Value::String(s.trim().to_string())
        }
    };
    let cors: Vec<Value> = d
        .cors_origins
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| Value::String(s.to_string()))
        .collect();

    set_path(cfg, "server", "bind_addr", Value::String(d.bind_addr.clone()));
    if let Ok(p) = d.port.trim().parse::<u16>() {
        set_path(cfg, "server", "port", Value::from(p));
    }
    set_path(cfg, "server", "api_key", Value::String(d.api_key.clone()));
    set_path(
        cfg,
        "server",
        "max_loaded_models",
        Value::from(d.max_loaded_models),
    );
    set_path(cfg, "server", "cors_origins", Value::Array(cors));

    set_path(cfg, "model", "path", str_or_null(&d.model_path));
    set_path(cfg, "model", "hub", str_or_null(&d.model_hub));
    set_path(
        cfg,
        "model",
        "chat_template",
        Value::String(d.chat_template.clone()),
    );

    set_path(cfg, "inference", "ctx_size", Value::from(d.ctx_size));
    set_path(cfg, "inference", "batch_size", Value::from(d.batch_size));
    set_path(cfg, "inference", "threads", Value::from(d.threads));
    set_path(
        cfg,
        "inference",
        "kv_dtype",
        Value::String(d.kv_dtype.clone()),
    );
    set_path(cfg, "inference", "k_dtype", str_or_null(&d.k_dtype));
    set_path(cfg, "inference", "v_dtype", str_or_null(&d.v_dtype));
    set_path(
        cfg,
        "inference",
        "flash_attention",
        Value::Bool(d.flash_attention),
    );
    set_path(
        cfg,
        "inference",
        "speculative_ngram",
        Value::Bool(d.speculative_ngram),
    );
    set_path(
        cfg,
        "inference",
        "prefix_cache",
        Value::Bool(d.prefix_cache),
    );
    set_path(
        cfg,
        "inference",
        "keep_quant_raw",
        Value::Bool(d.keep_quant_raw),
    );

    set_path(cfg, "ui", "theme", Value::String(d.theme.clone()));
    set_path(cfg, "ui", "font_size", Value::from(d.font_size));
    set_path(cfg, "ui", "code_theme", Value::String(d.code_theme.clone()));
}

/// Set `cfg[section][key] = val`, creating the section object if the config (or
/// that section) isn't an object yet.
fn set_path(cfg: &mut serde_json::Value, section: &str, key: &str, val: serde_json::Value) {
    if !cfg.is_object() {
        *cfg = serde_json::Value::Object(serde_json::Map::new());
    }
    let obj = cfg.as_object_mut().expect("config is an object");
    let sec = obj
        .entry(section.to_string())
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    if !sec.is_object() {
        *sec = serde_json::Value::Object(serde_json::Map::new());
    }
    sec.as_object_mut()
        .expect("section is an object")
        .insert(key.to_string(), val);
}

// --- Status-view widgets --------------------------------------------------

/// A left-labelled key/value row for the Status cards.
fn kv_row(ui: &mut egui::Ui, k: &str, v: &str) {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(170.0, 16.0), Sense::hover());
        ui.painter().text(
            rect.left_center(),
            Align2::LEFT_CENTER,
            k,
            FontId::proportional(12.0),
            MUTED,
        );
        ui.label(RichText::new(v).color(TEXT).small());
    });
}

/// A tuner-winner row: label + value + an auto-applied / stored pill, or "—"
/// when the sweep hasn't produced a winner.
fn winner_row(ui: &mut egui::Ui, label: &str, value: Option<String>, auto_applied: bool) {
    ui.horizontal(|ui| {
        let (rect, _) = ui.allocate_exact_size(egui::vec2(170.0, 16.0), Sense::hover());
        ui.painter().text(
            rect.left_center(),
            Align2::LEFT_CENTER,
            label,
            FontId::proportional(12.0),
            MUTED,
        );
        match value {
            Some(v) => {
                ui.label(RichText::new(v).color(TEXT).small());
                if auto_applied {
                    pill(ui, "auto-applied", HEALTH_OK);
                } else {
                    pill(ui, "stored", MUTED);
                }
            }
            None => {
                ui.label(RichText::new("—").color(MUTED).small());
            }
        }
    });
}

/// The compute-backend panel body (CPU / SYCL / CUDA tiers), from
/// `/v1/capabilities`.
fn render_backends(ui: &mut egui::Ui, b: &rustllama_client::Backends) {
    // CPU — always the active fallback tier.
    ui.label(RichText::new("CPU").color(MUTED).small());
    ui.label(RichText::new("active").color(HEALTH_OK).strong());
    let simd = if b.cpu.simd_features.is_empty() {
        "scalar (non-x86)".to_string()
    } else {
        format!("SIMD: {}", b.cpu.simd_features.join(", "))
    };
    ui.label(RichText::new(simd).color(MUTED).small());
    ui.label(
        RichText::new(format!(
            "rayon matvec: {}",
            if b.cpu.parallel_matvec { "on (M≥256)" } else { "off" }
        ))
        .color(MUTED)
        .small(),
    );

    // SYCL.
    ui.add_space(8.0);
    ui.label(RichText::new("SYCL (GPU)").color(MUTED).small());
    if b.sycl.available {
        ui.label(
            RichText::new(b.sycl.backend.clone().unwrap_or_else(|| "active".into()))
                .color(HEALTH_OK)
                .strong(),
        );
        ui.label(
            RichText::new(format!(
                "devices: {} · preference: {}",
                b.sycl.device_count, b.sycl.preference
            ))
            .color(MUTED)
            .small(),
        );
        if b.sycl.l0_import_eligible {
            ui.label(
                RichText::new("L0 USM import fast-path eligible")
                    .color(HEALTH_OK)
                    .small(),
            );
        }
    } else {
        ui.label(RichText::new("unavailable").color(MUTED).strong());
        ui.label(RichText::new("no SYCL GPU visible").color(MUTED).small());
    }

    // CUDA.
    ui.add_space(8.0);
    ui.label(RichText::new("CUDA (GPU)").color(MUTED).small());
    match &b.cuda {
        Some(c) if c.available => {
            ui.label(RichText::new("active").color(HEALTH_OK).strong());
            ui.label(
                RichText::new(format!(
                    "{}/{} kernel-ready · driver {}",
                    c.compute_ready,
                    c.device_count,
                    c.driver.clone().unwrap_or_else(|| "n/a".into())
                ))
                .color(MUTED)
                .small(),
            );
        }
        _ => {
            ui.label(RichText::new("unavailable").color(MUTED).strong());
            ui.label(RichText::new("no NVIDIA GPU visible").color(MUTED).small());
        }
    }
}

/// One compute-inventory row: backend badge + label/name on the left, a memory
/// meter + utilization on the right. Every numeric is optional (renders "n/a").
fn inventory_row(
    ui: &mut egui::Ui,
    badge: &str,
    label: &str,
    name: &str,
    used: Option<u64>,
    total: Option<u64>,
    util: Option<f64>,
) {
    egui::Frame::none()
        .fill(ELEVATED2)
        .rounding(Rounding::same(8.0))
        .stroke(Stroke::new(1.0, BORDER))
        .inner_margin(Margin::symmetric(12.0, 8.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        pill(ui, badge, ACCENT);
                        ui.label(RichText::new(label).strong().color(TEXT).small());
                    });
                    ui.label(RichText::new(name).color(MUTED).small());
                });
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    let util_txt = util
                        .map(|u| format!("{u:.0}%"))
                        .unwrap_or_else(|| "n/a".into());
                    ui.label(RichText::new(util_txt).color(TEXT).small());
                    ui.label(RichText::new("util").color(MUTED).small());
                    if let Some(t) = total {
                        ui.add_space(10.0);
                        let mem_txt = match used {
                            Some(u) => format!("{} / {}", fmt_gib(u), fmt_gib(t)),
                            None => format!("{} total", fmt_gib(t)),
                        };
                        ui.label(RichText::new(mem_txt).color(TEXT).small());
                        let frac = used.map(|u| u as f32 / t as f32).unwrap_or(0.0);
                        thin_meter(ui, frac, 80.0);
                    }
                });
            });
        });
    ui.add_space(6.0);
}

/// A small egui_plot line sparkline for a Status live-metrics tile: `label` +
/// the current `value` over a compact filled line chart of the ring-buffer
/// series. `y_max` pins the ceiling for a capped series (ctx_size, max_pending);
/// `None` auto-scales to the series' own peak. Panning / zooming are disabled —
/// it's a read-only glance widget.
fn sparkline(
    ui: &mut egui::Ui,
    id: &str,
    label: &str,
    value: &str,
    series: &VecDeque<f64>,
    color: Color32,
    y_max: Option<f64>,
) {
    ui.allocate_ui(egui::vec2(184.0, 88.0), |ui| {
        ui.vertical(|ui| {
            ui.label(RichText::new(label).color(MUTED).small());
            ui.label(RichText::new(value).color(TEXT).heading());
            let points: Vec<[f64; 2]> = series
                .iter()
                .enumerate()
                .map(|(i, &y)| [i as f64, y])
                .collect();
            let line = egui_plot::Line::new(points).color(color).width(1.5).fill(0.0);
            let mut plot = egui_plot::Plot::new(id.to_string())
                .height(40.0)
                .width(180.0)
                .show_axes([false, false])
                .show_grid([false, false])
                .show_x(false)
                .show_y(false)
                .allow_zoom(false)
                .allow_drag(false)
                .allow_scroll(false)
                .allow_boxed_zoom(false)
                .include_y(0.0);
            if let Some(m) = y_max {
                plot = plot.include_y(m.max(1.0));
            }
            plot.show(ui, |pui| pui.line(line));
        });
    });
}

// --- Settings-view widgets ------------------------------------------------

/// A live / reload / restart section badge (color-coded like Settings.tsx).
fn section_badge(ui: &mut egui::Ui, kind: &str) {
    let color = match kind {
        "live" => HEALTH_OK,
        "reload" => ACCENT,
        "restart" => WARN,
        _ => MUTED,
    };
    pill(ui, kind, color);
}

/// A collapsible Settings section (default-open), its hot-apply `tag` shown as
/// a badge at the top of the body.
fn collapsing_section(
    ui: &mut egui::Ui,
    title: &str,
    tag: &str,
    add: impl FnOnce(&mut egui::Ui),
) {
    egui::CollapsingHeader::new(RichText::new(title).strong().color(TEXT))
        .id_salt(("settings_sec", title))
        .default_open(true)
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                section_badge(ui, tag);
            });
            add(ui);
        });
    ui.add_space(6.0);
}

/// A collapsible read-only Settings section (no hot-apply badge).
fn collapsing_section_ro(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui)) {
    egui::CollapsingHeader::new(RichText::new(title).strong().color(TEXT))
        .id_salt(("settings_sec_ro", title))
        .default_open(true)
        .show(ui, |ui| add(ui));
    ui.add_space(6.0);
}

/// A labelled single-line text field (with an optional hint line).
fn form_text(ui: &mut egui::Ui, label: &str, hint: &str, value: &mut String) {
    ui.add_space(6.0);
    ui.label(RichText::new(label).color(TEXT).small().strong());
    ui.add(egui::TextEdit::singleline(value).desired_width(f32::INFINITY));
    if !hint.is_empty() {
        ui.label(RichText::new(hint).color(DISABLED).small());
    }
}

/// A labelled numeric drag field, clamped to `range` (the range is f64 so a
/// single helper serves both the u32 and u64 fields).
fn form_num<N: egui::emath::Numeric>(
    ui: &mut egui::Ui,
    label: &str,
    hint: &str,
    value: &mut N,
    range: std::ops::RangeInclusive<f64>,
) {
    ui.add_space(6.0);
    ui.label(RichText::new(label).color(TEXT).small().strong());
    ui.add(egui::DragValue::new(value).range(range));
    if !hint.is_empty() {
        ui.label(RichText::new(hint).color(DISABLED).small());
    }
}

/// A labelled checkbox field.
fn form_bool(ui: &mut egui::Ui, label: &str, hint: &str, value: &mut bool) {
    ui.add_space(6.0);
    ui.checkbox(value, RichText::new(label).color(TEXT).small());
    if !hint.is_empty() {
        ui.label(RichText::new(hint).color(DISABLED).small());
    }
}

/// A labelled dropdown over `options`; the empty option renders "— inherit —"
/// (used by the optional per-channel K/V dtype overrides).
fn form_combo(
    ui: &mut egui::Ui,
    id: &str,
    label: &str,
    hint: &str,
    value: &mut String,
    options: &[&str],
) {
    ui.add_space(6.0);
    ui.label(RichText::new(label).color(TEXT).small().strong());
    egui::ComboBox::from_id_salt(id)
        .width(220.0)
        .selected_text(if value.is_empty() {
            "— inherit —".to_string()
        } else {
            value.clone()
        })
        .show_ui(ui, |ui| {
            for opt in options {
                let disp = if opt.is_empty() { "— inherit —" } else { opt };
                ui.selectable_value(value, opt.to_string(), disp);
            }
        });
    if !hint.is_empty() {
        ui.label(RichText::new(hint).color(DISABLED).small());
    }
}

/// The Profiles section: a picker + Apply. Applying merges the chosen profile's
/// SPARSE overrides on the server; blocked while the form has unsaved edits
/// (the merge would otherwise discard them).
fn settings_profiles_section(
    ui: &mut egui::Ui,
    profiles: &[String],
    selected: &mut String,
    dirty: bool,
    saving: bool,
    actions: &mut Vec<Action>,
) {
    egui::CollapsingHeader::new(RichText::new("Profiles").strong().color(TEXT))
        .id_salt(("settings_sec", "Profiles"))
        .default_open(true)
        .show(ui, |ui| {
            ui.label(
                RichText::new(
                    "Quick-switch between named [[profiles]] in config.toml. Applying merges the \
                     profile's SPARSE overrides into the on-disk config — only the sections it \
                     defines change.",
                )
                .color(MUTED)
                .small(),
            );
            ui.add_space(6.0);
            if profiles.is_empty() {
                ui.label(
                    RichText::new("No profiles defined. Add [[profiles]] blocks to config.toml.")
                        .color(MUTED)
                        .small(),
                );
            } else {
                ui.horizontal(|ui| {
                    egui::ComboBox::from_id_salt("profile_combo")
                        .width(220.0)
                        .selected_text(if selected.is_empty() {
                            "— select —".to_string()
                        } else {
                            selected.clone()
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(selected, String::new(), "— select —");
                            for p in profiles {
                                ui.selectable_value(selected, p.clone(), p.as_str());
                            }
                        });
                    let can = !selected.is_empty() && !dirty && !saving;
                    if ui
                        .add_enabled(can, egui::Button::new("Apply profile"))
                        .clicked()
                    {
                        actions.push(Action::ApplyProfile(selected.clone()));
                    }
                });
                if dirty {
                    ui.label(
                        RichText::new("Discard or save your current edits before applying a profile.")
                            .color(WARN)
                            .small(),
                    );
                }
            }
        });
    ui.add_space(6.0);
}

// --- Phase-5 Decide / Quantize / history / image helpers ------------------

/// The quantize target dtypes the pipeline supports, ordered lowest→highest
/// bpw so the dropdown reads smallest-to-biggest output (mirrors Quantize.tsx's
/// `TARGETS`). `(dtype name accepted by `parse_dtype_name`, display label)`.
const QUANT_TARGETS: &[(&str, &str)] = &[
    ("iq1_s", "IQ1_S — 1.56 bpw"),
    ("iq1_m", "IQ1_M — 1.75 bpw"),
    ("tq1_0", "TQ1_0 — 1.69 bpw"),
    ("tq2_0", "TQ2_0 — 2.0 bpw"),
    ("iq2_xxs", "IQ2_XXS — 2.06 bpw"),
    ("iq2_xs", "IQ2_XS — 2.31 bpw"),
    ("iq2_s", "IQ2_S — 2.56 bpw"),
    ("q2_k", "Q2_K — 2.625 bpw"),
    ("iq3_xxs", "IQ3_XXS — 3.06 bpw"),
    ("iq3_s", "IQ3_S — 3.44 bpw"),
    ("q3_k", "Q3_K — 3.44 bpw"),
    ("iq4_nl", "IQ4_NL — 4.5 bpw"),
    ("iq4_xs", "IQ4_XS — 4.25 bpw"),
    ("q4_0", "Q4_0 — 4.5 bpw"),
    ("q4_1", "Q4_1 — 5.0 bpw"),
    ("q4_k", "Q4_K (recommended) — 4.5 bpw"),
    ("q5_0", "Q5_0 — 5.5 bpw"),
    ("q5_1", "Q5_1 — 6.0 bpw"),
    ("q5_k", "Q5_K — 5.5 bpw"),
    ("q6_k", "Q6_K — 6.5 bpw"),
    ("q8_0", "Q8_0 — 8.5 bpw"),
    ("q8_1", "Q8_1 — 9.0 bpw"),
    ("q8_k", "Q8_K — 9.125 bpw"),
    ("bf16", "BF16 — 16 bpw"),
    ("f16", "F16 — 16 bpw"),
    ("f32", "F32 (no quantization) — 32 bpw"),
];

/// APEX mixed-precision tiers (mirrors Quantize.tsx's `APEX_TIERS`). `""` = none.
const APEX_TIERS: &[(&str, &str)] = &[
    ("", "(none) — uniform target across all tensors"),
    ("i-quality", "I-Quality — routed Q4_K/Q6_K, shared Q8_0, attn Q6_K"),
    ("quality", "Quality — routed Q3_K/Q5_K, shared Q8_0, attn Q6_K"),
    ("balanced", "Balanced — routed Q3_K/Q4_K, shared Q6_K, attn Q5_K"),
    ("mini", "Mini — routed Q2_K/Q4_K, shared Q5_K, attn Q5_K"),
    ("nano", "Nano — routed Q2_K, shared Q4_K, attn Q4_K"),
];

/// Look up the display label for the selected quantize target (falls back to
/// the raw name if it isn't in the table).
fn quant_target_label(name: &str) -> &str {
    QUANT_TARGETS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, l)| *l)
        .unwrap_or(name)
}

/// Look up the display label for the selected APEX tier.
fn apex_tier_label(name: &str) -> &str {
    APEX_TIERS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, l)| *l)
        .unwrap_or("(none)")
}

/// A dynamic add/remove list of single-line string fields (Decide options /
/// levels). Blank rows are tolerated (filtered out before the request). `noun`
/// labels the add button + remove tooltips.
fn string_list_editor(ui: &mut egui::Ui, items: &mut Vec<String>, noun: &str) {
    let mut remove: Option<usize> = None;
    for (i, item) in items.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.add(
                egui::TextEdit::singleline(item)
                    .desired_width(ui.available_width() - 34.0)
                    .hint_text(format!("{noun} {}", i + 1)),
            );
            if ui
                .add(egui::Button::new(RichText::new("✕").color(DANGER)).small())
                .on_hover_text(format!("Remove this {noun}"))
                .clicked()
            {
                remove = Some(i);
            }
        });
        ui.add_space(2.0);
    }
    if let Some(i) = remove {
        // Keep at least one row so the list is never empty to edit.
        if items.len() > 1 {
            items.remove(i);
        } else if let Some(s) = items.get_mut(0) {
            s.clear();
        }
    }
    if ui
        .button(RichText::new(format!("＋ Add {noun}")).small())
        .clicked()
    {
        items.push(String::new());
    }
}

/// Render a Decide result: winner header + a "calibrated"/"raw" badge, then a
/// probability bar per option with the winner highlighted (mirrors Decide.tsx).
fn render_decide_result(
    ui: &mut egui::Ui,
    r: &DecideResult,
    tab: DecideTab,
    labels: &[String],
) {
    // Header line + calibration badge.
    ui.horizontal(|ui| {
        let header = match tab {
            DecideTab::Boolean => {
                let p = r.probability.unwrap_or(0.0);
                let yes = r.value.as_bool().unwrap_or(p >= 0.5);
                format!("{} ({:.1}%)", if yes { "YES" } else { "NO" }, p * 100.0)
            }
            _ => r
                .value
                .as_str()
                .map(|s| s.to_string())
                .unwrap_or_else(|| r.value.to_string()),
        };
        ui.label(RichText::new(header).strong().color(TEXT).size(15.0));
        if tab == DecideTab::Score {
            if let Some(sc) = r.score {
                ui.label(RichText::new(format!("expected {sc:.2}")).color(MUTED).small());
            }
        }
        if r.calibrated {
            pill(ui, "calibrated", HEALTH_OK);
        } else {
            pill(ui, "raw confidence", MUTED);
        }
        if r.declined {
            pill(ui, "declined", WARN);
        }
    });
    ui.add_space(10.0);

    // Bars: choice/score come back as an array; boolean is a single P(yes)
    // expanded to yes/no.
    let bars: Vec<(String, f32)> = match tab {
        DecideTab::Boolean => {
            let p = r.probability.unwrap_or(0.0);
            vec![("yes".to_string(), p), ("no".to_string(), 1.0 - p)]
        }
        _ => {
            let probs = r.probabilities.clone().unwrap_or_default();
            probs
                .iter()
                .enumerate()
                .map(|(i, &p)| (labels.get(i).cloned().unwrap_or_else(|| format!("#{i}")), p))
                .collect()
        }
    };
    let winner = bars
        .iter()
        .enumerate()
        .max_by(|a, b| a.1 .1.partial_cmp(&b.1 .1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i);
    for (i, (label, p)) in bars.iter().enumerate() {
        let p = *p;
        ui.horizontal(|ui| {
            let is_win = winner == Some(i);
            let cap = RichText::new(label.as_str())
                .color(if is_win { TEXT } else { MUTED })
                .small();
            ui.add_sized([130.0, 16.0], egui::Label::new(cap).truncate());
            // Bar trough + fill.
            let (rect, _) =
                ui.allocate_exact_size(egui::vec2(ui.available_width() - 60.0, 12.0), Sense::hover());
            let rounding = Rounding::same(6.0);
            ui.painter().rect_filled(rect, rounding, FIELD_BG);
            let f = p.clamp(0.0, 1.0);
            if f > 0.0 {
                let fill =
                    Rect::from_min_size(rect.min, egui::vec2(rect.width() * f, rect.height()));
                ui.painter()
                    .rect_filled(fill, rounding, if is_win { ACCENT } else { ELEVATED2 });
            }
            ui.label(
                RichText::new(format!("{:.1}%", p * 100.0))
                    .color(if is_win { TEXT } else { MUTED })
                    .small(),
            );
        });
        ui.add_space(4.0);
    }
}

/// Render the quantize result summary (mirrors Quantize.tsx's result grid).
fn render_quantize_result(ui: &mut egui::Ui, s: &QuantizeSummary) {
    ui.label(RichText::new("Quantize complete").strong().color(HEALTH_OK));
    ui.add_space(8.0);
    kv_row(ui, "Default target", &s.target);
    kv_row(ui, "Block layers detected", &s.n_layers.to_string());
    kv_row(
        ui,
        "Tensors re-encoded",
        &format!("{} of {}", s.tensors_requantized, s.tensors_total),
    );
    kv_row(ui, "Passthrough", &s.tensors_passthrough.to_string());
    kv_row(ui, "Source size", &human_bytes(s.bytes_in));
    let pct = if s.bytes_in > 0 {
        (s.bytes_out as f64 / s.bytes_in as f64) * 100.0
    } else {
        0.0
    };
    kv_row(
        ui,
        "Output size",
        &format!("{} ({pct:.1}% of original)", human_bytes(s.bytes_out)),
    );
    kv_row(ui, "Elapsed", &format!("{:.2}s", s.elapsed_ms / 1000.0));
}

/// One conversation row in the history sidebar. Returns `(load_clicked,
/// delete_clicked)`. The active conversation is highlighted.
fn conversation_row(ui: &mut egui::Ui, c: &ConversationSummary, active: bool) -> (bool, bool) {
    let mut load = false;
    let mut delete = false;
    let fill = if active { SELECT_BG } else { ELEVATED };
    egui::Frame::none()
        .fill(fill)
        .rounding(Rounding::same(7.0))
        .inner_margin(Margin::symmetric(8.0, 6.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                let title = if c.title.trim().is_empty() {
                    "Untitled".to_string()
                } else {
                    c.title.clone()
                };
                let resp = ui.add(
                    egui::Label::new(
                        RichText::new(title).color(if active { TEXT } else { MUTED }).small(),
                    )
                    .truncate()
                    .sense(Sense::click()),
                );
                if resp.clicked() {
                    load = true;
                }
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    if ui
                        .add(egui::Button::new(RichText::new("✕").color(DANGER)).small())
                        .on_hover_text("Delete this conversation")
                        .clicked()
                    {
                        delete = true;
                    }
                });
            });
        });
    (load, delete)
}

/// A staged-image chip in the composer: name + size + a remove (✕) button.
fn image_chip(
    ui: &mut egui::Ui,
    name: &str,
    bytes: u64,
    remove: &mut Option<usize>,
    idx: usize,
) {
    egui::Frame::none()
        .fill(ELEVATED2)
        .rounding(Rounding::same(6.0))
        .stroke(Stroke::new(1.0, BORDER))
        .inner_margin(Margin::symmetric(7.0, 3.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new("🖼").small());
                let short = if name.chars().count() > 20 {
                    let head: String = name.chars().take(18).collect();
                    format!("{head}…")
                } else {
                    name.to_string()
                };
                ui.label(RichText::new(short).color(TEXT).small())
                    .on_hover_text(format!("{name} · {}", human_bytes(bytes)));
                if ui
                    .add(egui::Button::new(RichText::new("✕").color(DANGER)).small())
                    .clicked()
                {
                    *remove = Some(idx);
                }
            });
        });
}

/// A short, single-line conversation title from the first user message.
fn truncate_title(s: &str) -> String {
    let one_line: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() > 48 {
        let head: String = one_line.chars().take(46).collect();
        format!("{head}…")
    } else if one_line.is_empty() {
        "New chat".into()
    } else {
        one_line
    }
}

/// Guess an image MIME type from a file extension for the `data:` URI (the
/// server sniffs the real bytes, but a correct type is polite / cheap).
fn mime_from_path(path: &std::path::Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        Some("gif") => "image/gif",
        _ => "application/octet-stream",
    }
}

/// Suggest an output path beside the source, tagged with the target dtype
/// (e.g. `model.gguf` + `q4_k` → `model.Q4_K.gguf`).
fn suggest_quant_output(src: &std::path::Path, target: &str) -> String {
    let stem = src
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let out = src.with_file_name(format!("{stem}.{}.gguf", target.to_uppercase()));
    out.display().to_string()
}

/// Run the in-process quantize pipeline — the SAME code path as the Tauri
/// `quantize_model` command (`app/src-tauri/src/main.rs`): parse the target
/// dtype, open the source GGUF, build a [`QuantizePlan`] (uniform default +
/// optional APEX rules + optional recipe rules + optional LM-head passthrough),
/// and run `quantize_gguf_to_path`. Returns a display summary or a message.
/// Called ONLY from the [`GuiApp::spawn_quantize`] worker thread — this is
/// CPU-bound and can run for minutes.
fn run_quantize_job(
    input: &str,
    output: &str,
    target: &str,
    apex: Option<&str>,
    recipe: Option<&str>,
    keep_output: bool,
) -> std::result::Result<QuantizeSummary, String> {
    use rustllama_gguf::{
        apex::{build_apex_rules, ApexTier},
        quantize::{quantize_gguf_to_path, QuantizePlan},
        recipe::{parse_dtype_name, parse_recipe_file},
        Gguf,
    };

    let target_dtype =
        parse_dtype_name(target).ok_or_else(|| format!("unknown target dtype {target:?}"))?;
    let src = Gguf::open(input).map_err(|e| format!("open source: {e}"))?;

    let mut plan = QuantizePlan::uniform(target_dtype);
    let n_layers = infer_n_layers(&src);
    if let Some(tier_name) = apex.filter(|t| !t.is_empty()) {
        let tier = ApexTier::parse(tier_name)
            .ok_or_else(|| format!("unknown APEX tier {tier_name:?}"))?;
        plan.add_rules(build_apex_rules(tier, n_layers));
    }
    if let Some(path) = recipe.filter(|p| !p.is_empty()) {
        let rules = parse_recipe_file(path).map_err(|e| format!("recipe parse: {e}"))?;
        plan.add_rules(rules);
    }
    if keep_output {
        // Keep the LM head / output projection at source precision — the
        // usual recommendation for sub-4-bpw targets (llama.cpp's
        // `--leave-output-tensor`). Mirrors the Tauri command.
        plan.passthrough_prefixes.push("output.".into());
        plan.passthrough_prefixes.push("lm_head.".into());
        plan.passthrough_prefixes.push("head.".into());
    }

    let start = std::time::Instant::now();
    let stats =
        quantize_gguf_to_path(&src, output, &plan).map_err(|e| format!("pipeline: {e}"))?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(QuantizeSummary {
        tensors_total: stats.tensors_total,
        tensors_requantized: stats.tensors_requantized,
        tensors_passthrough: stats.tensors_passthrough,
        bytes_in: stats.bytes_in,
        bytes_out: stats.bytes_out,
        elapsed_ms,
        target: target_dtype.as_str().to_string(),
        n_layers,
    })
}

/// Layer-count heuristic mirroring the Tauri command's `infer_n_layers`:
/// `{arch}.block_count` metadata, else the max `blk.N.*` tensor index + 1.
fn infer_n_layers(src: &rustllama_gguf::Gguf) -> usize {
    use rustllama_gguf::MetadataValue;
    if let Some(arch) = src.architecture() {
        let key = format!("{arch}.block_count");
        if let Some(value) = src.metadata_get(&key) {
            if let Some(n) = match value {
                MetadataValue::U32(v) => Some(*v as usize),
                MetadataValue::U64(v) => Some(*v as usize),
                MetadataValue::I32(v) if *v >= 0 => Some(*v as usize),
                MetadataValue::I64(v) if *v >= 0 => Some(*v as usize),
                _ => None,
            } {
                return n;
            }
        }
    }
    let mut max_idx: Option<usize> = None;
    for t in src.tensors() {
        if let Some(rest) = t.name.strip_prefix("blk.") {
            if let Some(dot) = rest.find('.') {
                if let Ok(idx) = rest[..dot].parse::<usize>() {
                    max_idx = Some(max_idx.map_or(idx, |m| m.max(idx)));
                }
            }
        }
    }
    max_idx.map_or(0, |i| i + 1)
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
