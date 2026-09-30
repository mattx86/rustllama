//! The Phase-1 chat app: [`GuiApp`], its async bridge, and the egui layout.

use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

use egui::Color32;
use egui_commonmark::{CommonMarkCache, CommonMarkViewer};
use futures::StreamExt;
use rustllama_client::{
    ChatEvent, ChatMessage, ChatRequest, Client, LoadModelParams, StreamOptions, Usage,
};

// --- Theme tokens ---------------------------------------------------------
// Approximate the app's design tokens (LM-Studio-ish dark). `Color32::from_rgb`
// is const, so these live as module constants and can be used anywhere without
// recomputing. Light theme is a later phase.
const BG: Color32 = Color32::from_rgb(0x17, 0x18, 0x1c); // window / panel bg
const ELEVATED: Color32 = Color32::from_rgb(0x1e, 0x1f, 0x25); // cards, inactive widgets
const ELEVATED2: Color32 = Color32::from_rgb(0x26, 0x28, 0x32); // hovered / strokes
const ACCENT: Color32 = Color32::from_rgb(0x61, 0x72, 0xf3); // indigo
const TEXT: Color32 = Color32::from_rgb(0xe7, 0xe8, 0xee);
const MUTED: Color32 = Color32::from_rgb(0x9a, 0x9c, 0xab);
const FIELD_BG: Color32 = Color32::from_rgb(0x12, 0x13, 0x16); // TextEdit / scroll troughs
const SELECT_BG: Color32 = Color32::from_rgb(0x2a, 0x30, 0x55);
const HEALTH_OK: Color32 = Color32::from_rgb(0x3f, 0xb9, 0x50);
const HEALTH_BAD: Color32 = Color32::from_rgb(0xe5, 0x48, 0x4d);

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
}

/// Messages sent from spawned tokio tasks back to the UI thread over an
/// `std::sync::mpsc` channel. Drained non-blocking each frame (see
/// [`GuiApp::update`]).
enum UiMsg {
    /// `/v1/models` `data[]` ids.
    Models(Vec<String>),
    /// `/healthz` reachable?
    Health(bool),
    /// A streamed chat content delta.
    ChatDelta(String),
    /// The chat stream drained cleanly; carries the final usage block (which,
    /// under `include_usage`, arrives *after* the finish chunk — see
    /// [`GuiApp::spawn_chat`]).
    ChatDone(Option<Usage>),
    /// A chat stream failed.
    ChatError(String),
    /// A model was promoted to the server default.
    ModelLoaded(String),
    /// Generic status/error line for the model bar.
    Err(String),
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

    // --- model state ---
    models: Vec<String>,
    selected_model: Option<String>,
    loaded_model: Option<String>,

    // --- chat state ---
    transcript: Vec<Turn>,
    /// In-progress assistant text, shown live while `streaming`.
    pending: String,
    input: String,
    streaming: bool,

    // --- status ---
    health: Option<bool>,
    last_usage: Option<Usage>,
    status: Option<String>,

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

        let app = Self {
            base_url,
            rt,
            client: Arc::new(client),
            ctx: cc.egui_ctx.clone(),
            tx,
            rx,
            models: Vec::new(),
            selected_model: None,
            loaded_model: None,
            transcript: Vec::new(),
            pending: String::new(),
            input: String::new(),
            streaming: false,
            health: None,
            last_usage: None,
            status: None,
            md_cache: CommonMarkCache::default(),
        };

        // Kick off the initial loads (health probe + model list).
        app.spawn_health();
        app.spawn_refresh_models();
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

    fn spawn_refresh_models(&self) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            match client.list_models().await {
                Ok(v) => {
                    let _ = tx.send(UiMsg::Models(parse_model_ids(&v)));
                }
                Err(e) => {
                    let _ = tx.send(UiMsg::Err(format!("list models: {e}")));
                }
            }
            ctx.request_repaint();
        });
    }

    fn spawn_load(&self, model_id: String) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            // The ids come from `/v1/models` (the already-loaded set), so we
            // treat the id as a hub ref and try `load_model` best-effort — a
            // failure here is usually just "already loaded" and is non-fatal.
            // We then promote the model to the server default so chat targets
            // it. TODO(phase2): a real model manager (pull / load-by-path /
            // unload) with explicit source selection.
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

    fn spawn_unload(&self, model_id: String) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            match client.unload_model(&model_id).await {
                Ok(_) => {
                    // Re-list so the dropped model leaves the combo.
                    if let Ok(v) = client.list_models().await {
                        let _ = tx.send(UiMsg::Models(parse_model_ids(&v)));
                    }
                }
                Err(e) => {
                    let _ = tx.send(UiMsg::Err(format!("unload: {e}")));
                }
            }
            ctx.request_repaint();
        });
    }

    fn spawn_chat(&self, model: String, messages: Vec<ChatMessage>) {
        let (client, tx, ctx) = (self.client.clone(), self.tx.clone(), self.ctx.clone());
        self.rt.spawn(async move {
            let req = ChatRequest {
                model,
                messages,
                temperature: None,
                top_p: None,
                top_k: None,
                max_tokens: None,
                repeat_penalty: None,
                stream: true,
                stream_options: Some(StreamOptions {
                    include_usage: true,
                }),
                // CLARIFY opt-in: let the model pause and ask instead of
                // guessing. Phase 1 renders the question inline (see AskUser).
                allow_clarify: Some(true),
            };

            // Event-ordering NOTE: with `stream_options.include_usage` the
            // server emits the Usage event in a *separate* final SSE chunk that
            // arrives AFTER Finish. So we do NOT commit on Finish — we
            // accumulate `usage` and send `ChatDone` once the stream fully
            // drains (the loop exits after `[DONE]`), which guarantees the
            // usage stats ride along with the completion.
            let mut usage: Option<Usage> = None;
            let mut errored = false;
            match client.chat_stream(req).await {
                Ok(mut stream) => {
                    while let Some(ev) = stream.next().await {
                        match ev {
                            Ok(ChatEvent::Content(t)) => {
                                let _ = tx.send(UiMsg::ChatDelta(t));
                                ctx.request_repaint(); // wake the UI per token
                            }
                            Ok(ChatEvent::Usage(u)) => usage = Some(u),
                            Ok(ChatEvent::AskUser {
                                prompt, options, ..
                            }) => {
                                // Phase 1: fold the clarify question into the
                                // visible stream. TODO(phase2): inline choice
                                // buttons that round-trip the answer.
                                let mut m = format!("\n\n**Clarification needed:** {prompt}");
                                if !options.is_empty() {
                                    m.push_str(&format!("\n\n_Options: {}_", options.join(", ")));
                                }
                                let _ = tx.send(UiMsg::ChatDelta(m));
                                ctx.request_repaint();
                            }
                            Ok(ChatEvent::Error(e)) => {
                                errored = true;
                                let _ = tx.send(UiMsg::ChatError(e));
                                ctx.request_repaint();
                            }
                            // Start / Finish / ToolCalls: nothing to render in
                            // Phase 1 (tool execution is a later phase).
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

    fn send_message(&mut self) {
        if self.streaming {
            return;
        }
        let text = self.input.trim().to_string();
        if text.is_empty() {
            return;
        }
        let model = self.current_model();
        if model.is_empty() {
            self.status = Some("Select and load a model first.".into());
            return;
        }

        self.input.clear();
        self.transcript.push(Turn::user(text));
        self.pending.clear();
        self.last_usage = None;
        self.status = None;
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
        self.spawn_chat(model, messages);
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
            UiMsg::Health(ok) => self.health = Some(ok),
            UiMsg::ChatDelta(t) => self.pending.push_str(&t),
            UiMsg::ChatDone(usage) => {
                if !self.pending.is_empty() {
                    let text = std::mem::take(&mut self.pending);
                    self.transcript.push(Turn::assistant(text));
                }
                self.streaming = false;
                self.last_usage = usage;
            }
            UiMsg::ChatError(e) => {
                // Commit any partial text so it isn't lost, then surface the error.
                if !self.pending.is_empty() {
                    let text = std::mem::take(&mut self.pending);
                    self.transcript.push(Turn::assistant(text));
                }
                self.streaming = false;
                tracing::warn!(target: "rustllama_gui", "chat error: {e}");
                self.status = Some(format!("Error: {e}"));
            }
            UiMsg::ModelLoaded(id) => {
                self.loaded_model = Some(id.clone());
                self.selected_model = Some(id);
                self.status = Some("Model loaded.".into());
                // A load may have pulled in a not-yet-listed model.
                self.spawn_refresh_models();
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
        // 1) Drain the async channel into state. We must NOT block — `try_recv`
        //    pulls whatever arrived since the last frame. (Collect first so the
        //    receiver borrow is released before `handle_msg`, which may itself
        //    spawn tasks.)
        let mut incoming = Vec::new();
        while let Ok(msg) = self.rx.try_recv() {
            incoming.push(msg);
        }
        for msg in incoming {
            self.handle_msg(msg);
        }

        // Actions collected from immediate-mode widgets, applied after the
        // panels close so the spawn helpers get clean access to `self`.
        let mut do_load = false;
        let mut do_unload = false;
        let mut do_refresh = false;
        let mut do_send = false;

        // 2) Top model bar.
        let models = self.models.clone(); // cheap; avoids nested-closure borrows
        egui::TopBottomPanel::top("model_bar")
            .frame(egui::Frame::default().fill(BG).inner_margin(egui::Margin::symmetric(10.0, 8.0)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new("Model").color(MUTED));
                    egui::ComboBox::from_id_salt("model_combo")
                        .width(280.0)
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
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let (col, tip) = match self.health {
                            Some(true) => (HEALTH_OK, "server: healthy"),
                            Some(false) => (HEALTH_BAD, "server: unreachable"),
                            None => (MUTED, "server: unknown"),
                        };
                        let (rect, resp) =
                            ui.allocate_exact_size(egui::vec2(14.0, 14.0), egui::Sense::hover());
                        ui.painter().circle_filled(rect.center(), 5.0, col);
                        resp.on_hover_text(format!("{tip}\n{}", self.base_url));
                        if let Some(s) = &self.status {
                            ui.label(egui::RichText::new(s).color(MUTED).small());
                        }
                    });
                });
            });

        // 3) Bottom composer.
        egui::TopBottomPanel::bottom("composer")
            .frame(egui::Frame::default().fill(BG).inner_margin(egui::Margin::symmetric(10.0, 8.0)))
            .show(ctx, |ui| {
                ui.add_sized(
                    [ui.available_width(), 72.0],
                    egui::TextEdit::multiline(&mut self.input)
                        .hint_text("Message the model…  (Ctrl+Enter to send)")
                        .desired_rows(3),
                );
                // Computed AFTER the TextEdit so this frame's typing is applied.
                let can_send = !self.streaming && !self.input.trim().is_empty();
                ui.horizontal(|ui| {
                    if ui
                        .add_enabled(can_send, egui::Button::new("Send"))
                        .clicked()
                    {
                        do_send = true;
                    }
                    if self.streaming {
                        ui.spinner();
                        ui.label(egui::RichText::new("generating…").color(MUTED));
                    } else if let Some(u) = &self.last_usage {
                        ui.label(egui::RichText::new(format_usage(u)).color(MUTED).small());
                    }
                });
                // Ctrl+Enter sends. Enter alone stays a newline in the multiline
                // field; Ctrl+Enter isn't consumed by the TextEdit so the global
                // input check sees it.
                let ctrl_enter =
                    ui.input(|i| i.key_pressed(egui::Key::Enter) && i.modifiers.ctrl);
                if ctrl_enter && can_send {
                    do_send = true;
                }
            });

        // 4) Central transcript. Bind the fields the closure needs up front as
        //    disjoint borrows (shared `transcript`/`pending`, mut `md_cache`) so
        //    the borrow checker is happy inside the scroll area.
        let transcript = &self.transcript;
        let pending = &self.pending;
        let streaming = self.streaming;
        let cache = &mut self.md_cache;
        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(BG).inner_margin(egui::Margin::same(12.0)))
            .show(ctx, |ui| {
                egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .stick_to_bottom(true)
                    .show(ui, |ui| {
                        if transcript.is_empty() && !streaming {
                            ui.add_space(48.0);
                            ui.vertical_centered(|ui| {
                                ui.label(
                                    egui::RichText::new("Start a conversation")
                                        .heading()
                                        .color(MUTED),
                                );
                            });
                            return;
                        }
                        for turn in transcript {
                            render_turn(ui, cache, turn);
                        }
                        if streaming {
                            ui.add_space(6.0);
                            ui.label(egui::RichText::new("Assistant").strong().color(ACCENT));
                            if pending.is_empty() {
                                ui.label(egui::RichText::new("…").color(MUTED));
                            } else {
                                CommonMarkViewer::new().show(ui, cache, pending);
                            }
                        }
                    });
            });

        // 5) Apply the actions gathered above, now that `self` is free again.
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
        if do_send {
            self.send_message();
        }
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

/// One transcript turn. User/system turns render as plain (wrapped) text in a
/// subtle card; assistant turns render as markdown via `CommonMarkViewer`.
fn render_turn(ui: &mut egui::Ui, cache: &mut CommonMarkCache, turn: &Turn) {
    ui.add_space(6.0);
    match turn.role.as_str() {
        "user" => {
            ui.label(egui::RichText::new("You").strong().color(MUTED));
            egui::Frame::none()
                .fill(ELEVATED)
                .rounding(egui::Rounding::same(8.0))
                .inner_margin(egui::Margin::same(8.0))
                .show(ui, |ui| {
                    ui.label(egui::RichText::new(turn.content.as_str()).color(TEXT));
                });
        }
        "system" => {
            ui.label(egui::RichText::new("System").italics().color(MUTED));
            ui.label(egui::RichText::new(turn.content.as_str()).color(MUTED));
        }
        _ => {
            ui.label(egui::RichText::new("Assistant").strong().color(ACCENT));
            CommonMarkViewer::new().show(ui, cache, turn.content.as_str());
        }
    }
    ui.add_space(4.0);
    ui.separator();
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

/// Install the dark visuals once at startup. Called from [`GuiApp::new`].
fn configure_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    let mut v = egui::Visuals::dark();

    v.override_text_color = Some(TEXT);
    v.panel_fill = BG;
    v.window_fill = BG;
    v.extreme_bg_color = FIELD_BG; // TextEdit / scroll troughs
    v.faint_bg_color = ELEVATED;
    v.hyperlink_color = ACCENT;
    v.window_stroke = egui::Stroke::new(1.0, ELEVATED2);
    v.selection.bg_fill = SELECT_BG;
    v.selection.stroke = egui::Stroke::new(1.0, ACCENT);

    let round = egui::Rounding::same(8.0);

    v.widgets.noninteractive.bg_fill = ELEVATED;
    v.widgets.noninteractive.weak_bg_fill = ELEVATED;
    v.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, ELEVATED2);
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
