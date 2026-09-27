//! Full-screen ncurses-style chat TUI for `rustllama chat --tui`.
//!
//! A ratatui + crossterm terminal UI over the SAME HTTP client + SSE
//! streaming path the line REPL uses (`Client::chat_stream` →
//! `ChatEvent`). Layout: a status bar (model / server / tok-s), a
//! scrollable transcript pane, and an input box. Tokens stream into the
//! transcript live while the input stays responsive.
//!
//! Concurrency: a dedicated OS thread reads terminal input events onto a
//! channel, and each chat turn is consumed by a spawned tokio task that
//! forwards deltas onto another channel — so the async render loop never
//! blocks on `event::poll`, and streaming works on any tokio runtime.

use std::io::{self};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use futures::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::{
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph};
use rustllama_client::{ChatEvent, ChatMessage, ChatRequest, Client};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};

/// One transcript turn.
struct Turn {
    role: String, // "user" | "assistant" | "error" | "info"
    content: String,
}

/// Messages the per-turn streaming task forwards to the render loop.
enum StreamMsg {
    Delta(String),
    Usage(String),
    Done,
    Err(String),
    /// An informational line to show as a transcript turn (e.g. the result
    /// of an async `/model list` / `/model load` command).
    Info(String),
    /// A background model op succeeded and promoted `id` to the server
    /// default — route this chat to it too. Carries a note to display.
    SetModel {
        id: String,
        note: String,
    },
}

/// TUI state.
struct App {
    system: Option<String>,
    turns: Vec<Turn>,
    input: String,
    scroll: u16,
    follow: bool,
    streaming: bool,
    model_label: String,
    model_id: String,
    server_url: String,
    status: String,
    // sampling knobs (mirrors the REPL defaults + /commands)
    temperature: f32,
    max_tokens: u32,
    top_p: f32,
    top_k: u32,
    repeat_penalty: f32,
    should_quit: bool,
}

impl App {
    fn last_assistant_mut(&mut self) -> Option<&mut Turn> {
        self.turns.iter_mut().rev().find(|t| t.role == "assistant")
    }

    /// Build the OpenAI-style message list from the transcript + system.
    fn messages(&self) -> Vec<ChatMessage> {
        let mut m = Vec::new();
        if let Some(sys) = &self.system {
            m.push(ChatMessage {
                role: "system".into(),
                content: sys.clone(),
            });
        }
        for t in &self.turns {
            // Skip the empty assistant placeholder for the turn currently
            // being generated (it has no content yet), so we send only
            // real history + the new user message.
            if t.role == "user" || (t.role == "assistant" && !t.content.is_empty()) {
                m.push(ChatMessage {
                    role: t.role.clone(),
                    content: t.content.clone(),
                });
            }
        }
        m
    }

    fn request(&self) -> ChatRequest {
        ChatRequest {
            model: self.model_label.clone(),
            messages: self.messages(),
            temperature: Some(self.temperature),
            top_p: Some(self.top_p),
            top_k: Some(self.top_k),
            max_tokens: Some(self.max_tokens),
            repeat_penalty: Some(self.repeat_penalty),
            stream: true,
            stream_options: Some(rustllama_client::StreamOptions {
                include_usage: true,
            }),
        }
    }
}

pub async fn run(
    config_path: &std::path::Path,
    system: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
    initial: Vec<ChatMessage>,
) -> Result<()> {
    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let server_url = crate::resolve_base_url(&cfg, base_url.as_deref());
    let client = Arc::new(Client::new(server_url.as_str())?);

    // Fail fast if no server is reachable (same contract as the REPL).
    let model_id = match client.healthz().await {
        Ok(v) => v
            .get("model_id")
            .and_then(|s| s.as_str())
            .unwrap_or("(unknown)")
            .to_string(),
        Err(e) => anyhow::bail!(
            "no rustllama server reachable at {server_url}: {e}\n\
             run `rustllama serve` in another terminal first."
        ),
    };

    // Terminal setup (alternate screen + raw mode).
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let res = event_loop(
        &mut terminal,
        client,
        server_url,
        model_id,
        system,
        model,
        initial,
    )
    .await;

    // Restore terminal on every path.
    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();
    terminal.show_cursor().ok();
    res
}

async fn event_loop<B: Backend>(
    terminal: &mut Terminal<B>,
    client: Arc<Client>,
    server_url: String,
    model_id: String,
    system: Option<String>,
    model: Option<String>,
    initial: Vec<ChatMessage>,
) -> Result<()> {
    // Seed a resumed transcript (`chat --resume`): a leading system
    // message fills the system slot when `--system` wasn't given; the
    // rest become visible transcript turns so the next turn continues on.
    let mut system = system;
    let mut turns: Vec<Turn> = Vec::new();
    for m in initial {
        if m.role == "system" {
            if system.is_none() {
                system = Some(m.content);
            }
        } else {
            turns.push(Turn {
                role: m.role,
                content: m.content,
            });
        }
    }
    let mut app = App {
        system,
        turns,
        input: String::new(),
        scroll: 0,
        follow: true,
        streaming: false,
        model_label: model.unwrap_or_default(),
        model_id,
        server_url,
        status: "ready".into(),
        temperature: 0.7,
        max_tokens: 512,
        top_p: 0.95,
        top_k: 40,
        repeat_penalty: 1.1,
        should_quit: false,
    };

    // Dedicated blocking input reader → channel, so the async loop never
    // blocks on `event::poll`. A stop flag lets it exit on teardown.
    let (ev_tx, mut ev_rx): (UnboundedSender<Event>, UnboundedReceiver<Event>) =
        unbounded_channel();
    let stop = Arc::new(AtomicBool::new(false));
    let stop_reader = stop.clone();
    let reader = std::thread::spawn(move || {
        while !stop_reader.load(Ordering::Relaxed) {
            match event::poll(Duration::from_millis(100)) {
                Ok(true) => {
                    if let Ok(ev) = event::read() {
                        if ev_tx.send(ev).is_err() {
                            break;
                        }
                    }
                }
                Ok(false) => {}
                Err(_) => break,
            }
        }
    });

    // Persistent stream channel; each turn's task clones the sender.
    let (stream_tx, mut stream_rx) = unbounded_channel::<StreamMsg>();

    // Initial paint.
    terminal.draw(|f| ui(f, &mut app))?;

    while !app.should_quit {
        let mut dirty = false;
        tokio::select! {
            biased;
            Some(ev) = ev_rx.recv() => {
                if handle_event(&mut app, ev, &client, &stream_tx) {
                    dirty = true;
                }
            }
            Some(msg) = stream_rx.recv() => {
                apply_stream_msg(&mut app, msg);
                dirty = true;
            }
            // Fallback tick so a resize / late paint still refreshes.
            _ = tokio::time::sleep(Duration::from_millis(120)) => {
                dirty = true;
            }
        }
        if dirty {
            terminal.draw(|f| ui(f, &mut app))?;
        }
    }

    // Teardown: signal + join the reader thread.
    stop.store(true, Ordering::Relaxed);
    let _ = reader.join();
    Ok(())
}

/// Returns true when the app state changed (needs a redraw).
fn handle_event(
    app: &mut App,
    ev: Event,
    client: &Arc<Client>,
    stream_tx: &UnboundedSender<StreamMsg>,
) -> bool {
    let key = match ev {
        Event::Key(k) if k.kind == KeyEventKind::Press => k,
        Event::Resize(_, _) => return true,
        _ => return false,
    };
    match (key.code, key.modifiers) {
        (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
            app.should_quit = true;
        }
        (KeyCode::Esc, _) => {
            app.should_quit = true;
        }
        (KeyCode::PageUp, _) => {
            app.follow = false;
            app.scroll = app.scroll.saturating_sub(5);
        }
        (KeyCode::PageDown, _) => {
            app.scroll = app.scroll.saturating_add(5);
        }
        (KeyCode::Up, _) => {
            app.follow = false;
            app.scroll = app.scroll.saturating_sub(1);
        }
        (KeyCode::Down, _) => {
            app.scroll = app.scroll.saturating_add(1);
        }
        (KeyCode::Enter, _) if !app.streaming => {
            let text = app.input.trim().to_string();
            app.input.clear();
            if text.is_empty() {
                return true;
            }
            if let Some(rest) = text.strip_prefix('/') {
                handle_command(app, rest, client, stream_tx);
                return true;
            }
            app.turns.push(Turn {
                role: "user".into(),
                content: text,
            });
            app.turns.push(Turn {
                role: "assistant".into(),
                content: String::new(),
            });
            app.streaming = true;
            app.follow = true;
            app.status = "generating…".into();
            spawn_stream(client.clone(), app.request(), stream_tx.clone());
        }
        (KeyCode::Backspace, _) if !app.streaming => {
            app.input.pop();
        }
        (KeyCode::Char(c), _) if !app.streaming => {
            app.input.push(c);
        }
        _ => return false,
    }
    true
}

fn apply_stream_msg(app: &mut App, msg: StreamMsg) {
    match msg {
        StreamMsg::Delta(t) => {
            if let Some(a) = app.last_assistant_mut() {
                a.content.push_str(&t);
            }
        }
        StreamMsg::Usage(u) => {
            app.status = u;
        }
        StreamMsg::Done => {
            app.streaming = false;
            if app.status.starts_with("generating") {
                app.status = "ready".into();
            }
        }
        StreamMsg::Err(e) => {
            app.streaming = false;
            app.turns.push(Turn {
                role: "error".into(),
                content: e,
            });
            app.status = "error".into();
        }
        StreamMsg::Info(text) => {
            app.turns.push(Turn {
                role: "info".into(),
                content: text,
            });
        }
        StreamMsg::SetModel { id, note } => {
            app.model_label = id;
            app.turns.push(Turn {
                role: "info".into(),
                content: note,
            });
        }
    }
}

fn spawn_stream(client: Arc<Client>, req: ChatRequest, tx: UnboundedSender<StreamMsg>) {
    tokio::spawn(async move {
        match client.chat_stream(req).await {
            Ok(mut stream) => {
                while let Some(ev) = stream.next().await {
                    match ev {
                        Ok(ChatEvent::Content(t)) => {
                            let _ = tx.send(StreamMsg::Delta(t));
                        }
                        Ok(ChatEvent::Usage(u)) => {
                            let decode_ms = u.decode_ms.unwrap_or(0.0);
                            let tps = if decode_ms > 0.0 && u.completion_tokens > 0 {
                                u.completion_tokens as f64 / (decode_ms / 1000.0)
                            } else {
                                0.0
                            };
                            let _ = tx.send(StreamMsg::Usage(format!(
                                "{} tok in {:.1}s ({:.1} tok/s)",
                                u.completion_tokens,
                                decode_ms / 1000.0,
                                tps
                            )));
                        }
                        Ok(ChatEvent::Finish(_)) => break,
                        Ok(ChatEvent::Start) => {}
                        Ok(ChatEvent::Error(e)) => {
                            let _ = tx.send(StreamMsg::Err(e));
                            return;
                        }
                        Err(e) => {
                            let _ = tx.send(StreamMsg::Err(e.to_string()));
                            return;
                        }
                    }
                }
                let _ = tx.send(StreamMsg::Done);
            }
            Err(e) => {
                let _ = tx.send(StreamMsg::Err(e.to_string()));
            }
        }
    });
}

/// In-TUI slash commands. These mirror the `rustllama` CLI verbs — the
/// `/model …` group tracks `rustllama model` and `/chat …` tracks
/// `rustllama chat` history — so users learn one vocabulary. Server-side
/// model ops run on a spawned task and report back via `StreamMsg` so the
/// render loop never blocks on a load.
fn handle_command(
    app: &mut App,
    rest: &str,
    client: &Arc<Client>,
    tx: &UnboundedSender<StreamMsg>,
) {
    let cmd = rest.split_whitespace().next().unwrap_or("");
    let arg = rest[cmd.len()..].trim().to_string();
    match cmd {
        "exit" | "quit" => app.should_quit = true,
        "reset" | "clear" => {
            app.turns.clear();
            app.status = "history cleared".into();
        }
        "system" => {
            app.system = if arg.is_empty() { None } else { Some(arg) };
            app.status = "system prompt updated".into();
        }
        // `/model …` mirrors `rustllama model`.
        "model" => handle_model_command(app, &arg, client, tx),
        // `/chat …` mirrors `rustllama chat` history (sync file store).
        "chat" => handle_chat_command(app, &arg),
        "temp" => set_f32(&mut app.temperature, &arg, &mut app.status, "temp"),
        "top_p" => set_f32(&mut app.top_p, &arg, &mut app.status, "top_p"),
        "repeat_penalty" => set_f32(
            &mut app.repeat_penalty,
            &arg,
            &mut app.status,
            "repeat_penalty",
        ),
        "top_k" => {
            if let Ok(v) = arg.parse() {
                app.top_k = v;
                app.status = format!("top_k = {v}");
            }
        }
        "max_tokens" => {
            if let Ok(v) = arg.parse() {
                app.max_tokens = v;
                app.status = format!("max_tokens = {v}");
            }
        }
        "help" => {
            app.turns.push(Turn {
                role: "info".into(),
                content: "commands (mirror the CLI): /model <id|list|load <ref>|default <ref>|\
                          unload <id>>  ·  /chat <list|save [name]|resume <name>|delete <name>>  ·  \
                          /system <text>  /reset  /temp <f>  /top_p <f>  /top_k <n>  \
                          /repeat_penalty <f>  /max_tokens <n>  /help  /quit   ·   \
                          keys: Enter send · PgUp/PgDn scroll · Esc/Ctrl-C quit"
                    .into(),
            });
        }
        other => {
            app.status = format!("unknown command: /{other} (try /help)");
        }
    }
}

/// `/model …` — the TUI half of `rustllama model`. `list` / `load` /
/// `default` / `unload` touch the server, so they run on a spawned task
/// that reports back via `StreamMsg`; a bare id is a local quick-switch.
fn handle_model_command(
    app: &mut App,
    arg: &str,
    client: &Arc<Client>,
    tx: &UnboundedSender<StreamMsg>,
) {
    let sub = arg.split_whitespace().next().unwrap_or("");
    let subarg = arg[sub.len()..].trim().to_string();
    match sub {
        "" => {
            let shown = if app.model_label.is_empty() {
                "(server default)".to_string()
            } else {
                app.model_label.clone()
            };
            app.status =
                format!("model: {shown}  ·  /model <id|list|load <ref>|default <ref>|unload <id>>");
        }
        "list" => {
            let client = client.clone();
            let tx = tx.clone();
            let active = app.model_label.clone();
            app.status = "listing models…".into();
            tokio::spawn(async move {
                let line = match client.list_models().await {
                    Ok(v) => {
                        let arr = v
                            .get("data")
                            .and_then(|d| d.as_array())
                            .cloned()
                            .unwrap_or_default();
                        if arr.is_empty() {
                            "(no models loaded)".to_string()
                        } else {
                            let mut s = String::from("available models:");
                            for m in &arr {
                                let id = m.get("id").and_then(|x| x.as_str()).unwrap_or("?");
                                let is_default = m
                                    .get("is_default")
                                    .and_then(|b| b.as_bool())
                                    .unwrap_or(false);
                                let marker = if is_default { " (default)" } else { "" };
                                let active_mark =
                                    if id == active || (active.is_empty() && is_default) {
                                        " *"
                                    } else {
                                        ""
                                    };
                                s.push_str(&format!("\n  {id}{marker}{active_mark}"));
                            }
                            s
                        }
                    }
                    Err(e) => format!("[error listing models: {e}]"),
                };
                let _ = tx.send(StreamMsg::Info(line));
            });
        }
        "load" | "default" => {
            if subarg.is_empty() {
                app.status = format!("usage: /model {sub} <hub-ref|path>");
                return;
            }
            let promote = sub == "default";
            let client = client.clone();
            let tx = tx.clone();
            let target = subarg;
            app.status = "loading…".into();
            tokio::spawn(async move {
                let mut params = rustllama_client::LoadModelParams::default();
                match crate::classify_swap_target(&target) {
                    crate::SwapTarget::Hub(h) => params.hub = Some(h),
                    crate::SwapTarget::Path(p) => params.path = Some(p),
                }
                match client.load_model(&params).await {
                    Ok(resp) => {
                        let new_id = resp
                            .get("model_id")
                            .and_then(|s| s.as_str())
                            .unwrap_or("?")
                            .to_string();
                        if promote {
                            match client.set_default_model(&new_id).await {
                                Ok(_) => {
                                    let _ = tx.send(StreamMsg::SetModel {
                                        id: new_id.clone(),
                                        note: format!("loaded `{new_id}` (now default)"),
                                    });
                                }
                                Err(e) => {
                                    let _ = tx.send(StreamMsg::Info(format!(
                                        "loaded `{new_id}` but failed to promote: {e}"
                                    )));
                                }
                            }
                        } else {
                            let _ = tx.send(StreamMsg::Info(format!(
                                "loaded `{new_id}` (not default; /model {new_id} to route here)"
                            )));
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(StreamMsg::Info(format!("load failed: {e}")));
                    }
                }
            });
        }
        "unload" => {
            if subarg.is_empty() {
                app.status = "usage: /model unload <id>".into();
                return;
            }
            let client = client.clone();
            let tx = tx.clone();
            let id = subarg;
            tokio::spawn(async move {
                // Refuse to unload the current default (same as the CLI).
                let is_default = match client.list_models().await {
                    Ok(v) => v
                        .get("data")
                        .and_then(|d| d.as_array())
                        .map(|arr| {
                            arr.iter().any(|m| {
                                m.get("id").and_then(|s| s.as_str()) == Some(id.as_str())
                                    && m.get("is_default")
                                        .and_then(|b| b.as_bool())
                                        .unwrap_or(false)
                            })
                        })
                        .unwrap_or(false),
                    Err(_) => false,
                };
                let line = if is_default {
                    format!("`{id}` is the default model and can't be unloaded; /model default <other> first")
                } else {
                    match client.unload_model(&id).await {
                        Ok(_) => format!("unloaded `{id}`"),
                        Err(e) => format!("error: {e}"),
                    }
                };
                let _ = tx.send(StreamMsg::Info(line));
            });
        }
        // Bare `/model <id>` — quick-switch this chat's routing.
        _ => {
            app.model_label = arg.to_string();
            app.status = format!("model → {arg}");
        }
    }
}

/// `/chat …` — the TUI half of `rustllama chat` history, over the
/// file-backed saved-session store (all sync).
fn handle_chat_command(app: &mut App, arg: &str) {
    let sub = arg.split_whitespace().next().unwrap_or("");
    let subarg = arg[sub.len()..].trim().to_string();
    match sub {
        "" | "list" => match crate::session::list() {
            Ok(entries) if entries.is_empty() => {
                app.turns.push(Turn {
                    role: "info".into(),
                    content: "(no saved sessions)".into(),
                });
            }
            Ok(entries) => {
                let mut s = String::from("saved sessions:");
                for e in entries {
                    s.push_str(&format!(
                        "\n  {}  (model={}, msgs={})",
                        e.name, e.model, e.message_count
                    ));
                }
                app.turns.push(Turn {
                    role: "info".into(),
                    content: s,
                });
            }
            Err(e) => app.status = format!("error listing sessions: {e}"),
        },
        "save" => {
            let name = if subarg.is_empty() {
                format!("session-{}", crate::session_default_name())
            } else {
                subarg
            };
            let msgs = app.messages();
            match crate::session::save(&name, &app.model_label, &msgs) {
                Ok(_) => app.status = format!("saved session `{name}` ({} msgs)", msgs.len()),
                Err(e) => app.status = format!("save failed: {e}"),
            }
        }
        "resume" | "load" => {
            if subarg.is_empty() {
                app.status = "usage: /chat resume <name>".into();
                return;
            }
            match crate::session::load(&subarg) {
                Ok(sess) => {
                    app.turns.clear();
                    for m in sess.messages {
                        if m.role == "system" {
                            app.system = Some(m.content);
                        } else {
                            app.turns.push(Turn {
                                role: m.role,
                                content: m.content,
                            });
                        }
                    }
                    if !sess.model.is_empty() {
                        app.model_label = sess.model;
                    }
                    app.status = format!("resumed `{}`", sess.name);
                }
                Err(e) => app.status = format!("resume failed: {e}"),
            }
        }
        "delete" => {
            if subarg.is_empty() {
                app.status = "usage: /chat delete <name>".into();
                return;
            }
            match crate::session::delete(&subarg) {
                Ok(()) => app.status = format!("deleted session `{subarg}`"),
                Err(e) => app.status = format!("delete failed: {e}"),
            }
        }
        other => {
            app.status = format!("unknown: /chat {other} (list|save|resume|delete)");
        }
    }
}

fn set_f32(slot: &mut f32, arg: &str, status: &mut String, name: &str) {
    if let Ok(v) = arg.parse() {
        *slot = v;
        *status = format!("{name} = {v}");
    }
}

// ---- rendering ----

fn ui(f: &mut Frame, app: &mut App) {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // status bar
            Constraint::Min(3),    // transcript
            Constraint::Length(3), // input box
        ])
        .split(area);

    // --- status bar ---
    let indicator = if app.streaming { "● " } else { "  " };
    let status = Line::from(vec![
        Span::styled(
            " rustllama chat ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled(indicator, Style::default().fg(Color::Green)),
        Span::styled(
            format!(
                "{} ",
                if app.model_label.is_empty() {
                    app.model_id.as_str()
                } else {
                    app.model_label.as_str()
                }
            ),
            Style::default().fg(Color::Yellow),
        ),
        Span::styled(
            format!("@ {}  ", app.server_url),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(
            format!("[{}]", app.status),
            Style::default().fg(Color::Gray),
        ),
    ]);
    f.render_widget(Paragraph::new(status), chunks[0]);

    // --- transcript ---
    let transcript_block = Block::default().borders(Borders::ALL).title(" transcript ");
    let inner = transcript_block.inner(chunks[1]);
    let content_w = inner.width.max(1) as usize;
    let view_h = inner.height.max(1);

    let mut lines: Vec<Line> = Vec::new();
    for t in &app.turns {
        let (prefix, style) = match t.role.as_str() {
            "user" => (
                "You",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            "assistant" => (
                "AI",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
            "error" => (
                "!!",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            _ => ("··", Style::default().fg(Color::DarkGray)),
        };
        let body_style = if t.role == "error" {
            Style::default().fg(Color::Red)
        } else if t.role == "info" {
            Style::default().fg(Color::DarkGray)
        } else {
            Style::default()
        };
        // "Prefix: " then wrapped body; continuation lines indented.
        let head = format!("{prefix}: ");
        let indent = " ".repeat(head.len().min(content_w.saturating_sub(1)));
        let wrapped = wrap_text(&t.content, content_w.saturating_sub(head.len()).max(1));
        if wrapped.is_empty() {
            lines.push(Line::from(vec![Span::styled(head.clone(), style)]));
        }
        for (i, w) in wrapped.iter().enumerate() {
            if i == 0 {
                lines.push(Line::from(vec![
                    Span::styled(head.clone(), style),
                    Span::styled(w.clone(), body_style),
                ]));
            } else {
                lines.push(Line::from(vec![
                    Span::raw(indent.clone()),
                    Span::styled(w.clone(), body_style),
                ]));
            }
        }
        lines.push(Line::from("")); // blank separator between turns
    }

    // Auto-follow: pin scroll to the bottom unless the user scrolled up.
    let total = lines.len() as u16;
    let max_scroll = total.saturating_sub(view_h);
    if app.follow {
        app.scroll = max_scroll;
    } else if app.scroll > max_scroll {
        app.scroll = max_scroll;
        app.follow = app.scroll == max_scroll; // re-pin if we hit bottom
    }

    let transcript = Paragraph::new(Text::from(lines))
        .block(transcript_block)
        .scroll((app.scroll, 0));
    f.render_widget(transcript, chunks[1]);

    // --- input box ---
    let title = if app.streaming {
        " input (generating…) "
    } else {
        " input "
    };
    let input_block = Block::default().borders(Borders::ALL).title(title);
    let input_inner = input_block.inner(chunks[2]);
    let prompt = "> ";
    let input = Paragraph::new(Line::from(vec![
        Span::styled(prompt, Style::default().fg(Color::DarkGray)),
        Span::raw(app.input.as_str()),
    ]))
    .block(input_block);
    f.render_widget(input, chunks[2]);

    // Cursor at the end of the input (only when not streaming).
    if !app.streaming {
        let cx = input_inner.x + (prompt.len() + app.input.chars().count()) as u16;
        let cy = input_inner.y;
        if cx < input_inner.x + input_inner.width {
            f.set_cursor_position((cx, cy));
        }
    }
}

/// Word-wrap `s` to `width` columns (ASCII-ish; splits over-long words).
/// Preserves explicit newlines. Exact line count → exact scroll math.
fn wrap_text(s: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut out = Vec::new();
    for logical in s.split('\n') {
        if logical.is_empty() {
            out.push(String::new());
            continue;
        }
        let mut cur = String::new();
        for word in logical.split(' ') {
            let wlen = word.chars().count();
            if wlen > width {
                // Flush current, then hard-split the long word.
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                let mut chars: Vec<char> = word.chars().collect();
                while chars.len() > width {
                    out.push(chars[..width].iter().collect());
                    chars.drain(..width);
                }
                cur = chars.into_iter().collect();
            } else if cur.is_empty() {
                cur = word.to_string();
            } else if cur.chars().count() + 1 + wlen <= width {
                cur.push(' ');
                cur.push_str(word);
            } else {
                out.push(std::mem::take(&mut cur));
                cur = word.to_string();
            }
        }
        out.push(cur);
    }
    out
}
