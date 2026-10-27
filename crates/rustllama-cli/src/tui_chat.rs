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
use rustllama_client::{
    ChatEvent, ChatMessage, ChatRequest, Client, InferenceStats, StatsTimer, ToolCall, Usage,
};
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// One transcript turn.
struct Turn {
    role: String, // "user" | "assistant" | "error" | "info"
    content: String,
}

/// Messages the per-turn streaming task forwards to the render loop.
enum StreamMsg {
    Delta(String),
    /// Final per-request token counts + timings from the server `usage` block
    /// (requested via `stream_options.include_usage`). Combined with the
    /// client-measured [`StatsTimer`] into the live stats footer.
    Usage(Usage),
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
    /// CLARIFY: the model called `ask_user`. Carries the question + the
    /// selectable options; the render loop stashes them as a pending
    /// selection and the next digit/Enter answers.
    AskUser {
        prompt: String,
        options: Vec<String>,
    },
    /// The model proposed tool calls. With auto-tools ON they're accepted
    /// silently (a note); OFF they become a pending approve/deny prompt.
    ToolCalls {
        calls: Vec<ToolCall>,
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
    // Sampling knobs. `None` = the user has NOT set this field, so the request
    // OMITS it (serde `skip_serializing_if`) and the server applies its
    // per-model recommended sampling (or its generic default). The `/temp`,
    // `/top_p`, `/top_k`, `/repeat_penalty` commands flip the matching field to
    // `Some`, which then overrides the recommendation. `max_tokens` is a hard
    // output cap, not a sampling recommendation, so it is always sent — seeded
    // to a generous context-aware default (`/max_tokens <n>` overrides).
    temperature: Option<f32>,
    max_tokens: u32,
    top_p: Option<f32>,
    top_k: Option<u32>,
    repeat_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    should_quit: bool,
    /// Auto-run tools: when true, proposed tool calls are accepted silently;
    /// when false (default), they raise an approve/deny prompt. Toggled with
    /// Ctrl-T. (The chat itself never executes tools — this only gates the
    /// confirm prompt.)
    auto_tools: bool,
    /// CLARIFY: when true (default), the server may inject the reserved
    /// `ask_user` tool so the model can pause and ask a clarifying question
    /// with selectable options instead of guessing. Toggled with Ctrl-K or
    /// `/clarify`. OFF sends the plain (no-tool) request path.
    clarify_enabled: bool,
    /// CLARIFY: a pending `ask_user` question `(prompt, options)`. While set,
    /// a digit / Enter answers it instead of sending a normal message.
    pending_question: Option<(String, Vec<String>)>,
    /// A pending tool-call approve/deny (auto-tools OFF). Holds the proposed
    /// calls; `y`/Enter approves, `n` denies.
    pending_tools: Option<Vec<ToolCall>>,
    /// Live inference stats for the footer. `stats_timer` is the
    /// client-measured timing for the IN-FLIGHT turn (TTFT / decode wall);
    /// `pending_usage` stashes the server `usage` chunk (which arrives just
    /// before Done); `last_stats` is the finished-turn snapshot shown when
    /// idle. `ctx_size` is the loaded model's context length (from
    /// `/v1/metrics`) for the context-usage readout — `0` when unknown.
    stats_timer: Option<StatsTimer>,
    pending_usage: Option<Usage>,
    last_stats: Option<InferenceStats>,
    ctx_size: u32,
}

impl App {
    fn last_assistant_mut(&mut self) -> Option<&mut Turn> {
        self.turns.iter_mut().rev().find(|t| t.role == "assistant")
    }

    /// Start client-side timing for a new streamed turn and clear the previous
    /// turn's stats. Called from every stream-start site (send, clarify answer,
    /// tool deny) so the TTFT clock starts at request-send.
    fn begin_turn_stats(&mut self) {
        self.stats_timer = Some(StatsTimer::start());
        self.pending_usage = None;
        self.last_stats = None;
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
                // Strip a prior assistant turn's reasoning span before re-sending
                // it as context: Qwen3 (and other reasoning models) expect only
                // the final answer of earlier turns, not the `<think>` scratchpad
                // — feeding the reasoning back drifts the context. Display-only
                // elsewhere; here it shapes the OUTGOING history, not the stored
                // transcript.
                let content = if t.role == "assistant" {
                    rustllama_client::strip_reasoning(&t.content).to_string()
                } else {
                    t.content.clone()
                };
                m.push(ChatMessage {
                    role: t.role.clone(),
                    content,
                });
            }
        }
        m
    }

    fn request(&self) -> ChatRequest {
        ChatRequest {
            model: self.model_label.clone(),
            messages: self.messages(),
            // Send a sampling field ONLY when the user explicitly set it;
            // otherwise pass `None` so it is omitted and the server applies its
            // per-model recommendation. `max_tokens` is always sent (hard cap).
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            max_tokens: Some(self.max_tokens),
            repeat_penalty: self.repeat_penalty,
            presence_penalty: self.presence_penalty,
            seed: None,
            stream: true,
            stream_options: Some(rustllama_client::StreamOptions {
                include_usage: true,
            }),
            // `None` (not `Some(false)`) when off, so the OFF request is
            // wire-identical to the plain path (no `ask_user` tool).
            allow_clarify: self.clarify_enabled.then_some(true),
        }
    }
}

pub async fn run(
    config_path: &std::path::Path,
    system: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
    initial: Vec<ChatMessage>,
    sampling: crate::ChatSampling,
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

    // The loaded model's context length, for the stats footer's context-usage
    // readout. `/v1/metrics.ctx_size` is the live value (engine `n_ctx()`);
    // fall back to the on-disk `[inference].ctx_size`, else 0 (unknown → the
    // footer just omits the context piece). Best-effort — a miss never blocks
    // the chat.
    let ctx_size: u32 = match client.metrics().await {
        Ok(m) if m.ctx_size > 0 => m.ctx_size as u32,
        _ => cfg.inference.ctx_size,
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
        ctx_size,
        sampling,
    )
    .await;

    // Restore terminal on every path.
    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();
    terminal.show_cursor().ok();
    res
}

#[allow(clippy::too_many_arguments)]
async fn event_loop<B: Backend>(
    terminal: &mut Terminal<B>,
    client: Arc<Client>,
    server_url: String,
    model_id: String,
    system: Option<String>,
    model: Option<String>,
    initial: Vec<ChatMessage>,
    ctx_size: u32,
    sampling: crate::ChatSampling,
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
        // Seeded from the `chat` CLI flags; each stays `None` unless its flag
        // was given, so the server's per-model recommendation still applies.
        temperature: sampling.temperature,
        // Generous, ctx-aware (the model's context length fetched above), so
        // normal + long replies — and a reasoning model's `<think>` block —
        // aren't truncated. `/max_tokens <n>` still overrides.
        max_tokens: crate::default_max_tokens(ctx_size),
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        repeat_penalty: sampling.repeat_penalty,
        presence_penalty: sampling.presence_penalty,
        should_quit: false,
        auto_tools: false,
        clarify_enabled: true,
        pending_question: None,
        pending_tools: None,
        stats_timer: None,
        pending_usage: None,
        last_stats: None,
        ctx_size,
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

    // Streaming redraw coalescing. Painting one frame per streamed token floods
    // a terminal, and over SSH that flood is exactly what tears / ghosts the
    // transcript — the model output is fine, it's the UPDATE VOLUME the
    // transport mishandles. So we DRAIN every token already queued into app
    // state and repaint at most once per `REDRAW_MIN_INTERVAL`, while still
    // painting PROMPTLY on anything the user needs to see at once: input edits,
    // the first token of a reply, and every stream BOUNDARY (done / error /
    // info / ask-user / tool-calls). A delta we throttle away sets
    // `pending_redraw`, and a short timer then flushes it so even a stalled
    // stream still shows its last tokens. This changes only HOW OFTEN we
    // repaint, never what is displayed.
    const REDRAW_MIN_INTERVAL: Duration = Duration::from_millis(40);
    let mut last_draw = std::time::Instant::now();
    let mut pending_redraw = false;

    // Initial paint.
    terminal.draw(|f| ui(f, &mut app))?;

    while !app.should_quit {
        let since_draw = last_draw.elapsed();
        // Time until the throttle window reopens (0 once we're past it).
        let throttle_wait = REDRAW_MIN_INTERVAL.saturating_sub(since_draw);
        let mut draw_now = false;

        tokio::select! {
            biased;
            Some(ev) = ev_rx.recv() => {
                // Input / resize: repaint immediately so typing stays crisp.
                if handle_event(&mut app, ev, &client, &stream_tx) {
                    draw_now = true;
                }
            }
            Some(msg) = stream_rx.recv() => {
                // A boundary (done/error/info/ask/tools) must show at once, and
                // so should the FIRST token of a reply (the empty assistant
                // placeholder getting its first text).
                let mut force = stream_msg_is_boundary(&msg)
                    || (matches!(&msg, StreamMsg::Delta(_))
                        && app
                            .last_assistant_mut()
                            .map(|a| a.content.is_empty())
                            .unwrap_or(false));
                apply_stream_msg(&mut app, msg);
                // Coalesce: fold in every token already waiting so one repaint
                // covers them all instead of one-per-token.
                while let Ok(m) = stream_rx.try_recv() {
                    force |= stream_msg_is_boundary(&m);
                    apply_stream_msg(&mut app, m);
                }
                if force || since_draw >= REDRAW_MIN_INTERVAL {
                    draw_now = true;
                } else {
                    pending_redraw = true;
                }
            }
            // Throttle flush: a delta we skipped repaints once the window reopens.
            _ = tokio::time::sleep(throttle_wait), if pending_redraw => {
                draw_now = true;
            }
            // Fallback tick so a resize / late paint / live stats still refresh.
            _ = tokio::time::sleep(Duration::from_millis(120)) => {
                draw_now = true;
            }
        }
        if draw_now {
            terminal.draw(|f| ui(f, &mut app))?;
            last_draw = std::time::Instant::now();
            pending_redraw = false;
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

    // Ctrl-T toggles auto-run-tools at any time (even mid-prompt).
    if key.code == KeyCode::Char('t') && key.modifiers.contains(KeyModifiers::CONTROL) {
        app.auto_tools = !app.auto_tools;
        app.status = if app.auto_tools {
            "auto-tools: ON".into()
        } else {
            "auto-tools: OFF".into()
        };
        return true;
    }
    // Ctrl-K toggles CLARIFY (the model may ask a clarifying question).
    if key.code == KeyCode::Char('k') && key.modifiers.contains(KeyModifiers::CONTROL) {
        app.clarify_enabled = !app.clarify_enabled;
        app.status = if app.clarify_enabled {
            "clarify: ON".into()
        } else {
            "clarify: OFF".into()
        };
        return true;
    }
    // Ctrl-C always quits, even with a selection pending.
    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
        app.should_quit = true;
        return true;
    }

    // A pending CLARIFY question captures input: a digit (1-9) or Enter
    // picks an option, which becomes a user turn + continues the stream.
    if let Some((_, options)) = app.pending_question.clone() {
        match key.code {
            KeyCode::Esc => app.should_quit = true,
            KeyCode::Char(c) if c.is_ascii_digit() && c != '0' => {
                let idx = (c as u8 - b'1') as usize;
                if idx < options.len() {
                    answer_pending_question(app, client, stream_tx, options[idx].clone());
                }
            }
            KeyCode::Enter => {
                if let Some(first) = options.first().cloned() {
                    answer_pending_question(app, client, stream_tx, first);
                }
            }
            _ => return false,
        }
        return true;
    }

    // A pending tool-call confirm (auto-tools OFF): y/a/Enter accepts (the
    // chat has no executor, so this just dismisses), n/d denies + continues.
    if app.pending_tools.is_some() {
        match key.code {
            KeyCode::Esc => app.should_quit = true,
            KeyCode::Char('y') | KeyCode::Char('a') | KeyCode::Enter => {
                app.pending_tools = None;
                app.turns.push(Turn {
                    role: "info".into(),
                    content: "tool call approved (chat has no executor; nothing ran)".into(),
                });
                app.status = "ready".into();
            }
            KeyCode::Char('n') | KeyCode::Char('d') => {
                app.pending_tools = None;
                deny_pending_tools(app, client, stream_tx);
            }
            _ => return false,
        }
        return true;
    }

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
            app.begin_turn_stats();
            app.streaming = true;
            app.follow = true;
            app.status = "generating…".into();
            spawn_stream(client.clone(), app.request(), stream_tx.clone());
        }
        (KeyCode::Backspace, _) if !app.streaming => {
            app.input.pop();
        }
        // Many terminals (and SSH / tmux / some pod consoles) send Ctrl-H
        // (0x08) for the Backspace key; crossterm decodes that as
        // Char('h')+CONTROL. Without this arm it would fall through to the
        // printable-char arm below and TYPE an 'h' instead of deleting.
        (KeyCode::Char('h'), m) if m.contains(KeyModifiers::CONTROL) && !app.streaming => {
            app.input.pop();
        }
        // Ctrl-U clears the line, Ctrl-W deletes the last word — the usual
        // readline edits users expect in an input box.
        (KeyCode::Char('u'), m) if m.contains(KeyModifiers::CONTROL) && !app.streaming => {
            app.input.clear();
        }
        (KeyCode::Char('w'), m) if m.contains(KeyModifiers::CONTROL) && !app.streaming => {
            let keep = app.input.trim_end();
            let cut = keep.rfind(char::is_whitespace).map(|i| i + 1).unwrap_or(0);
            app.input.truncate(cut);
        }
        // Printable text — but NOT control/alt-modified combos, which would
        // otherwise type the bare letter (e.g. an unhandled Ctrl-A → 'a').
        // SHIFT is allowed so capitals still work.
        (KeyCode::Char(c), m)
            if !app.streaming
                && !m.contains(KeyModifiers::CONTROL)
                && !m.contains(KeyModifiers::ALT) =>
        {
            app.input.push(c);
        }
        _ => return false,
    }
    true
}

fn apply_stream_msg(app: &mut App, msg: StreamMsg) {
    match msg {
        StreamMsg::Delta(t) => {
            // The first non-empty delta marks time-to-first-token; each delta
            // bumps the client-side fallback decode count.
            if let Some(timer) = app.stats_timer.as_mut() {
                timer.on_token();
            }
            if let Some(a) = app.last_assistant_mut() {
                a.content.push_str(&t);
            }
        }
        StreamMsg::Usage(u) => {
            // Stash the server's exact token counts; finalized at Done (the
            // usage chunk arrives just before it under include_usage).
            app.pending_usage = Some(u);
        }
        StreamMsg::Done => {
            app.streaming = false;
            // Freeze the finished-turn stats: client-measured TTFT/decode wall
            // + the server usage counts + the model's context size.
            if let Some(timer) = app.stats_timer.take() {
                app.last_stats = Some(timer.snapshot(app.pending_usage.as_ref(), app.ctx_size));
            }
            if app.status.starts_with("generating") {
                app.status = "ready".into();
            }
        }
        StreamMsg::Err(e) => {
            app.streaming = false;
            app.stats_timer = None;
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
        StreamMsg::AskUser { prompt, options } => {
            // Reuse the streaming placeholder (it never got content) as the
            // question turn — so it's part of the transcript once answered —
            // and stash it as a pending selection.
            app.streaming = false;
            drop_empty_assistant(app);
            app.turns.push(Turn {
                role: "assistant".into(),
                content: prompt.clone(),
            });
            app.status = "choose an option (1-9 / Enter)".into();
            app.pending_question = Some((prompt, options));
        }
        StreamMsg::ToolCalls { calls } => {
            app.streaming = false;
            drop_empty_assistant(app);
            let summary = summarize_tool_calls(&calls);
            if app.auto_tools {
                // Accept silently — the chat can't execute tools, so this
                // is just an acknowledgement note.
                app.turns.push(Turn {
                    role: "info".into(),
                    content: format!("auto-accepted tool call: {summary}"),
                });
                app.status = "ready".into();
            } else {
                app.turns.push(Turn {
                    role: "info".into(),
                    content: format!("model proposes tool call: {summary}"),
                });
                app.status = "approve? y = accept / n = deny".into();
                app.pending_tools = Some(calls);
            }
        }
    }
}

/// Whether a stream message is a BOUNDARY that must repaint at once rather
/// than a plain token that can be coalesced with its neighbors. Everything
/// that changes the transcript's structure or ends the stream is a boundary;
/// only `Delta` (a token) and `Usage` (a silent counts stash, always trailed
/// by a `Done`) are coalescable.
fn stream_msg_is_boundary(msg: &StreamMsg) -> bool {
    !matches!(msg, StreamMsg::Delta(_) | StreamMsg::Usage(_))
}

/// Drop a trailing empty assistant placeholder (the one a turn pushes before
/// streaming) when the turn produced no text — e.g. an `ask_user` / tool-call
/// terminus — so the transcript doesn't show a blank "AI:" bubble.
fn drop_empty_assistant(app: &mut App) {
    if let Some(last) = app.turns.last() {
        if last.role == "assistant" && last.content.is_empty() {
            app.turns.pop();
        }
    }
}

/// One-line summary of proposed tool calls for the transcript.
fn summarize_tool_calls(calls: &[ToolCall]) -> String {
    calls
        .iter()
        .map(|c| format!("{}({})", c.name, c.arguments))
        .collect::<Vec<_>>()
        .join(", ")
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
                            // Forward the raw usage block; the render loop folds
                            // it into the client-measured stats footer.
                            let _ = tx.send(StreamMsg::Usage(u));
                        }
                        Ok(ChatEvent::AskUser {
                            prompt, options, ..
                        }) => {
                            let _ = tx.send(StreamMsg::AskUser { prompt, options });
                        }
                        Ok(ChatEvent::ToolCalls(calls)) => {
                            let _ = tx.send(StreamMsg::ToolCalls { calls });
                        }
                        // Do NOT break on Finish: under `include_usage` the
                        // server emits the `usage` block in a SEPARATE final SSE
                        // chunk that arrives AFTER Finish, so breaking here would
                        // drop the token counts (and with them the context-usage
                        // + prefill stats). Keep draining until the stream ends
                        // naturally ([DONE]); `Done` is sent below. Mirrors the
                        // GUI's streaming loop.
                        Ok(ChatEvent::Finish(_)) => {}
                        Ok(ChatEvent::Start(_)) => {}
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

/// Answer a pending CLARIFY question: record the chosen option as a user
/// turn (the question is already an assistant turn) and continue the stream.
fn answer_pending_question(
    app: &mut App,
    client: &Arc<Client>,
    stream_tx: &UnboundedSender<StreamMsg>,
    answer: String,
) {
    app.pending_question = None;
    app.turns.push(Turn {
        role: "user".into(),
        content: answer,
    });
    app.turns.push(Turn {
        role: "assistant".into(),
        content: String::new(),
    });
    app.begin_turn_stats();
    app.streaming = true;
    app.follow = true;
    app.status = "generating…".into();
    spawn_stream(client.clone(), app.request(), stream_tx.clone());
}

/// Deny a proposed tool call: push a brief "don't run that" user turn and
/// continue the stream so the model answers directly instead.
fn deny_pending_tools(app: &mut App, client: &Arc<Client>, stream_tx: &UnboundedSender<StreamMsg>) {
    app.turns.push(Turn {
        role: "user".into(),
        content: "Please don't run that tool — answer directly instead.".into(),
    });
    app.turns.push(Turn {
        role: "assistant".into(),
        content: String::new(),
    });
    app.begin_turn_stats();
    app.streaming = true;
    app.follow = true;
    app.status = "generating…".into();
    spawn_stream(client.clone(), app.request(), stream_tx.clone());
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
        "temp" | "temperature" => set_f32(&mut app.temperature, &arg, &mut app.status, "temp"),
        "top_p" => set_f32(&mut app.top_p, &arg, &mut app.status, "top_p"),
        "repeat_penalty" => set_f32(
            &mut app.repeat_penalty,
            &arg,
            &mut app.status,
            "repeat_penalty",
        ),
        "presence_penalty" => set_f32(
            &mut app.presence_penalty,
            &arg,
            &mut app.status,
            "presence_penalty",
        ),
        "top_k" => {
            if let Ok(v) = arg.parse() {
                app.top_k = Some(v);
                app.status = format!("top_k = {v}");
            }
        }
        "max_tokens" => {
            if let Ok(v) = arg.parse() {
                app.max_tokens = v;
                app.status = format!("max_tokens = {v}");
            }
        }
        // Toggle auto-run-tools (same as Ctrl-T). `on`/`off` force a state.
        "auto" => {
            app.auto_tools = match arg.as_str() {
                "on" => true,
                "off" => false,
                _ => !app.auto_tools,
            };
            app.status = format!("auto-tools: {}", if app.auto_tools { "ON" } else { "OFF" });
        }
        // Toggle CLARIFY (same as Ctrl-K). `on`/`off` force a state.
        "clarify" => {
            app.clarify_enabled = match arg.as_str() {
                "on" => true,
                "off" => false,
                _ => !app.clarify_enabled,
            };
            app.status = format!(
                "clarify: {}",
                if app.clarify_enabled { "ON" } else { "OFF" }
            );
        }
        "help" => {
            // One command per line, each with a short inline description.
            app.turns.push(Turn {
                role: "info".into(),
                content: "commands (mirror the CLI):\n\
                     \n\
                     /model <id|list|load <ref>|default <ref>|unload <id>>   manage loaded models\n\
                     /chat <list|save [name]|resume <name>|delete <name>>    saved-chat history\n\
                     /system <text>          set the system prompt\n\
                     /reset                  clear the conversation\n\
                     /temp <f>               sampling temperature (alias: /temperature)\n\
                     /top_p <f>              nucleus sampling cutoff\n\
                     /top_k <n>              keep top-K tokens; 0 = full vocab\n\
                     /repeat_penalty <f>     discourage repeats; 1.0 = off\n\
                     /presence_penalty <f>   curb run-on / emoji spiral; unset = model default\n\
                     /max_tokens <n>         per-turn output cap\n\
                     /auto [on|off]          auto-run proposed tool calls\n\
                     /clarify [on|off]       let the model ask clarifying questions\n\
                     /help                   show this help\n\
                     /quit                   exit the chat\n\
                     \n\
                     keys: Enter send · PgUp/PgDn scroll · Ctrl-T auto-tools · Ctrl-K clarify · Esc/Ctrl-C quit"
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

/// Parse `arg` as an f32 and record it as an EXPLICIT user override (`Some`)
/// in `slot`. An unset slot stays `None`, so `request()` omits that field and
/// the server applies its per-model recommendation.
fn set_f32(slot: &mut Option<f32>, arg: &str, status: &mut String, name: &str) {
    if let Ok(v) = arg.parse() {
        *slot = Some(v);
        *status = format!("{name} = {v}");
    }
}

// ---- rendering ----

/// An assistant message split into its chain-of-thought reasoning span and the
/// final answer, for DISPLAY.
struct ReasonSplit {
    /// The reasoning text (the `<think>` scratchpad), to render dimmed. Empty
    /// when the model emitted no reasoning.
    reasoning: String,
    /// The answer (reasoning stripped), to render normally. Empty while the
    /// reasoning is still streaming (no `</think>` yet).
    answer: String,
    /// Whether a reasoning span was detected at all.
    has_reasoning: bool,
}

/// Split an assistant message into `(reasoning, answer)` for DISPLAY.
///
/// Reasoning models (Qwen3, …) stream a `<think>…</think>` scratchpad. Rather
/// than hide it behind a frozen `[thinking…]` marker (which looks identical
/// whether the model is working or stuck), the TUI STREAMS the reasoning live
/// in a dimmed style and renders the answer normally once `</think>` arrives —
/// the same split the native GUI's "Reasoning" section makes.
///
///   - **Closed span** — a normal `<think>…</think>` pair OR Qwen3's
///     prompt-opened form (the completion carries only the closing tag because
///     the template opens `<think>` in the PROMPT): `reasoning` = the text up to
///     `</think>` (any leading `<think>` removed), `answer` = the shared
///     `strip_reasoning` cut (so the displayed answer matches what the OUTGOING
///     history re-sends).
///   - **Open span** — an `<think>` seen, no `</think>` yet (reasoning still
///     streaming): `reasoning` = the text after `<think>`, `answer` = empty.
///   - **No think tags** → `answer` = the content, `reasoning` empty.
///
/// Display-only; the stored content (and thus history) is untouched.
fn split_reasoning(content: &str) -> ReasonSplit {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    if let Some(ci) = content.find(CLOSE) {
        // Closed: reasoning is everything before `</think>` (drop a leading
        // `<think>` if present — prompt-opened Qwen3 has none). The answer uses
        // the SAME shared cut the REPL/GUI/outgoing-history path uses.
        let head = &content[..ci];
        let reasoning = match head.rfind(OPEN) {
            Some(oi) => &head[oi + OPEN.len()..],
            None => head,
        };
        return ReasonSplit {
            reasoning: reasoning.trim().to_string(),
            answer: rustllama_client::strip_reasoning(content).to_string(),
            has_reasoning: true,
        };
    }
    if let Some(oi) = content.find(OPEN) {
        // Open: reasoning still streaming, no close yet. Show it live (dimmed).
        return ReasonSplit {
            reasoning: content[oi + OPEN.len()..].trim_start().to_string(),
            answer: String::new(),
            has_reasoning: true,
        };
    }
    // No think tags → render the whole content as the answer.
    ReasonSplit {
        reasoning: String::new(),
        answer: content.to_string(),
        has_reasoning: false,
    }
}

fn ui(f: &mut Frame, app: &mut App) {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // status bar
            Constraint::Length(1), // stats footer
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
        // Auto-run-tools indicator (Ctrl-T toggles).
        Span::styled(
            format!("  tools:{}", if app.auto_tools { "auto" } else { "ask" }),
            Style::default().fg(if app.auto_tools {
                Color::Magenta
            } else {
                Color::DarkGray
            }),
        ),
        // CLARIFY indicator (Ctrl-K toggles).
        Span::styled(
            format!(
                "  clarify:{}",
                if app.clarify_enabled { "on" } else { "off" }
            ),
            Style::default().fg(if app.clarify_enabled {
                Color::Cyan
            } else {
                Color::DarkGray
            }),
        ),
    ]);
    f.render_widget(Paragraph::new(status), chunks[0]);

    // --- stats footer ---
    // Live during a stream (client-measured TTFT + decode so far), frozen to
    // the last turn's full figures when idle. See `InferenceStats`.
    let stats = if app.streaming {
        app.stats_timer
            .as_ref()
            .map(|t| t.snapshot(app.pending_usage.as_ref(), app.ctx_size))
    } else {
        app.last_stats.clone()
    };
    let stats_text = stats.map(|s| s.format_compact()).unwrap_or_default();
    let stats_line = Line::from(vec![
        Span::styled(
            " stats ",
            Style::default().fg(Color::Black).bg(Color::DarkGray),
        ),
        Span::raw(" "),
        Span::styled(
            if stats_text.is_empty() {
                "—".to_string()
            } else {
                stats_text
            },
            Style::default().fg(Color::Gray),
        ),
    ]);
    f.render_widget(Paragraph::new(stats_line), chunks[1]);

    // --- transcript ---
    // Only a TOP border (the ` transcript ` title line). The left/right columns
    // and the bottom row are content, by design: the full box ring was dropped
    // so the pane uses the whole width AND can never smear a scrolled glyph onto
    // a side/bottom border (the XShell→SSH artifact). See the border-integrity
    // regression test below.
    let transcript_block = Block::default().borders(Borders::TOP).title(" transcript ");
    let inner = transcript_block.inner(chunks[2]);
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
        // "Prefix: " then wrapped body; continuation lines indented. Assistant
        // turns split their chain-of-thought `<think>…</think>` span out: the
        // reasoning streams LIVE in a dimmed (DarkGray) style — so a long / slow
        // / stuck think phase shows as visible progress, not a frozen marker —
        // and the answer renders normally once `</think>` arrives. Everything
        // else renders verbatim in its role style.
        let head = format!("{prefix}: ");
        // DISPLAY width of the head, not its byte length: the info prefix "··"
        // is U+00B7 ×2 = 4 bytes / 2 columns, so `head.len()` over-indented and
        // over-subtracted by the byte/column gap. The indent is that many
        // spaces (1 column each) so continuation lines align under the body.
        let head_w = UnicodeWidthStr::width(head.as_str());
        let indent = " ".repeat(head_w.min(content_w.saturating_sub(1)));
        let body_width = content_w.saturating_sub(head_w).max(1);

        // Ordered, per-segment-styled body: assistant turns emit a dimmed
        // reasoning segment (when present) then the normal answer; every other
        // role is a single verbatim segment.
        let segments: Vec<(String, Style)> = if t.role == "assistant" {
            let split = split_reasoning(&t.content);
            let mut segs: Vec<(String, Style)> = Vec::new();
            if split.has_reasoning && !split.reasoning.is_empty() {
                segs.push((split.reasoning, Style::default().fg(Color::DarkGray)));
            }
            // The answer is empty while the reasoning is still streaming; it
            // fills in once `</think>` arrives.
            if !split.answer.is_empty() {
                segs.push((split.answer, Style::default()));
            }
            segs
        } else {
            vec![(t.content.clone(), body_style)]
        };

        if t.role == "assistant" {
            // AI responses: put the `AI:` label on its OWN line, then render the
            // body starting on the next line FLUSH-LEFT (no label-width indent),
            // wrapped to the full content width. Content-intrinsic indentation
            // (markdown lists / code blocks carry their own leading spaces) is
            // preserved because we simply don't ADD an indent. The label line
            // alone also covers the empty-placeholder case while a reply streams.
            lines.push(Line::from(vec![Span::styled(head.clone(), style)]));
            for (text, seg_style) in &segments {
                for w in wrap_text(text, content_w) {
                    lines.push(Line::from(vec![Span::styled(w, *seg_style)]));
                }
            }
        } else {
            // Other roles (You / info / error): the head leads the first wrapped
            // line; later lines indent under the body. A turn with no body still
            // shows its head.
            let mut first_line = true;
            for (text, seg_style) in &segments {
                for w in wrap_text(text, body_width) {
                    if first_line {
                        lines.push(Line::from(vec![
                            Span::styled(head.clone(), style),
                            Span::styled(w, *seg_style),
                        ]));
                        first_line = false;
                    } else {
                        lines.push(Line::from(vec![
                            Span::raw(indent.clone()),
                            Span::styled(w, *seg_style),
                        ]));
                    }
                }
            }
            if first_line {
                lines.push(Line::from(vec![Span::styled(head.clone(), style)]));
            }
        }
        lines.push(Line::from("")); // blank separator between turns
    }

    // CLARIFY: render the pending question's numbered options so the user
    // knows what a digit / Enter will pick.
    if let Some((_, options)) = &app.pending_question {
        lines.push(Line::from(vec![Span::styled(
            "Choose an option (press its number, or Enter for 1):",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )]));
        for (i, opt) in options.iter().enumerate() {
            lines.push(Line::from(vec![
                Span::styled(format!("  {}. ", i + 1), Style::default().fg(Color::Cyan)),
                Span::raw(opt.clone()),
            ]));
        }
        lines.push(Line::from(""));
    }
    // Tool-confirm (auto-tools OFF): prompt approve/deny.
    if app.pending_tools.is_some() {
        lines.push(Line::from(vec![Span::styled(
            "Run this tool call?  y = accept   n = deny (Ctrl-T = auto-run)",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )]));
        lines.push(Line::from(""));
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
    // Belt-and-braces: the branches above already keep `scroll` in range, but
    // an explicit clamp guarantees it is always a valid offset (never past the
    // content) regardless of how it was last mutated by a key event.
    app.scroll = app.scroll.min(max_scroll);

    // Render the bordered Block and the scrolled transcript as TWO widgets, not
    // one `Paragraph::block(..).scroll(..)`. The Block owns the full `chunks[2]`
    // (its border ring); the Paragraph is clipped to the Block's `inner` area
    // and never sees the border cells at all. Drawn together in one pass, the
    // scrolled content can paint over the frame — on a narrow/odd terminal (an
    // XShell→SSH session) that shows as the box border vanishing and glyphs
    // smearing along the edge as the user scrolls. Clipping the paragraph to
    // `inner` makes that structurally impossible: the border stays solid and no
    // scrolled cell can bleed past the frame.
    f.render_widget(transcript_block, chunks[2]);
    let transcript = Paragraph::new(Text::from(lines)).scroll((app.scroll, 0));
    f.render_widget(transcript, inner);

    // --- input box ---
    let title = if app.streaming {
        " input (generating…) "
    } else {
        " input "
    };
    let input_block = Block::default().borders(Borders::ALL).title(title);
    let input_inner = input_block.inner(chunks[3]);
    let prompt = "> ";
    let input = Paragraph::new(Line::from(vec![
        Span::styled(prompt, Style::default().fg(Color::DarkGray)),
        Span::raw(app.input.as_str()),
    ]))
    .block(input_block);
    f.render_widget(input, chunks[3]);

    // Cursor at the end of the input (only when not streaming).
    if !app.streaming {
        let cx = input_inner.x + (prompt.len() + app.input.chars().count()) as u16;
        let cy = input_inner.y;
        if cx < input_inner.x + input_inner.width {
            f.set_cursor_position((cx, cy));
        }
    }
}

/// Word-wrap `s` to `width` DISPLAY columns, preserving explicit newlines and
/// hard-splitting any single word wider than the line.
///
/// Width is measured with `unicode-width` — an emoji / CJK glyph is 2 cells, a
/// combining mark 0 — NOT a char- or byte-count. That is the fix for the
/// transcript-desync bug: the old `word.chars().count()` under-counted wide
/// glyphs, so a line it thought fit `width` actually painted WIDER than `width`
/// and ratatui (which renders each pre-wrapped `Line` to a fixed-width cell
/// row) pushed the overflow onto the next row — characters "slid" onto the
/// wrong line on a multibyte reply. Each returned segment now fits in `width`
/// columns, so the manual pre-wrap matches what ratatui paints and the scroll
/// line-count stays exact.
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
            let wlen = UnicodeWidthStr::width(word);
            if wlen > width {
                // Flush the current line, then hard-split the over-long word by
                // display columns so a wide glyph never straddles the boundary.
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                let mut seg = String::new();
                let mut seg_w = 0usize;
                for ch in word.chars() {
                    let cw = UnicodeWidthChar::width(ch).unwrap_or(0);
                    if seg_w + cw > width && !seg.is_empty() {
                        out.push(std::mem::take(&mut seg));
                        seg_w = 0;
                    }
                    seg.push(ch);
                    seg_w += cw;
                }
                cur = seg;
            } else if cur.is_empty() {
                cur = word.to_string();
            } else if UnicodeWidthStr::width(cur.as_str()) + 1 + wlen <= width {
                // +1 for the joining space. `cur`'s width is recomputed rather
                // than tracked so there is no dead-store bookkeeping on the
                // hard-split path; `cur` is at most `width` cells, so this stays
                // cheap for a transcript line.
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Baseline the wrapper relies on: `unicode-width` measures an emoji / CJK
    /// glyph as 2 cells and ASCII as 1 (a char-count treats both as 1, which is
    /// what desynced the terminal columns). Also: the info prefix "··" is two
    /// U+00B7 — 2 columns but 4 bytes — so byte length is the wrong indent.
    #[test]
    fn display_width_counts_wide_glyphs_as_two() {
        assert_eq!(UnicodeWidthStr::width("a"), 1);
        assert_eq!(UnicodeWidthChar::width('a'), Some(1));
        assert_eq!(UnicodeWidthStr::width("🦙"), 2); // llama emoji
        assert_eq!(UnicodeWidthStr::width("世"), 2); // CJK
        assert_eq!(UnicodeWidthStr::width("··"), 2); // info prefix, 2 cols…
        assert_eq!("··".len(), 4); // …but 4 bytes
    }

    /// Every wrapped segment fits the target DISPLAY width — a run of width-2
    /// glyphs wraps at half as many glyphs per line as a char-count would.
    #[test]
    fn wrap_text_breaks_on_display_width_not_char_count() {
        // Eight llama emoji (one word, no spaces) = 16 columns. At width 8 that
        // is 4 glyphs per line → two lines, each exactly 8 columns wide.
        let wrapped = wrap_text("🦙🦙🦙🦙🦙🦙🦙🦙", 8);
        assert_eq!(wrapped.len(), 2);
        for line in &wrapped {
            assert!(
                UnicodeWidthStr::width(line.as_str()) <= 8,
                "segment wider than the column: {line:?}"
            );
        }
        assert_eq!(wrapped[0].chars().count(), 4); // 4 wide glyphs = 8 cols
    }

    /// Word-level wrapping respects display width: a wide word that would spill
    /// past the column moves to the next line instead of overflowing it.
    #[test]
    fn wrap_text_wraps_words_by_width() {
        // "hi" (2) + " " + "世界" (4) = 7 → fits width 7 on one line.
        assert_eq!(wrap_text("hi 世界", 7), vec!["hi 世界".to_string()]);
        // At width 4 the wide word can't join (2+1+4 = 7 > 4) → two lines.
        let w = wrap_text("hi 世界", 4);
        assert_eq!(w, vec!["hi".to_string(), "世界".to_string()]);
        for line in &w {
            assert!(UnicodeWidthStr::width(line.as_str()) <= 4);
        }
    }

    /// Explicit newlines stay as separate logical lines — an exact line count
    /// is what keeps the transcript scroll math right.
    #[test]
    fn wrap_text_preserves_newlines() {
        assert_eq!(
            wrap_text("a\n\nb", 10),
            vec!["a".to_string(), String::new(), "b".to_string()]
        );
    }

    /// A minimal `App` for driving the real `ui(f, &mut app)` under a
    /// `TestBackend`. Fields match the `event_loop` defaults; stats start
    /// cleared so an idle footer is deterministic.
    fn test_app() -> App {
        App {
            system: None,
            turns: Vec::new(),
            input: String::new(),
            scroll: 0,
            follow: true,
            streaming: false,
            model_label: "test-model".into(),
            model_id: "test-model".into(),
            server_url: "http://127.0.0.1:11434".into(),
            status: "ready".into(),
            temperature: None,
            // Mirror the real `event_loop` default: the ctx-aware value, not 512.
            max_tokens: crate::default_max_tokens(8192),
            top_p: None,
            top_k: None,
            repeat_penalty: None,
            presence_penalty: None,
            should_quit: false,
            auto_tools: false,
            clarify_enabled: true,
            pending_question: None,
            pending_tools: None,
            stats_timer: None,
            pending_usage: None,
            last_stats: None,
            ctx_size: 8192,
        }
    }

    /// A fresh `App` (the user has run no `/temp` etc.) must build a request
    /// that OMITS every sampling field, so the server's per-model recommended
    /// sampling applies. Setting one via the real `/command` path flips only
    /// that field to `Some`; `max_tokens` (a hard cap) is always sent.
    #[test]
    fn fresh_request_omits_sampling_until_set() {
        // `handle_command` needs a client + stream channel; the sampling
        // `/commands` touch neither, so a never-used local client is fine
        // (constructing a reqwest client does no I/O and needs no runtime).
        let client = Arc::new(Client::new("http://127.0.0.1:11434").unwrap());
        let (tx, _rx) = unbounded_channel::<StreamMsg>();

        let mut app = test_app();
        let req = app.request();
        assert_eq!(req.temperature, None);
        assert_eq!(req.top_p, None);
        assert_eq!(req.top_k, None);
        assert_eq!(req.repeat_penalty, None);
        // Hard cap, always sent — the ctx-aware default (ctx 8192 → 8192), not 512.
        assert_eq!(req.max_tokens, Some(crate::default_max_tokens(8192)));
        assert_ne!(req.max_tokens, Some(512));

        // `/top_k 50` sets ONLY top_k; the rest stay omitted.
        handle_command(&mut app, "top_k 50", &client, &tx);
        let req = app.request();
        assert_eq!(req.top_k, Some(50));
        assert_eq!(req.temperature, None);
        assert_eq!(req.top_p, None);
        assert_eq!(req.repeat_penalty, None);

        // `/temp 0.3` then also sets temperature, leaving the others as-is.
        handle_command(&mut app, "temp 0.3", &client, &tx);
        let req = app.request();
        assert_eq!(req.temperature, Some(0.3));
        assert_eq!(req.top_k, Some(50));
        assert_eq!(req.top_p, None);
        assert_eq!(req.repeat_penalty, None);
    }

    /// Emulate SSE token deltas: a few characters at a time, never splitting a
    /// `char` — so a multi-byte glyph (the trailing emoji) always arrives whole
    /// in one delta, exactly as a real tokenizer emits it.
    fn chunk_tokens(s: &str) -> Vec<String> {
        let chars: Vec<char> = s.chars().collect();
        chars.chunks(3).map(|c| c.iter().collect()).collect()
    }

    /// GOAL 1 — the decisive app-bug-vs-terminal test.
    ///
    /// Drive the REAL render function `ui(f, &mut app)` under a ratatui
    /// `TestBackend`, simulating the streaming loop: push a user turn + an
    /// empty assistant placeholder, then append the reply TOKEN-BY-TOKEN
    /// (`terminal.draw` after each delta, exactly like `event_loop`). The reply
    /// is a multi-paragraph answer with a fenced code block and a trailing
    /// emoji — the shape that desyncs a width-naive renderer. After the final
    /// frame, the on-screen buffer (`backend().buffer()`, which accumulates
    /// ratatui's per-frame DIFFS — so any ghost / stale cell would survive in
    /// it) MUST equal a single-shot render of the same full content.
    ///
    /// If this ever fails, the corruption is APP-side (the auto-follow
    /// `scroll`/`max_scroll` recompute as the line count grows, the
    /// `Paragraph::scroll` interaction, or a widget not clearing its area) and
    /// the fix belongs here. If it passes (as it does), the app render is
    /// byte-for-cell correct and the on-terminal ghosting an XShell→SSH user
    /// sees is TERMINAL / TRANSPORT-side (redraw-volume + SSH), which the
    /// redraw coalescing in `event_loop` is what addresses.
    #[test]
    fn streaming_render_equals_single_shot_render() {
        use ratatui::backend::TestBackend;

        let reply = "Here is a short explanation of the idea.\n\n\
                     It takes two paragraphs, and then a fenced code block:\n\n\
                     ```\n\
                     console.log(\"hello, world — from inside a fenced code block\");\n\
                     ```\n\n\
                     That is all. Thanks for reading 😊";

        // A small terminal so the reply exceeds the viewport — this exercises
        // the auto-follow scroll / `max_scroll` recompute every frame as the
        // line count grows, and forces the code line to wrap.
        let (w, h) = (50u16, 16u16);

        // --- render incrementally, one frame per token, like the loop ---
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        let mut app = test_app();
        app.turns.push(Turn {
            role: "user".into(),
            content: "explain the idea".into(),
        });
        app.turns.push(Turn {
            role: "assistant".into(),
            content: String::new(),
        });
        app.streaming = true;
        app.follow = true;
        term.draw(|f| ui(f, &mut app)).unwrap(); // first paint (empty assistant)
        for tok in chunk_tokens(reply) {
            if let Some(a) = app.last_assistant_mut() {
                a.content.push_str(&tok);
            }
            term.draw(|f| ui(f, &mut app)).unwrap();
        }
        app.streaming = false;
        term.draw(|f| ui(f, &mut app)).unwrap(); // final paint, stream done
        let streamed = term.backend().buffer().clone();

        // --- render the same full content in a single shot ---
        let mut term2 = Terminal::new(TestBackend::new(w, h)).unwrap();
        let mut app2 = test_app();
        app2.turns.push(Turn {
            role: "user".into(),
            content: "explain the idea".into(),
        });
        app2.turns.push(Turn {
            role: "assistant".into(),
            content: reply.to_string(),
        });
        app2.streaming = false;
        app2.follow = true;
        term2.draw(|f| ui(f, &mut app2)).unwrap();
        let single = term2.backend().buffer().clone();

        assert_eq!(
            streamed, single,
            "incremental-stream buffer differs from the single-shot render — \
             app-side streaming corruption"
        );
    }

    /// A tall transcript (content far exceeds the viewport) for the scroll test.
    fn tall_app() -> App {
        let mut a = test_app();
        for i in 0..12 {
            a.turns.push(Turn {
                role: "user".into(),
                content: format!(
                    "question number {i} with enough words to wrap across the narrow pane"
                ),
            });
            a.turns.push(Turn {
                role: "assistant".into(),
                content: format!(
                    "answer number {i}: here is a multi-line reply.\n\
                     second line of the answer.\nthird line with a tail 😊"
                ),
            });
        }
        a.streaming = false;
        a
    }

    /// The transcript pane's rect, laid out exactly as `ui` does (the third
    /// slot: status bar, stats footer, transcript, input box).
    fn transcript_rect(w: u16, h: u16) -> Rect {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Min(3),
                Constraint::Length(3),
            ])
            .split(Rect::new(0, 0, w, h))[2]
    }

    fn is_box_glyph(sym: &str) -> bool {
        // A space here would mean the border VANISHED at that cell — the exact
        // bug under test — so it is NOT accepted.
        matches!(
            sym,
            "│" | "─" | "┌" | "┐" | "└" | "┘" | "├" | "┤" | "┬" | "┴" | "┼"
        )
    }

    /// The transcript block now draws only a TOP border (the ` transcript `
    /// title line); the left/right columns and the bottom row are content, by
    /// design (the box ring was dropped so the pane uses the full width and can
    /// never smear a scrolled glyph onto a side/bottom border over XShell→SSH).
    /// So the integrity check is now "the top line is intact": the title
    /// renders and the line reaches the right edge (not overwritten by bled
    /// content). The stronger no-smear guarantee is the fresh-render `assert_eq!`
    /// in the caller.
    fn assert_border_intact(buf: &ratatui::buffer::Buffer, w: u16, h: u16) {
        let t = transcript_rect(w, h);
        let y = t.top();
        let mut top_row = String::new();
        for x in t.left()..t.right() {
            top_row.push_str(buf[(x, y)].symbol());
        }
        assert!(
            top_row.contains("transcript"),
            "transcript title missing from top row: {top_row:?}"
        );
        let r = buf[(t.right() - 1, y)].symbol().to_string();
        assert!(
            is_box_glyph(&r),
            "top border overwritten at right edge: {r:?}"
        );
    }

    /// GOAL 1 — manual-scroll / border-integrity regression.
    ///
    /// The reported XShell→SSH symptom is the transcript's BOX BORDER vanishing
    /// and glyphs SMEARING as the user scrolls (not only during streaming).
    /// Drive the real `ui` through a sequence of scroll offsets — top, middle,
    /// PAST the end (over-max), and back up — on ONE `TestBackend` (so a stale
    /// cell from a prior offset would survive into the buffer). At every offset
    /// the block border must stay intact, and the final buffer must equal a
    /// FRESH single-shot render at that offset (no smear / stale cells).
    ///
    /// With the transcript rendered as a Block clipped-to-`inner` Paragraph
    /// (not `Paragraph::block(..).scroll(..)`), scrolled content can never
    /// touch the border, so this holds.
    #[test]
    fn manual_scroll_keeps_border_and_leaves_no_smear() {
        use ratatui::backend::TestBackend;
        let (w, h) = (40u16, 12u16);

        // Visit several offsets in sequence on one terminal.
        let offsets = [0u16, 6, 100, 3];
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        let mut app = tall_app();
        for &off in &offsets {
            app.follow = false;
            app.scroll = off;
            term.draw(|f| ui(f, &mut app)).unwrap();
            assert_border_intact(term.backend().buffer(), w, h);
        }

        // The last offset we rendered (3) with no follow — a fresh render at the
        // same state must match cell-for-cell (any smear left by the earlier
        // offsets, including the over-max bottom pin, would show up as a diff).
        let mut fresh = Terminal::new(TestBackend::new(w, h)).unwrap();
        let mut app2 = tall_app();
        app2.follow = false;
        app2.scroll = 3;
        fresh.draw(|f| ui(f, &mut app2)).unwrap();

        assert_eq!(
            term.backend().buffer(),
            fresh.backend().buffer(),
            "scrolling through offsets left stale / smeared cells vs a fresh render"
        );
    }

    /// Concatenate every cell symbol (row-major, a newline per row) so a
    /// rendered-text substring check is easy.
    fn buffer_text(buf: &ratatui::buffer::Buffer) -> String {
        let area = *buf.area();
        let mut s = String::new();
        for y in 0..area.height {
            for x in 0..area.width {
                s.push_str(buf[(x, y)].symbol());
            }
            s.push('\n');
        }
        s
    }

    /// The foreground color of the first cell whose symbol equals `needle`
    /// (scanning row-major). Used to assert a span's style landed in the buffer.
    fn first_cell_fg(buf: &ratatui::buffer::Buffer, needle: &str) -> Option<Color> {
        let area = *buf.area();
        for y in 0..area.height {
            for x in 0..area.width {
                if buf[(x, y)].symbol() == needle {
                    return Some(buf[(x, y)].fg);
                }
            }
        }
        None
    }

    /// FIX 3 — reasoning streams LIVE (dimmed), not a frozen "[thinking…]".
    ///
    /// While a `<think>` span is OPEN (streaming, no `</think>` yet) the real
    /// `ui` must paint the reasoning text itself in a dimmed (DarkGray) style —
    /// so the user sees progress token-by-token — rather than collapse it to a
    /// static marker that hides whether the model is working or stuck.
    #[test]
    fn open_think_span_streams_dimmed_reasoning_not_a_marker() {
        use ratatui::backend::TestBackend;
        let (w, h) = (60u16, 16u16);
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        let mut app = test_app();
        app.turns.push(Turn {
            role: "user".into(),
            content: "hi".into(),
        });
        // An OPEN think span: the unique 'Z' leads the reasoning so its styled
        // cell is easy to find; no closing tag yet (still streaming).
        app.turns.push(Turn {
            role: "assistant".into(),
            content: "<think>Zebra reasoning underway".into(),
        });
        app.streaming = true;
        term.draw(|f| ui(f, &mut app)).unwrap();
        let buf = term.backend().buffer();
        let text = buffer_text(buf);

        // The reasoning TEXT is on screen (streamed live)…
        assert!(
            text.contains("Zebra reasoning underway"),
            "reasoning text must stream live, got:\n{text}"
        );
        // …NOT hidden behind a frozen marker…
        assert!(
            !text.contains("[thinking"),
            "must not show the old static [thinking…] marker"
        );
        // …and it is dimmed (DarkGray), visibly distinct from a normal answer.
        assert_eq!(
            first_cell_fg(buf, "Z"),
            Some(Color::DarkGray),
            "streaming reasoning must render dimmed (DarkGray)"
        );
    }

    /// FIX 3 — once `</think>` arrives the answer renders normally and the
    /// reasoning stays styled distinctly (dimmed) from it.
    #[test]
    fn closed_think_span_renders_answer_normal_reasoning_dimmed() {
        use ratatui::backend::TestBackend;
        let (w, h) = (60u16, 16u16);
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        let mut app = test_app();
        app.turns.push(Turn {
            role: "user".into(),
            content: "hi".into(),
        });
        // Closed span: 'Z' leads the reasoning, 'Q' leads the answer (both
        // unique in the frame) so each segment's style is findable.
        app.turns.push(Turn {
            role: "assistant".into(),
            content: "<think>Zebra reasoning underway</think>Qwik final answer".into(),
        });
        app.streaming = false;
        term.draw(|f| ui(f, &mut app)).unwrap();
        let buf = term.backend().buffer();
        let text = buffer_text(buf);

        // Both the reasoning and the answer are on screen.
        assert!(text.contains("Zebra reasoning underway"), "reasoning kept");
        assert!(text.contains("Qwik final answer"), "answer rendered");

        let reasoning_fg = first_cell_fg(buf, "Z").expect("reasoning 'Z' present");
        let answer_fg = first_cell_fg(buf, "Q").expect("answer 'Q' present");
        // Reasoning dimmed; answer normal (buffer default fg = Reset); distinct.
        assert_eq!(reasoning_fg, Color::DarkGray, "reasoning stays dimmed");
        assert_eq!(
            answer_fg,
            Color::Reset,
            "answer renders in the normal style"
        );
        assert_ne!(
            reasoning_fg, answer_fg,
            "reasoning must be styled distinctly from the answer"
        );
    }
}
