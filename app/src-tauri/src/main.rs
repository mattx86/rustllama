//! rustllama dispatcher.
//!
//! Parses argv and routes:
//!   - `rustllama gui` → the Tauri builder (feature `gui` required)
//!   - everything else → `rustllama_cli::run`
//!
//! When the `gui` feature is built, the GUI subcommand also runs the
//! local HTTP server in-process on a tokio runtime thread before the
//! Tauri window opens — so the React app's `fetch('http://127.0.0.1:11434')`
//! calls find a live server out of the box. Standalone CLI use (`serve`,
//! `chat`, `pull`, etc.) doesn't touch the GUI code path.

use clap::Parser;
use rustllama_cli::{Cli, Command};

fn main() -> anyhow::Result<()> {
    // Autodetect + register GPU-runtime DLL dirs BEFORE any SYCL symbol
    // is touched (the kernel DLL is delay-loaded, so this in-process
    // PATH prepend takes effect when it binds). Makes `serve`/`gui`
    // work from a bare shell without run-sycl.bat. No-op off-Windows.
    rustllama_runtime::ensure_gpu_dll_search_paths();
    let cli = Cli::parse();

    match cli.command {
        Command::Gui => {
            #[cfg(feature = "gui")]
            {
                // The Tauri builder is synchronous (it owns the event
                // loop); the embedded server runs on a dedicated tokio
                // runtime spawned on a worker thread. We don't return
                // from `tauri::Builder::run` until the window closes.
                return run_gui();
            }
            #[cfg(not(feature = "gui"))]
            {
                anyhow::bail!(
                    "this binary was built without the `gui` feature; \
                     rebuild with `cargo build --release --features gui` \
                     after installing Node 20+ and pnpm (the frontend lives \
                     in app/ui)"
                );
            }
        }
        _ => {
            // Non-GUI commands route through `rustllama-cli`, which is
            // async itself — spin up a single-thread tokio runtime here
            // since the binary isn't `#[tokio::main]` anymore (Tauri
            // needs to own the main thread when the gui feature is on).
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(rustllama_cli::run(cli))
        }
    }
}

#[cfg(feature = "gui")]
fn run_gui() -> anyhow::Result<()> {
    use std::sync::Arc;
    use tracing_subscriber::fmt::writer::MakeWriterExt;

    // GUI processes have no console attached when launched from File
    // Explorer / Start menu, so stderr-only tracing is invisible.
    // Mirror logs into `<cache_dir>/gui.log` (rotating-append) so any
    // crash leaves a forensic trail the user can share. Errors here
    // fall back to a no-op writer — better to lose logs than to
    // refuse to launch.
    let log_path = rustllama_runtime::paths()
        .cache_dir
        .join("gui.log");
    let _ = std::fs::create_dir_all(&rustllama_runtime::paths().cache_dir);
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .ok();
    let log_path_for_panic = log_path.clone();
    if let Some(f) = log_file {
        let writer = std::sync::Mutex::new(f).with_max_level(tracing::Level::INFO);
        tracing_subscriber::fmt()
            .with_writer(writer)
            .with_ansi(false)
            .try_init()
            .ok();
    } else {
        tracing_subscriber::fmt::try_init().ok();
    }

    // Install the crash-log panic hook so any panic on either the
    // main Tauri thread OR the embedded server worker thread writes
    // a structured crash report to `<cache_dir>/crashes/<ts>.json`.
    // `set_hook` is process-wide, so calling once here covers all
    // future threads including the one we spawn below.
    rustllama_runtime::crash::install_panic_hook(env!("CARGO_PKG_VERSION"), "gui");

    tracing::info!(
        log_path = %log_path.display(),
        "rustllama GUI starting"
    );

    // Startup SYCL probe — runs unconditionally so the user can see
    // device-visibility status in `gui.log` without needing to
    // trigger a chat first. Shows up immediately after the
    // "rustllama GUI starting" line.
    match rustllama_kernels_sycl::device_count() {
        Ok(n) if n > 0 => {
            // Best-effort device-info dump for the first visible device.
            let info = rustllama_kernels_sycl::device_info(0).ok();
            tracing::info!(
                device_count = n,
                device_0_name = info.as_ref().map(|i| i.name.as_str()).unwrap_or("?"),
                device_0_driver = info.as_ref().map(|i| i.driver_version.as_str()).unwrap_or("?"),
                device_0_vendor_id = info.as_ref().map(|i| format!("0x{:04X}", i.vendor_id)).unwrap_or_default(),
                device_0_vram_mb = info.as_ref().map(|i| i.vram_bytes / (1024 * 1024)).unwrap_or(0),
                "SYCL device(s) detected — generation will dispatch to the GPU"
            );
        }
        Ok(_) => {
            tracing::warn!(
                "SYCL backend compiled in but `device_count() == 0` — \
                 no Intel GPU visible. Generation will use the CPU path. \
                 Check that the Intel graphics driver + oneAPI Level Zero \
                 loader are installed."
            );
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "SYCL device probe failed — generation will use the CPU path"
            );
        }
    }

    // Pre-flight: if a `rustllama serve` is already running and
    // bound to 127.0.0.1:11434, attach to it as a client instead of
    // spawning a second server (which would fail with WSAEADDRINUSE
    // / os error 10048 and leave the GUI showing "offline"). A
    // short TCP-connect probe is enough — we don't need to hit
    // /healthz since any listener on that port is presumably ours.
    let preflight = std::net::TcpStream::connect_timeout(
        &"127.0.0.1:11434".parse().expect("hardcoded socket addr"),
        std::time::Duration::from_millis(500),
    );
    let external_server_running = preflight.is_ok();
    tracing::info!(
        external_server_running,
        preflight_err = preflight.as_ref().err().map(|e| e.to_string()),
        "pre-flight TCP probe to 127.0.0.1:11434"
    );
    if external_server_running {
        tracing::info!(
            "found existing rustllama server on 127.0.0.1:11434; \
             attaching as client (skipping embedded server)"
        );
    } else {
        // Signal to the embedded server (via env var, since we route
        // through `rustllama_cli::run` which loads config from disk
        // independently of this process) that it should default
        // `[server].cors_origins` to `["*"]` when the user hasn't
        // configured anything. Without this the Tauri webview's
        // `http://tauri.localhost` origin can't read responses from
        // `http://127.0.0.1:11434` and every fetch reports as "offline".
        std::env::set_var("RUSTLLAMA_GUI_EMBEDDED", "1");

        // Start the local HTTP server on a worker thread BEFORE the Tauri
        // window opens, so the React app's first `getHealth()` poll
        // succeeds. The server stays alive for the lifetime of the
        // process; Tauri's window close triggers process exit, which
        // tears the server down.
        //
        // The worker thread catches any panic from the runtime + `run`
        // and writes it to the GUI log; the main thread keeps running so
        // the user at least sees a non-blank window with the inline
        // React error boundary.
        let _server_handle = std::thread::Builder::new()
            .name("rustllama-embedded-server".to_string())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let rt = tokio::runtime::Builder::new_multi_thread()
                        .enable_all()
                        .build()
                        .expect("build tokio runtime");
                    rt.block_on(async {
                        // Reuse the CLI's `serve` path so the embedded server
                        // sees the same config + model load logic as the
                        // standalone `rustllama serve` binary.
                        let serve = Cli {
                            config: None,
                            profile: None,
                            // Empty model list: the embedded server binds
                            // the configured host/port and loads `[model].path`
                            // if set; the GUI otherwise loads models on demand
                            // via its Models page (`POST /v1/models/load`).
                            command: Command::Serve {
                                model: Vec::new(),
                                ip: None,
                                port: None,
                                // Device-tier flags default off: the embedded
                                // server honors the `[inference]` config keys
                                // (cpu_enabled / gpu_enabled / disabled_cpus /
                                // vram_only) rather than CLI overrides.
                                no_cpu: false,
                                no_gpu: false,
                                disabled_cpus: None,
                                vram_only: false,
                            },
                        };
                        if let Err(e) = rustllama_cli::run(serve).await {
                            tracing::error!("embedded server exited: {e}");
                        }
                    });
                }));
                if let Err(payload) = result {
                    let msg = if let Some(s) = payload.downcast_ref::<&str>() {
                        (*s).to_string()
                    } else if let Some(s) = payload.downcast_ref::<String>() {
                        s.clone()
                    } else {
                        "<non-string panic payload>".to_string()
                    };
                    tracing::error!(
                        panic = msg,
                        log_path = %log_path_for_panic.display(),
                        "embedded server thread panicked"
                    );
                }
            })
            .map_err(|e| anyhow::anyhow!("spawn server thread: {e}"))?;
    }

    // Native egui GUI (replaces the former Tauri/webview window). The
    // embedded server spawned above serves the API on the config port; the
    // egui app drives it over HTTP via rustllama-client. The Tauri IPC
    // commands + tray helpers further down are now dead code, kept only
    // until the de-Tauri cleanup pass (dir rename + removal).
    let base_url = "http://127.0.0.1:11434".to_string(); // TODO: read [server].port
    rustllama_gui::run_ui(base_url)?;

    tracing::info!("rustllama GUI exiting cleanly");
    // Server thread keeps running until process exit; we don't await
    // its join handle because dropping it on shutdown is fine.
    let _ = Arc::new(());
    Ok(())
}

/// Tauri command surfaced to the frontend so the React app can
/// confirm the embedded server's base URL. The hard-coded localhost
/// URL is fine for v1 — when we add per-user port selection (config
/// `[server].port`), this becomes the single seam to plumb that
/// through.
#[cfg(feature = "gui")]
#[tauri::command]
fn server_url() -> String {
    "http://127.0.0.1:11434".to_string()
}

/// Build + install the system tray icon with a Show / Hide / Quit
/// menu. The window-close handler hides; the only way to actually
/// exit is the "Quit" menu item below. Tauri 2's tray API is
/// built-in — no extra plugin crate needed.
#[cfg(feature = "gui")]
fn install_tray(app: &tauri::AppHandle) -> tauri::Result<()> {
    use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
    use tauri::tray::{TrayIconBuilder, TrayIconEvent};
    use tauri::Manager;

    let show = MenuItem::with_id(app, "tray-show", "Show window", true, None::<&str>)?;
    let hide = MenuItem::with_id(app, "tray-hide", "Hide window", true, None::<&str>)?;
    let new_window =
        MenuItem::with_id(app, "tray-new-window", "New chat window", true, None::<&str>)?;
    let sep = PredefinedMenuItem::separator(app)?;
    let quit = MenuItem::with_id(app, "tray-quit", "Quit", true, None::<&str>)?;
    let menu = Menu::with_items(app, &[&show, &hide, &new_window, &sep, &quit])?;

    let _tray = TrayIconBuilder::with_id("rustllama-tray")
        .icon(app.default_window_icon().cloned().ok_or_else(|| {
            tauri::Error::AssetNotFound("default window icon".into())
        })?)
        .tooltip("rustllama")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id.as_ref() {
            "tray-show" => {
                if let Some(w) = app.get_webview_window("main") {
                    let _ = w.show();
                    let _ = w.set_focus();
                }
            }
            "tray-hide" => {
                if let Some(w) = app.get_webview_window("main") {
                    let _ = w.hide();
                }
            }
            "tray-new-window" => {
                // Pick an unused label by counting existing windows.
                // The tray menu's "New chat window" is the same
                // shape as the in-window React tab bar would use
                // (just a different trigger).
                let mut idx = 1;
                let existing: std::collections::HashSet<String> =
                    app.webview_windows().keys().cloned().collect();
                while existing.contains(&format!("chat-{idx}")) {
                    idx += 1;
                }
                let label = format!("chat-{idx}");
                let url = tauri::WebviewUrl::App("chat".into());
                let _ = tauri::WebviewWindowBuilder::new(app, &label, url)
                    .title(format!("rustllama — {label}"))
                    .inner_size(1200.0, 800.0)
                    .min_inner_size(800.0, 600.0)
                    .build();
            }
            "tray-quit" => {
                app.exit(0);
            }
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            // Left-click on the tray toggles the window — matches
            // the convention of every other tray app on Windows.
            if let TrayIconEvent::Click {
                button: tauri::tray::MouseButton::Left,
                button_state: tauri::tray::MouseButtonState::Up,
                ..
            } = event
            {
                if let Some(w) = tray.app_handle().get_webview_window("main") {
                    if w.is_visible().unwrap_or(false) {
                        let _ = w.hide();
                    } else {
                        let _ = w.show();
                        let _ = w.set_focus();
                    }
                }
            }
        })
        .build(app)?;
    Ok(())
}

// ===========================================================
// IPC commands — thin HTTP-client adapters for the React app.
// ===========================================================
//
// The React frontend can call any of these via `invoke('command_name', args)`
// instead of going through `fetch()` directly. v1 keeps both surfaces
// alive: the HTTP path is the existing wire and tests use it; the IPC
// path matches the plan's "GUI talks to engine via Tauri commands"
// shape and lets future hardening (e.g. per-command rate limiting in
// the Rust side) live in one place. Today they are 1:1 HTTP proxies.

#[cfg(feature = "gui")]
fn http_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()
        .expect("build reqwest client")
}

#[cfg(feature = "gui")]
fn base() -> String {
    "http://127.0.0.1:11434".to_string()
}

/// `invoke('list_models')` → JSON value matching the server's
/// `GET /v1/models` shape.
#[cfg(feature = "gui")]
#[tauri::command]
async fn list_models() -> Result<serde_json::Value, String> {
    http_client()
        .get(format!("{}/v1/models", base()))
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json::<serde_json::Value>()
        .await
        .map_err(|e| e.to_string())
}

/// `invoke('pull_model', { spec })` → triggers `POST /api/pull`.
/// Returns the final JSON status object; per-chunk progress streams
/// back via `pull://progress` events emitted from the chunked HTTP
/// body (the React frontend already consumes a streamed pull via
/// `fetch` today; this command is for callers that prefer IPC).
#[cfg(feature = "gui")]
#[tauri::command]
async fn pull_model(spec: String) -> Result<serde_json::Value, String> {
    http_client()
        .post(format!("{}/api/pull", base()))
        .json(&serde_json::json!({ "model": spec, "stream": false }))
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json::<serde_json::Value>()
        .await
        .map_err(|e| e.to_string())
}

/// `invoke('get_config')` → current `[server]`/`[model]`/`[inference]`/…
/// config snapshot. Server reads this from disk per call so live
/// edits via `config.toml` reload are picked up.
#[cfg(feature = "gui")]
#[tauri::command]
async fn get_config() -> Result<serde_json::Value, String> {
    http_client()
        .get(format!("{}/v1/config", base()))
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json::<serde_json::Value>()
        .await
        .map_err(|e| e.to_string())
}

/// `invoke('set_config', { patch })` → `PUT /v1/config`. The patch
/// shape mirrors the on-disk TOML structure (nested objects per
/// section). Server returns the merged config + per-section
/// hot-apply / restart flags so the UI knows which banner to show.
#[cfg(feature = "gui")]
#[tauri::command]
async fn set_config(patch: serde_json::Value) -> Result<serde_json::Value, String> {
    http_client()
        .put(format!("{}/v1/config", base()))
        .json(&patch)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json::<serde_json::Value>()
        .await
        .map_err(|e| e.to_string())
}

/// `invoke('get_server_status')` → `GET /healthz`. Aliased from
/// the plan's name; useful so the React app can `invoke` instead of
/// `fetch` during tray-toggle restore flows where fetch latency is
/// noticeable.
#[cfg(feature = "gui")]
#[tauri::command]
async fn get_server_status() -> Result<serde_json::Value, String> {
    http_client()
        .get(format!("{}/healthz", base()))
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json::<serde_json::Value>()
        .await
        .map_err(|e| e.to_string())
}

/// `invoke('ensure_server')` → either confirms a server is live on
/// 127.0.0.1:11434 (returns `{ already_running: true }`) or returns
/// `{ already_running: false }` if it isn't. v1 doesn't start a new
/// server from this command — the embedded server is launched at
/// GUI startup. A future iteration can re-spawn here for self-heal.
#[cfg(feature = "gui")]
#[tauri::command]
async fn ensure_server() -> Result<serde_json::Value, String> {
    let probe = std::net::TcpStream::connect_timeout(
        &"127.0.0.1:11434".parse().expect("hardcoded socket addr"),
        std::time::Duration::from_millis(250),
    );
    Ok(serde_json::json!({ "already_running": probe.is_ok() }))
}

/// `invoke('send_chat', { body })` with body matching the OpenAI
/// `POST /v1/chat/completions` schema. Returns the non-streamed
/// completion; for streaming, emit `chat://token` events from a
/// `tokio::spawn`'d task that reads the SSE response and call
/// `app.emit("chat://token", ...)` per chunk.
///
/// v1 routes the non-stream path. Streaming is wired via the
/// `stream_chat_command` below.
#[cfg(feature = "gui")]
#[tauri::command]
async fn send_chat(body: serde_json::Value) -> Result<serde_json::Value, String> {
    http_client()
        .post(format!("{}/v1/chat/completions", base()))
        .json(&body)
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json::<serde_json::Value>()
        .await
        .map_err(|e| e.to_string())
}

/// `invoke('stream_chat', { body })` → kicks off an SSE chat
/// completion against the embedded server. Per-token `chat://token`
/// events fire as content deltas arrive; `chat://done` carries the
/// final usage; `chat://error` carries any transport / server error.
/// The command itself returns immediately with the spawned task's
/// status — the caller subscribes to the three event channels.
#[cfg(feature = "gui")]
#[tauri::command]
async fn stream_chat(
    app: tauri::AppHandle,
    body: serde_json::Value,
) -> Result<serde_json::Value, String> {
    use futures::StreamExt;
    use tauri::Emitter;

    // Force streaming on so the server emits SSE; we ignore any
    // `stream: false` the caller passed.
    let mut body = body;
    if let Some(obj) = body.as_object_mut() {
        obj.insert("stream".into(), serde_json::Value::Bool(true));
    }

    let resp = http_client()
        .post(format!("{}/v1/chat/completions", base()))
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            let _ = app.emit("chat://error", serde_json::json!({ "error": e.to_string() }));
            e.to_string()
        })?;
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let text = resp.text().await.unwrap_or_default();
        let _ = app.emit(
            "chat://error",
            serde_json::json!({ "status": status, "body": text }),
        );
        return Err(format!("server returned {status}: {text}"));
    }

    // Walk the SSE byte stream. Each `data: …` line is one OpenAI
    // chunk; the terminating `data: [DONE]` triggers the done event.
    let mut stream = resp.bytes_stream();
    let mut buf = Vec::<u8>::new();
    while let Some(chunk) = stream.next().await {
        let bytes = chunk.map_err(|e| e.to_string())?;
        buf.extend_from_slice(&bytes);
        // SSE separates events with `\n\n`. Drain complete events
        // out of the buffer and leave any trailing partial frame.
        loop {
            let split = buf.windows(2).position(|w| w == b"\n\n");
            let Some(end) = split else { break };
            let event = buf.drain(..end + 2).collect::<Vec<u8>>();
            // Find a "data: …" line inside this event block (SSE
            // allows multiple data: lines but our server emits one).
            let event_str = String::from_utf8_lossy(&event);
            for line in event_str.lines() {
                if let Some(payload) = line.strip_prefix("data: ") {
                    if payload.trim() == "[DONE]" {
                        let _ = app.emit("chat://done", serde_json::json!({}));
                        return Ok(serde_json::json!({ "ok": true }));
                    }
                    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(payload) {
                        let _ = app.emit("chat://token", parsed);
                    }
                }
            }
        }
    }
    let _ = app.emit("chat://done", serde_json::json!({ "transport_eof": true }));
    Ok(serde_json::json!({ "ok": true }))
}

/// `invoke('get_tuning_state')` → `GET /v1/tuning_summary` for the
/// active SYCL device. Status page polls this on a slow tick.
#[cfg(feature = "gui")]
#[tauri::command]
async fn get_tuning_state() -> Result<serde_json::Value, String> {
    http_client()
        .get(format!("{}/v1/tuning_summary", base()))
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json::<serde_json::Value>()
        .await
        .map_err(|e| e.to_string())
}

/// `invoke('start_tune', { depth, mode })` → kicks off either
/// `/v1/tune/placement` or `/v1/tune/batch_size` depending on
/// `mode`. `depth` is `"quick"` (default) or `"thorough"`. Per-
/// candidate progress streams back via `tune://progress` events;
/// the final winner arrives in `tune://done`. Returns the routing
/// envelope (job id) for the caller to correlate against events.
#[cfg(feature = "gui")]
#[tauri::command]
async fn start_tune(
    app: tauri::AppHandle,
    mode: String,
    depth: Option<String>,
) -> Result<serde_json::Value, String> {
    use tauri::Emitter;
    let depth = depth.unwrap_or_else(|| "quick".into());
    let endpoint = match mode.as_str() {
        "placement" => "/v1/tune/placement",
        "batch_size" => "/v1/tune/batch_size",
        other => return Err(format!("unknown tune mode: {other}")),
    };
    // Emit a starting event so the UI can flip into "tuning…" state
    // before the (potentially long) HTTP request returns.
    let _ = app.emit(
        "tune://progress",
        serde_json::json!({ "mode": mode, "depth": depth, "phase": "starting" }),
    );
    let resp = http_client()
        .post(format!("{}{}", base(), endpoint))
        .json(&serde_json::json!({ "depth": depth }))
        .timeout(std::time::Duration::from_secs(15 * 60))
        .send()
        .await
        .map_err(|e| {
            let _ = app.emit(
                "tune://done",
                serde_json::json!({ "mode": mode, "ok": false, "error": e.to_string() }),
            );
            e.to_string()
        })?
        .json::<serde_json::Value>()
        .await
        .map_err(|e| e.to_string())?;
    let _ = app.emit(
        "tune://done",
        serde_json::json!({ "mode": mode, "ok": true, "result": resp.clone() }),
    );
    Ok(resp)
}

/// `invoke('cancel_tune')` → `POST /v1/cancel` to abort an in-flight
/// tune. Server-side cancellation is best-effort: kernel sweeps
/// finish their current measurement before exiting. Returns
/// `{ cancelled: true }` once the cancel signal is delivered.
#[cfg(feature = "gui")]
#[tauri::command]
async fn cancel_tune() -> Result<serde_json::Value, String> {
    http_client()
        .post(format!("{}/v1/cancel", base()))
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json::<serde_json::Value>()
        .await
        .map_err(|e| e.to_string())
}

// ===========================================================
// Multi-window: open additional chat windows.
// ===========================================================
//
// Each window is an independent Tauri WebviewWindow with its own
// React tree. They share the same embedded server (so loaded models +
// conversation history are shared), but each window has its own
// chat state — useful for comparing two prompts side-by-side or
// keeping a long-running RAG-query window open separately from a
// scratchpad. A "tab" in the GUI sense maps 1:1 to a window today;
// in-window tabs would require a router-level redesign of the React
// app, which is a follow-up.

/// `invoke('open_new_window', { label, route, title })` → spawns a
/// fresh webview window. `label` must be unique across open windows
/// (used as the window id); `route` is the React-router path the new
/// window opens on (e.g. `"/chat"`, `"/models"`); `title` is the
/// window title bar text. Returns the assigned label on success.
///
/// If `label` matches an already-open window, focuses that window
/// instead of opening a new one — the React tab bar uses this
/// behavior to switch between named windows.
#[cfg(feature = "gui")]
#[tauri::command]
async fn open_new_window(
    app: tauri::AppHandle,
    label: String,
    route: Option<String>,
    title: Option<String>,
) -> Result<String, String> {
    use tauri::Manager;
    if let Some(existing) = app.get_webview_window(&label) {
        let _ = existing.show();
        let _ = existing.set_focus();
        return Ok(label);
    }
    let route = route.unwrap_or_else(|| "/chat".into());
    let title = title.unwrap_or_else(|| format!("rustllama — {label}"));
    // `WebviewWindowBuilder::new` takes the AppHandle + a unique
    // label + a webview URL. Tauri serves the bundled frontend from
    // `tauri://localhost/`; opening on `/<route>` lands React Router
    // at the matching page.
    let url = tauri::WebviewUrl::App(route.trim_start_matches('/').into());
    let _w = tauri::WebviewWindowBuilder::new(&app, &label, url)
        .title(&title)
        .inner_size(1200.0, 800.0)
        .min_inner_size(800.0, 600.0)
        .build()
        .map_err(|e| format!("WebviewWindowBuilder::build: {e}"))?;
    Ok(label)
}

/// `invoke('list_open_windows')` → labels of every currently-open
/// webview window. The GUI's tab bar polls this on a slow tick to
/// reflect external window-close events that happen outside React
/// (user clicks the OS close button on a non-main window).
#[cfg(feature = "gui")]
#[tauri::command]
async fn list_open_windows(app: tauri::AppHandle) -> Result<Vec<String>, String> {
    use tauri::Manager;
    Ok(app
        .webview_windows()
        .keys()
        .cloned()
        .collect())
}

/// `invoke('quantize_model', { input, output, target, apex?, recipe?, keep_output? })` →
/// runs the [`rustllama_gguf::quantize`] pipeline synchronously.
/// Returns JSON `{ tensors_total, tensors_requantized, tensors_passthrough,
///   bytes_in, bytes_out, elapsed_ms, target }` on success.
///
/// Quantize jobs can take minutes on real models; the command runs
/// on a `spawn_blocking` task so the Tauri runtime stays responsive
/// for progress polling (a future iteration can emit
/// `quantize://progress` events; v1 is one-shot completion).
#[cfg(feature = "gui")]
#[tauri::command]
async fn quantize_model(
    input: String,
    output: String,
    target: String,
    apex: Option<String>,
    recipe: Option<String>,
    keep_output: Option<bool>,
) -> Result<serde_json::Value, String> {
    use rustllama_gguf::{
        apex::{build_apex_rules, ApexTier},
        quantize::{quantize_gguf_to_path, QuantizePlan},
        recipe::parse_recipe_file,
        GgmlType, Gguf, MetadataValue,
    };

    // Run the (CPU-bound) pipeline on a blocking worker so the
    // Tauri event loop stays responsive.
    let result = tokio::task::spawn_blocking(move || -> Result<serde_json::Value, String> {
        let target_dtype = parse_target_dtype_local(&target)
            .ok_or_else(|| format!("unknown target dtype {target:?}"))?;
        let src = Gguf::open(&input)
            .map_err(|e| format!("open source: {e}"))?;

        let mut plan = QuantizePlan::uniform(target_dtype);
        let n_layers = infer_n_layers(&src);
        if let Some(tier_name) = apex.as_deref() {
            let tier = ApexTier::parse(tier_name)
                .ok_or_else(|| format!("unknown APEX tier {tier_name:?}"))?;
            plan.add_rules(build_apex_rules(tier, n_layers));
        }
        if let Some(path) = recipe.as_deref() {
            let rules = parse_recipe_file(path)
                .map_err(|e| format!("recipe parse: {e}"))?;
            plan.add_rules(rules);
        }
        if keep_output.unwrap_or(false) {
            plan.passthrough_prefixes.push("output.".into());
            plan.passthrough_prefixes.push("lm_head.".into());
            plan.passthrough_prefixes.push("head.".into());
        }

        let start = std::time::Instant::now();
        let stats = quantize_gguf_to_path(&src, &output, &plan)
            .map_err(|e| format!("pipeline: {e}"))?;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let _ = GgmlType::F32;
        let _ = MetadataValue::U32(0);
        Ok(serde_json::json!({
            "tensors_total": stats.tensors_total,
            "tensors_requantized": stats.tensors_requantized,
            "tensors_passthrough": stats.tensors_passthrough,
            "bytes_in": stats.bytes_in,
            "bytes_out": stats.bytes_out,
            "elapsed_ms": elapsed_ms,
            "target": target_dtype.as_str(),
            "n_layers": n_layers,
        }))
    })
    .await
    .map_err(|e| format!("join: {e}"))??;

    Ok(result)
}

/// CLI-side `parse_target_dtype` is private to `rustllama-cli`;
/// re-implement here so the GUI doesn't depend on a non-public
/// helper. Kept in sync via the same lookup table.
#[cfg(feature = "gui")]
fn parse_target_dtype_local(name: &str) -> Option<rustllama_gguf::GgmlType> {
    rustllama_gguf::recipe::parse_dtype_name(name)
}

/// Same heuristic as `rustllama-cli`'s `infer_n_layers_from_gguf` —
/// extracted here so the Tauri command doesn't pull in a CLI
/// private helper. Looks up `{arch}.block_count` then falls back
/// to scanning `blk.N.*` tensor names.
#[cfg(feature = "gui")]
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

/// `invoke('close_window', { label })` → closes the named window.
/// The main window's close handler intercepts to hide-not-exit (via
/// `on_window_event`); secondary windows opened via
/// `open_new_window` actually close. Idempotent: closing a label
/// that doesn't exist is a no-op.
#[cfg(feature = "gui")]
#[tauri::command]
async fn close_window(app: tauri::AppHandle, label: String) -> Result<(), String> {
    use tauri::Manager;
    if let Some(w) = app.get_webview_window(&label) {
        // For the main window we hide (close-to-tray contract);
        // secondary windows actually close.
        if label == "main" {
            let _ = w.hide();
        } else {
            w.close().map_err(|e| format!("close: {e}"))?;
        }
    }
    Ok(())
}
