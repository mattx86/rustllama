//! Native egui/eframe desktop GUI for rustllama — Phase 2 (chat + model
//! management + status bar).
//!
//! This crate is deliberately **toolchain-free**: it depends only on
//! `rustllama-client` (a reqwest-only OpenAI/Ollama HTTP client) plus the egui
//! stack, so it builds with plain `cargo` — no Intel oneAPI / NVIDIA CUDA env.
//! It is *just* the UI + an async bridge to a running rustllama server. The
//! server-embedding lives in the `app/desktop` binary, which calls
//! [`run_ui`]; the GUI itself only ever talks to the server over HTTP.
//!
//! ## Scope
//! - **Chat** view: load/select a model, stream a conversation, render
//!   assistant turns as markdown.
//! - **Models** view: loaded + cached lists (load / unload / set-default /
//!   delete), a native "load from file" picker, and HuggingFace search →
//!   streamed pull with a live progress bar.
//! - A left icon-nav sidebar and an always-on status bar (RAM / per-GPU VRAM /
//!   tok-s / active model, polled from `/v1/metrics`).
//!
//! Settings, typed decisions, and quantization tooling are later phases — the
//! nav rail stubs them so their seams are visible but does not build them.
//!
//! ## The async/repaint bridge (the load-bearing gotcha)
//! egui is immediate-mode and only repaints on input or an explicit
//! [`egui::Context::request_repaint`]. All network work runs on a tokio
//! runtime built here in [`run_ui`]; each spawned task holds a cloned client, a
//! cloned `mpsc::Sender`, and a cloned [`egui::Context`], and after pushing a
//! message it calls `ctx.request_repaint()` — otherwise streamed tokens would
//! not appear until the next mouse move. The UI thread drains the receiver with
//! a non-blocking `try_recv` loop at the top of every frame. See `app.rs`.

mod app;

use app::GuiApp;

/// Launch the native GUI against `base_url` (e.g. `http://127.0.0.1:11434`).
///
/// Blocks until the window closes. Builds a multi-thread tokio runtime up front
/// and hands its [`Handle`](tokio::runtime::Handle) to the app; the runtime is
/// kept alive on this stack for the whole `run_native` call so in-flight tasks
/// keep running.
pub fn run_ui(base_url: String) -> anyhow::Result<()> {
    // Multi-thread runtime so a streaming chat and a list/load call can overlap
    // without one starving the other.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    let handle = rt.handle().clone();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("rustllama")
            .with_inner_size([1024.0, 720.0])
            .with_min_inner_size([640.0, 420.0]),
        ..Default::default()
    };

    eframe::run_native(
        "rustllama",
        options,
        // eframe 0.29's app-creator returns `Result<Box<dyn App>, _>`. The `?`
        // converts our `anyhow::Error` into the boxed error the creator expects.
        Box::new(move |cc| {
            let app = GuiApp::new(cc, base_url, handle)?;
            Ok(Box::new(app) as Box<dyn eframe::App>)
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe run_native failed: {e}"))?;

    // `rt` drops here, after the window has closed — it must outlive the whole
    // `run_native` call above so spawned tasks have a runtime to run on.
    Ok(())
}
