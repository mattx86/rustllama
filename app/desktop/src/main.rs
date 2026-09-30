//! rustllama dispatcher.
//!
//! Parses argv and routes:
//!   - `rustllama gui` → the native egui GUI (feature `gui` required)
//!   - everything else → `rustllama_cli::run`
//!
//! When the `gui` feature is built, the GUI subcommand also runs the
//! local HTTP server in-process on a tokio runtime thread before the
//! egui window opens — so the GUI's HTTP client (rustllama-client)
//! finds a live server out of the box. Standalone CLI use (`serve`,
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
                // `run_gui` owns the calling thread for the lifetime of
                // the egui window (eframe drives the event loop); the
                // embedded server runs on a dedicated tokio runtime
                // spawned on a worker thread. We don't return from
                // `run_ui` until the window closes.
                return run_gui();
            }
            #[cfg(not(feature = "gui"))]
            {
                anyhow::bail!(
                    "this binary was built without the `gui` feature; \
                     rebuild with `cargo build --release --features gui`"
                );
            }
        }
        _ => {
            // Non-GUI commands route through `rustllama-cli`, which is
            // async itself — spin up a multi-thread tokio runtime here
            // since the binary isn't `#[tokio::main]` (the gui arm needs
            // to own the main thread when the gui feature is on).
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            rt.block_on(rustllama_cli::run(cli))
        }
    }
}

#[cfg(feature = "gui")]
fn run_gui() -> anyhow::Result<()> {
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
    // main GUI thread OR the embedded server worker thread writes
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
        // independently of this process) that it is GUI-embedded: the
        // embedded start then skips the blocking first-load autotune
        // (that autotune runs as a separate subprocess with a progress
        // modal instead — see rustllama-server's autotune gate).
        std::env::set_var("RUSTLLAMA_GUI_EMBEDDED", "1");

        // Start the local HTTP server on a worker thread BEFORE the egui
        // window opens, so the GUI's first health poll succeeds. The
        // server stays alive for the lifetime of the process; the egui
        // window closing triggers process exit, which tears the server
        // down.
        //
        // The worker thread catches any panic from the runtime + `run`
        // and writes it to the GUI log; the main thread keeps running so
        // the egui window still comes up even if the server dies.
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

    // Native egui GUI. The embedded server spawned above serves the API
    // on the config port; the egui app drives it over HTTP via
    // rustllama-client. The server thread keeps running until process
    // exit (the OS reaps it when the window closes).
    let base_url = "http://127.0.0.1:11434".to_string(); // TODO: read [server].port
    rustllama_gui::run_ui(base_url)?;

    tracing::info!("rustllama GUI exiting cleanly");
    Ok(())
}
