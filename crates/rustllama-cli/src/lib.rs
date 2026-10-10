//! rustllama CLI subcommand dispatcher.

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};

mod gpu_parity;
mod session;
#[cfg(feature = "tui")]
mod tui_chat;

#[derive(Parser, Debug)]
#[command(
    name = "rustllama",
    version,
    about = "Local LLM runtime — Intel (SYCL) + NVIDIA (CUDA) GPU + CPU, OpenAI-compatible server, CLI, and GUI"
)]
pub struct Cli {
    /// Path to config file. Defaults to %APPDATA%\rustllama\config.toml.
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,

    /// Activate a named profile from the loaded config. Profiles live
    /// under `[[profiles]]` in the config file; each one overrides
    /// any subset of the top-level keys (model.path, inference.*,
    /// server.*, etc.). Useful for switching between "work" and
    /// "personal" model setups without juggling separate files.
    /// Falls back to no profile (= top-level config as-is) when unset
    /// or when the named profile doesn't exist (logs a warning).
    #[arg(long, global = true)]
    pub profile: Option<String>,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Print runtime diagnostics (oneAPI, SYCL devices, paths, config validity).
    Doctor {
        /// Run an end-to-end SYCL kernel smoke test (RMSNorm against
        /// a known input) on a host with a real Intel GPU + Level Zero /
        /// OpenCL runtime. Reports PASS/FAIL/SKIPPED so the user can
        /// verify GPU dispatch works before plumbing it into a model
        /// load. SKIPPED when no SYCL device is present (SYCL is always
        /// compiled in — there is no build flag to toggle).
        #[arg(long, default_value_t = false)]
        sycl_smoke: bool,
        /// Run the SYCL kernel parity + stability harness: every SYCL
        /// kernel family (packed matvecs, fused gate_up, the flash-
        /// attention chains) against its CPU reference on identical
        /// inputs, each probe in its own subprocess so a kernel that
        /// crashes or hangs the device (observed: DEVICE_LOST on
        /// Level Zero for IQ2/IQ3 matvecs) only takes its child down.
        /// Prints a per-kernel OK / MISCOMPUTE / KERNEL_ERR / CRASH /
        /// HANG matrix for the backend the launch PATH selects.
        #[arg(long, default_value_t = false)]
        sycl_parity: bool,
        /// Internal: run a single named parity probe in-process and
        /// print its machine-readable result line. Spawned by
        /// `--sycl-parity`; not for direct use.
        #[arg(long, hide = true)]
        sycl_parity_probe: Option<String>,
        /// Run the native CUDA kernel parity harness: every CUDA kernel
        /// (packed matvecs PTQ1_0/Q8_0/Q4_K/Q6_K, dense matvec, rmsnorm,
        /// rope, swiglu, embedding, flash-attn decode/prefill, argmax)
        /// against its CPU reference on identical inputs. In-process
        /// (NVIDIA target); SKIPs cleanly when no CUDA device is present.
        #[arg(long, default_value_t = false)]
        cuda_parity: bool,
        /// Run the CPU kernel self-parity harness: the CPU matvec SIMD
        /// (AVX-512 / AVX2 / NEON) and rayon-parallel paths against a
        /// naive scalar reference on this host's actual CPU (f32, f16,
        /// and the PTQ1_0 ternary fastdot/batched paths). In-process
        /// (no device to lose). The scalar-only quant matvecs are the
        /// reference themselves and are covered by the SYCL/CUDA modes.
        #[arg(long, default_value_t = false)]
        cpu_parity: bool,
        /// Run the native Metal (MLX) kernel parity harness: the Apple-Metal
        /// packed-quant matvecs (all K-quants, IQ grids, IQ4, MXFP, PTQ1_0) +
        /// dense matvec + rmsnorm against their CPU reference on identical
        /// inputs. In-process (Apple-Silicon target); SKIPs cleanly off Apple
        /// Silicon (no Metal device). These kernels were authored write-blind
        /// on a non-Apple host, so this is their first real validation.
        #[arg(long, default_value_t = false)]
        metal_parity: bool,
    },
    /// Print the rustllama version.
    Version,
    /// Start the OpenAI-compatible HTTP server in the foreground.
    Serve {
        /// Model(s) to load + auto-tune at startup — path(s) to `.gguf`
        /// file(s). Repeat `--model` to load several; the FIRST becomes
        /// the server default. Overrides `[model].path` from the config.
        /// When omitted, the configured model loads (or the server starts
        /// empty and you load a model via the GUI / `POST /v1/models/load`).
        #[arg(long, value_name = "GGUF")]
        model: Vec<std::path::PathBuf>,
        /// Bind address — overrides `[server].bind_addr`. Accepts an
        /// IPv4/IPv6 literal or hostname (IPv6 needs no brackets here,
        /// e.g. `--ip ::1`). Bind `0.0.0.0` (or `::`) to expose the
        /// server on the LAN; the startup banner and `/healthz` then
        /// surface reachable URLs.
        #[arg(long)]
        ip: Option<String>,
        /// Listen port — overrides `[server].port`.
        #[arg(long)]
        port: Option<u16>,
        /// Bearer-token API key clients must present as
        /// `Authorization: Bearer <token>`. Overrides `[server].api_key`
        /// and the `RUSTLLAMA_API_KEY` env var (flag > env > config).
        /// Remote (non-loopback) requests without a valid token get 401;
        /// loopback + `/healthz` always bypass. Empty/unset = open access.
        /// Set this to use token-auth clients (e.g. OpenCode) against the
        /// server — pair it with `--ip 0.0.0.0` to expose + require it on
        /// the LAN.
        #[arg(long, value_name = "TOKEN")]
        api_key: Option<String>,
        /// Disable the CPU compute tier (GPU-only placement). Overrides
        /// `[inference].cpu_enabled`. The load FAILS with a clear error if
        /// the model + KV cache don't fit GPU memory (no silent CPU spill).
        #[arg(long, default_value_t = false)]
        no_cpu: bool,
        /// Disable the GPU compute tier (CPU-only placement). Overrides
        /// `[inference].gpu_enabled`. All weights run on the CPU/RAM tier
        /// regardless of any GPU present — the mirror of `--no-cpu`.
        /// Passing both `--no-cpu` and `--no-gpu` is an error.
        #[arg(long, default_value_t = false)]
        no_gpu: bool,
        /// Comma-separated LOGICAL-PROCESSOR indices to exclude from the CPU
        /// pool, e.g. `--disabled-cpus 8,9,10,11` to keep work off the
        /// E-cores. Overrides `[inference].disabled_cpus`. Rayon workers are
        /// pinned off these cores.
        #[arg(long, value_name = "CSV")]
        disabled_cpus: Option<String>,
        /// Require weights to live in DEDICATED GPU VRAM only (no host-RAM
        /// residency / mmap); load FAILS if they don't fit. Overrides
        /// `[inference].vram_only`. No-op + warning on unified-memory GPUs.
        #[arg(long, default_value_t = false)]
        vram_only: bool,
    },
    /// Interactive chat against a running rustllama server (line REPL, or
    /// a full-screen ncurses-style TUI with `--tui`) — OR manage saved
    /// chat history via a subcommand (`chat list` / `show` / `export` /
    /// `delete` / `search`). With no subcommand, starts the interactive
    /// chat; the flags below apply only to that interactive mode.
    Chat {
        /// Manage saved chat history instead of chatting. When present,
        /// the interactive-mode flags below are ignored.
        #[command(subcommand)]
        history: Option<ChatCmd>,
        #[arg(long)]
        system: Option<String>,
        #[arg(long)]
        model: Option<String>,
        /// Resume a saved chat: seed the interactive session with its
        /// transcript so you can keep going. A numeric value resolves to
        /// a GUI conversation id; anything else to a saved-session name
        /// (see `chat list`). The saved model is used unless `--model`
        /// overrides it.
        #[arg(long, value_name = "ID_OR_NAME")]
        resume: Option<String>,
        /// Full server URL override (e.g. `http://127.0.0.1:11434`). Takes
        /// precedence over `--ip`/`--port`. Default: discover via the
        /// runtime record, else config.
        #[arg(long)]
        base_url: Option<String>,
        /// Server IP/host to connect to. Pairs with `--port`; the missing
        /// half defaults to the live server record, else config. Ignored
        /// when `--base-url` is given.
        #[arg(long)]
        ip: Option<String>,
        /// Server port to connect to. Pairs with `--ip`. Ignored when
        /// `--base-url` is given.
        #[arg(long)]
        port: Option<u16>,
        /// Launch the full-screen ncurses-style TUI (scrollable transcript
        /// pane + input box + status bar) instead of the line REPL.
        /// Requires a real terminal; falls back to the REPL when stdout
        /// isn't a TTY. (Built only with the `tui` feature — on by default.)
        #[arg(long, default_value_t = false)]
        tui: bool,
    },
    /// One-shot embedding. Sends `input` to `POST /v1/embeddings`
    /// and prints the resulting vector. Useful for `rustllama embed
    /// "text" | tee vec.json` shell pipelines and quick smoke-test
    /// of the embedding model.
    Embed {
        /// Input text. Required unless `--stdin` is set.
        #[arg(default_value = "")]
        input: String,
        /// Read input from stdin (appended to `input` arg if both supplied).
        #[arg(long)]
        stdin: bool,
        /// Embedding model id. Defaults to the server's configured
        /// `[embeddings]` slot — `rustllama-embeddings` in practice.
        #[arg(long)]
        model: Option<String>,
        /// MRL-truncate the result vector to the first N dimensions
        /// (model must be MRL-trained for the truncated vector to be
        /// semantically valid).
        #[arg(long)]
        dimensions: Option<u32>,
        /// Emit the full EmbeddingsResponse JSON instead of just the
        /// vector. Pairs with shell pipelines: `... --json | jq .data[0].embedding`.
        #[arg(long)]
        json: bool,
        /// Print every vector element. Default: print dim + first/last 8
        /// elements + L2 norm (avoids spamming the terminal with 4096
        /// floats by default).
        #[arg(long)]
        full: bool,
        /// Override server URL (default: discover via runtime record / config).
        #[arg(long)]
        base_url: Option<String>,
    },
    /// One-shot text completion. Sends a single prompt to
    /// `POST /v1/completions` (no chat template, no REPL) and
    /// prints the generated text. Designed for scripts / CI / quick
    /// smoke tests where the interactive REPL is overhead.
    Generate {
        /// Prompt text. Required unless `--stdin` is set.
        #[arg(default_value = "")]
        prompt: String,
        /// Read the prompt from stdin (appended to `prompt` arg if both supplied).
        #[arg(long)]
        stdin: bool,
        #[arg(long)]
        model: Option<String>,
        #[arg(long, default_value_t = 256)]
        max_tokens: u32,
        #[arg(long, default_value_t = 0.7)]
        temperature: f32,
        /// Top-p (nucleus sampling). `0.0` or `1.0` disables — keeps
        /// the full distribution. Defaults to the OpenAI baseline.
        #[arg(long, default_value_t = 0.95)]
        top_p: f32,
        /// Top-k. `0` disables — keeps the full vocab in the
        /// candidate set. Helpful for coding models where rare
        /// tokens matter (set to 0 or a high number).
        #[arg(long, default_value_t = 0)]
        top_k: u32,
        /// Repeat penalty. `1.0` disables. Typical values are
        /// `1.05`-`1.2`; higher discourages echoing the prompt /
        /// looping on a token.
        #[arg(long, default_value_t = 1.0)]
        repeat_penalty: f32,
        #[arg(long)]
        seed: Option<u64>,
        /// FIM suffix. When set, the request becomes a fill-in-the-middle
        /// completion (server wraps prefix+suffix in the model's FIM
        /// special tokens). Requires a model with FIM tokenizer support.
        #[arg(long)]
        suffix: Option<String>,
        /// Stream tokens to stdout as they arrive (long outputs aren't
        /// a silent wait). Mutually exclusive with `--json`.
        #[arg(long)]
        stream: bool,
        /// Emit the full CompletionResponse JSON instead of just text.
        /// Pairs with shell pipelines: `rustllama generate ... --json | jq .usage`.
        #[arg(long)]
        json: bool,
        /// Override server URL (default: discover via runtime record / config).
        #[arg(long)]
        base_url: Option<String>,
    },
    /// Manage models: cache (list/rm/inspect/pull), the running server's
    /// loaded set (load/unload/default), the config default (use), and
    /// benchmarking (bench).
    #[command(subcommand)]
    Model(ModelCmd),
    /// Manage the on-disk config.
    #[command(subcommand)]
    Config(ConfigCmd),
    /// Run the autotuner for the active device + model. `tune --show`
    /// prints the current tuner-cache state (the former `tuning show`).
    Tune {
        #[arg(long)]
        device: Option<String>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        thorough: bool,
        #[arg(long)]
        clear: bool,
        /// Print the current tuner-cache state for the active device
        /// (device fingerprint, cache path, per-model winners, and the
        /// live `[tuning].auto_apply_*` flags) instead of running a
        /// sweep — the former `tuning show`. Mutually exclusive with the
        /// sweep-mode flags.
        #[arg(long)]
        show: bool,
        /// Synthetic prompt length (tokens) shared by every measured
        /// sweep's prefill phase. Bigger = longer prefill, more
        /// representative of real workloads. Default 64. (The
        /// batch-size sweep uses its own fixed long prompt regardless,
        /// since it measures prefill throughput specifically.)
        #[arg(long, default_value_t = 64)]
        prompt_tokens: u32,
        /// Decode tokens per measurement run, shared by every measured
        /// sweep. Quick by default (16) — decode tok/s stabilizes within
        /// a few tokens, so 16 ranks candidates reliably; `--thorough`
        /// bumps this to 32. Pass a bigger value to override upward.
        #[arg(long, default_value_t = 16)]
        decode_tokens: u32,
        /// Repeats per candidate, shared by every measured sweep; the
        /// median score of these runs picks the winner. Quick by default
        /// (3); `--thorough` bumps this to 7. Pass a bigger value to
        /// override upward.
        #[arg(long, default_value_t = 3)]
        repeats: u32,
        /// After the kernel-LWS sweep completes and the cache is
        /// written, load the engine and run a synthetic decode pass
        /// to measure **end-to-end tok/s** with the populated cache.
        /// Reports the headline number the plan calls for as the win
        /// metric — closer to user-experienced throughput than the
        /// per-shape kernel µs the acceptance gate uses.
        ///
        /// The check loads the configured model once + runs 1 untimed
        /// warmup + `--repeats` timed generations of `--decode-tokens`
        /// each. Adds ~30 s on a 7B model; default off.
        #[arg(long)]
        measure_tok_s: bool,
        /// Static placement-sweep analyzer: enumerates `n_gpu_layers`
        /// candidates that fit a given VRAM budget for the configured
        /// model, prints estimated VRAM cost per candidate and the
        /// recommended setting. Doesn't measure tok/s yet — that's a
        /// follow-up. Useful answer to "what should I set
        /// `[inference].n_gpu_layers` to for my hardware?" without
        /// guessing.
        #[arg(long)]
        placement: bool,
        /// VRAM budget in MiB for the placement sweep. Defaults to
        /// `0` = AUTO-DETECT the dispatch GPU's VRAM (CUDA → SYCL),
        /// so the sweep budgets against the REAL card. The old fixed
        /// 4 GiB default silently capped placement on any larger GPU —
        /// e.g. only ~20 of 28 layers on a 16 GiB card, leaving the
        /// rest on CPU in the decode critical path. Pass an explicit
        /// value to override (e.g. to reserve headroom on a shared
        /// box). Honored only with `--placement`.
        #[arg(long, default_value_t = 0)]
        vram_mb: u64,
        /// VRAM headroom in MiB kept free after the placement sweep
        /// fits weights + KV cache. Defaults to 256 MiB. Honored only
        /// with `--placement`.
        #[arg(long, default_value_t = 256)]
        vram_headroom_mb: u64,
        /// Context window assumed when sizing the KV-cache cost in
        /// the placement sweep. Defaults to the loaded config's
        /// `[inference].ctx_size`. Honored only with `--placement`.
        #[arg(long)]
        placement_ctx: Option<u32>,
        /// Run the dynamic measurement half of the placement sweep:
        /// load the model once, vary `n_gpu_layers` per candidate
        /// that fits the budget, time decode tok/s, pick the winner
        /// by measured throughput (not just "most-GPU that fits"
        /// — which can be slower on shared-LPDDR integrated GPUs).
        /// Slow: loads the full model + runs N candidates × `--repeats`
        /// decode runs. Honored only with `--placement`.
        #[arg(long)]
        measure: bool,
        /// Batch-size sweep: vary `[inference].batch_size` (the
        /// prefill chunk in tokens) across the candidate list and
        /// pick the value that maximizes prefill throughput on a
        /// synthetic long prompt. Mutually exclusive with the
        /// kernel-LWS sweep + `--placement`.
        #[arg(long)]
        batch_size: bool,
        /// Candidate batch sizes for the sweep, comma-separated.
        /// Defaults match the plan's prescribed grid. Honored only
        /// with `--batch-size`.
        #[arg(long, default_value = "128,256,512,1024,2048")]
        batch_candidates: String,
        /// CPU threads sweep: vary `[inference].threads` across a
        /// candidate grid and pick the value that maximizes decode
        /// tok/s. Mutually exclusive with `--placement` /
        /// `--batch-size` / the default kernel-LWS sweep.
        ///
        /// The CPU kernel layer's parallel matvec (rayon-backed) is
        /// opt-in per-call-site today (`matvec_f16_w_f32_a_parallel`).
        /// Until every hot matvec is flipped to the parallel
        /// variant, the sweep mostly measures rayon pool-init
        /// overhead — useful as a smoke test but not the full perf
        /// picture. See `install_thread_pool` in `rustllama-kernels-cpu`.
        #[arg(long)]
        threads: bool,
        /// Candidate thread counts for the sweep, comma-separated.
        /// When empty, derives the plan's grid `{1, 2, 4, ncores/2,
        /// ncores, 2*ncores}` from the host's CPU count.
        #[arg(long, default_value = "")]
        threads_candidates: String,
        /// KV-dtype sweep: try `f32 / q8_0 / tq1 / tq2 / tq4 / tq8 /
        /// nvfp4` (or a subset via `--kv-dtype-candidates`) and pick
        /// the kv_dtype that maximizes decode tok/s. Reloads the model
        /// per candidate — KV cache shape changes per dtype so a hot
        /// swap is not possible. Writes the winner to the tuner cache
        /// under `kv_dtype`; engine auto-applies on next load when
        /// `[tuning].auto_apply_kv_dtype = true`.
        #[arg(long)]
        kv_dtype: bool,
        /// KV-dtype candidates (comma-separated). Default covers every
        /// v1-supported value. Honored only with `--kv-dtype`.
        #[arg(long, default_value = "f32,q8_0,q4_0,tq1,tq2,tq4,tq8,nvfp4,mxfp4,mxfp6,mxfp8")]
        kv_dtype_candidates: String,
        /// Flash-attention on/off sweep. Reuses one loaded engine
        /// (flash is a hot toggle). Writes the winner under
        /// `flash_attention`.
        #[arg(long)]
        flash_attention: bool,
        /// KV-cache layout sweep (`contiguous` vs `paged`). Reloads
        /// per candidate (cache shape changes). Paged today only
        /// supports `kv_dtype = f32`; non-F32 paged falls through as
        /// a failed candidate.
        #[arg(long)]
        kv_layout: bool,
        /// KV-layout candidates (comma-separated). Default
        /// `"contiguous,paged"`. Honored only with `--kv-layout`.
        #[arg(long, default_value = "contiguous,paged")]
        kv_layout_candidates: String,
        /// E2: sweep the flash-attention KV-length threshold to find
        /// the host-specific break-even between flash-decode and
        /// standard attention. Persists winner under `flash_kv_min`.
        /// (Decode length is clamped to 2× the largest candidate at
        /// run-time so the threshold actually affects dispatch.)
        #[arg(long)]
        flash_kv_min: bool,
        #[arg(long, default_value = "64,128,256,512,1024")]
        flash_kv_min_candidates: String,
        /// E2: sweep the prefix-cache snapshot pool depth. Persists
        /// winner under `prefix_cache_max_snapshots`.
        #[arg(long)]
        prefix_snapshots: bool,
        #[arg(long, default_value = "1,2,4,8")]
        prefix_snapshots_candidates: String,
        /// E2: sweep the paged-KV page size (token-count per page).
        /// Only meaningful when `[inference].kv_cache_layout = "paged"`.
        /// Persists winner under `kv_page_size`.
        #[arg(long)]
        kv_page_size: bool,
        #[arg(long, default_value = "8,16,32,64")]
        kv_page_size_candidates: String,
        /// E2: sweep KV_TILE for the SYCL flash-attn-v3 decode kernel.
        /// Persists winner under `flash_v3_kv_tile`. Only meaningful when
        /// a SYCL device is present; with none, every candidate measures
        /// the same CPU path.
        #[arg(long)]
        flash_v3_kv_tile: bool,
        #[arg(long, default_value = "16,32,64")]
        flash_v3_kv_tile_candidates: String,
        /// MoE placement + q★ co-execution sweep: A/B `uniform`,
        /// `experts_cpu`, and CPU/GPU expert-split candidates by
        /// measured decode tok/s (MoE models only). A non-uniform
        /// winner must beat uniform by ≥5% median. Persists winner
        /// under `moe_placement` + `moe_gpu_split_permille`; applied
        /// automatically on the next load (gated by
        /// `[tuning].auto_apply_moe_placement`).
        #[arg(long)]
        moe_placement: bool,
        /// MTP / NextN self-speculation A/B sweep: measure decode tok/s
        /// with MTP self-speculative decode off vs on and pick the
        /// faster arm. Hybrid + NextN-head models only; a non-capable
        /// model records `false` (nothing to tune). Persists winner
        /// under `speculative_mtp`; auto-applied on next load when
        /// `[tuning].auto_apply_speculative_mtp = true`.
        #[arg(long)]
        speculative_mtp: bool,
        /// Repeats per arm for the MTP sweep (median picks the winner).
        #[arg(long, default_value_t = 3)]
        speculative_mtp_repeats: u32,
        /// Chunked-parallel SSM (DeltaNet) prefill A/B sweep: measure
        /// prefill tok/s with the chunked path off vs on and pick the
        /// faster arm. Hybrid models only; a non-hybrid model records
        /// `false` (nothing to tune). Persists winner under
        /// `ssm_prefill_chunked`; auto-applied on next load when
        /// `[tuning].auto_apply_ssm_prefill_chunked = true`.
        #[arg(long)]
        ssm_prefill_chunked: bool,
        /// Repeats per arm for the chunked-SSM-prefill sweep.
        #[arg(long, default_value_t = 3)]
        ssm_prefill_chunked_repeats: u32,
        /// Decision-calibration (temperature scaling for typed
        /// decisions). Loads the model, fits a calibration temperature
        /// over the labeled decision set, and persists it to the tuner
        /// cache. Primarily exists so `tune --all` can run this stage
        /// as an isolated subprocess (fresh SYCL/USM state); rarely
        /// invoked directly.
        #[arg(long)]
        decision_calibrate: bool,
        /// Per-device perf measurement (each GPU + the CPU tier's short-
        /// synthetic decode tok/s), persisted under `per_device_perf` to
        /// feed the heat placement planner. Primarily exists so `tune --all`
        /// can run this model-reloading stage as an isolated subprocess
        /// (fresh SYCL/USM state — see Stage 1b); rarely invoked directly.
        /// Honors the shared `--prompt-tokens` / `--decode-tokens` /
        /// `--repeats` sizing.
        #[arg(long)]
        per_device_perf: bool,
        /// On-device GPU-kernel validation: run the parity probes (CUDA
        /// tensor-core GEMM, SYCL XMX) and persist a pass/fail verdict per
        /// kernel under `kernel_verdicts`, which the dispatch layer reads to
        /// AUTO-ENABLE each specialized path only where it matches the CPU
        /// reference on this machine. Primarily a `tune --all` sub-stage
        /// (Stage 1c); replaces the removed `RUSTLLAMA_FP4_TC` /
        /// `_FP8_WGMMA` / `_SYCL_XMX` env gates.
        #[arg(long)]
        validate_kernels: bool,
        /// **Comprehensive autotune**: run every sweep in coordinate-
        /// descent order — kernel LWS → kv_dtype → flash_attention →
        /// kv_cache_layout → placement → batch_size → threads — and
        /// persist every winner to the cache. Earlier winners feed the
        /// next stage's measurement config so the chosen value is
        /// best-given-prior-stages (greedy; not joint-optimal but
        /// converges fast on a quiet host). Total runtime ~5–30 min
        /// per model depending on `--thorough`.
        #[arg(long)]
        all: bool,
        /// With `--all`: skip stages whose winner is already cached
        /// for this `(device, model)`. Re-running `tune --all` after
        /// a previous successful run becomes near-instant — only
        /// stages that didn't complete (or whose cache entry was
        /// invalidated by a driver/device change) actually run.
        /// Defaults to `false` for backward compatibility. Pass
        /// `--skip-cached` to enable. Use `--force` to invalidate
        /// and re-run everything.
        #[arg(long)]
        skip_cached: bool,
        /// With `--all --skip-cached`: invalidate cached winners
        /// for the current `(device, model)` and re-run every stage
        /// from scratch. Useful after a driver upgrade if you
        /// suspect the cached values are stale.
        #[arg(long)]
        force: bool,
    },
    /// Re-quantize a GGUF model file to a smaller target dtype.
    /// Reads any of the 26 supported quant formats; writes any of
    /// the encoder-supported targets (Q4_0/1, Q5_0/1, Q8_0/1, Q2/3/4/5/6/8_K,
    /// TQ1_0, TQ2_0, IQ4_NL, IQ4_XS, F32, F16, BF16).
    ///
    /// Single-pass streaming: each tensor dequants → re-encodes →
    /// writes one at a time. Peak memory is one tensor's
    /// worth, not the whole model. 1D "norm"-style tensors are
    /// passed through at source dtype automatically (override via
    /// `--no-passthrough` if you really want to re-encode them).
    ///
    /// With `--to-mlx`, the output is instead an **Apple MLX** model
    /// *directory* (mlx-lm affine format): each 2-D weight is
    /// group-affine quantized (`--mlx-bits` / `--mlx-group-size`) into
    /// the `<name>.weight`/`.scales`/`.biases` triple, 1-D tensors pass
    /// through, and a `config.json` (with the `quantization` block) +
    /// copied `tokenizer.json` are written beside `model.safetensors`.
    /// `--target`/`--recipe`/`--apex`/`--imatrix` are ignored in this
    /// mode.
    Quantize {
        /// Source GGUF file.
        #[arg(long)]
        input: String,
        /// Destination. A GGUF file by default (created/truncated); with
        /// `--to-mlx` this is an MLX model **directory** (created).
        #[arg(long)]
        output: String,
        /// Write an Apple MLX affine model directory instead of a GGUF
        /// file. Changes `--output` semantics to a directory.
        #[arg(long, default_value_t = false)]
        to_mlx: bool,
        /// MLX affine bits per weight (`--to-mlx` only). One of
        /// 2,3,4,5,6,8. Default 4 (the mlx-lm default).
        #[arg(long, default_value_t = 4)]
        mlx_bits: u32,
        /// MLX affine group size (`--to-mlx` only) — elements per
        /// (scale, bias) group along the input dim. One of 32,64,128.
        /// Default 64 (the mlx-lm default).
        #[arg(long, default_value_t = 64)]
        mlx_group_size: usize,
        /// Default target dtype for all quantizable weight tensors
        /// that no recipe or APEX rule matches. Accepts canonical
        /// ggml names (case-insensitive): q4_0, q4_1, q5_0, q5_1,
        /// q8_0, q8_1, q2_k, q3_k, q4_k, q5_k, q6_k, q8_k, tq1_0,
        /// tq2_0, iq4_nl, iq4_xs, f32, f16, bf16. Required for the
        /// GGUF→GGUF path; ignored (and optional) with `--to-mlx`.
        #[arg(long)]
        target: Option<String>,
        /// Disable the 1D-tensor passthrough heuristic. Without
        /// this, norms / biases stay at source dtype.
        #[arg(long, default_value_t = false)]
        no_passthrough: bool,
        /// Keep the LM head (`output.weight` / `lm_head.weight` /
        /// `head.weight`) at its source dtype. Mirrors llama.cpp's
        /// `--leave-output-tensor` and the `_M` variant convention.
        #[arg(long, default_value_t = false)]
        keep_output: bool,
        /// Path to a recipe file: one rule per line as
        /// `<pattern> <dtype>` where `<pattern>` is a tensor name
        /// (exact) or a glob with `*` wildcards. Walked top-to-
        /// bottom; first matching rule per tensor wins. Comments
        /// start with `#`. Stacks with `--apex` (APEX rules are
        /// installed first, user rules override).
        #[arg(long)]
        recipe: Option<String>,
        /// Apply a built-in APEX (Adaptive Precision for EXpert
        /// models) profile. Valid tiers, in decreasing size order:
        /// `i-quality`, `quality`, `balanced`, `mini`, `nano`.
        /// Generates per-tensor + per-layer rules based on the
        /// source model's layer count. Pairs with `--target` —
        /// tensors that no APEX rule matches fall back to
        /// `--target`.
        #[arg(long)]
        apex: Option<String>,
        /// Path to an importance-matrix (`.rlim`) file produced by the
        /// `imatrix` subcommand. When set, Q2_K/Q4_K/Q5_K/IQ1_S tensors
        /// are encoded with per-input-column importance weighting,
        /// which is essential for usable ≤2-bpw quants (e.g. IQ1_S).
        /// With `RUSTLLAMA_IQ_GPU=1` the IQ1_S weighting runs on the GPU
        /// (weighted grid-search kernel); K-quants weight on CPU. Without
        /// the GPU opt-in it uses the AVX2/scalar CPU weighted encode.
        #[arg(long)]
        imatrix: Option<String>,
    },
    /// Generate an importance matrix (`.rlim`) by running a calibration
    /// corpus through a model and accumulating per-input-column
    /// mean-squared activation per weight tensor. Feed the result to
    /// `quantize --imatrix` for usable low-bpw quants (IQ1_S/Q2_K),
    /// which are incoherent without importance weighting.
    Imatrix {
        /// Model to calibrate against. Any dtype works; a
        /// higher-precision source yields a cleaner imatrix.
        #[arg(long)]
        model: String,
        /// Calibration text file (UTF-8). Tokenized, then run through
        /// prefill in `--ctx-size` windows.
        #[arg(long)]
        calibration: String,
        /// Output `.rlim` path.
        #[arg(long)]
        output: String,
        /// Max calibration tokens to process. Default 4096; enough to
        /// stabilize column statistics without a long run.
        #[arg(long, default_value_t = 4096)]
        max_tokens: u32,
        /// Prefill window size (tokens per forward chunk). Default 2048.
        #[arg(long, default_value_t = 2048)]
        ctx_size: usize,
    },
    /// Calibrate a K-cache mean-centering bias sidecar (PrismML
    /// `kv_bar` GGUF format). Runs a prompt set through the model
    /// with q4_0 KV, accumulates per-(layer, kv-head, channel) K
    /// means at cache-write time — in the whitened basis when KV
    /// whitening is active — and writes `<model stem>.kvbias.gguf`.
    /// The serve path auto-discovers that sidecar on the next load.
    /// Subtraction is exactly softmax-invariant; the win is lower
    /// q4_0 quantization error on K channels with a large mean.
    KvCalibrate {
        /// Model GGUF path. Defaults to the configured `[model].path`.
        #[arg(long)]
        model: Option<std::path::PathBuf>,
        /// Prompt file, one prompt per line (UTF-8). Defaults to a
        /// small built-in mixed-domain corpus.
        #[arg(long)]
        prompts: Option<std::path::PathBuf>,
        /// Output sidecar path. Defaults to `<model stem>.kvbias.gguf`
        /// beside the model.
        #[arg(long)]
        output: Option<std::path::PathBuf>,
        /// Context window for the calibration engine. Default 4096.
        #[arg(long, default_value_t = 4096)]
        ctx_size: usize,
    },
    /// Launch the Tauri GUI.
    Gui,
    /// Run as a Language Server Protocol bridge over stdio. Editors
    /// that target this binary (Helix, Zed, Neovim with built-in LSP)
    /// get AI inline completion via the existing FIM endpoint. Point
    /// `--base-url` at a running `rustllama serve` instance.
    Lsp {
        /// Base URL of the rustllama HTTP server.
        #[arg(long, default_value = "http://127.0.0.1:11434")]
        base_url: String,
        /// Model id sent in the `model` field of `/v1/completions`. Empty
        /// means "use server default".
        #[arg(long, default_value = "")]
        model: String,
        /// Max tokens per inline completion. Editors typically want a
        /// few hundred at most.
        #[arg(long, default_value_t = 128)]
        max_tokens: u32,
        /// Sampling temperature. 0.0 (default) is greedy / deterministic
        /// — the right call for code.
        #[arg(long, default_value_t = 0.0)]
        temperature: f32,
    },
}

#[derive(Subcommand, Debug)]
pub enum ModelCmd {
    /// List cached models.
    List,
    /// Download a GGUF from HuggingFace into the local cache.
    Pull {
        /// `owner/repo:filename` reference.
        hub_ref: String,
    },
    /// Remove a cached model file from disk.
    Rm { hub_ref_or_path: String },
    /// Set the active model in the config (`[model].path`).
    Use { hub_ref_or_path: String },
    /// Print GGUF metadata for a model file: architecture, quantization,
    /// context length, layer count, vocab, tensor stats. Use `--tensors`
    /// to dump the full tensor table.
    Inspect {
        /// Local `.gguf` path. (Not a hub ref — use `model pull` first.)
        path: std::path::PathBuf,
        /// Print every tensor name, shape, and dtype.
        #[arg(long)]
        tensors: bool,
        /// Emit machine-readable JSON instead of the human-readable
        /// summary. Handy for the future GUI's model card.
        #[arg(long)]
        json: bool,
    },
    /// Load a model into the running server WITHOUT making it the
    /// default (`POST /v1/models/load`). Route to it explicitly via the
    /// `model` field on requests, or promote it later with
    /// `model default`. The target is a HuggingFace ref
    /// (`owner/repo:filename`, cached — `model pull` first) or an
    /// absolute `.gguf` path.
    Load {
        /// Hub ref `owner/repo:filename` or absolute path to a `.gguf`.
        target: String,
        /// Override server URL (default: discover via runtime record / config).
        #[arg(long)]
        base_url: Option<String>,
        /// Context length for the freshly-loaded model.
        #[arg(long)]
        ctx_size: Option<usize>,
        /// KV-cache dtype: `f32` (default) or `q8_0`.
        #[arg(long)]
        kv_dtype: Option<String>,
    },
    /// Unload a model from the running server, freeing its RAM/VRAM. The
    /// current default model can't be unloaded — promote another with
    /// `model default <id>` first.
    Unload {
        /// The loaded model's id (see `model list` / `model default`).
        id: String,
        /// Override server URL (default: discover via runtime record / config).
        #[arg(long)]
        base_url: Option<String>,
    },
    /// Change the running server's default model, loading it first if
    /// needed (`POST /v1/models/load` then `POST /v1/models/default`).
    /// In-flight requests against the old default finish normally. The
    /// target is a HuggingFace ref (`owner/repo:filename`, cached) or an
    /// absolute `.gguf` path.
    Default {
        /// Hub ref `owner/repo:filename` or absolute path to a `.gguf`.
        target: String,
        /// Override server URL (default: discover via runtime record / config).
        #[arg(long)]
        base_url: Option<String>,
        /// Context length for the freshly-loaded model.
        #[arg(long)]
        ctx_size: Option<usize>,
        /// KV-cache dtype: `f32` (default) or `q8_0`.
        #[arg(long)]
        kv_dtype: Option<String>,
    },
    /// Synthetic prefill + decode benchmark against the configured
    /// model. Reports prefill latency, decode tok/s, and total wall
    /// time so the user can empirically check whether perf knobs
    /// (`flash_attention`, `kv_dtype`, SYCL dispatch) actually help.
    Bench {
        /// Override `[model].path` for this run only — bench a specific
        /// cached GGUF without editing config.toml. Composes with the
        /// flash/kv_dtype/ctx_size overrides.
        #[arg(long)]
        model: Option<String>,
        /// Toggle flash-attention for this run; defaults to the
        /// config value.
        #[arg(long)]
        flash: Option<bool>,
        /// Override `[inference].kv_dtype`. Useful for A/B'ing
        /// memory + speed across f32 / q8_0 / tq* / nvfp4 without
        /// editing the config.
        #[arg(long)]
        kv_dtype: Option<String>,
        /// Context window for the benchmark. Defaults to
        /// `[inference].ctx_size`.
        #[arg(long)]
        ctx_size: Option<usize>,
        /// Number of synthetic prompt tokens to prefill. Default 512.
        #[arg(long, default_value_t = 512)]
        prompt_tokens: u32,
        /// Number of tokens to decode. Default 64.
        #[arg(long, default_value_t = 64)]
        decode_tokens: u32,
        /// Number of repeats; reports min/median/max of each stage.
        /// Default 3.
        #[arg(long, default_value_t = 3)]
        repeats: u32,
    },
}

/// Subcommands for `rustllama chat` that manage saved chat history —
/// a unified view over BOTH stores: REPL-saved sessions (created via
/// `/save_session`, keyed by name) and the GUI's sqlite conversations
/// (the `/api/conversations` store, keyed by numeric id). Commands that
/// take an identifier resolve a purely-numeric value to a conversation
/// id and anything else to a session name. (Conversation access needs
/// the `history` feature; on a slim build those paths report so.)
#[derive(Subcommand, Debug)]
pub enum ChatCmd {
    /// List saved chat history — both saved sessions and conversations,
    /// newest first.
    List {
        /// Emit machine-readable JSON instead of the tabular view.
        #[arg(long)]
        json: bool,
    },
    /// Print one chat's transcript. A numeric ID is a conversation; any
    /// other value is a saved-session name.
    Show {
        /// Conversation id (numeric) or saved-session name.
        id: String,
        /// Render as plain text (default) or markdown.
        #[arg(long, value_parser = ["text", "markdown"], default_value = "text")]
        format: String,
    },
    /// Search saved sessions for a substring (case-insensitive). Prints
    /// every (session, message) hit so users can grep their chat history.
    Search {
        query: String,
        /// Limit to sessions whose name contains this substring.
        #[arg(long)]
        name: Option<String>,
        /// Match case-sensitively. Default is case-insensitive.
        #[arg(long)]
        case_sensitive: bool,
    },
    /// Delete a chat. A numeric ID is a conversation; any other value is
    /// a saved-session name. Pass the exact identifier to avoid accidents.
    Delete {
        /// Conversation id (numeric) or saved-session name.
        id: String,
    },
    /// Export one chat to a file. A numeric ID is a conversation; any
    /// other value is a saved-session name. Markdown is the friendly
    /// transcript; JSON is the raw on-disk shape.
    Export {
        /// Conversation id (numeric) or saved-session name.
        id: String,
        #[arg(long, value_parser = ["markdown", "md", "json"], default_value = "markdown")]
        format: String,
        /// Output path. If omitted, writes to stdout.
        #[arg(long)]
        out: Option<std::path::PathBuf>,
    },
}

#[derive(Subcommand, Debug)]
pub enum ConfigCmd {
    /// Print the resolved config.
    Show,
    /// Get a value at a dotted path (e.g. `inference.n_gpu_layers`).
    Get { key: String },
    /// Set a value at a dotted path.
    Set { key: String, value: String },
    /// Open the config in `$EDITOR` (falls back to notepad on Windows).
    Edit,
    /// Print the resolved config-file path.
    Path,
    /// Validate the config without starting the server. Surfaces
    /// profile-name typos, model-path mutual-exclusion errors, and
    /// other schema mismatches before they bite at startup.
    Validate {
        /// Optional profile name to apply on top of the base config
        /// before validating (matches the global `--profile` flag).
        #[arg(long)]
        profile: Option<String>,
    },
}

pub async fn run(cli: Cli) -> anyhow::Result<()> {
    // Register GPU-runtime DLL search paths first, before any SYCL
    // symbol binds (delay-loaded). Lets the CLI serve path find the
    // oneAPI DLLs without run-sycl.bat. No-op off-Windows / no toolkit.
    rustllama_runtime::ensure_gpu_dll_search_paths();
    init_tracing();

    // GPU path wants PACKED (Raw) weights: the USM pre-upload + the fast packed
    // matvec kernel only cover `*Raw` tensors, so when a GPU is present we keep
    // weights raw process-wide (unless the user already set the env). Set here
    // at the CLI ENTRY — before `serve` spawns the mandatory-autotune
    // `tune --all` subprocess — so the SUBPROCESS inherits it too; otherwise its
    // measurement loads dequant to F16 and hybrid (DeltaNet) decode crawls on
    // CPU. CPU-only hosts keep the F16 default (faster on CPU). See
    // project-hybrid-forward-perf.
    if std::env::var_os("RUSTLLAMA_KEEP_QUANT_RAW").is_none() {
        let gpu_present = rustllama_kernels_sycl::device_count()
            .map(|n| n > 0)
            .unwrap_or(false)
            || rustllama_kernels_cuda::device_count() > 0;
        if gpu_present {
            std::env::set_var("RUSTLLAMA_KEEP_QUANT_RAW", "1");
        }
    }

    // First thing: resolve paths (auto-detects portable mode from env
    // var or sibling `portable.flag`). Doing this before any path
    // consumer kicks in ensures everyone sees the same roots.
    let paths = rustllama_runtime::paths();
    if paths.portable {
        tracing::info!(
            data_root = %paths.cache_dir.parent().map(|p| p.display().to_string()).unwrap_or_default(),
            "portable mode active"
        );
    }

    // Install the crash-log panic hook. Tagged with the subcommand name
    // so post-mortem inspectors can tell `serve` panics from `lsp` ones.
    let tag = match &cli.command {
        Command::Version => "version",
        Command::Doctor { .. } => "doctor",
        Command::Serve { .. } => "serve",
        Command::Chat { .. } => "chat",
        Command::Generate { .. } => "generate",
        Command::Embed { .. } => "embed",
        Command::Model(_) => "model",
        Command::Config(_) => "config",
        Command::Tune { .. } => "tune",
        Command::Quantize { .. } => "quantize",
        Command::Imatrix { .. } => "imatrix",
        Command::KvCalibrate { .. } => "kv-calibrate",
        Command::Gui => "gui",
        Command::Lsp { .. } => "lsp",
    };
    rustllama_runtime::crash::install_panic_hook(env!("CARGO_PKG_VERSION"), tag);

    let config_path = cli
        .config
        .clone()
        .or_else(rustllama_config::default_config_path)
        .ok_or_else(|| anyhow::anyhow!("cannot resolve config path"))?;

    match cli.command {
        Command::Version => {
            println!("rustllama {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Command::Doctor {
            sycl_smoke,
            sycl_parity,
            sycl_parity_probe,
            cuda_parity,
            cpu_parity,
            metal_parity,
        } => {
            if let Some(probe) = sycl_parity_probe {
                crate::gpu_parity::run_probe(&probe)
            } else if sycl_parity {
                crate::gpu_parity::run_parent()
            } else if cuda_parity {
                crate::gpu_parity::run_cuda_parity()
            } else if cpu_parity {
                crate::gpu_parity::run_cpu_parity()
            } else if metal_parity {
                crate::gpu_parity::run_metal_parity()
            } else {
                doctor(&config_path, sycl_smoke).await
            }
        }
        Command::Serve {
            model,
            ip,
            port,
            api_key,
            no_cpu,
            no_gpu,
            disabled_cpus,
            vram_only,
        } => {
            serve(
                &config_path,
                model,
                ip,
                port,
                api_key,
                no_cpu,
                no_gpu,
                disabled_cpus,
                vram_only,
                cli.profile.as_deref(),
            )
            .await
        }
        Command::Chat {
            history,
            system,
            model,
            resume,
            base_url,
            ip,
            port,
            tui,
        } => {
            // A history subcommand (`chat list` / `show` / …) manages
            // saved chats instead of starting the interactive session.
            if let Some(cmd) = history {
                return run_chat_history(cmd);
            }
            // `--resume` seeds the interactive session with a saved chat's
            // transcript; the saved model fills in when `--model` is unset.
            let (initial, model) = match resume {
                Some(id_or_name) => {
                    let (msgs, model_hint) = load_resumable_chat(&id_or_name)?;
                    println!("resumed `{id_or_name}` ({} messages)", msgs.len());
                    (msgs, model.or(model_hint))
                }
                None => (Vec::new(), model),
            };
            // Fold `--ip`/`--port` into a concrete base-URL override
            // (`--base-url` wins if given); `None` falls through to the
            // usual runtime-record / config discovery.
            let base_url = resolve_chat_base_url(&config_path, base_url, ip, port);
            if tui {
                // TUI needs a real terminal; fall back to the REPL when
                // stdout is piped/redirected.
                use std::io::IsTerminal as _;
                let is_tty = std::io::stdout().is_terminal() && std::io::stdin().is_terminal();
                #[cfg(feature = "tui")]
                {
                    if is_tty {
                        return crate::tui_chat::run(
                            &config_path,
                            system,
                            model,
                            base_url,
                            initial,
                        )
                        .await;
                    }
                    eprintln!(
                        "chat --tui: stdout/stdin is not a TTY; falling back to the line REPL."
                    );
                }
                #[cfg(not(feature = "tui"))]
                {
                    let _ = is_tty;
                    eprintln!(
                        "chat --tui: this binary was built without the `tui` feature; \
                         using the line REPL."
                    );
                }
            }
            chat_repl(&config_path, system, model, base_url, initial).await
        }
        Command::Generate {
            prompt,
            stdin,
            model,
            max_tokens,
            temperature,
            top_p,
            top_k,
            repeat_penalty,
            seed,
            suffix,
            stream,
            json,
            base_url,
        } => {
            generate_oneshot(
                &config_path,
                prompt,
                stdin,
                model,
                max_tokens,
                temperature,
                top_p,
                top_k,
                repeat_penalty,
                seed,
                suffix,
                stream,
                json,
                base_url,
            )
            .await
        }
        Command::Embed {
            input,
            stdin,
            model,
            dimensions,
            json,
            full,
            base_url,
        } => {
            embed_oneshot(
                &config_path,
                input,
                stdin,
                model,
                dimensions,
                json,
                full,
                base_url,
            )
            .await
        }
        Command::Model(ModelCmd::List) => models_list(),
        Command::Model(ModelCmd::Pull { hub_ref }) => pull(&hub_ref).await,
        Command::Model(ModelCmd::Rm { hub_ref_or_path }) => models_rm(&hub_ref_or_path),
        Command::Model(ModelCmd::Use { hub_ref_or_path }) => {
            models_use(&config_path, &hub_ref_or_path)
        }
        Command::Model(ModelCmd::Inspect {
            path,
            tensors,
            json,
        }) => models_inspect(&path, tensors, json),
        Command::Model(ModelCmd::Load {
            target,
            base_url,
            ctx_size,
            kv_dtype,
        }) => {
            // `load` = register on the server without promoting to default.
            swap_model(
                &config_path,
                &target,
                base_url.as_deref(),
                ctx_size,
                kv_dtype.as_deref(),
                true, // no_default
            )
            .await
        }
        Command::Model(ModelCmd::Default {
            target,
            base_url,
            ctx_size,
            kv_dtype,
        }) => {
            // `default` = load (if needed) then promote to the server default.
            swap_model(
                &config_path,
                &target,
                base_url.as_deref(),
                ctx_size,
                kv_dtype.as_deref(),
                false, // promote to default
            )
            .await
        }
        Command::Model(ModelCmd::Unload { id, base_url }) => {
            unload_model(&config_path, &id, base_url.as_deref()).await
        }
        Command::Model(ModelCmd::Bench {
            model,
            flash,
            kv_dtype,
            ctx_size,
            prompt_tokens,
            decode_tokens,
            repeats,
        }) => bench(
            &config_path,
            cli.profile.as_deref(),
            model.as_deref(),
            flash,
            kv_dtype.as_deref(),
            ctx_size,
            prompt_tokens,
            decode_tokens,
            repeats,
        ),
        Command::Config(ConfigCmd::Show) => config_show(&config_path),
        Command::Config(ConfigCmd::Path) => {
            println!("{}", config_path.display());
            Ok(())
        }
        Command::Config(ConfigCmd::Get { key }) => config_get(&config_path, &key),
        Command::Config(ConfigCmd::Set { key, value }) => config_set(&config_path, &key, &value),
        Command::Config(ConfigCmd::Edit) => config_edit(&config_path),
        Command::Config(ConfigCmd::Validate { profile }) => {
            config_validate(&config_path, profile.as_deref().or(cli.profile.as_deref()))
        }
        Command::Tune {
            device,
            model,
            thorough,
            clear,
            show,
            prompt_tokens,
            decode_tokens,
            repeats,
            measure_tok_s,
            placement,
            vram_mb,
            vram_headroom_mb,
            placement_ctx,
            measure,
            batch_size,
            batch_candidates,
            threads,
            threads_candidates,
            kv_dtype,
            kv_dtype_candidates,
            flash_attention,
            kv_layout,
            kv_layout_candidates,
            flash_kv_min,
            flash_kv_min_candidates,
            prefix_snapshots,
            prefix_snapshots_candidates,
            kv_page_size,
            kv_page_size_candidates,
            flash_v3_kv_tile,
            flash_v3_kv_tile_candidates,
            moe_placement,
            speculative_mtp,
            speculative_mtp_repeats,
            ssm_prefill_chunked,
            ssm_prefill_chunked_repeats,
            decision_calibrate,
            per_device_perf,
            validate_kernels,
            all,
            skip_cached,
            force,
        } => {
            // `tune --show` just prints the tuner-cache state (the former
            // `tuning show`) and runs no sweep.
            if show {
                return cmd_tuning_show(&config_path);
            }
            // The batch-size sweep measures prefill throughput, so it
            // uses a long synthetic prompt rather than the shared
            // `--prompt-tokens` (which is sized for decode measurements).
            // Quick by default (256 tokens keeps prefill dominant while
            // finishing fast; prefill tok/s ranking is size-independent);
            // `--thorough` restores the old exhaustive 2048-token prompt.
            const BATCH_SWEEP_PROMPT_TOKENS: u32 = 2048;
            const QUICK_BATCH_SWEEP_PROMPT_TOKENS: u32 = 256;
            let batch_prompt_tokens = if thorough {
                BATCH_SWEEP_PROMPT_TOKENS
            } else {
                QUICK_BATCH_SWEEP_PROMPT_TOKENS
            };
            // Quick-by-default measurement sizing for the decode sweeps.
            // `--thorough` restores the old exhaustive decode-tokens (32)
            // and repeats (7). `.max()` keeps any user override that goes
            // UPWARD (e.g. `--decode-tokens 64` / `--repeats 7`), so the
            // flags still let a user tune more precisely on demand. The
            // mandatory first-load autotune no longer passes `--thorough`,
            // so it runs quick. (The shared `--prompt-tokens` is already
            // small — it only sizes the excluded prefill + KV footprint of
            // the decode sweeps — so it needs no quick/thorough split.)
            let eff_decode_tokens = if thorough {
                decode_tokens.max(32)
            } else {
                decode_tokens
            };
            let eff_repeats = if thorough { repeats.max(7) } else { repeats };
            // Quick-by-default batch-size candidate grid. The quick batch sweep
            // uses a 256-token prompt (QUICK_BATCH_SWEEP_PROMPT_TOKENS), so any
            // chunk-size candidate >= that prompt length degenerates to a single
            // chunk and just re-measures the same thing — trim the grid to the
            // two candidates that actually differ at 256 tokens. `--thorough`
            // restores the full user-supplied grid (default 128..=2048) paired
            // with the 2048-token prefill. batch_size is the slowest single
            // stage, so cutting redundant candidates is the biggest quick win.
            let eff_batch_candidates: &str = if thorough {
                batch_candidates.as_str()
            } else {
                "128,256"
            };
            let exclusive = [
                placement,
                batch_size,
                threads,
                kv_dtype,
                flash_attention,
                kv_layout,
                flash_kv_min,
                prefix_snapshots,
                kv_page_size,
                flash_v3_kv_tile,
                moe_placement,
                speculative_mtp,
                ssm_prefill_chunked,
                decision_calibrate,
                per_device_perf,
                validate_kernels,
                all,
            ]
            .iter()
            .filter(|x| **x)
            .count();
            if exclusive > 1 {
                anyhow::bail!(
                    "tune: --placement / --batch-size / --threads / --kv-dtype / \
                     --flash-attention / --kv-layout / --flash-kv-min / \
                     --prefix-snapshots / --kv-page-size / --flash-v3-kv-tile / \
                     --moe-placement / --speculative-mtp / --ssm-prefill-chunked / \
                     --decision-calibrate / --per-device-perf / --all are exclusive; \
                     pick at most one"
                );
            }
            if all {
                return cmd_tune_all(
                    &config_path,
                    device,
                    model,
                    thorough,
                    clear,
                    vram_mb,
                    vram_headroom_mb,
                    placement_ctx,
                    prompt_tokens,
                    eff_decode_tokens,
                    eff_repeats,
                    eff_batch_candidates,
                    batch_prompt_tokens,
                    eff_repeats,
                    &threads_candidates,
                    eff_decode_tokens,
                    eff_repeats,
                    &kv_dtype_candidates,
                    prompt_tokens,
                    eff_decode_tokens,
                    eff_repeats,
                    prompt_tokens,
                    eff_decode_tokens,
                    eff_repeats,
                    &kv_layout_candidates,
                    prompt_tokens,
                    eff_decode_tokens,
                    eff_repeats,
                    skip_cached,
                    force,
                );
            }
            if kv_dtype {
                return cmd_tune_kv_dtype(
                    &config_path,
                    model,
                    &kv_dtype_candidates,
                    prompt_tokens,
                    decode_tokens,
                    repeats,
                );
            }
            if flash_attention {
                return cmd_tune_flash_attention(
                    &config_path,
                    model,
                    prompt_tokens,
                    decode_tokens,
                    repeats,
                );
            }
            if kv_layout {
                return cmd_tune_kv_layout(
                    &config_path,
                    model,
                    &kv_layout_candidates,
                    prompt_tokens,
                    decode_tokens,
                    repeats,
                );
            }
            if flash_kv_min {
                return cmd_tune_flash_kv_min(
                    &config_path,
                    model,
                    &flash_kv_min_candidates,
                    prompt_tokens,
                    decode_tokens,
                    repeats,
                );
            }
            if prefix_snapshots {
                return cmd_tune_prefix_snapshots(
                    &config_path,
                    model,
                    &prefix_snapshots_candidates,
                    prompt_tokens,
                    decode_tokens,
                    repeats,
                );
            }
            if kv_page_size {
                return cmd_tune_kv_page_size(
                    &config_path,
                    model,
                    &kv_page_size_candidates,
                    prompt_tokens,
                    decode_tokens,
                    repeats,
                );
            }
            if moe_placement {
                return cmd_tune_moe_placement(
                    &config_path,
                    model,
                    prompt_tokens,
                    decode_tokens,
                    repeats,
                );
            }
            if flash_v3_kv_tile {
                return cmd_tune_flash_v3_kv_tile(
                    &config_path,
                    model,
                    &flash_v3_kv_tile_candidates,
                    prompt_tokens,
                    decode_tokens,
                    repeats,
                );
            }
            if speculative_mtp {
                return cmd_tune_speculative_mtp(&config_path, model, speculative_mtp_repeats);
            }
            if ssm_prefill_chunked {
                return cmd_tune_ssm_prefill_chunked(
                    &config_path,
                    model,
                    ssm_prefill_chunked_repeats,
                );
            }
            if decision_calibrate {
                return cmd_decision_calibrate(&config_path, model);
            }
            if per_device_perf {
                return cmd_tune_per_device_perf(
                    &config_path,
                    model,
                    prompt_tokens,
                    decode_tokens,
                    repeats,
                );
            }
            if validate_kernels {
                return cmd_tune_validate_kernels();
            }
            if threads {
                return cmd_tune_threads(
                    &config_path,
                    model,
                    &threads_candidates,
                    decode_tokens,
                    repeats,
                );
            }
            if batch_size {
                cmd_tune_batch_size(
                    &config_path,
                    model,
                    eff_batch_candidates,
                    batch_prompt_tokens,
                    eff_repeats,
                )
            } else if placement {
                cmd_tune_placement(
                    &config_path,
                    model,
                    vram_mb,
                    vram_headroom_mb,
                    placement_ctx,
                    measure,
                    prompt_tokens,
                    decode_tokens,
                    repeats,
                )
            } else {
                cmd_tune(
                    &config_path,
                    device,
                    model,
                    thorough,
                    clear,
                    measure_tok_s,
                    decode_tokens,
                    repeats,
                )
            }
        }
        #[cfg(feature = "encoder")]
        Command::Quantize {
            input,
            output,
            to_mlx,
            mlx_bits,
            mlx_group_size,
            target,
            no_passthrough,
            keep_output,
            recipe,
            apex,
            imatrix,
        } => cmd_quantize(
            &input,
            &output,
            to_mlx,
            mlx_bits,
            mlx_group_size,
            target.as_deref(),
            no_passthrough,
            keep_output,
            recipe.as_deref(),
            apex.as_deref(),
            imatrix.as_deref(),
        ),
        #[cfg(not(feature = "encoder"))]
        Command::Quantize { .. } => anyhow::bail!(
            "this build was compiled without the `encoder` feature — \
             `rustllama quantize` is unavailable (use a default-features build)"
        ),
        #[cfg(feature = "encoder")]
        Command::KvCalibrate {
            model,
            prompts,
            output,
            ctx_size,
        } => cmd_kv_calibrate(&config_path, model, prompts, output, ctx_size),
        #[cfg(not(feature = "encoder"))]
        Command::KvCalibrate { .. } => anyhow::bail!(
            "this build was compiled without the `encoder` feature — \
             `rustllama kv-calibrate` is unavailable (use a default-features build)"
        ),
        Command::Imatrix {
            model,
            calibration,
            output,
            max_tokens,
            ctx_size,
        } => cmd_imatrix(&model, &calibration, &output, max_tokens, ctx_size),
        Command::Gui => {
            // The GUI is served by the Tauri host binary (built with
            // `--features gui`), which intercepts `gui` before delegating
            // here. Reaching this arm means we're the headless build.
            anyhow::bail!(
                "this is a headless build with no GUI; install/run the \
                 desktop (GUI) build of rustllama to use `gui`"
            );
        }
        Command::Lsp {
            base_url,
            model,
            max_tokens,
            temperature,
        } => run_lsp(base_url, model, max_tokens, temperature),
    }
}

#[cfg(feature = "encoder")]
#[allow(clippy::too_many_arguments)]
fn cmd_quantize(
    input: &str,
    output: &str,
    to_mlx: bool,
    mlx_bits: u32,
    mlx_group_size: usize,
    target_name: Option<&str>,
    no_passthrough: bool,
    keep_output: bool,
    recipe_path: Option<&str>,
    apex_tier: Option<&str>,
    imatrix_path: Option<&str>,
) -> anyhow::Result<()> {
    use rustllama_gguf::{
        apex::{build_apex_rules, ApexTier},
        imatrix::Imatrix,
        quantize::QuantizePlan,
        recipe::parse_recipe_file,
        GgmlType, Gguf, MetadataValue,
    };

    // `--to-mlx` writes an Apple MLX affine model directory instead of a
    // GGUF file — an entirely separate produce path (GGUF dequant → MLX
    // group-affine quant → safetensors/config writer). The GGUF→GGUF
    // levers (`--target`/recipe/apex/imatrix) don't apply.
    if to_mlx {
        if target_name.is_some()
            || recipe_path.is_some()
            || apex_tier.is_some()
            || imatrix_path.is_some()
        {
            eprintln!(
                "quantize: --to-mlx ignores --target/--recipe/--apex/--imatrix \
                 (MLX uses --mlx-bits / --mlx-group-size)"
            );
        }
        return cmd_quantize_to_mlx(input, output, mlx_bits, mlx_group_size);
    }

    // GGUF → GGUF path. `--target` is required here.
    let target_name = target_name.ok_or_else(|| {
        anyhow::anyhow!(
            "quantize: --target is required for the GGUF→GGUF path \
             (or pass --to-mlx to write an MLX directory)"
        )
    })?;
    let target = parse_target_dtype(target_name)?;
    let src = Gguf::open(input)
        .map_err(|e| anyhow::anyhow!("failed to open source GGUF {input:?}: {e}"))?;

    let mut plan = QuantizePlan::uniform(target);

    // Pull layer count from source metadata for APEX. Try the
    // common architecture-prefixed key (`{arch}.block_count`); if
    // the source doesn't have one, fall back to the per-layer tensor
    // count via name pattern scan. APEX rules without a real layer
    // count gracefully no-op the layer-position gradient.
    let n_layers = infer_n_layers_from_gguf(&src);

    if let Some(tier_name) = apex_tier {
        let tier = ApexTier::parse(tier_name).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown APEX tier {tier_name:?}. Valid: i-quality, quality, balanced, mini, nano"
            )
        })?;
        plan.add_rules(build_apex_rules(tier, n_layers));
        println!(
            "quantize: {} profile applied ({} blocks → {} APEX rules)",
            tier.label(),
            n_layers,
            plan.rules.len()
        );
    }

    if let Some(path) = recipe_path {
        let rules =
            parse_recipe_file(path).map_err(|e| anyhow::anyhow!("recipe parse failed: {e}"))?;
        let added = rules.len();
        plan.add_rules(rules);
        println!("quantize: loaded {added} rule(s) from recipe {path}");
    }

    // Optional importance matrix. When present, the encode runs on the
    // CPU path with per-column weighting (the GPU IQ encoder has no
    // weighted variant yet).
    let imatrix = match imatrix_path {
        Some(path) => {
            let im = Imatrix::load(path)
                .map_err(|e| anyhow::anyhow!("imatrix load failed ({path}): {e}"))?;
            println!(
                "quantize: loaded imatrix with {} tensor entr(ies) from {path} \
                 — Q2_K/Q4_K/Q5_K/IQ1_S tensors will be importance-weighted",
                im.len()
            );
            Some(im)
        }
        None => None,
    };

    if !no_passthrough {
        // The pipeline's `is_quantizable_tensor` already handles
        // "1D norm / bias" classification (anything whose dequant
        // would be a no-op stays raw). Nothing extra to wire here.
    }
    if keep_output {
        plan.passthrough_prefixes.push("output.".into());
        plan.passthrough_prefixes.push("lm_head.".into());
        plan.passthrough_prefixes.push("head.".into());
    }

    let start = std::time::Instant::now();
    // Use the mmap-backed pipeline so encoded bytes flow straight
    // into the output file via memory-mapped pages — eliminates the
    // per-tensor heap-allocated `enc_buf` that the Vec-backed
    // `quantize_gguf` would otherwise hold. Peak per-tensor RAM
    // becomes whatever the OS chooses to keep resident from the
    // mapped region; on memory-tight hosts that's a few MB rather
    // than the full encoded tensor.
    //
    // `RUSTLLAMA_IQ_GPU=1` opts into the experimental SYCL GPU path
    // for IQ1_S codebook search. The encoder uploads the IQ1_S grid
    // to USM once and dispatches one batched call per delta-sign
    // per tensor. Other tensor dtypes use the existing rayon CPU
    // path. If the GPU encoder fails to initialize (no Intel SYCL
    // device, oneAPI not on PATH), we fall back to the CPU pipeline
    // automatically with a warning. **Hardware-validation pending**
    // — opt-in only.
    let stats = if let Some(im) = imatrix.as_ref() {
        // imatrix path. With RUSTLLAMA_IQ_GPU=1 we use the GPU
        // importance-weighted encoder (IQ1_S on GPU; Q2_K/Q4_K/Q5_K
        // fall to the CPU weighted encoder inside the pipeline). Without
        // the GPU opt-in (or if the encoder fails to init) we use the
        // CPU weighted pipeline.
        let use_gpu = std::env::var("RUSTLLAMA_IQ_GPU")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let gpu_encoder = if use_gpu {
            match rustllama_kernels_sycl::iq_encoder::SyclIqEncoder::new(0) {
                Ok(encoder) => {
                    println!(
                        "quantize: imatrix + IQ GPU encoder enabled (IQ1_S weighted on GPU; \
                         K-quants weighted on CPU)"
                    );
                    Some(encoder)
                }
                Err(e) => {
                    eprintln!(
                        "quantize: RUSTLLAMA_IQ_GPU=1 set but encoder unavailable ({e}); \
                         using the CPU weighted encode"
                    );
                    None
                }
            }
        } else {
            None
        };
        if let Some(encoder) = gpu_encoder.as_ref() {
            rustllama_gguf::quantize::quantize_gguf_to_path_with_encoder_imatrix(
                &src,
                output,
                &plan,
                encoder,
                Some(im),
            )
            .map_err(|e| anyhow::anyhow!("quantize pipeline failed: {e}"))?
        } else {
            rustllama_gguf::quantize::quantize_gguf_to_path_with_imatrix(
                &src,
                output,
                &plan,
                Some(im),
            )
            .map_err(|e| anyhow::anyhow!("quantize pipeline failed: {e}"))?
        }
    } else {
        let use_gpu = std::env::var("RUSTLLAMA_IQ_GPU")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if use_gpu {
            match rustllama_kernels_sycl::iq_encoder::SyclIqEncoder::new(0) {
                Ok(encoder) => {
                    println!("quantize: IQ GPU encoder enabled (experimental — needs hardware validation)");
                    rustllama_gguf::quantize::quantize_gguf_to_path_with_encoder(
                        &src, output, &plan, &encoder,
                    )
                    .map_err(|e| anyhow::anyhow!("quantize pipeline failed: {e}"))?
                }
                Err(e) => {
                    eprintln!(
                        "quantize: RUSTLLAMA_IQ_GPU=1 set but encoder unavailable ({e}); falling back to CPU"
                    );
                    rustllama_gguf::quantize::quantize_gguf_to_path(&src, output, &plan)
                        .map_err(|e| anyhow::anyhow!("quantize pipeline failed: {e}"))?
                }
            }
        } else {
            rustllama_gguf::quantize::quantize_gguf_to_path(&src, output, &plan)
                .map_err(|e| anyhow::anyhow!("quantize pipeline failed: {e}"))?
        }
    };
    let elapsed = start.elapsed();

    let mb_in = stats.bytes_in as f64 / (1024.0 * 1024.0);
    let mb_out = stats.bytes_out as f64 / (1024.0 * 1024.0);
    let ratio = if stats.bytes_in > 0 {
        stats.bytes_out as f64 / stats.bytes_in as f64
    } else {
        1.0
    };
    println!(
        "quantize: {} tensor(s) re-encoded, {} passthrough ({} total)",
        stats.tensors_requantized, stats.tensors_passthrough, stats.tensors_total,
    );
    println!(
        "          {:.1} MiB → {:.1} MiB ({:.1}% of original) in {:.2}s",
        mb_in,
        mb_out,
        ratio * 100.0,
        elapsed.as_secs_f64()
    );
    println!("          default target dtype: {}", target.as_str());
    let _ = GgmlType::F32;
    let _ = MetadataValue::U32(0);

    Ok(())
}

/// `rustllama quantize --to-mlx`: convert a GGUF model into an Apple
/// **MLX affine** model directory (mlx-lm / mlx-community layout). Each
/// 2-D weight whose input dim is a multiple of `group_size` is
/// group-affine quantized into the MLX `<name>.weight`/`.scales`/
/// `.biases` triple; 1-D tensors (and any 2-D weight whose input dim
/// doesn't divide `group_size`) pass through full precision. GGUF tensor
/// names are preserved, so the Phase-A MLX loader
/// (`rustllama_safetensors::load_mlx_dir`) reads them straight back.
///
/// # Caveats
///
/// - **Name fidelity.** Names are kept verbatim (`blk.0.attn_q.*`), NOT
///   rewritten to mlx-lm's HF module paths (`model.layers.0...`). The
///   file round-trips through rustllama's own loader (which keys on the
///   `.scales`/`.biases` siblings, not fixed names), but is not a
///   drop-in for upstream mlx-lm tooling until a name remap lands.
/// - **Dims > 2** (e.g. stacked MoE expert tensors) pass through as f32
///   rather than being quantized — correct but large; MoE→MLX is a
///   later slice.
/// - **Single shard.** Always one `model.safetensors`; no sharding.
/// - **Affine only.** Emits `mode="affine"`; the MXFP/NVFP4 MLX modes
///   are out of scope (the loader rejects them on read too).
#[cfg(feature = "encoder")]
fn cmd_quantize_to_mlx(
    input: &str,
    output: &str,
    bits: u32,
    group_size: usize,
) -> anyhow::Result<()> {
    use rustllama_gguf::{quantize::dequant_tensor_to_f32, Gguf};
    use rustllama_kernels_cpu::mlx_affine::quantize_mlx_affine;
    use rustllama_safetensors::{map_gguf_to_hf, write_mlx_dir, MlxFullDtype, MlxWriteTensor};

    // Geometry guards mirror the MLX affine format + the loader's
    // validate(): bits ∈ {2,3,4,5,6,8}, group_size ∈ {32,64,128}.
    if !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) {
        anyhow::bail!("--mlx-bits {bits} invalid (want one of 2,3,4,5,6,8)");
    }
    if !matches!(group_size, 32 | 64 | 128) {
        anyhow::bail!("--mlx-group-size {group_size} invalid (want one of 32,64,128)");
    }

    let src = Gguf::open(input)
        .map_err(|e| anyhow::anyhow!("failed to open source GGUF {input:?}: {e}"))?;
    let out_dir = std::path::Path::new(output);

    let start = std::time::Instant::now();
    let mut tensors: Vec<MlxWriteTensor> = Vec::with_capacity(src.tensors().len());
    let mut n_quant = 0usize;
    let mut n_full = 0usize;
    let mut bytes_in = 0u64;

    for t in src.tensors() {
        let n = t.element_count() as usize;
        let src_bytes = src
            .tensor_bytes(&t.name)
            .ok_or_else(|| anyhow::anyhow!("source tensor {:?} has no data region", t.name))?;
        bytes_in += src_bytes.len() as u64;

        // Dequant the whole tensor to f32. GGUF data is row-major with
        // the inner (input) dim contiguous — exactly the axis MLX groups
        // along, so no transpose is needed.
        let mut f32_buf = vec![0f32; n];
        dequant_tensor_to_f32(t.dtype, src_bytes, &mut f32_buf);

        // Translate the GGUF tensor name to its mlx-lm / HF counterpart so
        // the written checkpoint is a drop-in for upstream mlx-lm tooling
        // (`model.layers.0.self_attn.q_proj.weight`, not `blk.0.attn_q.weight`).
        let hf_name = map_gguf_to_hf(&t.name);

        // GGUF dims are [n_in (contiguous), n_out, ...]; a 2-D weight is
        // [n_in, n_out]. MLX's logical shape is [out_features,
        // in_features] = [n_out, n_in], with groups along n_in.
        let is_2d = t.dims.len() == 2;
        let in_features = t.dims.first().copied().unwrap_or(0);
        if is_2d && in_features > 0 && in_features % group_size as u64 == 0 {
            let out_features = t.dims[1];
            // A quantized linear with no clean HF inverse would produce a
            // checkpoint upstream can't load — bail rather than emit a
            // wrong name. (The standard Llama/Qwen2 set always maps.)
            let hf_full = hf_name.ok_or_else(|| {
                anyhow::anyhow!(
                    "cannot map GGUF tensor {:?} to an mlx-lm / HF module name — \
                     quantize --to-mlx supports the standard dense Llama/Qwen2 tensor \
                     set; a fused or architecture-specific weight has no clean HF inverse",
                    t.name
                )
            })?;
            let (packed, scales, biases) = quantize_mlx_affine(&f32_buf, group_size, bits);
            // Module path = HF name minus the `.weight` suffix; the writer
            // re-appends `.weight`/`.scales`/`.biases`.
            let module = hf_full
                .strip_suffix(".weight")
                .unwrap_or(&hf_full)
                .to_string();
            tensors.push(MlxWriteTensor::Quant {
                name: module,
                packed,
                scales,
                biases,
                group_size,
                bits,
                shape: vec![out_features, in_features],
            });
            n_quant += 1;
        } else {
            // 1-D norms/biases, a group-misaligned 2-D weight, or a
            // >2-D tensor: carry through full precision (f32, lossless).
            // Map the name when we can; an unmapped passthrough (aux tensor
            // outside the standard set) keeps its GGUF name with a warning
            // rather than aborting the whole conversion.
            if is_2d {
                eprintln!(
                    "quantize: {} [{}x{}] input dim not a multiple of group_size {} \
                     — passing through full precision",
                    t.name, in_features, t.dims[1], group_size
                );
            }
            let name = match hf_name {
                Some(h) => h,
                None => {
                    eprintln!(
                        "quantize: GGUF tensor {:?} has no mlx-lm / HF name mapping — \
                         writing it under its original name (mlx-lm may ignore it)",
                        t.name
                    );
                    t.name.clone()
                }
            };
            let bytes: Vec<u8> = f32_buf.iter().flat_map(|v| v.to_le_bytes()).collect();
            tensors.push(MlxWriteTensor::Full {
                name,
                dtype: MlxFullDtype::F32,
                shape: t.dims.clone(),
                bytes,
            });
            n_full += 1;
        }
    }

    // Minimal, honest config.json provenance beyond the `quantization`
    // block the writer adds (the MLX loader needs only `quantization`).
    let mut extra = serde_json::Map::new();
    if let Some(arch) = src.architecture() {
        extra.insert("architectures".into(), serde_json::json!([arch]));
        extra.insert("model_type".into(), serde_json::json!(arch));
    }
    extra.insert("quantized_by".into(), serde_json::json!("rustllama"));

    // Copy a sibling `tokenizer.json` if the source GGUF has one beside
    // it (GGUF embeds its tokenizer in metadata, but a directory export
    // may also ship the HF tokenizer.json). The writer copies only when
    // the path exists.
    let tok_src = std::path::Path::new(input)
        .parent()
        .map(|p| p.join("tokenizer.json"));
    let tok_src_ref = tok_src.as_deref().filter(|p| p.exists());

    write_mlx_dir(out_dir, &tensors, group_size, bits, extra, tok_src_ref)
        .map_err(|e| anyhow::anyhow!("MLX write failed: {e}"))?;

    let elapsed = start.elapsed();
    println!(
        "quantize --to-mlx: {} tensor(s) — {} quantized (affine {}-bit, g={}), {} passthrough",
        tensors.len(),
        n_quant,
        bits,
        group_size,
        n_full,
    );
    println!(
        "          source {:.1} MiB -> MLX directory {} in {:.2}s",
        bytes_in as f64 / (1024.0 * 1024.0),
        out_dir.display(),
        elapsed.as_secs_f64(),
    );
    if tok_src_ref.is_some() {
        println!("          copied tokenizer.json");
    }
    Ok(())
}

/// `rustllama tune --moe-placement`: A/B the MoE placement modes and
/// q★ co-execution splits by measured decode throughput, persist the
/// winner to the tuner cache. See
/// `measurement::measure_moe_placement_candidates` for the candidate
/// set and the ≥5%-win gate.
fn cmd_tune_moe_placement(
    config_path: &std::path::Path,
    model_override: Option<String>,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
) -> anyhow::Result<()> {
    use rustllama_engine::measurement::{measure_moe_placement_candidates, MeasurementConfig};

    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let model_path = effective_model_path(model_override, &cfg, config_path)?;
    if repeats == 0 {
        anyhow::bail!("--moe-placement-repeats must be >= 1");
    }
    let (k, _) = cfg.inference.resolved_kv_dtypes();
    let m_cfg = MeasurementConfig {
        kv_dtype: parse_kv_dtype(k)?,
        kv_cache_layout: cfg.inference.kv_cache_layout.clone(),
        flash_attention: cfg.inference.flash_attention,
        n_gpu_layers: cfg.inference.n_gpu_layers,
    };
    let ctx_size =
        (prompt_tokens as usize + decode_tokens as usize + 8).max(cfg.inference.ctx_size as usize);

    println!("rustllama tune --moe-placement");
    println!("  model         = {}", model_path.display());
    println!("  ctx_size      = {ctx_size}");
    println!("  prompt_tokens = {prompt_tokens}");
    println!("  decode_tokens = {decode_tokens}");
    println!("  repeats       = {repeats} (median per candidate)");
    println!();

    let report = measure_moe_placement_candidates(
        &model_path,
        ctx_size,
        prompt_tokens,
        decode_tokens,
        repeats,
        &m_cfg,
    )?;
    println!("  model loaded in {:.1} ms", report.load_ms);
    println!();
    println!(
        "  {:>12}  {:>7}  {:>9}  {:>11}  {:>11}  notes",
        "candidate", "split\u{2030}", "warmup", "median tps", "p100 tps"
    );
    for c in &report.candidates {
        match c.median_tps {
            Some(m) => println!(
                "  {:>12}  {:>7}  {:>7.0}ms  {:>11.2}  {:>11.2}",
                c.label,
                c.gpu_split_permille,
                c.warmup_ms,
                m,
                c.max_tps.unwrap_or(0.0),
            ),
            None => println!(
                "  {:>12}  {:>7}  {:>7.0}ms  {:>11}  {:>11}  {}",
                c.label,
                c.gpu_split_permille,
                c.warmup_ms,
                "-",
                "-",
                c.error.as_deref().unwrap_or("failed"),
            ),
        }
    }
    println!();
    match &report.winner {
        Some((placement, split)) => {
            println!(
                "  winner: {placement} (split {split}\u{2030}) at {:.2} tok/s \
                 (uniform baseline {:.2} tok/s)",
                report.winner_tps, report.baseline_tps
            );
            persist_moe_placement_winner(placement, *split)?;
            println!(
                "  applied automatically on next load when \
                 [tuning].auto_apply_moe_placement = true"
            );
        }
        None => println!("  no candidate produced a successful run; nothing persisted"),
    }
    Ok(())
}

/// Promote the typed `[inference]` memory knobs to the process env
/// vars the models/engine crates read at load time. A user-set env
/// var always wins (lets the tuner or a power user sweep a knob
/// without rewriting config). Called by `serve` and `bench` before
/// the engine loads — the accessors behind these vars cache on first
/// read, so promotion must precede `CpuEngine::load*`.
/// Promote the compute-tier knobs (`disabled_cpus`, `cpu_enabled`,
/// `gpu_enabled`, `vram_only`) into the env the runtime/engine/server read.
/// The CPU
/// analog of the `disabled_gpus` promotion. Only sets a var when the
/// setting is non-default and the user hasn't already set the env var
/// (a user-set env var wins). Idempotent — safe to call more than once.
fn promote_cpu_tier_env_from_config(inference: &rustllama_config::InferenceConfig) {
    // CPU disable-list: read by the thread-pool affinity pinning + the
    // tuner's CPU-config-aware fingerprint (mirror of the GPU disable-list).
    if !inference.disabled_cpus.is_empty() && std::env::var_os("RUSTLLAMA_DISABLED_CPUS").is_none() {
        let list = inference
            .disabled_cpus
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",");
        std::env::set_var("RUSTLLAMA_DISABLED_CPUS", &list);
        tracing::info!(disabled_cpus = %list, "CPU disable-list applied from config");
    }
    // CPU tier on/off + VRAM-only mode: promote so the server inventory
    // (`/v1/capabilities`, `/v1/tuning_summary`) and any child process
    // reflect them.
    if !inference.cpu_enabled && std::env::var_os("RUSTLLAMA_CPU_ENABLED").is_none() {
        std::env::set_var("RUSTLLAMA_CPU_ENABLED", "0");
        tracing::info!("cpu_enabled=false: GPU-only placement (no CPU weight tier)");
    }
    // GPU tier on/off (mirror of cpu_enabled). Promote so the server
    // inventory + any child process see a CPU-only placement.
    if !inference.gpu_enabled && std::env::var_os("RUSTLLAMA_GPU_ENABLED").is_none() {
        std::env::set_var("RUSTLLAMA_GPU_ENABLED", "0");
        tracing::info!("gpu_enabled=false: CPU-only placement (no GPU weight tier)");
    }
    if inference.vram_only && std::env::var_os("RUSTLLAMA_VRAM_ONLY").is_none() {
        std::env::set_var("RUSTLLAMA_VRAM_ONLY", "1");
        tracing::info!("vram_only=true: weights required in dedicated VRAM only");
    }
}

fn promote_memory_env_from_config(
    inference: &rustllama_config::InferenceConfig,
    tuning: &rustllama_config::TuningConfig,
) {
    if inference.moe_expert_cache_mb > 0
        && std::env::var_os("RUSTLLAMA_MOE_EXPERT_CACHE_MB").is_none()
    {
        std::env::set_var(
            "RUSTLLAMA_MOE_EXPERT_CACHE_MB",
            inference.moe_expert_cache_mb.to_string(),
        );
        tracing::info!(
            budget_mb = inference.moe_expert_cache_mb,
            "moe_expert_cache_mb: expert-pin cache enabled from config"
        );
    }
    if inference.zerocopy_weights && std::env::var_os("RUSTLLAMA_ZEROCOPY_WEIGHTS").is_none() {
        std::env::set_var("RUSTLLAMA_ZEROCOPY_WEIGHTS", "1");
    }
    let lock_ram = inference.lock_ram_mb.trim();
    if !lock_ram.is_empty()
        && lock_ram != "0"
        && std::env::var_os("RUSTLLAMA_LOCK_RAM_MB").is_none()
    {
        std::env::set_var("RUSTLLAMA_LOCK_RAM_MB", lock_ram);
    }
    // GPU disable-list: promote `[inference].disabled_gpus` into the env
    // both the metrics probe and the SYCL dispatch selection read.
    if !inference.disabled_gpus.is_empty() && std::env::var_os("RUSTLLAMA_DISABLED_GPUS").is_none()
    {
        let list = inference
            .disabled_gpus
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",");
        std::env::set_var("RUSTLLAMA_DISABLED_GPUS", &list);
        tracing::info!(disabled_gpus = %list, "GPU disable-list applied from config");
    }
    // CPU compute tier (disable-list + on/off + VRAM-only mode).
    promote_cpu_tier_env_from_config(inference);
    // MoE routed-expert placement is fully AUTO: the tuner-cache winner
    // (populated by `rustllama tune --moe-placement`, gated by
    // `[tuning].auto_apply_moe_placement`) decides it — there is no
    // user-facing config key; the heat planner / tuner owns the
    // routed-expert tier. No cache entry (or auto-apply off) ⇒ uniform
    // (a no-op). Promoted to the env vars accel reads at first use.
    let (experts_cpu, gpu_split) =
        match moe_placement_from_cache_or_default(tuning.auto_apply_moe_placement) {
            Some((mode, split)) => {
                tracing::info!(
                    placement = %mode,
                    gpu_split_permille = split,
                    "moe placement (auto): applied tuner-cache winner"
                );
                (mode == "experts_cpu", split)
            }
            None => (false, 0),
        };
    if experts_cpu && std::env::var_os("RUSTLLAMA_MOE_EXPERTS_CPU").is_none() {
        std::env::set_var("RUSTLLAMA_MOE_EXPERTS_CPU", "1");
        tracing::info!(
            "moe_placement: routed experts pinned to CPU (attn/shared stay GPU-eligible)"
        );
    }
    if gpu_split > 0 && std::env::var_os("RUSTLLAMA_MOE_GPU_SPLIT_PERMILLE").is_none() {
        std::env::set_var("RUSTLLAMA_MOE_GPU_SPLIT_PERMILLE", gpu_split.to_string());
        tracing::info!(
            gpu_split_permille = gpu_split,
            "moe_placement: q★ CPU/GPU expert co-execution enabled"
        );
    }
    if inference.prefill_batched && std::env::var_os("RUSTLLAMA_PREFILL_BATCHED").is_none() {
        std::env::set_var("RUSTLLAMA_PREFILL_BATCHED", "1");
        tracing::info!(
            "prefill_batched: chunked prefill routes through the batched flash-prefill path"
        );
    }
    if inference.memory_budget.trim().eq_ignore_ascii_case("auto")
        && std::env::var_os("RUSTLLAMA_MEMORY_BUDGET").is_none()
    {
        std::env::set_var("RUSTLLAMA_MEMORY_BUDGET", "auto");
        tracing::info!(
            "memory_budget=auto: expert-cache budget will be planned from measured RAM at load"
        );
    }
    if inference.prefill_batched_hybrid
        && std::env::var_os("RUSTLLAMA_PREFILL_BATCHED_HYBRID").is_none()
    {
        std::env::set_var("RUSTLLAMA_PREFILL_BATCHED_HYBRID", "1");
        tracing::info!(
            "prefill_batched_hybrid: hybrid chunk prefill with grouped-expert execution enabled"
        );
    }
    if inference.ssm_prefill_chunked
        && std::env::var_os("RUSTLLAMA_SSM_PREFILL_CHUNKED").is_none()
    {
        std::env::set_var("RUSTLLAMA_SSM_PREFILL_CHUNKED", "1");
        tracing::info!(
            "ssm_prefill_chunked: chunked-parallel SSM (DeltaNet) prefill path enabled from config"
        );
    }
    if inference.kv_persist_mb > 0 && std::env::var_os("RUSTLLAMA_KV_PERSIST_MB").is_none() {
        std::env::set_var(
            "RUSTLLAMA_KV_PERSIST_MB",
            inference.kv_persist_mb.to_string(),
        );
        tracing::info!(
            cap_mb = inference.kv_persist_mb,
            "kv_persist_mb: warm-restart KV persistence enabled"
        );
    }
}

/// Generate an importance matrix by running calibration text through
/// the model's forward pass with the activation collector active.
fn cmd_imatrix(
    model_path: &str,
    calibration_path: &str,
    output_path: &str,
    max_tokens: u32,
    ctx_size: usize,
) -> anyhow::Result<()> {
    use rustllama_engine::{imatrix_collect, CpuEngine, SamplingParams};
    use rustllama_gguf::imatrix::Imatrix;

    // Disable the forward-pass fusions that bypass the base
    // `matvec_tensor_dispatch` hook, so every quantizable tensor is
    // recorded exactly once. Set before load — the `*_enabled()`
    // accessors cache their value on first read.
    for k in [
        "RUSTLLAMA_MOE_GATE_UP_FUSED",
        "RUSTLLAMA_FUSED_OUT_PROJ_NORM",
        "RUSTLLAMA_MIXED_PRECISION_MATVEC",
        "RUSTLLAMA_QKV_FUSED",
    ] {
        std::env::set_var(k, "0");
    }

    let ctx = ctx_size.max(1);
    let kv_dtype = parse_kv_dtype("f32")?;
    println!("imatrix: loading {model_path} (ctx {ctx})");
    let mut cpu = CpuEngine::load_with_options_and_layout(
        std::path::Path::new(model_path),
        ctx,
        true, // with tokenizer (needed to tokenize the calibration text)
        kv_dtype,
        "contiguous",
    )
    .map_err(|e| anyhow::anyhow!("load failed: {e}"))?;
    cpu.set_prefix_cache(false);

    let text = std::fs::read_to_string(calibration_path)
        .map_err(|e| anyhow::anyhow!("read calibration {calibration_path}: {e}"))?;
    let mut ids = cpu
        .tokenize_chunks([text], true)
        .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;
    if max_tokens > 0 && ids.len() > max_tokens as usize {
        ids.truncate(max_tokens as usize);
    }
    if ids.is_empty() {
        anyhow::bail!("imatrix: calibration produced 0 tokens");
    }
    println!(
        "imatrix: {} calibration tokens, prefill window {ctx}",
        ids.len()
    );

    // Greedy, single-token decode — we discard outputs; only the
    // prefill forward (which fires every matvec) matters.
    let sampling = SamplingParams {
        temperature: 0.0,
        max_tokens: 1,
        ..SamplingParams::default()
    };

    imatrix_collect::begin();
    let t = std::time::Instant::now();
    let n_chunks = ids.len().div_ceil(ctx);
    for (i, chunk) in ids.chunks(ctx).enumerate() {
        cpu.generate_token_ids(chunk, 1, &sampling)
            .map_err(|e| anyhow::anyhow!("calibration forward failed: {e}"))?;
        println!(
            "  calibrated chunk {}/{n_chunks} ({} tokens)",
            i + 1,
            chunk.len()
        );
    }
    let importance = imatrix_collect::finish();
    let secs = t.elapsed().as_secs_f64();

    if importance.is_empty() {
        anyhow::bail!(
            "imatrix: collector recorded 0 tensors — the forward path did not route \
             through matvec_tensor_dispatch (check model arch support)"
        );
    }
    let mut im = Imatrix::new();
    for (name, v) in importance {
        im.insert(name, v);
    }
    im.save(output_path)
        .map_err(|e| anyhow::anyhow!("write imatrix {output_path}: {e}"))?;
    println!(
        "imatrix: wrote {} tensor entr(ies) to {output_path} in {secs:.1}s",
        im.len()
    );
    println!("          feed to: quantize --imatrix {output_path} --recipe <recipe> --target f32");
    Ok(())
}

/// Built-in mixed-domain calibration prompts for `kv-calibrate` when
/// the user provides no `--prompts` file. Small on purpose: K-channel
/// means stabilize quickly (they are first moments, not tail
/// statistics), and each prompt runs a full prefill on what may be a
/// 27B-class model.
#[cfg(feature = "encoder")]
const KV_CALIB_BUILTIN_PROMPTS: &[&str] = &[
    "The three primary colors of light are red, green, and blue, while subtractive \
     pigments use cyan, magenta, and yellow. Color perception arises from the \
     differential stimulation of cone cells in the retina.",
    "fn quicksort<T: Ord>(v: &mut [T]) { if v.len() <= 1 { return; } let p = \
     partition(v); let (lo, hi) = v.split_at_mut(p); quicksort(lo); \
     quicksort(&mut hi[1..]); }",
    "In 1905, Albert Einstein published four papers that reshaped physics: the \
     photoelectric effect, Brownian motion, special relativity, and mass-energy \
     equivalence. Each addressed a standing puzzle of classical theory.",
    "Dear team, following up on yesterday's incident review: the root cause was a \
     stale cache entry that survived the deploy. Action items are listed below, \
     with owners and due dates. Please confirm by Friday.",
    "SELECT customer_id, SUM(total) AS revenue FROM orders WHERE placed_at >= \
     '2026-01-01' GROUP BY customer_id HAVING SUM(total) > 1000 ORDER BY revenue \
     DESC LIMIT 50;",
    "The recipe serves four: dice two onions and soften them in olive oil over \
     medium heat, add crushed garlic and cumin, then the lentils and stock. \
     Simmer twenty-five minutes and finish with lemon.",
    "El aprendizaje automático permite a los sistemas mejorar su rendimiento con \
     la experiencia. Los modelos de lenguaje se entrenan con grandes corpus de \
     texto y luego se ajustan para tareas específicas.",
    "Quarterly results exceeded guidance, with revenue up 14% year over year and \
     operating margin expanding 220 basis points. Management raised the full-year \
     outlook, citing strong demand and easing input costs.",
];

/// `rustllama kv-calibrate` — see the `Command::KvCalibrate` docs.
/// Runs prompts through a q4_0-KV engine with the calibration
/// accumulator armed, then writes the fork-format `kv_bar` sidecar.
#[cfg(feature = "encoder")]
/// Labeled decision items for decision-calibration. Each is a framed
/// context, a candidate option set, and the ground-truth index. Mixed
/// difficulty (easy → less-obvious) so the temperature fit sees a spread of
/// confidences. Options carry a leading space to match byte-level-BPE
/// continuations. Override with `--items <file.jsonl>` for a custom corpus.
struct DecisionCalibItem {
    context: &'static str,
    options: &'static [&'static str],
    correct: usize,
}

const YESNO: &[&str] = &[" yes", " no"];

const DECISION_CALIB_BUILTIN_ITEMS: &[DecisionCalibItem] = &[
    DecisionCalibItem { context: "Question: Is water wet?\nAnswer:", options: YESNO, correct: 0 },
    DecisionCalibItem { context: "Question: Is the sun cold?\nAnswer:", options: YESNO, correct: 1 },
    DecisionCalibItem { context: "Question: Is Paris the capital of France?\nAnswer:", options: YESNO, correct: 0 },
    DecisionCalibItem { context: "Question: Is 7 an even number?\nAnswer:", options: YESNO, correct: 1 },
    DecisionCalibItem { context: "Question: Do fish breathe air with lungs?\nAnswer:", options: YESNO, correct: 1 },
    DecisionCalibItem { context: "Question: Is ice frozen water?\nAnswer:", options: YESNO, correct: 0 },
    DecisionCalibItem { context: "Question: Is the Earth flat?\nAnswer:", options: YESNO, correct: 1 },
    DecisionCalibItem { context: "Question: Can humans breathe underwater without equipment?\nAnswer:", options: YESNO, correct: 1 },
    DecisionCalibItem { context: "Question: Is 100 greater than 10?\nAnswer:", options: YESNO, correct: 0 },
    DecisionCalibItem { context: "Question: Is a tomato a type of animal?\nAnswer:", options: YESNO, correct: 1 },
    DecisionCalibItem { context: "Question: Does the moon orbit the Earth?\nAnswer:", options: YESNO, correct: 0 },
    DecisionCalibItem { context: "Question: Is fire cold to the touch?\nAnswer:", options: YESNO, correct: 1 },
    DecisionCalibItem { context: "Question: Are there 12 months in a year?\nAnswer:", options: YESNO, correct: 0 },
    DecisionCalibItem { context: "Question: Is the ocean made of fresh water?\nAnswer:", options: YESNO, correct: 1 },
    DecisionCalibItem { context: "Question: Do plants produce oxygen?\nAnswer:", options: YESNO, correct: 0 },
    DecisionCalibItem { context: "Question: Is a decade 100 years?\nAnswer:", options: YESNO, correct: 1 },
    // Sentiment classification (Choice).
    DecisionCalibItem { context: "Classify the sentiment.\nReview: \"I absolutely love this, best purchase ever!\"\nSentiment:", options: &[" positive", " negative", " neutral"], correct: 0 },
    DecisionCalibItem { context: "Classify the sentiment.\nReview: \"Terrible quality, broke on day one.\"\nSentiment:", options: &[" positive", " negative", " neutral"], correct: 1 },
    DecisionCalibItem { context: "Classify the sentiment.\nReview: \"It arrived on time and works as described.\"\nSentiment:", options: &[" positive", " negative", " neutral"], correct: 2 },
    DecisionCalibItem { context: "Classify the sentiment.\nReview: \"Waste of money, would not recommend.\"\nSentiment:", options: &[" positive", " negative", " neutral"], correct: 1 },
    // Topic / intent classification (Choice).
    DecisionCalibItem { context: "Which topic?\nText: \"The quarterly earnings beat analyst expectations.\"\nTopic:", options: &[" finance", " sports", " cooking"], correct: 0 },
    DecisionCalibItem { context: "Which topic?\nText: \"He scored a hat-trick in the final minutes.\"\nTopic:", options: &[" finance", " sports", " cooking"], correct: 1 },
    DecisionCalibItem { context: "Which topic?\nText: \"Simmer the sauce for twenty minutes, stirring often.\"\nTopic:", options: &[" finance", " sports", " cooking"], correct: 2 },
];

/// Fit the temperature `T` that minimizes mean NLL of the correct option
/// under `softmax(z / T)`, via golden-section search over `[0.25, 5]`.
/// Returns `(T, nll_before@1.0, nll_after@T)`.
fn fit_decision_temperature(items: &[(Vec<f32>, usize)]) -> (f32, f32, f32) {
    let nll = |t: f32| -> f32 {
        let n = items.len().max(1) as f64;
        let mut sum = 0.0f64;
        for (z, correct) in items {
            let scaled: Vec<f32> = z.iter().map(|v| v / t).collect();
            let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let lse = max + scaled.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
            sum += (lse - scaled[*correct]) as f64;
        }
        (sum / n) as f32
    };
    let nll_before = nll(1.0);
    let (mut lo, mut hi) = (0.25f32, 5.0f32);
    let gr = 0.618_034f32;
    let mut c = hi - gr * (hi - lo);
    let mut d = lo + gr * (hi - lo);
    let (mut fc, mut fd) = (nll(c), nll(d));
    for _ in 0..48 {
        if fc < fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - gr * (hi - lo);
            fc = nll(c);
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + gr * (hi - lo);
            fd = nll(d);
        }
    }
    let t = 0.5 * (lo + hi);
    (t, nll_before, nll(t))
}

/// Fit split-conformal nonconformity quantiles at each requested `coverage`
/// from the temperature-calibrated decision logits. For each item the
/// nonconformity score is `1 − softmax(z / T)[correct]`. The returned map
/// is keyed by `"{:.2}"`-formatted coverage; the value `q` is the empirical
/// quantile at rank `⌈(n+1)·coverage⌉` (clamped into `[1, n]`). A downstream
/// prediction set = every option with calibrated prob ≥ `1 − q` then has the
/// standard marginal coverage guarantee. Returns an empty map when there are
/// no items.
fn fit_conformal_quantiles(
    items: &[(Vec<f32>, usize)],
    temperature: f32,
    coverages: &[f32],
) -> std::collections::HashMap<String, f32> {
    let mut out = std::collections::HashMap::new();
    let n = items.len();
    if n == 0 {
        return out;
    }
    let t = if temperature > 0.0 { temperature } else { 1.0 };
    // Nonconformity per item: 1 − calibrated P(true label).
    let mut scores: Vec<f32> = items
        .iter()
        .map(|(z, correct)| {
            let scaled: Vec<f32> = z.iter().map(|v| v / t).collect();
            let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scaled.iter().map(|v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let p_true = if sum > 0.0 { exps[*correct] / sum } else { 0.0 };
            1.0 - p_true
        })
        .collect();
    scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    for &cov in coverages {
        let rank = (((n + 1) as f32) * cov).ceil() as usize;
        let idx = rank.clamp(1, n) - 1;
        out.insert(format!("{cov:.2}"), scores[idx]);
    }
    out
}

/// Decision-calibration (a stage of `tune --all`): load the model, score
/// each labeled decision item via `score_continuations`, fit the decision-
/// probability temperature, and persist it per-model in the tuner cache.
fn cmd_decision_calibrate(config_path: &std::path::Path, model: Option<String>) -> anyhow::Result<()> {
    use rustllama_engine::CpuEngine;

    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let model_path: std::path::PathBuf = effective_model_path(model, &cfg, config_path)?;
    let model_key = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown-model")
        .to_string();

    std::env::set_var("RUSTLLAMA_KEEP_QUANT_RAW", "1");
    std::env::set_var("RUSTLLAMA_ZEROCOPY_WEIGHTS", "1");
    if std::env::var("RUSTLLAMA_PREFILL_BATCHED_HYBRID").is_err() {
        std::env::set_var("RUSTLLAMA_PREFILL_BATCHED_HYBRID", "1");
    }

    println!(
        "decision-calibrate: loading {} (f32 KV, {} items)",
        model_path.display(),
        DECISION_CALIB_BUILTIN_ITEMS.len()
    );
    let mut cpu = CpuEngine::load_with_options_and_layout(
        &model_path,
        2048,
        true,
        rustllama_engine::KvDtype::F32,
        "contiguous",
    )
    .map_err(|e| anyhow::anyhow!("load failed: {e}"))?;
    cpu.set_prefix_cache(false);

    let tokenizer = cpu
        .tokenizer()
        .ok_or_else(|| anyhow::anyhow!("model has no tokenizer"))?;

    let mut items: Vec<(Vec<f32>, usize)> = Vec::new();
    for it in DECISION_CALIB_BUILTIN_ITEMS {
        let ctx_ids: Vec<i32> = tokenizer
            .encode(it.context, tokenizer.add_bos_token())
            .map_err(|e| anyhow::anyhow!("encode context: {e}"))?
            .into_iter()
            .map(|t| t as i32)
            .collect();
        let opt_toks: Vec<Vec<u32>> = it
            .options
            .iter()
            .map(|o| tokenizer.encode(o, false).unwrap_or_default())
            .collect();
        if opt_toks.iter().any(|o| o.is_empty()) {
            continue;
        }
        // Length-normalized (mean) per-option logprob = the decision logit.
        let raw = cpu
            .score_continuations(&ctx_ids, &opt_toks)
            .map_err(|e| anyhow::anyhow!("score: {e}"))?;
        let z: Vec<f32> = raw
            .iter()
            .zip(opt_toks.iter())
            .map(|(s, t)| {
                if t.is_empty() {
                    *s
                } else {
                    *s / t.len() as f32
                }
            })
            .collect();
        items.push((z, it.correct));
    }
    if items.is_empty() {
        anyhow::bail!("decision-calibrate: no scorable items");
    }

    let (temperature, nll_before, nll_after) = fit_decision_temperature(&items);
    // Split-conformal calibration: on the SAME labeled corpus, using the
    // fitted temperature, the nonconformity score of each item is
    // `1 − calibrated P(true label)`. For each target coverage the
    // quantile `q = sorted[⌈(n+1)·coverage⌉ − 1]` (clamped to the corpus)
    // is the split-conformal threshold — every option with calibrated
    // prob ≥ `1 − q` enters the prediction set. Serialized per coverage so
    // the `/v1/decide/choice?coverage=` endpoint can build guaranteed sets.
    let conformal_q = fit_conformal_quantiles(&items, temperature, &[0.80, 0.90, 0.95]);
    // Accuracy at T=1 for a human-readable sanity line.
    let acc = items
        .iter()
        .filter(|(z, c)| {
            z.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i == *c)
                .unwrap_or(false)
        })
        .count();
    println!(
        "decision-calibrate: T={temperature:.3}  NLL {nll_before:.3} -> {nll_after:.3}  \
         (argmax accuracy {acc}/{} on the corpus)",
        items.len()
    );

    let calib = rustllama_tuner::DecisionCalibration {
        temperature,
        nll_before,
        nll_after,
        n_items: items.len() as u32,
        conformal_q,
    };
    match persist_decision_calibration(&model_key, calib) {
        Ok(Some(path)) => println!("decision-calibrate: persisted to {}", path.display()),
        Ok(None) => {
            println!("decision-calibrate: no SYCL device — calibration not cached (measured only)")
        }
        Err(e) => tracing::warn!(error = %e, "decision-calibrate: cache write failed"),
    }
    Ok(())
}

/// Persist a decision calibration into the per-device tuner cache under
/// `model_key`. Returns the cache path, or `None` when no device fingerprint
/// is available (mock / no-GPU build).
fn persist_decision_calibration(
    model_key: &str,
    calib: rustllama_tuner::DecisionCalibration,
) -> anyhow::Result<Option<std::path::PathBuf>> {
    let Some(cache_dir) = rustllama_tuner::default_cache_dir() else {
        return Ok(None);
    };
    let (key, device) = rustllama_tuner::cache_context();
    let mut t = rustllama_tuner::load_cache(&cache_dir, &key)
        .ok()
        .flatten()
        .unwrap_or_else(|| rustllama_tuner::TuningResult::empty(key.clone(), device));
    t.decision_calibration.insert(model_key.to_string(), calib);
    rustllama_tuner::save_cache(&cache_dir, &t)?;
    Ok(Some(rustllama_tuner::cache_path_for(&cache_dir, &key)))
}

fn cmd_kv_calibrate(
    config_path: &std::path::Path,
    model: Option<std::path::PathBuf>,
    prompts: Option<std::path::PathBuf>,
    output: Option<std::path::PathBuf>,
    ctx_size: usize,
) -> anyhow::Result<()> {
    use rustllama_engine::{kv_bias, CpuEngine, SamplingParams};

    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let model_path = effective_model_path(
        model.map(|p| p.to_string_lossy().into_owned()),
        &cfg,
        config_path,
    )?;
    let output_path = output.unwrap_or_else(|| model_path.with_extension("kvbias.gguf"));

    let prompt_texts: Vec<String> = match &prompts {
        Some(p) => std::fs::read_to_string(p)
            .map_err(|e| anyhow::anyhow!("read prompts {}: {e}", p.display()))?
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect(),
        None => KV_CALIB_BUILTIN_PROMPTS
            .iter()
            .map(|s| s.to_string())
            .collect(),
    };
    if prompt_texts.is_empty() {
        anyhow::bail!("kv-calibrate: prompt file contains no non-empty lines");
    }

    // Load the way the serve path loads big ternary models: raw quant
    // blocks + zero-copy mmap (an owned copy of a 27B-class model
    // would double RAM), and the chunk-batched hybrid prefill so 8
    // prompts don't take half an hour at serial per-token speed. Set
    // before load — the `*_enabled()` accessors cache on first read.
    std::env::set_var("RUSTLLAMA_KEEP_QUANT_RAW", "1");
    std::env::set_var("RUSTLLAMA_ZEROCOPY_WEIGHTS", "1");
    if std::env::var("RUSTLLAMA_PREFILL_BATCHED_HYBRID").is_err() {
        std::env::set_var("RUSTLLAMA_PREFILL_BATCHED_HYBRID", "1");
    }

    let ctx = ctx_size.max(64);
    println!(
        "kv-calibrate: loading {} (ctx {ctx}, kv q4_0, {} prompt(s))",
        model_path.display(),
        prompt_texts.len()
    );
    let mut cpu = CpuEngine::load_with_options_and_layout(
        &model_path,
        ctx,
        true,
        rustllama_engine::KvDtype::Q4_0,
        "contiguous",
    )
    .map_err(|e| anyhow::anyhow!("load failed: {e}"))?;
    // Prefix-cache reuse would skip KV writes for shared prefixes and
    // starve the accumulator; calibration wants every row observed.
    cpu.set_prefix_cache(false);

    let mcfg = cpu.config();
    let (n_layers, head_dim, n_kv_heads) = (mcfg.n_layers, mcfg.head_dim, mcfg.n_kv_heads);
    let whitening =
        rustllama_engine::kv_whitening_active_for(rustllama_engine::KvDtype::Q4_0, head_dim);
    println!(
        "kv-calibrate: whitening basis {} (recorded in the sidecar)",
        if whitening { "ACTIVE" } else { "inactive" }
    );

    let sampling = SamplingParams {
        temperature: 0.0,
        max_tokens: 1,
        ..SamplingParams::default()
    };

    kv_bias::calib_begin(n_layers);
    let t = std::time::Instant::now();
    for (i, text) in prompt_texts.iter().enumerate() {
        let mut ids = cpu
            .tokenize_chunks([text.clone()], true)
            .map_err(|e| anyhow::anyhow!("tokenize failed: {e}"))?;
        if ids.len() > ctx.saturating_sub(1) {
            ids.truncate(ctx - 1);
        }
        if ids.is_empty() {
            continue;
        }
        cpu.generate_token_ids(&ids, 1, &sampling)
            .map_err(|e| anyhow::anyhow!("calibration forward failed: {e}"))?;
        println!(
            "  prompt {}/{} calibrated ({} tokens)",
            i + 1,
            prompt_texts.len(),
            ids.len()
        );
    }
    let acc = kv_bias::calib_take()
        .ok_or_else(|| anyhow::anyhow!("kv-calibrate: accumulator vanished mid-run"))?;
    let bias = acc.into_bias(whitening);
    let n_centered = bias.n_centered();
    if n_centered == 0 {
        anyhow::bail!(
            "kv-calibrate: no K rows were observed — the forward path did not hit \
             the Q4_0 KV write arms (is this model's attention path supported?)"
        );
    }
    bias.save_with_geometry(&output_path, head_dim, n_kv_heads)
        .map_err(|e| anyhow::anyhow!(e))?;
    println!(
        "kv-calibrate: wrote {} centered layer(s) to {} in {:.1}s",
        n_centered,
        output_path.display(),
        t.elapsed().as_secs_f64()
    );
    println!(
        "               serve auto-discovers it beside the model when kv q4_0 is active; \
         disable with RUSTLLAMA_KV_BIAS=0"
    );
    Ok(())
}

/// Look up the source model's transformer block count by checking
/// `{arch}.block_count` metadata; falls back to scanning tensor
/// names for the highest `blk.{i}.*` index. Returns 0 if neither
/// signal is present — APEX layer-position gradient then no-ops.
fn infer_n_layers_from_gguf(src: &rustllama_gguf::Gguf) -> usize {
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
    // Fallback: scan tensor names for the highest `blk.N.*` index.
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

/// True when the GGUF declares a hybrid transformer+SSM architecture — the
/// same trigger `LlamaConfig::from_gguf` uses (`{arch}.full_attention_interval`
/// AND `{arch}.ssm.state_size` both present). Used by the kv_dtype sweep, which
/// MEASURES the full candidate grid on hybrids (qwen35 / qwen35moe / Ornith) and
/// lets its coherence gate decide which quant KV (if any) is honored — so the
/// autotune, not a manual `RUSTLLAMA_HYBRID_KV_ANY`, is the authority on quant
/// hybrid KV. See [`cmd_tune_kv_dtype`].
fn gguf_is_hybrid(model_path: &std::path::Path) -> bool {
    let Ok(g) = rustllama_gguf::Gguf::open(model_path) else {
        return false;
    };
    let Some(arch) = g.architecture() else {
        return false;
    };
    g.metadata_get(&format!("{arch}.full_attention_interval"))
        .is_some()
        && g.metadata_get(&format!("{arch}.ssm.state_size")).is_some()
}

/// Parse the user's `--target <name>` string into a [`GgmlType`].
/// Case-insensitive; accepts the canonical ggml names. Returns a
/// clean error listing supported targets if the name doesn't match.
#[cfg(feature = "encoder")]
fn parse_target_dtype(name: &str) -> anyhow::Result<rustllama_gguf::GgmlType> {
    use rustllama_gguf::GgmlType;
    let lower = name.to_ascii_lowercase();
    let dtype = match lower.as_str() {
        "f32" => GgmlType::F32,
        "f16" => GgmlType::F16,
        "bf16" => GgmlType::Bf16,
        "q4_0" => GgmlType::Q4_0,
        "q4_1" => GgmlType::Q4_1,
        "q5_0" => GgmlType::Q5_0,
        "q5_1" => GgmlType::Q5_1,
        "q8_0" => GgmlType::Q8_0,
        "q8_1" => GgmlType::Q8_1,
        "q2_k" => GgmlType::Q2_K,
        "q3_k" => GgmlType::Q3_K,
        "q4_k" => GgmlType::Q4_K,
        "q5_k" => GgmlType::Q5_K,
        "q6_k" => GgmlType::Q6_K,
        "q8_k" => GgmlType::Q8_K,
        "tq1_0" => GgmlType::TQ1_0,
        "tq2_0" => GgmlType::TQ2_0,
        "iq4_nl" => GgmlType::IQ4_NL,
        "iq4_xs" => GgmlType::IQ4_XS,
        "iq2_xxs" => GgmlType::IQ2_XXS,
        "iq2_xs" => GgmlType::IQ2_XS,
        "iq2_s" => GgmlType::IQ2_S,
        "iq3_xxs" => GgmlType::IQ3_XXS,
        "iq3_s" => GgmlType::IQ3_S,
        "iq1_s" => GgmlType::IQ1_S,
        "iq1_m" => GgmlType::IQ1_M,
        other => anyhow::bail!(
            "unknown target dtype {other:?}. Supported: f32, f16, bf16, q4_0, q4_1, \
             q5_0, q5_1, q8_0, q8_1, q2_k, q3_k, q4_k, q5_k, q6_k, q8_k, tq1_0, \
             tq2_0, iq2_xxs, iq4_nl, iq4_xs. (IQ1_S/M, IQ2_XS, IQ2_S, IQ3_* land in a later release.)"
        ),
    };
    Ok(dtype)
}

fn run_lsp(
    base_url: String,
    model: String,
    max_tokens: u32,
    temperature: f32,
) -> anyhow::Result<()> {
    let cfg = rustllama_lsp::LspConfig {
        base_url,
        model,
        max_tokens,
        temperature,
        ..rustllama_lsp::LspConfig::default()
    };
    let session = rustllama_lsp::Session::new(cfg)?;
    // Take exclusive locks on stdin/stdout — LSP needs raw byte I/O,
    // and any other writer to stdout would corrupt the framed responses.
    let stdin = std::io::stdin().lock();
    let stdout = std::io::stdout().lock();
    session
        .run(stdin, stdout)
        .map_err(|e| anyhow::anyhow!("LSP session error: {e}"))?;
    Ok(())
}

fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let _ = fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .try_init();
}

// ----- doctor -----

async fn doctor(config_path: &std::path::Path, sycl_smoke: bool) -> anyhow::Result<()> {
    println!("rustllama {} doctor", env!("CARGO_PKG_VERSION"));
    println!();

    let paths = rustllama_runtime::paths();
    println!("install:");
    println!(
        "  mode     = {}",
        if paths.portable { "portable" } else { "system" }
    );
    if let Ok(exe) = std::env::current_exe() {
        println!("  exe      = {}", exe.display());
    }

    println!();
    println!("config:");
    println!("  path     = {}", config_path.display());
    match rustllama_config::load(config_path) {
        Ok(_) => println!("  status   = ok"),
        Err(e) => println!("  status   = ERROR: {e}"),
    }

    println!();
    println!("runtime paths:");
    match rustllama_runtime::runtime_dir() {
        Ok(p) => println!("  runtime  = {}", p.display()),
        Err(e) => println!("  runtime  = ERROR: {e}"),
    }
    if let Some(d) = rustllama_hub::default_cache_dir() {
        println!("  models   = {}", d.display());
    }
    if let Some(d) = rustllama_tuner::default_cache_dir() {
        println!("  tuning   = {}", d.display());
    }
    println!("  sessions = {}", paths.sessions_dir.display());
    println!("  crashes  = {}", paths.crash_log_dir.display());

    println!();
    println!("server discovery:");
    match rustllama_runtime::read_alive_record() {
        Ok(Some(rec)) => println!(
            "  running on {}:{} (pid {}, owner {})",
            rec.bind_addr, rec.port, rec.pid, rec.owner
        ),
        Ok(None) => println!("  none running"),
        Err(e) => println!("  ERROR: {e}"),
    }

    println!();
    println!("toolchains:");
    let oneapi = std::env::var("ONEAPI_ROOT")
        .ok()
        .filter(|p| std::path::Path::new(p).exists())
        .or_else(|| {
            let p = "C:\\Program Files (x86)\\Intel\\oneAPI";
            std::path::Path::new(p).exists().then(|| p.to_string())
        });
    match oneapi {
        Some(p) => println!("  oneAPI   = found at {p}"),
        None => println!("  oneAPI   = NOT FOUND (install Base Toolkit 2025.0+ for GPU build)"),
    }

    println!();
    println!("sycl:");
    match rustllama_kernels_sycl::device_count() {
        Ok(0) => {
            println!("  feature  = compiled in");
            println!("  devices  = 0 (no Intel GPU visible — check drivers / run `sycl-ls`)");
        }
        Ok(n) => {
            println!("  feature  = compiled in");
            println!(
                "  devices  = {n} (oneAPI sees {n} Intel GPU{})",
                if n == 1 { "" } else { "s" }
            );
        }
        Err(rustllama_kernels_sycl::SyclError::Unavailable) => {
            // Real-only build: SYCL is always compiled in, so Unavailable
            // means the SYCL/oneAPI runtime DLLs couldn't be loaded (not a
            // build variant).
            println!("  feature  = compiled in (SYCL runtime not loadable)");
            println!(
                "  devices  = n/a — Intel oneAPI runtime not found; install oneAPI or fix PATH"
            );
        }
        Err(e) => {
            println!("  feature  = compiled in");
            println!("  devices  = ERROR: {e}");
        }
    }

    // CUDA is a first-class peer of SYCL: the native compute kernels are
    // always compiled in, and NVIDIA GPUs are inventoried here exactly like
    // Intel GPUs above. The CUDA *driver* (dlopen, no toolkit) enumerates the
    // GPUs; `kernels_cuda::device_count()` reports how many the CUDA *runtime*
    // can actually launch kernels on ("compute-ready"). On a non-NVIDIA host
    // this prints `devices = 0`, mirroring the `sycl:` section on non-Intel.
    println!();
    println!("cuda:");
    let nvidia = rustllama_runtime::gpu_detect::detect_nvidia();
    let cuda_compute = rustllama_kernels_cuda::device_count();
    match nvidia.as_ref().filter(|i| !i.gpus.is_empty()) {
        Some(info) => {
            let n = info.gpus.len();
            println!("  feature  = compiled in");
            println!(
                "  devices  = {n} (CUDA driver sees {n} NVIDIA GPU{}; {cuda_compute} compute-ready)",
                if n == 1 { "" } else { "s" }
            );
            println!("  driver   = {}", info.driver_version_str());
            for g in &info.gpus {
                println!(
                    "    [{}] {} — {} MiB, compute {}.{}",
                    g.index,
                    g.name,
                    g.total_mem_bytes / (1024 * 1024),
                    g.compute_capability.0,
                    g.compute_capability.1
                );
            }
            if cuda_compute == 0 {
                println!(
                    "  note     = GPUs visible to the driver but the CUDA runtime could \
                     not launch kernels (driver/runtime mismatch?) — CUDA compute disabled"
                );
            }
        }
        None => {
            println!("  feature  = compiled in");
            println!("  devices  = 0 (no NVIDIA GPU visible to the CUDA driver)");
        }
    }

    if sycl_smoke {
        println!();
        println!("sycl smoke (RMSNorm parity vs CPU reference):");
        match run_sycl_rmsnorm_smoke() {
            SmokeResult::Pass { max_abs_err } => {
                println!("  result   = PASS  (max |gpu - cpu| = {max_abs_err:.4e})")
            }
            SmokeResult::Fail { reason } => println!("  result   = FAIL: {reason}"),
            SmokeResult::Skipped { reason } => println!("  result   = SKIPPED ({reason})"),
        }

        // Same harness via the new f32-facing SyclAccelerator wrapper.
        // Exercises the conversion path the engine forward pass will
        // use once kernel dispatches are wired in. Pinning parity at
        // this layer (not just the raw FFI) is the safety gate.
        println!();
        println!("sycl smoke (all kernels via SyclAccelerator wrapper):");
        match run_sycl_accel_smoke() {
            AccelSmoke::Pass(r) => {
                println!(
                    "  rmsnorm  = PASS  (max |gpu - cpu| = {:.4e})",
                    r.rmsnorm_err
                );
                println!(
                    "  embed    = PASS  (max |gpu - cpu| = {:.4e})",
                    r.embedding_err
                );
                println!("  silu_mul = PASS  (max |gpu - cpu| = {:.4e})", r.silu_err);
                println!("  rope     = PASS  (max |gpu - cpu| = {:.4e})", r.rope_err);
                println!(
                    "  softmax  = PASS  (max |gpu - cpu| = {:.4e})",
                    r.softmax_err
                );
                println!("  gemm     = PASS  (max |gpu - cpu| = {:.4e})", r.gemm_err);
            }
            AccelSmoke::Fail { reason } => println!("  result   = FAIL: {reason}"),
            AccelSmoke::Skipped { reason } => {
                println!("  result   = SKIPPED ({reason})")
            }
        }
    }

    Ok(())
}

enum AccelSmoke {
    Pass(AccelSmokeResults),
    Fail { reason: String },
    Skipped { reason: String },
}

struct AccelSmokeResults {
    rmsnorm_err: f32,
    embedding_err: f32,
    silu_err: f32,
    rope_err: f32,
    softmax_err: f32,
    gemm_err: f32,
}

/// Exercises [`rustllama_engine::SyclAccelerator`]'s parity-check
/// methods. These run the f32 → f16 → kernel → f16 → f32 round-trip
/// the engine forward pass will use, so a PASS here means the wrapper
/// + the raw FFI are both correct.
///
/// Tolerances are scaled to each kernel's compute depth:
///   - rmsnorm: d=4096 reduction → ~1e-2 typical f16-accumulation error
///   - embedding: memcpy only → ~1e-4 (single-bit f16 round-trip)
///   - silu_mul: per-element exp/sigmoid → ~5e-3 (sigmoid is f16-sensitive)
///   - rope: cos/sin per element → ~1e-2 (trig + f16 round-trip)
///   - softmax_attn: max + exp + sum → ~5e-3 normalised, bounded
///   - gemm: K=64 reduction (small), ~5e-2 (f16 mul accumulates)
fn run_sycl_accel_smoke() -> AccelSmoke {
    use rustllama_engine::SyclAccelerator;
    let mut a = match SyclAccelerator::try_new(0) {
        Ok(a) => a,
        Err(e) => {
            return AccelSmoke::Skipped {
                reason: e.to_string(),
            }
        }
    };
    let rmsnorm_err = match a.parity_check_rmsnorm(2, 4096, 1e-2) {
        Ok(e) => e,
        Err(e) => {
            return AccelSmoke::Fail {
                reason: format!("rmsnorm: {e}"),
            }
        }
    };
    let embedding_err = match a.parity_check_embedding(1024, 256, 1e-4) {
        Ok(e) => e,
        Err(e) => {
            return AccelSmoke::Fail {
                reason: format!("embedding: {e}"),
            }
        }
    };
    let silu_err = match a.parity_check_silu_mul(4096, 5e-3) {
        Ok(e) => e,
        Err(e) => {
            return AccelSmoke::Fail {
                reason: format!("silu_mul: {e}"),
            }
        }
    };
    // n_heads=4, head_dim=128 covers the typical llama config; pos=17
    // is arbitrary but non-zero so cos/sin aren't trivial.
    let rope_err = match a.parity_check_rope(4, 128, 17, 1e-2) {
        Ok(e) => e,
        Err(e) => {
            return AccelSmoke::Fail {
                reason: format!("rope: {e}"),
            }
        }
    };
    let softmax_err = match a.parity_check_softmax_attn(2, 4, 64, 5e-3) {
        Ok(e) => e,
        Err(e) => {
            return AccelSmoke::Fail {
                reason: format!("softmax_attn: {e}"),
            }
        }
    };
    // Small GEMM — 32x32 @ 32x16. Big enough to exercise the tiled
    // dispatch, small enough that the kernel runs in <10ms even on
    // Iris Xe under the H2D + D2H envelope.
    let gemm_err = match a.parity_check_gemm(32, 16, 32, 5e-2) {
        Ok(e) => e,
        Err(e) => {
            return AccelSmoke::Fail {
                reason: format!("gemm: {e}"),
            }
        }
    };
    AccelSmoke::Pass(AccelSmokeResults {
        rmsnorm_err,
        embedding_err,
        silu_err,
        rope_err,
        softmax_err,
        gemm_err,
    })
}

enum SmokeResult {
    Pass { max_abs_err: f32 },
    Fail { reason: String },
    Skipped { reason: String },
}

/// End-to-end SYCL kernel sanity check. Runs `rsl_rmsnorm` against a
/// pure-Rust f32 reference for a small fixed input and compares the
/// outputs within f16 round-trip tolerance. Used by `rustllama doctor
/// --sycl-smoke` so the user can verify that their oneAPI build can
/// dispatch a GPU kernel end-to-end before plumbing it into a model
/// load. Mirrors the `sycl_rmsnorm_matches_cpu_reference` integration
/// test in `rustllama-kernels-sycl` but is callable from production
/// code, not just `#[test]`.
fn run_sycl_rmsnorm_smoke() -> SmokeResult {
    let count = match rustllama_kernels_sycl::device_count() {
        Ok(c) => c,
        Err(e) => {
            return SmokeResult::Skipped {
                reason: e.to_string(),
            }
        }
    };
    if count == 0 {
        return SmokeResult::Skipped {
            reason: "no SYCL GPUs visible".to_string(),
        };
    }
    let mut stream = match rustllama_kernels_sycl::create_stream(0) {
        Ok(s) => s,
        Err(e) => {
            return SmokeResult::Fail {
                reason: format!("create_stream: {e}"),
            }
        }
    };

    let n_rows: u32 = 2;
    let d: u32 = 64;
    let eps: f32 = 1e-5;

    let mut x = vec![0f32; (n_rows * d) as usize];
    for i in 0..d as usize {
        x[i] = (i as f32) * 0.05;
        x[d as usize + i] = if i % 2 == 0 { 0.3 } else { -0.3 };
    }
    let mut w = vec![0f32; d as usize];
    for i in 0..d as usize {
        w[i] = 0.5 + 0.5 * (i as f32 * 0.1).sin();
    }

    let x_f16: Vec<u16> = x
        .iter()
        .map(|v| half::f16::from_f32(*v).to_bits())
        .collect();
    let w_f16: Vec<u16> = w
        .iter()
        .map(|v| half::f16::from_f32(*v).to_bits())
        .collect();
    let x_round: Vec<f32> = x_f16
        .iter()
        .map(|b| half::f16::from_bits(*b).to_f32())
        .collect();
    let w_round: Vec<f32> = w_f16
        .iter()
        .map(|b| half::f16::from_bits(*b).to_f32())
        .collect();

    let mut y_ref = vec![0f32; (n_rows * d) as usize];
    for r in 0..n_rows as usize {
        let base = r * d as usize;
        let mut sum_sq = 0.0f32;
        for i in 0..d as usize {
            let v = x_round[base + i];
            sum_sq += v * v;
        }
        let scale = 1.0f32 / (sum_sq / d as f32 + eps).sqrt();
        for i in 0..d as usize {
            y_ref[base + i] = x_round[base + i] * scale * w_round[i];
        }
    }

    let mut y_gpu_bits = vec![0u16; (n_rows * d) as usize];
    if let Err(e) = rustllama_kernels_sycl::rmsnorm(
        &mut stream,
        &x_f16,
        &w_f16,
        &mut y_gpu_bits,
        n_rows,
        d,
        eps,
    ) {
        return SmokeResult::Fail {
            reason: format!("rmsnorm dispatch: {e}"),
        };
    }

    let mut max_abs_err = 0.0f32;
    for i in 0..(n_rows * d) as usize {
        let gpu = half::f16::from_bits(y_gpu_bits[i]).to_f32();
        let diff = (gpu - y_ref[i]).abs();
        if diff > max_abs_err {
            max_abs_err = diff;
        }
    }
    // 1e-2 absolute matches the parity test tolerance — f16 round-trip
    // alone costs ~2-3 decimal digits, so anything noticeably tighter
    // is going to false-fail on legitimate rounding.
    const TOL: f32 = 1e-2;
    if max_abs_err > TOL {
        SmokeResult::Fail {
            reason: format!("max |gpu - cpu| = {max_abs_err:.4e} exceeds tolerance {TOL:.0e}"),
        }
    } else {
        SmokeResult::Pass { max_abs_err }
    }
}

// ----- serve -----

/// One-time startup probe: detect and log which compute backend the
/// runtime will use, and make CPU fallback explicit in the logs. This is
/// the operator-facing answer to "is my GPU actually being used?" — a
/// driver problem, a missing oneAPI runtime, or a GPU-less container all
/// surface here as a clean "using CPU" line rather than a silent fallback.
///
/// Detection is failure-safe: `device_count()` returns `Err(Unavailable)`
/// on a mock (CPU-only) build, `Ok(0)` when the SYCL runtime is compiled
/// in but enumerates no usable device (no driver, no GPU passthrough in
/// Docker, broken oneAPI install — the delay-load guard turns all of
/// these into `Ok(0)` instead of an abort), and `Ok(n)` on success. The
/// engine's per-kernel dispatch already falls back to CPU on its own; this
/// only reports the decision and lists the devices a multi-GPU box exposes.
fn log_backend_selection() {
    // Multi-vendor GPU inventory. NVIDIA GPUs come from the CUDA driver
    // (runtime dlopen — no CUDA toolkit needed to *detect* them); Intel
    // GPUs from the SYCL / Level-Zero layer. Today Intel GPUs are
    // dispatched via the SYCL backend and NVIDIA GPUs are inventory-only
    // (the native CUDA compute backend, rustllama-kernels-cuda, is a
    // roadmap item). Auto-selection + Intel/NVIDIA mix-and-match key off
    // this full set.

    // ---- NVIDIA (CUDA driver + native compute crate) ----
    // Driver inventory (dlopen, no toolkit) enumerates the GPUs; the
    // native compute crate (rustllama-kernels-cuda, nvcc-built) reports
    // whether the CUDA *runtime* can actually create a stream + launch
    // kernels on them. Both non-zero ⇒ the CUDA backend is usable.
    let nvidia = rustllama_runtime::gpu_detect::detect_nvidia();
    let cuda_compute_devices = rustllama_kernels_cuda::device_count();
    if let Some(info) = &nvidia {
        for g in &info.gpus {
            tracing::info!(
                vendor = "nvidia",
                device = g.index,
                name = %g.name,
                vram_mb = g.total_mem_bytes / (1024 * 1024),
                compute_cap = %format!("{}.{}", g.compute_capability.0, g.compute_capability.1),
                cuda_driver = %info.driver_version_str(),
                cuda_compute = cuda_compute_devices > 0,
                "GPU detected (NVIDIA/CUDA — native compute kernels available; \
                 verify with `doctor --cuda-parity`)"
            );
        }
    }
    let nvidia_count = nvidia.as_ref().map(|i| i.gpus.len()).unwrap_or(0);

    // ---- Intel (SYCL / Level Zero) ----
    // `device_count()` counts SYCL device *enumerations*, not physical
    // GPUs: one Iris Xe is exposed TWICE — once via Level Zero, once via
    // OpenCL (same name + VRAM, different driver). Dedup by physical
    // identity (name, vendor, VRAM) so the inventory reports GPUs, not
    // backend views. A GPU seen via multiple *different* backends
    // collapses to one; two identical GPUs on the *same* backend stay
    // distinct (their shared backend name repeats within the group).
    let intel_count = match rustllama_kernels_sycl::device_count() {
        Ok(n) if n > 0 => {
            use std::collections::BTreeMap;
            // physical key -> list of (device_idx, driver, backend, xmx_capable)
            let mut groups: BTreeMap<(String, u32, u64), Vec<(u32, String, &'static str, bool)>> =
                BTreeMap::new();
            for idx in 0..n {
                match rustllama_kernels_sycl::device_info(idx) {
                    Ok(info) => {
                        let backend =
                            rustllama_kernels_sycl::current_backend_name(idx).unwrap_or("unknown");
                        groups
                            .entry((info.name.clone(), info.vendor_id, info.vram_mb()))
                            .or_default()
                            .push((idx, info.driver_version.clone(), backend, info.xmx_capable));
                    }
                    Err(e) => tracing::info!(
                        vendor = "intel",
                        device = idx,
                        error = %e,
                        "Intel GPU present but device-info query failed"
                    ),
                }
            }
            let mut physical_total = 0usize;
            for ((name, _vendor, vram_mb), entries) in &groups {
                // Physical count = the most times any single backend
                // appears in this group (identical GPUs on one backend
                // → N; one GPU via L0+OpenCL → both appear once → 1).
                let mut per_backend: BTreeMap<&str, usize> = BTreeMap::new();
                for (_, _, b, _) in entries {
                    *per_backend.entry(*b).or_default() += 1;
                }
                let physical = per_backend.values().copied().max().unwrap_or(1);
                physical_total += physical;
                let backends: Vec<&str> = per_backend.keys().copied().collect();
                let indices: Vec<u32> = entries.iter().map(|(i, _, _, _)| *i).collect();
                // bf16 XMX/DPAS capable (Arc Xe-HPG / PVC); same across identical
                // devices in a group. Informational — the XMX GEMM is opt-in.
                let xmx_capable = entries.iter().any(|(_, _, _, x)| *x);
                // Active backend = the one dispatch actually uses: Level
                // Zero if available, else OpenCL, else whatever's present
                // (matches `first_enabled_sycl_device_index`).
                let active_backend = if per_backend.contains_key("level_zero") {
                    "level_zero"
                } else if per_backend.contains_key("opencl") {
                    "opencl"
                } else {
                    backends.first().copied().unwrap_or("unknown")
                };
                tracing::info!(
                    vendor = "intel",
                    name = %name,
                    vram_mb = vram_mb,
                    physical_gpus = physical,
                    device_indices = ?indices,
                    backends = %backends.join(", "),
                    active_backend = active_backend,
                    xmx_capable = xmx_capable,
                    "GPU detected (Intel/SYCL{})",
                    if entries.len() > physical {
                        " — one physical GPU exposed via Level Zero + OpenCL; \
                         dispatching via Level Zero"
                    } else {
                        ""
                    }
                );
            }
            physical_total
        }
        // Ok(0) = SYCL compiled in but no device; Err(Unavailable) =
        // CPU-only (mock) build; Err(other) = probe failure. All → no
        // usable Intel GPU via SYCL.
        _ => 0,
    };

    // ---- Effective backend summary ----
    if intel_count + nvidia_count == 0 {
        tracing::info!(
            "compute backend = CPU: no usable GPU detected (no SYCL/CUDA GPU, \
             or the GPU runtime failed to load); all inference runs on CPU"
        );
    } else {
        tracing::info!(
            intel_gpus = intel_count,
            nvidia_gpus = nvidia_count,
            cuda_compute_devices,
            "GPU inventory: {intel_count} Intel (SYCL) + {nvidia_count} NVIDIA (CUDA, \
             {cuda_compute_devices} compute-ready). Intel GPUs dispatch via the SYCL \
             backend; the native CUDA kernels are built in — validate them with \
             `doctor --cuda-parity`. Per-model CPU/GPU placement is chosen by the \
             VRAM-fit planner / tuner."
        );
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
async fn serve(
    config_path: &std::path::Path,
    models: Vec<std::path::PathBuf>,
    ip: Option<String>,
    port: Option<u16>,
    api_key: Option<String>,
    no_cpu: bool,
    no_gpu: bool,
    disabled_cpus: Option<String>,
    vram_only: bool,
    profile: Option<&str>,
) -> anyhow::Result<()> {
    use rustllama_engine::paged_batch::PagedBatchEngine;
    use rustllama_engine::{CpuEngine, Engine};
    use rustllama_server::{AppState, ServingModel};
    use std::net::SocketAddr;

    let mut cfg = rustllama_config::load_with_profile(config_path, profile).unwrap_or_default();
    if let Some(name) = profile {
        if !name.is_empty() {
            println!("rustllama: profile `{name}` active");
        }
    }

    // `--api-key` overrides `[server].api_key` (and the RUSTLLAMA_API_KEY env
    // override already folded in by config load) — flag > env > config. This
    // is the bearer token remote clients must present as
    // `Authorization: Bearer <token>`; empty keeps open access.
    if let Some(key) = api_key {
        cfg.server.api_key = key;
    }

    // Device / memory-tier CLI flags override the corresponding
    // `[inference]` config keys, exactly like `--ip`/`--port` override
    // `[server]`. These then flow to the engine + server via config +
    // `promote_memory_env_from_config` (the env vars runtime/engine read).
    if no_cpu {
        cfg.inference.cpu_enabled = false;
    }
    if no_gpu {
        cfg.inference.gpu_enabled = false;
    }
    // Both tiers off leaves the placement planner nothing to target. Config
    // `validate()` catches this at load, but the CLI flags override cfg AFTER
    // load, so re-check here.
    if !cfg.inference.cpu_enabled && !cfg.inference.gpu_enabled {
        anyhow::bail!(
            "--no-cpu and --no-gpu (or [inference].cpu_enabled/gpu_enabled) are both \
             disabled — no compute tier remains; enable at least one"
        );
    }
    if vram_only {
        cfg.inference.vram_only = true;
    }
    if let Some(csv) = disabled_cpus.as_deref() {
        cfg.inference.disabled_cpus = csv
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter_map(|t| t.trim().parse::<u32>().ok())
            .collect();
    }
    // Promote the CPU-tier settings to env EARLY (before the server binds),
    // so the inventory surfaces (`/v1/capabilities`, `/v1/tuning_summary`)
    // reflect them even when the server starts with no model loaded. The
    // per-model `promote_memory_env_from_config` below is idempotent (a set
    // env var wins), so this doesn't double-apply.
    promote_cpu_tier_env_from_config(&cfg.inference);

    // Single-instance guard: take the advisory `runtime/server.lock`
    // and hold it for the whole process lifetime. A second server
    // (CLI or the GUI's embedded serve) against the same runtime dir
    // then fails fast here instead of racing on the bind port + the
    // `runtime/server.json` discovery record. The guard drops when
    // `serve` returns (or the process exits), releasing the lock. A
    // stale lock from a crashed server is auto-released by the OS
    // (advisory lock tied to the file handle), so this never blocks on
    // a dead predecessor. A filesystem error is non-fatal — we log and
    // serve without the guard rather than refuse to start.
    let _server_lock: Option<rustllama_runtime::ServerLock> =
        match rustllama_runtime::acquire_server_lock() {
            Ok(Some(guard)) => Some(guard),
            Ok(None) => anyhow::bail!(
                "another rustllama server is already running against this runtime \
                 directory (runtime/server.lock is held). Stop it first, or point \
                 this instance at a separate install/data directory."
            ),
            Err(e) => {
                tracing::warn!(error = %e, "could not acquire runtime/server.lock; \
                    starting without the single-instance guard");
                None
            }
        };

    // Install the Job Object sandbox before any heavy allocation
    // (model load, engine spawn). The memory cap acts on every
    // subsequent malloc, so attaching here catches a malformed-GGUF
    // runaway alloc cleanly. Non-zero `sandbox_memory_limit_mb`
    // engages the limiter; `0` keeps the legacy unlimited path.
    if cfg.server.sandbox_memory_limit_mb > 0 {
        let bytes = (cfg.server.sandbox_memory_limit_mb as u64) * 1024 * 1024;
        match rustllama_runtime::sandbox::install(rustllama_runtime::sandbox::SandboxConfig {
            memory_limit_bytes: bytes,
            kill_on_close: true,
        }) {
            Ok(()) => tracing::info!(
                memory_limit_mb = cfg.server.sandbox_memory_limit_mb,
                "process sandbox engaged"
            ),
            Err(e) => tracing::warn!(
                error = %e,
                "sandbox install failed — continuing without sandboxing"
            ),
        }
    }

    // Detect and log the compute backend up-front so a CPU fallback
    // (missing driver, GPU-less container, broken oneAPI) is visible in
    // the startup log rather than inferred from performance.
    log_backend_selection();

    let bind_addr = ip.unwrap_or(cfg.server.bind_addr);
    let port = port.unwrap_or(cfg.server.port);
    let addr: SocketAddr = format!("{}:{}", authority_host(&bind_addr), port)
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid --ip {bind_addr} / --port {port}: {e}"))?;

    let max_pending = cfg.server.max_pending_per_model as usize;
    let concurrency = cfg.server.concurrency.max(1) as usize;
    // Host-environment hint for tool/function-calling: server-wide,
    // computed once, injected into tools requests so the model picks the
    // right shell dialect. Mirror `[server].tool_environment_hint` into
    // the server's process-global toggle (default true).
    rustllama_server::env_hint::set_tool_environment_hint_enabled(
        cfg.server.tool_environment_hint,
    );
    // Fused-decode continuous batching is opt-in via `[server].fused_decode`.
    // It is NOT the default: the contiguous CpuEngine path keeps prefix-cache,
    // string stop-sequences, and grammar constraints, which the paged path does
    // not yet support — so single-user chat stays on the richer backend.
    // `PagedBatchEngine` builds its own paged KV pool, so `fused_decode` alone is
    // sufficient; we no longer silently no-op when `kv_cache_layout` is left at
    // the "contiguous" default. Warn on the mismatch instead of ignoring the flag.
    let use_fused_decode = cfg.server.fused_decode;
    if use_fused_decode && cfg.inference.kv_cache_layout != "paged" {
        tracing::warn!(
            kv_cache_layout = %cfg.inference.kv_cache_layout,
            "fused_decode is enabled; PagedBatchEngine uses its own paged KV pool \
             regardless of [inference].kv_cache_layout"
        );
    }

    // Optional engine: when `[model].path` is configured we load it
    // up-front and seed the registry with it. When it's empty (fresh
    // install, or the user wants to load models exclusively via the
    // GUI / `POST /v1/models/load`) the server starts with an empty
    // registry. No mock fallback — that was a footgun where every
    // chat emitted canned tokens against a "live"-looking server.
    //
    // Two backends:
    //   - **Fused decode** (`[server].fused_decode = true` +
    //     `[inference].kv_cache_layout = "paged"`): load
    //     [`PagedBatchEngine`]. One shared paged KV pool + driver
    //     thread arbitrates fused multi-slot decode across up to
    //     `concurrency` HTTP requests.
    //   - **CpuEngine + multi-flight** (default): existing path,
    //     `concurrency` independent forks via `with_concurrency`.
    // Startup model list. Explicit `--model` paths win over `[model].path`;
    // the FIRST entry becomes the server default (the rest are registered
    // alongside it and reachable by id / via `swap`). Empty → start with an
    // empty registry (load later via the GUI / `POST /v1/models/load`).
    let startup_models: Vec<std::path::PathBuf> = if !models.is_empty() {
        models
    } else {
        cfg.model.path.clone().into_iter().collect()
    };
    // Resolve each `--model` entry the same way `model use`/`rm` do: an
    // existing path (absolute or CWD-relative) is kept verbatim, otherwise
    // the string is treated as a hub ref / cached id and mapped to its
    // location under `models/<org>__<repo>/<file>`. Without this,
    // `serve --model <the exact ref you pulled>` failed with "not found"
    // even though the file sat in the cache — the user had to spell out the
    // full nested path. An entry that resolves to nothing is left as typed
    // so the existence check below reports it against the user's spelling.
    let startup_models: Vec<std::path::PathBuf> = startup_models
        .iter()
        .map(|m| resolve_model_path(&m.to_string_lossy()).unwrap_or_else(|_| m.clone()))
        .collect();
    for m in &startup_models {
        if !m.exists() {
            anyhow::bail!(
                "--model {}: not found (pass a `.gguf` file, a `.safetensors` \
                 AWQ/GPTQ checkpoint, an MLX model directory, or a hub ref / \
                 cached id like `org/repo:file.gguf` you've `pull`ed)",
                m.display()
            );
        }
    }

    // Load (and, on first sight, auto-tune) each startup model. Autotune is
    // now MANDATORY before a model is served: the one-time full sweep runs
    // (progress echoed to this console) whenever the model has never been
    // tuned on this device, populating the tuner-cache winners the load below
    // reads (kv_dtype / placement / per-device perf + the coherence
    // guardrail). An already-cached model is an INSTANT no-op; a failed sweep
    // is non-fatal (the load falls back to config + guardrail defaults).
    //
    // This runs for the GUI's embedded server too (`RUSTLLAMA_GUI_EMBEDDED`):
    // a startup model configured via `[model].path` is NOT marked ready until
    // its tune completes — the server binds its port only after the model is
    // tuned + loaded, so the health/ready signal flips true only post-tune.
    // (The GUI's on-demand loads go through POST /v1/models/load, which runs
    // the same mandatory tune behind its progress modal.)
    let mut loaded_startup: Vec<(ServingModel, String)> = Vec::new();
    for path in &startup_models {
        // Mandatory first-load auto-tune for THIS model, before it loads.
        // The tune subprocess loads via the GGUF path, so a model DIRECTORY
        // (MLX / safetensors export) has no GGUF to tune — skip the sweep;
        // it loads with engine defaults (F32 contiguous KV).
        if !path.is_dir() {
            let path = path.clone();
            let _ = tokio::task::spawn_blocking(move || {
                rustllama_server::maybe_first_load_autotune(&path, true)
            })
            .await;
        }
        let loaded_one: (ServingModel, String) = if use_fused_decode {
            let max_ctx = cfg.inference.ctx_size as usize;
            tracing::info!(
                model = %path.display(),
                max_ctx,
                max_slots = concurrency,
                "loading PagedBatchEngine (fused-decode CB mode)"
            );
            let paged = PagedBatchEngine::load(path, max_ctx, concurrency as u32).map_err(|e| {
                anyhow::anyhow!("failed to load fused-decode model {}: {e}", path.display())
            })?;
            let model_id = paged.model_id().to_string();
            let paged = Arc::new(paged);
            let serving = ServingModel::new_fused_decode_paged(
                paged,
                model_id.clone(),
                max_pending,
                concurrency,
            );
            (serving, model_id)
        } else {
            let max_ctx = cfg.inference.ctx_size as usize;
            tracing::info!(model = %path.display(), max_ctx, "loading CPU engine");
            // Honor `[inference].keep_quant_raw` — AND force it on whenever a
            // GPU is present, because the GPU path wants PACKED (Raw) weights:
            // the USM pre-upload only covers `*Raw` tensors and packed USM
            // matvec is the fast GPU kernel. Without this, sub-L3 tensors
            // dequant to F16, skip the pre-upload, and run on CPU (the hybrid
            // <1 tok/s regression — even after adding hybrid_layers to the
            // pre-upload, F16-dequanted tensors wouldn't be covered). CPU-only
            // serving keeps the config default (F16 is faster on CPU).
            let gpu_present = rustllama_kernels_sycl::device_count()
                .map(|n| n > 0)
                .unwrap_or(false)
                || rustllama_kernels_cuda::device_count() > 0;
            if cfg.inference.keep_quant_raw || gpu_present {
                std::env::set_var("RUSTLLAMA_KEEP_QUANT_RAW", "1");
                tracing::info!(
                    gpu_present,
                    "keep_quant_raw enabled: quantized tensors stay in raw GGUF form \
                     (packed USM pre-upload + packed GPU matvec; skips F16 dequant)"
                );
            }
            promote_memory_env_from_config(&cfg.inference, &cfg.tuning);
            // Chunked-parallel SSM (DeltaNet) prefill: honor the tuner-cache
            // winner too. The config→env promotion already ran inside
            // `promote_memory_env_from_config` above; this additionally enables
            // the path when the cache holds a `true` winner and config left it
            // off. A user-set env var wins over both (checked via `is_none`).
            if ssm_prefill_chunked_from_cache_or_default(cfg.tuning.auto_apply_ssm_prefill_chunked)
                == Some(true)
                && std::env::var_os("RUSTLLAMA_SSM_PREFILL_CHUNKED").is_none()
            {
                std::env::set_var("RUSTLLAMA_SSM_PREFILL_CHUNKED", "1");
                tracing::info!("tuner cache: applied ssm_prefill_chunked winner");
            }
            // Export the autotuned flash-attn-v3 KV_TILE to the env
            // var the SYCL kernel reads at dispatch time. Only set
            // when the cache holds an entry AND the user hasn't
            // already set the env var (user override wins). Missing
            // cache → leave unset → kernel defaults to 32.
            if std::env::var_os("RUSTLLAMA_FLASH_V3_KV_TILE").is_none() {
                if let Some(t) =
                    flash_v3_kv_tile_from_cache_or_default(cfg.tuning.auto_apply_flash_v3_kv_tile)
                {
                    tracing::info!(applied = t, "tuner cache: applied flash_v3_kv_tile winner");
                    std::env::set_var("RUSTLLAMA_FLASH_V3_KV_TILE", t.to_string());
                }
            }
            // Promote the typed `[inference].flash_attention_kv_min` to
            // the env var the kernel-dispatch site reads. Precedence:
            //   1. User-set env var (lets the tuner sweep without
            //      rewriting config).
            //   2. Tuner cache winner (auto-apply via
            //      `flash_kv_min_from_cache_or_default`).
            //   3. The typed config value.
            if std::env::var_os("RUSTLLAMA_FLASH_KV_LEN_MIN").is_none() {
                let applied = flash_kv_min_from_cache_or_default(
                    cfg.inference.flash_attention_kv_min,
                    cfg.tuning.auto_apply_flash_kv_min,
                );
                if applied != cfg.inference.flash_attention_kv_min {
                    tracing::info!(
                        configured = cfg.inference.flash_attention_kv_min,
                        applied = applied,
                        "tuner cache: applied flash_attention_kv_min winner"
                    );
                }
                std::env::set_var("RUSTLLAMA_FLASH_KV_LEN_MIN", applied.to_string());
            }
            // Resolve the per-channel K and V dtypes from config. If
            // [inference].k_dtype / v_dtype are set they override
            // [inference].kv_dtype; otherwise both fall back to the
            // homogeneous value. Engine storage today is a coupled
            // `KvLayer` enum that pairs K and V into one variant
            // per dtype, so K's dtype is what reaches the kernel
            // when the two differ. We warn rather than refuse so
            // configs already targeting future per-side storage
            // continue to load.
            let (k_str, v_str) = cfg.inference.resolved_kv_dtypes();
            if k_str != v_str {
                tracing::warn!(
                    k = k_str,
                    v = v_str,
                    "[inference].k_dtype and v_dtype differ; engine storage \
                         currently couples K/V — V's setting will mirror K until \
                         per-side KV storage lands"
                );
            }
            // Tuner-cache override for kv_dtype + kv_cache_layout:
            // when the user has the corresponding auto_apply flag
            // on (default) AND a cached winner exists, use that
            // instead of the config value. Both fall back to the
            // config string on cache miss / no-SYCL builds, so
            // fresh installs behave identically to the legacy path.
            let configured_kv_dtype = k_str.to_string();
            let cache_kv = kv_dtype_from_cache_or_default(cfg.tuning.auto_apply_kv_dtype);
            let from_cache = cache_kv.is_some();
            let applied_kv_dtype_str = cache_kv.unwrap_or_else(|| configured_kv_dtype.clone());
            if applied_kv_dtype_str != configured_kv_dtype {
                tracing::info!(
                    configured = %configured_kv_dtype,
                    applied = %applied_kv_dtype_str,
                    "tuner cache: applied kv_dtype winner"
                );
            }
            // Coherence guardrail (1d): auto-downgrade an aggressive
            // quantized KV cache to f32 when the model isn't
            // validated for it (no kvbias calibration sidecar) —
            // prevents the Qwen 2.5 gibberish class silently. When the
            // dtype came from the tuner CACHE, trust it (force=true): the
            // tuner already validated the winner via its 0.90 coherence
            // gate, so a coherent q4_0 winner isn't second-guessed back
            // to f32. A user-CONFIGURED (non-cache) dtype keeps the strict
            // guard (force only when RUSTLLAMA_FORCE_QUANT_KV is set).
            let (safe_kv_dtype_str, kv_guard_warn) = rustllama_config::coherence_safe_kv_dtype(
                &applied_kv_dtype_str,
                path,
                from_cache || rustllama_config::force_quant_kv_from_env(),
            );
            if let Some(w) = kv_guard_warn {
                tracing::warn!("{w}");
            }
            // A tuner-cache kv_dtype winner is coherence-gated by the sweep —
            // now including hybrids (the sweep measures the full grid on them).
            // Signal the engine's load-time hybrid-KV coercion to honor the
            // validated dtype verbatim, so the autotune — not a manual env var
            // — is the authority on quant hybrid KV. Scoped to a non-f32 winner
            // (nothing to lift otherwise); harmless on non-hybrids, where the
            // coercion never fires.
            if from_cache && !safe_kv_dtype_str.eq_ignore_ascii_case("f32") {
                std::env::set_var("RUSTLLAMA_HYBRID_KV_ANY", "1");
            }
            let kv_dtype = parse_kv_dtype(&safe_kv_dtype_str)?;

            let configured_layout = cfg.inference.kv_cache_layout.clone();
            let applied_layout =
                kv_cache_layout_from_cache_or_default(cfg.tuning.auto_apply_kv_cache_layout)
                    .unwrap_or_else(|| configured_layout.clone());
            if applied_layout != configured_layout {
                tracing::info!(
                    configured = %configured_layout,
                    applied = %applied_layout,
                    "tuner cache: applied kv_cache_layout winner"
                );
            }

            let applied_page_size = kv_page_size_from_cache_or_default(
                cfg.inference.kv_page_size,
                cfg.tuning.auto_apply_kv_page_size,
            );
            if applied_page_size != cfg.inference.kv_page_size {
                tracing::info!(
                    configured = cfg.inference.kv_page_size,
                    applied = applied_page_size,
                    "tuner cache: applied kv_page_size winner"
                );
            }
            let mut cpu = if path.is_dir() {
                // MLX / safetensors model DIRECTORY: route through the
                // format-detecting loader (dequant-to-f16 for MLX). It fixes
                // F32 contiguous KV, so the GGUF-oriented kv_dtype / layout /
                // page-size winners resolved above don't apply here.
                tracing::info!(
                    model = %path.display(),
                    "loading model directory via load_auto (MLX / safetensors)"
                );
                CpuEngine::load_auto(path, max_ctx).map_err(|e| {
                    anyhow::anyhow!("failed to load model {}: {e}", path.display())
                })?
            } else {
                CpuEngine::load_with_options_layout_and_page_size(
                    path,
                    max_ctx,
                    true,
                    kv_dtype,
                    &applied_layout,
                    applied_page_size,
                )
                .map_err(|e| {
                    anyhow::anyhow!("failed to load model {}: {e}", path.display())
                })?
            };
            // K-cache mean-centering bias sidecar (explicit path
            // from config, else auto-discovery beside the model).
            // A malformed / basis-mismatched sidecar fails the
            // load loudly — a silent skip would quietly degrade
            // quantized-KV quality instead.
            cpu.attach_kv_bias(
                cfg.inference
                    .kv_bias_path
                    .as_deref()
                    .map(std::path::Path::new),
            )
            .map_err(|e| anyhow::anyhow!("kv-bias attach failed: {e}"))?;
            // Vision tower ([model].mmproj): attach post-load so it
            // composes with the real KV dtype/layout the serve path
            // chose (the legacy load_with_mmproj entry hardcoded
            // F32). Placeholder default is the Qwen-VL pad token;
            // the engine derives the vision start/end wrapper from
            // the mmproj's projector family.
            if let Some(mmproj) = cfg.model.mmproj.as_ref() {
                cpu.attach_mmproj(
                    mmproj,
                    "<|image_pad|>",
                    rustllama_engine::PlaceholderMode::OnePerImage,
                )
                .map_err(|e| anyhow::anyhow!("mmproj attach failed: {e}"))?;
            }
            cpu.set_prefix_cache(cfg.inference.prefix_cache);
            let applied_snapshots = prefix_snapshots_from_cache_or_default(
                cfg.inference.prefix_cache_max_snapshots,
                cfg.tuning.auto_apply_prefix_cache_max_snapshots,
            );
            if applied_snapshots != cfg.inference.prefix_cache_max_snapshots {
                tracing::info!(
                    configured = cfg.inference.prefix_cache_max_snapshots,
                    applied = applied_snapshots,
                    "tuner cache: applied prefix_cache_max_snapshots winner"
                );
            }
            cpu.set_prefix_cache_max_snapshots(applied_snapshots as usize);
            // Consumer-side hooks for the tuner cache: when
            // `[tuning].auto_apply_*` is true (default) AND the
            // cache holds an entry for this device + model, the
            // cached winner overrides the config default. Both
            // hooks fall back to the config value on cache miss
            // or no-SYCL builds — backward compat for fresh
            // installs and mock-mode tests.
            let model_key = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown-model")
                .to_string();
            let applied_batch_size = batch_size_from_cache_or_default(
                cfg.inference.batch_size,
                cfg.tuning.auto_apply_batch_size,
            );
            if applied_batch_size != cfg.inference.batch_size {
                tracing::info!(
                    configured = cfg.inference.batch_size,
                    applied = applied_batch_size,
                    "tuner cache: applied batch_size winner"
                );
            }
            cpu.set_prefill_chunk_size(applied_batch_size as usize);
            cpu.set_max_tool_iterations(cfg.inference.max_tool_iterations);
            if let Some(draft_path) = cfg.inference.speculative_draft_path.as_deref() {
                // Two-model speculative decoding. The draft loads
                // with F32 KV (it's small; exactness helps
                // acceptance) and shares nothing with the target
                // except the tokenizer — which is verified, not
                // assumed: accept/reject compares token IDs.
                let draft_path = std::path::Path::new(draft_path);
                let draft = CpuEngine::load_with_options_and_layout(
                    draft_path,
                    max_ctx,
                    true,
                    rustllama_engine::KvDtype::F32,
                    "contiguous",
                )
                .map_err(|e| {
                    anyhow::anyhow!(
                        "failed to load speculative draft model {}: {e}",
                        draft_path.display()
                    )
                })?;
                cpu.draft_compatible_with(&draft)
                    .map_err(|e| anyhow::anyhow!("speculative draft model rejected: {e}"))?;
                let k = cfg.inference.speculative_draft_k.max(1);
                cpu.set_draft_speculative(Some((
                    std::sync::Arc::new(draft) as std::sync::Arc<dyn rustllama_engine::Engine>,
                    k,
                )));
                tracing::info!(
                    draft = %draft_path.display(),
                    k,
                    "draft-model speculative decoding enabled (raw-softmax \
                     sampling; grammar requests fall back to the classic sampler)"
                );
            }
            if cfg.inference.speculative_ngram {
                cpu.set_ngram_speculative(Some(
                    rustllama_engine::speculative::NgramDrafterConfig {
                        n_match: cfg.inference.ngram_n_match as usize,
                        n_draft: cfg.inference.ngram_n_draft as usize,
                        ..Default::default()
                    },
                ));
                tracing::info!(
                    n_match = cfg.inference.ngram_n_match,
                    n_draft = cfg.inference.ngram_n_draft,
                    "n-gram speculative decoding enabled (raw-softmax sampling; \
                         grammar requests fall back to the classic sampler)"
                );
            }
            // MTP / NextN self-speculation (hybrid + NextN-head models
            // only; a silent no-op otherwise). Takes precedence over
            // n-gram at dispatch time when both are enabled. Tuner-cache
            // winner (gated by auto_apply_speculative_mtp) overrides the
            // config default; falls back to config on cache miss.
            let applied_mtp =
                speculative_mtp_from_cache_or_default(cfg.tuning.auto_apply_speculative_mtp)
                    .unwrap_or(cfg.inference.speculative_mtp);
            if applied_mtp != cfg.inference.speculative_mtp {
                tracing::info!(
                    configured = cfg.inference.speculative_mtp,
                    applied = applied_mtp,
                    "tuner cache: applied speculative_mtp winner"
                );
            }
            cpu.set_mtp_speculative(applied_mtp);
            if applied_mtp {
                tracing::info!(
                    "MTP / NextN self-speculative decoding enabled (raw-softmax \
                         sampling; used only on hybrid models with a NextN head; \
                         grammar requests fall back to the classic sampler)"
                );
            }
            let applied_flash =
                flash_attention_from_cache_or_default(cfg.tuning.auto_apply_flash_attention)
                    .unwrap_or(cfg.inference.flash_attention);
            if applied_flash != cfg.inference.flash_attention {
                tracing::info!(
                    configured = cfg.inference.flash_attention,
                    applied = applied_flash,
                    "tuner cache: applied flash_attention winner"
                );
            }
            cpu.set_flash_attention(applied_flash);
            // Pull CPU-force patterns from
            // `[inference].placement.overrides` FIRST — the placement
            // decision below (VRAM-fit planner + first-load probe)
            // must exclude pinned tensors from its per-layer byte
            // counts and its timing. V1 ignores any non-"cpu" device
            // targets (multi-GPU routing is a later milestone).
            let cpu_patterns: Vec<String> = cfg
                .inference
                .placement
                .overrides
                .iter()
                .filter(|o| o.device.eq_ignore_ascii_case("cpu"))
                .map(|o| o.pattern.clone())
                .collect();
            if !cpu_patterns.is_empty() {
                tracing::info!(
                    n_patterns = cpu_patterns.len(),
                    "placement.overrides: pinning {} pattern(s) to CPU",
                    cpu_patterns.len(),
                );
            }
            cpu.set_cpu_force_patterns(cpu_patterns);
            // CPU/GPU placement, chosen by precedence:
            //   1. HARD device-tier guardrail (`cpu_enabled=false` /
            //      `vram_only`): measured-perf heat placement (VRAM-fit
            //      fallback pre-tune) that REFUSES the load on overflow;
            //   2. else an explicit `[inference].n_gpu_layers` override
            //      (any value other than the AUTO sentinel), honored only
            //      when the GPU tier is enabled;
            //   3. else a flat placement winner cached for this
            //      (device, model) from `rustllama tune --placement`;
            //   4. else AUTO: measured-perf heat placement, which subsumes
            //      the flat layer cutoff + the MoE experts→CPU split and
            //      falls back to the VRAM-fit planner before the tune has
            //      measured this device.
            let cached_placement = cfg
                .tuning
                .auto_apply_placement
                .then(|| placement_cache_lookup(&model_key))
                .flatten();
            // An explicit user override is honored only while the GPU tier is
            // enabled; `gpu_enabled=false` forces the all-CPU auto path.
            let override_n = if cfg.inference.gpu_enabled {
                cfg.inference.n_gpu_layers_override()
            } else {
                None
            };
            // Device-tier options for the placement planner. `cpu_enabled` /
            // `gpu_enabled` / `vram_only` come from config (already overridden
            // by the `serve --no-cpu` / `--no-gpu` / `--vram-only` flags above).
            let placement_opts = rustllama_engine::placement_auto::AutoPlacementOpts {
                cpu_enabled: cfg.inference.cpu_enabled,
                gpu_enabled: cfg.inference.gpu_enabled,
                vram_only: cfg.inference.vram_only,
                ..Default::default()
            };
            // `cpu_enabled = false` (GPU-only) and `vram_only` are HARD
            // device-tier constraints: run the planner and REFUSE the load
            // when the model doesn't satisfy them, instead of silently
            // spilling weights to the CPU / host RAM.
            let force_gpu_fit = !cfg.inference.cpu_enabled || cfg.inference.vram_only;
            let applied_n_gpu_layers = if force_gpu_fit {
                let decision = cpu.auto_place_heat(&placement_opts);
                if decision.vram_only_noop {
                    tracing::warn!(
                        "vram_only requested but this host has no separate dedicated VRAM \
                         (unified-memory GPU / CPU-only) — vram_only is a no-op; weights use \
                         the normal RAM/USM residency path"
                    );
                }
                if let Some(err) = &decision.placement_error {
                    anyhow::bail!("model placement failed for {}: {err}", path.display());
                }
                tracing::info!(
                    n_gpu_layers = decision.n_gpu_layers,
                    total_layers = decision.total_layers,
                    cpu_enabled = cfg.inference.cpu_enabled,
                    vram_only = cfg.inference.vram_only,
                    reason = %decision.reason,
                    "device-tier placement (cpu_enabled/vram_only guardrail)",
                );
                decision.n_gpu_layers
            } else if let Some(n) = override_n {
                tracing::info!(
                    model = %model_key,
                    applied = n,
                    "explicit [inference].n_gpu_layers override (bypassing auto placement)"
                );
                n
            } else if let Some(n) = cached_placement {
                tracing::info!(
                    model = %model_key,
                    applied = n,
                    "tuner cache: applied n_gpu_layers winner"
                );
                n
            } else {
                // AUTO (the zero-config default): measured-perf heat placement.
                // Installs a per-tensor device plan when the tune has measured
                // this device; otherwise falls back to the VRAM-fit planner
                // (fit the model + KV cache into GPU VRAM, spill the coldest
                // layers to CPU/RAM). GPU VRAM is real added capacity
                // (integrated GPUs included).
                let decision = cpu.auto_place_heat(&placement_opts);
                tracing::info!(
                    n_gpu_layers = decision.n_gpu_layers,
                    total_layers = decision.total_layers,
                    device_global_mem_bytes = decision.device_global_mem_bytes,
                    effective_budget_bytes = decision.effective_budget_bytes,
                    gpu_resident_bytes = decision.gpu_resident_bytes,
                    reason = %decision.reason,
                    "auto placement: measured-perf heat plan (VRAM-fit fallback pre-tune)",
                );
                decision.n_gpu_layers
            };
            cpu.set_n_gpu_layers(applied_n_gpu_layers);
            let id = cpu.model_id().to_string();
            let cpu = Arc::new(cpu);
            // Pre-upload packed-quant weights to USM on a blocking-
            // pool thread so the first chat doesn't pay the lazy
            // weight-cache miss cost. No-op when SYCL is disabled or
            // no GPU is visible — won't block startup on CPU-only.
            cpu.warmup_for_sycl_async().await;
            let serving = ServingModel {
                engine: cpu.clone() as Arc<dyn Engine>,
                cpu_engine: Some(cpu),
                model_id: id.clone(),
                gate: ServingModel::new_gate(),
                scheduler: ServingModel::new_scheduler(),
                pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
                max_pending,
                last_used: Arc::new(std::sync::atomic::AtomicU64::new(0)),
                multi: None,
            }
            .with_concurrency(concurrency);
            (serving, id)
        };
        loaded_startup.push(loaded_one);
    }
    if startup_models.is_empty() {
        tracing::info!(
            "no model configured; starting with empty registry — load a model \
             via the GUI Models page or `POST /v1/models/load`"
        );
    }

    // Seed the registry: the first startup model is the default; register
    // any remaining ones alongside it. The loaded cap is bumped so none of
    // the explicitly-requested startup models is immediately LRU-evicted
    // (0 keeps the "unlimited registry" semantics).
    let (state, model_id_for_record): (AppState, Option<String>) = if loaded_startup.is_empty() {
        (
            AppState::empty(
                env!("CARGO_PKG_VERSION").to_string(),
                cfg.server.max_loaded_models as usize,
                max_pending,
            )
            .with_config_path(config_path.to_path_buf()),
            None,
        )
    } else {
        let cfg_cap = cfg.server.max_loaded_models as usize;
        let cap = if cfg_cap == 0 {
            0
        } else {
            cfg_cap.max(loaded_startup.len())
        };
        let mut it = loaded_startup.into_iter();
        let (first_serving, first_id) = it.next().unwrap();
        let record_id = first_id.clone();
        let state =
            AppState::with_max_loaded(first_serving, env!("CARGO_PKG_VERSION").to_string(), cap)
                .with_config_path(config_path.to_path_buf());
        for (serving, model_id) in it {
            tracing::info!(model = %model_id, "registering additional startup model");
            state.upsert(serving).await;
        }
        (state, Some(record_id))
    };
    // Open the sqlite-backed conversations store so `/api/conversations`
    // becomes functional and the CLI's `conv` subcommands have data to
    // read. Failures here are non-fatal — the server still serves chats
    // even without history persistence; the user just loses the
    // sidebar / `conv` flow. Slim builds (`--no-default-features`)
    // compile the whole block out; /api/conversations 501s.
    #[cfg(feature = "history")]
    let state = match conv_db_path() {
        Some(path) => match rustllama_server::history::HistoryStore::open(path.clone()) {
            Ok(store) => {
                tracing::info!(path = %path.display(), "conversation history store ready");
                state.with_history(Arc::new(store))
            }
            Err(e) => {
                tracing::warn!(
                    path = %path.display(),
                    error = %e,
                    "failed to open conversation history store — /api/conversations will 501"
                );
                state
            }
        },
        None => state,
    };

    // Enable the lazy embedding-model slot when [embeddings] is
    // configured. The slot itself is empty until the first
    // /v1/embeddings request triggers the load — eager-load at
    // startup would hold ~200MB-2GB of weights for a feature
    // most servers never invoke.
    let state = if cfg.embeddings.path.is_some() || cfg.embeddings.hub.is_some() {
        tracing::info!("[embeddings] configured — lazy-load slot enabled");
        state.with_embedding_slot()
    } else {
        state
    };
    // Same lazy-load story for [reranker] — backs POST /v1/rerank
    // when configured, otherwise the endpoint 501s with the
    // "no model configured" diagnostic.
    let state = if cfg.reranker.path.is_some() || cfg.reranker.hub.is_some() {
        tracing::info!("[reranker] configured — lazy-load slot enabled");
        state.with_reranker_slot()
    } else {
        state
    };

    // Attach bearer-token auth when [server].api_key is non-empty.
    // The middleware bypasses /healthz and loopback connections so
    // the GUI's own fetches and local CLI clients keep working
    // without a header; LAN clients must present `Authorization:
    // Bearer <key>`. Empty key keeps the legacy open-access path.
    // Install the CPU thread pool early — rayon's global pool can
    // only be built once per process, and any later attempt is a
    // no-op. `0` (default) lets rayon's heuristic pick.
    //
    // When any logical processor is disabled (config `disabled_cpus` +/or
    // `RUSTLLAMA_DISABLED_CPUS`), build a pool sized to — and thread-
    // affinity-pinned onto — the ENABLED set so disabled cores run no
    // kernel work. With no disabled cores this is exactly the historical
    // path (no pinning), preserving current perf.
    {
        let disabled: std::collections::HashSet<u32> = cfg
            .inference
            .disabled_cpus
            .iter()
            .copied()
            .chain(rustllama_runtime::disabled_cpu_indices())
            .collect();
        if disabled.is_empty() {
            rustllama_kernels_cpu::install_thread_pool(cfg.inference.threads as usize);
        } else {
            let total = std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(1);
            let enabled: Vec<u32> = (0..total).filter(|i| !disabled.contains(i)).collect();
            let mut dl: Vec<u32> = disabled.iter().copied().collect();
            dl.sort_unstable();
            tracing::info!(
                disabled_cpus = ?dl,
                enabled_cores = enabled.len(),
                total_cores = total,
                "CPU affinity: pinning rayon workers to enabled logical processors"
            );
            rustllama_kernels_cpu::install_thread_pool_pinned(
                enabled,
                cfg.inference.threads as usize,
            );
        }
    }

    // Opportunistic-refine background task. Spawn before the server
    // starts taking requests so its first idle window doesn't miss
    // any. Idle threshold is 5 minutes (300s) per the plan's
    // "v1.1 — refine in idle time" description. No-op when the
    // config flag is off.
    if cfg.tuning.opportunistic_refine {
        let _ = rustllama_server::spawn_opportunistic_refine_task(state.clone(), 300);
        tracing::info!("opportunistic-refine task spawned (idle threshold = 5 min)");
    }

    let state = match rustllama_server::AuthState::new_with_rate_limit(
        &cfg.server.api_key,
        cfg.server.rate_limit_per_minute,
    ) {
        Some(auth) => {
            tracing::info!(
                rate_limit_per_minute = cfg.server.rate_limit_per_minute,
                require_auth_loopback = cfg.server.require_auth_loopback,
                "bearer-token auth enabled — {} still bypass",
                if cfg.server.require_auth_loopback {
                    "/healthz only"
                } else {
                    "loopback connections + /healthz"
                }
            );
            state.with_auth(auth.require_loopback_auth(cfg.server.require_auth_loopback))
        }
        None => state,
    };

    // Open the audit-log sink when [server].audit_log = true. Path
    // default is `<crash_log_dir>/audit.log.jsonl` so it lives
    // alongside crash dumps in the user-data layout (and gets
    // redirected the same way under portable mode). An open failure
    // is non-fatal — log a warning and keep serving without audit.
    let state = if cfg.server.audit_log {
        let resolved_path = if cfg.server.audit_log_path.is_empty() {
            rustllama_runtime::paths()
                .crash_log_dir
                .join("audit.log.jsonl")
        } else {
            std::path::PathBuf::from(&cfg.server.audit_log_path)
        };
        match rustllama_server::AuditSink::try_open(resolved_path.clone()) {
            Ok(sink) => {
                tracing::info!(path = %resolved_path.display(), "audit log enabled");
                state.with_audit_sink(sink)
            }
            Err(e) => {
                tracing::warn!(
                    path = %resolved_path.display(),
                    error = %e,
                    "failed to open audit log — continuing without"
                );
                state
            }
        }
    } else {
        state
    };

    let record = rustllama_runtime::ServerRecord {
        pid: std::process::id(),
        port,
        bind_addr: bind_addr.clone(),
        started_at: chrono_lite_now(),
        model_id: model_id_for_record,
        version: state.version.clone(),
        owner: "serve".to_string(),
    };
    rustllama_runtime::write_record(&record)?;

    // Compute + surface LAN URLs when the bind covers external
    // interfaces (0.0.0.0). Skip for localhost-only binds — the URL
    // list would just echo back the loopback and confuse users.
    if matches!(bind_addr.as_str(), "0.0.0.0" | "::") {
        let lan_urls = enumerate_lan_urls(port);
        if lan_urls.is_empty() {
            println!("listening on 0.0.0.0:{port} (no non-loopback IPv4 interfaces found)");
        } else {
            println!("rustllama listening on:");
            for url in &lan_urls {
                println!("  {url}");
            }
        }
        rustllama_server::set_lan_urls(lan_urls);
    } else {
        println!(
            "rustllama listening on http://{}:{port}",
            authority_host(&bind_addr)
        );
    }

    // When `serve` is invoked by the embedded GUI process (signalled
    // via the `RUSTLLAMA_GUI_EMBEDDED` env var that `run_gui` sets
    // before spawning us), and the user hasn't already configured
    // `[server].cors_origins`, default to wildcard so the webview's
    // `http://tauri.localhost` origin can read responses. Without
    // this every fetch from the GUI reports the server as "offline"
    // — the request reaches the server, but the browser blocks JS
    // from reading the response under same-origin-policy.
    //
    // Standalone `rustllama serve` keeps the existing empty default
    // (no CORS headers) so the server stays locked down by default.
    // Users running an external `rustllama serve` plus the GUI need
    // to set `[server].cors_origins = ["*"]` (or the specific Tauri
    // origin) in their config.
    let mut cors_origins = cfg.server.cors_origins.clone();
    let embedded = std::env::var("RUSTLLAMA_GUI_EMBEDDED").is_ok();
    if embedded && cors_origins.is_empty() {
        tracing::info!("embedded-GUI mode detected; defaulting cors_origins = [\"*\"]");
        cors_origins = vec!["*".to_string()];
    }
    // Always allow the Tauri webview origins so a separately-launched
    // `rustllama gui` can talk to a pre-existing `rustllama serve`
    // without the operator having to set CORS in config. The Tauri
    // 2 webview's `fetch()` carries `Origin: http://tauri.localhost`
    // on Windows (and the https variant on some platforms); adding
    // those is a no-op for non-GUI clients and doesn't expose the
    // server to web browsers (the schemes aren't reachable from
    // arbitrary pages).
    if !cors_origins.iter().any(|o| o == "*") {
        for origin in ["http://tauri.localhost", "https://tauri.localhost"] {
            if !cors_origins.iter().any(|o| o == origin) {
                cors_origins.push(origin.to_string());
            }
        }
    }
    let result = rustllama_server::run_with_cors(state, addr, &cors_origins).await;
    let _ = rustllama_runtime::remove_record();
    result?;
    Ok(())
}

/// Format a host for use in a URL / socket authority: bracket a bare
/// IPv6 literal (`::1` → `[::1]`), pass IPv4 / hostnames / already-
/// bracketed forms through unchanged.
fn authority_host(host: &str) -> String {
    if host.contains(':') && !host.starts_with('[') {
        format!("[{host}]")
    } else {
        host.to_string()
    }
}

/// Resolve the effective `chat` server URL from the mutually-layered
/// `--base-url` / `--ip` / `--port` flags:
///   - `--base-url` wins outright when present.
///   - otherwise, if either `--ip` or `--port` is given, build
///     `http://ip:port`, defaulting the missing half from the live
///     server record (else `[server]` config).
///   - if none are given, return `None` so the caller falls through to
///     the usual `resolve_base_url` discovery.
fn resolve_chat_base_url(
    config_path: &std::path::Path,
    base_url: Option<String>,
    ip: Option<String>,
    port: Option<u16>,
) -> Option<String> {
    if base_url.is_some() {
        return base_url;
    }
    if ip.is_none() && port.is_none() {
        return None;
    }
    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let (rec_host, rec_port) = match rustllama_runtime::read_alive_record() {
        Ok(Some(rec)) => (rec.bind_addr, rec.port),
        _ => (cfg.server.bind_addr.clone(), cfg.server.port),
    };
    // A server bound to any-interface isn't a usable client target;
    // dial loopback instead.
    let host = ip.unwrap_or_else(|| match rec_host.as_str() {
        "0.0.0.0" | "::" => "127.0.0.1".to_string(),
        _ => rec_host,
    });
    let port = port.unwrap_or(rec_port);
    Some(format!("http://{}:{}", authority_host(&host), port))
}

/// Enumerate non-loopback IPv4 addresses we can advertise as LAN URLs.
/// Uses tokio's `lookup_host`-friendly UDP-bind trick to find the
/// default outbound address, plus a scan of the interfaces table via
/// `sysinfo`. Returns a deduplicated, stable-ordered list.
fn enumerate_lan_urls(port: u16) -> Vec<String> {
    use std::net::UdpSocket;
    let mut out: Vec<String> = Vec::new();

    // Trick: connecting a UDP socket to a public-ish address picks the
    // route's source IP without sending any packets. Most users have
    // exactly one outbound interface; surfacing that one is the 99%
    // case.
    if let Ok(sock) = UdpSocket::bind("0.0.0.0:0") {
        // 8.8.8.8 is just a routing hint; no packets are sent.
        if sock.connect("8.8.8.8:53").is_ok() {
            if let Ok(local) = sock.local_addr() {
                if let std::net::IpAddr::V4(v4) = local.ip() {
                    if !v4.is_loopback() && !v4.is_unspecified() {
                        out.push(format!("http://{v4}:{port}"));
                    }
                }
            }
        }
    }

    // Always include localhost so the user can copy-paste even on a
    // machine with no LAN.
    out.push(format!("http://127.0.0.1:{port}"));

    // De-dupe while preserving insertion order (route-source IP first).
    let mut seen = std::collections::HashSet::new();
    out.retain(|u| seen.insert(u.clone()));
    out
}

fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("epoch-{secs}")
}

// ----- swap -----

/// Classify a `--target` argument as either a HuggingFace ref or a
/// local filesystem path. Hub refs contain `/` AND `:`; absolute paths
/// don't. (`C:\foo\bar.gguf` has a `:` but no `/`, so this still works
/// on Windows.)
fn classify_swap_target(target: &str) -> SwapTarget {
    let looks_like_hub = target.contains('/')
        && target.contains(':')
        && !target.starts_with(['/', '\\'])
        && !looks_like_windows_path(target);
    if looks_like_hub {
        SwapTarget::Hub(target.to_string())
    } else {
        SwapTarget::Path(std::path::PathBuf::from(target))
    }
}

fn looks_like_windows_path(s: &str) -> bool {
    let mut chars = s.chars();
    let (Some(c0), Some(c1)) = (chars.next(), chars.next()) else {
        return false;
    };
    c0.is_ascii_alphabetic() && c1 == ':'
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum SwapTarget {
    Hub(String),
    Path(std::path::PathBuf),
}

fn resolve_base_url(cfg: &rustllama_config::Config, override_url: Option<&str>) -> String {
    if let Some(u) = override_url {
        return u.to_string();
    }
    if let Ok(Some(rec)) = rustllama_runtime::read_alive_record() {
        return format!("http://{}:{}", rec.bind_addr, rec.port);
    }
    format!("http://{}:{}", cfg.server.bind_addr, cfg.server.port)
}

async fn swap_model(
    config_path: &std::path::Path,
    target: &str,
    base_url: Option<&str>,
    ctx_size: Option<usize>,
    kv_dtype: Option<&str>,
    no_default: bool,
) -> anyhow::Result<()> {
    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let server_url = resolve_base_url(&cfg, base_url);
    let client = rustllama_client::Client::new(server_url.as_str())?;

    // Make sure a server is up before we send anything.
    if let Err(e) = client.healthz().await {
        anyhow::bail!(
            "no rustllama server reachable at {server_url}: {e}\n\
             run `rustllama serve` in another terminal first."
        );
    }

    let mut params = rustllama_client::LoadModelParams {
        ctx_size,
        kv_dtype: kv_dtype.map(|s| s.to_string()),
        ..rustllama_client::LoadModelParams::default()
    };
    match classify_swap_target(target) {
        SwapTarget::Hub(h) => params.hub = Some(h),
        SwapTarget::Path(p) => params.path = Some(p),
    }

    println!("loading model on {server_url}...");
    let load_resp = client.load_model(&params).await?;
    let model_id = load_resp
        .get("model_id")
        .and_then(|s| s.as_str())
        .unwrap_or("?")
        .to_string();
    let prev = load_resp.get("previous_model_id").and_then(|s| s.as_str());
    if let Some(prev) = prev {
        println!("loaded `{model_id}` (replaced `{prev}`)");
    } else {
        println!("loaded `{model_id}`");
    }

    if !no_default {
        client.set_default_model(&model_id).await?;
        println!("set `{model_id}` as default");
    }
    Ok(())
}

/// `model unload <id>` — free a loaded model's RAM/VRAM on the running
/// server. Refuses to unload the current default model (promote another
/// with `model default <id>` first), so the server always keeps a
/// resolvable default while other models come and go.
async fn unload_model(
    config_path: &std::path::Path,
    id: &str,
    base_url: Option<&str>,
) -> anyhow::Result<()> {
    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let server_url = resolve_base_url(&cfg, base_url);
    let client = rustllama_client::Client::new(server_url.as_str())?;

    // Fail fast if no server is reachable, and learn the current default
    // in the same probe (`/healthz` reports the default model's id).
    let default_id = match client.healthz().await {
        Ok(v) => v
            .get("model_id")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string(),
        Err(e) => anyhow::bail!(
            "no rustllama server reachable at {server_url}: {e}\n\
             run `rustllama serve` in another terminal first."
        ),
    };
    if id == default_id {
        anyhow::bail!(
            "`{id}` is the current default model and can't be unloaded; \
             promote another first with `rustllama model default <id>`"
        );
    }

    client.unload_model(id).await?;
    println!("unloaded `{id}` from {server_url}");
    Ok(())
}

// ----- pull -----

async fn pull(hub_ref_str: &str) -> anyhow::Result<()> {
    use indicatif::{ProgressBar, ProgressStyle};
    use std::time::Duration;

    let hub_ref = rustllama_hub::HubRef::parse(hub_ref_str)?;
    let cache_dir = rustllama_hub::default_cache_dir()
        .ok_or_else(|| anyhow::anyhow!("could not resolve cache dir"))?;
    std::fs::create_dir_all(&cache_dir)?;

    let pb = ProgressBar::new_spinner();
    pb.enable_steady_tick(Duration::from_millis(120));
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner} {msg}")
            .unwrap_or_else(|_| ProgressStyle::default_spinner()),
    );
    pb.set_message(format!("downloading {}", hub_ref.filename));

    let dst = rustllama_hub::download(&hub_ref, &cache_dir, Some(&pb)).await?;
    pb.finish_and_clear();
    println!("✓ {}", dst.display());

    // Companion sidecars that live in the same repo: the `<stem>.kvbias.gguf`
    // KV-calibration file (its presence next to the GGUF is exactly what unlocks
    // a quantized KV cache for this model — see `coherence_safe_kv_dtype`) and an
    // mmproj vision projector for multimodal models. Fetch them too so a pulled
    // model lands complete; the single GGUF silently loses quant-KV coherence +
    // vision. Best-effort: listing the repo or any one companion failing never
    // fails the pull (the GGUF is already down).
    match rustllama_hub::hf_repo_files(&hub_ref.repo_id()).await {
        Ok(files) => {
            for comp in rustllama_hub::companion_sidecars(&hub_ref.filename, &files) {
                let cpb = ProgressBar::new_spinner();
                cpb.enable_steady_tick(Duration::from_millis(120));
                cpb.set_style(
                    ProgressStyle::default_spinner()
                        .template("{spinner} {msg}")
                        .unwrap_or_else(|_| ProgressStyle::default_spinner()),
                );
                cpb.set_message(format!("downloading companion {comp}"));
                let comp_ref = rustllama_hub::HubRef {
                    owner: hub_ref.owner.clone(),
                    repo: hub_ref.repo.clone(),
                    filename: comp.clone(),
                };
                match rustllama_hub::download(&comp_ref, &cache_dir, Some(&cpb)).await {
                    Ok(p) => {
                        cpb.finish_and_clear();
                        println!("✓ companion → {}", p.display());
                    }
                    Err(e) => {
                        cpb.finish_and_clear();
                        tracing::warn!(error = %e, companion = %comp, "companion sidecar download failed (continuing)");
                    }
                }
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "could not list repo files for companion sidecars (continuing)");
        }
    }

    // Fetch the HF model card README.md so `models inspect` / `/api/show`
    // can surface license, tags, base_model, etc. Best-effort: a missing
    // README, network hiccup, or 404 doesn't fail the pull — the GGUF is
    // already there.
    match rustllama_hub::model_card::fetch(&hub_ref).await {
        Ok(Some(card)) => match rustllama_hub::model_card::save(&card, &hub_ref, &cache_dir) {
            Ok(p) => println!("✓ model card → {}", p.display()),
            Err(e) => tracing::warn!(error = %e, "model card save failed"),
        },
        Ok(None) => tracing::info!("no README.md at HuggingFace for {hub_ref_str}"),
        Err(e) => tracing::warn!(error = %e, "model card fetch failed (continuing)"),
    }

    Ok(())
}

/// Synthetic prefill + decode benchmark. Loads the configured model
/// once, runs `repeats` rounds of (prefill of `prompt_tokens` tokens
/// + decode of `decode_tokens` tokens), and reports min / median /
/// max for each stage. Useful for empirically checking whether perf
/// knobs (`flash_attention`, `kv_dtype`) help on the local hardware.
///
/// `flash` / `kv_dtype` / `ctx_size` override the matching config
/// `rustllama tune` — autotuner CLI driver. Sweeps the candidate
/// local work-group sizes for the Q4_K packed USM matvec at each
/// distinct `(M, K)` shape the model uses, picks the winner per
/// shape, and writes the result to
/// `%LOCALAPPDATA%\rustllama\tuning\<device_fingerprint>.toml`.
/// The engine consumes this cache on next load via
/// [`rustllama_models::accel::tuned_q4k_lws`].
///
/// Defaults:
///   - `--device` = `sycl:0`
///   - `--model` = `[model].path` from the active config
///   - `--thorough` extends each sweep from the quick-default 3 timed
///     runs → 21 and disables the early-stop heuristic, slower but with
///     tighter confidence intervals. Without it (the default, incl. the
///     mandatory first-load autotune) the kernel LWS sweep runs the
///     quick 1-warmup/3-timed cadence.
///   - `--clear` deletes any existing cache for this device before
///     sweeping; without it, existing entries for shapes NOT in this
///     model are preserved (you can tune two models for the same
///     device into one cache file).
///
/// The SYCL runtime DLL search paths are set up in-process at CLI
/// startup (`rustllama_runtime::ensure_gpu_dll_search_paths`), so this
/// runs from a bare shell — no `run-sycl.bat` needed.
fn cmd_tune(
    config_path: &std::path::Path,
    device_spec: Option<String>,
    model_override: Option<String>,
    thorough: bool,
    clear: bool,
    measure_tok_s: bool,
    measure_tok_s_tokens: u32,
    measure_tok_s_repeats: u32,
) -> anyhow::Result<()> {
    use rustllama_gguf::{GgmlType, Gguf, TensorInfo};
    use rustllama_kernels_sycl as sk;
    use rustllama_tuner::{
        cache_path_for, default_cache_dir, fingerprint_device, load_cache, save_cache,
        set_tuned_lws_for, sweep, SweepConfig, TuningResult, KERNEL_PTQ1_0_PACKED_USM,
        KERNEL_Q4K_PACKED_USM, KERNEL_Q5K_PACKED_USM, KERNEL_Q6K_PACKED_USM,
        KERNEL_Q8_0_PACKED_USM, PACKED_USM_LWS_CANDIDATES,
    };

    // Describe one tunable packed-quant single-row matvec kernel.
    // Adding a new packed quant = add an entry here + match arm in
    // the `dispatch` field. The closure-based dispatch lets us reuse
    // one sweep loop for every kernel without per-kernel duplication.
    struct PackedKernel {
        kernel: &'static str,
        ggml_type: GgmlType,
        /// Bytes per output row for a given K. Validates the GGUF
        /// tensor size + sets the USM allocation size.
        row_bytes: fn(usize) -> usize,
        /// K alignment requirement (32 for Q8_0 single-block; 256
        /// for the K-quants' super-block size). Tensors whose K
        /// doesn't divide are skipped.
        k_align: usize,
    }

    // SAFETY for the dispatch closures below: all USM buffers
    // (w_buf, x_buf, out_buf) live for the duration of the closure
    // (enclosing scope), the stream outlives every call, and each
    // kernel `.wait()`s before returning.
    let kernels: &[PackedKernel] = &[
        PackedKernel {
            kernel: KERNEL_Q4K_PACKED_USM,
            ggml_type: GgmlType::Q4_K,
            row_bytes: |k| (k / 256) * 144,
            k_align: 256,
        },
        PackedKernel {
            kernel: KERNEL_PTQ1_0_PACKED_USM,
            ggml_type: GgmlType::PTQ1_0,
            row_bytes: |k| (k / 128) * 28,
            k_align: 128,
        },
        PackedKernel {
            kernel: KERNEL_Q5K_PACKED_USM,
            ggml_type: GgmlType::Q5_K,
            row_bytes: |k| (k / 256) * 176,
            k_align: 256,
        },
        PackedKernel {
            kernel: KERNEL_Q6K_PACKED_USM,
            ggml_type: GgmlType::Q6_K,
            row_bytes: |k| (k / 256) * 210,
            k_align: 256,
        },
        PackedKernel {
            kernel: KERNEL_Q8_0_PACKED_USM,
            ggml_type: GgmlType::Q8_0,
            row_bytes: |k| (k / 32) * 34,
            k_align: 32,
        },
    ];

    // 1. Resolve device + model.
    let device_idx: u32 = match device_spec.as_deref() {
        None => 0,
        Some(s) => {
            let trimmed = s.strip_prefix("sycl:").unwrap_or(s);
            trimmed
                .parse::<u32>()
                .map_err(|e| anyhow::anyhow!("invalid --device {s:?}: {e}"))?
        }
    };
    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let model_path = effective_model_path(model_override, &cfg, config_path)?;

    // An MLX model is a DIRECTORY of affine-quantized safetensors with no
    // GGUF tensor table, and here it runs on CPU (load_auto → load_mlx; no GPU
    // kernel dispatch), so this per-shape SYCL kernel-LWS sweep has nothing to
    // tune. Detect the directory and skip cleanly — returning Ok so the
    // `tune --all` orchestrator treats Stage 1 as a no-op rather than a failure
    // (opening a directory as a GGUF file would otherwise error at `Gguf::open`
    // below). Bails before even opening a SYCL device, so it is correct on a
    // CPU-only host too.
    if model_path.is_dir() {
        println!("  model       = {}", model_path.display());
        println!(
            "MLX / non-GGUF model directory — skipping GPU kernel-LWS sweep \
             (no packed-quant GGUF tensors; MLX runs on CPU)."
        );
        return Ok(());
    }

    // The kernel-LWS sweep tunes SYCL local-work-group sizes for the packed
    // matvec — it is SYCL-ONLY. On a host with no usable SYCL device (an
    // NVIDIA-only box / the DGX Spark, where SYCL is a no-op stub), opening
    // sycl:0 fails with "device index out of range" — not an error, the sweep
    // simply doesn't apply (CUDA/CPU use fixed launch geometry). Skip cleanly so
    // `tune --all` doesn't log a scary "stage 1 (kernel LWS) failed".
    if rustllama_kernels_sycl::device_count().unwrap_or(0) == 0 {
        println!(
            "  kernel LWS sweep: no SYCL device present — skipping \
             (SYCL-only; CUDA/CPU use fixed launch geometry)."
        );
        return Ok(());
    }

    // 2. Open SYCL device + fingerprint.
    let stream = sk::create_stream(device_idx)
        .map_err(|e| anyhow::anyhow!("create_stream(sycl:{device_idx}): {e}"))?;
    let fp = fingerprint_device(device_idx).ok_or_else(|| {
        anyhow::anyhow!(
            "could not fingerprint device sycl:{device_idx} — is an Intel GPU \
             present with its Level Zero / OpenCL runtime installed?"
        )
    })?;

    let cache_dir = default_cache_dir()
        .ok_or_else(|| anyhow::anyhow!("could not resolve default tuner cache dir"))?;
    std::fs::create_dir_all(&cache_dir)?;
    // The cache FILE is keyed by the whole-system fingerprint (works on
    // SYCL / CUDA / CPU hosts). `fp` remains the SYCL device we swept the
    // kernel LWS on — displayed below + stamped into the result's `device`.
    let key = rustllama_tuner::system_fingerprint();
    let cache_path = cache_path_for(&cache_dir, &key);

    println!("rustllama tune");
    println!("  device      = sycl:{device_idx} ({})", fp.name);
    println!("  driver      = {}", fp.driver_ver);
    println!("  fingerprint = {}", fp.slug());
    println!("  model       = {}", model_path.display());
    println!(
        "  mode        = {}",
        if thorough { "thorough" } else { "quick" }
    );
    println!("  cache path  = {}", cache_path.display());
    println!();

    if clear && cache_path.exists() {
        std::fs::remove_file(&cache_path)?;
        println!("cleared existing cache");
    }
    let mut tuning = load_cache(&cache_dir, &key)?
        .unwrap_or_else(|| TuningResult::empty(key.clone(), fp.clone()));

    // 3. Enumerate distinct shapes per kernel, dedup across the
    //    typical Q/O / K/V / gate-up / down sets.
    let gguf = Gguf::open(&model_path)?;
    type ShapeMap<'a> = std::collections::BTreeMap<(usize, usize), Vec<&'a TensorInfo>>;
    let mut per_kernel_shapes: std::collections::BTreeMap<&'static str, ShapeMap<'_>> =
        std::collections::BTreeMap::new();
    for k_def in kernels {
        let mut sm: ShapeMap<'_> = std::collections::BTreeMap::new();
        for info in gguf.tensors() {
            if info.dtype != k_def.ggml_type || info.dims.len() < 2 {
                continue;
            }
            let m = info.dims[0] as usize;
            let k = info.dims[1] as usize;
            if m == 0 || k == 0 || k % k_def.k_align != 0 {
                continue;
            }
            sm.entry((m, k)).or_default().push(info);
        }
        if !sm.is_empty() {
            per_kernel_shapes.insert(k_def.kernel, sm);
        }
    }

    if per_kernel_shapes.is_empty() {
        println!(
            "no tunable packed-quant tensors in {} — nothing to tune",
            model_path.display()
        );
        return Ok(());
    }

    let total_shapes: usize = per_kernel_shapes.values().map(|m| m.len()).sum();
    let total_tensors: usize = per_kernel_shapes
        .values()
        .flat_map(|m| m.values().map(|v| v.len()))
        .sum();
    println!(
        "found {} distinct shape(s) across {} tensor(s), spanning {} kernel(s):",
        total_shapes,
        total_tensors,
        per_kernel_shapes.len()
    );
    for (kname, shapes) in &per_kernel_shapes {
        println!("  [{kname}]");
        for (&(m, k), tensors) in shapes {
            let n = tensors.len();
            println!(
                "    M={m:5} K={k:5}  ({} tensor{})",
                n,
                if n == 1 { "" } else { "s" }
            );
        }
    }
    println!();

    let sweep_cfg = if thorough {
        SweepConfig {
            warmup_runs: 3,
            timed_runs: 21,
            early_stop_ratio: 0.0,
        }
    } else {
        SweepConfig::default()
    };

    let pb = indicatif::ProgressBar::new(total_shapes as u64);
    pb.set_style(
        indicatif::ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=> "),
    );

    let mut winners_total: usize = 0;
    let mut failed_total: usize = 0;
    // Per-shape (baseline_median_us, tuned_median_us) records collected
    // across every kernel. After the sweep we report the aggregate
    // speedup (sum_baseline / sum_tuned) — the plan's Phase 5 acceptance
    // gate is ≥1.5×. The geomean is reported alongside so a single
    // slow kernel doesn't dominate the headline number.
    //
    // Baseline = LWS=0 (driver default), the value the engine uses on
    // a cache miss. Tuned = the sweep's winning LWS for the shape.
    let mut speedup_records: Vec<(f64, f64)> = Vec::new();

    // 4. Per-kernel sweep loop. Each kernel reuses one sweep loop
    //    with a different FFI dispatch closure.
    for k_def in kernels {
        let shapes = match per_kernel_shapes.get(k_def.kernel) {
            Some(s) => s,
            None => continue,
        };
        pb.println(format!("== {} ==", k_def.kernel));
        for (&(m, k), tensors) in shapes {
            pb.set_message(format!("[{}] M={m} K={k}", k_def.kernel));
            let bytes_needed = (k_def.row_bytes)(k) * m;
            let rep = tensors[0];
            let mut w_buf: sk::SyclSharedBuffer<u8> =
                sk::SyclSharedBuffer::alloc(&stream, bytes_needed)?;
            match gguf.tensor_bytes(&rep.name) {
                Some(src) => {
                    let n = src.len().min(bytes_needed);
                    w_buf.as_mut_slice()[..n].copy_from_slice(&src[..n]);
                    if n < bytes_needed {
                        for b in w_buf.as_mut_slice()[n..].iter_mut() {
                            *b = 0;
                        }
                    }
                }
                None => {
                    for (i, b) in w_buf.as_mut_slice().iter_mut().enumerate() {
                        *b = ((i * 31 + 7) & 0xff) as u8;
                    }
                }
            }
            let mut x_buf: sk::SyclSharedBuffer<f32> = sk::SyclSharedBuffer::alloc(&stream, k)?;
            for (i, v) in x_buf.as_mut_slice().iter_mut().enumerate() {
                *v = (((i % 17) as f32) - 8.0) * 0.05;
            }
            let mut out_buf: sk::SyclSharedBuffer<f32> = sk::SyclSharedBuffer::alloc(&stream, m)?;

            let m_u32 = m as u32;
            let k_u32 = k as u32;
            let w_ptr = w_buf.as_ptr();
            let x_ptr = x_buf.as_ptr();
            let out_ptr = out_buf.as_mut_ptr();
            let kname = k_def.kernel;

            let result = sweep(PACKED_USM_LWS_CANDIDATES, sweep_cfg, |lws| {
                let r = unsafe {
                    match kname {
                        n if n == KERNEL_Q4K_PACKED_USM => sk::matvec_q4_k_packed_f32_usm_raw(
                            &stream, w_ptr, x_ptr, out_ptr, m_u32, k_u32, lws,
                        ),
                        n if n == KERNEL_PTQ1_0_PACKED_USM => sk::matvec_ptq1_0_packed_f32_usm_raw(
                            &stream, w_ptr, x_ptr, out_ptr, m_u32, k_u32, lws,
                        ),
                        n if n == KERNEL_Q5K_PACKED_USM => sk::matvec_q5_k_packed_f32_usm_raw(
                            &stream, w_ptr, x_ptr, out_ptr, m_u32, k_u32, lws,
                        ),
                        n if n == KERNEL_Q6K_PACKED_USM => sk::matvec_q6_k_packed_f32_usm_raw(
                            &stream, w_ptr, x_ptr, out_ptr, m_u32, k_u32, lws,
                        ),
                        n if n == KERNEL_Q8_0_PACKED_USM => sk::matvec_q8_0_packed_f32_usm_raw(
                            &stream, w_ptr, x_ptr, out_ptr, m_u32, k_u32, lws,
                        ),
                        other => Err(sk::SyclError::InvalidShape(format!(
                            "unknown packed-quant kernel: {other}"
                        ))),
                    }
                };
                r.map_err(|e| e.to_string())
            });

            match result.winner_pair() {
                Some((lws, median)) => {
                    set_tuned_lws_for(&mut tuning, k_def.kernel, m, k, lws);
                    winners_total += 1;
                    // Baseline measurement at LWS=0 (driver default,
                    // engine fallback on cache miss). One extra
                    // sweep call against a single-element candidate
                    // list reuses the same warmup/median harness.
                    let baseline = sweep(&[0u32], sweep_cfg, |lws| {
                        let r = unsafe {
                            match kname {
                                n if n == KERNEL_Q4K_PACKED_USM => {
                                    sk::matvec_q4_k_packed_f32_usm_raw(
                                        &stream, w_ptr, x_ptr, out_ptr, m_u32, k_u32, lws,
                                    )
                                }
                                n if n == KERNEL_PTQ1_0_PACKED_USM => {
                                    sk::matvec_ptq1_0_packed_f32_usm_raw(
                                        &stream, w_ptr, x_ptr, out_ptr, m_u32, k_u32, lws,
                                    )
                                }
                                n if n == KERNEL_Q5K_PACKED_USM => {
                                    sk::matvec_q5_k_packed_f32_usm_raw(
                                        &stream, w_ptr, x_ptr, out_ptr, m_u32, k_u32, lws,
                                    )
                                }
                                n if n == KERNEL_Q6K_PACKED_USM => {
                                    sk::matvec_q6_k_packed_f32_usm_raw(
                                        &stream, w_ptr, x_ptr, out_ptr, m_u32, k_u32, lws,
                                    )
                                }
                                n if n == KERNEL_Q8_0_PACKED_USM => {
                                    sk::matvec_q8_0_packed_f32_usm_raw(
                                        &stream, w_ptr, x_ptr, out_ptr, m_u32, k_u32, lws,
                                    )
                                }
                                other => Err(sk::SyclError::InvalidShape(format!(
                                    "unknown packed-quant kernel: {other}"
                                ))),
                            }
                        };
                        r.map_err(|e| e.to_string())
                    });
                    let baseline_us = baseline.winner_pair().map(|(_, m)| m.as_secs_f64() * 1e6);
                    let tuned_us = median.as_secs_f64() * 1e6;
                    if let Some(b) = baseline_us {
                        speedup_records.push((b, tuned_us));
                    }
                    let mut detail = String::new();
                    for c in &result.results {
                        if c.ok {
                            detail.push_str(&format!(
                                "  {}={:.1}us",
                                c.value,
                                c.median.as_secs_f64() * 1e6
                            ));
                        } else {
                            detail.push_str(&format!("  {}=FAIL", c.value));
                        }
                    }
                    let speedup_tag = match baseline_us {
                        Some(b) if tuned_us > 0.0 => {
                            format!("  (baseline LWS=0 {:.1} us → {:.2}x)", b, b / tuned_us)
                        }
                        _ => String::new(),
                    };
                    pb.println(format!(
                        "  M={m:5} K={k:5}  →  LWS={lws:3}  (median {:.1} us){detail}{speedup_tag}",
                        median.as_secs_f64() * 1e6,
                    ));
                }
                None => {
                    failed_total += 1;
                    pb.println(format!(
                        "  M={m:5} K={k:5}  →  all candidates failed; cache entry skipped"
                    ));
                }
            }
            pb.inc(1);
        }
    }
    pb.finish_with_message("done");

    tuning.last_tuned = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| format!("{}", d.as_secs()));
    save_cache(&cache_dir, &tuning)?;

    println!();
    println!("wrote cache to {}", cache_path.display());
    println!(
        "tuned {} shape(s); {} failed; engine picks up the cache on next load",
        winners_total, failed_total
    );

    // Phase 5 acceptance gate: aggregate per-shape speedup vs the
    // engine's cache-miss fallback (LWS=0). The plan target is
    // ≥1.5×; below that we WARN so the user knows the cache may
    // not be worth keeping on this device. Above, we PASS. Either
    // way the cache is written — the gate is informational, not
    // a hard reject (a user re-running on a different shape mix
    // may still benefit even if the synthetic-shape geomean is
    // below threshold).
    if !speedup_records.is_empty() {
        let sum_baseline: f64 = speedup_records.iter().map(|(b, _)| b).sum();
        let sum_tuned: f64 = speedup_records.iter().map(|(_, t)| t).sum();
        let aggregate = if sum_tuned > 0.0 {
            sum_baseline / sum_tuned
        } else {
            0.0
        };
        // Geometric mean of per-shape speedups. Bias-robust against a
        // single huge shape dominating the headline number.
        let geomean = {
            let log_sum: f64 = speedup_records
                .iter()
                .filter(|(_, t)| *t > 0.0)
                .map(|(b, t)| (b / t).ln())
                .sum();
            (log_sum / speedup_records.len() as f64).exp()
        };
        println!();
        println!("acceptance gate (Phase 5):");
        println!(
            "  aggregate speedup: {:.2}x  (sum baseline {:.0} us / sum tuned {:.0} us)",
            aggregate, sum_baseline, sum_tuned
        );
        println!(
            "  per-shape geomean: {:.2}x  (across {} shapes)",
            geomean,
            speedup_records.len()
        );
        if aggregate >= 1.5 {
            println!("  PASS  ≥1.5x target met — the cache will be a real win at runtime.");
        } else if aggregate >= 1.1 {
            println!(
                "  MARGINAL  below 1.5x target. Cache still written; engine will use it, \
                 but the win at runtime will be modest."
            );
        } else {
            println!(
                "  WEAK  near-baseline speedup. The cached LWS is barely better than the \
                 driver default on this device — consider re-running with `--thorough` if \
                 you didn't, or accept that this device's quant matvec is already near-optimal."
            );
        }
    } else {
        println!();
        println!("acceptance gate (Phase 5): no baseline records collected — skipped");
    }

    // End-to-end tok/s probe. Optional (`--measure-tok-s`) so callers
    // who just want the kernel-LWS sweep + cache write don't pay the
    // model-load + generation cost. When enabled: load the model
    // once with the freshly-written cache in place, run one warmup
    // + `--measure-tok-s-repeats` timed generations of
    // `--measure-tok-s-tokens` each, and report the median tok/s.
    // This is the plan's "end-to-end win metric" — the per-kernel
    // µs aggregate above is a proxy that doesn't reflect data-
    // movement / kernel-launch / sampling overhead.
    if measure_tok_s {
        println!();
        println!("end-to-end tok/s probe (with tuned cache):");
        match measure_end_to_end_tok_s(&model_path, measure_tok_s_tokens, measure_tok_s_repeats) {
            Ok(median) => println!("  median decode: {median:.2} tok/s"),
            Err(e) => println!("  probe failed: {e}"),
        }
    }

    Ok(())
}

/// CPU threads sweep. Loads the model once per candidate (rayon's
/// global pool can only be built per-process, so we exec each
/// candidate in a fresh child via re-invoking the bench loop within
/// this process — sized via `RAYON_NUM_THREADS` env var as a proxy
/// for `install_thread_pool` whose effect is one-shot).
///
/// Wall-time cost: `candidates × (model_load + repeats × decode)`.
/// For a 7B Q4_K_M with 6 candidates × 3 repeats × 32 decode tokens
/// that's a few minutes — well above the kernel-LWS sweep budget,
/// in line with the placement sweep.
fn cmd_tune_threads(
    config_path: &std::path::Path,
    model_override: Option<String>,
    candidates_csv: &str,
    decode_tokens: u32,
    repeats: u32,
) -> anyhow::Result<()> {
    use rustllama_engine::measurement::{measure_placement_candidates, MeasurementConfig};
    use rustllama_engine::KvDtype;
    use rustllama_tuner::PlacementPlan;

    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let model_path = effective_model_path(model_override, &cfg, config_path)?;

    // Derive candidate thread counts. Empty CSV → plan grid:
    //   {1, 2, 4, ncores/2, ncores, 2*ncores}
    // ncores from std::thread::available_parallelism (physical
    // cores when the OS exposes them; logical otherwise).
    let candidates: Vec<usize> = if candidates_csv.trim().is_empty() {
        let ncores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let mut grid = vec![1, 2, 4, (ncores / 2).max(1), ncores, ncores * 2];
        grid.sort_unstable();
        grid.dedup();
        grid
    } else {
        candidates_csv
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.parse::<usize>()
                    .map_err(|e| anyhow::anyhow!("invalid thread candidate {s:?}: {e}"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?
    };
    if candidates.is_empty() {
        anyhow::bail!("--threads-candidates produced an empty list");
    }

    println!("rustllama tune --threads");
    println!("  model         = {}", model_path.display());
    println!(
        "  candidates    = {}",
        candidates
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("  decode_tokens = {decode_tokens}");
    println!("  repeats       = {repeats} (median per candidate)");
    println!();
    println!("note: rayon's global pool is built once per process; this command");
    println!("respects the FIRST candidate only. To benchmark multiple thread");
    println!("counts cleanly, re-invoke with RAYON_NUM_THREADS=<n> per run and");
    println!("compare. The sweep below loads the model once and records the");
    println!("baseline tok/s with rayon's default — useful as a perf reference.");
    println!();

    // Use measure_placement_candidates with a single placement as the
    // timing harness — one candidate, but the existing load + warmup +
    // median-of-runs machinery is identical to what we want.
    //
    // Measure the ALL-CPU placement (n_gpu_layers = 0), NOT all-GPU:
    // this stage installs the rayon CPU thread pool, so the thing whose
    // effect we are trying to observe (thread count) only moves the
    // needle on the CPU decode path. Measuring with everything on the
    // GPU would drown the CPU-thread signal AND route through the
    // per-thread `UsmAttnContext` — and on hybrid (attn+DeltaNet) models
    // the all-GPU + `contiguous`-layout combination crashes USM K/V
    // allocation with an access violation (0xC0000005), which aborts the
    // whole `tune --all` process (a hard fault, not a catchable Err) and
    // silently skips the later stages (decision calibration, MTP,
    // chunked SSM). All-CPU + `paged` mirrors the safe, representative
    // path the engine actually runs for a CPU-placed model.
    let cand_plans = [PlacementPlan {
        n_gpu_layers: 0,
        overrides: Vec::new(),
    }];
    let m_cfg = MeasurementConfig {
        kv_dtype: KvDtype::F32,
        kv_cache_layout: "paged".to_string(),
        flash_attention: true,
        n_gpu_layers: 0,
    };
    let prompt_tokens = 32u32;
    let ctx_size = (prompt_tokens as usize + decode_tokens as usize + 8).max(128);

    // Install the first candidate as the rayon pool size before any
    // measurement. The remaining candidates print as "would-need
    // separate invocation" — the rayon one-shot constraint is
    // explicit in the help text above.
    let first = candidates[0];
    rustllama_kernels_cpu::install_thread_pool(first);

    let report = measure_placement_candidates(
        &model_path,
        &cand_plans,
        ctx_size,
        prompt_tokens,
        decode_tokens,
        repeats,
        &m_cfg,
    )
    .map_err(|e| anyhow::anyhow!("measure failed: {e}"))?;
    let cand = report
        .candidates
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no candidate result"))?;
    match cand.median_tps {
        Some(median) => println!(
            "  threads={first:3}: median {median:.2} tok/s (warmup {:.0}ms)",
            cand.warmup_ms
        ),
        None => println!(
            "  threads={first:3}: measurement failed: {}",
            cand.error.unwrap_or_default()
        ),
    }
    for &c in &candidates[1..] {
        println!("  threads={c:3}: skipped (re-invoke with RAYON_NUM_THREADS={c} to measure)");
    }
    Ok(())
}

/// Helper for the `--measure-tok-s` probe. Loads the engine with the
/// configured context size, runs `repeats + 1` decodes (first is
/// warmup, remaining are timed), returns the median tok/s.
fn measure_end_to_end_tok_s(
    model_path: &std::path::Path,
    decode_tokens: u32,
    repeats: u32,
) -> anyhow::Result<f64> {
    use rustllama_engine::measurement::{measure_placement_candidates, MeasurementConfig};
    use rustllama_engine::KvDtype;
    use rustllama_tuner::PlacementPlan;

    // Single placement candidate: keep whatever `n_gpu_layers` the
    // engine's own cache lookup will pick. Passing `u32::MAX` means
    // "all layers on GPU when the device has capacity"; the engine
    // overrides via the tuner cache if a per-model placement winner
    // is stored.
    let candidates = [PlacementPlan {
        n_gpu_layers: u32::MAX,
        overrides: Vec::new(),
    }];
    // Synthetic prompt of 32 tokens — small enough that prefill
    // doesn't dominate the measurement.
    let prompt_tokens = 32u32;
    let ctx_size = (prompt_tokens as usize + decode_tokens as usize + 8).max(128);
    let cfg = MeasurementConfig {
        kv_dtype: KvDtype::F32,
        kv_cache_layout: "contiguous".to_string(),
        flash_attention: true,
        n_gpu_layers: u32::MAX,
    };
    let report = measure_placement_candidates(
        model_path,
        &candidates,
        ctx_size,
        prompt_tokens,
        decode_tokens,
        repeats,
        &cfg,
    )
    .map_err(|e| anyhow::anyhow!("measure_placement_candidates: {e}"))?;
    let cand = report
        .candidates
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no candidate result"))?;
    cand.median_tps
        .ok_or_else(|| anyhow::anyhow!("no median_tps (warmup/runs failed)"))
}

/// Auto-detect the VRAM (in MiB) of the GPU the engine would dispatch on,
/// mirroring the matvec-dispatch precedence (CUDA → SYCL). Returns `None` on a
/// CPU-only host. Used by the placement sweep when `--vram-mb 0` (the default)
/// so it budgets against the REAL card, not a fixed 4 GiB guess that silently
/// capped placement on any larger GPU.
fn detect_dispatch_gpu_vram_mb() -> Option<u64> {
    if rustllama_kernels_cuda::device_count() > 0 {
        if let Some(info) = rustllama_runtime::gpu_detect::detect_nvidia() {
            if let Some(g) = info.gpus.iter().max_by_key(|g| g.total_mem_bytes) {
                return Some(g.total_mem_bytes / (1024 * 1024));
            }
        }
    }
    if let Ok(info) = rustllama_kernels_sycl::device_info(0) {
        let mb = info.vram_mb();
        if mb > 0 {
            return Some(mb);
        }
    }
    None
}

/// True when the dispatch GPU has DEDICATED VRAM (discrete NVIDIA/SYCL card) or
/// is a high-bandwidth unified GPU (Apple Metal) — i.e. NOT a weak integrated
/// GPU sharing system LPDDR. On such a GPU the most-GPU-that-fits placement is
/// always fastest for single-stream decode, so the per-candidate decode
/// micro-benchmark (which only disambiguates the shared-LPDDR CPU-vs-iGPU case)
/// is skipped — it mis-picked 20 of 28 layers on an RTX 2000 Ada where all-GPU
/// is 1.5x faster. Mirrors `multi_gpu::host_has_dedicated_vram`.
fn dispatch_gpu_has_dedicated_vram() -> bool {
    rustllama_kernels_cuda::device_count() > 0
        || rustllama_kernels_mlx::device_count() > 0
        || rustllama_kernels_sycl::device_info(0)
            .map(|i| !i.is_integrated)
            .unwrap_or(false)
}

/// Static placement-sweep analyzer. Given the configured model, the
/// supplied VRAM budget, and the context window, enumerates the
/// `n_gpu_layers` candidates that fit and prints a per-candidate
/// VRAM-cost breakdown. The top row is the recommended setting —
/// the most-GPU candidate that fits the budget after subtracting
/// headroom.
///
/// This is the static half of the placement sweep: pure GGUF
/// metadata + arithmetic, no model load, no measurement. The
/// dynamic measurement half (actually time generation tok/s per
/// candidate) is the next-up follow-up; with measurement absent,
/// the recommendation defaults to "most GPU that fits", which is
/// the right answer on dedicated VRAM. On Iris Xe's shared LPDDR,
/// the right answer can be lower because GPU + CPU compete for
/// bandwidth — pin a smaller `--vram-mb` to reflect that.
#[allow(clippy::too_many_arguments)]
fn cmd_tune_placement(
    config_path: &std::path::Path,
    model_override: Option<String>,
    vram_mb: u64,
    vram_headroom_mb: u64,
    placement_ctx_override: Option<u32>,
    measure: bool,
    measure_prompt_tokens: u32,
    measure_decode_tokens: u32,
    measure_repeats: u32,
) -> anyhow::Result<()> {
    use rustllama_tuner::placement::{
        block_vram_bytes, candidate_placements, embedding_vram_bytes, kv_cache_vram_bytes,
        lm_head_vram_bytes, DEFAULT_SWEEP_STEP,
    };

    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let model_path = effective_model_path(model_override, &cfg, config_path)?;
    let max_ctx = placement_ctx_override.unwrap_or(cfg.inference.ctx_size);

    // MLX / non-GGUF model directory: there is no GGUF tensor table to price a
    // VRAM-fit table from, and an MLX affine checkpoint loads + runs on CPU
    // here (load_auto → load_mlx, which ignores n_gpu_layers), so the only
    // placement is all-CPU (n_gpu_layers = 0). Persist that CPU placement
    // winner keyed by the model stem — the exact key the first-load autotune
    // gate (`server::autotune::is_untuned`) checks — so `tune --all` on an MLX
    // dir records a real, applicable placement instead of erroring on
    // `read_dims_and_quant_from_gguf` (`Gguf::open`) below. No per-candidate
    // measurement is needed (there is only one placement), which also keeps
    // this robust even when the MLX load is slow/heavy.
    if model_path.is_dir() {
        println!("rustllama tune --placement");
        println!(
            "  model       = {} (MLX / non-GGUF directory → CPU placement)",
            model_path.display()
        );
        println!("  ctx_size    = {max_ctx}");
        println!();
        let model_key = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown-model")
            .to_string();
        // n_gpu_layers = 0 → all-CPU; MLX affine runs on CPU regardless, so
        // this is both correct and the value the engine honors on reload.
        match persist_placement_winner(&model_key, 0) {
            Ok(()) => {
                println!("  → placement: n_gpu_layers = 0 (CPU; MLX affine runs on CPU here)")
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to persist MLX placement winner to tuner cache")
            }
        }
        return Ok(());
    }

    let (dims, quant) = read_dims_and_quant_from_gguf(&model_path)?;

    // AUTO VRAM (`--vram-mb 0`, the default): budget against the real dispatch
    // GPU instead of the old fixed 4 GiB, which capped placement on any larger
    // card (only ~20/28 layers on a 16 GiB GPU → the rest stranded on CPU).
    let vram_mb = if vram_mb == 0 {
        match detect_dispatch_gpu_vram_mb() {
            Some(mb) => {
                println!("  (auto-detected dispatch GPU VRAM: {mb} MiB)");
                mb
            }
            None => {
                println!("  (no GPU detected → all-CPU placement; 4096 MiB nominal budget)");
                4096
            }
        }
    } else {
        vram_mb
    };

    let vram_bytes = vram_mb * 1024 * 1024;
    let headroom_bytes = vram_headroom_mb * 1024 * 1024;
    let candidates = candidate_placements(
        dims,
        quant,
        vram_bytes,
        headroom_bytes,
        max_ctx,
        DEFAULT_SWEEP_STEP,
    );

    println!("rustllama tune --placement");
    println!("  model       = {}", model_path.display());
    println!(
        "  dims        = {} layers, d_model={}, d_ff={}, heads={}/{} kv, head_dim={}, vocab={}",
        dims.n_layers,
        dims.d_model,
        dims.d_ff,
        dims.n_heads,
        dims.n_kv_heads,
        dims.head_dim,
        dims.vocab_size
    );
    println!("  quant       = {:?}", quant);
    println!(
        "  vram budget = {} MiB (with {} MiB headroom; effective {} MiB)",
        vram_mb,
        vram_headroom_mb,
        vram_mb.saturating_sub(vram_headroom_mb),
    );
    println!("  ctx_size    = {} (KV cost scales with this)", max_ctx);
    println!();

    let block_cost = block_vram_bytes(dims, quant);
    let head_cost = lm_head_vram_bytes(dims, quant);
    let embed_cost = embedding_vram_bytes(dims, quant);
    let kv_cost = kv_cache_vram_bytes(dims, max_ctx);
    let mib = |b: u64| (b as f64) / (1024.0 * 1024.0);
    println!("  per-component cost:");
    println!("    block        = {:>8.1} MiB", mib(block_cost));
    println!("    lm_head      = {:>8.1} MiB", mib(head_cost));
    println!("    embed_table  = {:>8.1} MiB", mib(embed_cost));
    println!(
        "    kv_cache@{:>5} = {:>8.1} MiB (always device-resident)",
        max_ctx,
        mib(kv_cost)
    );
    println!();

    if candidates.is_empty() {
        println!(
            "no candidate fits — even the all-CPU baseline (just the KV cache) exceeds \
             the budget. Raise --vram-mb, lower --placement-ctx, or accept that this \
             model + context combo won't fit your GPU."
        );
        return Ok(());
    }

    println!("candidates that fit (most-GPU first; top row is the recommendation):");
    println!(
        "  {:>4}  {:>10}  {:>10}  {:>10}  notes",
        "n_gpu", "weights", "total", "vs budget"
    );
    for (i, plan) in candidates.iter().enumerate() {
        let n = plan.n_gpu_layers;
        let blocks_on_gpu = n.min(dims.n_layers) as u64;
        let weight_cost = blocks_on_gpu * block_cost
            + if n > dims.n_layers {
                head_cost + embed_cost
            } else {
                0
            };
        let total = weight_cost + kv_cost;
        let budget = vram_bytes.saturating_sub(headroom_bytes);
        let pct = if budget > 0 {
            100.0 * (total as f64) / (budget as f64)
        } else {
            0.0
        };
        let notes = if i == 0 {
            "  ← recommended"
        } else if n == 0 {
            "  (full CPU fallback)"
        } else if n > dims.n_layers {
            "  (+ LM head + embeddings)"
        } else {
            ""
        };
        println!(
            "  {:>4}  {:>7.1} MiB  {:>7.1} MiB  {:>7.1}%   {}",
            n,
            mib(weight_cost),
            mib(total),
            pct,
            notes
        );
    }
    println!();
    // On a dedicated-VRAM GPU the most-GPU-that-fits candidate (row 0) is always
    // fastest for single-stream decode; the per-candidate decode sweep only
    // disambiguates the weak shared-LPDDR integrated-GPU case, and its short
    // micro-benchmark otherwise just adds noise (it mis-picked 20/28 on a
    // discrete RTX 2000 Ada). Skip it there and take the recommendation.
    let dedicated = dispatch_gpu_has_dedicated_vram();
    let measure = measure && !dedicated;
    if dedicated {
        println!(
            "  (dedicated-VRAM GPU: taking the most-GPU-that-fits candidate; \
             skipping the per-candidate decode sweep — it only helps weak \
             integrated GPUs)"
        );
    }
    // Dynamic measurement half: optionally run each candidate
    // through the real engine and pick the winner by measured tok/s.
    let (winner_n_gpu, measured_winner) = if measure {
        let cfg_for_measure = rustllama_config::load(config_path).unwrap_or_default();
        let winner = measure_placement_candidates(
            &model_path,
            &candidates,
            max_ctx as usize,
            measure_prompt_tokens,
            measure_decode_tokens,
            measure_repeats,
            &cfg_for_measure,
        )?;
        (winner.unwrap_or(candidates[0].n_gpu_layers), winner)
    } else {
        (candidates[0].n_gpu_layers, None)
    };

    // Persist the chosen placement to the tuner cache so a future consumer-side
    // hook — and the first-load autotune gate (`server::autotune::is_untuned`,
    // keyed on this same per-model stem) — can read it back without re-running
    // the sweep. Fires when EITHER measurement produced a winner, OR the GPU has
    // dedicated VRAM, where the static most-GPU-that-fits pick (row 0) is always
    // the right answer (the per-candidate micro-bench is skipped there precisely
    // because it adds no information). On a shared-LPDDR INTEGRATED GPU without
    // `--measure` we still decline to cache — the static recommendation isn't
    // trustworthy there (it needs the decode sweep). Best-effort: a cache
    // failure is warned and doesn't fail the command.
    //
    // This is the fix for the dedicated-VRAM friction: previously `measure` was
    // forced off on a dedicated GPU, so `measured_winner` was always `None` and
    // nothing was persisted — `is_untuned` stayed true forever and `serve`
    // re-ran the full first-load sweep on EVERY load. Now a single fast
    // `tune --placement` (and the Stage-5 sub-stage of `tune --all`, which calls
    // this) persists the pick and `serve` skips the sweep thereafter.
    let persist_n = measured_winner.or(if dedicated { Some(winner_n_gpu) } else { None });
    if let Some(n) = persist_n {
        let model_key = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown-model")
            .to_string();
        if let Err(e) = persist_placement_winner(&model_key, n) {
            tracing::warn!(error = %e, "failed to persist placement winner to tuner cache");
        }
    }

    println!(
        "to apply: set `[inference].n_gpu_layers = {}` in {} (the GUI Settings page \
         has the field; or edit config.toml directly + the watcher will pick up the \
         change at next model reload).",
        winner_n_gpu,
        config_path.display()
    );
    if !measure && !dedicated {
        println!();
        println!(
            "tip: pass `--measure` to actually time each candidate — the static \
             recommendation is 'most-GPU that fits', which on shared-LPDDR \
             integrated GPUs can be slower than fewer layers on GPU."
        );
    }
    Ok(())
}

/// Drive the engine once per candidate placement, time decode tok/s,
/// pick the winner. Loads the model exactly once — `set_n_gpu_layers`
/// reconfigures the per-thread dispatch cutoff between runs so we
/// don't pay model-load latency per candidate.
///
/// Returns `Some(n_gpu_layers)` of the highest-throughput run, or
/// `None` if every candidate failed (unloadable model, generate
/// errored, etc.) — caller falls back to the static-analyzer pick.
///
/// `pub` so integration tests can exercise the measurement loop end-
/// to-end against synth GGUF fixtures.
pub fn measure_placement_candidates(
    model_path: &std::path::Path,
    candidates: &[rustllama_tuner::PlacementPlan],
    ctx_size: usize,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
    cfg: &rustllama_config::Config,
) -> anyhow::Result<Option<u32>> {
    use rustllama_engine::measurement::{
        measure_placement_candidates as engine_measure_placement, MeasurementConfig,
    };

    let (k, v) = cfg.inference.resolved_kv_dtypes();
    if k != v {
        tracing::warn!(
            k = k,
            v = v,
            "split K/V dtypes configured; placement measurement uses K's dtype \
             (engine storage couples K/V today)"
        );
    }
    let kv_dtype = parse_kv_dtype(&k.to_string())?;
    let m_cfg = MeasurementConfig {
        kv_dtype,
        kv_cache_layout: cfg.inference.kv_cache_layout.clone(),
        flash_attention: cfg.inference.flash_attention,
        n_gpu_layers: cfg.inference.n_gpu_layers,
    };

    println!();
    println!("=== dynamic measurement ===");
    println!("  prompt_tokens = {prompt_tokens}");
    println!("  decode_tokens = {decode_tokens}");
    println!("  repeats       = {repeats} (median per candidate)");
    println!();

    let report = engine_measure_placement(
        model_path,
        candidates,
        ctx_size,
        prompt_tokens,
        decode_tokens,
        repeats,
        &m_cfg,
    )?;

    println!("  model loaded in {:.1} ms", report.load_ms);
    println!();
    println!(
        "  {:>5}  {:>9}  {:>11}  {:>11}  notes",
        "n_gpu", "warmup", "median tps", "p100 tps"
    );
    for c in &report.candidates {
        match (c.median_tps, c.max_tps, c.error.as_ref()) {
            (Some(med), Some(max), _) => {
                let star = if report.winner == Some(c.n_gpu_layers) && med >= report.winner_tps {
                    "  ★ best"
                } else {
                    ""
                };
                println!(
                    "  {:>5}  {:>7.0} ms  {med:>7.2} tps  {max:>7.2} tps {star}",
                    c.n_gpu_layers, c.warmup_ms,
                );
            }
            (_, _, Some(err)) => {
                println!(
                    "  {:>5}  {:>7.0} ms  {:>11}  {:>11}  {err}",
                    c.n_gpu_layers, c.warmup_ms, "—", "—"
                );
            }
            _ => {}
        }
    }
    println!();
    match report.winner {
        Some(n) => println!(
            "  → measured winner: n_gpu_layers = {n} at {:.2} tok/s median",
            report.winner_tps
        ),
        None => println!("  → no candidate completed; recommendation falls back to static pick"),
    }
    Ok(report.winner)
}

/// Sweep `[inference].batch_size` (the prefill chunk size in tokens)
/// across a candidate list, drive a long synthetic prompt through
/// each, and pick the candidate that maximizes prefill throughput.
///
/// Decode tokens are pinned to 1 so the measurement isolates the
/// prefill path — decode is what the placement sweep covers.
/// Returns Ok(()) regardless of which candidate wins; the
/// recommendation is printed inline (no automatic config edit).
fn cmd_tune_batch_size(
    config_path: &std::path::Path,
    model_override: Option<String>,
    candidates_csv: &str,
    prompt_tokens: u32,
    repeats: u32,
) -> anyhow::Result<()> {
    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let model_path = effective_model_path(model_override, &cfg, config_path)?;
    let candidates: Vec<usize> = candidates_csv
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| anyhow::anyhow!("invalid batch candidate {s:?}: {e}"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    if candidates.is_empty() {
        anyhow::bail!("--batch-candidates produced an empty list");
    }
    if repeats == 0 {
        anyhow::bail!("--batch-repeats must be >= 1");
    }

    println!("rustllama tune --batch-size");
    println!("  model         = {}", model_path.display());
    println!(
        "  candidates    = {}",
        candidates
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("  prompt_tokens = {prompt_tokens}");
    println!("  repeats       = {repeats} (median per candidate)");
    println!();

    let winner =
        measure_batch_size_candidates(&model_path, &candidates, prompt_tokens, repeats, &cfg)?;

    if let Some(b) = winner {
        if let Err(e) = persist_batch_size_winner(b as u32) {
            tracing::warn!(error = %e, "failed to persist batch_size winner to tuner cache");
        }
    }

    println!();
    match winner {
        Some(b) => println!(
            "to apply: set `[inference].batch_size = {b}` in {} (the GUI Settings page \
             has the field; or edit config.toml directly + the watcher will pick up \
             the change at next model reload).",
            config_path.display()
        ),
        None => println!(
            "no candidate completed — model may be too small for a meaningful prefill \
             measurement, or all candidates errored. Try a longer --batch-prompt-tokens \
             or check the trace logs."
        ),
    }
    Ok(())
}

/// Sweep `[inference].kv_dtype` across the candidate set and pick
/// the value that maximizes decode tok/s. Persists the winner to the
/// tuner cache under `kv_dtype`. Model is reloaded per candidate
/// (cache shape changes per dtype).
/// Phase 3: measure per-device (each GPU + the CPU tier) short-synthetic-
/// decode tok/s and persist it under `TuningResult.per_device_perf`, so the
/// Phase 5 heat placement planner can rank tiers by MEASURED throughput. A
/// mandatory stage of `tune --all`; also runnable standalone.
fn cmd_tune_per_device_perf(
    config_path: &std::path::Path,
    model_override: Option<String>,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
) -> anyhow::Result<()> {
    use rustllama_engine::measurement::{measure_per_device_perf, MeasurementConfig};

    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let model_path = effective_model_path(model_override, &cfg, config_path)?;
    if repeats == 0 {
        anyhow::bail!("--repeats must be >= 1");
    }
    let (k, _) = cfg.inference.resolved_kv_dtypes();
    let m_cfg = MeasurementConfig {
        kv_dtype: parse_kv_dtype(k)?,
        kv_cache_layout: cfg.inference.kv_cache_layout.clone(),
        flash_attention: cfg.inference.flash_attention,
        n_gpu_layers: cfg.inference.n_gpu_layers,
    };
    let ctx_size =
        (prompt_tokens as usize + decode_tokens as usize + 8).max(cfg.inference.ctx_size as usize);

    println!("rustllama tune --per-device-perf");
    println!("  model         = {}", model_path.display());
    println!("  ctx_size      = {ctx_size}");
    println!("  prompt_tokens = {prompt_tokens}");
    println!("  decode_tokens = {decode_tokens}");
    println!("  repeats       = {repeats} (median per tier)");
    println!();

    let report = measure_per_device_perf(
        &model_path,
        ctx_size,
        prompt_tokens,
        decode_tokens,
        repeats,
        &m_cfg,
    )?;
    println!("  total load time {:.1} ms", report.load_ms);
    println!();
    for tier in &report.tiers {
        match (tier.median_tps, tier.error.as_ref()) {
            (Some(tps), _) => {
                println!("  {:<28} {tps:>8.2} tps   [{}]", tier.label, tier.slug)
            }
            (None, Some(err)) => {
                println!("  {:<28} {:>8}   [{}]  {err}", tier.label, "—", tier.slug)
            }
            _ => {}
        }
    }
    println!();
    if report.scores.is_empty() {
        println!("  → no tier produced a measurement; nothing persisted");
    } else {
        match persist_per_device_perf(&report.scores) {
            Ok(()) => println!("  → persisted {} device score(s)", report.scores.len()),
            Err(e) => {
                tracing::warn!(error = %e, "failed to persist per-device perf to tuner cache")
            }
        }
    }
    Ok(())
}

/// Merge the measured per-device tok/s scores into the tuner cache's
/// `per_device_perf` map. Graceful no-op when no cache dir resolves.
pub fn persist_per_device_perf(
    scores: &std::collections::HashMap<String, f32>,
) -> anyhow::Result<()> {
    use rustllama_tuner::{load_cache, save_cache, TuningResult};
    let Some(cache_dir) = rustllama_tuner::default_cache_dir() else {
        tracing::warn!("no tuner cache dir resolvable; skipping per-device perf persistence");
        return Ok(());
    };
    let (key, device) = rustllama_tuner::cache_context();
    std::fs::create_dir_all(&cache_dir)?;
    let mut tuning = load_cache(&cache_dir, &key)?
        .unwrap_or_else(|| TuningResult::empty(key.clone(), device.clone()));
    for (slug, tps) in scores {
        tuning.per_device_perf.insert(slug.clone(), *tps);
    }
    tuning.last_tuned = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| format!("{}", d.as_secs()));
    save_cache(&cache_dir, &tuning)?;
    Ok(())
}

/// Run the on-device GPU-kernel parity probes (CUDA tensor-core GEMM +
/// SYCL XMX) and persist a pass/fail verdict per kernel under
/// `TuningResult.kernel_verdicts`. The dispatch layer reads these to
/// AUTO-ENABLE each specialized path only where it matches the CPU
/// reference on this machine — replacing the removed `RUSTLLAMA_FP4_TC` /
/// `_FP8_WGMMA` / `_SYCL_XMX` env gates. A `tune --all` sub-stage (Stage
/// 1c); also runnable standalone. Model-independent (hardware only). Safe
/// on non-GPU hosts: every probe SKIPs → no verdict → nothing persisted.
fn cmd_tune_validate_kernels() -> anyhow::Result<()> {
    println!("rustllama tune --validate-kernels");
    println!("  running GPU kernel parity probes (CUDA tensor-core GEMM, SYCL XMX)…");
    println!();

    let mut verdicts: Vec<(String, Option<bool>)> = Vec::new();
    match crate::gpu_parity::collect_cuda_verdicts() {
        Ok(mut v) => verdicts.append(&mut v),
        Err(e) => tracing::warn!(error = %e, "CUDA kernel validation probe failed"),
    }
    verdicts.extend(crate::gpu_parity::collect_xmx_verdict());

    // Keep only decided verdicts (OK / MISCOMPUTE); SKIP (capability
    // absent) carries no information and is left unset.
    let decided: Vec<(String, bool)> = verdicts
        .into_iter()
        .filter_map(|(name, v)| v.map(|pass| (name, pass)))
        .collect();

    println!();
    if decided.is_empty() {
        println!("  → no GPU kernel produced a verdict (no capable device)");
    } else {
        let pass = decided.iter().filter(|(_, p)| *p).count();
        let fail = decided.len() - pass;
        println!("  → {} verdict(s): {pass} pass, {fail} fail", decided.len());
    }
    // Always persist — even with no verdicts it stamps `rustllama_version`,
    // so the first-load self-heal re-runs this once per build (on a version
    // change) instead of on every load on a GPU-less host.
    match persist_kernel_verdicts(&decided) {
        Ok(()) if !decided.is_empty() => {
            println!("  → persisted to tuner cache (auto-enables the passing paths)")
        }
        Ok(()) => {}
        Err(e) => tracing::warn!(error = %e, "failed to persist kernel verdicts to tuner cache"),
    }
    Ok(())
}

/// Merge GPU-kernel validation verdicts into the tuner cache's
/// `kernel_verdicts` map and stamp `rustllama_version`, so a later build
/// (whose kernels may differ) re-validates instead of trusting a stale
/// verdict. Graceful no-op when no cache dir resolves.
pub fn persist_kernel_verdicts(verdicts: &[(String, bool)]) -> anyhow::Result<()> {
    use rustllama_tuner::{load_cache, save_cache, set_verdict, TuningResult};
    let Some(cache_dir) = rustllama_tuner::default_cache_dir() else {
        tracing::warn!("no tuner cache dir resolvable; skipping kernel-verdict persistence");
        return Ok(());
    };
    let (key, device) = rustllama_tuner::cache_context();
    std::fs::create_dir_all(&cache_dir)?;
    let mut tuning = load_cache(&cache_dir, &key)?
        .unwrap_or_else(|| TuningResult::empty(key.clone(), device.clone()));
    for (name, pass) in verdicts {
        set_verdict(&mut tuning, name, *pass);
    }
    tuning.rustllama_version = env!("CARGO_PKG_VERSION").to_string();
    tuning.last_tuned = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| format!("{}", d.as_secs()));
    save_cache(&cache_dir, &tuning)?;
    Ok(())
}

fn cmd_tune_kv_dtype(
    config_path: &std::path::Path,
    model_override: Option<String>,
    candidates_csv: &str,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
) -> anyhow::Result<()> {
    use rustllama_engine::measurement::{measure_kv_dtype_candidates, MeasurementConfig};

    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let model_path = effective_model_path(model_override, &cfg, config_path)?;

    // Hybrid (transformer+SSM) models: the hybrid attention forward implements
    // EVERY KV dtype (f32/q4_0 have dedicated calibrated arms; q8_0 / TurboQuant
    // / NVFP4 / MXFP route through the same per-dtype flash decode/prefill
    // kernels the dense path uses). Rather than restrict hybrids to {f32,q4_0}
    // and require a manual `RUSTLLAMA_HYBRID_KV_ANY` to try the rest, the sweep
    // now MEASURES the full candidate grid on hybrids too and lets its
    // coherence-first gate decide: an aggressive quant that diverges from the
    // f32 reference (e.g. 1-bit tq1) is rejected automatically, a coherent one
    // is adopted + persisted. We set the hybrid-KV-any flag for THIS tune
    // subprocess so each candidate is honored verbatim during measurement —
    // otherwise the engine's load-time hybrid coercion would force every
    // non-f32/q4_0 candidate to f32 and the sweep would grade f32 against
    // itself. The persisted winner is a coherence-VALIDATED choice the serve
    // path then honors with no user env var. Non-hybrids are unaffected.
    let is_hybrid = gguf_is_hybrid(&model_path);
    if is_hybrid {
        std::env::set_var("RUSTLLAMA_HYBRID_KV_ANY", "1");
    }

    let candidates: Vec<rustllama_engine::KvDtype> = candidates_csv
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(parse_kv_dtype)
        .collect::<anyhow::Result<Vec<_>>>()?;
    if candidates.is_empty() {
        anyhow::bail!("--kv-dtype-candidates produced an empty list");
    }
    if repeats == 0 {
        anyhow::bail!("--kv-dtype-repeats must be >= 1");
    }
    // Coherence needs a longer decode sample than the shared tune default (16):
    // the gate is an EXACT top-1 match, and a marginally-lossy KV dtype can agree
    // for the first few tokens then diverge — so grade over >=64 decode tokens
    // for a robust verdict. Cheap next to the per-candidate model load; only this
    // sweep is affected (placement/other sweeps keep the caller's value).
    let decode_tokens = decode_tokens.max(64);

    let m_cfg = MeasurementConfig {
        // Initial value is irrelevant — measure_kv_dtype_candidates overrides
        // it per candidate. We still need a valid one for the struct init.
        kv_dtype: candidates[0],
        kv_cache_layout: cfg.inference.kv_cache_layout.clone(),
        flash_attention: cfg.inference.flash_attention,
        n_gpu_layers: cfg.inference.n_gpu_layers,
    };
    let ctx_size =
        (prompt_tokens as usize + decode_tokens as usize + 8).max(cfg.inference.ctx_size as usize);

    println!("rustllama tune --kv-dtype");
    println!("  model         = {}", model_path.display());
    println!(
        "  candidates    = {}",
        candidates
            .iter()
            .map(kv_dtype_to_str)
            .collect::<Vec<_>>()
            .join(", ")
    );
    if is_hybrid {
        println!(
            "  hybrid model: sweeping full grid — coherence gate decides which \
             quant KV (if any) is honored"
        );
    }
    println!("  ctx_size      = {ctx_size}");
    println!("  prompt_tokens = {prompt_tokens}");
    println!("  decode_tokens = {decode_tokens}");
    println!("  repeats       = {repeats} (median per candidate)");
    println!();

    let report = measure_kv_dtype_candidates(
        &model_path,
        &candidates,
        ctx_size,
        prompt_tokens,
        decode_tokens,
        repeats,
        &m_cfg,
    )?;
    println!("  total load time {:.1} ms", report.total_load_ms);
    println!();

    // Coherence baseline: prefer F32 if present; else fall back
    // to the highest-precision candidate that produced tokens.
    // This is the reference for top-1 agreement scoring.
    let reference_dt = if candidates.contains(&rustllama_engine::KvDtype::F32) {
        rustllama_engine::KvDtype::F32
    } else {
        // Pick the highest-bits candidate that ran successfully.
        report
            .candidates
            .iter()
            .filter(|c| c.decode_tokens.is_some())
            .max_by(|a, b| {
                a.kv_dtype
                    .approx_bits_per_element()
                    .partial_cmp(&b.kv_dtype.approx_bits_per_element())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|c| c.kv_dtype)
            .unwrap_or(rustllama_engine::KvDtype::F32)
    };
    let agreements = report.agreement_vs(reference_dt);

    println!(
        "  {:>6}  {:>9}  {:>11}  {:>11}  {:>7}  {:>9}  notes",
        "dtype", "warmup", "median tps", "p100 tps", "bpe", "vs-ref"
    );
    // Coherence threshold: a candidate must agree with the reference dtype on
    // 100% of the measured decode tokens (EXACT top-1 match) to remain eligible.
    // A quant KV is adopted only when it is LOSSLESS on the probe — anything that
    // diverges even once falls back to the next-smaller coherent dtype (and
    // ultimately f32). This is deliberately strict: KV error compounds over
    // context, so "near-coherent" is not good enough to ship as the auto default.
    const COHERENCE_THRESHOLD: f32 = 1.0;

    // Speed-first-if-it-fits selection budget. Among coherent
    // candidates we want the FASTEST that still fits memory, only
    // downgrading to a smaller (slower) KV dtype under genuine
    // memory pressure. Estimate each candidate's K+V cache at the
    // DEPLOYMENT context (`[inference].ctx_size`, not the tiny probe
    // ctx used for the sweep) by scaling the f16-based
    // `kv_cache_vram_bytes` by the dtype's bits/16, and budget it
    // against available system RAM.
    //
    // RAM (not VRAM) is the pool here because (a) this stage runs
    // BEFORE the placement stage, so we don't yet know if the model
    // will be GPU- or CPU-placed, and (b) the primary target is a
    // unified-memory iGPU where the KV cache lives in shared RAM
    // anyway. On a discrete GPU the placement stage's own VRAM gate
    // remains the authoritative fit check; this budget is a coarse
    // guard that keeps us from choosing a KV dtype whose cache would
    // blow out host memory at the configured context. A budget of 0
    // (dims unreadable / RAM query failed) disables the gate and
    // honors the pure speed-first pick.
    let deploy_ctx = cfg.inference.ctx_size.max(1);
    // f16-based KV cache size at the deployment context (0 if dims
    // couldn't be read → fit gate disabled below). The per-dtype
    // estimate scales this by bits/16.
    let f16_base_for_fit: u64 = match read_dims_and_quant_from_gguf(&model_path) {
        Ok((dims, _quant)) => rustllama_tuner::placement::kv_cache_vram_bytes(dims, deploy_ctx),
        Err(e) => {
            tracing::warn!(error = %e, "kv_dtype: could not read dims for fit budget; speed-first without the fit gate");
            0
        }
    };
    // Budget: half of available RAM (leaves headroom for weights +
    // activations + OS). 0 base ⇒ 0 budget ⇒ gate disabled (pure
    // speed-first).
    let kv_budget_bytes: u64 = if f16_base_for_fit == 0 {
        0
    } else {
        let avail = rustllama_runtime::memory_info().available_bytes;
        let budget = avail / 2;
        let mib = |b: u64| (b as f64) / (1024.0 * 1024.0);
        println!(
            "  fit budget    = {:.0} MiB (½ of {:.0} MiB available RAM); KV@ctx{} f16 base {:.0} MiB",
            mib(budget),
            mib(avail),
            deploy_ctx,
            mib(f16_base_for_fit),
        );
        budget
    };
    let kv_bytes_for = |dt: rustllama_engine::KvDtype| -> u64 {
        ((f16_base_for_fit as f64) * (dt.approx_bits_per_element() as f64) / 16.0) as u64
    };

    // Speed-first-if-it-fits: fastest coherent dtype that fits the
    // budget, falling back to the smallest coherent under memory
    // pressure (see `pick_speed_first_if_fits`).
    let chosen_pick =
        report.pick_speed_first_if_fits(COHERENCE_THRESHOLD, kv_budget_bytes, kv_bytes_for);
    for c in &report.candidates {
        let name = kv_dtype_to_str(&c.kv_dtype);
        let bpe = c.kv_dtype.approx_bits_per_element();
        let agree_str = agreements
            .iter()
            .find(|(dt, _)| *dt == c.kv_dtype)
            .and_then(|(_, a)| *a)
            .map(|a| format!("{:.0}%", a * 100.0))
            .unwrap_or_else(|| "—".to_string());
        let pick_marker = if chosen_pick == Some(c.kv_dtype) {
            "  ★ pick"
        } else {
            ""
        };
        match (c.median_tps, c.max_tps, c.error.as_ref()) {
            (Some(med), Some(max), _) => {
                println!(
                    "  {name:>6}  {:>7.0} ms  {med:>7.2} tps  {max:>7.2} tps  {bpe:>5.1}b  {agree_str:>9}{pick_marker}",
                    c.warmup_ms,
                );
            }
            (_, _, Some(err)) => {
                println!(
                    "  {name:>6}  {:>7.0} ms  {:>11}  {:>11}  {bpe:>5.1}b  {agree_str:>9}  {err}",
                    c.warmup_ms, "—", "—"
                );
            }
            _ => {}
        }
    }
    println!();
    // Selection ladder:
    //   1. Speed-first-if-it-fits pick: the FASTEST coherent dtype
    //      (≥COHERENCE_THRESHOLD agreement vs the reference) whose KV
    //      cache fits the memory budget; smallest coherent under
    //      memory pressure.
    //   2. Fall back to the pure tok/s winner if no candidate
    //      cleared the coherence bar.
    //   3. Else nothing ran — leave the config value as-is.
    let chosen = chosen_pick.or(report.winner);
    match chosen {
        Some(dt) => {
            let s = kv_dtype_to_str(&dt);
            let ref_s = kv_dtype_to_str(&reference_dt);
            let how = if chosen_pick == Some(dt) {
                let fits = kv_budget_bytes == 0 || kv_bytes_for(dt) <= kv_budget_bytes;
                if fits {
                    format!(
                        "speed-first: fastest dtype with ≥{:.0}% top-1 agreement vs {ref_s} that fits the memory budget",
                        COHERENCE_THRESHOLD * 100.0
                    )
                } else {
                    format!(
                        "memory-pressure fallback: smallest dtype with ≥{:.0}% top-1 agreement vs {ref_s} (nothing faster fit the budget)",
                        COHERENCE_THRESHOLD * 100.0
                    )
                }
            } else {
                "tok/s-only (no candidate cleared the coherence bar)".to_string()
            };
            println!("  → pick: kv_dtype = {s}  [{how}]");
            if let Err(e) = persist_kv_dtype_winner(&s) {
                tracing::warn!(error = %e, "failed to persist kv_dtype winner to tuner cache");
            }
        }
        None => println!("  → no candidate completed; falling back to config value"),
    }
    Ok(())
}

/// Sweep `[inference].flash_attention = true vs false`, pick the one
/// that maximizes decode tok/s. Reuses a single loaded engine — flash
/// is a hot toggle. Persists winner under `flash_attention`.
fn cmd_tune_flash_attention(
    config_path: &std::path::Path,
    model_override: Option<String>,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
) -> anyhow::Result<()> {
    use rustllama_engine::measurement::{measure_flash_attention_candidates, MeasurementConfig};

    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let model_path = effective_model_path(model_override, &cfg, config_path)?;
    if repeats == 0 {
        anyhow::bail!("--flash-repeats must be >= 1");
    }

    let (k, _) = cfg.inference.resolved_kv_dtypes();
    let m_cfg = MeasurementConfig {
        kv_dtype: parse_kv_dtype(k)?,
        kv_cache_layout: cfg.inference.kv_cache_layout.clone(),
        flash_attention: cfg.inference.flash_attention,
        n_gpu_layers: cfg.inference.n_gpu_layers,
    };
    let ctx_size =
        (prompt_tokens as usize + decode_tokens as usize + 8).max(cfg.inference.ctx_size as usize);

    println!("rustllama tune --flash-attention");
    println!("  model         = {}", model_path.display());
    println!("  candidates    = on, off");
    println!("  ctx_size      = {ctx_size}");
    println!("  prompt_tokens = {prompt_tokens}");
    println!("  decode_tokens = {decode_tokens}");
    println!("  repeats       = {repeats} (median per candidate)");
    println!();

    let report = measure_flash_attention_candidates(
        &model_path,
        ctx_size,
        prompt_tokens,
        decode_tokens,
        repeats,
        &m_cfg,
    )?;
    println!("  model loaded in {:.1} ms", report.load_ms);
    println!();
    println!(
        "  {:>5}  {:>9}  {:>11}  {:>11}  notes",
        "flash", "warmup", "median tps", "p100 tps"
    );
    for c in &report.candidates {
        let name = if c.flash_attention { "on" } else { "off" };
        match (c.median_tps, c.max_tps, c.error.as_ref()) {
            (Some(med), Some(max), _) => {
                let star = if report.winner == Some(c.flash_attention) && med >= report.winner_tps {
                    "  ★ best"
                } else {
                    ""
                };
                println!(
                    "  {name:>5}  {:>7.0} ms  {med:>7.2} tps  {max:>7.2} tps {star}",
                    c.warmup_ms,
                );
            }
            (_, _, Some(err)) => {
                println!(
                    "  {name:>5}  {:>7.0} ms  {:>11}  {:>11}  {err}",
                    c.warmup_ms, "—", "—"
                );
            }
            _ => {}
        }
    }
    println!();
    match report.winner {
        Some(flag) => {
            println!(
                "  → measured winner: flash_attention = {flag} at {:.2} tok/s median",
                report.winner_tps
            );
            if let Err(e) = persist_flash_attention_winner(flag) {
                tracing::warn!(error = %e, "failed to persist flash_attention winner");
            }
        }
        None => println!("  → no candidate completed; falling back to config value"),
    }
    Ok(())
}

/// A/B MTP (NextN) self-speculative decode by measured decode tok/s.
/// Loads the model once and flips the hot `set_mtp_speculative` toggle
/// between arms (no reload). Hybrid + NextN-head models only: a
/// non-capable model records `Some(false)` (nothing to tune) and the
/// axis is skipped. Persists winner under `speculative_mtp`; engine
/// auto-applies on next load when `[tuning].auto_apply_speculative_mtp
/// = true`.
fn cmd_tune_speculative_mtp(
    config_path: &std::path::Path,
    model_override: Option<String>,
    repeats: u32,
) -> anyhow::Result<()> {
    use rustllama_engine::measurement::{measure_speculative_mtp, MeasurementConfig};

    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let model_path = effective_model_path(model_override, &cfg, config_path)?;
    if repeats == 0 {
        anyhow::bail!("--speculative-mtp-repeats must be >= 1");
    }

    let (k, _) = cfg.inference.resolved_kv_dtypes();
    let m_cfg = MeasurementConfig {
        kv_dtype: parse_kv_dtype(k)?,
        kv_cache_layout: cfg.inference.kv_cache_layout.clone(),
        flash_attention: cfg.inference.flash_attention,
        n_gpu_layers: cfg.inference.n_gpu_layers,
    };

    println!("rustllama tune --speculative-mtp");
    println!("  model   = {}", model_path.display());
    println!("  repeats = {repeats} (median per arm)");
    println!();

    let report = measure_speculative_mtp(&model_path, &m_cfg, repeats as usize)
        .map_err(|e| anyhow::anyhow!("MTP measurement failed: {e}"))?;

    // A non-capable model (not hybrid, or no NextN head) can never use MTP —
    // record `false` so the axis reads as tuned (an instant no-op next run)
    // rather than re-measuring every load.
    if !report.capable {
        println!("  skipped: no NextN head (MTP n/a)");
        if let Err(e) = persist_speculative_mtp_winner(false) {
            tracing::warn!(error = %e, "failed to persist speculative_mtp winner (false)");
        }
        return Ok(());
    }

    let fmt = |t: Option<f64>| {
        t.map(|v| format!("{v:.2} tps"))
            .unwrap_or_else(|| "—".to_string())
    };
    println!("  off (classic decode) = {}", fmt(report.off_tps));
    println!("  on  (MTP self-spec)  = {}", fmt(report.on_tps));
    if let Some(err) = report.error.as_ref() {
        println!("  note: {err}");
    }
    // A non-capable / off-wins model records `false` (Some(false) or a None
    // winner both collapse to disable).
    let winner = report.winner.unwrap_or(false);
    println!();
    println!("  → pick: speculative_mtp = {winner}");
    if let Err(e) = persist_speculative_mtp_winner(winner) {
        tracing::warn!(error = %e, "failed to persist speculative_mtp winner");
    }
    Ok(())
}

/// A/B the chunked-parallel SSM (DeltaNet) prefill path by measured
/// prefill tok/s. Loads the model once and toggles the
/// `RUSTLLAMA_SSM_PREFILL_CHUNKED` env var (read per-forward) between
/// arms (no reload). Hybrid models only: a non-hybrid model records
/// `Some(false)` (nothing to tune). Persists winner under
/// `ssm_prefill_chunked`; engine auto-applies on next load when
/// `[tuning].auto_apply_ssm_prefill_chunked = true`.
fn cmd_tune_ssm_prefill_chunked(
    config_path: &std::path::Path,
    model_override: Option<String>,
    repeats: u32,
) -> anyhow::Result<()> {
    use rustllama_engine::measurement::{measure_ssm_prefill_chunked, MeasurementConfig};

    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let model_path = effective_model_path(model_override, &cfg, config_path)?;
    if repeats == 0 {
        anyhow::bail!("--ssm-prefill-chunked-repeats must be >= 1");
    }

    println!("rustllama tune --ssm-prefill-chunked");
    println!("  model   = {}", model_path.display());
    println!("  repeats = {repeats} (median per arm)");
    println!();

    // The chunked-parallel SSM prefill path only exists on hybrid
    // (transformer+SSM / DeltaNet) models; on anything else the env toggle
    // changes nothing and both arms measure the same path. Skip the sweep and
    // record "off" so the axis reads as tuned.
    if !gguf_is_hybrid(&model_path) {
        println!("  skipped: not a hybrid model");
        if let Err(e) = persist_ssm_prefill_chunked_winner(false) {
            tracing::warn!(error = %e, "failed to persist ssm_prefill_chunked winner (false)");
        }
        return Ok(());
    }

    let (k, _) = cfg.inference.resolved_kv_dtypes();
    let m_cfg = MeasurementConfig {
        kv_dtype: parse_kv_dtype(k)?,
        kv_cache_layout: cfg.inference.kv_cache_layout.clone(),
        flash_attention: cfg.inference.flash_attention,
        n_gpu_layers: cfg.inference.n_gpu_layers,
    };

    let report = measure_ssm_prefill_chunked(&model_path, &m_cfg, repeats as usize)
        .map_err(|e| anyhow::anyhow!("SSM prefill measurement failed: {e}"))?;

    let fmt = |t: Option<f64>| {
        t.map(|v| format!("{v:.2} tps"))
            .unwrap_or_else(|| "—".to_string())
    };
    println!(
        "  off (sequential prefill) = {}",
        fmt(report.off_prefill_tps)
    );
    println!(
        "  on  (chunked-parallel)   = {}",
        fmt(report.on_prefill_tps)
    );
    if let Some(err) = report.error.as_ref() {
        println!("  note: {err}");
    }
    let winner = report.winner.unwrap_or(false);
    println!();
    println!("  → pick: ssm_prefill_chunked = {winner}");
    if let Err(e) = persist_ssm_prefill_chunked_winner(winner) {
        tracing::warn!(error = %e, "failed to persist ssm_prefill_chunked winner");
    }
    Ok(())
}

/// Sweep `[inference].flash_attention_kv_min` across candidate
/// thresholds. The kernel-dispatch site reads
/// `RUSTLLAMA_FLASH_KV_LEN_MIN`; we set it per candidate and time
/// decode tok/s, picking the host-specific break-even crossover
/// between flash-decode and standard attention. Persists winner
/// under `flash_kv_min`.
fn cmd_tune_flash_kv_min(
    config_path: &std::path::Path,
    model_override: Option<String>,
    candidates_csv: &str,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
) -> anyhow::Result<()> {
    use rustllama_engine::measurement::{measure_flash_kv_min_candidates, MeasurementConfig};

    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let model_path = effective_model_path(model_override, &cfg, config_path)?;
    if repeats == 0 {
        anyhow::bail!("--flash-kv-min-repeats must be >= 1");
    }
    let candidates: Vec<u32> = candidates_csv
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u32>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| anyhow::anyhow!("invalid --flash-kv-min-candidates list: {e}"))?;
    if candidates.is_empty() {
        anyhow::bail!("--flash-kv-min-candidates produced no values");
    }

    let (k, _) = cfg.inference.resolved_kv_dtypes();
    let m_cfg = MeasurementConfig {
        kv_dtype: parse_kv_dtype(k)?,
        kv_cache_layout: cfg.inference.kv_cache_layout.clone(),
        flash_attention: cfg.inference.flash_attention,
        n_gpu_layers: cfg.inference.n_gpu_layers,
    };
    // Decode must exceed the largest candidate or the threshold
    // doesn't actually affect dispatch (we'd measure the same path
    // for every candidate). Bump it to 2× the largest candidate.
    let largest = *candidates.iter().max().unwrap_or(&256);
    let decode_tokens = decode_tokens.max(largest * 2);
    let ctx_size =
        (prompt_tokens as usize + decode_tokens as usize + 8).max(cfg.inference.ctx_size as usize);

    println!("rustllama tune --flash-kv-min");
    println!("  model         = {}", model_path.display());
    println!(
        "  candidates    = {}",
        candidates
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("  ctx_size      = {ctx_size}");
    println!("  prompt_tokens = {prompt_tokens}");
    println!("  decode_tokens = {decode_tokens} (clamped to 2× largest candidate)");
    println!("  repeats       = {repeats} (median per candidate)");
    println!();

    let report = measure_flash_kv_min_candidates(
        &model_path,
        &candidates,
        ctx_size,
        prompt_tokens,
        decode_tokens,
        repeats,
        &m_cfg,
    )?;
    println!("  model loaded in {:.1} ms", report.load_ms);
    println!();
    println!(
        "  {:>6}  {:>9}  {:>11}  {:>11}  notes",
        "kv_min", "warmup", "median tps", "p100 tps"
    );
    for c in &report.candidates {
        match (c.median_tps, c.max_tps, c.error.as_ref()) {
            (Some(med), Some(max), _) => {
                let star = if report.winner == Some(c.kv_min) && med >= report.winner_tps {
                    "  ★ best"
                } else {
                    ""
                };
                println!(
                    "  {:>6}  {:>7.0} ms  {med:>7.2} tps  {max:>7.2} tps {star}",
                    c.kv_min, c.warmup_ms,
                );
            }
            (_, _, Some(err)) => println!(
                "  {:>6}  {:>7.0} ms  {:>11}  {:>11}  {err}",
                c.kv_min, c.warmup_ms, "—", "—"
            ),
            _ => {}
        }
    }
    println!();
    match report.winner {
        Some(v) => {
            println!(
                "  → measured winner: flash_kv_min = {v} at {:.2} tok/s median",
                report.winner_tps
            );
            if let Err(e) = persist_flash_kv_min_winner(v) {
                tracing::warn!(error = %e, "failed to persist flash_kv_min winner");
            }
        }
        None => println!("  → no candidate completed; falling back to config value"),
    }
    Ok(())
}

/// Sweep `[inference].prefix_cache_max_snapshots` across pool
/// depths. Persists winner under `prefix_cache_max_snapshots`.
fn cmd_tune_prefix_snapshots(
    config_path: &std::path::Path,
    model_override: Option<String>,
    candidates_csv: &str,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
) -> anyhow::Result<()> {
    use rustllama_engine::measurement::{measure_prefix_snapshots_candidates, MeasurementConfig};

    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let model_path = effective_model_path(model_override, &cfg, config_path)?;
    if repeats == 0 {
        anyhow::bail!("--prefix-snapshots-repeats must be >= 1");
    }
    let candidates: Vec<u32> = candidates_csv
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u32>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| anyhow::anyhow!("invalid --prefix-snapshots-candidates list: {e}"))?;
    if candidates.is_empty() {
        anyhow::bail!("--prefix-snapshots-candidates produced no values");
    }

    let (k, _) = cfg.inference.resolved_kv_dtypes();
    let m_cfg = MeasurementConfig {
        kv_dtype: parse_kv_dtype(k)?,
        kv_cache_layout: cfg.inference.kv_cache_layout.clone(),
        flash_attention: cfg.inference.flash_attention,
        n_gpu_layers: cfg.inference.n_gpu_layers,
    };
    let ctx_size =
        (prompt_tokens as usize + decode_tokens as usize + 8).max(cfg.inference.ctx_size as usize);

    println!("rustllama tune --prefix-snapshots");
    println!("  model         = {}", model_path.display());
    println!(
        "  candidates    = {}",
        candidates
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("  ctx_size      = {ctx_size}");
    println!("  prompt_tokens = {prompt_tokens}");
    println!("  decode_tokens = {decode_tokens}");
    println!("  repeats       = {repeats} (median per candidate)");
    println!();

    let report = measure_prefix_snapshots_candidates(
        &model_path,
        &candidates,
        ctx_size,
        prompt_tokens,
        decode_tokens,
        repeats,
        &m_cfg,
    )?;
    println!("  model loaded in {:.1} ms", report.load_ms);
    println!();
    println!(
        "  {:>5}  {:>9}  {:>11}  {:>11}  notes",
        "depth", "warmup", "median tps", "p100 tps"
    );
    for c in &report.candidates {
        match (c.median_tps, c.max_tps, c.error.as_ref()) {
            (Some(med), Some(max), _) => {
                let star = if report.winner == Some(c.max_snapshots) && med >= report.winner_tps {
                    "  ★ best"
                } else {
                    ""
                };
                println!(
                    "  {:>5}  {:>7.0} ms  {med:>7.2} tps  {max:>7.2} tps {star}",
                    c.max_snapshots, c.warmup_ms,
                );
            }
            (_, _, Some(err)) => println!(
                "  {:>5}  {:>7.0} ms  {:>11}  {:>11}  {err}",
                c.max_snapshots, c.warmup_ms, "—", "—"
            ),
            _ => {}
        }
    }
    println!();
    match report.winner {
        Some(v) => {
            println!(
                "  → measured winner: prefix_cache_max_snapshots = {v} at {:.2} tok/s median",
                report.winner_tps
            );
            if let Err(e) = persist_prefix_snapshots_winner(v) {
                tracing::warn!(error = %e, "failed to persist prefix_snapshots winner");
            }
        }
        None => println!("  → no candidate completed; falling back to config value"),
    }
    Ok(())
}

/// Sweep `KV_TILE` for the SYCL flash-attn-v3 decode kernel. The
/// kernel reads `RUSTLLAMA_FLASH_V3_KV_TILE` at dispatch time; we
/// set it per candidate, time decode tok/s, and persist the winner
/// to the tuner cache. Only meaningful on real-SYCL builds; on
/// mock-feature builds every candidate measures the same CPU code
/// path.
fn cmd_tune_flash_v3_kv_tile(
    config_path: &std::path::Path,
    model_override: Option<String>,
    candidates_csv: &str,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
) -> anyhow::Result<()> {
    use rustllama_engine::measurement::{measure_flash_v3_kv_tile_candidates, MeasurementConfig};

    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let model_path = effective_model_path(model_override, &cfg, config_path)?;
    if repeats == 0 {
        anyhow::bail!("--flash-v3-kv-tile-repeats must be >= 1");
    }
    let candidates: Vec<u32> = candidates_csv
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u32>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| anyhow::anyhow!("invalid --flash-v3-kv-tile-candidates list: {e}"))?;
    if candidates.is_empty() {
        anyhow::bail!("--flash-v3-kv-tile-candidates produced no values");
    }
    for c in &candidates {
        if !matches!(c, 16 | 32 | 64) {
            tracing::warn!(
                candidate = c,
                "KV_TILE candidate outside the kernel's supported set {{16, 32, 64}}; \
                 the kernel will fall back to 32 for this value"
            );
        }
    }

    let (k, _) = cfg.inference.resolved_kv_dtypes();
    let m_cfg = MeasurementConfig {
        kv_dtype: parse_kv_dtype(k)?,
        kv_cache_layout: cfg.inference.kv_cache_layout.clone(),
        flash_attention: cfg.inference.flash_attention,
        n_gpu_layers: cfg.inference.n_gpu_layers,
    };
    let ctx_size =
        (prompt_tokens as usize + decode_tokens as usize + 8).max(cfg.inference.ctx_size as usize);

    println!("rustllama tune --flash-v3-kv-tile");
    println!("  model         = {}", model_path.display());
    println!(
        "  candidates    = {}",
        candidates
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("  ctx_size      = {ctx_size}");
    println!("  prompt_tokens = {prompt_tokens}");
    println!("  decode_tokens = {decode_tokens}");
    println!("  repeats       = {repeats} (median per candidate)");
    println!();

    let report = measure_flash_v3_kv_tile_candidates(
        &model_path,
        &candidates,
        ctx_size,
        prompt_tokens,
        decode_tokens,
        repeats,
        &m_cfg,
    )?;
    println!("  model loaded in {:.1} ms", report.load_ms);
    println!();
    println!(
        "  {:>5}  {:>9}  {:>11}  {:>11}  notes",
        "tile", "warmup", "median tps", "p100 tps"
    );
    for c in &report.candidates {
        match (c.median_tps, c.max_tps, c.error.as_ref()) {
            (Some(med), Some(max), _) => {
                let star = if report.winner == Some(c.kv_tile) && med >= report.winner_tps {
                    "  ★ best"
                } else {
                    ""
                };
                println!(
                    "  {:>5}  {:>7.0} ms  {med:>7.2} tps  {max:>7.2} tps {star}",
                    c.kv_tile, c.warmup_ms,
                );
            }
            (_, _, Some(err)) => println!(
                "  {:>5}  {:>7.0} ms  {:>11}  {:>11}  {err}",
                c.kv_tile, c.warmup_ms, "—", "—"
            ),
            _ => {}
        }
    }
    println!();
    match report.winner {
        Some(v) => {
            println!(
                "  → measured winner: flash_v3_kv_tile = {v} at {:.2} tok/s median",
                report.winner_tps
            );
            if let Err(e) = persist_flash_v3_kv_tile_winner(v) {
                tracing::warn!(error = %e, "failed to persist flash_v3_kv_tile winner");
            }
        }
        None => println!("  → no candidate completed; falling back to kernel default (32)"),
    }
    Ok(())
}

/// Sweep `[inference].kv_page_size` across candidate token-counts-
/// per-page. Reloads the engine per candidate; only meaningful when
/// `kv_cache_layout = "paged"`. Persists winner under `kv_page_size`.
fn cmd_tune_kv_page_size(
    config_path: &std::path::Path,
    model_override: Option<String>,
    candidates_csv: &str,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
) -> anyhow::Result<()> {
    use rustllama_engine::measurement::{measure_kv_page_size_candidates, MeasurementConfig};

    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let model_path = effective_model_path(model_override, &cfg, config_path)?;
    if repeats == 0 {
        anyhow::bail!("--kv-page-size-repeats must be >= 1");
    }
    if cfg.inference.kv_cache_layout != "paged" {
        tracing::warn!(
            layout = %cfg.inference.kv_cache_layout,
            "kv_page_size sweep is a no-op on non-paged layouts \
             (all candidates measure the same path)"
        );
    }
    let candidates: Vec<u32> = candidates_csv
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<u32>())
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| anyhow::anyhow!("invalid --kv-page-size-candidates list: {e}"))?;
    if candidates.is_empty() {
        anyhow::bail!("--kv-page-size-candidates produced no values");
    }

    let (k, _) = cfg.inference.resolved_kv_dtypes();
    let m_cfg = MeasurementConfig {
        kv_dtype: parse_kv_dtype(k)?,
        kv_cache_layout: cfg.inference.kv_cache_layout.clone(),
        flash_attention: cfg.inference.flash_attention,
        n_gpu_layers: cfg.inference.n_gpu_layers,
    };
    let ctx_size =
        (prompt_tokens as usize + decode_tokens as usize + 8).max(cfg.inference.ctx_size as usize);

    println!("rustllama tune --kv-page-size");
    println!("  model         = {}", model_path.display());
    println!("  layout        = {}", cfg.inference.kv_cache_layout);
    println!(
        "  candidates    = {}",
        candidates
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
    println!("  ctx_size      = {ctx_size}");
    println!("  prompt_tokens = {prompt_tokens}");
    println!("  decode_tokens = {decode_tokens}");
    println!("  repeats       = {repeats} (median per candidate)");
    println!();

    let report = measure_kv_page_size_candidates(
        &model_path,
        &candidates,
        ctx_size,
        prompt_tokens,
        decode_tokens,
        repeats,
        &m_cfg,
    )?;
    println!("  total load time {:.1} ms", report.total_load_ms);
    println!();
    println!(
        "  {:>5}  {:>9}  {:>11}  {:>11}  notes",
        "page", "warmup", "median tps", "p100 tps"
    );
    for c in &report.candidates {
        match (c.median_tps, c.max_tps, c.error.as_ref()) {
            (Some(med), Some(max), _) => {
                let star = if report.winner == Some(c.page_size) && med >= report.winner_tps {
                    "  ★ best"
                } else {
                    ""
                };
                println!(
                    "  {:>5}  {:>7.0} ms  {med:>7.2} tps  {max:>7.2} tps {star}",
                    c.page_size, c.warmup_ms,
                );
            }
            (_, _, Some(err)) => println!(
                "  {:>5}  {:>7.0} ms  {:>11}  {:>11}  {err}",
                c.page_size, c.warmup_ms, "—", "—"
            ),
            _ => {}
        }
    }
    println!();
    match report.winner {
        Some(v) => {
            println!(
                "  → measured winner: kv_page_size = {v} at {:.2} tok/s median",
                report.winner_tps
            );
            if let Err(e) = persist_kv_page_size_winner(v) {
                tracing::warn!(error = %e, "failed to persist kv_page_size winner");
            }
        }
        None => println!("  → no candidate completed; falling back to config value"),
    }
    Ok(())
}

/// Sweep `[inference].kv_cache_layout` ("contiguous" vs "paged"),
/// pick the value that maximizes decode tok/s. Reloads per candidate.
/// Persists winner under `kv_cache_layout`.
fn cmd_tune_kv_layout(
    config_path: &std::path::Path,
    model_override: Option<String>,
    candidates_csv: &str,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
) -> anyhow::Result<()> {
    use rustllama_engine::measurement::{measure_kv_layout_candidates, MeasurementConfig};

    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let model_path = effective_model_path(model_override, &cfg, config_path)?;
    let candidates: Vec<String> = candidates_csv
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if candidates.is_empty() {
        anyhow::bail!("--kv-layout-candidates produced an empty list");
    }
    if repeats == 0 {
        anyhow::bail!("--kv-layout-repeats must be >= 1");
    }

    let (k, _) = cfg.inference.resolved_kv_dtypes();
    let m_cfg = MeasurementConfig {
        kv_dtype: parse_kv_dtype(k)?,
        // Overridden per candidate by the measurement loop.
        kv_cache_layout: candidates[0].clone(),
        flash_attention: cfg.inference.flash_attention,
        n_gpu_layers: cfg.inference.n_gpu_layers,
    };
    let ctx_size =
        (prompt_tokens as usize + decode_tokens as usize + 8).max(cfg.inference.ctx_size as usize);

    println!("rustllama tune --kv-layout");
    println!("  model         = {}", model_path.display());
    println!("  candidates    = {}", candidates.join(", "));
    println!("  ctx_size      = {ctx_size}");
    println!("  prompt_tokens = {prompt_tokens}");
    println!("  decode_tokens = {decode_tokens}");
    println!("  repeats       = {repeats} (median per candidate)");
    println!();

    let report = measure_kv_layout_candidates(
        &model_path,
        &candidates,
        ctx_size,
        prompt_tokens,
        decode_tokens,
        repeats,
        &m_cfg,
    )?;
    println!("  total load time {:.1} ms", report.total_load_ms);
    println!();
    println!(
        "  {:>10}  {:>9}  {:>11}  {:>11}  notes",
        "layout", "warmup", "median tps", "p100 tps"
    );
    for c in &report.candidates {
        let name = c.kv_cache_layout.as_str();
        match (c.median_tps, c.max_tps, c.error.as_ref()) {
            (Some(med), Some(max), _) => {
                let star = if report.winner.as_deref() == Some(name) && med >= report.winner_tps {
                    "  ★ best"
                } else {
                    ""
                };
                println!(
                    "  {name:>10}  {:>7.0} ms  {med:>7.2} tps  {max:>7.2} tps {star}",
                    c.warmup_ms,
                );
            }
            (_, _, Some(err)) => {
                println!(
                    "  {name:>10}  {:>7.0} ms  {:>11}  {:>11}  {err}",
                    c.warmup_ms, "—", "—"
                );
            }
            _ => {}
        }
    }
    println!();
    match &report.winner {
        Some(layout) => {
            println!(
                "  → measured winner: kv_cache_layout = {layout} at {:.2} tok/s median",
                report.winner_tps
            );
            if let Err(e) = persist_kv_cache_layout_winner(layout) {
                tracing::warn!(error = %e, "failed to persist kv_cache_layout winner");
            }
        }
        None => println!("  → no candidate completed; falling back to config value"),
    }
    Ok(())
}

/// Comprehensive autotune: chain every sweep in coordinate-descent
/// order, feeding each stage's winner into the next stage's
/// measurement config. Persists all winners to the cache.
///
/// Order: kernel LWS → kv_dtype → flash_attention → kv_cache_layout
/// → placement (measured) → batch_size → threads. The first stage is
/// the existing per-shape kernel LWS sweep; the rest are the new
/// engine-level sweeps. Stages that depend on a prior winner (e.g.
/// placement uses the chosen kv_dtype + flash setting) consult the
/// cache between phases rather than re-passing values through call
/// arguments.
/// Run one `tune --<stage>` as an isolated subprocess of this binary.
///
/// The autotuner reloads the model many times per `tune --all` run,
/// and on this SYCL stack the cumulative USM / stream state across
/// ~16 in-process reloads faults the driver (0xC0000005) mid-sweep —
/// a hard access violation that `catch_unwind` cannot contain and that
/// would otherwise abort the whole `tune --all` process, losing every
/// later stage AND making the mandatory first-load autotune report a
/// failure (even though the earlier stages' winners persisted). Running
/// the crash-prone model-reloading stages as fresh child processes —
/// mirroring `doctor --sycl-parity`, which subprocesses each probe so a
/// DEVICE_LOST only kills its child — resets that per-process state to
/// zero, so each stage loads into a clean driver. Every model-reloading
/// stage of `tune --all` (1b/per-device-perf, 2/kv_dtype, 3/flash,
/// 4/kv_layout, 5/placement, 6/batch_size, and 7–10) goes through here;
/// only Stage 1 (the pure SYCL kernel-LWS shape sweep, which loads no
/// full model) stays in-process, so the parent performs ZERO model
/// loads and can never reach the fault threshold.
///
/// `extra_args` are the stage's candidate + prompt/decode-token + repeats
/// flags, forwarded verbatim so a user's `tune --all` sizing overrides
/// (`--kv-dtype-candidates`, `--batch-candidates`, `--prompt-tokens`, …)
/// reach the child rather than being silently dropped. The child's own
/// `--<stage>` dispatch parses them exactly as a standalone single-stage
/// invocation would, so behavior matches the former in-process call.
/// Stages 7–10 pass no extra args (their subcommands use the same
/// quick-by-default sizing, escalated together by `--thorough`).
///
/// Each stage subcommand persists its own winner to the shared
/// tuner-cache TOML, so a child that succeeds contributes exactly what
/// the in-process call would; a child that crashes is logged and the
/// parent moves on. The child inherits the parent's env (keep_quant_raw
/// etc.) and never passes `--force`/`--clear` — the parent already
/// cleared the cache once at the top of the sweep, and each child only
/// adds its key (so an earlier stage's winner is never wiped mid-run).
fn run_tune_stage_subprocess(
    config_path: &std::path::Path,
    model: &Option<String>,
    stage_flag: &str,
    thorough: bool,
    stage_label: &str,
    extra_args: &[String],
) -> anyhow::Result<()> {
    let exe = std::env::current_exe()
        .map_err(|e| anyhow::anyhow!("cannot resolve current exe for tune subprocess: {e}"))?;
    let mut cmd = std::process::Command::new(exe);
    // `--config` is a global arg; forward the parent's resolved path so
    // the child keys the same cache + honors the same config overrides.
    cmd.arg("--config").arg(config_path);
    cmd.arg("tune").arg(stage_flag);
    if let Some(m) = model {
        cmd.arg("--model").arg(m);
    }
    if thorough {
        cmd.arg("--thorough");
    }
    // Stage-specific candidate/sizing flags (empty for stages 7–10).
    for a in extra_args {
        cmd.arg(a);
    }
    let status = cmd
        .status()
        .map_err(|e| anyhow::anyhow!("failed to spawn `tune {stage_flag}` subprocess: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        // A crashed child (access violation, DEVICE_LOST) surfaces here
        // as a non-success status; the caller logs and continues so the
        // remaining stages still run.
        anyhow::bail!("`tune {stage_label}` subprocess exited with {status}")
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_tune_all(
    config_path: &std::path::Path,
    device_spec: Option<String>,
    model_override: Option<String>,
    thorough: bool,
    clear: bool,
    vram_mb: u64,
    vram_headroom_mb: u64,
    placement_ctx: Option<u32>,
    measure_prompt_tokens: u32,
    measure_decode_tokens: u32,
    measure_repeats: u32,
    batch_candidates: &str,
    // The batch-size stage (6) now runs as a subprocess; its prefill prompt
    // length is derived by the child from `--thorough` (there is no
    // `--batch-prompt-tokens` flag to forward), so this caller-supplied
    // sizing is no longer consumed here. Kept for call-site stability;
    // prefixed to mark unused.
    _batch_prompt_tokens: u32,
    batch_repeats: u32,
    // Threads stage (7) now runs as a subprocess with its own quick-by-default
    // sizing, so these caller-supplied sizings are no longer consumed here.
    // Kept in the signature for call-site stability; prefixed to mark unused.
    _threads_candidates: &str,
    _threads_decode_tokens: u32,
    _threads_repeats: u32,
    kv_dtype_candidates: &str,
    kv_dtype_prompt_tokens: u32,
    kv_dtype_decode_tokens: u32,
    kv_dtype_repeats: u32,
    flash_prompt_tokens: u32,
    flash_decode_tokens: u32,
    flash_repeats: u32,
    kv_layout_candidates: &str,
    kv_layout_prompt_tokens: u32,
    kv_layout_decode_tokens: u32,
    kv_layout_repeats: u32,
    skip_cached: bool,
    force: bool,
) -> anyhow::Result<()> {
    println!("rustllama tune --all");
    println!("  mode = {}", if thorough { "thorough" } else { "quick" });
    if skip_cached {
        println!("  skip-cached = on (stages with a persisted winner will be skipped)");
    }
    if force {
        println!("  force       = on (cache will be invalidated; every stage re-runs)");
    }
    println!();

    // Resolve the cache KEY (whole-system fingerprint) + cache dir so the
    // skip-cached path can peek at which stages already have winners. The key
    // resolves on SYCL / CUDA / CPU hosts alike, so the snapshot + force-wipe
    // target the same cache file the engine + per-stage persistence use.
    // (`--device` / `device_spec` is still forwarded to the Stage-1 kernel-LWS
    // sweep, which needs a concrete SYCL device index.)
    let cache_dir = rustllama_tuner::default_cache_dir();
    let cache_key = rustllama_tuner::system_fingerprint();

    // `--force` invalidates the cache before stage 1 runs.
    // Stage 1 (`cmd_tune`) also honors the `clear` arg by deleting
    // its own cache file, so passing `clear=true` from here would
    // double-wipe — we do the wipe explicitly here so any later
    // stage that loads the cache sees the empty state too.
    if force {
        if let Some(dir) = cache_dir.as_ref() {
            let path = rustllama_tuner::cache_path_for(dir, &cache_key);
            if path.exists() {
                if let Err(e) = std::fs::remove_file(&path) {
                    tracing::warn!(error = %e, path = %path.display(),
                        "force: failed to delete cache file; continuing anyway");
                } else {
                    println!("force: cleared {}", path.display());
                }
            }
        }
    }

    // Snapshot the cache *after* a potential force-wipe so the
    // skip checks below see the post-wipe state.
    let cached: Option<rustllama_tuner::TuningResult> = if skip_cached && !force {
        cache_dir
            .as_ref()
            .and_then(|dir| rustllama_tuner::load_cache(dir, &cache_key).ok().flatten())
    } else {
        None
    };

    let has_kernels = cached
        .as_ref()
        .map(|t| !t.kernels.is_empty())
        .unwrap_or(false);
    let has_kv_dtype = cached.as_ref().and_then(|t| t.kv_dtype.clone()).is_some();
    let has_flash = cached.as_ref().and_then(|t| t.flash_attention).is_some();
    let has_kv_layout = cached
        .as_ref()
        .and_then(|t| t.kv_cache_layout.clone())
        .is_some();
    // Placement is PER-MODEL (keyed by the model's file stem), so it must
    // only be treated as "already cached" when the cache holds an entry for
    // THIS model — not merely for some other model. The previous
    // `!placement.is_empty()` skipped Stage 5 for any never-tuned model, so
    // its placement was never persisted and `is_untuned` stayed true forever
    // (re-running first-load tuning on every load). Derive the stem from
    // `--model` (a `.gguf` path in the first-load-autotune subprocess) or the
    // config's `[model].path`. (NB: placement is a per-DEVICE property today;
    // a whole-system fingerprint is the planned evolution — see the mixed
    // multi-GPU/CPU case — but the per-model keying is correct either way.)
    // Derive the per-model cache key from the RESOLVED model path, so a hub
    // ref / cached id and the on-disk path it resolves to yield the same stem
    // (the subprocess stages resolve `--model` the same way).
    let tune_model_key: Option<String> = {
        let cfg = rustllama_config::load(config_path).unwrap_or_default();
        effective_model_path(model_override.clone(), &cfg, config_path)
            .ok()
            .and_then(|p| p.file_stem().and_then(|s| s.to_str()).map(str::to_string))
    };
    let has_placement = match (cached.as_ref(), tune_model_key.as_ref()) {
        (Some(t), Some(k)) => t.placement.contains_key(k),
        _ => false,
    };
    let has_batch_size = cached.as_ref().and_then(|t| t.batch_size).is_some();
    let has_threads = cached.as_ref().and_then(|t| t.threads.clone()).is_some();
    let has_decision_calibration = cached
        .as_ref()
        .map(|t| !t.decision_calibration.is_empty())
        .unwrap_or(false);
    // Phase 3: per-device (GPU + CPU tier) measured decode tok/s. Feeds the
    // Phase 5 heat placement planner, which requires MEASURED perf.
    let has_per_device_perf = cached
        .as_ref()
        .map(|t| !t.per_device_perf.is_empty())
        .unwrap_or(false);
    // Kernel verdicts are cached ONLY if present AND stamped by the
    // CURRENT build — a different build may have changed the kernels (e.g.
    // a Blackwell GEMM fix), so re-validate on a version change. This is
    // the self-heal that auto-refreshes verdicts after an upgrade without
    // the user clearing any cache.
    let has_kernel_verdicts = cached
        .as_ref()
        .map(|t| {
            !t.kernel_verdicts.is_empty() && t.rustllama_version == env!("CARGO_PKG_VERSION")
        })
        .unwrap_or(false);
    // MTP self-speculation + chunked SSM prefill: both persist a
    // `Some(bool)` winner (including `Some(false)` for a non-capable /
    // non-hybrid model), so `.is_some()` is the "already tuned" probe.
    let has_speculative_mtp = cached
        .as_ref()
        .map(|t| t.speculative_mtp.is_some())
        .unwrap_or(false);
    let has_ssm_prefill_chunked = cached
        .as_ref()
        .map(|t| t.ssm_prefill_chunked.is_some())
        .unwrap_or(false);

    let skip = |populated: bool| skip_cached && !force && populated;

    // Stage 1 (kernel LWS) is the ONLY stage that runs in-process: with
    // `measure_tok_s = false` (passed below) it loads NO full model — it
    // opens the GGUF tensor table and sweeps packed-quant matvec LWS on a
    // single SYCL stream. Every stage AFTER it reloads the model (each via
    // a `measure_*` helper), and on this SYCL stack ~16 cumulative
    // in-process reloads fault the driver with a 0xC0000005 access
    // violation mid-sweep. So stages 1b-6 (like 7-10) now run as ISOLATED
    // SUBPROCESSES via `run_tune_stage_subprocess`: each child loads the
    // model into fresh SYCL/USM state, the parent performs ZERO model
    // loads for the whole `--all` run, and a child crash is contained +
    // logged so the remaining stages still run. Coordinate descent is
    // preserved — stages run sequentially and each child reads the
    // current on-disk tuner cache and writes its winner back before the
    // next stage spawns. The stages' candidate/sizing overrides are
    // forwarded to each child (see the per-stage `args_*` below), so a
    // user's `tune --all --kv-dtype-candidates …` / `--thorough` is not
    // silently dropped; the mandatory first-load autotune uses defaults
    // either way.
    println!("Stage 1/10: kernel LWS (per-shape packed-quant matvec)");
    println!("--------");
    let model_clone_for_stages = model_override.clone();
    if skip(has_kernels) {
        println!("(skipped: kernel LWS winners already cached)");
    } else if let Err(e) = cmd_tune(
        config_path,
        device_spec,
        model_override,
        thorough,
        clear,
        false,
        0,
        0,
    ) {
        tracing::warn!(error = %e, "stage 1 (kernel LWS) failed; continuing with remaining stages");
    }

    println!();
    println!("Stage 1b/10: per-device perf (each GPU + CPU tier decode tok/s — heat placement)");
    println!("--------");
    if skip(has_per_device_perf) {
        println!("(skipped: per-device perf already cached)");
    } else {
        // Reuse the placement measurement's prompt/decode sizing so the
        // synthetic decode is representative of the placement sweep.
        let args_1b = vec![
            "--prompt-tokens".to_string(),
            measure_prompt_tokens.to_string(),
            "--decode-tokens".to_string(),
            measure_decode_tokens.to_string(),
            "--repeats".to_string(),
            measure_repeats.to_string(),
        ];
        if let Err(e) = run_tune_stage_subprocess(
            config_path,
            &model_clone_for_stages,
            "--per-device-perf",
            thorough,
            "1b (per-device perf)",
            &args_1b,
        ) {
            tracing::warn!(error = %e, "stage 1b (per-device perf) failed; continuing");
        }
    }

    println!();
    println!("Stage 1c/10: validate GPU kernels (tensor-core GEMM / XMX parity → auto-enable)");
    println!("--------");
    if skip(has_kernel_verdicts) {
        println!("(skipped: kernel verdicts already cached for this build)");
    } else {
        // Hardware-only (model-independent) + fast; still a subprocess for a
        // clean GPU context, consistent with the other stages.
        if let Err(e) = run_tune_stage_subprocess(
            config_path,
            &model_clone_for_stages,
            "--validate-kernels",
            thorough,
            "1c (validate kernels)",
            &[],
        ) {
            tracing::warn!(error = %e, "stage 1c (validate kernels) failed; continuing");
        }
    }

    println!();
    println!("Stage 2/10: kv_dtype  (coherence-first: smallest KV dtype whose decode matches F32)");
    println!("--------");
    if skip(has_kv_dtype) {
        println!("(skipped: kv_dtype winner already cached)");
    } else {
        let args_2 = vec![
            "--prompt-tokens".to_string(),
            kv_dtype_prompt_tokens.to_string(),
            "--decode-tokens".to_string(),
            kv_dtype_decode_tokens.to_string(),
            "--repeats".to_string(),
            kv_dtype_repeats.to_string(),
            "--kv-dtype-candidates".to_string(),
            kv_dtype_candidates.to_string(),
        ];
        if let Err(e) = run_tune_stage_subprocess(
            config_path,
            &model_clone_for_stages,
            "--kv-dtype",
            thorough,
            "2 (kv_dtype)",
            &args_2,
        ) {
            tracing::warn!(error = %e, "stage 2 (kv_dtype) failed; continuing");
        }
    }

    println!();
    println!("Stage 3/10: flash_attention");
    println!("--------");
    if skip(has_flash) {
        println!("(skipped: flash_attention winner already cached)");
    } else {
        let args_3 = vec![
            "--prompt-tokens".to_string(),
            flash_prompt_tokens.to_string(),
            "--decode-tokens".to_string(),
            flash_decode_tokens.to_string(),
            "--repeats".to_string(),
            flash_repeats.to_string(),
        ];
        if let Err(e) = run_tune_stage_subprocess(
            config_path,
            &model_clone_for_stages,
            "--flash-attention",
            thorough,
            "3 (flash_attention)",
            &args_3,
        ) {
            tracing::warn!(error = %e, "stage 3 (flash_attention) failed; continuing");
        }
    }

    println!();
    println!("Stage 4/10: kv_cache_layout");
    println!("--------");
    if skip(has_kv_layout) {
        println!("(skipped: kv_cache_layout winner already cached)");
    } else {
        let args_4 = vec![
            "--prompt-tokens".to_string(),
            kv_layout_prompt_tokens.to_string(),
            "--decode-tokens".to_string(),
            kv_layout_decode_tokens.to_string(),
            "--repeats".to_string(),
            kv_layout_repeats.to_string(),
            "--kv-layout-candidates".to_string(),
            kv_layout_candidates.to_string(),
        ];
        if let Err(e) = run_tune_stage_subprocess(
            config_path,
            &model_clone_for_stages,
            "--kv-layout",
            thorough,
            "4 (kv_cache_layout)",
            &args_4,
        ) {
            tracing::warn!(error = %e, "stage 4 (kv_cache_layout) failed; continuing");
        }
    }

    println!();
    println!("Stage 5/10: placement (measured)");
    println!("--------");
    if skip(has_placement) {
        println!("(skipped: placement winner already cached)");
    } else {
        // `--measure` selects the dynamic measurement half (the in-process
        // call forced `measure = true`); VRAM budget + optional ctx are
        // forwarded so the child fits the same candidate table.
        let mut args_5 = vec![
            "--measure".to_string(),
            "--vram-mb".to_string(),
            vram_mb.to_string(),
            "--vram-headroom-mb".to_string(),
            vram_headroom_mb.to_string(),
            "--prompt-tokens".to_string(),
            measure_prompt_tokens.to_string(),
            "--decode-tokens".to_string(),
            measure_decode_tokens.to_string(),
            "--repeats".to_string(),
            measure_repeats.to_string(),
        ];
        if let Some(ctx) = placement_ctx {
            args_5.push("--placement-ctx".to_string());
            args_5.push(ctx.to_string());
        }
        if let Err(e) = run_tune_stage_subprocess(
            config_path,
            &model_clone_for_stages,
            "--placement",
            thorough,
            "5 (placement)",
            &args_5,
        ) {
            tracing::warn!(error = %e, "stage 5 (placement) failed; continuing");
        }
    }

    println!();
    println!("Stage 6/10: batch_size");
    println!("--------");
    if skip(has_batch_size) {
        println!("(skipped: batch_size winner already cached)");
    } else {
        // `batch_candidates` / `batch_repeats` are already the effective
        // (quick/thorough-resolved) values; the child re-applies the same
        // `--thorough` transform idempotently (quick forces "128,256" +
        // a 256-token prefill regardless, which is what these resolve to).
        let args_6 = vec![
            "--batch-candidates".to_string(),
            batch_candidates.to_string(),
            "--repeats".to_string(),
            batch_repeats.to_string(),
        ];
        if let Err(e) = run_tune_stage_subprocess(
            config_path,
            &model_clone_for_stages,
            "--batch-size",
            thorough,
            "6 (batch_size)",
            &args_6,
        ) {
            tracing::warn!(error = %e, "stage 6 (batch_size) failed; continuing");
        }
    }

    // Stages 7-10 also run as ISOLATED SUBPROCESSES (they reload the model
    // too). They forward no candidate/sizing flags — their subcommands use
    // the same quick-by-default sizing, escalated together by --thorough.
    println!();
    println!("Stage 7/10: threads");
    println!("--------");
    if skip(has_threads) {
        println!("(skipped: threads winner already cached)");
    } else if let Err(e) = run_tune_stage_subprocess(
        config_path,
        &model_clone_for_stages,
        "--threads",
        thorough,
        "7 (threads)",
        &[],
    ) {
        tracing::warn!(error = %e, "stage 7 (threads) failed; continuing");
    }

    println!();
    println!("Stage 8/10: decision calibration (temperature scaling)");
    println!("--------");
    if skip(has_decision_calibration) {
        println!("(skipped: decision calibration already cached)");
    } else if let Err(e) = run_tune_stage_subprocess(
        config_path,
        &model_clone_for_stages,
        "--decision-calibrate",
        thorough,
        "8 (decision calibration)",
        &[],
    ) {
        tracing::warn!(error = %e, "stage 8 (decision calibration) failed; continuing");
    }

    println!();
    println!("Stage 9/10: MTP self-speculation (decode tok/s)");
    println!("--------");
    if skip(has_speculative_mtp) {
        println!("(skipped: speculative_mtp winner already cached)");
    } else if let Err(e) = run_tune_stage_subprocess(
        config_path,
        &model_clone_for_stages,
        "--speculative-mtp",
        thorough,
        "9 (MTP self-speculation)",
        &[],
    ) {
        tracing::warn!(error = %e, "stage 9 (MTP self-speculation) failed; continuing");
    }

    println!();
    println!("Stage 10/10: chunked SSM prefill (prefill tok/s)");
    println!("--------");
    if skip(has_ssm_prefill_chunked) {
        println!("(skipped: ssm_prefill_chunked winner already cached)");
    } else if let Err(e) = run_tune_stage_subprocess(
        config_path,
        &model_clone_for_stages,
        "--ssm-prefill-chunked",
        thorough,
        "10 (chunked SSM prefill)",
        &[],
    )
    {
        tracing::warn!(error = %e, "stage 10 (chunked SSM prefill) failed; continuing");
    }

    println!();
    println!("=== tune --all complete ===");
    println!("All winners persisted to the tuner cache; the engine will pick");
    println!("them up on next model load (when [tuning].auto_apply_* = true).");
    Ok(())
}

/// Map a `KvDtype` back to its config-string form. Pairs with
/// `parse_kv_dtype` so a winner survives a cache round-trip.
fn kv_dtype_to_str(dt: &rustllama_engine::KvDtype) -> String {
    match dt {
        rustllama_engine::KvDtype::F32 => "f32".to_string(),
        rustllama_engine::KvDtype::Q8_0 => "q8_0".to_string(),
        rustllama_engine::KvDtype::Tq(bits) => format!("tq{bits}"),
        rustllama_engine::KvDtype::Nvfp4 => "nvfp4".to_string(),
        rustllama_engine::KvDtype::Q4_0 => "q4_0".to_string(),
        rustllama_engine::KvDtype::Mxfp4 => "mxfp4".to_string(),
        rustllama_engine::KvDtype::Mxfp6 => "mxfp6".to_string(),
        rustllama_engine::KvDtype::Mxfp8 => "mxfp8".to_string(),
    }
}

/// Persist the kv_dtype winner to the tuner cache. Graceful no-op
/// when no SYCL device is visible (mirrors `persist_batch_size_winner`).
pub fn persist_kv_dtype_winner(kv_dtype: &str) -> anyhow::Result<()> {
    let Some(cache_dir) = rustllama_tuner::default_cache_dir() else {
        tracing::warn!("no tuner cache dir resolvable; skipping kv_dtype persistence");
        return Ok(());
    };
    let (key, device) = rustllama_tuner::cache_context();
    let path = persist_kv_dtype_winner_to(&cache_dir, &key, &device, kv_dtype)?;
    println!("saved kv_dtype winner to {}", path.display());
    Ok(())
}

pub fn persist_kv_dtype_winner_to(
    cache_dir: &std::path::Path,
    key: &str,
    device: &rustllama_tuner::DeviceFingerprint,
    kv_dtype: &str,
) -> anyhow::Result<std::path::PathBuf> {
    use rustllama_tuner::{cache_path_for, load_cache, save_cache, TuningResult};
    std::fs::create_dir_all(cache_dir)?;
    let mut tuning =
        load_cache(cache_dir, key)?.unwrap_or_else(|| TuningResult::empty(key.to_string(), device.clone()));
    tuning.kv_dtype = Some(kv_dtype.to_string());
    tuning.last_tuned = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| format!("{}", d.as_secs()));
    save_cache(cache_dir, &tuning)?;
    Ok(cache_path_for(cache_dir, key))
}

pub fn persist_flash_attention_winner(flash: bool) -> anyhow::Result<()> {
    let Some(cache_dir) = rustllama_tuner::default_cache_dir() else {
        tracing::warn!("no tuner cache dir resolvable; skipping flash_attention persistence");
        return Ok(());
    };
    let (key, device) = rustllama_tuner::cache_context();
    let path = persist_flash_attention_winner_to(&cache_dir, &key, &device, flash)?;
    println!("saved flash_attention winner to {}", path.display());
    Ok(())
}

pub fn persist_flash_attention_winner_to(
    cache_dir: &std::path::Path,
    key: &str,
    device: &rustllama_tuner::DeviceFingerprint,
    flash: bool,
) -> anyhow::Result<std::path::PathBuf> {
    use rustllama_tuner::{cache_path_for, load_cache, save_cache, TuningResult};
    std::fs::create_dir_all(cache_dir)?;
    let mut tuning =
        load_cache(cache_dir, key)?.unwrap_or_else(|| TuningResult::empty(key.to_string(), device.clone()));
    tuning.flash_attention = Some(flash);
    tuning.last_tuned = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| format!("{}", d.as_secs()));
    save_cache(cache_dir, &tuning)?;
    Ok(cache_path_for(cache_dir, key))
}

pub fn persist_speculative_mtp_winner(v: bool) -> anyhow::Result<()> {
    let Some(cache_dir) = rustllama_tuner::default_cache_dir() else {
        tracing::warn!("no tuner cache dir resolvable; skipping speculative_mtp persistence");
        return Ok(());
    };
    let (key, device) = rustllama_tuner::cache_context();
    let path = persist_speculative_mtp_winner_to(&cache_dir, &key, &device, v)?;
    println!("saved speculative_mtp winner to {}", path.display());
    Ok(())
}

pub fn persist_speculative_mtp_winner_to(
    cache_dir: &std::path::Path,
    key: &str,
    device: &rustllama_tuner::DeviceFingerprint,
    v: bool,
) -> anyhow::Result<std::path::PathBuf> {
    use rustllama_tuner::{cache_path_for, load_cache, save_cache, TuningResult};
    std::fs::create_dir_all(cache_dir)?;
    let mut tuning =
        load_cache(cache_dir, key)?.unwrap_or_else(|| TuningResult::empty(key.to_string(), device.clone()));
    tuning.speculative_mtp = Some(v);
    tuning.last_tuned = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| format!("{}", d.as_secs()));
    save_cache(cache_dir, &tuning)?;
    Ok(cache_path_for(cache_dir, key))
}

pub fn persist_ssm_prefill_chunked_winner(v: bool) -> anyhow::Result<()> {
    let Some(cache_dir) = rustllama_tuner::default_cache_dir() else {
        tracing::warn!("no tuner cache dir resolvable; skipping ssm_prefill_chunked persistence");
        return Ok(());
    };
    let (key, device) = rustllama_tuner::cache_context();
    let path = persist_ssm_prefill_chunked_winner_to(&cache_dir, &key, &device, v)?;
    println!("saved ssm_prefill_chunked winner to {}", path.display());
    Ok(())
}

pub fn persist_ssm_prefill_chunked_winner_to(
    cache_dir: &std::path::Path,
    key: &str,
    device: &rustllama_tuner::DeviceFingerprint,
    v: bool,
) -> anyhow::Result<std::path::PathBuf> {
    use rustllama_tuner::{cache_path_for, load_cache, save_cache, TuningResult};
    std::fs::create_dir_all(cache_dir)?;
    let mut tuning =
        load_cache(cache_dir, key)?.unwrap_or_else(|| TuningResult::empty(key.to_string(), device.clone()));
    tuning.ssm_prefill_chunked = Some(v);
    tuning.last_tuned = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| format!("{}", d.as_secs()));
    save_cache(cache_dir, &tuning)?;
    Ok(cache_path_for(cache_dir, key))
}

pub fn persist_kv_cache_layout_winner(layout: &str) -> anyhow::Result<()> {
    let Some(cache_dir) = rustllama_tuner::default_cache_dir() else {
        tracing::warn!("no tuner cache dir resolvable; skipping kv_cache_layout persistence");
        return Ok(());
    };
    let (key, device) = rustllama_tuner::cache_context();
    let path = persist_kv_cache_layout_winner_to(&cache_dir, &key, &device, layout)?;
    println!("saved kv_cache_layout winner to {}", path.display());
    Ok(())
}

pub fn persist_kv_cache_layout_winner_to(
    cache_dir: &std::path::Path,
    key: &str,
    device: &rustllama_tuner::DeviceFingerprint,
    layout: &str,
) -> anyhow::Result<std::path::PathBuf> {
    use rustllama_tuner::{cache_path_for, load_cache, save_cache, TuningResult};
    std::fs::create_dir_all(cache_dir)?;
    let mut tuning =
        load_cache(cache_dir, key)?.unwrap_or_else(|| TuningResult::empty(key.to_string(), device.clone()));
    tuning.kv_cache_layout = Some(layout.to_string());
    tuning.last_tuned = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| format!("{}", d.as_secs()));
    save_cache(cache_dir, &tuning)?;
    Ok(cache_path_for(cache_dir, key))
}

// ============================================================
// E2 autotuner candidates: flash_kv_min + prefix_cache_max_snapshots
// ============================================================

pub fn persist_flash_kv_min_winner(kv_min: u32) -> anyhow::Result<()> {
    let Some(cache_dir) = rustllama_tuner::default_cache_dir() else {
        tracing::warn!("no tuner cache dir resolvable; skipping flash_kv_min persistence");
        return Ok(());
    };
    let (key, device) = rustllama_tuner::cache_context();
    let path = persist_flash_kv_min_winner_to(&cache_dir, &key, &device, kv_min)?;
    println!("saved flash_kv_min winner to {}", path.display());
    Ok(())
}

pub fn persist_flash_kv_min_winner_to(
    cache_dir: &std::path::Path,
    key: &str,
    device: &rustllama_tuner::DeviceFingerprint,
    kv_min: u32,
) -> anyhow::Result<std::path::PathBuf> {
    use rustllama_tuner::{cache_path_for, load_cache, save_cache, TuningResult};
    std::fs::create_dir_all(cache_dir)?;
    let mut tuning =
        load_cache(cache_dir, key)?.unwrap_or_else(|| TuningResult::empty(key.to_string(), device.clone()));
    tuning.flash_kv_min = Some(kv_min);
    tuning.last_tuned = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| format!("{}", d.as_secs()));
    save_cache(cache_dir, &tuning)?;
    Ok(cache_path_for(cache_dir, key))
}

pub fn persist_prefix_snapshots_winner(depth: u32) -> anyhow::Result<()> {
    let Some(cache_dir) = rustllama_tuner::default_cache_dir() else {
        tracing::warn!(
            "no tuner cache dir resolvable; skipping prefix_cache_max_snapshots persistence"
        );
        return Ok(());
    };
    let (key, device) = rustllama_tuner::cache_context();
    let path = persist_prefix_snapshots_winner_to(&cache_dir, &key, &device, depth)?;
    println!(
        "saved prefix_cache_max_snapshots winner to {}",
        path.display()
    );
    Ok(())
}

pub fn persist_prefix_snapshots_winner_to(
    cache_dir: &std::path::Path,
    key: &str,
    device: &rustllama_tuner::DeviceFingerprint,
    depth: u32,
) -> anyhow::Result<std::path::PathBuf> {
    use rustllama_tuner::{cache_path_for, load_cache, save_cache, TuningResult};
    std::fs::create_dir_all(cache_dir)?;
    let mut tuning =
        load_cache(cache_dir, key)?.unwrap_or_else(|| TuningResult::empty(key.to_string(), device.clone()));
    tuning.prefix_cache_max_snapshots = Some(depth);
    tuning.last_tuned = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| format!("{}", d.as_secs()));
    save_cache(cache_dir, &tuning)?;
    Ok(cache_path_for(cache_dir, key))
}

/// Consumer-side hook for flash_attention_kv_min. Returns the
/// `config_default` when auto-apply is off OR the cache lacks an
/// entry; otherwise returns the cached winner.
pub fn flash_kv_min_from_cache_or_default(config_default: u32, auto_apply: bool) -> u32 {
    if !auto_apply {
        return config_default;
    }
    let Some(cache_dir) = rustllama_tuner::default_cache_dir() else {
        return config_default;
    };
    let key = rustllama_tuner::system_fingerprint();
    flash_kv_min_from_cache_or_default_to(&cache_dir, &key, config_default)
}

pub fn flash_kv_min_from_cache_or_default_to(
    cache_dir: &std::path::Path,
    key: &str,
    config_default: u32,
) -> u32 {
    rustllama_tuner::load_cache(cache_dir, key)
        .ok()
        .flatten()
        .and_then(|t| t.flash_kv_min)
        .unwrap_or(config_default)
}

/// Consumer-side hook for prefix_cache_max_snapshots.
pub fn prefix_snapshots_from_cache_or_default(config_default: u32, auto_apply: bool) -> u32 {
    if !auto_apply {
        return config_default;
    }
    let Some(cache_dir) = rustllama_tuner::default_cache_dir() else {
        return config_default;
    };
    let key = rustllama_tuner::system_fingerprint();
    prefix_snapshots_from_cache_or_default_to(&cache_dir, &key, config_default)
}

pub fn prefix_snapshots_from_cache_or_default_to(
    cache_dir: &std::path::Path,
    key: &str,
    config_default: u32,
) -> u32 {
    rustllama_tuner::load_cache(cache_dir, key)
        .ok()
        .flatten()
        .and_then(|t| t.prefix_cache_max_snapshots)
        .unwrap_or(config_default)
}

pub fn persist_kv_page_size_winner(page_size: u32) -> anyhow::Result<()> {
    let Some(cache_dir) = rustllama_tuner::default_cache_dir() else {
        tracing::warn!("no tuner cache dir resolvable; skipping kv_page_size persistence");
        return Ok(());
    };
    let (key, device) = rustllama_tuner::cache_context();
    let path = persist_kv_page_size_winner_to(&cache_dir, &key, &device, page_size)?;
    println!("saved kv_page_size winner to {}", path.display());
    Ok(())
}

pub fn persist_kv_page_size_winner_to(
    cache_dir: &std::path::Path,
    key: &str,
    device: &rustllama_tuner::DeviceFingerprint,
    page_size: u32,
) -> anyhow::Result<std::path::PathBuf> {
    use rustllama_tuner::{cache_path_for, load_cache, save_cache, TuningResult};
    std::fs::create_dir_all(cache_dir)?;
    let mut tuning =
        load_cache(cache_dir, key)?.unwrap_or_else(|| TuningResult::empty(key.to_string(), device.clone()));
    tuning.kv_page_size = Some(page_size);
    tuning.last_tuned = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| format!("{}", d.as_secs()));
    save_cache(cache_dir, &tuning)?;
    Ok(cache_path_for(cache_dir, key))
}

/// Consumer-side hook for AUTO MoE placement: the cached
/// `(placement, gpu_split_permille)` winner when `auto_apply` is on AND a
/// sweep has run; `None` otherwise (caller falls back to `"uniform"` +
/// split 0). MoE placement is always auto — there is no config key.
pub fn moe_placement_from_cache_or_default(auto_apply: bool) -> Option<(String, u32)> {
    if !auto_apply {
        return None;
    }
    let cache_dir = rustllama_tuner::default_cache_dir()?;
    let key = rustllama_tuner::system_fingerprint();
    moe_placement_from_cache_or_default_to(&cache_dir, &key)
}

pub fn moe_placement_from_cache_or_default_to(
    cache_dir: &std::path::Path,
    key: &str,
) -> Option<(String, u32)> {
    let t = rustllama_tuner::load_cache(cache_dir, key).ok().flatten()?;
    let placement = t.moe_placement?;
    Some((placement, t.moe_gpu_split_permille.unwrap_or(0)))
}

pub fn persist_moe_placement_winner(placement: &str, split_permille: u32) -> anyhow::Result<()> {
    let Some(cache_dir) = rustllama_tuner::default_cache_dir() else {
        tracing::warn!("no tuner cache dir resolvable; skipping moe_placement persistence");
        return Ok(());
    };
    let (key, device) = rustllama_tuner::cache_context();
    let path = persist_moe_placement_winner_to(&cache_dir, &key, &device, placement, split_permille)?;
    println!("saved moe_placement winner to {}", path.display());
    Ok(())
}

pub fn persist_moe_placement_winner_to(
    cache_dir: &std::path::Path,
    key: &str,
    device: &rustllama_tuner::DeviceFingerprint,
    placement: &str,
    split_permille: u32,
) -> anyhow::Result<std::path::PathBuf> {
    use rustllama_tuner::{cache_path_for, load_cache, save_cache, TuningResult};
    std::fs::create_dir_all(cache_dir)?;
    let mut tuning =
        load_cache(cache_dir, key)?.unwrap_or_else(|| TuningResult::empty(key.to_string(), device.clone()));
    tuning.moe_placement = Some(placement.to_string());
    tuning.moe_gpu_split_permille = Some(split_permille);
    tuning.last_tuned = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| format!("{}", d.as_secs()));
    save_cache(cache_dir, &tuning)?;
    Ok(cache_path_for(cache_dir, key))
}

/// Consumer-side hook for kv_page_size — read the cached winner
/// when `auto_apply` is on AND a cache entry exists; otherwise
/// return the config default.
pub fn kv_page_size_from_cache_or_default(config_default: u32, auto_apply: bool) -> u32 {
    if !auto_apply {
        return config_default;
    }
    let Some(cache_dir) = rustllama_tuner::default_cache_dir() else {
        return config_default;
    };
    let key = rustllama_tuner::system_fingerprint();
    kv_page_size_from_cache_or_default_to(&cache_dir, &key, config_default)
}

pub fn kv_page_size_from_cache_or_default_to(
    cache_dir: &std::path::Path,
    key: &str,
    config_default: u32,
) -> u32 {
    rustllama_tuner::load_cache(cache_dir, key)
        .ok()
        .flatten()
        .and_then(|t| t.kv_page_size)
        .unwrap_or(config_default)
}

pub fn persist_flash_v3_kv_tile_winner(kv_tile: u32) -> anyhow::Result<()> {
    let Some(cache_dir) = rustllama_tuner::default_cache_dir() else {
        tracing::warn!("no tuner cache dir resolvable; skipping flash_v3_kv_tile persistence");
        return Ok(());
    };
    let (key, device) = rustllama_tuner::cache_context();
    let path = persist_flash_v3_kv_tile_winner_to(&cache_dir, &key, &device, kv_tile)?;
    println!("saved flash_v3_kv_tile winner to {}", path.display());
    Ok(())
}

pub fn persist_flash_v3_kv_tile_winner_to(
    cache_dir: &std::path::Path,
    key: &str,
    device: &rustllama_tuner::DeviceFingerprint,
    kv_tile: u32,
) -> anyhow::Result<std::path::PathBuf> {
    use rustllama_tuner::{cache_path_for, load_cache, save_cache, TuningResult};
    std::fs::create_dir_all(cache_dir)?;
    let mut tuning =
        load_cache(cache_dir, key)?.unwrap_or_else(|| TuningResult::empty(key.to_string(), device.clone()));
    tuning.flash_v3_kv_tile = Some(kv_tile);
    tuning.last_tuned = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| format!("{}", d.as_secs()));
    save_cache(cache_dir, &tuning)?;
    Ok(cache_path_for(cache_dir, key))
}

/// Consumer-side hook for the flash-attn-v3 KV_TILE. Returns the
/// cached winner if auto-apply is on AND a cache entry exists;
/// otherwise `None` so the kernel uses its built-in default (32).
pub fn flash_v3_kv_tile_from_cache_or_default(auto_apply: bool) -> Option<u32> {
    if !auto_apply {
        return None;
    }
    let cache_dir = rustllama_tuner::default_cache_dir()?;
    let key = rustllama_tuner::system_fingerprint();
    flash_v3_kv_tile_from_cache_or_default_to(&cache_dir, &key)
}

pub fn flash_v3_kv_tile_from_cache_or_default_to(
    cache_dir: &std::path::Path,
    key: &str,
) -> Option<u32> {
    rustllama_tuner::load_cache(cache_dir, key)
        .ok()
        .flatten()
        .and_then(|t| t.flash_v3_kv_tile)
}

/// Consumer-side hook for kv_dtype: look up the cached winner and
/// return it as a config-string when auto-apply is on AND a cache
/// entry exists. Otherwise returns `None` so the caller falls back
/// to the config value.
pub fn kv_dtype_from_cache_or_default(auto_apply: bool) -> Option<String> {
    if !auto_apply {
        return None;
    }
    let cache_dir = rustllama_tuner::default_cache_dir()?;
    let key = rustllama_tuner::system_fingerprint();
    kv_dtype_from_cache_or_default_to(&cache_dir, &key)
}

pub fn kv_dtype_from_cache_or_default_to(
    cache_dir: &std::path::Path,
    key: &str,
) -> Option<String> {
    rustllama_tuner::load_cache(cache_dir, key)
        .ok()
        .flatten()
        .and_then(|t| t.kv_dtype)
}

pub fn flash_attention_from_cache_or_default(auto_apply: bool) -> Option<bool> {
    if !auto_apply {
        return None;
    }
    let cache_dir = rustllama_tuner::default_cache_dir()?;
    let key = rustllama_tuner::system_fingerprint();
    flash_attention_from_cache_or_default_to(&cache_dir, &key)
}

pub fn flash_attention_from_cache_or_default_to(
    cache_dir: &std::path::Path,
    key: &str,
) -> Option<bool> {
    rustllama_tuner::load_cache(cache_dir, key)
        .ok()
        .flatten()
        .and_then(|t| t.flash_attention)
}

pub fn speculative_mtp_from_cache_or_default(auto_apply: bool) -> Option<bool> {
    if !auto_apply {
        return None;
    }
    let cache_dir = rustllama_tuner::default_cache_dir()?;
    let key = rustllama_tuner::system_fingerprint();
    speculative_mtp_from_cache_or_default_to(&cache_dir, &key)
}

pub fn speculative_mtp_from_cache_or_default_to(
    cache_dir: &std::path::Path,
    key: &str,
) -> Option<bool> {
    rustllama_tuner::load_cache(cache_dir, key)
        .ok()
        .flatten()
        .and_then(|t| t.speculative_mtp)
}

pub fn ssm_prefill_chunked_from_cache_or_default(auto_apply: bool) -> Option<bool> {
    if !auto_apply {
        return None;
    }
    let cache_dir = rustllama_tuner::default_cache_dir()?;
    let key = rustllama_tuner::system_fingerprint();
    ssm_prefill_chunked_from_cache_or_default_to(&cache_dir, &key)
}

pub fn ssm_prefill_chunked_from_cache_or_default_to(
    cache_dir: &std::path::Path,
    key: &str,
) -> Option<bool> {
    rustllama_tuner::load_cache(cache_dir, key)
        .ok()
        .flatten()
        .and_then(|t| t.ssm_prefill_chunked)
}

pub fn kv_cache_layout_from_cache_or_default(auto_apply: bool) -> Option<String> {
    if !auto_apply {
        return None;
    }
    let cache_dir = rustllama_tuner::default_cache_dir()?;
    let key = rustllama_tuner::system_fingerprint();
    kv_cache_layout_from_cache_or_default_to(&cache_dir, &key)
}

pub fn kv_cache_layout_from_cache_or_default_to(
    cache_dir: &std::path::Path,
    key: &str,
) -> Option<String> {
    rustllama_tuner::load_cache(cache_dir, key)
        .ok()
        .flatten()
        .and_then(|t| t.kv_cache_layout)
}

/// Update the tuner cache's `placement[model_key]` slot with the
/// measured winner. Convenience wrapper that resolves the default
/// cache dir + auto-detects the SYCL device 0 fingerprint, then
/// dispatches to [`persist_placement_winner_to`]. Returns `Ok(())`
/// even when no SYCL device is detected — without a fingerprint
/// there's no cache file to write to, so this is a graceful no-op
/// rather than a hard error.
pub fn persist_placement_winner(model_key: &str, n_gpu_layers: u32) -> anyhow::Result<()> {
    let Some(cache_dir) = rustllama_tuner::default_cache_dir() else {
        tracing::warn!("no tuner cache dir resolvable; skipping placement persistence");
        return Ok(());
    };
    let (key, device) = rustllama_tuner::cache_context();
    let path = persist_placement_winner_to(&cache_dir, &key, &device, model_key, n_gpu_layers)?;
    println!("saved placement winner to {}", path.display());
    Ok(())
}

/// Lower-level placement persister: writes directly to `cache_dir`
/// for the supplied `fp`. Returns the path written to. Exposed
/// separately so tests can drive it against a tempdir + synthetic
/// fingerprint without needing a real SYCL device, and so future
/// callers (GUI, scripted tuning runs) can target an alternate
/// cache root.
pub fn persist_placement_winner_to(
    cache_dir: &std::path::Path,
    key: &str,
    device: &rustllama_tuner::DeviceFingerprint,
    model_key: &str,
    n_gpu_layers: u32,
) -> anyhow::Result<std::path::PathBuf> {
    use rustllama_tuner::{cache_path_for, load_cache, save_cache, PlacementPlan, TuningResult};
    std::fs::create_dir_all(cache_dir)?;
    let mut tuning =
        load_cache(cache_dir, key)?.unwrap_or_else(|| TuningResult::empty(key.to_string(), device.clone()));
    tuning.placement.insert(
        model_key.to_string(),
        PlacementPlan {
            n_gpu_layers,
            overrides: Vec::new(),
        },
    );
    tuning.last_tuned = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| format!("{}", d.as_secs()));
    save_cache(cache_dir, &tuning)?;
    Ok(cache_path_for(cache_dir, key))
}

/// Update the tuner cache's `batch_size` slot. Convenience wrapper
/// around [`persist_batch_size_winner_to`]; same graceful-on-no-
/// device behavior as [`persist_placement_winner`].
///
/// Batch size is not per-model — it's a per-device characteristic
/// of prefill scheduling — so it lives at the top of the cache,
/// not under a model key.
pub fn persist_batch_size_winner(batch_size: u32) -> anyhow::Result<()> {
    let Some(cache_dir) = rustllama_tuner::default_cache_dir() else {
        tracing::warn!("no tuner cache dir resolvable; skipping batch_size persistence");
        return Ok(());
    };
    let (key, device) = rustllama_tuner::cache_context();
    let path = persist_batch_size_winner_to(&cache_dir, &key, &device, batch_size)?;
    println!("saved batch_size winner to {}", path.display());
    Ok(())
}

/// Lower-level batch-size persister; see
/// [`persist_placement_winner_to`] for the dispatch shape.
pub fn persist_batch_size_winner_to(
    cache_dir: &std::path::Path,
    key: &str,
    device: &rustllama_tuner::DeviceFingerprint,
    batch_size: u32,
) -> anyhow::Result<std::path::PathBuf> {
    use rustllama_tuner::{cache_path_for, load_cache, save_cache, TuningResult};
    std::fs::create_dir_all(cache_dir)?;
    let mut tuning =
        load_cache(cache_dir, key)?.unwrap_or_else(|| TuningResult::empty(key.to_string(), device.clone()));
    tuning.batch_size = Some(batch_size);
    tuning.last_tuned = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()
        .map(|d| format!("{}", d.as_secs()));
    save_cache(cache_dir, &tuning)?;
    Ok(cache_path_for(cache_dir, key))
}

/// Consumer-side hook for the placement axis: look up the cached
/// `n_gpu_layers` winner for `(device, model_key)` and return it
/// if the auto-apply flag is on AND the cache has an entry;
/// otherwise return `config_default` unchanged.
///
/// `auto_apply` is the `[tuning].auto_apply_placement` flag — when
/// `false`, the function returns `config_default` without touching
/// the cache. When `true`, a cache miss also returns `config_default`
/// (so a fresh install with no tuner history behaves identically to
/// the legacy config-only path).
///
/// Resolves the default cache dir + auto-detects SYCL device 0.
/// Returns the config default verbatim when either resolution
/// fails — no warning logs, this is the silent fallback path that
/// every non-SYCL build hits.
pub fn placement_from_cache_or_default(
    model_key: &str,
    config_default: u32,
    auto_apply: bool,
) -> u32 {
    if !auto_apply {
        return config_default;
    }
    let Some(cache_dir) = rustllama_tuner::default_cache_dir() else {
        return config_default;
    };
    let key = rustllama_tuner::system_fingerprint();
    placement_from_cache_or_default_to(&cache_dir, &key, model_key, config_default)
}

/// Lower-level placement consumer; takes an explicit cache dir +
/// fingerprint so tests + alternate-cache flows can drive it
/// directly without needing a real SYCL device.
pub fn placement_from_cache_or_default_to(
    cache_dir: &std::path::Path,
    key: &str,
    model_key: &str,
    config_default: u32,
) -> u32 {
    match rustllama_tuner::load_cache(cache_dir, key) {
        Ok(Some(tuning)) => tuning
            .placement
            .get(model_key)
            .map(|p| p.n_gpu_layers)
            .unwrap_or(config_default),
        _ => config_default,
    }
}

/// Peek the tuner cache for a measured `n_gpu_layers` winner for
/// `(device 0, model_key)`. Returns `None` when there is no entry — an
/// unseen model — which is what triggers the first-load auto-tune probe.
/// Unlike [`placement_from_cache_or_default`], this distinguishes "no
/// cache entry" from "cached value happens to equal the config default",
/// a distinction the `serve` placement precedence needs.
pub fn placement_cache_lookup(model_key: &str) -> Option<u32> {
    let cache_dir = rustllama_tuner::default_cache_dir()?;
    let key = rustllama_tuner::system_fingerprint();
    match rustllama_tuner::load_cache(&cache_dir, &key) {
        Ok(Some(tuning)) => tuning.placement.get(model_key).map(|p| p.n_gpu_layers),
        _ => None,
    }
}

/// Consumer-side hook for the batch-size axis. Same shape as
/// [`placement_from_cache_or_default`] but for the per-device
/// `batch_size` slot — not keyed by model, since the optimal
/// prefill chunk is a device characteristic, not a model one.
pub fn batch_size_from_cache_or_default(config_default: u32, auto_apply: bool) -> u32 {
    if !auto_apply {
        return config_default;
    }
    let Some(cache_dir) = rustllama_tuner::default_cache_dir() else {
        return config_default;
    };
    let key = rustllama_tuner::system_fingerprint();
    batch_size_from_cache_or_default_to(&cache_dir, &key, config_default)
}

/// Lower-level batch-size consumer; see
/// [`placement_from_cache_or_default_to`] for the contract.
pub fn batch_size_from_cache_or_default_to(
    cache_dir: &std::path::Path,
    key: &str,
    config_default: u32,
) -> u32 {
    match rustllama_tuner::load_cache(cache_dir, key) {
        Ok(Some(tuning)) => tuning.batch_size.unwrap_or(config_default),
        _ => config_default,
    }
}

/// CLI parity with the GUI Status page's "Tuner cache" panel:
/// print the per-device tuner state in human-readable form.
/// Returns Ok(()) regardless of whether a SYCL device is visible —
/// the no-device path prints a clear "rebuild with --features
/// sycl" hint rather than erroring.
///
/// `pub` so the dispatcher in `run()` can call it; also lets
/// integration tests drive the formatter against a tempdir-backed
/// cache via the lower-level helpers.
pub fn cmd_tuning_show(config_path: &std::path::Path) -> anyhow::Result<()> {
    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    println!("rustllama tuning show");
    println!("  config              = {}", config_path.display());
    println!(
        "  auto_apply_placement = {}",
        cfg.tuning.auto_apply_placement
    );
    println!(
        "  auto_apply_batch_size= {}",
        cfg.tuning.auto_apply_batch_size
    );
    println!();

    let Some(cache_dir) = rustllama_tuner::default_cache_dir() else {
        println!(
            "no tuner cache directory resolvable — this is unusual; check that \
             %LOCALAPPDATA% / $XDG_DATA_HOME is writable, or set \
             [tuning].cache_dir in the config to an explicit path."
        );
        return Ok(());
    };
    println!("  cache dir           = {}", cache_dir.display());

    // The cache is keyed by the whole-system fingerprint now, so it resolves
    // on SYCL / CUDA / CPU hosts alike. The SYCL device (if any) is only shown
    // for context — its absence no longer hides the cache.
    let key = rustllama_tuner::system_fingerprint();
    match rustllama_tuner::fingerprint_active_device() {
        Some(fp) => {
            println!("  device              = {} ({})", fp.name, fp.slug());
            println!("  driver              = {}", fp.driver_ver);
            println!("  vram                = {} MiB", fp.vram_mb);
        }
        None => {
            println!("  device              = (no SYCL device — CUDA/CPU host)");
        }
    }
    println!("  cache key           = {key}");
    let cache_path = rustllama_tuner::cache_path_for(&cache_dir, &key);
    println!("  cache file          = {}", cache_path.display());

    let tuning = match rustllama_tuner::load_cache(&cache_dir, &key)? {
        Some(t) => t,
        None => {
            println!();
            println!("cache file not found — run `rustllama tune` to populate it:");
            println!("  rustllama tune                                # kernel-LWS sweep");
            println!(
                "  rustllama tune --placement --measure          # placement winner per model"
            );
            println!("  rustllama tune --batch-size                   # prefill chunk-size winner");
            return Ok(());
        }
    };

    println!();
    println!(
        "  last tuned          = {}",
        tuning.last_tuned.as_deref().unwrap_or("(never)")
    );
    println!("  kernel-LWS shapes   = {}", tuning.kernels.len());
    println!();

    println!("placement winners ({}):", tuning.placement.len());
    if tuning.placement.is_empty() {
        println!("  (none — run `rustllama tune --placement --measure` for a model to populate)");
    } else {
        for (model_key, plan) in &tuning.placement {
            println!("  {model_key:<40}  n_gpu_layers = {}", plan.n_gpu_layers);
        }
    }
    println!();
    println!("batch_size winner:");
    match tuning.batch_size {
        Some(b) => println!("  {b}"),
        None => {
            println!("  (none — run `rustllama tune --batch-size` to populate)")
        }
    }
    Ok(())
}

/// Drive the engine once per batch-size candidate, time the prefill
/// phase of a fixed-length synthetic prompt, pick the candidate with
/// the highest prefill tok/s.
///
/// Loads the model exactly once — `set_prefill_chunk_size` is a
/// cheap setter that takes effect on the next generate call.
///
/// Returns `Some(batch_size)` of the highest-throughput candidate,
/// or `None` if every candidate failed.
///
/// `pub` so integration tests can exercise the measurement loop
/// against synth fixtures.
pub fn measure_batch_size_candidates(
    model_path: &std::path::Path,
    candidates: &[usize],
    prompt_tokens: u32,
    repeats: u32,
    cfg: &rustllama_config::Config,
) -> anyhow::Result<Option<usize>> {
    use rustllama_engine::measurement::{
        measure_batch_size_candidates as engine_measure_batch, MeasurementConfig,
    };

    let (k, v) = cfg.inference.resolved_kv_dtypes();
    if k != v {
        tracing::warn!(
            k = k,
            v = v,
            "split K/V dtypes configured; batch-size measurement uses K's dtype \
             (engine storage couples K/V today)"
        );
    }
    let kv_dtype = parse_kv_dtype(&k.to_string())?;
    let m_cfg = MeasurementConfig {
        kv_dtype,
        kv_cache_layout: cfg.inference.kv_cache_layout.clone(),
        flash_attention: cfg.inference.flash_attention,
        n_gpu_layers: cfg.inference.n_gpu_layers,
    };

    let report = engine_measure_batch(model_path, candidates, prompt_tokens, repeats, &m_cfg)?;

    println!("  model loaded in {:.1} ms", report.load_ms);
    println!();
    println!(
        "  {:>5}  {:>9}  {:>13}  {:>13}  notes",
        "batch", "warmup", "median pf tps", "p100 pf tps"
    );
    for c in &report.candidates {
        match (c.median_tps, c.max_tps, c.error.as_ref()) {
            (Some(med), Some(max), _) => {
                let star = if report.winner == Some(c.batch_size) && med >= report.winner_tps {
                    "  ★ best"
                } else {
                    ""
                };
                println!(
                    "  {:>5}  {:>7.0} ms  {med:>9.1} tps  {max:>9.1} tps {star}",
                    c.batch_size, c.warmup_ms,
                );
            }
            (_, _, Some(err)) => {
                println!(
                    "  {:>5}  {:>7.0} ms  {:>13}  {:>13}  {err}",
                    c.batch_size, c.warmup_ms, "—", "—"
                );
            }
            _ => {}
        }
    }
    println!();
    match report.winner {
        Some(b) => println!(
            "  → measured winner: batch_size = {b} at {:.1} prefill tok/s median",
            report.winner_tps
        ),
        None => println!("  → no candidate completed"),
    }
    Ok(report.winner)
}

/// Read just enough GGUF metadata to populate `ModelDims` + infer
/// the weight quant. Mirrors the inspector's metadata-key extraction
/// (architecture-qualified key with bare-key fallback for older
/// converters) but keeps the dependency surface tiny.
///
/// Quant detection scans the tensor table for the dominant
/// quantized weight format (Q4_K, Q5_K, Q8_0, F16) — matches what
/// llama.cpp + rustllama_hub's model-card path use to label the
/// model. Tensors at other dtypes (norms in F32, etc.) don't count
/// against the dominant pick.
///
/// `pub` so integration tests can exercise the metadata-extraction
/// path against synth fixtures without running the whole CLI.
pub fn read_dims_and_quant_from_gguf(
    path: &std::path::Path,
) -> anyhow::Result<(
    rustllama_tuner::placement::ModelDims,
    rustllama_tuner::placement::WeightQuant,
)> {
    use rustllama_gguf::{GgmlType, Gguf, MetadataValue};
    use rustllama_tuner::placement::{ModelDims, WeightQuant};

    let gguf = Gguf::open(path).map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))?;
    let arch = gguf
        .architecture()
        .ok_or_else(|| anyhow::anyhow!("GGUF has no general.architecture metadata"))?
        .to_string();

    let key = |k: &str| -> Option<u32> {
        let full = format!("{arch}.{k}");
        gguf.metadata_get(&full)
            .or_else(|| gguf.metadata_get(k))
            .and_then(MetadataValue::as_u32)
    };
    let n_layers =
        key("block_count").ok_or_else(|| anyhow::anyhow!("GGUF missing `{arch}.block_count`"))?;
    let d_model = key("embedding_length")
        .ok_or_else(|| anyhow::anyhow!("GGUF missing `{arch}.embedding_length`"))?;
    let d_ff = key("feed_forward_length")
        .ok_or_else(|| anyhow::anyhow!("GGUF missing `{arch}.feed_forward_length`"))?;
    let n_heads = key("attention.head_count")
        .ok_or_else(|| anyhow::anyhow!("GGUF missing `{arch}.attention.head_count`"))?;
    // GQA off → kv_heads == heads. Standard llama-family default.
    let n_kv_heads = key("attention.head_count_kv").unwrap_or(n_heads);
    // head_dim is implied as d_model / n_heads when the explicit key
    // isn't present — historic GGUFs predate the explicit field.
    let head_dim = key("attention.key_length")
        .or_else(|| key("rope.dimension_count"))
        .unwrap_or(d_model / n_heads.max(1));
    let vocab_size = gguf
        .metadata_get("tokenizer.ggml.tokens")
        .and_then(|v| match v {
            MetadataValue::Array(a) => Some(a.len() as u32),
            _ => None,
        })
        .ok_or_else(|| anyhow::anyhow!("GGUF missing tokenizer.ggml.tokens"))?;

    // MoE expert counts — zero on dense GGUFs.
    let n_experts = key("expert_count").unwrap_or(0);
    let n_experts_used = if n_experts >= 2 {
        key("expert_used_count").unwrap_or(1)
    } else {
        0
    };
    let n_experts_shared = if n_experts >= 2 {
        key("expert_shared_count").unwrap_or(0)
    } else {
        0
    };

    // Quant detection: tally bytes per dtype across the tensor
    // table, pick the dominant quantized one. Norms / biases in F32
    // are tiny and don't sway the pick. GgmlType doesn't implement
    // Ord (deliberate — enum ordering would be a footgun) so we
    // tally into a Vec keyed by the type's canonical name string.
    let mut bytes_by: Vec<(GgmlType, u64)> = Vec::new();
    for t in gguf.tensors() {
        match bytes_by.iter_mut().find(|(d, _)| *d == t.dtype) {
            Some(entry) => entry.1 += t.byte_size,
            None => bytes_by.push((t.dtype, t.byte_size)),
        }
    }
    let dominant = bytes_by
        .iter()
        .max_by_key(|(_, b)| *b)
        .map(|(t, _)| *t)
        .unwrap_or(GgmlType::F16);
    let quant = match dominant {
        GgmlType::Q4_K => WeightQuant::Q4_K_M,
        GgmlType::Q5_K => WeightQuant::Q5_K_M,
        GgmlType::Q8_0 => WeightQuant::Q8_0,
        GgmlType::F16 | GgmlType::Bf16 => WeightQuant::F16,
        // K-quants below 4-bit + IQ formats aren't in the tuner's
        // pricing table (v1 only models the four locked formats).
        // Approximate to the nearest priced quant so the user still
        // gets a sweep — log a warning so they know the estimate
        // is an over-approximation.
        other => {
            tracing::warn!(
                dtype = ?other,
                "tuner placement: dominant quant not in v1 pricing table — \
                 approximating as Q4_K_M (slight VRAM over-estimate)"
            );
            WeightQuant::Q4_K_M
        }
    };

    Ok((
        ModelDims {
            n_layers,
            d_model,
            d_ff,
            n_heads,
            n_kv_heads,
            head_dim,
            vocab_size,
            n_experts,
            n_experts_used,
            n_experts_shared,
        },
        quant,
    ))
}

/// fields without writing to disk. The synthetic prompt is a
/// monotonic token sequence (1, 2, 3, …) clipped to the vocab — any
/// real model has a vocab in the thousands, so this exercises the
/// full pipeline without depending on a tokenizer.
#[allow(clippy::too_many_arguments)]
fn bench(
    config_path: &std::path::Path,
    profile: Option<&str>,
    model_override: Option<&str>,
    flash_override: Option<bool>,
    kv_dtype_override: Option<&str>,
    ctx_size_override: Option<usize>,
    prompt_tokens: u32,
    decode_tokens: u32,
    repeats: u32,
) -> anyhow::Result<()> {
    use rustllama_engine::{CpuEngine, SamplingParams};

    let cfg = rustllama_config::load_with_profile(config_path, profile)?;
    let model_path =
        effective_model_path(model_override.map(str::to_string), &cfg, config_path)?;
    let ctx_size = ctx_size_override.unwrap_or(cfg.inference.ctx_size as usize);
    // Override-or-config K/V dtype. When K and V differ in config,
    // bench uses K's dtype (matches engine storage which couples
    // K/V today) and warns. `--kv-dtype` overrides both.
    let kv_dtype_str = match kv_dtype_override {
        Some(s) => s.to_string(),
        None => {
            let (k, v) = cfg.inference.resolved_kv_dtypes();
            if k != v {
                tracing::warn!(
                    k = k,
                    v = v,
                    "split K/V dtypes configured; bench uses K's dtype \
                     (engine storage couples K/V today)"
                );
            }
            k.to_string()
        }
    };
    let kv_dtype = parse_kv_dtype(&kv_dtype_str)?;
    let flash_enabled = flash_override.unwrap_or(cfg.inference.flash_attention);

    println!("rustllama bench");
    println!("  model         = {}", model_path.display());
    println!("  ctx_size      = {ctx_size}");
    println!("  kv_dtype      = {kv_dtype_str}");
    println!("  flash         = {flash_enabled}");
    println!("  prompt_tokens = {prompt_tokens}");
    println!("  decode_tokens = {decode_tokens}");
    println!("  repeats       = {repeats}");
    println!();

    // Same memory-knob promotion as `serve` — bench is the A/B tool
    // for the expert cache / pagelock / zero-copy settings, so it
    // must load the model under the same env the server would.
    promote_memory_env_from_config(&cfg.inference, &cfg.tuning);
    let load_start = std::time::Instant::now();
    let mut cpu = CpuEngine::load_with_options_and_layout(
        &model_path,
        ctx_size,
        true,
        kv_dtype,
        &cfg.inference.kv_cache_layout,
    )
    .map_err(|e| anyhow::anyhow!("load failed: {e}"))?;
    cpu.set_flash_attention(flash_enabled);
    // Resolve `n_gpu_layers` the SAME way `serve` does, rather than passing the
    // raw config value. A fresh config's AUTO sentinel (N_GPU_LAYERS_AUTO = 999)
    // is NOT a layer count: the engine's per-tensor device plan (VRAM-fit /
    // measured-perf heat) is installed only when AUTO is resolved through
    // `auto_place_heat`. Handing the sentinel straight to `set_n_gpu_layers`
    // left the model all-CPU, so bench silently measured the WRONG device (looked
    // like a 2x "regression" that was really a CPU run). Precedence mirrors the
    // serve load path: hard GPU-fit guardrail → explicit override → cached tuner
    // winner → AUTO heat/VRAM-fit.
    let placement_opts = rustllama_engine::placement_auto::AutoPlacementOpts {
        cpu_enabled: cfg.inference.cpu_enabled,
        gpu_enabled: cfg.inference.gpu_enabled,
        vram_only: cfg.inference.vram_only,
        ..Default::default()
    };
    let override_n = if cfg.inference.gpu_enabled {
        cfg.inference.n_gpu_layers_override()
    } else {
        None
    };
    let model_key = model_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown-model")
        .to_string();
    let cached_placement = cfg
        .tuning
        .auto_apply_placement
        .then(|| placement_cache_lookup(&model_key))
        .flatten();
    let force_gpu_fit = !cfg.inference.cpu_enabled || cfg.inference.vram_only;
    let applied_n_gpu_layers = if force_gpu_fit {
        cpu.auto_place_heat(&placement_opts).n_gpu_layers
    } else if let Some(n) = override_n {
        n
    } else if let Some(n) = cached_placement {
        n
    } else {
        cpu.auto_place_heat(&placement_opts).n_gpu_layers
    };
    if applied_n_gpu_layers != cfg.inference.n_gpu_layers {
        tracing::info!(
            configured = cfg.inference.n_gpu_layers,
            applied = applied_n_gpu_layers,
            "bench placement: resolved n_gpu_layers (AUTO → VRAM-fit/heat plan)"
        );
    }
    cpu.set_n_gpu_layers(applied_n_gpu_layers);
    cpu.set_prefix_cache(false); // bench is comparable across repeats only without prefix reuse
    if cfg.inference.speculative_ngram {
        cpu.set_ngram_speculative(Some(rustllama_engine::speculative::NgramDrafterConfig {
            n_match: cfg.inference.ngram_n_match as usize,
            n_draft: cfg.inference.ngram_n_draft as usize,
            ..Default::default()
        }));
    }
    // MTP / NextN self-speculation (hybrid + NextN-head models only).
    cpu.set_mtp_speculative(cfg.inference.speculative_mtp);
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    println!("  load          = {load_ms:.1} ms");
    // Make the device placement explicit so a CPU-vs-GPU run is never
    // mistaken for a kernel regression (the AUTO-sentinel trap).
    println!(
        "  n_gpu_layers  = {applied_n_gpu_layers} / {} total",
        cpu.n_layers()
    );
    // Surface MoE info post-load so users comparing MoE vs dense
    // tok/s rows can see at a glance which row is which.
    if let Some(moe) = cpu.llama_config().moe.as_ref() {
        let shared = moe.n_experts_shared;
        if shared > 0 {
            println!(
                "  moe           = {} routed, top-{}, {shared} shared",
                moe.n_experts, moe.n_experts_used,
            );
        } else {
            println!(
                "  moe           = {} routed, top-{}",
                moe.n_experts, moe.n_experts_used,
            );
        }
    }

    // Synthetic prompt: monotonic ids (1, 2, 3, …) clipped to vocab.
    let vocab = cpu.vocab_size() as i32;
    let prompt_ids: Vec<i32> = (0..prompt_tokens as i32)
        .map(|i| 1 + (i % (vocab - 1)))
        .collect();

    // Deterministic / greedy sampling for the bench. We override
    // only the fields that need to differ from `Default` rather
    // than enumerating every field — the struct has grown over
    // time (mirostat, logprobs, grammar, ...) and the bench
    // doesn't care about most of them.
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        typical_p: 1.0,
        repeat_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 0,
        max_tokens: decode_tokens,
        stop: Vec::new(),
        ..SamplingParams::default()
    };

    let mut prefill_ms = Vec::with_capacity(repeats as usize);
    let mut decode_tps = Vec::with_capacity(repeats as usize);
    let mut wall_ms = Vec::with_capacity(repeats as usize);

    for r in 0..repeats {
        cpu.clear_prefix_cache();
        let t = std::time::Instant::now();
        let _generated = cpu
            .generate_token_ids(&prompt_ids, decode_tokens, &sampling)
            .map_err(|e| anyhow::anyhow!("generate failed: {e}"))?;
        let wall = t.elapsed().as_secs_f64() * 1000.0;
        let stats = cpu.last_request_stats();
        let pf_ms = stats.prefill_ms;
        let dc_ms = stats.decode_ms;
        let tok_s = if dc_ms > 0.0 {
            (stats.tokens_generated as f64) / (dc_ms / 1000.0)
        } else {
            0.0
        };
        println!(
            "  run {:>2}/{:<2}  prefill = {pf_ms:>8.1} ms  decode = {dc_ms:>8.1} ms  ({tok_s:>6.2} tok/s)  wall = {wall:>8.1} ms",
            r + 1,
            repeats,
        );
        prefill_ms.push(pf_ms);
        decode_tps.push(tok_s);
        wall_ms.push(wall);
    }

    fn stats(label: &str, mut xs: Vec<f64>, unit: &str) {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = xs.len();
        let min = xs[0];
        let max = xs[n - 1];
        let median = if n % 2 == 1 {
            xs[n / 2]
        } else {
            (xs[n / 2 - 1] + xs[n / 2]) / 2.0
        };
        println!("  {label:<14}: min {min:>8.2} {unit}  median {median:>8.2} {unit}  max {max:>8.2} {unit}");
    }
    println!();
    stats("prefill", prefill_ms, "ms");
    stats("decode", decode_tps, "tok/s");
    stats("wall", wall_ms, "ms");
    // MoE tiered-expert engine: how many matvecs ran against a promoted
    // (device-resident) expert during the bench. Non-zero confirms the
    // (auto-enabled) device tier actually engaged (vs silently falling back);
    // omitted when 0 so non-MoE / non-promoted benches stay quiet.
    let dev_resident = rustllama_engine::moe_dev_resident_hits();
    if dev_resident > 0 {
        println!();
        println!("  moe device-resident expert matvecs = {dev_resident}");
    }
    Ok(())
}

/// Map a `kv_dtype` config string to the enum the engine expects.
/// Accepts common spellings:
///   - `""` / `"f32"` / `"fp32"` / `"float32"` → `F32` (default).
///   - `"q8"` / `"q8_0"` / `"int8"` → `Q8_0` (per-row 8-bit + scale).
///   - `"tq1"` / `"tq2"` / `"tq4"` / `"tq8"` → TurboQuant
///     (Walsh–Hadamard rotation + uniform N-bit quant).
///   - `"nvfp4"` → NVIDIA NVFP4 (E2M1 + per-block FP8 scale).
fn parse_kv_dtype(s: &str) -> anyhow::Result<rustllama_engine::KvDtype> {
    let lower = s.to_ascii_lowercase();
    if matches!(lower.as_str(), "" | "f32" | "fp32" | "float32") {
        return Ok(rustllama_engine::KvDtype::F32);
    }
    if matches!(lower.as_str(), "q8" | "q8_0" | "int8") {
        return Ok(rustllama_engine::KvDtype::Q8_0);
    }
    if let Some(dtype) = rustllama_engine::KvDtype::parse(&lower) {
        return Ok(dtype);
    }
    anyhow::bail!("unknown kv_dtype `{s}` (expected: f32, q8_0, q4_0, tq1, tq2, tq4, tq8, nvfp4)");
}

// ----- chat history (unified sessions + conversations) -----

/// Route a `rustllama chat <sub>` history command to the right store: a
/// purely-numeric identifier is a GUI conversation (sqlite, `history`
/// feature); anything else is a saved-session name (file-backed).
fn run_chat_history(cmd: ChatCmd) -> anyhow::Result<()> {
    match cmd {
        ChatCmd::List { json } => chat_history_list(json),
        ChatCmd::Search {
            query,
            name,
            case_sensitive,
        } => sessions_search(&query, case_sensitive, name.as_deref()),
        ChatCmd::Show { id, format } => match id.parse::<i64>() {
            Ok(cid) => {
                #[cfg(feature = "history")]
                {
                    conv_show(cid, &format)
                }
                #[cfg(not(feature = "history"))]
                {
                    let _ = &format;
                    anyhow::bail!(no_history_feature_msg(cid))
                }
            }
            Err(_) => sessions_show(&id, &format),
        },
        ChatCmd::Delete { id } => match id.parse::<i64>() {
            Ok(cid) => {
                #[cfg(feature = "history")]
                {
                    conv_delete(cid)
                }
                #[cfg(not(feature = "history"))]
                {
                    anyhow::bail!(no_history_feature_msg(cid))
                }
            }
            Err(_) => sessions_delete(&id),
        },
        ChatCmd::Export { id, format, out } => match id.parse::<i64>() {
            Ok(cid) => {
                #[cfg(feature = "history")]
                {
                    conv_export(cid, &format, out.as_deref())
                }
                #[cfg(not(feature = "history"))]
                {
                    let _ = (&format, &out);
                    anyhow::bail!(no_history_feature_msg(cid))
                }
            }
            Err(_) => sessions_export(&id, &format, out.as_deref()),
        },
    }
}

/// Message for the slim (no-`history`) build when a numeric id targets the
/// sqlite conversation store that isn't compiled in.
#[cfg(not(feature = "history"))]
fn no_history_feature_msg(id: i64) -> String {
    format!(
        "conversation {id}: this build was compiled without the `history` \
         feature (sqlite conversation store unavailable). Use a saved-session \
         name, or a default-features build."
    )
}

/// `chat list` — a unified view over both stores: file-backed saved
/// sessions (name-keyed) and, when the `history` feature is on, the GUI's
/// sqlite conversations (id-keyed). Prints each store as its own labeled
/// section so the two identifier spaces stay clear.
fn chat_history_list(json: bool) -> anyhow::Result<()> {
    if json {
        // Two labeled JSON arrays (one per store) rather than one blob —
        // the identifier spaces (name vs numeric id) are distinct.
        println!("// saved sessions");
        sessions_list(true)?;
        #[cfg(feature = "history")]
        {
            println!("// conversations");
            conv_list(true)?;
        }
    } else {
        println!("== Saved sessions (resume/show by name) ==");
        sessions_list(false)?;
        #[cfg(feature = "history")]
        {
            println!("\n== Conversations (resume/show by numeric id) ==");
            conv_list(false)?;
        }
    }
    Ok(())
}

/// Load a saved chat's transcript for `chat --resume`. A numeric id is a
/// GUI conversation (needs the `history` feature); anything else is a
/// saved-session name. Returns `(messages, model_hint)` — the hint is the
/// model the session was saved with, used when `--model` isn't given.
fn load_resumable_chat(
    id_or_name: &str,
) -> anyhow::Result<(Vec<rustllama_client::ChatMessage>, Option<String>)> {
    use rustllama_client::ChatMessage;
    if let Ok(cid) = id_or_name.parse::<i64>() {
        #[cfg(feature = "history")]
        {
            let store = open_conv_store()?;
            let conv = store
                .get_conversation(cid)?
                .ok_or_else(|| anyhow::anyhow!("conversation {cid} not found"))?;
            let msgs = conv
                .messages
                .into_iter()
                .map(|m| ChatMessage {
                    role: m.role,
                    content: m.content,
                })
                .collect();
            return Ok((msgs, None));
        }
        #[cfg(not(feature = "history"))]
        {
            anyhow::bail!(no_history_feature_msg(cid));
        }
    }
    let session =
        session::load(id_or_name).map_err(|e| anyhow::anyhow!("resume `{id_or_name}`: {e}"))?;
    Ok((session.messages, Some(session.model)))
}

// ----- sessions -----

fn sessions_list(json: bool) -> anyhow::Result<()> {
    let entries = session::list().map_err(|e| anyhow::anyhow!(e))?;
    if json {
        println!("{}", serde_json::to_string_pretty(&entries)?);
        return Ok(());
    }
    if entries.is_empty() {
        println!("(no saved sessions)");
        return Ok(());
    }
    println!("{:<32} {:<24} {:>5}  UPDATED_AT", "NAME", "MODEL", "MSGS");
    for e in entries {
        let model = if e.model.len() > 24 {
            format!("{}…", &e.model[..23])
        } else {
            e.model
        };
        println!(
            "{:<32} {:<24} {:>5}  {}",
            e.name, model, e.message_count, e.updated_at
        );
    }
    Ok(())
}

fn sessions_show(name: &str, format: &str) -> anyhow::Result<()> {
    let session = session::load(name).map_err(|e| anyhow::anyhow!(e))?;
    match format {
        "markdown" => {
            print!("{}", session::render_markdown(&session));
        }
        _ => {
            println!(
                "session `{}`  model={}  messages={}",
                session.name,
                session.model,
                session.messages.len()
            );
            println!(
                "created_at={}  updated_at={}",
                session.created_at, session.updated_at
            );
            println!("---");
            for m in &session.messages {
                println!("[{}]", m.role);
                println!("{}", m.content);
                println!();
            }
        }
    }
    Ok(())
}

fn sessions_search(
    query: &str,
    case_sensitive: bool,
    name_filter: Option<&str>,
) -> anyhow::Result<()> {
    if query.is_empty() {
        anyhow::bail!("search query must not be empty");
    }
    let hits =
        session::search(query, case_sensitive, name_filter).map_err(|e| anyhow::anyhow!(e))?;
    if hits.is_empty() {
        println!("(no matches)");
        return Ok(());
    }
    let mut current_session: Option<String> = None;
    for h in &hits {
        if current_session.as_deref() != Some(h.session_name.as_str()) {
            println!(
                "\n=== {} (updated_at={}) ===",
                h.session_name, h.session_updated_at
            );
            current_session = Some(h.session_name.clone());
        }
        println!("  [{}] msg #{}", h.role, h.message_index);
        // Indent the snippet by 4 so multi-line matches are visually
        // attached to the message header above.
        for line in h.snippet.lines() {
            println!("    {line}");
        }
    }
    println!("\n{} match(es) across {} session(s)", hits.len(), {
        let mut names = hits
            .iter()
            .map(|h| h.session_name.as_str())
            .collect::<Vec<_>>();
        names.sort();
        names.dedup();
        names.len()
    });
    Ok(())
}

fn sessions_delete(name: &str) -> anyhow::Result<()> {
    session::delete(name).map_err(|e| anyhow::anyhow!(e))?;
    println!("deleted session `{name}`");
    Ok(())
}

fn sessions_export(name: &str, format: &str, out: Option<&std::path::Path>) -> anyhow::Result<()> {
    let session = session::load(name).map_err(|e| anyhow::anyhow!(e))?;
    let content = match format {
        "json" => serde_json::to_string_pretty(&session)?,
        _ => session::render_markdown(&session),
    };
    match out {
        Some(path) => {
            std::fs::write(path, &content)?;
            println!("exported `{name}` → {}", path.display());
        }
        None => print!("{content}"),
    }
    Ok(())
}

// ----- models -----

fn models_list() -> anyhow::Result<()> {
    let cache_dir = rustllama_hub::default_cache_dir()
        .ok_or_else(|| anyhow::anyhow!("could not resolve cache dir"))?;
    // `list_cached_models` surfaces cached GGUF files AND MLX model
    // directories (mlx-lm / mlx-community layout), so both show up here and
    // can be loaded on the running server by name.
    let entries = rustllama_hub::list_cached_models(&cache_dir)?;
    if entries.is_empty() {
        println!("(no cached models in {})", cache_dir.display());
        return Ok(());
    }
    println!("{:<48} {:>10}  PATH", "NAME", "SIZE_MB");
    for entry in entries {
        // GGUF = the file size; MLX dir = the sum of its `*.safetensors`
        // shards (the weights), since the dir's own `metadata().len()` is
        // meaningless.
        let size_mb = if entry.is_dir {
            mlx_dir_weight_bytes(&entry.path) as f64 / (1024.0 * 1024.0)
        } else {
            std::fs::metadata(&entry.path)
                .map(|m| (m.len() as f64) / (1024.0 * 1024.0))
                .unwrap_or(0.0)
        };
        let label = if entry.is_dir {
            format!("{} (mlx)", entry.name)
        } else {
            entry.name.clone()
        };
        println!("{:<48} {:>10.1}  {}", label, size_mb, entry.path.display());
    }
    Ok(())
}

/// Total bytes of an MLX model directory's `*.safetensors` shards — the
/// weights that dominate its on-disk size. Best-effort; 0 on an unreadable
/// dir. (Mirrors the server's `/api/tags` size column.)
fn mlx_dir_weight_bytes(dir: &std::path::Path) -> u64 {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .filter(|e| {
                    e.path()
                        .extension()
                        .and_then(|x| x.to_str())
                        .map(|x| x.eq_ignore_ascii_case("safetensors"))
                        .unwrap_or(false)
                })
                .filter_map(|e| e.metadata().ok().map(|m| m.len()))
                .sum()
        })
        .unwrap_or(0)
}

fn models_rm(hub_ref_or_path: &str) -> anyhow::Result<()> {
    let path = resolve_model_path(hub_ref_or_path)?;
    if !path.exists() {
        anyhow::bail!("not in cache: {}", path.display());
    }
    std::fs::remove_file(&path)?;
    println!("removed {}", path.display());

    // Clean up an empty owner__repo directory if applicable.
    if let Some(parent) = path.parent() {
        if let Ok(mut it) = std::fs::read_dir(parent) {
            if it.next().is_none() {
                let _ = std::fs::remove_dir(parent);
            }
        }
    }
    Ok(())
}

fn models_use(config_path: &std::path::Path, hub_ref_or_path: &str) -> anyhow::Result<()> {
    let path = resolve_model_path(hub_ref_or_path)?;
    if !path.exists() {
        anyhow::bail!(
            "not in cache: {} (run `rustllama pull` first)",
            path.display()
        );
    }
    let mut cfg = rustllama_config::load(config_path).unwrap_or_default();
    cfg.model.path = Some(path.clone());
    cfg.model.hub = None;
    rustllama_config::save(config_path, &cfg)?;
    println!("active model: {}", path.display());
    Ok(())
}

fn models_inspect(path: &std::path::Path, list_tensors: bool, as_json: bool) -> anyhow::Result<()> {
    use rustllama_gguf::{Gguf, MetadataValue};

    let gguf = Gguf::open(path).map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))?;

    // ---- pull common metadata keys ----------------------------------
    let arch = gguf.architecture().unwrap_or("?").to_string();
    let model_name = gguf
        .metadata_get("general.name")
        .and_then(MetadataValue::as_string)
        .unwrap_or("(unnamed)")
        .to_string();
    let model_size_label = gguf
        .metadata_get("general.size_label")
        .and_then(MetadataValue::as_string)
        .map(str::to_string);
    let quant_label = gguf
        .metadata_get("general.file_type")
        .and_then(MetadataValue::as_u32)
        .map(|n| format!("file_type={n}"));

    let key = |k: &str| -> Option<u32> {
        // try `{arch}.{k}` first, fall back to bare `k` for older converters
        let full = format!("{arch}.{k}");
        gguf.metadata_get(&full)
            .or_else(|| gguf.metadata_get(k))
            .and_then(MetadataValue::as_u32)
    };
    let ctx_train = key("context_length");
    let n_layers = key("block_count");
    let d_model = key("embedding_length");
    let n_heads = key("attention.head_count");
    let n_kv_heads = key("attention.head_count_kv");
    let head_dim = key("attention.key_length").or_else(|| key("rope.dimension_count"));
    let vocab_size = gguf
        .metadata_get("tokenizer.ggml.tokens")
        .and_then(|v| match v {
            MetadataValue::Array(a) => Some(a.len() as u32),
            _ => None,
        });
    // MoE expert counts: filtered to >=2 so single-expert synthetic
    // GGUFs don't get reported as MoE; matches the convention used by
    // the HTTP `/api/gguf/inspect` endpoint.
    let n_experts = key("expert_count").filter(|&n| n >= 2);
    let n_experts_used = n_experts.and_then(|_| key("expert_used_count").or(Some(1)));
    let n_experts_shared = n_experts.and_then(|_| key("expert_shared_count").or(Some(0)));

    // ---- model card sidecar (if any) -------------------------------
    let card = rustllama_hub::model_card::load_for_gguf(path);

    // ---- tensor stats ----------------------------------------------
    let tensors = gguf.tensors();
    let total_tensors = tensors.len();
    let total_bytes: u64 = tensors.iter().map(|t| t.byte_size).sum();
    let total_params: u64 = tensors.iter().map(|t| t.element_count()).sum();
    // Histogram by dtype.
    let mut dtype_counts: std::collections::BTreeMap<&'static str, (u64, u64)> =
        std::collections::BTreeMap::new();
    for t in tensors {
        let name = ggml_type_name(t.dtype);
        let entry = dtype_counts.entry(name).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += t.byte_size;
    }
    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    if as_json {
        let dtype_json: Vec<_> = dtype_counts
            .iter()
            .map(|(name, (count, bytes))| {
                serde_json::json!({ "dtype": name, "tensor_count": count, "bytes": bytes })
            })
            .collect();
        let tensors_json: Option<Vec<_>> = if list_tensors {
            Some(
                tensors
                    .iter()
                    .map(|t| {
                        serde_json::json!({
                            "name": t.name,
                            "dtype": ggml_type_name(t.dtype),
                            "shape": t.dims,
                            "elements": t.element_count(),
                            "bytes": t.byte_size,
                            "offset": t.abs_offset,
                        })
                    })
                    .collect(),
            )
        } else {
            None
        };
        let payload = serde_json::json!({
            "path": path.display().to_string(),
            "architecture": arch,
            "name": model_name,
            "size_label": model_size_label,
            "quantization": quant_label,
            "context_length": ctx_train,
            "block_count": n_layers,
            "embedding_length": d_model,
            "head_count": n_heads,
            "head_count_kv": n_kv_heads,
            "head_dim": head_dim,
            "vocab_size": vocab_size,
            "n_experts": n_experts,
            "n_experts_used": n_experts_used,
            "n_experts_shared": n_experts_shared,
            "tensor_count": total_tensors,
            "total_params": total_params,
            "total_tensor_bytes": total_bytes,
            "file_bytes": file_size,
            "dtypes": dtype_json,
            "tensors": tensors_json,
            "model_card": card,
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    // ---- human-readable summary ------------------------------------
    println!("path:           {}", path.display());
    println!("file size:      {}", human_bytes(file_size));
    println!("architecture:   {arch}");
    println!("name:           {model_name}");
    if let Some(s) = &model_size_label {
        println!("size label:    {s}");
    }
    if let Some(q) = &quant_label {
        println!("quantization:  {q}");
    }
    print_kv("context_length", ctx_train);
    print_kv("block_count", n_layers);
    print_kv("embedding_length", d_model);
    print_kv("head_count", n_heads);
    print_kv("head_count_kv", n_kv_heads);
    print_kv("head_dim", head_dim);
    print_kv("vocab_size", vocab_size);
    if let Some(n) = n_experts {
        let used = n_experts_used.unwrap_or(0);
        let shared = n_experts_shared.unwrap_or(0);
        if shared > 0 {
            println!("MoE:            {n} routed, top-{used}, {shared} shared");
        } else {
            println!("MoE:            {n} routed, top-{used}");
        }
    }
    println!();
    println!("tensors:        {total_tensors}");
    println!("total params:   {}", human_count(total_params));
    println!("tensor bytes:   {}", human_bytes(total_bytes));
    println!();
    println!("dtype breakdown:");
    for (name, (count, bytes)) in &dtype_counts {
        println!("  {name:<6} {count:>6}  ({})", human_bytes(*bytes));
    }

    if let Some(c) = &card {
        println!();
        println!("model card (HuggingFace):");
        if let Some(url) = &c.source_url {
            println!("  source:        {url}");
        }
        if let Some(license) = &c.license {
            println!("  license:       {license}");
        }
        if let Some(library) = &c.library_name {
            println!("  library:       {library}");
        }
        if let Some(base) = &c.base_model {
            println!("  base model:    {base}");
        }
        if let Some(creator) = &c.model_creator {
            println!("  creator:       {creator}");
        }
        if let Some(quantized_by) = &c.quantized_by {
            println!("  quantized by:  {quantized_by}");
        }
        if !c.tags.is_empty() {
            println!("  tags:          {}", c.tags.join(", "));
        }
        if !c.language.is_empty() {
            println!("  languages:     {}", c.language.join(", "));
        }
        if let Some(desc) = &c.description {
            println!("  description:   {desc}");
        }
    }

    if list_tensors {
        println!();
        println!("tensor table:");
        println!(
            "  {:<40} {:<8} {:<20} {:>14}",
            "name", "dtype", "shape", "bytes"
        );
        for t in tensors {
            let shape = t
                .dims
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join("×");
            println!(
                "  {:<40} {:<8} {:<20} {:>14}",
                truncate(&t.name, 40),
                ggml_type_name(t.dtype),
                shape,
                human_bytes(t.byte_size),
            );
        }
    }

    Ok(())
}

fn ggml_type_name(t: rustllama_gguf::GgmlType) -> &'static str {
    // Delegate to the canonical name table so a new GgmlType can
    // never silently drift out of `models inspect` (this used to be
    // a hand-maintained duplicate match).
    t.as_str()
}

fn print_kv(key: &str, value: Option<u32>) {
    match value {
        Some(v) => println!("{key:<15}: {v}"),
        None => println!("{key:<15}: (not present)"),
    }
}

fn human_bytes(n: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = KIB * 1024;
    const GIB: u64 = MIB * 1024;
    if n >= GIB {
        format!("{:.2} GiB", n as f64 / GIB as f64)
    } else if n >= MIB {
        format!("{:.2} MiB", n as f64 / MIB as f64)
    } else if n >= KIB {
        format!("{:.2} KiB", n as f64 / KIB as f64)
    } else {
        format!("{n} B")
    }
}

fn human_count(n: u64) -> String {
    const M: u64 = 1_000_000;
    const B: u64 = 1_000_000_000;
    if n >= B {
        format!("{:.2} B", n as f64 / B as f64)
    } else if n >= M {
        format!("{:.1} M", n as f64 / M as f64)
    } else {
        format!("{n}")
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut out = String::with_capacity(max);
        out.push_str(&s[..max.saturating_sub(1)]);
        out.push('…');
        out
    }
}

/// Resolve the effective model for a subcommand: the `--model` override when
/// given, else `[model].path` from the config. EITHER source is run through
/// [`resolve_model_path`], so a hub ref (`org/repo:file.gguf`), a bare cached
/// id, or a path on disk works in either slot (CLAUDE.md notes `[model].path`
/// may itself be a hub ref). `config_path` names the config in the "no model"
/// error. This is the single front door every model-taking subcommand uses so
/// resolution is identical whether you typed `--model` or set the config.
fn effective_model_path(
    model_override: Option<String>,
    cfg: &rustllama_config::Config,
    config_path: &std::path::Path,
) -> anyhow::Result<PathBuf> {
    let spec = match model_override {
        Some(s) => s,
        None => cfg
            .model
            .path
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "no model: pass --model or set [model].path in {}",
                    config_path.display()
                )
            })?,
    };
    resolve_model_path(&spec)
}

fn resolve_model_path(s: &str) -> anyhow::Result<PathBuf> {
    let raw = PathBuf::from(s);
    // An existing path (absolute OR relative to the CWD) is used verbatim;
    // an absolute path that merely carries an extension also passes (lets a
    // not-yet-downloaded target resolve to where it WILL land — `model use`
    // / `pull` rely on naming a stable target before the file exists). Only
    // when neither holds do we consult the cache.
    if raw.exists() || (raw.is_absolute() && raw.extension().is_some()) {
        return Ok(raw);
    }
    let cache_dir = rustllama_hub::default_cache_dir()
        .ok_or_else(|| anyhow::anyhow!("could not resolve cache dir"))?;
    // Superset cache lookup — hub ref (`org/repo:file.gguf`), bare cached id /
    // file stem, or MLX model dir — the same resolver the server + GUI use.
    if let Some(found) = rustllama_hub::resolve_model_spec(s, &cache_dir) {
        return Ok(found);
    }
    // Not in the cache: a well-formed hub ref still resolves to its would-be
    // location so `model use org/repo:file` (before `pull`) names a stable
    // target; anything else is unresolvable.
    let hub_ref = rustllama_hub::HubRef::parse(s).map_err(|e| {
        anyhow::anyhow!("{s:?} is not an existing path, a cached id, or a valid hub ref: {e}")
    })?;
    Ok(hub_ref.local_path(&cache_dir))
}

// ----- config -----

fn config_show(config_path: &std::path::Path) -> anyhow::Result<()> {
    let cfg = rustllama_config::load(config_path)?;
    println!("{}", toml::to_string_pretty(&cfg)?);
    Ok(())
}

fn config_get(config_path: &std::path::Path, key: &str) -> anyhow::Result<()> {
    let cfg = rustllama_config::load(config_path)?;
    let val = toml::Value::try_from(&cfg)?;
    let leaf = lookup_dotted(&val, key).ok_or_else(|| anyhow::anyhow!("no value at `{key}`"))?;
    println!("{}", format_leaf(leaf));
    Ok(())
}

fn config_set(config_path: &std::path::Path, key: &str, value: &str) -> anyhow::Result<()> {
    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let mut val = toml::Value::try_from(&cfg)?;
    set_dotted(&mut val, key, parse_value(value))?;
    let updated: rustllama_config::Config = val.try_into()?;
    rustllama_config::save(config_path, &updated)?;
    println!("set {key} = {value}");
    Ok(())
}

fn config_edit(config_path: &std::path::Path) -> anyhow::Result<()> {
    if !config_path.exists() {
        let cfg = rustllama_config::Config::default();
        rustllama_config::save(config_path, &cfg)?;
    }
    let editor = std::env::var("EDITOR")
        .or_else(|_| std::env::var("VISUAL"))
        .unwrap_or_else(|_| "notepad".to_string());
    let status = std::process::Command::new(&editor)
        .arg(config_path)
        .status()
        .map_err(|e| anyhow::anyhow!("failed to launch editor {editor:?}: {e}"))?;
    if !status.success() {
        anyhow::bail!("editor exited with {:?}", status.code());
    }
    // Validate after the edit so a syntax error surfaces immediately.
    rustllama_config::load(config_path)?;
    println!("saved {}", config_path.display());
    Ok(())
}

/// Validate the config without starting the server. Surfaces parse
/// errors (with line numbers from `toml`), validation failures
/// (mutual-exclusive keys, invalid kv_dtype), and unknown profile
/// names. Exits non-zero on any problem so CI / scripts can gate on
/// `rustllama config validate --profile prod`.
fn config_validate(config_path: &std::path::Path, profile: Option<&str>) -> anyhow::Result<()> {
    if !config_path.exists() {
        println!("config file does not exist: {}", config_path.display());
        println!("(this is fine — defaults apply)");
        return Ok(());
    }
    let cfg = rustllama_config::load(config_path)?;
    println!("config: {} — parses + validates", config_path.display());

    // Profile check: name must exist if specified. Print all
    // available profiles regardless so the user sees what's there.
    let profile_names = cfg.profile_names();
    if profile_names.is_empty() {
        println!("profiles: <none defined>");
    } else {
        println!(
            "profiles: {} defined ({})",
            profile_names.len(),
            profile_names.join(", "),
        );
    }
    if let Some(name) = profile {
        if !name.is_empty() {
            if profile_names.iter().any(|p| *p == name) {
                println!("profile `{name}`: applies cleanly");
                // Apply on a clone to surface any post-merge issues
                // without keeping the mutation.
                let mut probe = cfg.clone();
                probe.apply_profile(name);
                rustllama_config::validate(&probe)?;
                println!("profile `{name}`: post-merge config validates");
            } else {
                anyhow::bail!(
                    "unknown profile `{name}` (available: {})",
                    if profile_names.is_empty() {
                        "<none>".to_string()
                    } else {
                        profile_names.join(", ")
                    }
                );
            }
        }
    }

    // KV dtype sanity check: `[inference].kv_dtype` is a string in
    // the schema; we ping `parse_kv_dtype` to catch bad spellings.
    match parse_kv_dtype(&cfg.inference.kv_dtype) {
        Ok(d) => println!("kv_dtype: `{}` → {:?}", cfg.inference.kv_dtype, d),
        Err(e) => anyhow::bail!("[inference].kv_dtype: {e}"),
    }

    Ok(())
}

// ----- conv (sqlite-backed conversation store) -----
// The whole section rides the default-on `history` feature; slim
// builds drop it together with the server's rusqlite dependency.

/// Default location for the conversations sqlite, matching where
/// `serve` opens its store. Lives under `<cache_dir>/conversations.db`
/// (the runtime's cache_dir is the same directory that holds the
/// model cache + tuning data).
#[cfg(feature = "history")]
fn conv_db_path() -> Option<std::path::PathBuf> {
    let paths = rustllama_runtime::paths();
    Some(paths.cache_dir.join("conversations.db"))
}

#[cfg(feature = "history")]
fn open_conv_store() -> anyhow::Result<rustllama_server::history::HistoryStore> {
    let path =
        conv_db_path().ok_or_else(|| anyhow::anyhow!("cannot resolve conversations.db path"))?;
    Ok(rustllama_server::history::HistoryStore::open(path)?)
}

#[cfg(feature = "history")]
fn conv_list(json: bool) -> anyhow::Result<()> {
    let store = open_conv_store()?;
    let rows = store.list_conversations()?;
    if json {
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    if rows.is_empty() {
        println!("no conversations stored");
        return Ok(());
    }
    println!("{:>6}  {:<20}  {}", "id", "updated_at", "title");
    for row in &rows {
        // updated_at is unix-seconds; render as a short relative
        // "<n>m ago" or absolute date depending on age. Keeps the
        // output scannable without pulling in a date-formatting
        // dependency.
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let age = now - row.updated_at;
        let when = if age < 60 {
            "just now".to_string()
        } else if age < 3600 {
            format!("{}m ago", age / 60)
        } else if age < 86_400 {
            format!("{}h ago", age / 3600)
        } else {
            format!("{}d ago", age / 86_400)
        };
        println!("{:>6}  {:<20}  {}", row.id, when, row.title);
    }
    Ok(())
}

#[cfg(feature = "history")]
fn conv_show(id: i64, format: &str) -> anyhow::Result<()> {
    let store = open_conv_store()?;
    let conv = store
        .get_conversation(id)?
        .ok_or_else(|| anyhow::anyhow!("conversation {id} not found"))?;
    match format {
        "markdown" => {
            println!("# {}\n", conv.summary.title);
            for m in &conv.messages {
                let role = match m.role.as_str() {
                    "user" => "User",
                    "assistant" => "Assistant",
                    "system" => "System",
                    "tool" => "Tool",
                    other => other,
                };
                println!("## {role}\n\n{}\n", m.content);
            }
        }
        _ => {
            // Plain text: just the messages, role-prefixed.
            println!("{}", conv.summary.title);
            println!("{}", "=".repeat(conv.summary.title.len().min(60)));
            for m in &conv.messages {
                println!("\n[{}]\n{}", m.role, m.content);
            }
        }
    }
    Ok(())
}

#[cfg(feature = "history")]
fn conv_export(id: i64, format: &str, out: Option<&std::path::Path>) -> anyhow::Result<()> {
    let store = open_conv_store()?;
    let conv = store
        .get_conversation(id)?
        .ok_or_else(|| anyhow::anyhow!("conversation {id} not found"))?;
    let body = match format {
        "markdown" | "md" => {
            let mut s = String::new();
            s.push_str(&format!("# {}\n\n", conv.summary.title));
            s.push_str(&format!(
                "<!-- created_at={}, updated_at={} -->\n\n",
                conv.summary.created_at, conv.summary.updated_at
            ));
            for m in &conv.messages {
                let role = match m.role.as_str() {
                    "user" => "User",
                    "assistant" => "Assistant",
                    "system" => "System",
                    "tool" => "Tool",
                    other => other,
                };
                s.push_str(&format!("## {role}\n\n{}\n\n", m.content));
            }
            s
        }
        "json" => serde_json::to_string_pretty(&conv)?,
        other => anyhow::bail!("unknown format `{other}` (expected: markdown, md, json)"),
    };
    match out {
        Some(path) => {
            std::fs::write(path, body)?;
            println!("wrote {}", path.display());
        }
        None => print!("{body}"),
    }
    Ok(())
}

#[cfg(feature = "history")]
fn conv_delete(id: i64) -> anyhow::Result<()> {
    let store = open_conv_store()?;
    if store.delete_conversation(id)? {
        println!("deleted conversation {id}");
    } else {
        anyhow::bail!("conversation {id} not found");
    }
    Ok(())
}

fn lookup_dotted<'a>(v: &'a toml::Value, key: &str) -> Option<&'a toml::Value> {
    let mut cur = v;
    for part in key.split('.') {
        cur = cur.as_table()?.get(part)?;
    }
    Some(cur)
}

fn set_dotted(v: &mut toml::Value, key: &str, new: toml::Value) -> anyhow::Result<()> {
    let parts: Vec<&str> = key.split('.').collect();
    if parts.is_empty() {
        anyhow::bail!("empty key");
    }
    let mut cur = v;
    for part in &parts[..parts.len() - 1] {
        let tbl = cur
            .as_table_mut()
            .ok_or_else(|| anyhow::anyhow!("path traverses through non-table at `{part}`"))?;
        cur = tbl
            .entry((*part).to_string())
            .or_insert_with(|| toml::Value::Table(toml::value::Table::new()));
    }
    let tbl = cur
        .as_table_mut()
        .ok_or_else(|| anyhow::anyhow!("non-table at parent of `{key}`"))?;
    tbl.insert(parts.last().unwrap().to_string(), new);
    Ok(())
}

fn parse_value(s: &str) -> toml::Value {
    // Try bool, integer, float, then bare string.
    if let Ok(b) = s.parse::<bool>() {
        return toml::Value::Boolean(b);
    }
    if let Ok(i) = s.parse::<i64>() {
        return toml::Value::Integer(i);
    }
    if let Ok(f) = s.parse::<f64>() {
        return toml::Value::Float(f);
    }
    toml::Value::String(s.to_string())
}

fn format_leaf(v: &toml::Value) -> String {
    match v {
        toml::Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

// ----- chat REPL -----

async fn chat_repl(
    config_path: &std::path::Path,
    system: Option<String>,
    model: Option<String>,
    base_url: Option<String>,
    initial: Vec<rustllama_client::ChatMessage>,
) -> anyhow::Result<()> {
    use rustllama_client::{ChatMessage, ChatRequest, Client};

    let cfg = rustllama_config::load(config_path).unwrap_or_default();

    // Resolve server URL: --base-url > running-server record > config.
    let server_url = resolve_base_url(&cfg, base_url.as_deref());
    let client = Client::new(server_url.as_str())?;

    // Sanity: probe /healthz so we fail fast if no server is reachable.
    match client.healthz().await {
        Ok(v) => {
            let id = v
                .get("model_id")
                .and_then(|s| s.as_str())
                .unwrap_or("(unknown)");
            println!("connected to {server_url} (model: {id})");
        }
        Err(e) => {
            anyhow::bail!(
                "no rustllama server reachable at {server_url}: {e}\n\
                 run `rustllama serve` in another terminal first."
            );
        }
    }

    let mut messages: Vec<ChatMessage> = Vec::new();
    if let Some(sys) = system {
        messages.push(ChatMessage {
            role: "system".into(),
            content: sys,
        });
    }
    // Seed with a resumed transcript (`chat --resume`) so the next turn
    // continues the saved conversation. Echo it back so the user sees the
    // context they're picking up from.
    if !initial.is_empty() {
        for m in &initial {
            println!("\x1b[2m[{}]\x1b[0m {}", m.role, m.content);
        }
        messages.extend(initial);
    }
    // Empty string means "let the server pick its default model". With
    // the multi-model registry, sending `model: "default"` would fail
    // resolution (no model is literally named "default"); empty is the
    // sentinel for "use default" per the server's resolve() contract.
    let mut model_label = model.unwrap_or_default();
    // Runtime-tweakable sampling. `/temp` and `/max_tokens` adjust these
    // mid-session without restarting the REPL.
    let mut temperature: f32 = 0.7;
    let mut max_tokens: u32 = 512;
    let mut top_p: f32 = 0.95;
    let mut top_k: u32 = 40;
    let mut repeat_penalty: f32 = 1.1;
    // Auto-run tools: OFF prompts approve/deny on each proposed tool call;
    // ON accepts silently. Toggled with `/auto`. (The REPL never executes
    // tools — this only gates the confirmation prompt.)
    let mut auto_tools = false;
    // CLARIFY: ON lets the model pause and ask a clarifying question (with
    // selectable options) instead of guessing — the server injects the
    // reserved `ask_user` tool. Default ON so this UI can surface clarifying
    // questions; toggled with `/clarify`. OFF sends the plain (no-tool)
    // request path, wire-identical to a client that never knew CLARIFY.
    let mut clarify_enabled = true;

    // rustyline editor with history under %APPDATA%\rustllama\chat_history
    let history_path = rustllama_config::default_config_path()
        .and_then(|p| p.parent().map(|d| d.join("chat_history")));
    let mut rl = rustyline::DefaultEditor::new()?;
    if let Some(ref hp) = history_path {
        let _ = rl.load_history(hp);
    }

    println!("type a message; /help for commands");
    println!(
        "\x1b[2mclarifying questions: {} (/clarify to toggle)\x1b[0m",
        if clarify_enabled { "on" } else { "off" }
    );
    loop {
        let line = match rl.readline("> ") {
            Ok(l) => l,
            Err(rustyline::error::ReadlineError::Interrupted)
            | Err(rustyline::error::ReadlineError::Eof) => break,
            Err(e) => return Err(e.into()),
        };
        // PowerShell on Windows pipes a UTF-8 BOM at stream start, which
        // would otherwise break the very first slash command (the BOM
        // appears before `/` and `strip_prefix('/')` fails). Strip it
        // defensively — the BOM is meaningless mid-stream too.
        let trimmed = line.trim().trim_start_matches('\u{feff}');
        if trimmed.is_empty() {
            continue;
        }
        let _ = rl.add_history_entry(trimmed);

        if let Some(rest) = trimmed.strip_prefix('/') {
            let cmd = rest.split_whitespace().next().unwrap_or("");
            let arg = rest[cmd.len()..].trim().to_string();
            match cmd {
                "exit" | "quit" => break,
                "help" => {
                    // Slash commands mirror the `rustllama` CLI verbs.
                    println!("commands (mirror the CLI verbs):");
                    println!("  /exit, /quit       quit");
                    println!("  /help              this list");
                    println!("  /reset, /clear     drop user+assistant history (system kept)");
                    println!("  /system <text>     replace the system prompt (omit text to clear)");
                    println!("  /regenerate        re-run the last user turn (drop prior assistant reply)");
                    println!("  model (like `rustllama model`):");
                    println!("    /model <id>          quick-switch THIS chat's model");
                    println!("    /model list          list models on the connected server");
                    println!("    /model load <ref>    load a model (no promote; hub ref or path)");
                    println!("    /model default <ref> load + promote to server default");
                    println!("    /model unload <id>   unload a model (not the default)");
                    println!("  chat history (like `rustllama chat`):");
                    println!("    /chat list           list saved sessions");
                    println!("    /chat save [name]    persist this conversation");
                    println!("    /chat resume <name>  reload a saved session into this chat");
                    println!("    /chat delete <name>  delete a saved session");
                    println!("  sampling / output:");
                    println!(
                        "    /temp <0..2>       sampling temperature (current: {temperature})"
                    );
                    println!("    /top_p <0..1>      nucleus-sampling cutoff (current: {top_p})");
                    println!(
                        "    /top_k <0|N>       keep top-K tokens; 0 = full vocab (current: {top_k})"
                    );
                    println!("    /repeat_penalty <0.5..2.0>  discourage repeats; 1.0 = off (current: {repeat_penalty})");
                    println!("    /max_tokens <n>    per-turn token cap (current: {max_tokens})");
                    println!(
                        "    /auto [on|off]     toggle auto-run of proposed tool calls (current: {})",
                        if auto_tools { "on" } else { "off" }
                    );
                    println!(
                        "    /clarify [on|off]  let the model ask clarifying questions (current: {})",
                        if clarify_enabled { "on" } else { "off" }
                    );
                    println!("    /save <path>       append the last assistant message to a file");
                    println!(
                        "    /transcript <path> dump the full conversation as markdown to a file"
                    );
                    continue;
                }
                "reset" | "clear" => {
                    messages.retain(|m| m.role == "system");
                    println!("[history cleared, system preserved]");
                    continue;
                }
                "system" => {
                    messages.retain(|m| m.role != "system");
                    if !arg.is_empty() {
                        messages.insert(
                            0,
                            ChatMessage {
                                role: "system".into(),
                                content: arg,
                            },
                        );
                    }
                    println!("[system prompt updated]");
                    continue;
                }
                "regenerate" | "regen" => {
                    // Drop the most recent assistant turn (if any) so
                    // the `messages` list ends with the last user turn,
                    // then re-run generation against that state.
                    if messages.last().map(|m| m.role.as_str()) == Some("assistant") {
                        messages.pop();
                    }
                    if !messages.iter().any(|m| m.role == "user") {
                        println!("no user turn to regenerate");
                        continue;
                    }
                    println!("[regenerating last turn]");
                    let req = ChatRequest {
                        model: model_label.clone(),
                        messages: messages.clone(),
                        temperature: Some(temperature),
                        top_p: Some(top_p),
                        top_k: Some(top_k),
                        max_tokens: Some(max_tokens),
                        repeat_penalty: Some(repeat_penalty),
                        seed: None,
                        stream: true,
                        stream_options: Some(rustllama_client::StreamOptions {
                            include_usage: true,
                        }),
                        // `None` (not `Some(false)`) when off, so the OFF
                        // request is wire-identical to the plain path.
                        allow_clarify: clarify_enabled.then_some(true),
                    };
                    drive_chat_stream(&client, req, &mut messages).await?;
                    continue;
                }
                // `/model …` mirrors the `rustllama model` CLI verbs:
                //   /model              show the active model + usage
                //   /model list         list loaded models (server)
                //   /model load <ref>   load without promoting to default
                //   /model default <r>  load + promote to server default
                //   /model unload <id>  unload (refuses the default)
                //   /model <id>         quick-switch THIS chat's routing
                "model" => {
                    let sub = arg.split_whitespace().next().unwrap_or("");
                    let subarg = arg[sub.len()..].trim().to_string();
                    match sub {
                        "" => {
                            let shown = if model_label.is_empty() {
                                "(server default)".to_string()
                            } else {
                                model_label.clone()
                            };
                            println!("current model: {shown}");
                            println!(
                                "usage: /model <id> | list | load <ref> | default <ref> | unload <id>"
                            );
                        }
                        "list" => match client.list_models().await {
                            Ok(v) => {
                                let data = v.get("data").and_then(|d| d.as_array());
                                match data {
                                    Some(arr) if !arr.is_empty() => {
                                        println!("available models:");
                                        for m in arr {
                                            let id =
                                                m.get("id").and_then(|s| s.as_str()).unwrap_or("?");
                                            let is_default = m
                                                .get("is_default")
                                                .and_then(|b| b.as_bool())
                                                .unwrap_or(false);
                                            let marker = if is_default { " (default)" } else { "" };
                                            let active = if *id == *model_label
                                                || (model_label.is_empty() && is_default)
                                            {
                                                " *"
                                            } else {
                                                ""
                                            };
                                            println!("  {id}{marker}{active}");
                                        }
                                    }
                                    _ => println!("(no models loaded)"),
                                }
                            }
                            Err(e) => println!("[error listing models: {e}]"),
                        },
                        "load" | "default" => {
                            if subarg.is_empty() {
                                println!("usage: /model {sub} <hub-ref|path>");
                                continue;
                            }
                            let promote = sub == "default";
                            let mut params = rustllama_client::LoadModelParams::default();
                            match classify_swap_target(&subarg) {
                                SwapTarget::Hub(h) => params.hub = Some(h),
                                SwapTarget::Path(p) => params.path = Some(p),
                            }
                            print!("[loading...");
                            let _ = std::io::Write::flush(&mut std::io::stdout());
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
                                                model_label = new_id.clone();
                                                println!(" loaded `{new_id}` (now default)]");
                                            }
                                            Err(e) => {
                                                println!(" loaded but failed to promote: {e}]")
                                            }
                                        }
                                    } else {
                                        println!(
                                            " loaded `{new_id}` (not default; /model {new_id} to route here)]"
                                        );
                                    }
                                }
                                Err(e) => println!(" failed: {e}]"),
                            }
                        }
                        "unload" => {
                            if subarg.is_empty() {
                                println!("usage: /model unload <id>");
                                continue;
                            }
                            // Refuse to unload the current default (same rule
                            // as the `rustllama model unload` CLI command).
                            let is_default = match client.list_models().await {
                                Ok(v) => v
                                    .get("data")
                                    .and_then(|d| d.as_array())
                                    .map(|arr| {
                                        arr.iter().any(|m| {
                                            m.get("id").and_then(|s| s.as_str())
                                                == Some(subarg.as_str())
                                                && m.get("is_default")
                                                    .and_then(|b| b.as_bool())
                                                    .unwrap_or(false)
                                        })
                                    })
                                    .unwrap_or(false),
                                Err(_) => false,
                            };
                            if is_default {
                                println!(
                                    "[`{subarg}` is the default model and can't be unloaded; /model default <other> first]"
                                );
                                continue;
                            }
                            match client.unload_model(&subarg).await {
                                Ok(_) => println!("[unloaded `{subarg}`]"),
                                Err(e) => println!("[error: {e}]"),
                            }
                        }
                        // Bare `/model <id>` — quick-switch this chat's routing.
                        _ => {
                            model_label = arg.clone();
                            println!("[switched to model: {model_label}]");
                        }
                    }
                    continue;
                }
                "temp" => {
                    match arg.parse::<f32>() {
                        Ok(t) if (0.0..=2.0).contains(&t) => {
                            temperature = t;
                            println!("[temperature = {temperature}]");
                        }
                        Ok(_) => println!("temperature out of range [0.0, 2.0]"),
                        Err(_) => println!("usage: /temp <0.0..2.0>"),
                    }
                    continue;
                }
                "max_tokens" => {
                    match arg.parse::<u32>() {
                        Ok(n) if n > 0 && n <= 32768 => {
                            max_tokens = n;
                            println!("[max_tokens = {max_tokens}]");
                        }
                        Ok(_) => println!("max_tokens out of range (1..=32768)"),
                        Err(_) => println!("usage: /max_tokens <n>"),
                    }
                    continue;
                }
                // Toggle auto-run-tools (accept proposed tool calls without
                // prompting). `on`/`off` force a state; bare toggles.
                "auto" => {
                    auto_tools = match arg.as_str() {
                        "on" => true,
                        "off" => false,
                        "" => !auto_tools,
                        other => {
                            println!("usage: /auto [on|off]  (got `{other}`)");
                            continue;
                        }
                    };
                    println!("[auto-tools {}]", if auto_tools { "on" } else { "off" });
                    continue;
                }
                "clarify" => {
                    clarify_enabled = match arg.as_str() {
                        "on" => true,
                        "off" => false,
                        "" => !clarify_enabled,
                        other => {
                            println!("usage: /clarify [on|off]  (got `{other}`)");
                            continue;
                        }
                    };
                    println!(
                        "[clarifying questions {}]",
                        if clarify_enabled { "on" } else { "off" }
                    );
                    continue;
                }
                "top_p" => {
                    match arg.parse::<f32>() {
                        Ok(p) if (0.0..=1.0).contains(&p) => {
                            top_p = p;
                            println!("[top_p = {top_p}]");
                        }
                        Ok(_) => println!("top_p out of range [0.0, 1.0]"),
                        Err(_) => println!("usage: /top_p <0.0..1.0>"),
                    }
                    continue;
                }
                "top_k" => {
                    match arg.parse::<u32>() {
                        Ok(k) if k <= 1_000_000 => {
                            top_k = k;
                            println!(
                                "[top_k = {top_k} ({})]",
                                if top_k == 0 { "disabled" } else { "active" }
                            );
                        }
                        Ok(_) => println!("top_k absurdly large"),
                        Err(_) => println!("usage: /top_k <0 = disabled | 1..>"),
                    }
                    continue;
                }
                "repeat_penalty" => {
                    match arg.parse::<f32>() {
                        Ok(rp) if (0.5..=2.0).contains(&rp) => {
                            repeat_penalty = rp;
                            println!("[repeat_penalty = {repeat_penalty}]");
                        }
                        Ok(_) => println!("repeat_penalty out of range [0.5, 2.0]"),
                        Err(_) => println!("usage: /repeat_penalty <0.5..2.0>"),
                    }
                    continue;
                }
                "save" => {
                    if arg.is_empty() {
                        println!("usage: /save <path>");
                        continue;
                    }
                    let last_assistant = messages
                        .iter()
                        .rev()
                        .find(|m| m.role == "assistant")
                        .map(|m| m.content.clone());
                    match last_assistant {
                        Some(text) => {
                            use std::io::Write;
                            let mut f = std::fs::OpenOptions::new()
                                .create(true)
                                .append(true)
                                .open(&arg)?;
                            writeln!(f, "{text}")?;
                            println!("appended to {arg}");
                        }
                        None => println!("no assistant response yet"),
                    }
                    continue;
                }
                "transcript" => {
                    if arg.is_empty() {
                        println!("usage: /transcript <path>");
                        continue;
                    }
                    use std::io::Write;
                    let mut f = std::fs::File::create(&arg)?;
                    writeln!(f, "# rustllama chat transcript")?;
                    writeln!(f, "model: {model_label}\n")?;
                    for m in &messages {
                        writeln!(f, "## {}\n", m.role)?;
                        writeln!(f, "{}\n", m.content)?;
                    }
                    println!("wrote transcript to {arg} ({} messages)", messages.len());
                    continue;
                }
                // `/chat …` mirrors the `rustllama chat` history verbs,
                // operating on the saved-session store + this live buffer:
                //   /chat list             list saved sessions
                //   /chat save [name]      persist the current conversation
                //   /chat resume <name>    replace the buffer with a session
                //   /chat delete <name>    delete a saved session
                "chat" => {
                    let sub = arg.split_whitespace().next().unwrap_or("");
                    let subarg = arg[sub.len()..].trim().to_string();
                    match sub {
                        "" | "list" => match session::list() {
                            Ok(entries) if entries.is_empty() => {
                                println!("(no saved sessions)");
                            }
                            Ok(entries) => {
                                println!("saved sessions (most-recently-updated first):");
                                for e in entries {
                                    println!(
                                        "  {:<24}  model={}  msgs={}  updated=epoch-{}",
                                        e.name, e.model, e.message_count, e.updated_at
                                    );
                                }
                            }
                            Err(e) => println!("[error listing sessions: {e}]"),
                        },
                        "save" => {
                            let name = if subarg.is_empty() {
                                format!("session-{}", session_default_name())
                            } else {
                                subarg
                            };
                            match session::save(&name, &model_label, &messages) {
                                Ok(path) => println!(
                                    "saved session `{name}` ({} messages) to {}",
                                    messages.len(),
                                    path.display()
                                ),
                                Err(e) => println!("[save failed: {e}]"),
                            }
                        }
                        "resume" | "load" => {
                            if subarg.is_empty() {
                                println!("usage: /chat resume <name>     (see /chat list)");
                                continue;
                            }
                            match session::load(&subarg) {
                                Ok(s) => {
                                    messages = s.messages;
                                    if !s.model.is_empty() {
                                        model_label = s.model;
                                    }
                                    println!(
                                        "resumed session `{}` ({} messages, model={})",
                                        s.name,
                                        messages.len(),
                                        model_label
                                    );
                                }
                                Err(e) => println!("[resume failed: {e}]"),
                            }
                        }
                        "delete" => {
                            if subarg.is_empty() {
                                println!("usage: /chat delete <name>     (see /chat list)");
                                continue;
                            }
                            match session::delete(&subarg) {
                                Ok(()) => println!("deleted session `{subarg}`"),
                                Err(e) => println!("[delete failed: {e}]"),
                            }
                        }
                        other => {
                            println!("unknown: /chat {other} (try list|save|resume|delete)");
                        }
                    }
                    continue;
                }
                other => {
                    println!("unknown command: /{other} (try /help)");
                    continue;
                }
            }
        }

        messages.push(ChatMessage {
            role: "user".into(),
            content: trimmed.to_string(),
        });

        // Drive the turn, then service any interactive follow-up (CLARIFY
        // option pick / tool-confirm) by appending a user turn and
        // re-issuing, until the model finishes normally.
        loop {
            let req = ChatRequest {
                model: model_label.clone(),
                messages: messages.clone(),
                temperature: Some(temperature),
                top_p: Some(top_p),
                top_k: Some(top_k),
                max_tokens: Some(max_tokens),
                repeat_penalty: Some(repeat_penalty),
                seed: None,
                stream: true,
                stream_options: Some(rustllama_client::StreamOptions {
                    include_usage: true,
                }),
                // `None` (not `Some(false)`) when off, so the OFF request is
                // wire-identical to the plain path (no `ask_user` tool).
                allow_clarify: clarify_enabled.then_some(true),
            };
            match drive_chat_stream(&client, req, &mut messages).await? {
                StreamOutcome::Normal => break,
                StreamOutcome::AskUser { prompt, options } => {
                    // Record the question as an assistant turn so the
                    // follow-up request is coherent, then read the pick.
                    messages.push(ChatMessage {
                        role: "assistant".into(),
                        content: prompt,
                    });
                    match read_clarify_selection(&mut rl, &options)? {
                        Some(answer) => {
                            println!("\x1b[2m[you chose: {answer}]\x1b[0m");
                            messages.push(ChatMessage {
                                role: "user".into(),
                                content: answer,
                            });
                            // Loop: re-issue with the answer appended.
                        }
                        None => break, // aborted — leave the question in view
                    }
                }
                StreamOutcome::ToolCalls(calls) => {
                    let summary = summarize_tool_calls(&calls);
                    if auto_tools {
                        println!("\x1b[2m[auto-accepted tool call: {summary}]\x1b[0m");
                        break;
                    }
                    println!("[model proposes tool call: {summary}]");
                    let ans = match rl.readline("run it? [y/N] ") {
                        Ok(l) => l,
                        Err(rustyline::error::ReadlineError::Interrupted)
                        | Err(rustyline::error::ReadlineError::Eof) => break,
                        Err(e) => return Err(e.into()),
                    };
                    let approved = matches!(
                        ans.trim().to_ascii_lowercase().as_str(),
                        "y" | "yes" | "a"
                    );
                    if approved {
                        // The REPL has no tool executor, so "approve" just
                        // acknowledges — nothing runs.
                        println!("[approved — the chat has no tool executor, so nothing ran]");
                        break;
                    }
                    // Deny: ask the model to answer directly, then re-issue.
                    messages.push(ChatMessage {
                        role: "user".into(),
                        content: "Please don't run that tool — answer directly instead."
                            .into(),
                    });
                }
            }
        }
        continue;
    }

    if let Some(hp) = history_path {
        let _ = rl.save_history(&hp);
    }
    println!("bye.");
    Ok(())
}

/// Default session name when `/save_session` is invoked without an
/// argument. Uses epoch-seconds so we never collide with an existing
/// session unless the user is calling /save_session twice in the same
/// second (in which case overwriting their own save is reasonable).
fn session_default_name() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "now".into())
}

/// Run one streaming chat round-trip from the REPL: send the request,
/// print tokens as they arrive, and stash the assembled assistant reply
/// onto `messages` (or pop the trailing user turn on error). Shared by
/// the main "user typed a message" path and the `/regenerate` command.
/// One-shot text completion via `POST /v1/completions`. Bypasses the
/// chat template — the prompt goes verbatim. Used by `rustllama
/// generate` for scripts / CI / quick smoke tests.
///
/// Reads stdin when `read_stdin` is true and appends it to `prompt`
/// (separated by `\n` when both are non-empty). When `as_json` is
/// true, the full `CompletionResponse` JSON is printed; otherwise
/// just the generated text (no trailing newline added — clients can
/// pipe directly without trimming).
#[allow(clippy::too_many_arguments)]
async fn generate_oneshot(
    config_path: &std::path::Path,
    prompt: String,
    read_stdin: bool,
    model: Option<String>,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    repeat_penalty: f32,
    seed: Option<u64>,
    suffix: Option<String>,
    stream: bool,
    as_json: bool,
    base_url: Option<String>,
) -> anyhow::Result<()> {
    use rustllama_client::{Client, CompletionRequest};
    use std::io::Read as _;

    if stream && as_json {
        anyhow::bail!("--stream and --json are mutually exclusive");
    }

    // Assemble the prompt: arg + optional stdin (newline-joined).
    let mut full_prompt = prompt;
    if read_stdin {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        let trimmed = buf.trim_end_matches('\n').to_string();
        if !trimmed.is_empty() {
            if !full_prompt.is_empty() {
                full_prompt.push('\n');
            }
            full_prompt.push_str(&trimmed);
        }
    }
    if full_prompt.is_empty() {
        anyhow::bail!("generate: empty prompt (pass arg or `--stdin`)");
    }

    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let server_url = resolve_base_url(&cfg, base_url.as_deref());
    let client = Client::new(server_url.as_str())?;

    let req = CompletionRequest {
        model: model.unwrap_or_default(),
        prompt: full_prompt,
        suffix,
        temperature: Some(temperature),
        top_p: Some(top_p),
        // `0` is the sentinel for "no top-k truncation"; passing
        // `None` drops the field from the wire JSON so the server
        // applies its own default. Mirror the same for
        // repeat_penalty = 1.0 (= "no penalty").
        top_k: if top_k == 0 { None } else { Some(top_k) },
        repeat_penalty: if (repeat_penalty - 1.0).abs() < 1e-6 {
            None
        } else {
            Some(repeat_penalty)
        },
        max_tokens: Some(max_tokens),
        stop: Vec::new(),
        seed,
        stream: false,
    };

    if stream {
        use futures::StreamExt as _;
        use rustllama_client::CompletionEvent;
        use std::io::Write as _;
        let mut stream = client
            .completions_stream(req)
            .await
            .map_err(|e| anyhow::anyhow!("/v1/completions stream failed: {e}"))?;
        while let Some(ev) = stream.next().await {
            match ev {
                Ok(CompletionEvent::Content(t)) => {
                    print!("{t}");
                    let _ = std::io::stdout().flush();
                }
                Ok(CompletionEvent::Finish(_)) => break,
                Ok(CompletionEvent::Error(e)) => {
                    anyhow::bail!("server error: {e}");
                }
                Ok(CompletionEvent::Usage(_)) => {
                    // Stats are tracked server-side; the CLI's
                    // streaming path stays output-pure (just text).
                }
                Err(e) => anyhow::bail!("stream error: {e}"),
            }
        }
        return Ok(());
    }

    let resp = client
        .completions(&req)
        .await
        .map_err(|e| anyhow::anyhow!("/v1/completions failed: {e}"))?;

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::to_value(&CompletionResponseShim {
                id: &resp.id,
                model: &resp.model,
                system_fingerprint: resp.system_fingerprint.as_deref(),
                choices: resp
                    .choices
                    .iter()
                    .map(|c| CompletionChoiceShim {
                        text: &c.text,
                        index: c.index,
                        finish_reason: c.finish_reason.as_deref(),
                    })
                    .collect(),
                usage: resp.usage.as_ref().map(|u| UsageShim {
                    prompt_tokens: u.prompt_tokens,
                    completion_tokens: u.completion_tokens,
                    total_tokens: u.total_tokens,
                    prefill_ms: u.prefill_ms,
                    decode_ms: u.decode_ms,
                    tokens_prefilled: u.tokens_prefilled,
                    cache_hit_tokens: u.cache_hit_tokens,
                }),
            })?)?
        );
    } else {
        // Print just the text from choice[0]. Editor scripts can
        // pipe directly: `rustllama generate "..." | tee out.txt`.
        let text = resp.choices.first().map(|c| c.text.as_str()).unwrap_or("");
        print!("{text}");
        let _ = std::io::Write::flush(&mut std::io::stdout());
    }
    Ok(())
}

/// Local shim structs used to re-serialize the `CompletionResponse`
/// for `--json` output. The client crate's response struct is
/// deserialize-only; round-tripping through these shims avoids
/// adding `Serialize` to the public client types.
#[derive(serde::Serialize)]
struct CompletionResponseShim<'a> {
    id: &'a str,
    model: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_fingerprint: Option<&'a str>,
    choices: Vec<CompletionChoiceShim<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<UsageShim>,
}

#[derive(serde::Serialize)]
struct CompletionChoiceShim<'a> {
    text: &'a str,
    index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'a str>,
}

#[derive(serde::Serialize)]
struct UsageShim {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    prefill_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decode_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tokens_prefilled: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_hit_tokens: Option<u32>,
}

/// One-shot embedding via `POST /v1/embeddings`. Symmetric to
/// `generate_oneshot` for the embedding model. Output modes:
///   - default: prints a summary (`dim=4096, ||v||=1.000, first=[...], last=[...]`)
///   - `--full`: prints every f32 on one line, space-separated
///   - `--json`: prints the full EmbeddingsResponse JSON
#[allow(clippy::too_many_arguments)]
async fn embed_oneshot(
    config_path: &std::path::Path,
    input: String,
    read_stdin: bool,
    model: Option<String>,
    dimensions: Option<u32>,
    as_json: bool,
    full: bool,
    base_url: Option<String>,
) -> anyhow::Result<()> {
    use rustllama_client::{Client, EmbeddingValue, EmbeddingsInput, EmbeddingsRequest};
    use std::io::Read as _;

    if as_json && full {
        anyhow::bail!("--full and --json are mutually exclusive");
    }

    let mut full_input = input;
    if read_stdin {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        let trimmed = buf.trim_end_matches('\n').to_string();
        if !trimmed.is_empty() {
            if !full_input.is_empty() {
                full_input.push('\n');
            }
            full_input.push_str(&trimmed);
        }
    }
    if full_input.is_empty() {
        anyhow::bail!("embed: empty input (pass arg or `--stdin`)");
    }

    let cfg = rustllama_config::load(config_path).unwrap_or_default();
    let server_url = resolve_base_url(&cfg, base_url.as_deref());
    let client = Client::new(server_url.as_str())?;

    let req = EmbeddingsRequest {
        model,
        input: EmbeddingsInput::Single(full_input),
        encoding_format: None,
        dimensions,
    };
    let resp = client
        .embeddings(&req)
        .await
        .map_err(|e| anyhow::anyhow!("/v1/embeddings failed: {e}"))?;

    if as_json {
        // Round-trip through a serializable shim — like generate
        // --json, this avoids adding Serialize to the public
        // deserialize-only response types.
        let vectors: Vec<serde_json::Value> = resp
            .data
            .iter()
            .map(|item| match &item.embedding {
                EmbeddingValue::Floats(v) => serde_json::json!({
                    "object": item.object,
                    "index": item.index,
                    "embedding": v,
                }),
                EmbeddingValue::Base64(s) => serde_json::json!({
                    "object": item.object,
                    "index": item.index,
                    "embedding": s,
                }),
            })
            .collect();
        let payload = serde_json::json!({
            "object": resp.object,
            "data": vectors,
            "model": resp.model,
            "usage": {
                "prompt_tokens": resp.usage.prompt_tokens,
                "total_tokens": resp.usage.total_tokens,
            },
        });
        println!("{}", serde_json::to_string_pretty(&payload)?);
        return Ok(());
    }

    let item = resp
        .data
        .first()
        .ok_or_else(|| anyhow::anyhow!("server returned no embedding vectors"))?;
    let vec_f32: Vec<f32> = match &item.embedding {
        EmbeddingValue::Floats(v) => v.clone(),
        EmbeddingValue::Base64(_) => anyhow::bail!(
            "embedding came back base64-encoded; pass `--json` to inspect raw or omit encoding_format"
        ),
    };

    if full {
        let strs: Vec<String> = vec_f32.iter().map(|x| format!("{x}")).collect();
        println!("{}", strs.join(" "));
    } else {
        // Compact summary: dim, L2 norm, first/last 8 elements.
        let norm: f32 = vec_f32.iter().map(|x| x * x).sum::<f32>().sqrt();
        let take = 8usize.min(vec_f32.len());
        let head = vec_f32[..take]
            .iter()
            .map(|x| format!("{x:.4}"))
            .collect::<Vec<_>>()
            .join(", ");
        let tail = if vec_f32.len() > 2 * take {
            let start = vec_f32.len() - take;
            let s = vec_f32[start..]
                .iter()
                .map(|x| format!("{x:.4}"))
                .collect::<Vec<_>>()
                .join(", ");
            format!(", …, {s}")
        } else {
            String::new()
        };
        println!(
            "dim={}, ||v||={:.4}, first={}: [{}{}]",
            vec_f32.len(),
            norm,
            take,
            head,
            tail,
        );
        println!(
            "(model={}, prompt_tokens={})",
            resp.model, resp.usage.prompt_tokens
        );
    }
    Ok(())
}

/// What a streaming turn ended in — drives the REPL's interactive follow-up
/// (CLARIFY option pick / tool-confirm) after [`drive_chat_stream`] returns.
enum StreamOutcome {
    /// A normal completion (or an error already reported); nothing to prompt.
    Normal,
    /// CLARIFY: the model asked the user a question with selectable options.
    AskUser {
        prompt: String,
        options: Vec<String>,
    },
    /// The model proposed one or more tool calls; the REPL applies its
    /// confirm / auto-run policy over them.
    ToolCalls(Vec<rustllama_client::ToolCall>),
}

async fn drive_chat_stream(
    client: &rustllama_client::Client,
    req: rustllama_client::ChatRequest,
    messages: &mut Vec<rustllama_client::ChatMessage>,
) -> anyhow::Result<StreamOutcome> {
    use futures::StreamExt as _;
    use rustllama_client::{ChatEvent, ChatMessage};
    use std::io::Write as _;

    let mut stream = match client.chat_stream(req).await {
        Ok(s) => s,
        Err(e) => {
            println!("error: {e}");
            // The REPL prepared `messages` so the request would be sent —
            // since the request failed, drop the trailing user turn so
            // the next request doesn't include an unanswered prompt.
            if messages.last().map(|m| m.role.as_str()) == Some("user") {
                messages.pop();
            }
            return Ok(StreamOutcome::Normal);
        }
    };

    let mut content = String::new();
    let mut got_error = false;
    let mut last_usage: Option<rustllama_client::Usage> = None;
    // CLARIFY / tool-confirm signals — surfaced by the client just before
    // Finish. Handled by the caller after this turn's output is flushed.
    let mut ask_user: Option<(String, Vec<String>)> = None;
    let mut tool_calls: Option<Vec<rustllama_client::ToolCall>> = None;
    while let Some(ev) = stream.next().await {
        match ev {
            Ok(ChatEvent::Start(_)) => {}
            Ok(ChatEvent::Content(t)) => {
                print!("{t}");
                let _ = std::io::stdout().flush();
                content.push_str(&t);
            }
            Ok(ChatEvent::AskUser { prompt, options, .. }) => {
                // Echo the question inline (it isn't content); the caller
                // prints the numbered options and reads a selection.
                print!("{prompt}");
                let _ = std::io::stdout().flush();
                ask_user = Some((prompt, options));
            }
            Ok(ChatEvent::ToolCalls(calls)) => {
                tool_calls = Some(calls);
            }
            Ok(ChatEvent::Usage(u)) => {
                // Save for printing after Finish — usage typically
                // arrives between the last Content chunk and Finish;
                // printing it here would interleave with token output.
                last_usage = Some(u);
            }
            Ok(ChatEvent::Finish(_reason)) => break,
            Ok(ChatEvent::Error(e)) => {
                println!("\n[server error: {e}]");
                got_error = true;
                break;
            }
            Err(e) => {
                println!("\n[stream error: {e}]");
                got_error = true;
                break;
            }
        }
    }
    println!();

    // Stats line — only when the server populated usage AND we
    // didn't hit an error. Shows "completion tok @ wallms (tok/s)"
    // so the user gets the same readout an editor would display.
    if !got_error {
        if let Some(u) = last_usage.as_ref() {
            let tok_s = match (u.decode_ms, u.completion_tokens) {
                (Some(ms), n) if ms > 0.0 && n > 0 => Some(n as f64 / (ms / 1000.0)),
                _ => None,
            };
            let mut parts: Vec<String> = Vec::new();
            parts.push(format!("{} tok", u.completion_tokens));
            if let Some(ms) = u.decode_ms {
                parts.push(format!("{:.0}ms decode", ms));
            }
            if let Some(ms) = u.prefill_ms {
                parts.push(format!("{:.0}ms prefill", ms));
            }
            if let Some(s) = tok_s {
                parts.push(format!("{:.1} tok/s", s));
            }
            if let Some(hit) = u.cache_hit_tokens {
                if hit > 0 {
                    parts.push(format!("{hit} cache-hit"));
                }
            }
            println!("\x1b[2m[{}]\x1b[0m", parts.join(", "));
        }
    }

    if got_error {
        if messages.last().map(|m| m.role.as_str()) == Some("user") {
            messages.pop();
        }
        return Ok(StreamOutcome::Normal);
    }
    if !content.is_empty() {
        messages.push(ChatMessage {
            role: "assistant".into(),
            content,
        });
    }
    // CLARIFY takes precedence over a tool-call terminus (mirrors the
    // server): the model paused to ask, so prompt the user for that.
    if let Some((prompt, options)) = ask_user {
        return Ok(StreamOutcome::AskUser { prompt, options });
    }
    if let Some(calls) = tool_calls {
        return Ok(StreamOutcome::ToolCalls(calls));
    }
    Ok(StreamOutcome::Normal)
}

/// Print numbered CLARIFY options and read the user's pick. A bare number
/// selects that option; any other non-empty line is used verbatim as a
/// freeform answer; an empty line defaults to the first option. Returns
/// `None` if the user aborted (Ctrl-C / EOF).
fn read_clarify_selection(
    rl: &mut rustyline::DefaultEditor,
    options: &[String],
) -> anyhow::Result<Option<String>> {
    println!();
    for (i, o) in options.iter().enumerate() {
        println!("  {}. {o}", i + 1);
    }
    let line = match rl.readline("choose> ") {
        Ok(l) => l,
        Err(rustyline::error::ReadlineError::Interrupted)
        | Err(rustyline::error::ReadlineError::Eof) => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(options.first().cloned());
    }
    if let Ok(n) = trimmed.parse::<usize>() {
        if n >= 1 && n <= options.len() {
            return Ok(Some(options[n - 1].clone()));
        }
    }
    Ok(Some(trimmed.to_string()))
}

/// One-line summary of proposed tool calls for the REPL.
fn summarize_tool_calls(calls: &[rustllama_client::ToolCall]) -> String {
    calls
        .iter()
        .map(|c| format!("{}({})", c.name, c.arguments))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authority_host_brackets_ipv6_only() {
        // IPv4 / hostnames pass through untouched.
        assert_eq!(authority_host("127.0.0.1"), "127.0.0.1");
        assert_eq!(authority_host("0.0.0.0"), "0.0.0.0");
        assert_eq!(authority_host("localhost"), "localhost");
        // Bare IPv6 literals get bracketed for the URL/socket authority.
        assert_eq!(authority_host("::1"), "[::1]");
        assert_eq!(authority_host("fe80::1"), "[fe80::1]");
        // Already-bracketed input is left alone (idempotent).
        assert_eq!(authority_host("[::1]"), "[::1]");
    }

    #[test]
    fn classify_swap_target_hub_ref() {
        assert_eq!(
            classify_swap_target(
                "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:qwen2.5-coder-7b-instruct-q4_k_m.gguf"
            ),
            SwapTarget::Hub(
                "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:qwen2.5-coder-7b-instruct-q4_k_m.gguf"
                    .to_string()
            )
        );
    }

    #[test]
    fn classify_swap_target_unix_absolute_path() {
        assert_eq!(
            classify_swap_target("/home/user/models/model.gguf"),
            SwapTarget::Path(std::path::PathBuf::from("/home/user/models/model.gguf"))
        );
    }

    #[test]
    fn classify_swap_target_windows_path() {
        // C:\models\foo.gguf — has a colon but isn't a hub ref.
        let win = "C:\\models\\foo.gguf";
        assert_eq!(
            classify_swap_target(win),
            SwapTarget::Path(std::path::PathBuf::from(win))
        );
        // Forward-slash variant: C:/models/foo.gguf — still a path.
        let win_fwd = "C:/models/foo.gguf";
        assert_eq!(
            classify_swap_target(win_fwd),
            SwapTarget::Path(std::path::PathBuf::from(win_fwd))
        );
    }

    #[test]
    fn classify_swap_target_bare_filename_is_path() {
        // No `/` and no `:` — definitely a path (relative file).
        assert_eq!(
            classify_swap_target("model.gguf"),
            SwapTarget::Path(std::path::PathBuf::from("model.gguf"))
        );
    }

    #[test]
    fn parse_kv_dtype_accepts_known_spellings() {
        use rustllama_engine::KvDtype;
        for s in ["", "f32", "F32", "fp32", "float32"] {
            assert!(
                matches!(parse_kv_dtype(s).unwrap(), KvDtype::F32),
                "expected F32 for {s:?}"
            );
        }
        for s in ["q8_0", "Q8_0", "q8", "int8"] {
            assert!(
                matches!(parse_kv_dtype(s).unwrap(), KvDtype::Q8_0),
                "expected Q8_0 for {s:?}"
            );
        }
        assert!(parse_kv_dtype("q4_0").is_err());
        assert!(parse_kv_dtype("nonsense").is_err());
    }
}
