//! Config schema, load/save, and live-reload watcher.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, RwLock};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub model: ModelConfig,
    pub inference: InferenceConfig,
    pub server: ServerConfig,
    pub ui: UiConfig,
    pub tuning: TuningConfig,
    /// Named profiles that override slices of the top-level config.
    /// Selected at startup via `rustllama --profile <name> …`, or
    /// programmatically via [`Config::apply_profile`]. Each profile
    /// is a sparse [`ProfileOverride`] — only the keys it cares about
    /// are present; everything else inherits the top level.
    #[serde(default)]
    pub profiles: Vec<ProfileOverride>,
    /// Named system-prompt library entries. The GUI Chat page renders
    /// these in a dropdown; selecting one prepends `{role: "system",
    /// content: <body>}` to the message list on every send. The
    /// optional `default_for_model` makes a prompt auto-select when
    /// the named model is loaded (string match against the model id
    /// surfaced by `/v1/models`).
    ///
    /// Example `config.toml`:
    ///
    /// ```toml
    /// [[system_prompts]]
    /// name = "Senior Rust reviewer"
    /// body = "You are a senior Rust engineer reviewing the user's code…"
    /// default_for_model = "qwen2.5-coder-7b-instruct-q4_k_m"
    ///
    /// [[system_prompts]]
    /// name = "Concise"
    /// body = "Answer in two sentences or fewer."
    /// ```
    #[serde(default)]
    pub system_prompts: Vec<SystemPrompt>,
    /// Embedding-model configuration. v1.1 foundation surface — the
    /// `/v1/embeddings` endpoint reads this to decide whether to
    /// 501 with "no embedding model configured" or "configured but
    /// not yet implemented" (BERT-family loader lands in a
    /// follow-up turn). Both fields empty = embeddings disabled.
    #[serde(default)]
    pub embeddings: EmbeddingsConfig,
    /// Reranker-model configuration. Backs `POST /v1/rerank` —
    /// a BGE-reranker-style cross-encoder that scores (query,
    /// document) pairs. Same shape + lazy-load pattern as
    /// `[embeddings]`. Both fields empty = reranker disabled and
    /// `/v1/rerank` returns 501.
    #[serde(default)]
    pub reranker: RerankerConfig,
}

/// Embedding-model configuration. Mirrors `[model]` for chat
/// models — mutually exclusive `path` / `hub` selectors. Loading
/// is not yet implemented; the `/v1/embeddings` endpoint reads
/// these to surface meaningful 501s.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct EmbeddingsConfig {
    /// Absolute path to a GGUF file. Mutually exclusive with `hub`.
    pub path: Option<PathBuf>,
    /// HuggingFace reference of the form `owner/repo:filename`.
    pub hub: Option<String>,
}

/// Reranker-model configuration. Same `path` / `hub` selector as
/// `[embeddings]`. Backs `POST /v1/rerank` (Cohere/Jina shape) —
/// scores (query, document) pairs through the loaded
/// cross-encoder. Points at a BGE-reranker GGUF (or any BERT GGUF
/// that carries a `cls.weight` / `classifier.weight` head).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct RerankerConfig {
    /// Absolute path to a GGUF file. Mutually exclusive with `hub`.
    pub path: Option<PathBuf>,
    /// HuggingFace reference of the form `owner/repo:filename`.
    pub hub: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct SystemPrompt {
    /// Display name shown in the GUI dropdown.
    pub name: String,
    /// Prompt body. Used verbatim as the `content` of a `role:
    /// "system"` message prepended to the request.
    pub body: String,
    /// Optional model id. When non-empty and equal to the active
    /// model's id, the GUI auto-selects this prompt on load. Empty
    /// = no auto-select.
    #[serde(default)]
    pub default_for_model: String,
}

/// A sparse override applied on top of the base [`Config`]. The
/// `[server]` / `[model]` / `[inference]` blocks use serde-flattened
/// optionals so a profile that only changes (say) `[server].port`
/// doesn't have to repeat the full base config.
///
/// Example `config.toml` fragment:
///
/// ```toml
/// [[profiles]]
/// name = "lan"
/// [profiles.server]
/// bind_addr = "0.0.0.0"
/// port = 11500
///
/// [[profiles]]
/// name = "small"
/// [profiles.model]
/// hub = "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF:qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
/// [profiles.inference]
/// ctx_size = 4096
/// ```
///
/// Then: `rustllama --profile lan serve` binds 0.0.0.0:11500;
/// `rustllama --profile small serve` loads the 0.5B model with a
/// 4 K context.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ProfileOverride {
    pub name: String,
    pub model: Option<ProfileModel>,
    pub inference: Option<ProfileInference>,
    pub server: Option<ProfileServer>,
}

/// Sparse `[model]` override. Each field is `Option` so omitting a
/// key leaves the base value intact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ProfileModel {
    pub path: Option<PathBuf>,
    pub hub: Option<String>,
    pub chat_template: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ProfileInference {
    pub n_gpu_layers: Option<u32>,
    pub ctx_size: Option<u32>,
    pub batch_size: Option<u32>,
    pub threads: Option<u32>,
    pub kv_dtype: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ProfileServer {
    pub bind_addr: Option<String>,
    pub port: Option<u16>,
    pub max_pending_per_model: Option<u32>,
    pub max_loaded_models: Option<u32>,
    pub concurrency: Option<u32>,
}

impl Config {
    /// Apply the named profile in-place. Returns `true` if a matching
    /// profile was found and applied, `false` otherwise. Unknown
    /// profile names are not an error — they just no-op so a stale
    /// `--profile` flag in a script doesn't crash startup.
    ///
    /// Application order: profile fields that are `Some(_)` overwrite
    /// the corresponding top-level fields. `None` fields are left
    /// untouched. This mirrors how TOML "deep merge" tools work, with
    /// the difference that profiles can't *remove* a base key —
    /// they can only set new values.
    pub fn apply_profile(&mut self, name: &str) -> bool {
        let Some(profile) = self.profiles.iter().find(|p| p.name == name).cloned() else {
            return false;
        };
        if let Some(m) = profile.model {
            if let Some(p) = m.path {
                self.model.path = Some(p);
            }
            if let Some(h) = m.hub {
                self.model.hub = Some(h);
            }
            if let Some(t) = m.chat_template {
                self.model.chat_template = t;
            }
        }
        if let Some(i) = profile.inference {
            if let Some(v) = i.n_gpu_layers {
                self.inference.n_gpu_layers = v;
            }
            if let Some(v) = i.ctx_size {
                self.inference.ctx_size = v;
            }
            if let Some(v) = i.batch_size {
                self.inference.batch_size = v;
            }
            if let Some(v) = i.threads {
                self.inference.threads = v;
            }
            if let Some(v) = i.kv_dtype {
                self.inference.kv_dtype = v;
            }
        }
        if let Some(s) = profile.server {
            if let Some(v) = s.bind_addr {
                self.server.bind_addr = v;
            }
            if let Some(v) = s.port {
                self.server.port = v;
            }
            if let Some(v) = s.max_pending_per_model {
                self.server.max_pending_per_model = v;
            }
            if let Some(v) = s.max_loaded_models {
                self.server.max_loaded_models = v;
            }
            if let Some(v) = s.concurrency {
                self.server.concurrency = v;
            }
        }
        true
    }

    /// Names of all defined profiles, in declaration order. Used by
    /// the CLI's `config show` to surface what's available without
    /// dumping the full profile bodies.
    pub fn profile_names(&self) -> Vec<&str> {
        self.profiles.iter().map(|p| p.name.as_str()).collect()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    /// Absolute path to a GGUF file on disk. Mutually exclusive with `hub`.
    pub path: Option<PathBuf>,
    /// HuggingFace reference of the form `owner/repo:filename`.
    pub hub: Option<String>,
    /// `"auto"` (read from GGUF), `"chatml"`, `"llama3"`, or an inline Jinja template.
    pub chat_template: String,
    /// Optional vision tower (mmproj GGUF) to attach alongside the
    /// text model. Enables image inputs on `/v1/chat/completions`
    /// (OpenAI image_url content blocks) and `rustllama chat --image`.
    /// Must be the mmproj paired with THIS text model — the loader
    /// verifies the projector/decoder dims and refuses mismatches.
    #[serde(default)]
    pub mmproj: Option<PathBuf>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            path: None,
            hub: None,
            chat_template: "auto".to_string(),
            mmproj: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceConfig {
    /// Advanced OVERRIDE for how many transformer layers are pinned to
    /// GPU VRAM. Its default / unset value is the sentinel
    /// [`N_GPU_LAYERS_AUTO`] (`999`, "all"), which means **AUTO**:
    /// placement is decided by the mandatory first-load autotune + the
    /// measured-perf heat planner (`CpuEngine::auto_place_heat`) — it
    /// fits the model + KV cache into GPU VRAM (all layers on GPU when
    /// they fit, else the largest hot prefix that fits with the coldest
    /// layers spilled to CPU/RAM), honoring `placement.overrides`.
    ///
    /// Set this to any OTHER value to OVERRIDE the auto plan and pin
    /// exactly that many layers to the GPU (`0` = all-CPU). This is an
    /// escape hatch for A/B'ing or scripted runs; leave it unset for the
    /// zero-config default. See
    /// [`InferenceConfig::n_gpu_layers_override`]. On SYCL/CUDA-
    /// unavailable hosts the planner falls back to all-CPU regardless.
    pub n_gpu_layers: u32,
    pub ctx_size: u32,
    pub batch_size: u32,
    pub threads: u32,
    pub flash_attention: bool,
    /// KV-length threshold below which the engine prefers the
    /// standard (3-pass) attention path over flash-decode. Flash has
    /// higher startup cost (online-softmax state init) and only wins
    /// once `kv_len` is large enough that the memory + cache behavior
    /// dominate. Default `256` — measured break-even on AVX-2 x86;
    /// raise on ARM / non-AVX-2 hosts where flash is still scalar.
    ///
    /// Honored alongside the `RUSTLLAMA_FLASH_KV_LEN_MIN` env var the
    /// engine already reads. When both are set, the env var wins
    /// (deliberate: lets the tuner sweep without rewriting config).
    #[serde(default = "default_flash_attention_kv_min")]
    pub flash_attention_kv_min: u32,
    /// GPU indices (in the stable RustLlama enumeration order shown in
    /// the GUI/CLI) to ignore entirely — excluded from the status bar and
    /// skipped for compute dispatch. Lets you pin work to a specific card
    /// on a multi-GPU box (e.g. `[0]` to ignore GPU 0). Empty = use all.
    /// Also settable via `RUSTLLAMA_DISABLED_GPUS` (comma-separated).
    #[serde(default)]
    pub disabled_gpus: Vec<u32>,
    /// Whether the CPU is used as a compute tier at all. `true` (the
    /// default) makes the CPU a first-class EQUAL tier that runs weights
    /// SECONDARY to the GPU — the VRAM-fit planner puts as many layers on
    /// the GPU as fit and spills the remainder onto the CPU/RAM tier.
    ///
    /// `false` ⇒ **GPU-only placement**: no weight layer is placed on the
    /// CPU. If the model + KV cache don't fit in GPU memory the load
    /// **fails with a clear error** instead of silently spilling to the
    /// CPU. (A minimal CPU thread pool is still created for non-weight
    /// host-side ops — tokenization, sampling scratch — it just holds no
    /// model weights.) Also settable via `RUSTLLAMA_CPU_ENABLED=0` and the
    /// `serve --no-cpu` flag.
    #[serde(default = "default_true")]
    pub cpu_enabled: bool,
    /// Whether the GPU is used as a compute/residency tier at all. `true`
    /// (the default) makes every ENABLED GPU a first-class tier the heat
    /// planner places hot weights on. The symmetric mirror of
    /// [`Self::cpu_enabled`]:
    ///
    /// `false` ⇒ **CPU-only placement**: no GPU tier is offered to the
    /// planner, so all weights live on the CPU/RAM tier (equivalent to
    /// forcing `n_gpu_layers = 0`), regardless of any GPU present. Use it
    /// to benchmark the pure-CPU path or to sideline a flaky GPU without
    /// editing the disable-list.
    ///
    /// Setting BOTH `cpu_enabled = false` and `gpu_enabled = false` is a
    /// validation error (there would be no compute tier). Also settable
    /// via `RUSTLLAMA_GPU_ENABLED=0` and the `serve --no-gpu` flag.
    #[serde(default = "default_true")]
    pub gpu_enabled: bool,
    /// Logical-processor indices (0-based, in OS enumeration order —
    /// P/E-core-aware on hybrid Intel parts) to EXCLUDE from the CPU
    /// compute pool. The rayon worker pool is sized to the enabled set
    /// (all logical procs minus this list, further capped by `threads`
    /// when non-zero) and each worker is pinned to a specific enabled
    /// logical processor via OS thread-affinity — so disabled cores run
    /// no kernel work. Empty (the default) = use every logical processor
    /// with no affinity pinning (preserves the historical pool behavior).
    /// Mirrors `disabled_gpus`. Also settable via `RUSTLLAMA_DISABLED_CPUS`
    /// (comma-separated) and the `serve --disabled-cpus <CSV>` flag.
    #[serde(default)]
    pub disabled_cpus: Vec<u32>,
    /// VRAM-only weight residency. `false` (the default) keeps the normal
    /// RAM/mmap + USM residency path. `true` forbids host-RAM weight
    /// residency and mmap: weights must live in **dedicated** GPU VRAM
    /// (device-local USM). If the weights don't fit dedicated VRAM the
    /// load **fails with a clear error**.
    ///
    /// DEGENERATE on **unified-memory** GPUs (integrated parts such as the
    /// Iris Xe, whose "VRAM" is a shared-LPDDR aperture with zero *separate*
    /// dedicated VRAM): there is no distinct device memory to pin weights
    /// into, so `vram_only` is treated as a **no-op and a warning is
    /// emitted**, and the load proceeds on the normal path. Also settable
    /// via `RUSTLLAMA_VRAM_ONLY=1` and the `serve --vram-only` flag.
    #[serde(default)]
    pub vram_only: bool,
    pub placement: PlacementConfig,
    /// Reuse the previous generation's KV cache when the new prompt shares
    /// a prefix with the previous one. Huge speedup for chat / coding-tool
    /// flows that resend long system prompts each turn. Default: true.
    pub prefix_cache: bool,
    /// How many cross-request KV snapshots to retain. Each snapshot is
    /// sized to the conversation it covers (not `ctx_size`), so memory
    /// scales with how many distinct prompt prefixes you keep warm.
    /// 0 disables the multi-snapshot pool — the engine still does single-
    /// snapshot LCP against the most recent request via `last_ids`.
    /// Default: 4 (≈ "two concurrent threads + scratch generation").
    pub prefix_cache_max_snapshots: u32,
    /// Dtype of the KV cache. `"tq4"` is the workspace default —
    /// `"q8_0"` quantizes each K/V row to 8 bits + a per-row f32 scale,
    /// cutting KV memory to ~26% of F32 at the cost of inline dequant
    /// in the attention loop. Drop-in: same forward semantics, just
    /// smaller. Greedy parity with F32 is asserted by the engine's
    /// integration test suite, and the kernel layer ships scalar /
    /// AVX2 / AVX-512 attention paths that all round-trip identically.
    ///
    /// Sample `config.toml` snippet:
    ///
    /// ```toml
    /// [inference]
    /// ctx_size = 32768       # fits ~4x more context in the same RAM at q8_0
    /// kv_dtype = "q8_0"
    /// ```
    ///
    /// Hot-load via the server endpoint:
    ///
    /// ```text
    /// POST /v1/internal/models/load
    /// { "model_id": "qwen2.5-coder-7b", "kv_dtype": "q8_0" }
    /// ```
    pub kv_dtype: String,
    /// Optional explicit K-cache dtype. When unset (the default),
    /// the engine uses [`Self::kv_dtype`] for both K and V. Set this
    /// alongside [`Self::v_dtype`] to use distinct per-channel
    /// dtypes — e.g. `k_dtype = "tq4"`, `v_dtype = "q8_0"` (V
    /// matters more for attention output quality, so a heavier
    /// V dtype + lighter K dtype is a common quality/memory
    /// trade-off).
    ///
    /// **Status:** the config + GUI accept split dtypes today, but
    /// engine storage couples K and V into a single `KvLayer` enum
    /// variant per dtype. When the resolved K and V differ, K's
    /// dtype is what actually reaches the attention kernels and a
    /// `WARN` is emitted at load. Per-side KV storage (the path
    /// that makes V's setting take effect) is a follow-up.
    #[serde(default)]
    pub k_dtype: Option<String>,
    /// Optional explicit V-cache dtype. See [`Self::k_dtype`].
    #[serde(default)]
    pub v_dtype: Option<String>,
    /// Optional K-cache mean-centering bias sidecar (GGUF, fork
    /// `kv_bar` format — see `rustllama-models::kv_bias`). When unset,
    /// the engine auto-discovers `<model stem>.kvbias.gguf` next to
    /// the model file. Applied only for quantized (q4_0) KV caches;
    /// exactly softmax-invariant. Disable at runtime with
    /// `RUSTLLAMA_KV_BIAS=0`.
    #[serde(default)]
    pub kv_bias_path: Option<String>,
    /// Cap on complete tool-call bodies the streaming grammar will
    /// accept per response. `0` disables the cap. Defaults to `8` —
    /// covers "agent picks a tool, sees the result, picks another"
    /// patterns with a couple of corrective retries. Set lower for
    /// strict single-shot tool use, higher for chains that legitimately
    /// need many calls.
    pub max_tool_iterations: u32,
    /// Keep quantized weights in their raw GGUF block form even when
    /// a tensor would fit in L3 cache after dequant-to-F16. Trades a
    /// small matvec speed-up on small tensors (attn_k / attn_v on
    /// smaller models) for **lower RAM usage** — typically saves
    /// 1-3 GB on a 7B-24B model. Recommended on hosts with <= 16 GB
    /// RAM running quantized models that approach the memory budget.
    /// Default: `false` (favor speed).
    pub keep_quant_raw: bool,
    /// KV-cache memory layout.
    ///
    /// - `"contiguous"` (default) — per-request `[n_kv_heads,
    ///   max_ctx, head_dim]` slab per layer. Simple, fast, but pins
    ///   `max_ctx`-sized memory per request regardless of how long
    ///   the conversation actually runs.
    /// - `"paged"` — fixed-size pages (default 16 tokens) allocated
    ///   from a shared pool. Each request holds a list of pages
    ///   and grows its allocation as the conversation extends. Same
    ///   forward-pass semantics (the engine gathers paged data
    ///   back into the slab shape the existing attention kernels
    ///   expect); the win is memory efficiency for short
    ///   conversations and the foundation for continuous
    ///   batching's shared KV pool.
    ///
    /// **Status (v1):** the config field accepts both values; the
    /// model's `forward_*_paged_f32` paths are bit-identical to
    /// the contiguous path for F32 KV. Engine consumption (single-
    /// slot generate loop dispatch) lands in 3.6d. Multi-slot
    /// continuous batching that actually benefits from paged
    /// storage lands in 3.7. Until then, `"paged"` runs the same
    /// single-request flow as `"contiguous"` — just routed through
    /// the page indirection so the wiring is exercised in real
    /// workloads before multi-slot demand on it.
    ///
    /// Sample `config.toml` snippet:
    ///
    /// ```toml
    /// [inference]
    /// kv_cache_layout = "paged"
    /// ```
    #[serde(default = "default_kv_cache_layout")]
    pub kv_cache_layout: String,
    /// Paged-KV page size in tokens. Ignored when
    /// `kv_cache_layout = "contiguous"`. Smaller pages reduce
    /// fragmentation on short conversations; larger pages improve
    /// SLM-tile efficiency in the GPU attention kernel. The vLLM-
    /// convention default of 16 is a reasonable starting point;
    /// `rustllama tune --kv-page-size` sweeps `{8, 16, 32, 64}` and
    /// caches the winner. `0` falls back to the built-in default
    /// inside `KvBackend::from_inference_config_with_page_size`.
    #[serde(default = "default_kv_page_size")]
    pub kv_page_size: u32,
    /// Enable n-gram (prompt-lookup) speculative decoding for text
    /// generation. Default `false`. When `true`, the engine drafts
    /// tokens by matching the trailing n-gram against earlier history
    /// and verifies them in one batched target forward per round — a
    /// pure win on repetitive / code-like output, with no second model.
    ///
    /// CAVEAT: the speculative verify samples from the target's raw
    /// softmax, so per-request `temperature` / `top_k` / `top_p` are not
    /// applied on this path (grammar-constrained requests automatically
    /// fall back to the classic sampler). Use when throughput on
    /// repetitive output matters more than exact sampling-temperature
    /// fidelity; a shared speculation-temperature knob is a follow-up.
    #[serde(default)]
    pub speculative_ngram: bool,
    /// MTP / NextN self-speculative decoding for hybrid models that
    /// carry a NextN head (`qwen35moe`-family `blk.{N}.nextn.*`).
    /// Default `false`. When `true`, grammar-free requests on a hybrid
    /// model with a NextN head route through the MTP driver: each main
    /// forward also runs the cheap NextN head to draft the +2 token,
    /// which the next forward verifies (accept commits 2 tokens per
    /// round, reject falls back to 1). Silently ignored on models
    /// without a NextN head (falls back to classic decode). Takes
    /// precedence over `speculative_ngram` when both are set. Same
    /// raw-softmax sampling caveat as the other speculative paths.
    #[serde(default)]
    pub speculative_mtp: bool,
    /// Use the chunked-parallel SSM prefill scan for hybrid (DeltaNet)
    /// models. Autotuner-selected; promoted to RUSTLLAMA_SSM_PREFILL_CHUNKED
    /// at serve load. Default false.
    #[serde(default)]
    pub ssm_prefill_chunked: bool,
    /// Trailing-token match window for the n-gram drafter. Default 3 —
    /// good for code without ballooning lookup cost. Ignored when
    /// `speculative_ngram = false`.
    #[serde(default = "default_ngram_n_match")]
    pub ngram_n_match: u32,
    /// Max tokens the n-gram drafter proposes per round. Default 4.
    /// Higher = bigger win on a successful speculation, more wasted
    /// verify compute on a miss. Ignored when `speculative_ngram = false`.
    #[serde(default = "default_ngram_n_draft")]
    pub ngram_n_draft: u32,
    /// Optional draft model GGUF for two-model speculative decoding.
    /// The draft must share the target's tokenizer EXACTLY (vocab
    /// size + special ids — verified at load, refused otherwise).
    /// When set, grammar-free requests draft `speculative_draft_k`
    /// tokens on the small model and verify them in one batched
    /// forward on the target; takes precedence over
    /// `speculative_ngram`. Same raw-softmax sampling caveat as the
    /// n-gram path.
    #[serde(default)]
    pub speculative_draft_path: Option<String>,
    /// Candidates per speculative round for the draft-model path.
    /// Default 8. Ignored when `speculative_draft_path` is unset.
    #[serde(default = "default_speculative_draft_k")]
    pub speculative_draft_k: u32,
    /// MoE expert-pin cache budget in MB. `0` (default) disables the
    /// cache. When set on a MoE model, the engine VirtualLocks the
    /// observed-hot routed experts' weight pages in RAM (LRU-evicting
    /// cold ones over budget), learns routing frequencies across
    /// sessions via a `<gguf>.rlusage` sidecar, pre-pins the learned
    /// hottest experts at load, and prefetches upcoming experts on a
    /// readahead thread. The big lever for MoE models that exceed —
    /// or crowd — physical RAM: ~20% of experts serve ~80% of tokens,
    /// and the OS pager doesn't know which. Requires file-backed
    /// (zero-copy) expert weights; the engine auto-enables zero-copy
    /// when this is set. Promoted to `RUSTLLAMA_MOE_EXPERT_CACHE_MB`
    /// at startup (a user-set env var wins).
    #[serde(default)]
    pub moe_expert_cache_mb: u64,
    /// Borrow raw quantized weights directly from the GGUF mmap
    /// instead of copying to owned heap (clean file-backed pages —
    /// never hit the pagefile). Required by the expert-pin cache
    /// (auto-enabled when `moe_expert_cache_mb > 0`). Promoted to
    /// `RUSTLLAMA_ZEROCOPY_WEIGHTS` at startup. Default `false`.
    #[serde(default)]
    pub zerocopy_weights: bool,
    /// VirtualLock hot weight tiers into RAM: `"0"`/empty = off
    /// (default), `"auto"` = lock the Tier-0 always-hot set within
    /// available RAM, or a number = MB budget. Promoted to
    /// `RUSTLLAMA_LOCK_RAM_MB` at startup (a user-set env var wins).
    #[serde(default = "default_lock_ram_mb")]
    pub lock_ram_mb: String,
    /// Batched chunk prefill: forward each prefill chunk through the
    /// multi-query flash-prefill kernels (one attention call per
    /// layer per chunk) instead of a serial `forward_one` per token.
    /// Applies to non-hybrid models on the contiguous KV backend, all
    /// KV dtypes. Bit-identical to the serial path up to fp reduction
    /// order (pinned by the model crate's parity tests). Promoted to
    /// `RUSTLLAMA_PREFILL_BATCHED` at startup (a user-set env var
    /// wins — set it to `0` to force the serial per-token loop).
    /// Default `true`: a large prefill speedup on long prompts, with
    /// prefill cancellation granularity becoming the chunk instead of
    /// the token.
    #[serde(default = "default_true")]
    pub prefill_batched: bool,
    /// Chunked prefill for **hybrid** (DeltaNet + MoE) models:
    /// grouped-expert MoE execution — each selected expert's weights
    /// are read once per chunk instead of once per token — plus
    /// layer-ahead expert-pool readahead. The dominant prefill win on
    /// qwen35moe-family models. Default `false` for one release:
    /// numerically it matches the serial path up to fp-association
    /// order, but this family's output quality has been sensitive, so
    /// flip it on after an A/B against `false` on your model (same
    /// prompt, compare coherence). Promoted to
    /// `RUSTLLAMA_PREFILL_BATCHED_HYBRID` at startup (a user-set env
    /// var wins).
    #[serde(default = "default_prefill_batched_hybrid")]
    pub prefill_batched_hybrid: bool,
    /// Warm-restart KV persistence (roadmap Phase 5, Colibrì-style):
    /// cap in MB for the `<gguf>.rlkv` sidecar the engine writes at
    /// graceful shutdown and reloads at start — a restarted server's
    /// first request whose prompt extends a persisted conversation
    /// skips re-prefill entirely. `0` (default) disables. v1 persists
    /// contiguous F32 KV only (the hybrid/qwen35moe configuration;
    /// hybrid entries carry their DeltaNet anchor state too). A hard
    /// kill loses the session — the write happens on drop. Promoted
    /// to `RUSTLLAMA_KV_PERSIST_MB` at startup (a user-set env var
    /// wins).
    #[serde(default)]
    pub kv_persist_mb: u64,
    /// Memory-budget mode: `"manual"` (default) keeps the explicit
    /// knobs (`moe_expert_cache_mb`, `lock_ram_mb`) authoritative;
    /// `"auto"` derives the expert-cache budget at engine load from
    /// measured available RAM minus an honest projection of the KV
    /// cache, recurrent state, forward scratch, and an OS reserve —
    /// so cache budgets never jointly overcommit the box. In auto
    /// mode a non-zero `moe_expert_cache_mb` acts as a *cap* on the
    /// computed budget. Promoted to `RUSTLLAMA_MEMORY_BUDGET` at
    /// startup (a user-set env var wins).
    #[serde(default = "default_memory_budget")]
    pub memory_budget: String,
}

fn default_memory_budget() -> String {
    "manual".to_string()
}

fn default_true() -> bool {
    true
}

/// Sentinel value for [`InferenceConfig::n_gpu_layers`] meaning "AUTO" —
/// placement is decided by the autotune + heat planner rather than pinned
/// to a fixed layer count. Historically the default ("all layers"); any
/// other configured value is treated as an explicit user override.
pub const N_GPU_LAYERS_AUTO: u32 = 999;

impl InferenceConfig {
    /// The explicit `n_gpu_layers` override, or `None` when the value is
    /// the [`N_GPU_LAYERS_AUTO`] sentinel (⇒ let the autotune + heat
    /// planner decide placement). `Some(n)` pins exactly `n` layers to the
    /// GPU, bypassing the auto plan.
    pub fn n_gpu_layers_override(&self) -> Option<u32> {
        if self.n_gpu_layers == N_GPU_LAYERS_AUTO {
            None
        } else {
            Some(self.n_gpu_layers)
        }
    }
}

fn default_prefill_batched_hybrid() -> bool {
    // Best-settings default: identity-verified on both hybrid archs,
    // ~1.5×+ prefill, no-op on non-hybrid. See the field doc.
    true
}

fn default_lock_ram_mb() -> String {
    // "auto" sizes + VirtualLocks the hot weight tier into RAM so
    // decode doesn't re-fault weight pages under memory pressure
    // (the thrash mode). Fail-soft: a failed lock just logs and
    // continues, so this is a safe universal default (best-settings
    // bake-in). Set "0" to disable, or an explicit MB cap.
    "auto".to_string()
}

fn default_kv_cache_layout() -> String {
    "contiguous".to_string()
}

fn default_flash_attention_kv_min() -> u32 {
    256
}

fn default_kv_page_size() -> u32 {
    16
}

fn default_ngram_n_match() -> u32 {
    3
}

fn default_speculative_draft_k() -> u32 {
    8
}

fn default_ngram_n_draft() -> u32 {
    4
}

impl InferenceConfig {
    /// Resolve the per-channel K and V dtype strings. `k_dtype` /
    /// `v_dtype` override `kv_dtype` when explicitly set; otherwise
    /// both fall back to `kv_dtype`. The returned strings are still
    /// raw config strings — callers parse them via
    /// `rustllama_engine::KvDtype::parse`. When K and V differ,
    /// callers should warn (current engine storage couples K and V,
    /// so K's dtype is what reaches the kernels) — see the docstring
    /// on [`Self::k_dtype`] for the roadmap.
    pub fn resolved_kv_dtypes(&self) -> (&str, &str) {
        let k = self.k_dtype.as_deref().unwrap_or(self.kv_dtype.as_str());
        let v = self.v_dtype.as_deref().unwrap_or(self.kv_dtype.as_str());
        (k, v)
    }
}

/// Coherence guardrail for the KV-cache dtype (productionization 1d).
///
/// Aggressive quantized KV caches (`q4_0`, `tq1`/`tq2`/`tq4`/`tq8`,
/// `nvfp4`) drift into gibberish on models NOT calibrated for them —
/// they assume the whitening + mean-centering-bias calibration tuned
/// for Bonsai's head_dim 256. On e.g. Qwen 2.5 (head_dim 128 GQA) a
/// 4-bit KV degrades to nonsense within a few dozen tokens
/// (root-caused 2026-09-20). This picks the SAFE effective dtype so a
/// mis-set config auto-adjusts instead of silently producing garbage.
///
/// Returns `(effective_dtype, Option<warning>)`:
/// - `f32` / `q8_0` (universally coherent) → returned unchanged.
/// - an aggressive quant that is VALIDATED → returned unchanged.
///   Validated means either `force` is set (env
///   `RUSTLLAMA_FORCE_QUANT_KV=1` / an explicit "I checked this" opt-in)
///   or a `<model_stem>.kvbias.gguf` calibration sidecar sits beside
///   the model (the Bonsai case — that sidecar is exactly the
///   calibration that makes 4-bit KV coherent).
/// - an aggressive quant that is NOT validated → downgraded to `f32`
///   with a warning string describing why.
pub fn coherence_safe_kv_dtype(
    requested: &str,
    model_path: &std::path::Path,
    force: bool,
) -> (String, Option<String>) {
    let r = requested.trim().to_ascii_lowercase();
    // Universally coherent: exact F32, or 8-bit (negligible drift) —
    // q8_0 and MXFP8 (E4M3 8-bit + per-32 E8M0 scale) are both ~8-bit
    // and don't need per-model calibration. mxfp4/mxfp6 stay aggressive.
    if r == "f32" || r == "q8_0" || r == "mxfp8" {
        return (requested.to_string(), None);
    }
    if force {
        return (requested.to_string(), None);
    }
    // Validated when the model ships a KV mean-centering-bias sidecar
    // (`<stem>.kvbias.gguf`) — the calibration that makes quant KV
    // coherent for that specific model.
    let sidecar = model_path.with_extension("kvbias.gguf");
    if sidecar.is_file() {
        return (requested.to_string(), None);
    }
    (
        "f32".to_string(),
        Some(format!(
            "KV dtype '{requested}' is an aggressive quantized cache not validated for this model \
             (no '{}' calibration sidecar); it can produce incoherent output on uncalibrated \
             models. Auto-using f32 for coherence. Set RUSTLLAMA_FORCE_QUANT_KV=1 to override if \
             you have verified this model.",
            sidecar
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
        )),
    )
}

/// True when the user explicitly opted into unvalidated quantized KV
/// via `RUSTLLAMA_FORCE_QUANT_KV`.
pub fn force_quant_kv_from_env() -> bool {
    std::env::var("RUSTLLAMA_FORCE_QUANT_KV")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

#[cfg(test)]
mod kv_guardrail_tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn f32_and_q8_0_always_pass() {
        let p = Path::new("C:/models/whatever.gguf");
        assert_eq!(coherence_safe_kv_dtype("f32", p, false).0, "f32");
        assert_eq!(coherence_safe_kv_dtype("q8_0", p, false).0, "q8_0");
        assert!(coherence_safe_kv_dtype("q8_0", p, false).1.is_none());
    }

    #[test]
    fn unvalidated_quant_downgrades_with_warning() {
        // No sidecar next to this (nonexistent) path → downgrade.
        let p = Path::new("C:/models/qwen2.5-coder.gguf");
        let (dt, warn) = coherence_safe_kv_dtype("q4_0", p, false);
        assert_eq!(dt, "f32");
        assert!(warn.is_some());
        for d in ["tq4", "tq1", "nvfp4"] {
            assert_eq!(coherence_safe_kv_dtype(d, p, false).0, "f32");
        }
    }

    #[test]
    fn force_overrides_downgrade() {
        let p = Path::new("C:/models/qwen2.5-coder.gguf");
        assert_eq!(coherence_safe_kv_dtype("q4_0", p, true).0, "q4_0");
        assert!(coherence_safe_kv_dtype("q4_0", p, true).1.is_none());
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            // Sentinel `N_GPU_LAYERS_AUTO` ("all"): unset ⇒ AUTO placement
            // via the mandatory first-load autotune + the measured-perf heat
            // planner (fit the model + KV cache into GPU VRAM, spill the
            // coldest layers to CPU/RAM). Any OTHER value is an explicit
            // user override — see the field doc + `n_gpu_layers_override`.
            n_gpu_layers: N_GPU_LAYERS_AUTO,
            ctx_size: 8192,
            batch_size: 512,
            threads: 0,
            flash_attention: true,
            flash_attention_kv_min: 256,
            disabled_gpus: Vec::new(),
            // CPU is a first-class EQUAL tier, enabled by default and used
            // secondary to the GPU (GPU-primary + CPU spill).
            cpu_enabled: true,
            // GPU tier enabled by default (mirror of cpu_enabled). false ⇒
            // CPU-only placement. Both false is a validation error.
            gpu_enabled: true,
            disabled_cpus: Vec::new(),
            vram_only: false,
            placement: PlacementConfig::default(),
            prefix_cache: true,
            prefix_cache_max_snapshots: 4,
            // F32 is the default KV dtype: universally coherent on
            // every model/architecture. Quantized KV caches (tq4 /
            // q4_0 / q8_0) cut KV memory but are only safe where
            // validated per-model — q4_0/tq4 assume the whitening +
            // mean-centering-bias calibration tuned for Bonsai's
            // head_dim 256, and on other models (e.g. Qwen2.5,
            // head_dim 128 GQA) 4-bit KV drifts into gibberish over
            // a few dozen tokens (confirmed 2026-09-20). So the safe
            // default is exact F32; opt into a quantized KV per-model
            // via `[inference].kv_dtype` once you've checked coherence.
            // The Settings page exposes this field as a dropdown.
            kv_dtype: "f32".to_string(),
            k_dtype: None,
            v_dtype: None,
            kv_bias_path: None,
            max_tool_iterations: 8,
            keep_quant_raw: false,
            kv_cache_layout: default_kv_cache_layout(),
            kv_page_size: default_kv_page_size(),
            speculative_ngram: false,
            speculative_mtp: false,
            ssm_prefill_chunked: false,
            speculative_draft_path: None,
            speculative_draft_k: default_speculative_draft_k(),
            ngram_n_match: 3,
            ngram_n_draft: 4,
            moe_expert_cache_mb: 0,
            zerocopy_weights: false,
            lock_ram_mb: default_lock_ram_mb(),
            prefill_batched: true,
            // Batched/hoisted hybrid-chunk prefill: identity-verified
            // token-for-token vs the per-token path on both hybrid
            // architectures (Bonsai dense-FFN + Qwen3.6 MoE), ~1.5×+
            // prefill; no-op on non-hybrid models. Default ON as a
            // best-settings bake-in (was gated OFF pending that A/B,
            // which passed 2026-09-20).
            prefill_batched_hybrid: true,
            kv_persist_mb: 0,
            memory_budget: default_memory_budget(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct PlacementConfig {
    pub overrides: Vec<TensorOverride>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorOverride {
    pub pattern: String,
    pub device: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub bind_addr: String,
    pub port: u16,
    pub api_key: String,
    pub cors_origins: Vec<String>,
    /// Per-model cap on `pending + in-flight` requests. Requests beyond
    /// this depth get rejected with `503` + `Retry-After: 2`. Tune up
    /// for high-throughput batch jobs; tune down for interactive UIs
    /// that prefer immediate-error over long waits.
    pub max_pending_per_model: u32,
    /// Cap on simultaneously warm-loaded models. When the cap is hit,
    /// new model loads evict the least-recently-resolved entry (except
    /// the current default, which is always pinned). `0` disables the
    /// cap — the registry grows unbounded.
    pub max_loaded_models: u32,
    /// Number of concurrent generations a single model serves in
    /// parallel. `1` (the default) is the safe single-flight path: one
    /// in-flight request per model, others queue on the gate. `>1`
    /// activates the multi-flight pool — each slot owns an independent
    /// KV cache via `CpuEngine::fork_for_concurrent_use`, so memory
    /// cost grows roughly linearly with concurrency (one KV cache per
    /// fork). Tune up for multi-user serving, leave at `1` for
    /// single-developer use (where one request at a time + the prefix
    /// cache gives the best end-to-end throughput).
    pub concurrency: u32,
    /// Use fused multi-slot decode instead of the per-fork
    /// multi-flight pool when serving concurrent requests. Requires
    /// `[inference] kv_cache_layout = "paged"` — fused decode is
    /// only wired for paged KV today.
    ///
    /// When `true` + paged KV: `ServingModel` builds a single
    /// `PagedBatchEngine` with `max_slots = concurrency`. M
    /// concurrent HTTP requests share one paged KV pool + one
    /// driver thread, with one fused decode kernel-launch
    /// sequence per tick (instead of M independent forks each
    /// issuing their own launches). On integrated GPUs where
    /// kernel-launch overhead is a fraction of decode time, this
    /// is the headline CB throughput win.
    ///
    /// When `false` (the default): falls back to the existing
    /// per-fork multi-flight pool. Concurrent requests still run
    /// in parallel, but each owns its own KV state — works with
    /// any `kv_cache_layout` (contiguous or paged).
    ///
    /// Requires `[inference].kv_cache_layout = "paged"` (the loader
    /// warns and ignores the flag otherwise, since fused decode shares
    /// one paged KV pool across concurrent requests).
    #[serde(default)]
    pub fused_decode: bool,
    /// Per-request audit log. When `true`, the server appends a
    /// JSONL entry (`ts_ms`, `method`, `path`, `query`, `status`,
    /// `latency_ms`) to [`Self::audit_log_path`] after each
    /// request. The Authorization header value is never logged;
    /// `api_key=…` query params are redacted to `api_key=***`.
    /// Off by default — opt-in because the file grows unbounded.
    /// Recommended companion to LAN-bind mode (where you want to
    /// know who hit what).
    #[serde(default)]
    pub audit_log: bool,
    /// Destination for the audit log. Empty = derive a default
    /// under the user-data dir (`<crash_log_dir>/audit.log.jsonl`).
    /// Honored only when [`Self::audit_log`] is true.
    #[serde(default)]
    pub audit_log_path: String,
    /// Per-key request rate limit (requests/minute). `0` (default)
    /// disables the limiter; non-zero caps the configured api_key's
    /// admission rate via a token bucket. Capacity equals the
    /// per-minute value so a single burst can use a minute's worth of
    /// budget. Exceeded requests get `429 Too Many Requests` with a
    /// `Retry-After` header.
    ///
    /// v1 has a single configured api_key, so the bucket is process-
    /// global. When multi-key auth lands, the limiter becomes per-key
    /// via a `HashMap<key_hash, Bucket>`.
    #[serde(default)]
    pub rate_limit_per_minute: u32,
    /// Process memory cap in MiB, enforced via the Windows Job Object
    /// sandbox. The OS kills the process if its working set exceeds
    /// this. `0` (default) disables the cap; on non-Windows targets
    /// the field is accepted but ignored.
    ///
    /// Defends against a malformed GGUF triggering a runaway
    /// allocation. 32 GiB is the recommended default for production;
    /// development laptops with 16 GiB total RAM should set 8000
    /// (8 GiB) to leave headroom for the OS.
    #[serde(default)]
    pub sandbox_memory_limit_mb: u32,
    /// Inject a concise, clearly-labeled `[Host environment]` system
    /// block into tool/function-calling requests describing the SERVER
    /// host (OS + arch + detected shells/tools) so the model picks the
    /// right shell syntax (PowerShell on Windows, bash on Linux) when it
    /// emits a shell/command tool call. `true` by default.
    ///
    /// NOTE: this describes the SERVER's host. For remote deployments
    /// where the tools actually execute on some OTHER machine, the
    /// operator should set this to `false` so the hint doesn't mislead
    /// the model about the execution environment.
    #[serde(default = "default_true")]
    pub tool_environment_hint: bool,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind_addr: "127.0.0.1".to_string(),
            port: 11434,
            api_key: String::new(),
            cors_origins: Vec::new(),
            max_pending_per_model: 4,
            max_loaded_models: 4,
            // Default to 1: all forks share the model's single
            // serial SyclWorker thread, so extra forks buy ZERO
            // request parallelism today — they only multiply KV-cache
            // RAM (one full max_ctx cache each) and let requests 2..N
            // pass admission just to sit invisibly in the worker's
            // queue with no first token. Metrics polling and /api/tags
            // don't need forks (they don't take the generation gate).
            // Raise this only alongside real multi-worker execution.
            concurrency: 1,
            fused_decode: false,
            audit_log: false,
            audit_log_path: String::new(),
            rate_limit_per_minute: 0,
            sandbox_memory_limit_mb: 0,
            tool_environment_hint: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct UiConfig {
    pub theme: String,
    pub font_size: u32,
    pub code_theme: String,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            theme: "system".to_string(),
            font_size: 14,
            code_theme: "github-dark".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TuningConfig {
    pub opportunistic_refine: bool,
    /// When `true` (default), the engine overrides
    /// `[inference].n_gpu_layers` with the per-model placement
    /// winner from the tuner cache if one exists for the current
    /// device + model. Run `rustllama tune --placement --measure`
    /// to populate. Set `false` to keep config-only behavior
    /// (useful for A/B'ing or scripted runs that need a known
    /// value).
    #[serde(default = "default_true_serde")]
    pub auto_apply_placement: bool,
    /// When `true` (default), the engine overrides
    /// `[inference].batch_size` with the per-device prefill-chunk
    /// winner from the tuner cache if one exists. Run
    /// `rustllama tune --batch-size` to populate.
    #[serde(default = "default_true_serde")]
    pub auto_apply_batch_size: bool,
    /// When `true` (default), the engine overrides
    /// `[inference].kv_dtype` with the per-device + per-model winner
    /// from the tuner cache when one exists. Run
    /// `rustllama tune --kv-dtype` (or `--all`) to populate. Cache
    /// entries are strings: `"f32"`, `"q8_0"`, `"tq1"`...`"tq8"`,
    /// `"nvfp4"`.
    #[serde(default = "default_true_serde")]
    pub auto_apply_kv_dtype: bool,
    /// When `true` (default), the engine overrides
    /// `[inference].flash_attention` with the per-device winner
    /// from the tuner cache. Flash-attention's win/loss varies by
    /// kernel and context length — let the tuner decide.
    #[serde(default = "default_true_serde")]
    pub auto_apply_flash_attention: bool,
    #[serde(default = "default_true_serde")]
    pub auto_apply_speculative_mtp: bool,
    #[serde(default = "default_true_serde")]
    pub auto_apply_ssm_prefill_chunked: bool,
    /// When `true` (default), the engine overrides
    /// `[inference].kv_cache_layout` with the cached winner.
    /// Contiguous wins for single-flight today; paged is wired but
    /// gates on continuous batching.
    #[serde(default = "default_true_serde")]
    pub auto_apply_kv_cache_layout: bool,
    /// When `true` (default), the engine overrides
    /// `[inference].flash_attention_kv_min` with the cached winner.
    /// Tuner sweep picks the host-specific break-even crossover
    /// for flash-decode vs standard attention.
    #[serde(default = "default_true_serde")]
    pub auto_apply_flash_kv_min: bool,
    /// When `true` (default), the engine overrides
    /// `[inference].prefix_cache_max_snapshots` with the cached
    /// winner. Workload-dependent; the tuner sweep validates "no
    /// regression from the configured depth" on a synthetic single-
    /// conversation pattern.
    #[serde(default = "default_true_serde")]
    pub auto_apply_prefix_cache_max_snapshots: bool,
    /// When `true` (default), the engine overrides
    /// `[inference].kv_page_size` with the cached winner. Only
    /// applies to `kv_cache_layout = "paged"`.
    #[serde(default = "default_true_serde")]
    pub auto_apply_kv_page_size: bool,
    /// When `true` (default), the engine exports the cached
    /// `flash_v3_kv_tile` (16/32/64) to `RUSTLLAMA_FLASH_V3_KV_TILE`
    /// at load time so the SYCL flash-attn-v3 decode kernel picks
    /// up the autotuned tile size on its next dispatch.
    #[serde(default = "default_true_serde")]
    pub auto_apply_flash_v3_kv_tile: bool,
    /// When `true` (default), routed-expert (MoE) placement resolves
    /// automatically from the tuner cache's `moe_placement` +
    /// `moe_gpu_split_permille` winners. Run `rustllama tune
    /// --moe-placement` to populate. With no cache entry (or this flag
    /// off), MoE tensors follow their layer's device (uniform). MoE
    /// placement is always auto — there is no user-facing config key;
    /// the heat planner / tuner owns the routed-expert tier decision.
    #[serde(default = "default_true_serde")]
    pub auto_apply_moe_placement: bool,
    /// Apply the per-model decision-probability calibration
    /// (fitted by the tune sweep's decision-calibration stage) to the
    /// `/v1/decide/*` endpoints.
    /// When off (or no cache entry), decisions return the model's raw
    /// (un-calibrated) softmax confidence.
    #[serde(default = "default_true_serde")]
    pub auto_apply_decision_calibration: bool,
}

fn default_true_serde() -> bool {
    true
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self {
            opportunistic_refine: false,
            auto_apply_placement: true,
            auto_apply_batch_size: true,
            auto_apply_kv_dtype: true,
            auto_apply_flash_attention: true,
            auto_apply_speculative_mtp: true,
            auto_apply_ssm_prefill_chunked: true,
            auto_apply_kv_cache_layout: true,
            auto_apply_flash_kv_min: true,
            auto_apply_prefix_cache_max_snapshots: true,
            auto_apply_kv_page_size: true,
            auto_apply_flash_v3_kv_tile: true,
            auto_apply_moe_placement: true,
            auto_apply_decision_calibration: true,
        }
    }
}

#[allow(clippy::derivable_impls)]
impl Default for Config {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            inference: InferenceConfig::default(),
            server: ServerConfig::default(),
            ui: UiConfig::default(),
            tuning: TuningConfig::default(),
            profiles: Vec::new(),
            system_prompts: Vec::new(),
            embeddings: EmbeddingsConfig::default(),
            reranker: RerankerConfig::default(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("toml decode: {0}")]
    TomlDe(#[from] toml::de::Error),
    #[error("toml encode: {0}")]
    TomlSer(#[from] toml::ser::Error),
    #[error("watcher: {0}")]
    Watcher(String),
    #[error("invalid config: {0}")]
    Validation(String),
}

pub type Result<T> = std::result::Result<T, ConfigError>;

pub fn default_config_path() -> Option<PathBuf> {
    Some(rustllama_runtime::paths().config_dir.join("config.toml"))
}

pub fn load(path: &Path) -> Result<Config> {
    if !path.exists() {
        return Ok(Config::default());
    }
    let s = std::fs::read_to_string(path)?;
    let cfg: Config = toml::from_str(&s)?;
    validate(&cfg)?;
    Ok(cfg)
}

/// Load the config and apply the named profile in one step. If
/// `profile` is `None` or empty, this is identical to [`load`].
/// Unknown profile names are warned-and-ignored rather than failing
/// — a stale CLI flag shouldn't crash startup.
pub fn load_with_profile(path: &Path, profile: Option<&str>) -> Result<Config> {
    let mut cfg = load(path)?;
    if let Some(name) = profile {
        if !name.is_empty() {
            if !cfg.apply_profile(name) {
                tracing::warn!(
                    profile = name,
                    available = ?cfg.profile_names(),
                    "unknown profile name — leaving base config as-is"
                );
            }
        }
    }
    Ok(cfg)
}

pub fn save(path: &Path, cfg: &Config) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let s = toml::to_string_pretty(cfg)?;
    std::fs::write(path, s)?;
    Ok(())
}

pub fn validate(cfg: &Config) -> Result<()> {
    match (cfg.model.path.as_ref(), cfg.model.hub.as_ref()) {
        (Some(_), Some(_)) => {
            return Err(ConfigError::Validation(
                "[model] specify exactly one of `path` or `hub`, not both".into(),
            ))
        }
        _ => {}
    }
    // `[embeddings]` and `[reranker]` mirror `[model]`'s mutually-exclusive
    // `path` / `hub` selectors — enforce the same rule so a config that sets
    // both fails loudly at load instead of silently preferring one.
    if cfg.embeddings.path.is_some() && cfg.embeddings.hub.is_some() {
        return Err(ConfigError::Validation(
            "[embeddings] specify exactly one of `path` or `hub`, not both".into(),
        ));
    }
    if cfg.reranker.path.is_some() && cfg.reranker.hub.is_some() {
        return Err(ConfigError::Validation(
            "[reranker] specify exactly one of `path` or `hub`, not both".into(),
        ));
    }
    let mb = cfg.inference.memory_budget.trim();
    if !(mb.eq_ignore_ascii_case("manual") || mb.eq_ignore_ascii_case("auto")) {
        return Err(ConfigError::Validation(format!(
            "[inference].memory_budget must be \"manual\" or \"auto\" (got \"{mb}\")"
        )));
    }
    // At least one compute tier must remain. Both tiers off leaves the
    // placement planner nothing to target.
    if !cfg.inference.cpu_enabled && !cfg.inference.gpu_enabled {
        return Err(ConfigError::Validation(
            "[inference].cpu_enabled and [inference].gpu_enabled are both false — no \
             compute tier remains; enable at least one"
                .to_string(),
        ));
    }
    Ok(())
}

/// Per-section changed flags. Subscribers inspect this to decide what to
/// hot-apply vs. require a model reload for.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ChangeSet {
    pub model: bool,
    pub inference: bool,
    pub server: bool,
    pub ui: bool,
    pub tuning: bool,
    /// `[embeddings]` changed.
    pub embeddings: bool,
    /// `[reranker]` changed.
    pub reranker: bool,
    /// The `[[system_prompts]]` library changed.
    pub system_prompts: bool,
    /// The `[[profiles]]` list changed.
    pub profiles: bool,
    /// The `[inference]` diff touches ONLY `moe_expert_cache_mb` —
    /// the one inference knob the server can hot-apply to a loaded
    /// engine at a request-safe point (elastic expert-cache budget,
    /// roadmap Phase 3). `false` whenever `inference` is `false` or
    /// any other inference field also changed.
    pub inference_budget_only: bool,
}

impl ChangeSet {
    pub fn any(&self) -> bool {
        self.model
            || self.inference
            || self.server
            || self.ui
            || self.tuning
            || self.embeddings
            || self.reranker
            || self.system_prompts
            || self.profiles
    }
    pub fn diff(old: &Config, new: &Config) -> Self {
        let inference = old.inference != new.inference;
        let inference_budget_only = inference && {
            let mut probe = old.inference.clone();
            probe.moe_expert_cache_mb = new.inference.moe_expert_cache_mb;
            probe == new.inference
        };
        Self {
            model: old.model != new.model,
            inference,
            server: old.server != new.server,
            ui: old.ui != new.ui,
            tuning: old.tuning != new.tuning,
            embeddings: old.embeddings != new.embeddings,
            reranker: old.reranker != new.reranker,
            system_prompts: old.system_prompts != new.system_prompts,
            profiles: old.profiles != new.profiles,
            inference_budget_only,
        }
    }
    /// Whether the changes can be applied without reloading the model.
    /// A budget-only inference change hot-applies (the server rebuilds
    /// the expert cache at a safe point), so it does not force a reload.
    pub fn requires_model_reload(&self) -> bool {
        self.model || (self.inference && !self.inference_budget_only)
    }
    /// Whether the changes require restarting the HTTP server to take effect.
    pub fn requires_server_restart(&self) -> bool {
        self.server
    }
}

#[derive(Debug, Clone)]
pub enum ConfigDelta {
    Reloaded { changes: ChangeSet },
    ParseError(String),
}

/// Recommended debounce interval for the file watcher.
pub const WATCH_DEBOUNCE: Duration = Duration::from_millis(200);

/// Live-reload handle. Wraps the current config behind a `RwLock` and
/// broadcasts a [`ConfigDelta`] on every detected change. Call
/// [`ConfigHandle::watch`] to enable filesystem watching; without that, the
/// handle is a pure in-memory store updated via [`ConfigHandle::reload`].
#[derive(Clone)]
pub struct ConfigHandle {
    current: Arc<RwLock<Config>>,
    tx: broadcast::Sender<ConfigDelta>,
    path: PathBuf,
}

impl ConfigHandle {
    pub fn new(path: PathBuf, cfg: Config) -> Self {
        let (tx, _) = broadcast::channel(16);
        Self {
            current: Arc::new(RwLock::new(cfg)),
            tx,
            path,
        }
    }

    pub fn from_path(path: PathBuf) -> Result<Self> {
        let cfg = load(&path)?;
        Ok(Self::new(path, cfg))
    }

    pub async fn snapshot(&self) -> Config {
        self.current.read().await.clone()
    }

    pub async fn reload(&self) -> Result<ChangeSet> {
        let new = load(&self.path)?;
        let mut current = self.current.write().await;
        let changes = ChangeSet::diff(&current, &new);
        *current = new;
        let _ = self.tx.send(ConfigDelta::Reloaded { changes });
        Ok(changes)
    }

    pub fn subscribe(&self) -> broadcast::Receiver<ConfigDelta> {
        self.tx.subscribe()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Start a debounced filesystem watcher on the config file. The returned
    /// [`WatchGuard`] owns the watcher thread — drop it to stop watching.
    ///
    /// On file change, this:
    ///   1. Parses the new file.
    ///   2. Diffs against the in-memory config.
    ///   3. Atomically swaps the in-memory copy.
    ///   4. Broadcasts [`ConfigDelta::Reloaded { changes }`].
    ///
    /// On parse error, the in-memory copy is *not* replaced and a
    /// [`ConfigDelta::ParseError`] is broadcast instead, so subscribers can
    /// surface a toast / warning without the engine flipping into an
    /// invalid state.
    pub fn watch(&self) -> Result<WatchGuard> {
        use notify::Watcher;
        use notify_debouncer_full::new_debouncer;

        let (tx, rx) = std::sync::mpsc::channel();
        let mut debouncer = new_debouncer(WATCH_DEBOUNCE, None, tx)
            .map_err(|e| ConfigError::Watcher(e.to_string()))?;

        // Watch the parent directory so atomic-rename saves (which the OS
        // delivers as Remove+Create) are caught reliably.
        let watch_target = self
            .path
            .parent()
            .filter(|p| p.exists())
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| self.path.clone());
        debouncer
            .watcher()
            .watch(&watch_target, notify::RecursiveMode::NonRecursive)
            .map_err(|e| ConfigError::Watcher(e.to_string()))?;

        let handle = self.clone();
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop_clone = stop.clone();
        let watched_path = self.path.clone();
        let join = std::thread::Builder::new()
            .name("rustllama-config-watcher".into())
            .spawn(move || {
                while !stop_clone.load(std::sync::atomic::Ordering::Acquire) {
                    let Ok(batch) = rx.recv() else { break };
                    // notify-debouncer-full delivers Result<Vec<DebouncedEvent>, Vec<Error>>.
                    let events = match batch {
                        Ok(evs) => evs,
                        Err(errs) => {
                            let msg = errs
                                .iter()
                                .map(|e| e.to_string())
                                .collect::<Vec<_>>()
                                .join("; ");
                            let _ = handle.tx.send(ConfigDelta::ParseError(format!(
                                "watcher: {msg}"
                            )));
                            continue;
                        }
                    };
                    if !events.iter().any(|e| {
                        e.event.paths.iter().any(|p| {
                            p == &watched_path
                                || p.file_name() == watched_path.file_name()
                        })
                    }) {
                        continue;
                    }
                    // Re-parse and broadcast.
                    let parsed = match load(&handle.path) {
                        Ok(cfg) => cfg,
                        Err(e) => {
                            let _ = handle.tx.send(ConfigDelta::ParseError(e.to_string()));
                            continue;
                        }
                    };
                    // Block in this thread on the async lock — we hold an
                    // owned Arc to current so we can use blocking_write.
                    let mut current = handle.current.blocking_write();
                    let changes = ChangeSet::diff(&current, &parsed);
                    *current = parsed;
                    drop(current);
                    if changes.any() {
                        let _ = handle.tx.send(ConfigDelta::Reloaded { changes });
                    }
                }
            })
            .map_err(|e| ConfigError::Watcher(format!("spawn watcher thread: {e}")))?;

        Ok(WatchGuard {
            _debouncer: debouncer,
            stop,
            join: Some(join),
        })
    }
}

/// Owns the filesystem watcher; dropping it stops the watch thread.
pub struct WatchGuard {
    _debouncer: notify_debouncer_full::Debouncer<
        notify::RecommendedWatcher,
        notify_debouncer_full::FileIdMap,
    >,
    stop: Arc<std::sync::atomic::AtomicBool>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl Drop for WatchGuard {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Release);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_roundtrip() {
        let c = Config::default();
        let s = toml::to_string(&c).unwrap();
        let back: Config = toml::from_str(&s).unwrap();
        assert_eq!(c.server.port, back.server.port);
    }

    #[test]
    fn validate_rejects_both_sources() {
        let mut c = Config::default();
        c.model.path = Some("a.gguf".into());
        c.model.hub = Some("a/b:c".into());
        assert!(validate(&c).is_err());
    }

    #[test]
    fn validate_rejects_both_embeddings_sources() {
        let mut c = Config::default();
        c.embeddings.path = Some("e.gguf".into());
        c.embeddings.hub = Some("a/b:e.gguf".into());
        assert!(validate(&c).is_err());
        // Either alone is fine.
        c.embeddings.hub = None;
        assert!(validate(&c).is_ok());
    }

    #[test]
    fn validate_rejects_both_reranker_sources() {
        let mut c = Config::default();
        c.reranker.path = Some("r.gguf".into());
        c.reranker.hub = Some("a/b:r.gguf".into());
        assert!(validate(&c).is_err());
        c.reranker.path = None;
        assert!(validate(&c).is_ok());
    }

    #[test]
    fn changeset_detects_embeddings_reranker_prompts_profiles() {
        // Edits touching only these sections must still be detected + notified
        // (previously diff()/any() ignored them, so the snapshot updated
        // silently and no ConfigDelta was broadcast).
        let base = Config::default();

        let mut b = base.clone();
        b.embeddings.hub = Some("a/b:e.gguf".into());
        let cs = ChangeSet::diff(&base, &b);
        assert!(cs.embeddings && cs.any());

        let mut b = base.clone();
        b.reranker.hub = Some("a/b:r.gguf".into());
        let cs = ChangeSet::diff(&base, &b);
        assert!(cs.reranker && cs.any());

        let mut b = base.clone();
        b.system_prompts.push(SystemPrompt {
            name: "x".into(),
            body: "y".into(),
            default_for_model: String::new(),
        });
        let cs = ChangeSet::diff(&base, &b);
        assert!(cs.system_prompts && cs.any());

        let mut b = base.clone();
        b.profiles.push(ProfileOverride {
            name: "p".into(),
            ..Default::default()
        });
        let cs = ChangeSet::diff(&base, &b);
        assert!(cs.profiles && cs.any());
    }

    #[test]
    fn changeset_detects_inference_change() {
        let a = Config::default();
        let mut b = Config::default();
        b.inference.n_gpu_layers = 99;
        let cs = ChangeSet::diff(&a, &b);
        assert!(cs.inference);
        assert!(!cs.server);
        assert!(cs.requires_model_reload());
        assert!(!cs.inference_budget_only);
    }

    #[test]
    fn changeset_budget_only_change_hot_applies() {
        // Touching ONLY moe_expert_cache_mb is the elastic-budget
        // class: flagged, and exempt from the model-reload demand.
        let a = Config::default();
        let mut b = Config::default();
        b.inference.moe_expert_cache_mb = 4096;
        let cs = ChangeSet::diff(&a, &b);
        assert!(cs.inference);
        assert!(cs.inference_budget_only);
        assert!(!cs.requires_model_reload());
    }

    #[test]
    fn changeset_budget_plus_other_field_still_reloads() {
        let a = Config::default();
        let mut b = Config::default();
        b.inference.moe_expert_cache_mb = 4096;
        b.inference.ctx_size = 4096;
        let cs = ChangeSet::diff(&a, &b);
        assert!(cs.inference);
        assert!(!cs.inference_budget_only);
        assert!(cs.requires_model_reload());
    }

    #[test]
    fn validate_rejects_both_tiers_disabled() {
        let mut c = Config::default();
        // Either tier alone is fine.
        c.inference.cpu_enabled = false;
        c.inference.gpu_enabled = true;
        assert!(validate(&c).is_ok());
        c.inference.cpu_enabled = true;
        c.inference.gpu_enabled = false;
        assert!(validate(&c).is_ok());
        // Both off leaves no compute tier — rejected.
        c.inference.cpu_enabled = false;
        c.inference.gpu_enabled = false;
        assert!(validate(&c).is_err());
    }

    #[test]
    fn n_gpu_layers_override_sentinel() {
        let mut c = Config::default();
        // Default is the AUTO sentinel ⇒ no override.
        assert_eq!(c.inference.n_gpu_layers_override(), None);
        c.inference.n_gpu_layers = 16;
        assert_eq!(c.inference.n_gpu_layers_override(), Some(16));
        c.inference.n_gpu_layers = 0;
        assert_eq!(c.inference.n_gpu_layers_override(), Some(0));
    }

    #[test]
    fn validate_rejects_bad_memory_budget_mode() {
        let mut c = Config::default();
        c.inference.memory_budget = "yes-please".into();
        assert!(validate(&c).is_err());
        c.inference.memory_budget = "AUTO".into();
        assert!(validate(&c).is_ok());
        c.inference.memory_budget = "manual".into();
        assert!(validate(&c).is_ok());
    }

    #[test]
    fn apply_profile_overrides_only_specified_keys() {
        // A profile that only touches `[server].port` and
        // `[inference].ctx_size` must leave every other base field
        // unchanged. Pins the partial-override semantics — full
        // replacement would be surprising and would force users to
        // re-specify every default when they want to bump one knob.
        let mut cfg = Config::default();
        cfg.profiles.push(ProfileOverride {
            name: "lan".to_string(),
            server: Some(ProfileServer {
                port: Some(11500),
                bind_addr: Some("0.0.0.0".to_string()),
                ..Default::default()
            }),
            inference: Some(ProfileInference {
                ctx_size: Some(16_384),
                ..Default::default()
            }),
            model: None,
        });

        let base_max_pending = cfg.server.max_pending_per_model;
        let base_threads = cfg.inference.threads;

        let applied = cfg.apply_profile("lan");
        assert!(applied, "profile must apply by name match");
        assert_eq!(cfg.server.port, 11500);
        assert_eq!(cfg.server.bind_addr, "0.0.0.0");
        assert_eq!(cfg.inference.ctx_size, 16_384);
        // Untouched fields preserved.
        assert_eq!(cfg.server.max_pending_per_model, base_max_pending);
        assert_eq!(cfg.inference.threads, base_threads);
    }

    #[test]
    fn apply_profile_unknown_name_is_noop_returns_false() {
        // Stale `--profile` flag in scripts should NOT crash startup.
        // The CLI logs a warning and falls back to the base config;
        // `apply_profile` returns false to give the caller the signal.
        let mut cfg = Config::default();
        let port_before = cfg.server.port;
        let applied = cfg.apply_profile("nonexistent");
        assert!(!applied);
        assert_eq!(cfg.server.port, port_before);
    }

    #[test]
    fn profile_round_trips_through_toml() {
        // Profiles must serialize / deserialize cleanly so `config
        // edit` doesn't lose the section. Build a config with one
        // profile, write it to toml, read it back, assert equality.
        let mut cfg = Config::default();
        cfg.profiles.push(ProfileOverride {
            name: "work".to_string(),
            server: Some(ProfileServer {
                concurrency: Some(4),
                ..Default::default()
            }),
            ..Default::default()
        });
        let s = toml::to_string(&cfg).expect("serialize");
        let reloaded: Config = toml::from_str(&s).expect("deserialize");
        assert_eq!(reloaded.profiles.len(), 1);
        assert_eq!(reloaded.profiles[0].name, "work");
        assert_eq!(
            reloaded.profiles[0].server.as_ref().unwrap().concurrency,
            Some(4),
        );
    }
}
