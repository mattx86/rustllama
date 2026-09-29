// Tiny HTTP client for the local rustllama server. The Tauri build
// embeds the server via the `serve` subcommand on the same host
// (default port 11434); standalone web builds talk to whatever
// origin the page was loaded from.
//
// All calls are plain fetch — no SDK dependency. Streaming uses the
// browser's ReadableStream so we can flush content_block deltas as
// they arrive.

const DEFAULT_BASE = "http://127.0.0.1:11434";

function baseUrl(): string {
  // Allow override via window.RUSTLLAMA_BASE (set via Tauri config or
  // dev-mode env injection). Falls back to localhost:11434.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const w = window as any;
  return (w.RUSTLLAMA_BASE as string | undefined) ?? DEFAULT_BASE;
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
  /** Attached images as data: URIs (png/jpeg). When present, the wire
   * layer sends OpenAI multimodal content blocks; the server decodes
   * them and the vision-enabled engine splices projected features. */
  images?: string[];
}

export interface ModelInfo {
  id: string;
  object?: string;
  created?: number;
  /** Present only when the entry is the loaded mixture-of-experts
   * chat model. Lets the Models page / picker render an
   * "N routed, top-K" badge without a separate /v1/capabilities call. */
  moe?: {
    n_experts: number;
    n_experts_used: number;
    n_experts_shared: number;
  };
}

export interface HealthInfo {
  status: string;
  model_id?: string;
  draining?: boolean;
  version?: string;
}

export interface UsageInfo {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  prefill_ms?: number;
  decode_ms?: number;
  cache_hit_tokens?: number;
  tokens_prefilled?: number;
}

export interface TagsModel {
  name: string;
  size: number;
  modified_at: string;
  details?: { family?: string; parameter_size?: string; quantization_level?: string };
}

export async function getHealth(): Promise<HealthInfo> {
  const r = await fetch(`${baseUrl()}/healthz`);
  if (!r.ok) throw new Error(`healthz ${r.status}`);
  return r.json();
}

export interface MetricsSnapshot {
  model_id: string;
  ctx_size: number;
  ctx_used: number;
  pending: number;
  max_pending: number;
  last_tok_s: number | null;
  /** Exponential moving average (alpha = 0.3) of recent decode tok/s.
   * Prefer this over `last_tok_s` for displayed throughput — the
   * per-request number swings 5× between a cold first probe and a
   * warm second. `null` until the first non-trivial generation. */
  ema_tok_s: number | null;
  last_prefill_ms: number | null;
  last_decode_ms: number | null;
  last_tokens_prefilled: number | null;
  last_tokens_generated: number | null;
  last_cache_hit_tokens: number | null;
  uptime_s: number;
  concurrency: number;
  /// One of "f32" / "q8_0" / "q4_0" / "tq1" / "tq2" / "tq4" / "tq8" / "nvfp4".
  /// `null` for mock-engine deployments that have no KV cache dtype.
  kv_dtype: string | null;
  /// Host total physical RAM in bytes.
  ram_total_bytes: number;
  /// Currently-available (free) RAM in bytes.
  ram_available_bytes: number;
  /// Commit limit in bytes (Windows: physical RAM + pagefile; Linux:
  /// CommitLimit). With `ram_*` this derives pagefile/swap usage.
  commit_total_bytes: number;
  /// Available commit (commit limit − commit charge) in bytes.
  commit_available_bytes: number;
  /// Number of SYCL devices the server can see (0 when no Intel GPU /
  /// oneAPI runtime is present; >0 when the oneAPI runtime detects an
  /// Intel GPU).
  sycl_device_count: number;
  /// Paged-KV pool: total pages allocated to the engine at load.
  /// `0` when the engine uses the contiguous KV layout (the
  /// default) — the Status page hides its "Paged KV pool" panel
  /// in that case.
  paged_total_pages: number;
  /// Paged-KV pool: pages currently on the free list.
  /// `paged_total_pages - paged_free_pages` is the live in-use
  /// count.
  paged_free_pages: number;
  /// Active slots in a fused-decode engine. Always `0` on
  /// `CpuEngine` paged backend (one slot per fork); nonzero only
  /// on `PagedBatchEngine` (the `[server].fused_decode = true`
  /// path). V1 placeholder always reports `0` until the
  /// driver-thread shared-counter wire-up lands.
  paged_active_slots: number;
  /// Cumulative-since-startup counters surfaced by `CpuEngine`.
  /// `null` for mock-engine deployments (no real generation has
  /// run). Operators watch `prefix_cache_hit_rate` to see whether
  /// the prefix cache is paying off; values close to `1.0` mean
  /// the cache is doing most of the prefill work.
  cumulative?: CumulativeStatsSnapshot | null;
  /// Resident set size of the rustllama process in bytes. Useful for
  /// pairing with `ram_available_bytes` to chart RAM pressure during
  /// long generations. `null` on platforms where the runtime can't
  /// read its own RSS.
  process_rss_bytes?: number | null;
  /// System-wide CPU utilization % (0..=100), for the CPU-over-RAM
  /// paired status-bar row. Delta since the previous poll.
  cpu_utilization_pct?: number;
  /// CPU brand string (e.g. "Intel Core i7-…") for the CPU-row tooltip.
  cpu_brand?: string;
  /// Page/Swap (system paging) disk I/O, bytes/sec, split read/write.
  page_io_read_bytes_per_sec?: number;
  page_io_write_bytes_per_sec?: number;
  /// Model I/O — the server process's disk I/O, bytes/sec, read/write.
  model_io_read_bytes_per_sec?: number;
  model_io_write_bytes_per_sec?: number;
  /// Intel GPU sensors via Level Zero Sysman. Each field is
  /// independently optional — Iris Xe integrated graphics typically
  /// exposes VRAM + freq but not temp/power; discrete Arc exposes all.
  /// `null` outer field on non-Intel hosts or when the driver doesn't
  /// expose Sysman.
  gpu_sysman?: GpuSysmanSnapshot | null;
  /// Per-physical-GPU VRAM (one entry per GPU detected). The status bar
  /// renders a VRAM usage bar per entry. Absent/empty when no GPU is
  /// visible; deduped so one physical GPU appears once.
  gpus?: GpuMetric[] | null;
}

/// Per-physical-GPU VRAM for the status bar. `vram_free_bytes` is populated
/// for both Intel (L0 Sysman) and NVIDIA (`cuMemGetInfo` via the dlopen'd
/// CUDA driver); `null` only when neither probe could report it.
export interface GpuMetric {
  /// Stable RustLlama enumeration index — fixed regardless of which GPUs
  /// are disabled, so the label stays consistent (ignoring GPU 0 leaves
  /// GPU 1 labeled "GPU 1"). This is the index the disable-list targets.
  index: number;
  /// "intel" | "nvidia" | "amd" | "gpu".
  vendor: string;
  name: string;
  vram_total_bytes?: number | null;
  vram_free_bytes?: number | null;
  /// GPU engine (compute/render) utilization %, 0..=100. `null`/absent
  /// when the backend doesn't expose it yet (the status bar shows the
  /// util bar as "n/a"). Intel engine-activity + NVIDIA NVML land in A3.
  utilization_pct?: number | null;
}

export interface GpuSysmanSnapshot {
  /// Total VRAM (or shared-memory equivalent on Iris Xe) in bytes.
  vram_total_bytes?: number | null;
  /// Currently-free VRAM in bytes.
  vram_free_bytes?: number | null;
  /// Max temperature across all sensors (°C). Often `null` on
  /// integrated parts.
  max_temp_c?: number | null;
  /// Cumulative energy counter (µJ). Pair with `energy_timestamp_us`
  /// from two samples to derive instantaneous power.
  energy_uj?: number | null;
  /// Energy-counter sample timestamp (µs).
  energy_timestamp_us?: number | null;
  /// GPU clock on the first frequency domain (MHz).
  gpu_freq_mhz?: number | null;
}

export interface CumulativeStatsSnapshot {
  total_requests: number;
  total_tokens_prefilled: number;
  total_cache_hit_tokens: number;
  total_tokens_generated: number;
  total_prefill_ms: number;
  total_decode_ms: number;
  /** Ratio of prompt tokens served from the prefix cache.
   * Range `[0.0, 1.0]`; `0.0` when no requests have completed. */
  prefix_cache_hit_rate: number;
  /** Speculation verify rounds that proposed ≥1 draft token.
   * Optional: absent on servers older than the counters. */
  total_spec_rounds?: number;
  /** Total draft tokens proposed (n-gram + draft-model). */
  total_spec_drafted?: number;
  /** Total draft tokens accepted by the target. */
  total_spec_accepted?: number;
  /** `total_spec_accepted / total_spec_drafted`; 0.0 before any
   * speculation has run. */
  spec_acceptance_rate?: number;
}

export async function getMetrics(): Promise<MetricsSnapshot> {
  const r = await fetch(`${baseUrl()}/v1/metrics`);
  if (!r.ok) throw new Error(`metrics ${r.status}`);
  return r.json();
}

export interface TokenizeResult {
  count: number;
  tokens: number[];
  model_id: string;
}

/// Server-side token count for the given text. Pass `addBos: false`
/// to match what the Chat page's send path uses (the chat template
/// prepends its own BOS as needed, so the meter shouldn't double-count).
export async function tokenizeCount(
  content: string,
  opts?: { model?: string; addBos?: boolean; signal?: AbortSignal },
): Promise<TokenizeResult> {
  const r = await fetch(`${baseUrl()}/v1/tokenize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      content,
      model: opts?.model,
      add_bos: opts?.addBos ?? false,
    }),
    signal: opts?.signal,
  });
  if (!r.ok) throw new Error(`tokenize ${r.status}`);
  return r.json();
}

export async function listModels(): Promise<ModelInfo[]> {
  const r = await fetch(`${baseUrl()}/v1/models`);
  if (!r.ok) throw new Error(`/v1/models ${r.status}`);
  const body = await r.json();
  return body.data ?? [];
}

export async function listOllamaTags(): Promise<TagsModel[]> {
  const r = await fetch(`${baseUrl()}/api/tags`);
  if (!r.ok) throw new Error(`/api/tags ${r.status}`);
  const body = await r.json();
  return body.models ?? [];
}

/// Load a model into the warm registry. Accepts any of:
///   - `path`: absolute `.gguf` path on disk.
///   - `hub`: HuggingFace ref `owner/repo:filename` (must already be
///     in the local cache; use `pullModel` first to download).
///   - `name`: short file-stem as surfaced by `/api/tags`. The server
///     walks the cache and matches `path.file_stem()`. This is the
///     shape the GUI Models page uses for cached rows.
export async function loadModel(opts: {
  path?: string;
  hub?: string;
  name?: string;
  ctxSize?: number;
  kvDtype?: string;
}): Promise<{ model_id: string; previous_model_id: string | null; is_default: boolean }> {
  const r = await fetch(`${baseUrl()}/v1/models/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      path: opts.path,
      hub: opts.hub,
      name: opts.name,
      ctx_size: opts.ctxSize,
      kv_dtype: opts.kvDtype,
    }),
  });
  if (!r.ok) throw new Error(`/v1/models/load ${r.status}: ${await r.text()}`);
  return r.json();
}

export interface GgufInspectDtypeBucket {
  dtype: string;
  tensor_count: number;
  bytes: number;
}

export interface GgufInspectTensorEntry {
  name: string;
  dtype: string;
  shape: number[];
  elements: number;
  bytes: number;
}

export interface GgufModelCard {
  license?: string;
  library_name?: string;
  tags: string[];
  base_model?: string;
  language: string[];
  model_creator?: string;
  quantized_by?: string;
  description?: string;
  source_url?: string;
}

export interface GgufInspectResult {
  path: string;
  architecture: string;
  name: string;
  size_label?: string;
  file_type?: number;
  context_length: number | null;
  block_count: number | null;
  embedding_length: number | null;
  head_count: number | null;
  head_count_kv: number | null;
  head_dim: number | null;
  vocab_size: number | null;
  // MoE fields are absent on dense GGUFs (server omits via
  // skip_serializing_if). When present, n_experts >= 2 and the
  // GUI Models page renders an "N experts, top-K routed" badge.
  n_experts?: number;
  n_experts_used?: number;
  n_experts_shared?: number;
  tensor_count: number;
  total_params: number;
  total_tensor_bytes: number;
  file_bytes: number;
  dtypes: GgufInspectDtypeBucket[];
  tensors?: GgufInspectTensorEntry[];
  model_card?: GgufModelCard;
}

/// Read-only metadata probe for a cached GGUF — does NOT load the
/// model. Pass exactly one of `path`, `hub`, or `name`. `name` (file
/// stem) is the shape `/api/tags` returns, so the Models page passes
/// it straight through. `includeTensors` toggles the per-tensor list
/// (~300 entries on a 7B model; off by default to keep payloads
/// small).
export async function inspectGguf(opts: {
  path?: string;
  hub?: string;
  name?: string;
  includeTensors?: boolean;
}): Promise<GgufInspectResult> {
  const r = await fetch(`${baseUrl()}/api/gguf/inspect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      path: opts.path,
      hub: opts.hub,
      name: opts.name,
      include_tensors: opts.includeTensors ?? false,
    }),
  });
  if (!r.ok) throw new Error(`/api/gguf/inspect ${r.status}: ${await r.text()}`);
  return r.json();
}

/// Drop a loaded model from the registry, freeing its RAM/VRAM.
/// "Eject" frees the last model too — the server tolerates an empty
/// registry and returns "model not loaded" until one is loaded again.
export async function unloadModel(modelId: string): Promise<void> {
  const r = await fetch(`${baseUrl()}/v1/models/unload`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: modelId }),
  });
  if (!r.ok) throw new Error(`/v1/models/unload ${r.status}: ${await r.text()}`);
}

/// Promote a loaded model to the default — subsequent requests
/// without an explicit `model` field route to this one.
export async function setDefaultModel(modelId: string): Promise<void> {
  const r = await fetch(`${baseUrl()}/v1/models/default`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: modelId }),
  });
  if (!r.ok) throw new Error(`/v1/models/default ${r.status}: ${await r.text()}`);
}

/// Delete a cached GGUF from disk. Ollama-shaped endpoint; takes the
/// `owner/repo:filename` style name surfaced by `/api/tags`.
export async function deleteModel(name: string): Promise<void> {
  const r = await fetch(`${baseUrl()}/api/delete`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!r.ok) throw new Error(`/api/delete ${r.status}: ${await r.text()}`);
}

// ----- /v1/config (full read/write of the on-disk TOML) ----

/// Hot-apply semantics for each config section. Surfaced from
/// `GET /v1/config` so the Settings page can label which fields
/// take effect immediately vs require a reload or restart.
export interface ConfigHotApply {
  model: "live" | "reload" | "restart";
  inference: "live" | "reload" | "restart";
  server: "live" | "reload" | "restart";
  ui: "live" | "reload" | "restart";
  tuning: "live" | "reload" | "restart";
}

/// Subset of the rust `Config` struct that the Settings page edits.
/// Keeps the wire shape loose (`unknown`) for sections we don't yet
/// expose in the form — those round-trip unchanged through `PUT`.
export interface AppConfig {
  model: {
    path?: string | null;
    hub?: string | null;
    chat_template: string;
  };
  inference: {
    n_gpu_layers: number;
    ctx_size: number;
    batch_size: number;
    threads: number;
    flash_attention: boolean;
    placement: { overrides: unknown[] };
    prefix_cache: boolean;
    prefix_cache_max_snapshots: number;
    kv_dtype: string;
    /// Optional per-channel K dtype override. `null` means use kv_dtype.
    /// Phase 3.5: when both k_dtype and v_dtype resolve to different
    /// values the engine rejects model load.
    k_dtype: string | null;
    /// Optional per-channel V dtype override. `null` means use kv_dtype.
    v_dtype: string | null;
    max_tool_iterations: number;
    keep_quant_raw: boolean;
    /// K-cache mean-centering bias sidecar path (fork kv_bar GGUF).
    /// null = auto-discover `<model stem>.kvbias.gguf` beside the model.
    kv_bias_path: string | null;
    /// N-gram (prompt-lookup) speculative decoding — no second model.
    speculative_ngram: boolean;
    /// Draft-model GGUF for two-model speculative decoding. Must share
    /// the target tokenizer exactly (verified at load). Takes
    /// precedence over the n-gram drafter.
    speculative_draft_path: string | null;
    /// Candidates per speculative round for the draft-model path.
    speculative_draft_k: number;
  };
  server: {
    bind_addr: string;
    port: number;
    api_key: string;
    cors_origins: string[];
    max_pending_per_model: number;
    max_loaded_models: number;
    concurrency: number;
    /// Per-request audit log. When true, the server appends a JSONL
    /// line per request to `audit_log_path`. Off by default — opt
    /// in for LAN-exposed deployments where you want to know who
    /// hit what. The log redacts api_key query params and never
    /// records the Authorization header value or request bodies.
    audit_log?: boolean;
    /// Destination for the audit log. Empty = derive a default
    /// under the user-data dir (`<crash_log_dir>/audit.log.jsonl`).
    audit_log_path?: string;
  };
  ui: {
    theme: string;
    font_size: number;
    code_theme: string;
  };
  tuning: Record<string, unknown>;
  profiles: ConfigProfile[];
  system_prompts: SystemPrompt[];
}

/// One entry of `[[system_prompts]]` in `config.toml`. The Chat
/// page renders these in a dropdown; selecting one prepends `{role:
/// "system", content: body}` to every send. `default_for_model`
/// auto-selects the prompt when the named model is loaded.
export interface SystemPrompt {
  name: string;
  body: string;
  default_for_model: string;
}

/// One entry of `[[profiles]]` in `config.toml`. Mirrors the Rust
/// `ProfileOverride` struct: every sparse section is optional, so a
/// profile that only retargets `[server].port` doesn't have to
/// declare the full `[model]` block.
export interface ConfigProfile {
  name: string;
  model?: {
    path?: string | null;
    hub?: string | null;
    chat_template?: string;
  } | null;
  inference?: {
    n_gpu_layers?: number;
    ctx_size?: number;
    batch_size?: number;
    threads?: number;
    kv_dtype?: string;
  } | null;
  server?: {
    bind_addr?: string;
    port?: number;
    max_pending_per_model?: number;
    max_loaded_models?: number;
    concurrency?: number;
  } | null;
}

export interface ConfigEnvelope {
  config: AppConfig;
  config_path: string | null;
  hot_apply: ConfigHotApply;
}

export async function getConfig(): Promise<ConfigEnvelope> {
  const r = await fetch(`${baseUrl()}/v1/config`);
  if (!r.ok) throw new Error(`/v1/config ${r.status}: ${await r.text()}`);
  return r.json();
}

export interface ConfigPutResult {
  changes: {
    model: boolean;
    inference: boolean;
    server: boolean;
    ui: boolean;
    tuning: boolean;
  };
  requires_model_reload: boolean;
  requires_server_restart: boolean;
}

export async function putConfig(cfg: AppConfig): Promise<ConfigPutResult> {
  const r = await fetch(`${baseUrl()}/v1/config`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cfg),
  });
  if (!r.ok) throw new Error(`PUT /v1/config ${r.status}: ${await r.text()}`);
  return r.json();
}

export interface CrashLogEntry {
  name: string;
  path: string;
  size_bytes: number;
  /// Unix epoch seconds when the crash happened. Parsed from the
  /// filename so it doesn't drift if the file is moved on disk.
  epoch_secs: number;
}

export interface CrashLogsList {
  dir: string;
  entries: CrashLogEntry[];
}

/// List crash logs the runtime panic hook has written, newest first.
/// Empty array (not an error) when no panic has ever occurred and
/// the crash dir doesn't exist yet.
export async function listCrashLogs(): Promise<CrashLogsList> {
  const r = await fetch(`${baseUrl()}/v1/crash_logs`);
  if (!r.ok) throw new Error(`/v1/crash_logs ${r.status}: ${await r.text()}`);
  return r.json();
}

/// Read a single crash log's body as plain text. `name` must come
/// from a prior `listCrashLogs()` call — server validates the shape
/// and rejects anything that doesn't match `crash-<digits>-<digits>.log`.
export async function getCrashLog(name: string): Promise<string> {
  const r = await fetch(`${baseUrl()}/v1/crash_logs/${encodeURIComponent(name)}`);
  if (!r.ok) throw new Error(`/v1/crash_logs/${name} ${r.status}: ${await r.text()}`);
  return r.text();
}

/// Remove a crash log from disk.
export async function deleteCrashLog(name: string): Promise<void> {
  const r = await fetch(`${baseUrl()}/v1/crash_logs/${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  if (!r.ok) throw new Error(`/v1/crash_logs/${name} ${r.status}: ${await r.text()}`);
}

export interface TuningDeviceFp {
  pci_id: number;
  name: string;
  driver_ver: string;
  vram_mb: number;
  slug: string;
}

export interface TuningPlacementEntry {
  model_key: string;
  n_gpu_layers: number;
}

export interface TuningKernelEntry {
  /// Cache HashMap key — typically `<kernel_name>:<shape_bucket>`.
  key: string;
  /// Canonical kernel name (e.g. `q4k_packed_usm`).
  kernel: string;
  /// Tuner-specific JSON payload. For packed matvec kernels this
  /// is `{"lws": <number>}`.
  params: unknown;
}

export interface TuningSummary {
  device: TuningDeviceFp | null;
  cache_path: string | null;
  cache_present: boolean;
  last_tuned: string | null;
  kernel_entry_count: number;
  kernels: TuningKernelEntry[];
  placement: TuningPlacementEntry[];
  batch_size: number | null;
  /// Per-device kv_dtype winner from `rustllama tune --kv-dtype`.
  kv_dtype: string | null;
  /// Per-device flash_attention winner.
  flash_attention: boolean | null;
  /// Per-device kv_cache_layout winner.
  kv_cache_layout: string | null;
  auto_apply_placement: boolean;
  auto_apply_batch_size: boolean;
  auto_apply_kv_dtype: boolean;
  auto_apply_flash_attention: boolean;
  auto_apply_kv_cache_layout: boolean;
}

export interface EmbeddingsRequestOpts {
  model?: string;
  input: string | string[];
  encodingFormat?: "float" | "base64";
  dimensions?: number;
}

export interface EmbeddingItem {
  object: "embedding";
  index: number;
  embedding: number[];
}

export interface EmbeddingsResponse {
  object: "list";
  data: EmbeddingItem[];
  model: string;
  usage: { prompt_tokens: number; total_tokens: number };
}

/// OpenAI-shaped POST /v1/embeddings client.
///
/// **v1.1 foundation status**: returns `501 Not Implemented`
/// today regardless of input, with a structured body that
/// distinguishes "no embedding model configured" from "configured
/// but loader is pending." Client wiring (RAG flows, Open WebUI,
/// rerankers) can be tested against this 501 today — when the
/// BERT/E5-family loader lands, no client changes needed.
export async function fetchEmbeddings(
  opts: EmbeddingsRequestOpts,
): Promise<EmbeddingsResponse> {
  const r = await fetch(`${baseUrl()}/v1/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: opts.model,
      input: opts.input,
      encoding_format: opts.encodingFormat,
      dimensions: opts.dimensions,
    }),
  });
  if (!r.ok) throw new Error(`/v1/embeddings ${r.status}: ${await r.text()}`);
  return r.json();
}

export interface AuditTailEntry {
  ts_ms: number;
  method: string;
  path: string;
  query: string;
  status: number;
  latency_ms: number;
}

export interface AuditTailResponse {
  /// Resolved on-disk path, or null when the file doesn't exist
  /// yet (no audit_log run + no `[server].audit_log = true`).
  path: string | null;
  /// Total entries in the whole file (larger than `entries.length`
  /// when more than `n` exist).
  total_entries: number;
  /// Most-recent first. Each entry's shape matches what the audit
  /// middleware writes — additional fields would pass through
  /// untouched.
  entries: AuditTailEntry[];
}

/// Fetch the last N audit log entries. The server reads the
/// configured `audit_log_path` (or derived default), parses JSONL,
/// returns the tail. Works whether or not `[server].audit_log` is
/// currently writing — the endpoint is read-only.
export async function getAuditLogTail(n?: number): Promise<AuditTailResponse> {
  const url = new URL(`${baseUrl()}/v1/audit_log/tail`);
  if (n !== undefined) url.searchParams.set("n", String(n));
  const r = await fetch(url.toString());
  if (!r.ok) throw new Error(`/v1/audit_log/tail ${r.status}: ${await r.text()}`);
  return r.json();
}

export interface TunePlacementCandidate {
  n_gpu_layers: number;
  warmup_ms: number;
  median_tps: number | null;
  max_tps: number | null;
  error: string | null;
}

export interface TunePlacementResponse {
  winner: number | null;
  winner_tps: number;
  load_ms: number;
  candidates: TunePlacementCandidate[];
  cache_path: string | null;
}

export interface TuneBatchSizeCandidate {
  batch_size: number;
  warmup_ms: number;
  median_tps: number | null;
  max_tps: number | null;
  error: string | null;
}

export interface TuneBatchSizeResponse {
  winner: number | null;
  winner_tps: number;
  load_ms: number;
  candidates: TuneBatchSizeCandidate[];
  cache_path: string | null;
}

/// Run the placement-tune measurement loop on the server. Blocks
/// for the duration of the sweep (30s–5min depending on model + N
/// candidates × repeats). Persists the winner to the per-device
/// tuner cache on success, so the next model load auto-applies it.
///
/// Pass exactly one of `modelName` (file stem from `/api/tags`,
/// the GUI's normal hook) or `modelPath` (absolute path, for
/// scripted use).
export async function runPlacementTune(opts: {
  modelName?: string;
  modelPath?: string;
  ctxSize?: number;
  promptTokens?: number;
  decodeTokens?: number;
  repeats?: number;
  vramMb?: number;
  vramHeadroomMb?: number;
}): Promise<TunePlacementResponse> {
  const r = await fetch(`${baseUrl()}/v1/tune/placement`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_name: opts.modelName,
      model_path: opts.modelPath,
      ctx_size: opts.ctxSize,
      prompt_tokens: opts.promptTokens,
      decode_tokens: opts.decodeTokens,
      repeats: opts.repeats,
      vram_mb: opts.vramMb,
      vram_headroom_mb: opts.vramHeadroomMb,
    }),
  });
  if (!r.ok) throw new Error(`/v1/tune/placement ${r.status}: ${await r.text()}`);
  return r.json();
}

export async function runBatchSizeTune(opts: {
  modelName?: string;
  modelPath?: string;
  candidates?: number[];
  promptTokens?: number;
  repeats?: number;
}): Promise<TuneBatchSizeResponse> {
  const r = await fetch(`${baseUrl()}/v1/tune/batch_size`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_name: opts.modelName,
      model_path: opts.modelPath,
      candidates: opts.candidates,
      prompt_tokens: opts.promptTokens,
      repeats: opts.repeats,
    }),
  });
  if (!r.ok) throw new Error(`/v1/tune/batch_size ${r.status}: ${await r.text()}`);
  return r.json();
}

/// Snapshot of the tuner-cache state for the active SYCL device,
/// plus the live `[tuning].auto_apply_*` flags. Drives the Status
/// page's Tuning panel. Returns gracefully-degraded fields when no
/// SYCL device is visible (no Intel GPU / oneAPI runtime).
export async function getTuningSummary(): Promise<TuningSummary> {
  const r = await fetch(`${baseUrl()}/v1/tuning_summary`);
  if (!r.ok) throw new Error(`/v1/tuning_summary ${r.status}: ${await r.text()}`);
  return r.json();
}

export interface TuningRecommendations {
  untuned_count: number;
  shapes: { kernel: string; m: number; k: number }[];
}

/// Distinct `(kernel, M, K)` matvec shapes the engine has dispatched
/// on this run without a cached LWS entry. The Status page polls this
/// to render a "N kernels untuned — Tune now" banner. `untuned_count`
/// is `0` when everything's tuned or when the engine hasn't dispatched
/// any USM matvec yet (no SYCL device present).
export async function getTuningRecommendations(): Promise<TuningRecommendations> {
  const r = await fetch(`${baseUrl()}/v1/tuning/recommendations`);
  if (!r.ok) throw new Error(`/v1/tuning/recommendations ${r.status}`);
  return r.json();
}

/// Clear the untuned-shape registry. Called after a successful tune
/// so the GUI banner clears without a full restart. Returns the new
/// (empty) snapshot for an immediate refresh.
export async function clearTuningRecommendations(): Promise<TuningRecommendations> {
  const r = await fetch(`${baseUrl()}/v1/tuning/recommendations`, {
    method: "POST",
  });
  if (!r.ok) throw new Error(`/v1/tuning/recommendations clear ${r.status}`);
  return r.json();
}

export interface CapabilitiesSnapshot {
  server_version: string;
  backends: BackendsSnapshot;
  // Other Capabilities fields exist on the wire (response_formats,
  // endpoints, embeddings, rerank, tools, history, audit_log, auth,
  // fim, moe) — the GUI doesn't consume them today, so we leave
  // them un-typed rather than pin a snapshot of a moving target.
  [key: string]: unknown;
}

export interface BackendsSnapshot {
  cpu: {
    available: boolean;
    /** Host SIMD features the runtime detected (avx, avx2, fma,
     * f16c, avx512f). Empty on non-x86. */
    simd_features: string[];
    /** Whether the rayon-parallel matvec path is active. */
    parallel_matvec: boolean;
    /** Logical CPU count the runtime sees (`available_parallelism`).
     * Absent on servers predating the CPU-tier inventory. */
    logical_cores?: number;
    /** Logical processors usable for compute = `logical_cores` minus
     * `disabled_cpus`. */
    enabled_cores?: number;
    /** Logical-processor indices excluded from the CPU pool
     * (`[inference].disabled_cpus`). Empty = every core is used. */
    disabled_cpus?: number[];
    /** Whether the CPU is an enabled compute tier (`false` = GPU-only
     * placement). */
    cpu_enabled?: boolean;
    /** Whether the GPU is an enabled compute tier (`false` = CPU-only
     * placement). */
    gpu_enabled?: boolean;
    /** Whether VRAM-only weight residency was requested. */
    vram_only?: boolean;
  };
  sycl: {
    /** True when the SYCL runtime enumerates ≥1 GPU. */
    available: boolean;
    device_count: number;
    /** "level_zero" | "opencl" | other backend label, null when
     * SYCL is unavailable. */
    backend: string | null;
    /** RUSTLLAMA_SYCL_BACKEND env var value or the default
     * "level_zero". */
    preference: string;
    /** True when backend == "level_zero" — the L0 USM import
     * fast-path is eligible. */
    l0_import_eligible: boolean;
  };
  /** CUDA (NVIDIA) backend — first-class peer of `sycl`. Absent on
   * servers predating the CUDA probe; all-false/zero on non-NVIDIA
   * hosts (the CUDA *driver* is dlopen'd, no toolkit needed to detect). */
  cuda?: {
    /** True when ≥1 NVIDIA GPU is visible to the CUDA driver. */
    available: boolean;
    /** NVIDIA GPUs the CUDA driver enumerates. */
    device_count: number;
    /** How many of those the native compute crate can launch kernels
     * on. `compute_ready == 0` with `device_count > 0` = driver/runtime
     * mismatch (GPU visible, kernels can't run). */
    compute_ready: number;
    /** CUDA driver version string when a GPU is present; null otherwise. */
    driver: string | null;
  };
  // `onednn` removed — oneDNN was ripped out of the runtime (2026-09-21);
  // the server's /v1/capabilities no longer reports it.
}

export async function getCapabilities(): Promise<CapabilitiesSnapshot> {
  const r = await fetch(`${baseUrl()}/v1/capabilities`);
  if (!r.ok) throw new Error(`/v1/capabilities ${r.status}`);
  return r.json();
}

export interface LanInfoResult {
  bind_addr: string;
  port: number;
  primary_lan_ip: string | null;
  api_key_set: boolean;
  /// URL another device should hit. Null when the server is
  /// loopback-bound or no routable interface was discovered.
  url: string | null;
  /// SVG markup encoding `url`. Null when `url` is null. Safe to
  /// embed via `dangerouslySetInnerHTML` — qrcode-rs SVG output
  /// contains only `<svg>` / `<path>` / `<rect>`, no scripts.
  qr_svg: string | null;
}

/// Discover the host's LAN IP + the URL a sibling device should
/// connect to + a QR encoding of that URL. The Settings page's LAN
/// access panel uses this so a phone can scan and reach the local
/// server.
export async function getLanInfo(): Promise<LanInfoResult> {
  const r = await fetch(`${baseUrl()}/v1/lan_info`);
  if (!r.ok) throw new Error(`/v1/lan_info ${r.status}: ${await r.text()}`);
  return r.json();
}

export interface ChatTemplatePreviewMessage {
  role: string;
  content: string;
}

export interface ChatTemplatePreviewResult {
  rendered: string;
  used_engine_specials: boolean;
}

/// Render a Jinja chat template against a sample conversation. The
/// Settings page uses this for the chat_template editor's live
/// preview — no model load, no tokenize, just runs the template
/// through the same minijinja env the engine uses, with the active
/// model's BOS/EOS forwarded when one is loaded.
export async function previewChatTemplate(opts: {
  template: string;
  messages: ChatTemplatePreviewMessage[];
  addGenerationPrompt?: boolean;
  signal?: AbortSignal;
}): Promise<ChatTemplatePreviewResult> {
  const r = await fetch(`${baseUrl()}/v1/chat/template/preview`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      template: opts.template,
      messages: opts.messages,
      add_generation_prompt: opts.addGenerationPrompt ?? true,
    }),
    signal: opts.signal,
  });
  if (!r.ok) {
    throw new Error(`/v1/chat/template/preview ${r.status}: ${await r.text()}`);
  }
  return r.json();
}

/// Switch the on-disk config to a named profile. Server merges the
/// profile's sparse overrides into the current config and saves it.
/// The on-disk watcher fires the same delta path PUT /v1/config
/// would, so live-applicable sections take effect immediately and
/// the response surfaces what needs a follow-up reload/restart.
export async function applyConfigProfile(name: string): Promise<ConfigPutResult> {
  const r = await fetch(`${baseUrl()}/v1/config/profile/apply`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!r.ok) {
    throw new Error(`/v1/config/profile/apply ${r.status}: ${await r.text()}`);
  }
  return r.json();
}

// ----- Conversation history (sqlite-backed; --features history) ----

export interface ConversationSummary {
  id: number;
  title: string;
  created_at: number;
  updated_at: number;
}

export interface StoredMessage {
  id: number;
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  created_at: number;
}

export interface Conversation extends ConversationSummary {
  messages: StoredMessage[];
}

/// `true` if the server has `--features history` and a store loaded.
/// Detected by probing the list endpoint — 501 means the feature is
/// off (server returns NOT_IMPLEMENTED) or no store is attached.
export async function historyAvailable(): Promise<boolean> {
  try {
    const r = await fetch(`${baseUrl()}/api/conversations`);
    return r.status !== 501 && r.status !== 404;
  } catch {
    return false;
  }
}

export async function listConversations(): Promise<ConversationSummary[]> {
  const r = await fetch(`${baseUrl()}/api/conversations`);
  if (r.status === 501 || r.status === 404) return [];
  if (!r.ok) throw new Error(`list conversations ${r.status}`);
  const body = await r.json();
  return body.conversations ?? [];
}

export async function getConversation(id: number): Promise<Conversation | null> {
  const r = await fetch(`${baseUrl()}/api/conversations/${id}`);
  if (r.status === 404) return null;
  if (!r.ok) throw new Error(`get conversation ${r.status}`);
  return r.json();
}

export async function createConversation(title: string): Promise<number> {
  const r = await fetch(`${baseUrl()}/api/conversations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!r.ok) throw new Error(`create conversation ${r.status}`);
  const body = await r.json();
  return body.id as number;
}

export async function appendMessage(
  id: number,
  role: StoredMessage["role"],
  content: string,
): Promise<void> {
  const r = await fetch(`${baseUrl()}/api/conversations/${id}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role, content }),
  });
  if (!r.ok) throw new Error(`append message ${r.status}`);
}

export async function deleteConversation(id: number): Promise<void> {
  const r = await fetch(`${baseUrl()}/api/conversations/${id}`, {
    method: "DELETE",
  });
  if (!r.ok && r.status !== 404) throw new Error(`delete conversation ${r.status}`);
}

export async function renameConversation(id: number, title: string): Promise<void> {
  const r = await fetch(`${baseUrl()}/api/conversations/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  if (!r.ok) throw new Error(`rename ${r.status}`);
}

export interface PullProgress {
  status: string;
  /// Bytes downloaded so far (present during the `downloading …` phase).
  completed?: number;
  /// Total bytes (Content-Length); 0/absent when the server can't report it.
  total?: number;
}

/// A GGUF-carrying model repo from `/api/hf/search`.
export interface HfModel {
  id: string;
  downloads: number;
  likes: number;
}

/// A `.gguf` file within a repo from `/api/hf/files`.
export interface HfFile {
  rfilename: string;
  size: number;
}

/// Realtime HuggingFace search for GGUF repos (proxied server-side).
export async function hfSearch(q: string, limit = 20): Promise<HfModel[]> {
  const r = await fetch(
    `${baseUrl()}/api/hf/search?q=${encodeURIComponent(q)}&limit=${limit}`,
  );
  if (!r.ok) throw new Error(`/api/hf/search ${r.status}: ${await r.text()}`);
  return r.json();
}

/// List the `.gguf` files in a repo (for picking which one to pull).
export async function hfFiles(repo: string): Promise<HfFile[]> {
  const r = await fetch(`${baseUrl()}/api/hf/files?repo=${encodeURIComponent(repo)}`);
  if (!r.ok) throw new Error(`/api/hf/files ${r.status}: ${await r.text()}`);
  return r.json();
}

export async function pullModel(
  modelRef: string,
  onProgress?: (p: PullProgress) => void,
): Promise<void> {
  // POST /api/pull streams NDJSON status updates. We surface each
  // line via the optional progress callback so the UI can show
  // "downloading…" / "verifying sha256" / etc.
  //
  // If any line carries `{"error": "..."}`, we stash the message,
  // finish draining the stream, then throw AFTER the loop so the
  // caller's `.catch` block runs. (Previously the throw lived inside
  // the per-line try/catch and was silently demoted to a progress
  // message, causing the Models page to overwrite the error with
  // "done" once the stream ended.)
  const r = await fetch(`${baseUrl()}/api/pull`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: modelRef, stream: true }),
  });
  if (!r.ok) {
    const body = await r.text();
    throw new Error(`/api/pull ${r.status}: ${body}`);
  }
  const reader = r.body?.getReader();
  if (!reader) return;
  const decoder = new TextDecoder();
  let buf = "";
  let serverError: string | null = null;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl: number;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      let obj:
        | { status?: string; error?: string; completed?: number; total?: number }
        | null = null;
      try {
        obj = JSON.parse(line);
      } catch {
        // Tolerate keepalive / partial frames — silently skip.
        continue;
      }
      if (obj?.status && onProgress)
        onProgress({
          status: obj.status,
          completed: obj.completed,
          total: obj.total,
        });
      if (obj?.error) {
        // Keep the first error we see — subsequent lines (if any)
        // are typically follow-on noise.
        if (!serverError) serverError = obj.error;
      }
    }
  }
  if (serverError) throw new Error(serverError);
}

// ----- Auto-tune: first-load sweep progress + manual re-tune -----

export interface TuneProgress {
  active: boolean;
  model_id: string;
  stage_idx: number;
  stage_total: number;
  stage_name: string;
  pct: number;
  line: string;
  log: string[];
  done: boolean;
  error: string | null;
  started_unix: number;
}

/// Poll the live auto-tune progress (GET /v1/tune/progress). Drives the
/// progress window shown during first-load auto-tune and manual re-tune.
export async function getTuneProgress(): Promise<TuneProgress> {
  const r = await fetch(`${baseUrl()}/v1/tune/progress`);
  if (!r.ok) throw new Error(`/v1/tune/progress ${r.status}: ${await r.text()}`);
  return r.json();
}

/// Force a re-tune of a known model (POST /v1/tune/model). `scope` picks
/// the full per-model sweep ("all", default) or just CPU-vs-GPU dispatch
/// ("placement"). The request blocks for the whole run (poll
/// getTuneProgress() meanwhile); after it resolves, reload the model to
/// apply the fresh winners.
export async function retuneModel(
  model: string,
  force = true,
  scope: "all" | "placement" = "all",
): Promise<void> {
  const r = await fetch(`${baseUrl()}/v1/tune/model`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model, force, scope }),
  });
  if (!r.ok) throw new Error(`/v1/tune/model ${r.status}: ${await r.text()}`);
}

// ----- Typed decisions -----

export interface DecideResult {
  // choice/score
  value?: string;
  index?: number;
  probabilities?: number[];
  logprobs?: number[];
  score?: number;
  // boolean
  probability?: number;
  confidence: number;
  calibrated: boolean;
}

/// Call a typed-decision endpoint. `kind` = "choice" | "score" | "boolean".
/// Body carries `context` (or `messages`) plus `options` / `levels` /
/// `question` depending on the kind.
export async function decide(
  kind: "choice" | "score" | "boolean",
  body: Record<string, unknown>,
): Promise<DecideResult> {
  const r = await fetch(`${baseUrl()}/v1/decide/${kind}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`/v1/decide/${kind} ${r.status}: ${await r.text()}`);
  return r.json();
}

export interface ChatStreamHandlers {
  onContent: (delta: string) => void;
  /** OpenAI-compat: every chunk carries `system_fingerprint`. The
   * Chat page tracks this across consecutive requests so a backend
   * swap mid-conversation (KV dtype change, model reload, server
   * version bump) can be surfaced to the user instead of silently
   * altering output behavior. `systemFingerprint` reflects the
   * value seen on the final non-`[DONE]` chunk. */
  onDone: (info?: {
    finishReason?: string;
    usage?: UsageInfo;
    systemFingerprint?: string;
  }) => void;
  onError: (err: Error) => void;
  /// Fired once when the first SSE chunk arrives, carrying the
  /// request id (`chatcmpl-…`). Callers stash this so the Stop
  /// button can POST `/v1/cancel` and actually stop the engine
  /// rather than just dropping SSE chunks on the client side.
  /// Optional — pre-existing callers ignore it.
  onStart?: (requestId: string) => void;
  /// CLARIFY: fired when the model calls the reserved `ask_user` tool.
  /// Carries the question `prompt` + selectable `options` for the UI to
  /// present a chooser (rides the terminal chunk, finish_reason "ask_user").
  onAskUser?: (q: AskUser) => void;
  /// Fired once, just before `onDone`, when the model proposed tool calls
  /// (finish_reason "tool_calls"), with the streamed deltas reassembled.
  /// The chat doesn't execute tools — this is for a confirm / auto-run
  /// policy over the proposal. Optional; pre-existing callers ignore it.
  onToolCalls?: (calls: ToolCall[]) => void;
}

/// CLARIFY question surfaced on `delta.ask_user`.
export interface AskUser {
  id: string;
  kind: string;
  prompt: string;
  options: string[];
}

/// A tool call reassembled from the streaming `tool_calls` deltas.
export interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

/// Cancel an in-flight streaming request by id. Server flips the
/// engine's cancel flag, generation stops mid-token, resources
/// freed immediately. The corresponding fetch on the client side
/// should still be aborted via its AbortSignal — the two are
/// complementary (one cancels the HTTP transport, the other the
/// upstream work).
export async function cancelRequest(id: string): Promise<void> {
  const r = await fetch(`${baseUrl()}/v1/cancel`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id }),
  });
  if (!r.ok) {
    // 404 for "request not registered" is common (race: completed
    // before our cancel arrived). Throw on other failures.
    if (r.status === 404) return;
    throw new Error(`/v1/cancel ${r.status}: ${await r.text()}`);
  }
}

/// Non-streaming /v1/chat/completions — send messages, get the full
/// assistant string back in one shot. Used by the chat's context
/// compaction ("ask the model to summarize the conversation so far"),
/// where streaming buys nothing. Text-only (images aren't relevant to a
/// summary). Defaults to a low temperature for a stable, faithful summary.
export async function chatOnce(
  messages: ChatMessage[],
  opts?: {
    model?: string;
    temperature?: number;
    maxTokens?: number;
    abort?: AbortSignal;
  },
): Promise<string> {
  const r = await fetch(`${baseUrl()}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: opts?.model,
      messages: messages.map((m) => ({ role: m.role, content: m.content })),
      temperature: opts?.temperature ?? 0.3,
      max_tokens: opts?.maxTokens ?? 1024,
      stream: false,
    }),
    signal: opts?.abort,
  });
  if (!r.ok) {
    throw new Error(`chat ${r.status}: ${await r.text().catch(() => "")}`);
  }
  const j = await r.json();
  return (j?.choices?.[0]?.message?.content as string) ?? "";
}

/// Streaming /v1/chat/completions client. Parses OpenAI SSE chunks
/// (lines prefixed `data: `) and dispatches `choices[0].delta.content`
/// to onContent. Handles the `[DONE]` terminator + optional usage chunk.
export async function streamChat(
  messages: ChatMessage[],
  opts: {
    model?: string;
    temperature?: number;
    maxTokens?: number;
    seed?: number;
    /// CLARIFY opt-in. When true, the server injects the reserved
    /// `ask_user` tool so the model may pause and ask a clarifying
    /// question (delivered to `onAskUser`) instead of guessing. Omitted
    /// from the request body when falsy, so the OFF request is
    /// wire-identical to the pre-CLARIFY plain path.
    allowClarify?: boolean;
    abort?: AbortSignal;
  },
  h: ChatStreamHandlers,
): Promise<void> {
  let resp: Response;
  try {
    resp = await fetch(`${baseUrl()}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: opts.model,
        // Messages with attached images go out as OpenAI multimodal
        // content-block arrays; text-only messages stay plain strings.
        messages: messages.map((m) =>
          m.images && m.images.length > 0
            ? {
                role: m.role,
                content: [
                  ...m.images.map((u) => ({
                    type: "image_url",
                    image_url: { url: u },
                  })),
                  { type: "text", text: m.content },
                ],
              }
            : { role: m.role, content: m.content },
        ),
        temperature: opts.temperature,
        max_tokens: opts.maxTokens,
        seed: opts.seed,
        stream: true,
        stream_options: { include_usage: true },
        // Only send the flag when enabled — an omitted field keeps the
        // request byte-identical to the plain (no-tool) path.
        ...(opts.allowClarify ? { allow_clarify: true } : {}),
      }),
      signal: opts.abort,
    });
  } catch (e) {
    h.onError(e instanceof Error ? e : new Error(String(e)));
    return;
  }
  if (!resp.ok) {
    h.onError(new Error(`chat/completions ${resp.status}: ${await resp.text()}`));
    return;
  }
  const reader = resp.body?.getReader();
  if (!reader) {
    h.onDone();
    return;
  }
  const decoder = new TextDecoder();
  let buf = "";
  let usage: UsageInfo | undefined;
  let finishReason: string | undefined;
  let systemFingerprint: string | undefined;
  // Fire onStart once when we see the first chunk with an id.
  // Server emits `chatcmpl-<…>` on every chunk, so any frame with
  // a string id works; capture-on-first prevents re-firing.
  let requestIdNotified = false;
  // Reassemble streaming `tool_calls` deltas (header + arg fragments),
  // keyed by index, then hand them to `onToolCalls` once at the end.
  const toolAcc: ToolCall[] = [];
  let toolCallsFlushed = false;
  const flushToolCalls = () => {
    if (!toolCallsFlushed && toolAcc.length > 0) {
      toolCallsFlushed = true;
      h.onToolCalls?.(toolAcc.filter((c) => c.name.length > 0));
    }
  };
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      // Process complete SSE lines (`data: {...}\n\n`).
      let nl: number;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl);
        buf = buf.slice(nl + 1);
        const data = line.startsWith("data: ") ? line.slice(6) : "";
        if (!data) continue;
        if (data.trim() === "[DONE]") {
          flushToolCalls();
          h.onDone({ finishReason, usage, systemFingerprint });
          return;
        }
        try {
          const obj = JSON.parse(data);
          if (!requestIdNotified && typeof obj.id === "string" && h.onStart) {
            h.onStart(obj.id);
            requestIdNotified = true;
          }
          // Every chunk carries the fingerprint at envelope level;
          // last-seen wins so a backend swap mid-stream (theoretically
          // possible if the server reloads the active model) is
          // detectable from the final value.
          if (typeof obj.system_fingerprint === "string") {
            systemFingerprint = obj.system_fingerprint;
          }
          const choice = obj.choices?.[0];
          const delta = choice?.delta?.content;
          if (typeof delta === "string" && delta.length > 0) {
            h.onContent(delta);
          }
          // Reassemble streaming tool-call deltas into `toolAcc`.
          const tcDeltas = choice?.delta?.tool_calls;
          if (Array.isArray(tcDeltas)) {
            for (const c of tcDeltas) {
              const idx = typeof c?.index === "number" ? c.index : 0;
              while (toolAcc.length <= idx)
                toolAcc.push({ id: "", name: "", arguments: "" });
              const slot = toolAcc[idx];
              if (typeof c?.id === "string" && c.id) slot.id = c.id;
              if (typeof c?.function?.name === "string" && c.function.name)
                slot.name = c.function.name;
              if (typeof c?.function?.arguments === "string")
                slot.arguments += c.function.arguments;
            }
          }
          // CLARIFY: the reserved `ask_user` tool rides the terminal chunk
          // as `delta.ask_user` (finish_reason "ask_user").
          const ask = choice?.delta?.ask_user;
          if (ask && typeof ask === "object") {
            h.onAskUser?.({
              id: typeof ask.id === "string" ? ask.id : "",
              kind: typeof ask.kind === "string" ? ask.kind : "clarify",
              prompt: typeof ask.prompt === "string" ? ask.prompt : "",
              options: Array.isArray(ask.options)
                ? ask.options.filter((o: unknown): o is string => typeof o === "string")
                : [],
            });
          }
          if (choice?.finish_reason) {
            finishReason = choice.finish_reason;
            // Surface any reassembled tool calls once the turn resolves.
            if (finishReason === "tool_calls") flushToolCalls();
          }
          if (obj.usage) {
            usage = obj.usage as UsageInfo;
          }
        } catch {
          // Tolerate keepalive comments / partial frames.
        }
      }
    }
    flushToolCalls();
    h.onDone({ finishReason, usage, systemFingerprint });
  } catch (e) {
    h.onError(e instanceof Error ? e : new Error(String(e)));
  }
}
