// Status page: server health, loaded model summary, available API
// endpoints, a quick API-call playground that reports tokens-per-second,
// and live sparkline charts of tok/s + ctx-used + pending requests
// driven by polling /v1/metrics at 1 Hz. Health + model list still
// refresh every 5 s on a slower cadence.

import { useEffect, useState } from "react";
import {
  clearTuningRecommendations,
  getCapabilities,
  getHealth,
  getMetrics,
  getTuningRecommendations,
  getTuningSummary,
  listModels,
  listOllamaTags,
  retuneModel,
  streamChat,
  type BackendsSnapshot,
  type MetricsSnapshot,
  type ModelInfo,
  type TagsModel,
  type TuningRecommendations,
  type TuningSummary,
} from "../api";
import TuneProgressModal from "../TuneProgressModal";

/// Render the tuner cache's `last_tuned` for humans. The server writes a
/// unix-epoch-seconds string; the CLI may write ISO-8601. Handle both, and
/// fall back to the raw value if it parses as neither.
function fmtLastTuned(v: string | null): string {
  if (!v) return "—";
  const s = v.trim();
  const d = /^\d+$/.test(s) ? new Date(Number(s) * 1000) : new Date(s);
  return isNaN(d.getTime()) ? v : d.toLocaleString();
}

interface ProbeResult {
  tokens: number;
  decode_ms?: number;
  prefill_ms?: number;
  cache_hit_tokens?: number;
  ok: boolean;
  err?: string;
}

/// How many metric samples to retain for the sparklines. At 1 Hz this
/// is one minute of history. Beyond that, the chart shape stops being
/// informative for short generations.
const METRICS_WINDOW = 60;

export default function StatusPage() {
  const [health, setHealth] = useState<{ status: string; draining?: boolean; version?: string; model_id?: string } | null>(null);
  const [healthErr, setHealthErr] = useState<string | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [probe, setProbe] = useState<ProbeResult | null>(null);
  const [probing, setProbing] = useState(false);
  const [metrics, setMetrics] = useState<MetricsSnapshot | null>(null);
  const [tokSeries, setTokSeries] = useState<number[]>([]);
  const [ctxSeries, setCtxSeries] = useState<number[]>([]);
  const [pendingSeries, setPendingSeries] = useState<number[]>([]);
  // GPU sensor series. Each is appended-and-trimmed on the same 1 Hz
  // tick as the tok/s sparklines so they line up visually. Only
  // rendered when the server's `/v1/metrics` reports a non-null
  // `gpu_sysman` snapshot — on non-Intel hosts (no Level Zero Sysman)
  // the section hides entirely.
  const [vramUsedSeries, setVramUsedSeries] = useState<number[]>([]);
  const [tempSeries, setTempSeries] = useState<number[]>([]);
  // Instantaneous power is derived from two consecutive energy
  // counter samples (Sysman exposes a cumulative energy register, not
  // a live power register). We keep the previous (uj, ts_us) pair so
  // the next tick can compute dW = dE / dt. `null` until the second
  // tick.
  const [powerSeries, setPowerSeries] = useState<number[]>([]);
  const [freqSeries, setFreqSeries] = useState<number[]>([]);
  // Previous energy-counter sample. Stored as a ref-style state so the
  // tick closure can read and replace it atomically.
  const [prevEnergy, setPrevEnergy] = useState<{ uj: number; ts_us: number } | null>(null);
  /// Snapshot of the tuner-cache state for the active device. Polled
  /// on a slow tick (10s) — the cache only changes when the user
  /// runs `rustllama tune`, so refreshing more often would be wasted
  /// fetch traffic.
  const [tuning, setTuning] = useState<TuningSummary | null>(null);
  const [tuningErr, setTuningErr] = useState<string | null>(null);
  /// Pull-shaped TuningRecommended event. Polled alongside the tuning
  /// summary on the same 10 s tick. When `untuned_count > 0` the GUI
  /// renders a yellow banner with "Tune now" + the kernel list; the
  /// "Run …" buttons stay in place below it.
  const [recs, setRecs] = useState<TuningRecommendations | null>(null);
  /// Active compute backends — CPU SIMD features, SYCL GPU
  /// availability + L0/OpenCL backend, oneDNN GEMM-fallback
  /// status. Polled once at mount (the data doesn't change
  /// mid-process — adding a GPU requires a server restart) and
  /// rendered in the Backends panel.
  const [backends, setBackends] = useState<BackendsSnapshot | null>(null);
  /// Collapsed by default — 30+ kernel rows would otherwise dominate
  /// the page. Toggled by the "Show / hide" link next to the header.
  const [showKernels, setShowKernels] = useState(false);
  /// Cached-model file stems for the tune model picker. Populated
  /// from `/api/tags` (which surfaces what `rustllama_hub::list_cached`
  /// finds). Empty array → the run-tune buttons are disabled with
  /// a "no cached models" hint.
  const [tuneCachedModels, setTuneCachedModels] = useState<TagsModel[]>([]);
  const [tuneSelectedModel, setTuneSelectedModel] = useState<string>("");
  /// Which tune is in flight: "placement" / "batch_size" / null
  /// (idle). When non-null the buttons disable + the panel shows a
  /// "tuning…" line with elapsed seconds.
  const [tuneRunning, setTuneRunning] = useState<null | "model" | "cpu_gpu" | "all_models">(null);
  const [tuneRunStart, setTuneRunStart] = useState<number>(0);
  // Progress modal shown while a tune subprocess runs.
  const [tuneModalOpen, setTuneModalOpen] = useState(false);
  // While tuning all models: {i: current 1-based, n: total} for the label.
  const [tuneAllIdx, setTuneAllIdx] = useState<{ i: number; n: number } | null>(null);
  /// Outcome of the most recent run-tune action. Null = idle, string
  /// for either a success message ("winner: n_gpu_layers=16 at
  /// 11.2 tok/s") or an error ("error: …"). Drives the inline
  /// result line below the buttons.
  const [tuneStatus, setTuneStatus] = useState<string | null>(null);

  useEffect(() => {
    let cancel = false;
    const tick = async () => {
      try {
        const tags = await listOllamaTags();
        if (cancel) return;
        setTuneCachedModels(tags);
        // Auto-select the first cached model when the picker hasn't
        // been touched yet, so the buttons are immediately actionable.
        if (tags.length > 0) {
          setTuneSelectedModel((prev) => prev || tags[0].name);
        }
      } catch {
        // Hub puller may be unreachable — leave the dropdown empty.
      }
    };
    tick();
    const id = setInterval(tick, 15_000);
    return () => {
      cancel = true;
      clearInterval(id);
    };
  }, []);

  // Re-render the elapsed-time counter every second while a tune
  // is in flight. The actual fetch is fired-and-awaited; this
  // interval just refreshes the "tuning… 23s" label.
  useEffect(() => {
    if (!tuneRunning) return;
    const id = setInterval(() => {
      // Force a re-render by setting state to a fresh-but-equal value;
      // React skips the rerender if state is reference-equal, so we
      // just bump a synthetic counter. Simpler: rely on `setTuneRunStart`
      // staying stable while we read Date.now() in the render path.
      // No-op tick is cheap; the panel reads Date.now() - tuneRunStart.
      setTuneRunning((v) => v);
    }, 1000);
    return () => clearInterval(id);
  }, [tuneRunning]);

  // Two consolidated actions, both driven by the safe subprocess tuner
  // (which mmap-shares weights instead of loading a second copy in-process
  // — the old in-process placement/batch-size endpoints could OOM-crash the
  // server on a big model). "model" = full per-model sweep; "cpu_gpu" = just
  // the CPU-vs-GPU dispatch (placement) measurement. The progress window
  // polls /v1/tune/progress while the sweep runs.
  const startTune = async (kind: "model" | "cpu_gpu") => {
    if (!tuneSelectedModel || tuneRunning) return;
    setTuneRunning(kind);
    setTuneRunStart(Date.now());
    setTuneStatus(null);
    setTuneModalOpen(true);
    try {
      await retuneModel(tuneSelectedModel, true, kind === "model" ? "all" : "placement");
      setTuneStatus(
        kind === "model"
          ? "model tune complete — reload the model to apply the new winners"
          : "CPU/GPU dispatch tune complete — reload the model to apply",
      );
      try {
        setTuning(await getTuningSummary());
      } catch {
        /* leave stale state; next poll will refresh */
      }
      try {
        setRecs(await clearTuningRecommendations());
      } catch {
        /* non-fatal; next poll picks up the new state */
      }
    } catch (e) {
      setTuneStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setTuneRunning(null);
      setTuneModalOpen(false);
    }
  };

  // Full sweep for every cached model, one after another. The progress
  // window shows each model's stages as the loop advances (the server's
  // progress state carries the model currently being tuned).
  const startTuneAll = async () => {
    if (tuneRunning || tuneCachedModels.length === 0) return;
    const models = tuneCachedModels.map((m) => m.name);
    setTuneRunning("all_models");
    setTuneRunStart(Date.now());
    setTuneStatus(null);
    setTuneModalOpen(true);
    try {
      for (let i = 0; i < models.length; i++) {
        setTuneAllIdx({ i: i + 1, n: models.length });
        await retuneModel(models[i], true, "all");
      }
      setTuneStatus(
        `tuned all ${models.length} model${models.length === 1 ? "" : "s"} — reload to apply`,
      );
      try {
        setTuning(await getTuningSummary());
      } catch {
        /* next poll refreshes */
      }
      try {
        setRecs(await clearTuningRecommendations());
      } catch {
        /* non-fatal */
      }
    } catch (e) {
      setTuneStatus(`error: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setTuneRunning(null);
      setTuneAllIdx(null);
      setTuneModalOpen(false);
    }
  };

  useEffect(() => {
    let cancel = false;
    const tick = async () => {
      try {
        const t = await getTuningSummary();
        if (cancel) return;
        setTuning(t);
        setTuningErr(null);
      } catch (e) {
        if (cancel) return;
        setTuningErr(e instanceof Error ? e.message : String(e));
      }
      try {
        const r = await getTuningRecommendations();
        if (cancel) return;
        setRecs(r);
      } catch {
        // Endpoint missing on older builds — leave recs null and the
        // banner just doesn't render.
      }
    };
    tick();
    const id = setInterval(tick, 10_000);
    return () => {
      cancel = true;
      clearInterval(id);
    };
  }, []);

  // Backends snapshot. Static for the process lifetime, so a single
  // mount-time fetch is enough — no interval poll. Server restarts
  // are infrequent and trigger a page refresh anyway.
  useEffect(() => {
    let cancel = false;
    (async () => {
      try {
        const caps = await getCapabilities();
        if (!cancel) setBackends(caps.backends);
      } catch {
        // Endpoint missing or fetch failed; panel hides.
      }
    })();
    return () => {
      cancel = true;
    };
  }, []);

  useEffect(() => {
    let cancel = false;
    const tick = async () => {
      try {
        const h = await getHealth();
        if (cancel) return;
        setHealth(h);
        setHealthErr(null);
      } catch (e) {
        if (cancel) return;
        setHealth(null);
        setHealthErr(e instanceof Error ? e.message : String(e));
      }
      try {
        const m = await listModels();
        if (!cancel) setModels(m);
      } catch {
        // /v1/models may degrade independently — keep prior list.
      }
    };
    tick();
    const id = setInterval(tick, 5000);
    return () => {
      cancel = true;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    let cancel = false;
    const tick = async () => {
      try {
        const m = await getMetrics();
        if (cancel) return;
        setMetrics(m);
        // Append-and-trim each series. The chart reads the array
        // directly so React's shallow-compare on Array identity is
        // what triggers a re-render.
        // Chart the smoothed EMA when available so the line doesn't
        // jitter wildly between cold and warm requests. Falls back
        // to the per-request value before the first generation.
        setTokSeries((prev) => trim([...prev, m.ema_tok_s ?? m.last_tok_s ?? 0]));
        setCtxSeries((prev) => trim([...prev, m.ctx_used]));
        setPendingSeries((prev) => trim([...prev, m.pending]));
        // GPU sensor sparklines. Each series only appends when its
        // probe returned a real value — None inputs are skipped so the
        // chart shape isn't polluted with zeros on partial sensor sets
        // (e.g. Iris Xe exposes VRAM + freq but not temp/power).
        const s = m.gpu_sysman;
        if (s) {
          if (typeof s.vram_total_bytes === "number" && typeof s.vram_free_bytes === "number") {
            const usedGb = (s.vram_total_bytes - s.vram_free_bytes) / 1_073_741_824;
            setVramUsedSeries((prev) => trim([...prev, usedGb]));
          }
          if (typeof s.max_temp_c === "number") {
            setTempSeries((prev) => trim([...prev, s.max_temp_c as number]));
          }
          if (typeof s.gpu_freq_mhz === "number") {
            setFreqSeries((prev) => trim([...prev, s.gpu_freq_mhz as number]));
          }
          // Derive instantaneous power from two consecutive energy
          // samples. Energy is µJ, timestamps are µs ⇒ W = dE / dt.
          // Skip the first tick (no prior sample) and skip ticks
          // where dt <= 0 (clock jitter / sysman replay).
          if (typeof s.energy_uj === "number" && typeof s.energy_timestamp_us === "number") {
            const cur = { uj: s.energy_uj, ts_us: s.energy_timestamp_us };
            if (prevEnergy && cur.ts_us > prevEnergy.ts_us) {
              const dE = cur.uj - prevEnergy.uj;
              const dt = cur.ts_us - prevEnergy.ts_us;
              const watts = dE / dt; // µJ/µs == W
              if (watts >= 0 && watts < 1000) {
                setPowerSeries((prev) => trim([...prev, watts]));
              }
            }
            setPrevEnergy(cur);
          }
        }
      } catch {
        // Metrics may be transiently unavailable during model swap
        // — keep the existing series so the chart doesn't flicker.
      }
    };
    tick();
    const id = setInterval(tick, 1000);
    return () => {
      cancel = true;
      clearInterval(id);
    };
  }, []);

  const runProbe = async () => {
    setProbing(true);
    setProbe(null);
    let tokens = 0;
    await streamChat(
      [{ role: "user", content: "Write one short sentence." }],
      { temperature: 0.7, maxTokens: 64 },
      {
        onContent: () => {
          tokens++;
        },
        onDone: (info) => {
          setProbe({
            tokens,
            decode_ms: info?.usage?.decode_ms,
            prefill_ms: info?.usage?.prefill_ms,
            cache_hit_tokens: info?.usage?.cache_hit_tokens,
            ok: true,
          });
          setProbing(false);
        },
        onError: (e) => {
          setProbe({ tokens, ok: false, err: e.message });
          setProbing(false);
        },
      },
    );
  };

  const probeTps =
    probe?.ok && probe.decode_ms && probe.decode_ms > 0
      ? (probe.tokens / (probe.decode_ms / 1000)).toFixed(1)
      : null;

  return (
    <div style={{ padding: 20 }}>
      <h2 style={{ margin: "0 0 16px 0", fontSize: 16, fontWeight: 600 }}>Status</h2>

      <section style={card}>
        <h3 style={cardHeader}>Server</h3>
        {healthErr && <div style={errBox}>error: {healthErr}</div>}
        {!healthErr && (
          <>
            <KV k="Status" v={health?.draining ? "draining" : health?.status ?? "—"} />
            <KV k="Version" v={health?.version ?? "—"} />
            <KV
              k="Loaded model"
              v={(() => {
                if (!health?.model_id) return "—";
                // Pair the model id with the MoE shorthand when the
                // active model is MoE — saved from the /v1/models
                // entry that matches the active model id. Avoids
                // round-tripping through /v1/capabilities for the
                // same data.
                const active = models.find((m) => m.id === health.model_id);
                if (active?.moe) {
                  const shared = active.moe.n_experts_shared;
                  const suffix =
                    shared > 0
                      ? `MoE ${active.moe.n_experts}×${active.moe.n_experts_used}+${shared}s`
                      : `MoE ${active.moe.n_experts}×${active.moe.n_experts_used}`;
                  return (
                    <span>
                      {health.model_id}
                      <span
                        style={{
                          marginLeft: 8,
                          padding: "1px 6px",
                          background: "rgba(163, 113, 247, 0.15)",
                          color: "var(--ll-purple)",
                          border: "1px solid rgba(163, 113, 247, 0.5)",
                          borderRadius: 10,
                          fontSize: 10,
                          fontFamily: "ui-monospace, monospace",
                        }}
                        title={
                          shared > 0
                            ? `${active.moe.n_experts} routed, top-${active.moe.n_experts_used}, ${shared} shared`
                            : `${active.moe.n_experts} routed, top-${active.moe.n_experts_used}`
                        }
                      >
                        {suffix}
                      </span>
                    </span>
                  );
                }
                return health.model_id;
              })()}
            />
          </>
        )}
      </section>

      {backends && (
        <section style={{ ...card, marginTop: 16 }}>
          <h3 style={cardHeader}>
            Backends
            <span style={{ marginLeft: 8, fontSize: 11, color: "var(--ll-text-muted)" }}>
              (compute dispatch paths)
            </span>
          </h3>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            {/* CPU column */}
            <div>
              <div style={{ fontSize: 12, color: "var(--ll-text-muted)" }}>CPU 0</div>
              <div
                style={{
                  fontSize: 18,
                  fontWeight: 600,
                  color: "var(--ll-green)",
                  marginTop: 2,
                }}
              >
                active
              </div>
              <div style={{ fontSize: 11, color: "var(--ll-text-muted)", marginTop: 6 }}>
                {backends.cpu.simd_features.length > 0 ? (
                  <>SIMD: {backends.cpu.simd_features.join(", ")}</>
                ) : (
                  <>scalar (non-x86)</>
                )}
              </div>
              <div style={{ fontSize: 11, color: "var(--ll-text-muted)", marginTop: 2 }}>
                rayon matvec:{" "}
                <code style={inlineCode}>
                  {backends.cpu.parallel_matvec ? "on (M≥256)" : "off"}
                </code>
              </div>
            </div>

            {/* SYCL column */}
            <div>
              <div style={{ fontSize: 12, color: "var(--ll-text-muted)" }}>SYCL (GPU)</div>
              <div
                style={{
                  fontSize: 18,
                  fontWeight: 600,
                  color: backends.sycl.available ? "var(--ll-green)" : "var(--ll-text-faint)",
                  marginTop: 2,
                }}
              >
                {backends.sycl.available
                  ? backends.sycl.backend ?? "active"
                  : "unavailable"}
              </div>
              {backends.sycl.available ? (
                <>
                  <div style={{ fontSize: 11, color: "var(--ll-text-muted)", marginTop: 6 }}>
                    devices: <code style={inlineCode}>{backends.sycl.device_count}</code>
                  </div>
                  {metrics?.gpus && metrics.gpus.length > 0 && (
                    <div style={{ fontSize: 11, color: "var(--ll-text-muted)", marginTop: 4 }}>
                      {metrics.gpus.map((g) => (
                        <div
                          key={g.index}
                          style={{ display: "flex", gap: 6, alignItems: "center" }}
                          title={g.name}
                        >
                          <code style={inlineCode}>GPU {g.index}</code>
                          <span
                            style={{
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                              whiteSpace: "nowrap",
                            }}
                          >
                            {g.name}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                  <div style={{ fontSize: 11, color: "var(--ll-text-muted)", marginTop: 2 }}>
                    preference: <code style={inlineCode}>{backends.sycl.preference}</code>
                  </div>
                  <div style={{ fontSize: 11, marginTop: 2 }}>
                    {backends.sycl.l0_import_eligible ? (
                      <span style={{ color: "var(--ll-green)" }}>L0 USM import fast-path eligible</span>
                    ) : (
                      <span style={{ color: "var(--ll-yellow)" }}>
                        L0 fast-path disabled (using {backends.sycl.backend ?? "fallback"})
                      </span>
                    )}
                  </div>
                </>
              ) : (
                <div style={{ fontSize: 11, color: "var(--ll-text-muted)", marginTop: 6 }}>
                  no SYCL GPU visible
                  {backends.sycl.preference !== "level_zero" && (
                    <> · preference=<code style={inlineCode}>{backends.sycl.preference}</code></>
                  )}
                </div>
              )}
            </div>

            {/* oneDNN column removed — oneDNN was ripped out (2026-09-21);
                the server no longer reports `backends.onednn`, so reading
                it here crashed the whole Status page. */}
          </div>
        </section>
      )}

      {metrics && (
        <section style={{ ...card, marginTop: 16 }}>
          <h3 style={cardHeader}>
            Compute inventory
            <span style={{ marginLeft: 8, fontSize: 11, color: "var(--ll-text-muted)" }}>
              (every GPU + the CPU tier)
            </span>
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {(metrics.gpus ?? []).map((g) => {
              const used =
                typeof g.vram_total_bytes === "number" &&
                typeof g.vram_free_bytes === "number"
                  ? g.vram_total_bytes - g.vram_free_bytes
                  : null;
              return (
                <DeviceRow
                  key={`gpu-${g.index}`}
                  badge={gpuBackendBadge(g.vendor)}
                  badgeColor={vendorColor(g.vendor)}
                  badgeTitle={gpuBackendTitle(g.vendor, backends)}
                  label={`GPU ${g.index}`}
                  name={g.name}
                  memLabel="VRAM"
                  usedBytes={used}
                  totalBytes={
                    typeof g.vram_total_bytes === "number" ? g.vram_total_bytes : null
                  }
                  utilPct={
                    typeof g.utilization_pct === "number" ? g.utilization_pct : null
                  }
                />
              );
            })}
            {/* CPU tier — the always-present fallback compute tier. Core
                counts + tier flags come from /v1/capabilities (backends.cpu);
                brand, RAM, and utilization come from /v1/metrics. */}
            <DeviceRow
              badge="CPU"
              badgeColor="var(--ll-text-muted)"
              label="CPU"
              name={metrics.cpu_brand ?? "Host CPU"}
              sub={cpuSubline(backends)}
              memLabel="RAM"
              usedBytes={
                typeof metrics.ram_total_bytes === "number" &&
                typeof metrics.ram_available_bytes === "number"
                  ? metrics.ram_total_bytes - metrics.ram_available_bytes
                  : null
              }
              totalBytes={
                typeof metrics.ram_total_bytes === "number" ? metrics.ram_total_bytes : null
              }
              utilPct={
                typeof metrics.cpu_utilization_pct === "number"
                  ? metrics.cpu_utilization_pct
                  : null
              }
            />
          </div>
          {(metrics.gpus?.length ?? 0) === 0 && (
            <div style={{ marginTop: 8, fontSize: 11, color: "var(--ll-text-muted)" }}>
              No GPU visible — inference runs on the CPU tier.
            </div>
          )}
        </section>
      )}

      <section style={{ ...card, marginTop: 16 }}>
        <h3 style={cardHeader}>Live metrics</h3>
        {metrics ? (
          <>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
              <ChartTile
                label="Tokens / second"
                value={
                  metrics.ema_tok_s
                    ? metrics.ema_tok_s.toFixed(1)
                    : metrics.last_tok_s
                      ? metrics.last_tok_s.toFixed(1)
                      : "—"
                }
                series={tokSeries}
                stroke="var(--ll-green)"
                fill="rgba(63, 185, 80, 0.12)"
              />
              <ChartTile
                label="KV context used"
                value={`${metrics.ctx_used} / ${metrics.ctx_size}`}
                series={ctxSeries}
                yMax={metrics.ctx_size}
                stroke="var(--ll-accent)"
                fill="rgba(88, 166, 255, 0.12)"
              />
              <ChartTile
                label="Pending requests"
                value={`${metrics.pending} / ${metrics.max_pending}`}
                series={pendingSeries}
                yMax={metrics.max_pending}
                stroke="var(--ll-yellow)"
                fill="rgba(210, 153, 34, 0.12)"
              />
            </div>
            <div style={{ marginTop: 10, fontSize: 12, color: "var(--ll-text-muted)" }}>
              Last request: prefill {metrics.last_prefill_ms?.toFixed(1) ?? "—"} ms,
              decode {metrics.last_decode_ms?.toFixed(1) ?? "—"} ms,
              cache hits {metrics.last_cache_hit_tokens ?? 0},
              generated {metrics.last_tokens_generated ?? 0}.
              Uptime {formatUptime(metrics.uptime_s)}.
            </div>
            <div style={{ marginTop: 6, fontSize: 12, color: "var(--ll-text-muted)", display: "flex", gap: 14, flexWrap: "wrap" }}>
              <span>
                KV dtype: <code style={inlineCode}>{metrics.kv_dtype ?? "—"}</code>
              </span>
              <span>
                Concurrency: <code style={inlineCode}>{metrics.concurrency}</code>
                {metrics.concurrency > 1 && (
                  <span style={{ marginLeft: 4, color: "var(--ll-green)" }}>
                    (multi-flight)
                  </span>
                )}
              </span>
            </div>
          </>
        ) : (
          <div style={{ color: "var(--ll-text-muted)", fontSize: 13 }}>Waiting for first /v1/metrics tick…</div>
        )}
      </section>

      {metrics?.gpu_sysman && (
        <section style={{ ...card, marginTop: 16 }}>
          <h3 style={cardHeader}>
            GPU sensors
            <span
              style={{ marginLeft: 8, fontSize: 11, color: "var(--ll-text-muted)" }}
              title="Level Zero Sysman snapshot. Iris Xe integrated graphics typically exposes VRAM + freq but not temp/power; discrete Arc exposes all four. Section hides on non-Intel hosts."
            >
              (Level Zero Sysman)
            </span>
          </h3>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 12 }}>
            {typeof metrics.gpu_sysman.vram_total_bytes === "number" &&
              typeof metrics.gpu_sysman.vram_free_bytes === "number" && (
                <ChartTile
                  label="VRAM used (GB)"
                  value={`${(
                    (metrics.gpu_sysman.vram_total_bytes - metrics.gpu_sysman.vram_free_bytes) /
                    1_073_741_824
                  ).toFixed(2)} / ${(metrics.gpu_sysman.vram_total_bytes / 1_073_741_824).toFixed(1)}`}
                  series={vramUsedSeries}
                  yMax={metrics.gpu_sysman.vram_total_bytes / 1_073_741_824}
                  stroke="var(--ll-purple)"
                  fill="rgba(163, 113, 247, 0.12)"
                />
              )}
            {typeof metrics.gpu_sysman.gpu_freq_mhz === "number" && (
              <ChartTile
                label="GPU clock (MHz)"
                value={metrics.gpu_sysman.gpu_freq_mhz.toFixed(0)}
                series={freqSeries}
                stroke="var(--ll-accent)"
                fill="rgba(88, 166, 255, 0.12)"
              />
            )}
            {typeof metrics.gpu_sysman.max_temp_c === "number" ? (
              <ChartTile
                label="GPU temp (°C)"
                value={metrics.gpu_sysman.max_temp_c.toFixed(1)}
                series={tempSeries}
                stroke="var(--ll-red)"
                fill="rgba(248, 81, 73, 0.12)"
              />
            ) : (
              <div style={{ padding: 10, color: "var(--ll-text-muted)", fontSize: 12 }}>
                <div style={{ fontWeight: 600, color: "var(--ll-text)" }}>GPU temp</div>
                <div style={{ marginTop: 4 }}>
                  not exposed on this device — integrated graphics typically omit
                  the temperature probe
                </div>
              </div>
            )}
            {powerSeries.length > 0 ? (
              <ChartTile
                label="GPU power (W)"
                value={powerSeries[powerSeries.length - 1].toFixed(2)}
                series={powerSeries}
                stroke="var(--ll-yellow)"
                fill="rgba(210, 153, 34, 0.12)"
              />
            ) : (
              <div style={{ padding: 10, color: "var(--ll-text-muted)", fontSize: 12 }}>
                <div style={{ fontWeight: 600, color: "var(--ll-text)" }}>GPU power</div>
                <div style={{ marginTop: 4 }}>
                  {metrics.gpu_sysman.energy_uj != null
                    ? "waiting for two samples to derive watts…"
                    : "energy counter not exposed on this device"}
                </div>
              </div>
            )}
          </div>
          <div style={{ marginTop: 8, fontSize: 11, color: "var(--ll-text-muted)" }}>
            VRAM + freq update every second. Power is derived from
            consecutive cumulative-energy reads (dE/dt). On Iris Xe this is
            shared LPDDR memory, so VRAM total reflects host RAM pressure.
          </div>
        </section>
      )}

      {metrics && metrics.paged_total_pages > 0 && (
        <section style={{ ...card, marginTop: 16 }}>
          <h3 style={cardHeader}>Paged KV pool</h3>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
            <div>
              <div style={{ fontSize: 12, color: "var(--ll-text-muted)" }}>Total pages</div>
              <div style={{ fontSize: 24, fontWeight: 600, color: "var(--ll-text)" }}>
                {metrics.paged_total_pages.toLocaleString()}
              </div>
              <div style={{ fontSize: 11, color: "var(--ll-text-muted)" }}>
                16 tokens/page · {(metrics.paged_total_pages * 16).toLocaleString()} cache slots
              </div>
            </div>
            <div>
              <div style={{ fontSize: 12, color: "var(--ll-text-muted)" }}>In use</div>
              <div style={{ fontSize: 24, fontWeight: 600, color: "var(--ll-text)" }}>
                {(metrics.paged_total_pages - metrics.paged_free_pages).toLocaleString()}
              </div>
              <div style={{ fontSize: 11, color: "var(--ll-text-muted)" }}>
                {metrics.paged_total_pages > 0
                  ? Math.round(
                      ((metrics.paged_total_pages - metrics.paged_free_pages) /
                        metrics.paged_total_pages) *
                        100,
                    )
                  : 0}
                % occupied
              </div>
            </div>
            <div>
              <div style={{ fontSize: 12, color: "var(--ll-text-muted)" }}>Active slots</div>
              <div style={{ fontSize: 24, fontWeight: 600, color: "var(--ll-text)" }}>
                {metrics.paged_active_slots}
              </div>
              <div style={{ fontSize: 11, color: "var(--ll-text-muted)" }}>
                {metrics.paged_active_slots === 0
                  ? "idle / single-flight"
                  : "fused-decode CB"}
              </div>
            </div>
          </div>
          <div
            style={{
              marginTop: 12,
              height: 8,
              background: "var(--ll-bg-elev)",
              borderRadius: 4,
              overflow: "hidden",
              border: "1px solid var(--ll-border-strong)",
            }}
            title={`${metrics.paged_total_pages - metrics.paged_free_pages} of ${metrics.paged_total_pages} pages in use`}
          >
            <div
              style={{
                height: "100%",
                width: `${
                  metrics.paged_total_pages > 0
                    ? ((metrics.paged_total_pages - metrics.paged_free_pages) /
                        metrics.paged_total_pages) *
                      100
                    : 0
                }%`,
                background: "var(--ll-accent)",
                transition: "width 200ms",
              }}
            />
          </div>
          <div style={{ marginTop: 8, fontSize: 12, color: "var(--ll-text-muted)" }}>
            Shared paged KV pool. New requests allocate pages on prefill;
            completion releases them. When occupancy hits 100%, new admissions
            queue in the scheduler.
          </div>
        </section>
      )}

      {metrics?.cumulative && metrics.cumulative.total_requests > 0 && (
        <section style={{ ...card, marginTop: 16 }}>
          <h3 style={cardHeader}>Cumulative (since server start)</h3>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
            <div>
              <div style={{ fontSize: 12, color: "var(--ll-text-muted)" }}>Requests</div>
              <div style={{ fontSize: 24, fontWeight: 600, color: "var(--ll-text)" }}>
                {metrics.cumulative.total_requests.toLocaleString()}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 12, color: "var(--ll-text-muted)" }}>Tokens generated</div>
              <div style={{ fontSize: 24, fontWeight: 600, color: "var(--ll-text)" }}>
                {metrics.cumulative.total_tokens_generated.toLocaleString()}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 12, color: "var(--ll-text-muted)" }}>Prompt tokens prefilled</div>
              <div style={{ fontSize: 24, fontWeight: 600, color: "var(--ll-text)" }}>
                {metrics.cumulative.total_tokens_prefilled.toLocaleString()}
              </div>
            </div>
            <div>
              <div style={{ fontSize: 12, color: "var(--ll-text-muted)" }}>
                Prefix-cache hits
                <span
                  style={{ marginLeft: 6, fontSize: 11, color: "var(--ll-text-muted)" }}
                  title="Prompt tokens that were served from the live longest-common-prefix cache + multi-snapshot pool + extended-LCP. Higher = the prefix cache is doing more work."
                >
                  ⓘ
                </span>
              </div>
              <div style={{ fontSize: 24, fontWeight: 600, color: "var(--ll-text)" }}>
                {metrics.cumulative.total_cache_hit_tokens.toLocaleString()}
                <span style={{ marginLeft: 8, fontSize: 13, color: "var(--ll-text-muted)" }}>
                  ({(metrics.cumulative.prefix_cache_hit_rate * 100).toFixed(1)}%)
                </span>
              </div>
            </div>
            {(metrics.cumulative.total_spec_rounds ?? 0) > 0 && (
              <div>
                <div style={{ fontSize: 12, color: "var(--ll-text-muted)" }}>
                  Speculation accepted
                  <span
                    style={{ marginLeft: 6, fontSize: 11, color: "var(--ll-text-muted)" }}
                    title="Draft tokens the target model accepted, out of all draft tokens proposed (n-gram + draft-model speculation). Higher = speculative decoding is paying off."
                  >
                    ⓘ
                  </span>
                </div>
                <div style={{ fontSize: 24, fontWeight: 600, color: "var(--ll-text)" }}>
                  {(metrics.cumulative.total_spec_accepted ?? 0).toLocaleString()}
                  <span style={{ marginLeft: 8, fontSize: 13, color: "var(--ll-text-muted)" }}>
                    / {(metrics.cumulative.total_spec_drafted ?? 0).toLocaleString()}{" "}
                    ({((metrics.cumulative.spec_acceptance_rate ?? 0) * 100).toFixed(1)}%)
                  </span>
                </div>
              </div>
            )}
          </div>
          {/* Visual: green bar = hit-rate fraction. Lets operators
              eyeball "is the prefix cache paying off?" without
              parsing a percent. */}
          <div style={{ marginTop: 12 }}>
            <div
              style={{
                height: 6,
                background: "var(--ll-border)",
                borderRadius: 3,
                overflow: "hidden",
              }}
              title={`${(metrics.cumulative.prefix_cache_hit_rate * 100).toFixed(1)}% of prompt tokens served from cache`}
            >
              <div
                style={{
                  height: "100%",
                  width: `${metrics.cumulative.prefix_cache_hit_rate * 100}%`,
                  background: "var(--ll-green)",
                  transition: "width 200ms",
                }}
              />
            </div>
          </div>
        </section>
      )}

      <section style={{ ...card, marginTop: 16 }}>
        <h3 style={cardHeader}>API surface</h3>
        <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13, color: "var(--ll-text)" }}>
          {[
            ["POST /v1/chat/completions", "OpenAI chat (stream + non-stream)"],
            ["POST /v1/completions", "OpenAI completions / FIM"],
            ["POST /v1/messages", "Anthropic Messages API"],
            ["GET  /v1/models", "list loaded models"],
            ["POST /api/chat", "Ollama chat"],
            ["POST /api/generate", "Ollama generate"],
            ["POST /api/show", "Ollama model info"],
            ["GET  /api/tags", "Ollama cached + loaded models"],
            ["POST /api/pull", "HF download via Ollama shape"],
            ["POST /v1/cancel", "cancel an in-flight request"],
            ["GET  /healthz", "liveness + drain state"],
          ].map(([p, d]) => (
            <li key={p} style={{ margin: "4px 0" }}>
              <code style={code}>{p}</code>
              <span style={{ color: "var(--ll-text-muted)", marginLeft: 8 }}>{d}</span>
            </li>
          ))}
        </ul>
      </section>

      {recs && recs.untuned_count > 0 && (
        <section
          style={{
            ...card,
            marginTop: 16,
            background: "rgba(210, 153, 34, 0.08)",
            borderColor: "rgba(210, 153, 34, 0.5)",
          }}
        >
          <h3 style={{ ...cardHeader, color: "var(--ll-yellow)" }}>
            {recs.untuned_count} kernel{recs.untuned_count === 1 ? "" : "s"} untuned
          </h3>
          <p style={{ fontSize: 12, color: "var(--ll-text)", margin: "4px 0 8px 0", lineHeight: 1.5 }}>
            The engine dispatched these matvec shapes without a cached LWS
            (work-group size) entry. Running with the kernel default works,
            but typically leaves 1.5–3× on the table. Use the buttons below
            to populate the cache.
          </p>
          <details style={{ fontSize: 12, color: "var(--ll-text-muted)" }}>
            <summary style={{ cursor: "pointer" }}>
              Show shapes ({recs.shapes.length})
            </summary>
            <ul style={{ margin: "6px 0 0 0", paddingLeft: 16 }}>
              {recs.shapes.map((s, i) => (
                <li key={i}>
                  <code style={code}>
                    {s.kernel} M={s.m} K={s.k}
                  </code>
                </li>
              ))}
            </ul>
          </details>
        </section>
      )}

      <section style={{ ...card, marginTop: 16 }}>
        <h3 style={cardHeader}>Tuner cache</h3>
        {tuningErr && (
          <div style={{ color: "var(--ll-red)", fontSize: 12, marginBottom: 8 }}>
            failed to read: {tuningErr}
          </div>
        )}
        {!tuning ? (
          <div style={{ color: "var(--ll-text-muted)", fontSize: 13 }}>loading…</div>
        ) : (
          <>
            {tuning.device === null ? (
              <p style={{ fontSize: 12, color: "var(--ll-text-muted)", margin: 0, lineHeight: 1.5 }}>
                No SYCL device visible — the tuner cache is keyed by device,
                so there is nothing to surface. Install an Intel GPU + the
                oneAPI runtime to enable device tuning.
              </p>
            ) : (
              <>
                <div style={{ display: "grid", gridTemplateColumns: "150px 1fr", rowGap: 4, fontSize: 13 }}>
                  <span style={{ color: "var(--ll-text-muted)" }}>Device</span>
                  <code style={code}>{tuning.device.name}</code>
                  <span style={{ color: "var(--ll-text-muted)" }}>Driver</span>
                  <code style={code}>{tuning.device.driver_ver}</code>
                  <span style={{ color: "var(--ll-text-muted)" }}>VRAM</span>
                  <span>{tuning.device.vram_mb} MiB</span>
                  <span style={{ color: "var(--ll-text-muted)" }}>Cache path</span>
                  <code style={{ ...code, fontSize: 11 }}>{tuning.cache_path ?? "—"}</code>
                  <span style={{ color: "var(--ll-text-muted)" }}>Cache present</span>
                  <span>
                    {tuning.cache_present ? (
                      <span style={badgeOk}>yes</span>
                    ) : (
                      <span style={badgeWarn}>none — run rustllama tune</span>
                    )}
                  </span>
                  <span style={{ color: "var(--ll-text-muted)" }}>Last tuned</span>
                  <span>{fmtLastTuned(tuning.last_tuned)}</span>
                  <span style={{ color: "var(--ll-text-muted)" }}>Tuned kernel shapes</span>
                  <span>
                    {tuning.kernel_entry_count}
                    {tuning.kernel_entry_count > 0 && (
                      <>
                        {" · "}
                        <a
                          href="#"
                          onClick={(e) => {
                            e.preventDefault();
                            setShowKernels((v) => !v);
                          }}
                          style={{ color: "var(--ll-accent)", fontSize: 11 }}
                        >
                          {showKernels ? "hide" : "show"}
                        </a>
                      </>
                    )}
                  </span>
                  <span style={{ color: "var(--ll-text-muted)" }}>Batch-size winner</span>
                  <span>
                    {tuning.batch_size !== null ? (
                      <>
                        <code style={code}>{tuning.batch_size}</code>{" "}
                        {tuning.auto_apply_batch_size ? (
                          <span style={badgeOk}>auto-applied</span>
                        ) : (
                          <span style={badgeMuted}>stored, not applied</span>
                        )}
                      </>
                    ) : (
                      <span style={{ color: "var(--ll-text-muted)" }}>—</span>
                    )}
                  </span>
                  <span style={{ color: "var(--ll-text-muted)" }}>KV-dtype winner</span>
                  <span>
                    {tuning.kv_dtype !== null ? (
                      <>
                        <code style={code}>{tuning.kv_dtype}</code>{" "}
                        {tuning.auto_apply_kv_dtype ? (
                          <span style={badgeOk}>auto-applied</span>
                        ) : (
                          <span style={badgeMuted}>stored, not applied</span>
                        )}
                      </>
                    ) : (
                      <span style={{ color: "var(--ll-text-muted)" }}>—</span>
                    )}
                  </span>
                  <span style={{ color: "var(--ll-text-muted)" }}>Flash-attention winner</span>
                  <span>
                    {tuning.flash_attention !== null ? (
                      <>
                        <code style={code}>{tuning.flash_attention ? "on" : "off"}</code>{" "}
                        {tuning.auto_apply_flash_attention ? (
                          <span style={badgeOk}>auto-applied</span>
                        ) : (
                          <span style={badgeMuted}>stored, not applied</span>
                        )}
                      </>
                    ) : (
                      <span style={{ color: "var(--ll-text-muted)" }}>—</span>
                    )}
                  </span>
                  <span style={{ color: "var(--ll-text-muted)" }}>KV-layout winner</span>
                  <span>
                    {tuning.kv_cache_layout !== null ? (
                      <>
                        <code style={code}>{tuning.kv_cache_layout}</code>{" "}
                        {tuning.auto_apply_kv_cache_layout ? (
                          <span style={badgeOk}>auto-applied</span>
                        ) : (
                          <span style={badgeMuted}>stored, not applied</span>
                        )}
                      </>
                    ) : (
                      <span style={{ color: "var(--ll-text-muted)" }}>—</span>
                    )}
                  </span>
                </div>

                <div
                  style={{
                    marginTop: 12,
                    paddingTop: 12,
                    borderTop: "1px solid var(--ll-border)",
                  }}
                >
                  <div
                    style={{
                      fontSize: 11,
                      color: "var(--ll-text-muted)",
                      textTransform: "uppercase",
                      letterSpacing: 0.3,
                      marginBottom: 6,
                    }}
                  >
                    Run tune
                  </div>
                  {tuneCachedModels.length === 0 ? (
                    <div style={{ color: "var(--ll-text-muted)", fontSize: 12 }}>
                      No cached models. Pull one from the Models page first.
                    </div>
                  ) : (
                    <>
                      <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                        <select
                          value={tuneSelectedModel}
                          onChange={(e) => setTuneSelectedModel(e.target.value)}
                          disabled={tuneRunning !== null}
                          style={{
                            background: "var(--ll-bg)",
                            color: "var(--ll-text)",
                            border: "1px solid var(--ll-border-strong)",
                            borderRadius: 4,
                            padding: "4px 8px",
                            fontSize: 12,
                            fontFamily: "ui-monospace, monospace",
                          }}
                        >
                          {tuneCachedModels.map((m) => (
                            <option key={m.name} value={m.name}>
                              {m.name}
                            </option>
                          ))}
                        </select>
                        <button
                          onClick={() => startTune("model")}
                          disabled={tuneRunning !== null || !tuneSelectedModel}
                          style={
                            tuneRunning !== null || !tuneSelectedModel
                              ? btnPrimaryDisabled
                              : btnPrimary
                          }
                          title="Full per-model sweep: KV-dtype coherence, CPU/GPU dispatch, kernels, batch size — persisted to the tuner cache"
                        >
                          {tuneRunning === "model"
                            ? `tuning… ${Math.floor((Date.now() - tuneRunStart) / 1000)}s`
                            : "Tune model"}
                        </button>
                        <button
                          onClick={() => startTune("cpu_gpu")}
                          disabled={tuneRunning !== null || !tuneSelectedModel}
                          style={
                            tuneRunning !== null || !tuneSelectedModel
                              ? btnPrimaryDisabled
                              : btnPrimary
                          }
                          title="Measure CPU vs GPU dispatch (n_gpu_layers) and persist the faster placement"
                        >
                          {tuneRunning === "cpu_gpu"
                            ? `tuning… ${Math.floor((Date.now() - tuneRunStart) / 1000)}s`
                            : "Tune CPU & GPUs"}
                        </button>
                        <button
                          onClick={() => startTuneAll()}
                          disabled={tuneRunning !== null || tuneCachedModels.length === 0}
                          style={
                            tuneRunning !== null || tuneCachedModels.length === 0
                              ? btnPrimaryDisabled
                              : btnPrimary
                          }
                          title="Run the full per-model sweep for every cached model, one after another"
                        >
                          {tuneRunning === "all_models"
                            ? `tuning ${tuneAllIdx ? `${tuneAllIdx.i}/${tuneAllIdx.n}` : ""}… ${Math.floor((Date.now() - tuneRunStart) / 1000)}s`
                            : "Tune all models"}
                        </button>
                      </div>
                      <div style={{ fontSize: 11, color: "var(--ll-text-muted)", marginTop: 6 }}>
                        Runs in a background process (safe on large models) and
                        shows a progress window. Closing this page does NOT
                        cancel the in-flight sweep on the server.
                      </div>
                      {tuneStatus && (
                        <div
                          style={{
                            marginTop: 8,
                            padding: "6px 10px",
                            background: tuneStatus.startsWith("error")
                              ? "rgba(248, 81, 73, 0.08)"
                              : "rgba(63, 185, 80, 0.08)",
                            border: tuneStatus.startsWith("error")
                              ? "1px solid rgba(248, 81, 73, 0.4)"
                              : "1px solid rgba(63, 185, 80, 0.4)",
                            borderRadius: 4,
                            color: tuneStatus.startsWith("error")
                              ? "var(--ll-red)"
                              : "var(--ll-green)",
                            fontSize: 12,
                            fontFamily: "ui-monospace, monospace",
                          }}
                        >
                          {tuneStatus}
                        </div>
                      )}
                      <TuneProgressModal
                        open={tuneModalOpen}
                        subtitle={
                          tuneAllIdx
                            ? `model ${tuneAllIdx.i} of ${tuneAllIdx.n}`
                            : tuneSelectedModel
                        }
                      />
                    </>
                  )}
                </div>

                <div style={{ marginTop: 12 }}>
                  <div style={{ fontSize: 11, color: "var(--ll-text-muted)", textTransform: "uppercase", letterSpacing: 0.3, marginBottom: 4 }}>
                    Placement winners ({tuning.placement.length}){" "}
                    {tuning.auto_apply_placement ? (
                      <span style={badgeOk}>auto-applied</span>
                    ) : (
                      <span style={badgeMuted}>stored, not applied</span>
                    )}
                  </div>
                  {tuning.placement.length === 0 ? (
                    <div style={{ color: "var(--ll-text-muted)", fontSize: 12 }}>
                      None — run <code style={code}>rustllama tune --placement --measure</code> for a model to populate.
                    </div>
                  ) : (
                    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                      <thead>
                        <tr style={{ color: "var(--ll-text-muted)", textAlign: "left" }}>
                          <th style={{ padding: "4px 6px", fontWeight: 500 }}>Model</th>
                          <th style={{ padding: "4px 6px", fontWeight: 500 }}>n_gpu_layers</th>
                        </tr>
                      </thead>
                      <tbody>
                        {tuning.placement.map((p) => (
                          <tr key={p.model_key} style={{ borderTop: "1px solid var(--ll-border)" }}>
                            <td style={{ padding: "4px 6px" }}>
                              <code style={code}>{p.model_key}</code>
                            </td>
                            <td style={{ padding: "4px 6px" }}>{p.n_gpu_layers}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                </div>
                {showKernels && tuning.kernels.length > 0 && (
                  <div style={{ marginTop: 12 }}>
                    <div style={{ fontSize: 11, color: "var(--ll-text-muted)", textTransform: "uppercase", letterSpacing: 0.3, marginBottom: 4 }}>
                      Kernel-LWS entries ({tuning.kernels.length})
                    </div>
                    <div style={{ maxHeight: 240, overflowY: "auto", border: "1px solid var(--ll-border)", borderRadius: 4 }}>
                      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                        <thead style={{ position: "sticky", top: 0, background: "var(--ll-bg-elev)" }}>
                          <tr style={{ color: "var(--ll-text-muted)", textAlign: "left" }}>
                            <th style={{ padding: "4px 6px", fontWeight: 500 }}>Key</th>
                            <th style={{ padding: "4px 6px", fontWeight: 500 }}>Kernel</th>
                            <th style={{ padding: "4px 6px", fontWeight: 500 }}>Params</th>
                          </tr>
                        </thead>
                        <tbody>
                          {tuning.kernels.map((k) => (
                            <tr key={k.key} style={{ borderTop: "1px solid var(--ll-border)" }}>
                              <td style={{ padding: "4px 6px" }}>
                                <code style={{ ...code, fontSize: 10 }}>{k.key}</code>
                              </td>
                              <td style={{ padding: "4px 6px" }}>
                                <code style={{ ...code, fontSize: 10 }}>{k.kernel}</code>
                              </td>
                              <td style={{ padding: "4px 6px" }}>
                                <code style={{ ...code, fontSize: 10 }}>
                                  {JSON.stringify(k.params)}
                                </code>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </>
            )}
          </>
        )}
      </section>

      <section style={{ ...card, marginTop: 16 }}>
        <h3 style={cardHeader}>Loaded models ({models.length})</h3>
        {models.length === 0 && (
          <div style={{ color: "var(--ll-text-muted)", fontSize: 13 }}>None — load via Models page.</div>
        )}
        {models.map((m) => (
          <div key={m.id} style={{ padding: "6px 0", borderTop: "1px solid var(--ll-border)", fontSize: 13 }}>
            <code style={code}>{m.id}</code>
          </div>
        ))}
      </section>

      <section style={{ ...card, marginTop: 16 }}>
        <h3 style={cardHeader}>Throughput probe</h3>
        <p style={{ fontSize: 12, color: "var(--ll-text-muted)", margin: "0 0 12px 0" }}>
          Fires one short chat request and reports tokens / second from the
          server's <code style={code}>usage.decode_ms</code> field. "This
          probe" is the raw single-request number; "Smoothed (EMA)" is the
          exponential moving average across the last several requests
          (alpha = 0.2, so each new probe contributes ~20%).
        </p>
        <button onClick={runProbe} disabled={probing} style={btnPrimary}>
          {probing ? "running…" : "Run probe"}
        </button>
        {probe && probe.ok && (
          <div style={{ marginTop: 12, fontSize: 13, color: "var(--ll-text)" }}>
            <KV k="Tokens emitted" v={probe.tokens} />
            <KV k="Decode time" v={probe.decode_ms ? `${probe.decode_ms.toFixed(1)} ms` : "—"} />
            <KV k="Prefill time" v={probe.prefill_ms ? `${probe.prefill_ms.toFixed(1)} ms` : "—"} />
            <KV k="Cache hits" v={probe.cache_hit_tokens ?? 0} />
            <KV k="This probe" v={probeTps ? `${probeTps} tok/s` : "—"} />
            <KV
              k="Smoothed (EMA)"
              v={
                metrics?.ema_tok_s
                  ? `${metrics.ema_tok_s.toFixed(1)} tok/s`
                  : "—"
              }
            />
          </div>
        )}
        {probe && !probe.ok && <div style={errBox}>{probe.err}</div>}
      </section>
    </div>
  );
}

function KV({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div style={{ display: "flex", padding: "6px 0", borderTop: "1px solid var(--ll-border)", fontSize: 13 }}>
      <span style={{ width: 160, color: "var(--ll-text-muted)" }}>{k}</span>
      <code style={{ ...code, background: "transparent", padding: 0 }}>{v}</code>
    </div>
  );
}

const badgeBase: React.CSSProperties = {
  display: "inline-block",
  padding: "1px 6px",
  borderRadius: 8,
  fontSize: 10,
  fontWeight: 600,
  letterSpacing: 0.3,
  textTransform: "uppercase",
  marginLeft: 6,
};
const badgeOk: React.CSSProperties = {
  ...badgeBase,
  background: "rgba(63, 185, 80, 0.15)",
  color: "var(--ll-green)",
  border: "1px solid rgba(63, 185, 80, 0.5)",
};
const badgeWarn: React.CSSProperties = {
  ...badgeBase,
  background: "rgba(210, 153, 34, 0.15)",
  color: "var(--ll-yellow)",
  border: "1px solid rgba(210, 153, 34, 0.5)",
};
const badgeMuted: React.CSSProperties = {
  ...badgeBase,
  background: "transparent",
  color: "var(--ll-text-muted)",
  border: "1px solid var(--ll-border-strong)",
};
const card: React.CSSProperties = {
  background: "var(--ll-bg-elev)",
  border: "1px solid var(--ll-border)",
  borderRadius: 6,
  padding: 16,
};
const cardHeader: React.CSSProperties = { margin: "0 0 8px 0", fontSize: 14, fontWeight: 600 };
const code: React.CSSProperties = {
  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
  background: "var(--ll-bg)",
  padding: "1px 6px",
  borderRadius: 3,
  fontSize: 12,
};
// Slightly tighter monospace badge used for inline values (KV dtype,
// concurrency level) on the Live metrics card.
const inlineCode: React.CSSProperties = {
  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
  background: "var(--ll-bg)",
  padding: "0 4px",
  borderRadius: 3,
  fontSize: 11,
  color: "var(--ll-text)",
};
const btnPrimary: React.CSSProperties = {
  padding: "6px 18px",
  background: "var(--ll-green)",
  color: "white",
  border: "1px solid var(--ll-green)",
  borderRadius: 4,
  cursor: "pointer",
  fontSize: 13,
};
const btnPrimaryDisabled: React.CSSProperties = {
  ...btnPrimary,
  opacity: 0.45,
  cursor: "not-allowed",
};
const errBox: React.CSSProperties = {
  padding: 10,
  background: "var(--ll-red-soft)",
  border: "1px solid var(--ll-red)",
  borderRadius: 4,
  color: "var(--ll-red)",
  fontSize: 13,
};

function trim(arr: number[]): number[] {
  return arr.length > METRICS_WINDOW ? arr.slice(arr.length - METRICS_WINDOW) : arr;
}

function formatUptime(s: number): string {
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

interface ChartTileProps {
  label: string;
  value: string;
  series: number[];
  stroke: string;
  fill: string;
  /// Optional fixed Y-axis ceiling. When unset, the chart auto-scales
  /// to the series's own max so peaks always touch the top of the box.
  /// Use this for series with a known cap (ctx_size, max_pending) to
  /// keep the scale stable across time.
  yMax?: number;
}

/// Inline-SVG sparkline. Avoids pulling in a chart library so the
/// frontend stays buildable without an extra `pnpm install`. The
/// chart is purely presentational: it reads the series, computes a
/// path, and renders it as a stroked line + filled area underneath.
function ChartTile({ label, value, series, stroke, fill, yMax }: ChartTileProps) {
  const w = 160;
  const h = 40;
  const max = Math.max(yMax ?? 0, ...series, 1);
  const n = series.length;
  // Need at least 2 points to draw a line. Otherwise just render
  // the label + value with an empty box.
  const path =
    n >= 2
      ? series
          .map((y, i) => {
            const x = (i / (n - 1)) * w;
            const yy = h - (y / max) * h;
            return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${yy.toFixed(1)}`;
          })
          .join(" ")
      : "";
  const area = path ? `${path} L${w},${h} L0,${h} Z` : "";
  return (
    <div>
      <div style={{ fontSize: 11, color: "var(--ll-text-muted)", marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 18, color: "var(--ll-text)", fontVariantNumeric: "tabular-nums" }}>{value}</div>
      <svg width={w} height={h} style={{ display: "block", marginTop: 4 }}>
        {area && <path d={area} fill={fill} />}
        {path && <path d={path} fill="none" stroke={stroke} strokeWidth={1.5} />}
      </svg>
    </div>
  );
}

// ----- Compute inventory (multi-GPU + CPU tier) ------------------------
//
// One row per compute device: every GPU (from /v1/metrics `gpus[]`) and the
// CPU tier (metrics flat CPU fields + /v1/capabilities `backends.cpu`).
// Purely presentational + defensive — any missing numeric field renders
// "n/a" rather than throwing.

/// GB rendering of a byte count (1 decimal). Guards non-finite input.
function fmtGb(bytes: number): string {
  if (!Number.isFinite(bytes)) return "n/a";
  return (bytes / 1_073_741_824).toFixed(1);
}

/// GPU→backend badge label. Intel/AMD dispatch through the SYCL kernels;
/// NVIDIA through the CUDA kernels. The metrics row already carries the
/// vendor; /v1/capabilities refines the tooltip (see `gpuBackendTitle`).
function gpuBackendBadge(vendor: string): string {
  if (vendor === "nvidia") return "CUDA";
  if (vendor === "intel" || vendor === "amd" || vendor === "gpu") return "SYCL";
  return "GPU";
}

/// Tooltip detail for a GPU's backend badge, sourced from /v1/capabilities
/// so the capability probe is load-bearing: SYCL backend name for Intel,
/// CUDA driver version + kernel-ready count for NVIDIA. Falls back to a
/// plain label when capabilities haven't loaded.
function gpuBackendTitle(vendor: string, b: BackendsSnapshot | null): string {
  if (vendor === "nvidia") {
    if (!b?.cuda) return "CUDA dispatch";
    const drv = b.cuda.driver ? `driver ${b.cuda.driver}` : "driver n/a";
    return `CUDA · ${drv} · ${b.cuda.compute_ready}/${b.cuda.device_count} kernel-ready`;
  }
  if (b?.sycl?.available) return `SYCL · ${b.sycl.backend ?? "backend n/a"}`;
  return "SYCL dispatch";
}

/// Vendor accent color for the device badge (NVIDIA green, Intel blue,
/// AMD red, other purple).
function vendorColor(vendor: string): string {
  switch (vendor) {
    case "nvidia":
      return "var(--ll-green)";
    case "intel":
      return "var(--ll-accent)";
    case "amd":
      return "var(--ll-red)";
    default:
      return "var(--ll-purple)";
  }
}

/// CPU-row sub-line: logical/enabled cores + SIMD + tier flags from the
/// capabilities backend snapshot. Empty string until capabilities load —
/// the row still renders brand + RAM + util from metrics alone.
function cpuSubline(b: BackendsSnapshot | null): string {
  if (!b?.cpu || typeof b.cpu.logical_cores !== "number") return "";
  const parts: string[] = [];
  const logical = b.cpu.logical_cores;
  const enabled = b.cpu.enabled_cores;
  parts.push(
    typeof enabled === "number" && enabled !== logical
      ? `${enabled}/${logical} cores enabled`
      : `${logical} logical cores`,
  );
  if (b.cpu.simd_features && b.cpu.simd_features.length > 0) {
    parts.push(b.cpu.simd_features.join("/"));
  }
  if (b.cpu.cpu_enabled === false) parts.push("tier disabled");
  if (b.cpu.gpu_enabled === false) parts.push("gpu disabled");
  if (b.cpu.vram_only) parts.push("vram-only");
  return parts.join(" · ");
}

interface DeviceRowProps {
  badge: string;
  badgeColor: string;
  badgeTitle?: string;
  label: string;
  name: string;
  sub?: string;
  memLabel: string;
  usedBytes: number | null;
  totalBytes: number | null;
  utilPct: number | null;
}

/// One compute-device row: identity (backend badge + label + name), a
/// used/free/total memory bar, and a utilization readout. Every numeric
/// field is independently optional — missing values render "n/a" and the
/// bar collapses to empty rather than crashing the page.
function DeviceRow(p: DeviceRowProps) {
  const pct =
    p.usedBytes != null && p.totalBytes != null && p.totalBytes > 0
      ? Math.min(100, Math.max(0, (p.usedBytes / p.totalBytes) * 100))
      : null;
  const barColor =
    pct != null && pct >= 90
      ? "var(--ll-red)"
      : pct != null && pct >= 75
        ? "var(--ll-yellow)"
        : "var(--ll-accent)";
  return (
    <div style={deviceRow}>
      {/* identity */}
      <div style={{ minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span
            title={p.badgeTitle}
            style={{ ...deviceBadge, color: p.badgeColor, borderColor: p.badgeColor }}
          >
            {p.badge}
          </span>
          <span style={{ fontSize: 12, fontWeight: 600, color: "var(--ll-text)" }}>
            {p.label}
          </span>
        </div>
        <div
          title={p.name}
          style={{
            fontSize: 12,
            color: "var(--ll-text-muted)",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            marginTop: 2,
          }}
        >
          {p.name}
        </div>
        {p.sub && (
          <div style={{ fontSize: 10, color: "var(--ll-text-faint)", marginTop: 1 }}>{p.sub}</div>
        )}
      </div>
      {/* memory bar */}
      <div style={{ minWidth: 0 }}>
        <div
          style={{
            fontSize: 11,
            color: "var(--ll-text-muted)",
            marginBottom: 3,
            display: "flex",
            justifyContent: "space-between",
            gap: 8,
          }}
        >
          <span>{p.memLabel}</span>
          <span style={{ fontVariantNumeric: "tabular-nums" }}>
            {p.totalBytes != null
              ? p.usedBytes != null
                ? `${fmtGb(p.usedBytes)} / ${fmtGb(p.totalBytes)} GB${
                    pct != null ? ` · ${pct.toFixed(0)}%` : ""
                  }`
                : `${fmtGb(p.totalBytes)} GB total`
              : "n/a"}
          </span>
        </div>
        <div
          style={memTrack}
          title={
            p.usedBytes != null && p.totalBytes != null
              ? `${fmtGb(p.totalBytes - p.usedBytes)} GB free of ${fmtGb(p.totalBytes)} GB`
              : undefined
          }
        >
          {pct != null && <div style={{ ...memFill, width: `${pct}%`, background: barColor }} />}
        </div>
        <div style={{ fontSize: 10, color: "var(--ll-text-faint)", marginTop: 2 }}>
          {p.usedBytes != null && p.totalBytes != null
            ? `${fmtGb(p.totalBytes - p.usedBytes)} GB free`
            : p.totalBytes == null
              ? "capacity unknown"
              : "free unknown"}
        </div>
      </div>
      {/* utilization */}
      <div style={{ textAlign: "right" }}>
        <div style={{ fontSize: 10, color: "var(--ll-text-muted)" }}>util</div>
        <div
          style={{
            fontSize: 15,
            fontWeight: 600,
            color: "var(--ll-text)",
            fontVariantNumeric: "tabular-nums",
          }}
        >
          {p.utilPct != null ? `${p.utilPct.toFixed(0)}%` : "n/a"}
        </div>
      </div>
    </div>
  );
}

const deviceRow: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "minmax(150px, 1.3fr) minmax(160px, 2fr) 64px",
  gap: 12,
  alignItems: "center",
  padding: "8px 10px",
  background: "var(--ll-bg)",
  border: "1px solid var(--ll-border)",
  borderRadius: 6,
};
const deviceBadge: React.CSSProperties = {
  display: "inline-block",
  padding: "0 6px",
  borderRadius: 4,
  fontSize: 9,
  fontWeight: 700,
  letterSpacing: 0.4,
  textTransform: "uppercase",
  border: "1px solid",
  lineHeight: "15px",
  background: "transparent",
};
const memTrack: React.CSSProperties = {
  height: 8,
  background: "var(--ll-bg-elev)",
  border: "1px solid var(--ll-border-strong)",
  borderRadius: 4,
  overflow: "hidden",
};
const memFill: React.CSSProperties = {
  height: "100%",
  background: "var(--ll-accent)",
  transition: "width 300ms",
};
