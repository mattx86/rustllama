// Models page: full lifecycle for cached + loaded models.
//
//   - Top section: currently loaded models (from `/v1/models`).
//     Shows which is the default; lets the user promote a different
//     model with one click, or unload (drops it from the warm pool).
//
//   - Middle section: pull form (HuggingFace ref → streamed progress).
//
//   - Bottom section: cached models on disk (`/api/tags`). Each row
//     has Load + Delete actions. Load is the path that promotes a
//     cached GGUF into the warm registry; Delete removes the file
//     entirely.
//
// All actions optimistically reload both lists on success.

import { useCallback, useEffect, useState } from "react";
import {
  deleteModel,
  getMetrics,
  inspectGguf,
  listOllamaTags,
  hfFiles,
  hfSearch,
  loadModel,
  pullModel,
  retuneModel,
  setDefaultModel,
  unloadModel,
  type GgufInspectResult,
  type HfFile,
  type HfModel,
  type ModelInfo,
  type TagsModel,
} from "../api";
import TuneProgressModal from "../TuneProgressModal";

/// Curated set of known-to-work GGUFs for users who want a one-click
/// starting point. Each entry includes the HuggingFace ref, an
/// approximate disk size, and the minimum free-RAM we'd recommend
/// before pulling. The list is intentionally short — the goal is
/// "pick one and go", not a comprehensive directory.
///
/// Bytes are approximations from the published GGUFs; the actual
/// download size may differ by a few percent. RAM hints assume the
/// default `keep_quant_raw = false` (some F16 dequant overhead).
interface SuggestedModel {
  label: string;
  hubRef: string;
  diskBytes: number;
  ramHintBytes: number;
  family: string;
  use: string;
}

const SUGGESTED_MODELS: SuggestedModel[] = [
  {
    label: "Qwen2.5-Coder 0.5B Q4_K_M",
    hubRef:
      "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF:qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
    diskBytes: 398 * 1_048_576,
    ramHintBytes: 1 * 1_073_741_824,
    family: "qwen2",
    use: "Smallest known-working coder. Quick test of the engine — fits anywhere.",
  },
  {
    label: "Llama-3.2 1B Instruct Q8_0",
    hubRef:
      "bartowski/Llama-3.2-1B-Instruct-GGUF:Llama-3.2-1B-Instruct-Q8_0.gguf",
    diskBytes: 1320 * 1_048_576,
    ramHintBytes: 3 * 1_073_741_824,
    family: "llama",
    use: "General chat. Conservative quant — full Q8 precision.",
  },
  {
    label: "Qwen2.5-Coder 1.5B Q4_K_M",
    hubRef:
      "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF:qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
    diskBytes: 1020 * 1_048_576,
    ramHintBytes: 3 * 1_073_741_824,
    family: "qwen2",
    use: "Better coder than the 0.5B, still fits comfortably.",
  },
  {
    label: "Qwen2.5-Coder 7B Q4_K_M",
    hubRef:
      "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:qwen2.5-coder-7b-instruct-q4_k_m.gguf",
    diskBytes: 4680 * 1_048_576,
    ramHintBytes: 8 * 1_073_741_824,
    family: "qwen2",
    use: "The mainline coding-LLM target. Needs ~8 GB free RAM.",
  },
  {
    label: "Mistral-7B-Instruct v0.2 Q4_K_M",
    hubRef:
      "TheBloke/Mistral-7B-Instruct-v0.2-GGUF:mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    diskBytes: 4370 * 1_048_576,
    ramHintBytes: 7 * 1_073_741_824,
    family: "mistral",
    use: "Classic Mistral 7B. Strong all-rounder.",
  },
];

function humanBytes(n: number): string {
  if (n === 0) return "—";
  const u = ["B", "KiB", "MiB", "GiB", "TiB"];
  let i = 0;
  let v = n;
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v < 10 ? 2 : 1)} ${u[i]}`;
}

export default function ModelsPage() {
  const [loaded, setLoaded] = useState<(ModelInfo & { is_default?: boolean })[]>([]);
  const [cached, setCached] = useState<TagsModel[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [pullRef, setPullRef] = useState("");
  const [pullStatus, setPullStatus] = useState<string | null>(null);
  const [pulling, setPulling] = useState(false);
  // Download progress bar: 0-100 while a pull streams byte counts, else null.
  const [pullPct, setPullPct] = useState<number | null>(null);
  // Realtime HuggingFace search: query → repo results → per-repo GGUF files.
  const [hfQuery, setHfQuery] = useState("");
  const [hfResults, setHfResults] = useState<HfModel[]>([]);
  const [hfSearching, setHfSearching] = useState(false);
  const [hfRepo, setHfRepo] = useState<string | null>(null);
  const [hfRepoFiles, setHfRepoFiles] = useState<HfFile[]>([]);
  const [hfFilesLoading, setHfFilesLoading] = useState(false);
  const [hfErr, setHfErr] = useState<string | null>(null);
  // Live RAM info — drives the "fits / tight / too big" annotation on
  // the suggested-models card.
  const [ramAvail, setRamAvail] = useState<number | null>(null);
  const [ramTotal, setRamTotal] = useState<number | null>(null);
  /// Drag-and-drop state. `true` while the OS reports a file is
  /// being dragged over the window; flips back to `false` on drop /
  /// leave. Renders a full-page overlay so the user can see they're
  /// in drop-target mode.
  const [dragHover, setDragHover] = useState(false);
  /// Status line for the most recent drop: "loading <name>...",
  /// "loaded as <model_id>", or "error: ...". Cleared on reload.
  const [dropStatus, setDropStatus] = useState<string | null>(null);
  /// Inspector state. `inspectFor` is the cached-model name being
  /// shown; the modal is open iff it's non-null. Result is the JSON
  /// from `/api/gguf/inspect`. `inspectIncludeTensors` controls
  /// whether the per-tensor list is fetched — off by default since a
  /// 7B model has ~300 tensors and they're not always interesting.
  const [inspectFor, setInspectFor] = useState<string | null>(null);
  const [inspectData, setInspectData] = useState<GgufInspectResult | null>(null);
  const [inspectBusy, setInspectBusy] = useState(false);
  const [inspectErr, setInspectErr] = useState<string | null>(null);
  const [inspectIncludeTensors, setInspectIncludeTensors] = useState(false);
  /// Load / auto-tune progress modal. Open while a load or re-tune is in
  /// flight; the modal polls /v1/tune/progress and renders the sweep bar
  /// when a first-load (or forced) tune is running.
  const [progressOpen, setProgressOpen] = useState(false);
  const [progressModel, setProgressModel] = useState("");

  useEffect(() => {
    let cancel = false;
    const tick = async () => {
      try {
        const m = await getMetrics();
        if (cancel) return;
        setRamAvail(m.ram_available_bytes);
        setRamTotal(m.ram_total_bytes);
      } catch {
        /* keep prior values */
      }
    };
    tick();
    const id = setInterval(tick, 5000);
    return () => {
      cancel = true;
      clearInterval(id);
    };
  }, []);

  const reload = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const [loadedNow, cachedNow] = await Promise.all([
        // /v1/models returns an OpenAI-shaped list; the rustllama
        // extension fields include `is_default` per entry.
        fetch("http://127.0.0.1:11434/v1/models").then(async (r) => {
          if (!r.ok) throw new Error(`/v1/models ${r.status}`);
          const body = await r.json();
          return (body.data ?? []) as (ModelInfo & { is_default?: boolean })[];
        }),
        listOllamaTags(),
      ]);
      setLoaded(loadedNow);
      setCached(cachedNow);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    reload();
  }, [reload]);

  // Drag-and-drop GGUF support. Tauri 2's webview drag-drop event
  // delivers OS-absolute file paths (not File objects), which is
  // exactly what `/v1/models/load` accepts via the `path` field.
  // Browser dev-mode (vite without Tauri) doesn't expose
  // `getCurrentWebview`; the dynamic import + try/catch keeps the
  // page functional in both contexts (vite dev silently no-ops).
  useEffect(() => {
    let unlisten: (() => void) | null = null;
    let cancelled = false;
    (async () => {
      try {
        const mod = await import("@tauri-apps/api/webview");
        if (cancelled) return;
        const wv = mod.getCurrentWebview();
        const u = await wv.onDragDropEvent((event) => {
          const t = event.payload.type;
          if (t === "over") {
            setDragHover(true);
            return;
          }
          if (t === "leave") {
            setDragHover(false);
            return;
          }
          if (t === "drop") {
            setDragHover(false);
            const paths = (event.payload as { paths?: string[] }).paths ?? [];
            const ggufs = paths.filter((p) =>
              p.toLowerCase().endsWith(".gguf"),
            );
            if (paths.length === 0) {
              return;
            }
            if (ggufs.length === 0) {
              setDropStatus(
                `error: dropped ${paths.length} file(s) but none had a .gguf extension`,
              );
              return;
            }
            (async () => {
              for (const p of ggufs) {
                const name = p.split(/[\\/]/).pop() ?? p;
                setDropStatus(`loading ${name}…`);
                try {
                  const res = await loadModel({ path: p });
                  setDropStatus(`loaded ${res.model_id}`);
                  await reload();
                } catch (e) {
                  setDropStatus(
                    `error loading ${name}: ${
                      e instanceof Error ? e.message : String(e)
                    }`,
                  );
                  // Don't abort the loop — try the next file even
                  // if one fails. A user dropping a folder of
                  // GGUFs shouldn't lose the rest on one bad header.
                }
              }
            })();
          }
        });
        if (cancelled) {
          u();
          return;
        }
        unlisten = u;
      } catch {
        // Not running in a Tauri webview (vite dev, or Tauri API
        // unavailable). Drop support is graceful no-op here — the
        // existing Pull / cached-model affordances still work.
      }
    })();
    return () => {
      cancelled = true;
      if (unlisten) {
        unlisten();
      }
    };
  }, [reload]);

  const doPull = async (explicitRef?: string) => {
    const target = (explicitRef ?? pullRef).trim();
    if (!target || pulling) return;
    if (!target.includes("/")) {
      setPullStatus("error: expected HuggingFace ref `owner/repo:filename`");
      return;
    }
    if (explicitRef) setPullRef(explicitRef);
    setPulling(true);
    setPullPct(null);
    setPullStatus("starting…");
    try {
      await pullModel(target, (p) => {
        if (p.total && p.total > 0 && p.completed != null) {
          const pct = Math.min(100, (p.completed / p.total) * 100);
          setPullPct(pct);
          setPullStatus(
            `${p.status} — ${pct.toFixed(0)}% (${humanBytes(p.completed)} / ${humanBytes(p.total)})`,
          );
        } else {
          setPullStatus(p.status);
        }
      });
      setPullStatus("done");
      reload();
    } catch (e) {
      setPullStatus(`error: ${e instanceof Error ? e.message : e}`);
    } finally {
      setPulling(false);
      // Remove the progress bar once the download finishes (the status
      // line keeps the "done"/error text).
      setPullPct(null);
    }
  };

  // Debounced realtime HF search — fires 300ms after the user stops typing.
  useEffect(() => {
    const q = hfQuery.trim();
    if (q.length < 2) {
      setHfResults([]);
      setHfSearching(false);
      return;
    }
    setHfSearching(true);
    const id = window.setTimeout(async () => {
      try {
        const rows = await hfSearch(q, 20);
        setHfResults(rows);
        setHfErr(null);
      } catch (e) {
        setHfErr(e instanceof Error ? e.message : String(e));
        setHfResults([]);
      } finally {
        setHfSearching(false);
      }
    }, 300);
    return () => window.clearTimeout(id);
  }, [hfQuery]);

  // Expand a searched repo into its concrete .gguf files.
  const pickRepo = async (repo: string) => {
    setHfRepo(repo);
    setHfRepoFiles([]);
    setHfFilesLoading(true);
    try {
      setHfRepoFiles(await hfFiles(repo));
      setHfErr(null);
    } catch (e) {
      setHfErr(e instanceof Error ? e.message : String(e));
    } finally {
      setHfFilesLoading(false);
    }
  };

  // Choose a specific file → stage it in the pull field, collapse search.
  const pickFile = (repo: string, rfilename: string) => {
    setPullRef(`${repo}:${rfilename}`);
    setHfRepo(null);
    setHfResults([]);
    setHfQuery("");
  };

  const doLoad = async (name: string) => {
    setBusyId(name);
    setProgressModel(name);
    setProgressOpen(true);
    try {
      // `/api/tags` returns file stems (e.g. "Qwen2.5-Coder-0.5B")
      // for cached models, which need server-side resolution. The
      // HuggingFace ref shape (`owner/repo:filename.gguf`) goes
      // straight through the hub path; everything else uses the
      // `name` field that walks the cache for a matching stem.
      // On a first-ever load the server runs the tuning sweep before
      // returning — the modal shows its progress.
      const isHub = name.includes("/") && name.includes(":");
      await loadModel(isHub ? { hub: name } : { name });
      await reload();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyId(null);
      setProgressOpen(false);
    }
  };

  /// Force a full re-tune of an already-known model, then reload it so the
  /// fresh winners (kv_dtype coherence, CPU/GPU dispatch, etc.) apply.
  const doRetune = async (modelId: string) => {
    setBusyId(modelId);
    setProgressModel(modelId);
    setProgressOpen(true);
    try {
      await retuneModel(modelId, true);
      await loadModel({ name: modelId });
      await reload();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyId(null);
      setProgressOpen(false);
    }
  };

  const doSetDefault = async (modelId: string) => {
    setBusyId(modelId);
    try {
      await setDefaultModel(modelId);
      await reload();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyId(null);
    }
  };

  const doUnload = async (modelId: string) => {
    setBusyId(modelId);
    try {
      await unloadModel(modelId);
      await reload();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyId(null);
    }
  };

  const doInspect = async (name: string, includeTensors: boolean) => {
    setInspectFor(name);
    setInspectBusy(true);
    setInspectErr(null);
    setInspectIncludeTensors(includeTensors);
    try {
      const res = await inspectGguf({ name, includeTensors });
      setInspectData(res);
    } catch (e) {
      setInspectErr(e instanceof Error ? e.message : String(e));
      setInspectData(null);
    } finally {
      setInspectBusy(false);
    }
  };

  const closeInspect = () => {
    setInspectFor(null);
    setInspectData(null);
    setInspectErr(null);
  };

  const doDelete = async (name: string) => {
    if (!confirm(`Delete cached model ${name}? This removes the GGUF from disk.`)) return;
    setBusyId(name);
    try {
      await deleteModel(name);
      await reload();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div style={{ padding: 20, position: "relative" }}>
      {dragHover && (
        <div style={dropOverlay}>
          <div style={dropOverlayInner}>
            <div style={{ fontSize: 28, fontWeight: 600, marginBottom: 8 }}>
              Drop GGUF to load
            </div>
            <div style={{ fontSize: 13, color: "var(--ll-text-muted)" }}>
              Files are loaded directly from their on-disk path — no copy.
            </div>
          </div>
        </div>
      )}

      {inspectFor && (
        <div style={modalBackdrop} onClick={closeInspect}>
          <div style={modalPanel} onClick={(e) => e.stopPropagation()}>
            <header style={modalHeader}>
              <h3 style={{ margin: 0, fontSize: 14, fontWeight: 600 }}>
                Inspect: <code style={code}>{inspectFor}</code>
              </h3>
              <button onClick={closeInspect} style={btnSecondary}>
                Close
              </button>
            </header>
            <div style={modalBody}>
              {inspectBusy && (
                <div style={{ color: "var(--ll-text-muted)", fontSize: 13 }}>reading GGUF…</div>
              )}
              {inspectErr && <div style={errBox}>{inspectErr}</div>}
              {inspectData && !inspectBusy && (
                <InspectView
                  data={inspectData}
                  includeTensors={inspectIncludeTensors}
                  onLoadTensors={() => {
                    if (inspectFor) doInspect(inspectFor, true);
                  }}
                />
              )}
            </div>
          </div>
        </div>
      )}

      <header style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>Models</h2>
        <button onClick={reload} disabled={loading} style={btnSecondary}>
          {loading ? "loading…" : "Refresh"}
        </button>
      </header>

      {err && <div style={errBox}>{err}</div>}
      {dropStatus && (
        <div
          style={
            dropStatus.startsWith("error") ? dropStatusError : dropStatusInfo
          }
        >
          {dropStatus}
        </div>
      )}

      <section style={card}>
        <h3 style={cardHeader}>Discover — suggested models</h3>
        <p style={{ fontSize: 12, color: "var(--ll-text-muted)", margin: "0 0 12px 0" }}>
          Curated GGUFs known to work on this engine.{" "}
          {ramTotal !== null && ramAvail !== null && (
            <>
              You have{" "}
              <code style={code}>
                {(ramAvail / 1_073_741_824).toFixed(1)} /{" "}
                {(ramTotal / 1_073_741_824).toFixed(1)} GB
              </code>{" "}
              free.
            </>
          )}
        </p>
        {SUGGESTED_MODELS.map((m) => {
          const fits = ramAvail !== null && ramAvail > m.ramHintBytes * 1.1;
          const tight =
            ramAvail !== null &&
            ramAvail > m.ramHintBytes * 0.9 &&
            ramAvail <= m.ramHintBytes * 1.1;
          const tooBig = ramAvail !== null && ramAvail <= m.ramHintBytes * 0.9;
          const fitBadge = fits ? (
            <span style={fitOk}>fits</span>
          ) : tight ? (
            <span style={fitTight}>tight</span>
          ) : tooBig ? (
            <span style={fitNo}>too big</span>
          ) : null;
          return (
            <div
              key={m.hubRef}
              style={{
                padding: "10px 0",
                borderTop: "1px solid var(--ll-border)",
                display: "grid",
                gridTemplateColumns: "1fr auto",
                gap: 12,
                alignItems: "center",
              }}
            >
              <div>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <strong style={{ fontSize: 13 }}>{m.label}</strong>
                  {fitBadge}
                  <span style={{ fontSize: 11, color: "var(--ll-text-muted)" }}>
                    {(m.diskBytes / 1_048_576).toFixed(0)} MiB ·{" "}
                    {(m.ramHintBytes / 1_073_741_824).toFixed(1)} GB free recommended
                  </span>
                </div>
                <div style={{ fontSize: 12, color: "var(--ll-text-muted)", marginTop: 2 }}>
                  {m.use}
                </div>
                <code style={{ ...code, marginTop: 4, display: "inline-block" }}>
                  {m.hubRef}
                </code>
              </div>
              <button
                onClick={() => {
                  void doPull(m.hubRef);
                }}
                disabled={pulling || tooBig}
                title={tooBig ? "Not enough RAM for this model" : ""}
                style={btnPrimary}
              >
                {pulling && pullRef === m.hubRef ? "pulling…" : "Pull"}
              </button>
            </div>
          );
        })}
      </section>

      <section style={{ ...card, marginTop: 20 }}>
        <h3 style={cardHeader}>Loaded ({loaded.length})</h3>
        {loaded.length === 0 && !loading && (
          <div style={{ color: "var(--ll-text-muted)", fontSize: 13 }}>No models loaded. Pick one from the cached list below.</div>
        )}
        {loaded.map((m) => (
          <div
            key={m.id}
            style={{
              padding: "10px 0",
              borderTop: "1px solid var(--ll-border)",
              display: "flex",
              alignItems: "center",
              gap: 8,
            }}
          >
            <code style={{ ...code, flex: 1 }}>{m.id}</code>
            {m.moe && (
              <span
                style={moeBadge}
                title={
                  m.moe.n_experts_shared > 0
                    ? `Mixture-of-experts: ${m.moe.n_experts} routed, top-${m.moe.n_experts_used}, ${m.moe.n_experts_shared} shared (DeepSeek-V3 style)`
                    : `Mixture-of-experts: ${m.moe.n_experts} routed, top-${m.moe.n_experts_used}`
                }
              >
                MoE {m.moe.n_experts}×{m.moe.n_experts_used}
                {m.moe.n_experts_shared > 0 && `+${m.moe.n_experts_shared}s`}
              </span>
            )}
            {m.is_default ? (
              <span style={defaultBadge}>default</span>
            ) : (
              <button
                onClick={() => doSetDefault(m.id)}
                disabled={busyId === m.id}
                style={btnSecondary}
              >
                Set default
              </button>
            )}
            <button
              onClick={() => doRetune(m.id)}
              disabled={busyId === m.id}
              title="Re-run the full tuning sweep (KV coherence, CPU/GPU dispatch, kernels…) and reload to apply"
              style={btnSecondary}
            >
              Re-tune
            </button>
            <button
              onClick={() => doUnload(m.id)}
              disabled={busyId === m.id}
              title="Unload this model, freeing its RAM/VRAM"
              style={btnSecondary}
            >
              Unload
            </button>
          </div>
        ))}
      </section>

      <section style={{ ...card, marginTop: 20 }}>
        <h3 style={cardHeader}>Pull from HuggingFace</h3>
        <p style={{ fontSize: 12, color: "var(--ll-text-muted)", margin: "0 0 12px 0" }}>
          Format: <code style={code}>owner/repo:filename.gguf</code> — e.g.{" "}
          <code style={code}>Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF:qwen2.5-coder-0.5b-instruct-q4_k_m.gguf</code>
        </p>

        {/* Realtime HuggingFace search: type → GGUF repos → files. */}
        <div style={{ position: "relative", marginBottom: 12 }}>
          <input
            value={hfQuery}
            onChange={(e) => setHfQuery(e.target.value)}
            placeholder="🔍  Search HuggingFace for GGUF models…"
            style={input}
          />
          {(hfSearching || hfResults.length > 0 || hfRepo || hfErr) && (
            <div
              style={{
                position: "absolute",
                top: "100%",
                left: 0,
                right: 0,
                zIndex: 50,
                marginTop: 4,
                maxHeight: 320,
                overflowY: "auto",
                background: "var(--ll-bg-elevated, var(--ll-bg))",
                border: "1px solid var(--ll-border)",
                borderRadius: 8,
                boxShadow: "0 8px 24px rgba(0,0,0,0.35)",
              }}
            >
              {hfSearching && (
                <div style={{ padding: "8px 12px", fontSize: 12, color: "var(--ll-text-faint)" }}>
                  searching…
                </div>
              )}
              {hfErr && (
                <div style={{ padding: "8px 12px", fontSize: 12, color: "var(--ll-red)" }}>
                  {hfErr}
                </div>
              )}
              {!hfRepo &&
                hfResults.map((m) => (
                  <div
                    key={m.id}
                    onClick={() => pickRepo(m.id)}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      gap: 10,
                      padding: "8px 12px",
                      cursor: "pointer",
                      borderTop: "1px solid var(--ll-border)",
                    }}
                  >
                    <span style={{ fontWeight: 600, fontSize: 13 }}>{m.id}</span>
                    <span style={{ color: "var(--ll-text-faint)", fontSize: 11, whiteSpace: "nowrap" }}>
                      ↓ {m.downloads.toLocaleString()} · ♥ {m.likes.toLocaleString()}
                    </span>
                  </div>
                ))}
              {hfRepo && (
                <>
                  <div
                    onClick={() => setHfRepo(null)}
                    style={{ padding: "8px 12px", fontSize: 12, cursor: "pointer", color: "var(--ll-text-muted)" }}
                  >
                    ← {hfRepo} (back to results)
                  </div>
                  {hfFilesLoading && (
                    <div style={{ padding: "8px 12px", fontSize: 12, color: "var(--ll-text-faint)" }}>
                      loading files…
                    </div>
                  )}
                  {hfRepoFiles.map((f) => (
                    <div
                      key={f.rfilename}
                      onClick={() => pickFile(hfRepo, f.rfilename)}
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        gap: 10,
                        padding: "8px 12px",
                        cursor: "pointer",
                        borderTop: "1px solid var(--ll-border)",
                      }}
                    >
                      <span style={{ fontSize: 13, wordBreak: "break-all" }}>{f.rfilename}</span>
                      <span style={{ color: "var(--ll-text-faint)", fontSize: 11, whiteSpace: "nowrap" }}>
                        {humanBytes(f.size)}
                      </span>
                    </div>
                  ))}
                  {!hfFilesLoading && hfRepoFiles.length === 0 && (
                    <div style={{ padding: "8px 12px", fontSize: 12, color: "var(--ll-text-faint)" }}>
                      no .gguf files in this repo
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>

        <div style={{ display: "flex", gap: 8 }}>
          <input
            value={pullRef}
            onChange={(e) => setPullRef(e.target.value)}
            placeholder="owner/repo:filename.gguf"
            disabled={pulling}
            style={input}
            onKeyDown={(e) => {
              if (e.key === "Enter") doPull();
            }}
          />
          <button onClick={() => doPull()} disabled={pulling || !pullRef.trim()} style={btnPrimary}>
            {pulling ? "pulling…" : "Pull"}
          </button>
        </div>
        {pullStatus && (
          <div
            style={{
              marginTop: 12,
              fontSize: 12,
              color: pullStatus.startsWith("error") ? "var(--ll-red)" : "var(--ll-text-muted)",
              fontFamily: "ui-monospace, monospace",
            }}
          >
            {pullStatus}
          </div>
        )}
        {pullPct != null && (
          <div
            style={{
              marginTop: 8,
              height: 8,
              borderRadius: 5,
              background: "var(--ll-border)",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                height: "100%",
                width: `${pullPct}%`,
                background: "var(--ll-accent, #4f8cff)",
                borderRadius: 5,
                transition: "width 0.3s ease",
              }}
            />
          </div>
        )}
      </section>

      <section style={{ ...card, marginTop: 20 }}>
        <h3 style={cardHeader}>My Models ({cached.length})</h3>
        {cached.length === 0 && !loading && (
          <div style={{ color: "var(--ll-text-muted)", fontSize: 13 }}>
            No models on disk yet — pull one from Discover above.
          </div>
        )}
        {cached.length > 0 && (
          <div style={{ display: "grid", gap: 10 }}>
            {cached.map((m) => {
              const isBusy = busyId === m.name;
              return (
                <div
                  key={m.name}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                    padding: "12px 14px",
                    background: "var(--ll-bg-elev-2)",
                    border: "1px solid var(--ll-border)",
                    borderRadius: 10,
                  }}
                >
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div
                      style={{
                        fontSize: 14,
                        fontWeight: 600,
                        overflow: "hidden",
                        textOverflow: "ellipsis",
                        whiteSpace: "nowrap",
                      }}
                      title={m.name}
                    >
                      {m.name.replace(/\.gguf$/i, "")}
                    </div>
                    <div
                      style={{
                        display: "flex",
                        gap: 6,
                        marginTop: 7,
                        flexWrap: "wrap",
                      }}
                    >
                      {m.details?.family && (
                        <span className="ll-badge">{m.details.family}</span>
                      )}
                      {m.details?.quantization_level && (
                        <span className="ll-badge">
                          {m.details.quantization_level}
                        </span>
                      )}
                      <span className="ll-badge">{humanBytes(m.size)}</span>
                    </div>
                    {isBusy && (
                      <div style={busyHint}>
                        Loading… large GGUFs (10+ GB) can take 30–90 s — the
                        server mmaps + copies weights into per-layer tensors.
                      </div>
                    )}
                  </div>
                  <div style={{ display: "flex", gap: 6, flex: "none" }}>
                    <button
                      onClick={() => doInspect(m.name, false)}
                      disabled={isBusy}
                      style={btnSecondary}
                      title="Show metadata + tensor stats without loading the model"
                    >
                      Inspect
                    </button>
                    <button
                      onClick={() => doLoad(m.name)}
                      disabled={isBusy}
                      className="ll-btn ll-btn-primary"
                    >
                      {isBusy ? "Loading…" : "Load"}
                    </button>
                    <button
                      onClick={() => doDelete(m.name)}
                      disabled={isBusy}
                      style={btnDanger}
                      title="Delete from disk"
                    >
                      Delete
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>

      <TuneProgressModal open={progressOpen} subtitle={progressModel} />
    </div>
  );
}

function InspectView({
  data,
  includeTensors,
  onLoadTensors,
}: {
  data: GgufInspectResult;
  includeTensors: boolean;
  onLoadTensors: () => void;
}) {
  const fmtNum = (n: number | null | undefined) =>
    n === null || n === undefined ? "—" : n.toLocaleString();
  // Total params as billions/millions for at-a-glance reading; the
  // raw count is in the field below for anyone who wants it.
  const paramLabel = (() => {
    const p = data.total_params;
    if (p >= 1e9) return `${(p / 1e9).toFixed(2)} B`;
    if (p >= 1e6) return `${(p / 1e6).toFixed(0)} M`;
    return p.toLocaleString();
  })();

  return (
    <div>
      <section style={inspectSection}>
        <h4 style={inspectH4}>Architecture</h4>
        <div style={inspectGrid}>
          <Field label="Architecture" value={data.architecture} />
          <Field label="Name" value={data.name} />
          {data.size_label && <Field label="Size label" value={data.size_label} />}
          <Field label="Parameters" value={paramLabel} />
          <Field label="Context length" value={fmtNum(data.context_length)} />
          <Field label="Block count (layers)" value={fmtNum(data.block_count)} />
          <Field label="Embedding length" value={fmtNum(data.embedding_length)} />
          <Field label="Head count" value={fmtNum(data.head_count)} />
          <Field label="Head count (KV)" value={fmtNum(data.head_count_kv)} />
          <Field label="Head dim" value={fmtNum(data.head_dim)} />
          <Field label="Vocab size" value={fmtNum(data.vocab_size)} />
          {data.file_type !== undefined && (
            <Field label="File type" value={`#${data.file_type}`} />
          )}
          {data.n_experts !== undefined && (
            <Field
              label="MoE experts"
              value={
                data.n_experts_shared && data.n_experts_shared > 0
                  ? `${data.n_experts} routed, top-${data.n_experts_used ?? "?"}, ${data.n_experts_shared} shared`
                  : `${data.n_experts} routed, top-${data.n_experts_used ?? "?"}`
              }
            />
          )}
        </div>
      </section>

      <section style={inspectSection}>
        <h4 style={inspectH4}>File</h4>
        <div style={inspectGrid}>
          <Field
            label="On-disk size"
            value={`${humanBytes(data.file_bytes)} (${data.file_bytes.toLocaleString()} bytes)`}
          />
          <Field label="Tensor count" value={data.tensor_count.toLocaleString()} />
          <Field label="Tensor bytes" value={humanBytes(data.total_tensor_bytes)} />
          <Field
            label="Path"
            value={<code style={code}>{data.path}</code>}
          />
        </div>
      </section>

      <section style={inspectSection}>
        <h4 style={inspectH4}>Dtype breakdown</h4>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ color: "var(--ll-text-muted)", textAlign: "left" }}>
              <th style={th}>Dtype</th>
              <th style={th}>Tensors</th>
              <th style={th}>Bytes</th>
              <th style={th}>% of file</th>
            </tr>
          </thead>
          <tbody>
            {data.dtypes.map((d) => (
              <tr key={d.dtype} style={{ borderTop: "1px solid var(--ll-border)" }}>
                <td style={td}>
                  <code style={code}>{d.dtype}</code>
                </td>
                <td style={td}>{d.tensor_count.toLocaleString()}</td>
                <td style={td}>{humanBytes(d.bytes)}</td>
                <td style={td}>
                  {data.file_bytes > 0
                    ? `${((d.bytes / data.file_bytes) * 100).toFixed(1)}%`
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {data.model_card && (
        <section style={inspectSection}>
          <h4 style={inspectH4}>Model card</h4>
          <div style={inspectGrid}>
            {data.model_card.license && (
              <Field label="License" value={data.model_card.license} />
            )}
            {data.model_card.base_model && (
              <Field label="Base model" value={data.model_card.base_model} />
            )}
            {data.model_card.model_creator && (
              <Field label="Creator" value={data.model_card.model_creator} />
            )}
            {data.model_card.quantized_by && (
              <Field label="Quantized by" value={data.model_card.quantized_by} />
            )}
            {data.model_card.tags.length > 0 && (
              <Field
                label="Tags"
                value={data.model_card.tags.slice(0, 8).join(", ")}
              />
            )}
            {data.model_card.source_url && (
              <Field
                label="Source"
                value={
                  <a
                    href={data.model_card.source_url}
                    target="_blank"
                    rel="noreferrer"
                    style={{ color: "var(--ll-accent)" }}
                  >
                    {data.model_card.source_url}
                  </a>
                }
              />
            )}
          </div>
          {data.model_card.description && (
            <div style={{ fontSize: 12, color: "var(--ll-text)", marginTop: 8 }}>
              {data.model_card.description}
            </div>
          )}
        </section>
      )}

      <section style={inspectSection}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: 8,
          }}
        >
          <h4 style={{ ...inspectH4, margin: 0 }}>
            Tensors {includeTensors && `(${data.tensors?.length ?? 0})`}
          </h4>
          {!includeTensors && (
            <button onClick={onLoadTensors} style={btnSecondary}>
              Load tensor list
            </button>
          )}
        </div>
        {!includeTensors && (
          <div style={{ fontSize: 12, color: "var(--ll-text-muted)" }}>
            Per-tensor list omitted to keep the response small. Click "Load
            tensor list" to fetch all {data.tensor_count} entries.
          </div>
        )}
        {includeTensors && data.tensors && (
          <div style={{ maxHeight: 360, overflowY: "auto", border: "1px solid var(--ll-border)", borderRadius: 4 }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead style={{ position: "sticky", top: 0, background: "var(--ll-bg-elev)" }}>
                <tr style={{ color: "var(--ll-text-muted)", textAlign: "left" }}>
                  <th style={th}>Name</th>
                  <th style={th}>Dtype</th>
                  <th style={th}>Shape</th>
                  <th style={th}>Bytes</th>
                </tr>
              </thead>
              <tbody>
                {data.tensors.map((t) => (
                  <tr key={t.name} style={{ borderTop: "1px solid var(--ll-border)" }}>
                    <td style={td}>
                      <code style={{ ...code, fontSize: 10 }}>{t.name}</code>
                    </td>
                    <td style={td}>
                      <code style={code}>{t.dtype}</code>
                    </td>
                    <td style={td}>{t.shape.join(" × ")}</td>
                    <td style={td}>{humanBytes(t.bytes)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}

function Field({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div>
      <div style={{ fontSize: 10, color: "var(--ll-text-muted)", textTransform: "uppercase", letterSpacing: 0.3 }}>
        {label}
      </div>
      <div style={{ fontSize: 13, color: "var(--ll-text)", marginTop: 2 }}>{value}</div>
    </div>
  );
}

const card: React.CSSProperties = {
  background: "var(--ll-bg-elev)",
  border: "1px solid var(--ll-border)",
  borderRadius: 6,
  padding: 16,
};
const cardHeader: React.CSSProperties = { margin: "0 0 12px 0", fontSize: 14, fontWeight: 600 };
const code: React.CSSProperties = {
  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
  background: "var(--ll-bg)",
  padding: "1px 6px",
  borderRadius: 3,
  fontSize: 12,
};
const input: React.CSSProperties = {
  flex: 1,
  padding: 8,
  background: "var(--ll-bg)",
  color: "var(--ll-text)",
  border: "1px solid var(--ll-border-strong)",
  borderRadius: 4,
  fontSize: 13,
  fontFamily: "inherit",
};
const btnPrimary: React.CSSProperties = {
  padding: "0 18px",
  background: "var(--ll-green)",
  color: "white",
  border: "1px solid var(--ll-green)",
  borderRadius: 4,
  cursor: "pointer",
  fontSize: 13,
};
const btnSecondary: React.CSSProperties = {
  padding: "6px 12px",
  background: "transparent",
  color: "var(--ll-text)",
  border: "1px solid var(--ll-border-strong)",
  borderRadius: 4,
  cursor: "pointer",
  fontSize: 12,
};
const btnDanger: React.CSSProperties = {
  padding: "6px 12px",
  background: "transparent",
  color: "var(--ll-red)",
  border: "1px solid var(--ll-red)",
  borderRadius: 4,
  cursor: "pointer",
  fontSize: 12,
};
const defaultBadge: React.CSSProperties = {
  padding: "2px 8px",
  background: "rgba(63, 185, 80, 0.15)",
  color: "var(--ll-green)",
  border: "1px solid rgba(63, 185, 80, 0.5)",
  borderRadius: 12,
  fontSize: 11,
  fontWeight: 500,
  letterSpacing: 0.3,
  textTransform: "uppercase",
};

/// MoE badge — compact "MoE 8×2" / "MoE 256×8+1s" rendering.
/// Purple-ish to distinguish from the green "default" badge so
/// the loaded-models row stays scannable. Hover-title carries the
/// full "N routed, top-K, M shared" wording for clients with
/// dense vs MoE side by side.
const moeBadge: React.CSSProperties = {
  padding: "2px 8px",
  background: "rgba(163, 113, 247, 0.15)",
  color: "var(--ll-purple)",
  border: "1px solid rgba(163, 113, 247, 0.5)",
  borderRadius: 12,
  fontSize: 11,
  fontWeight: 500,
  letterSpacing: 0.3,
  fontFamily: "ui-monospace, SFMono-Regular, monospace",
};
const th: React.CSSProperties = {
  padding: "8px 6px",
  fontWeight: 500,
  fontSize: 11,
  textTransform: "uppercase",
  letterSpacing: 0.4,
};
const td: React.CSSProperties = { padding: "8px 6px" };
const errBox: React.CSSProperties = {
  padding: 10,
  background: "var(--ll-red-soft)",
  border: "1px solid var(--ll-red)",
  borderRadius: 4,
  color: "var(--ll-red)",
  fontSize: 13,
  marginBottom: 16,
  whiteSpace: "pre-wrap",
  wordBreak: "break-word",
};
const busyHint: React.CSSProperties = {
  marginTop: 6,
  fontSize: 11,
  color: "var(--ll-text-muted)",
  fontStyle: "italic",
};
const badge: React.CSSProperties = {
  padding: "1px 8px",
  fontSize: 10,
  fontWeight: 600,
  letterSpacing: 0.3,
  textTransform: "uppercase",
  borderRadius: 10,
};
const fitOk: React.CSSProperties = {
  ...badge,
  background: "rgba(63, 185, 80, 0.15)",
  color: "var(--ll-green)",
  border: "1px solid rgba(63, 185, 80, 0.5)",
};
const fitTight: React.CSSProperties = {
  ...badge,
  background: "rgba(210, 153, 34, 0.15)",
  color: "var(--ll-yellow)",
  border: "1px solid rgba(210, 153, 34, 0.5)",
};
const fitNo: React.CSSProperties = {
  ...badge,
  background: "rgba(248, 81, 73, 0.15)",
  color: "var(--ll-red)",
  border: "1px solid rgba(248, 81, 73, 0.5)",
};
const dropOverlay: React.CSSProperties = {
  position: "fixed",
  inset: 0,
  background: "rgba(13, 17, 23, 0.85)",
  border: "3px dashed var(--ll-green)",
  zIndex: 1000,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  pointerEvents: "none",
};
const dropOverlayInner: React.CSSProperties = {
  textAlign: "center",
  color: "var(--ll-text)",
};
const dropStatusInfo: React.CSSProperties = {
  padding: 10,
  background: "rgba(63, 185, 80, 0.08)",
  border: "1px solid rgba(63, 185, 80, 0.4)",
  borderRadius: 4,
  color: "var(--ll-green)",
  fontSize: 13,
  marginBottom: 16,
  fontFamily: "ui-monospace, monospace",
};
const dropStatusError: React.CSSProperties = {
  ...dropStatusInfo,
  background: "rgba(248, 81, 73, 0.08)",
  border: "1px solid rgba(248, 81, 73, 0.4)",
  color: "var(--ll-red)",
};
const modalBackdrop: React.CSSProperties = {
  position: "fixed",
  inset: 0,
  background: "rgba(0, 0, 0, 0.6)",
  zIndex: 999,
  display: "flex",
  alignItems: "flex-start",
  justifyContent: "center",
  paddingTop: 60,
};
const modalPanel: React.CSSProperties = {
  width: "min(820px, 92vw)",
  maxHeight: "calc(100vh - 100px)",
  background: "var(--ll-bg)",
  border: "1px solid var(--ll-border-strong)",
  borderRadius: 8,
  display: "flex",
  flexDirection: "column",
  overflow: "hidden",
};
const modalHeader: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  padding: "12px 16px",
  borderBottom: "1px solid var(--ll-border-strong)",
  background: "var(--ll-bg-elev)",
};
const modalBody: React.CSSProperties = {
  padding: 16,
  overflowY: "auto",
  flex: 1,
};
const inspectSection: React.CSSProperties = {
  marginBottom: 20,
};
const inspectH4: React.CSSProperties = {
  margin: "0 0 10px 0",
  fontSize: 11,
  fontWeight: 600,
  color: "var(--ll-text-muted)",
  textTransform: "uppercase",
  letterSpacing: 0.5,
};
const inspectGrid: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
  gap: 12,
};
