// Settings page. Edits `config.toml` via PUT /v1/config. Sections
// marked "live" hot-apply via the on-disk watcher; "reload" sections
// surface a reload-required banner; the "server" section needs a
// process restart and the banner says so explicitly.

import { useEffect, useState } from "react";
import {
  applyConfigProfile,
  deleteCrashLog,
  getAuditLogTail,
  getConfig,
  getCrashLog,
  getHealth,
  getLanInfo,
  getMetrics,
  listCrashLogs,
  previewChatTemplate,
  putConfig,
  type AppConfig,
  type AuditTailResponse,
  type ConfigEnvelope,
  type ConfigProfile,
  type CrashLogEntry,
  type CrashLogsList,
  type HealthInfo,
  type LanInfoResult,
  type MetricsSnapshot,
} from "../api";
import { CODE_THEMES } from "../themes";

type SaveStatus =
  | { kind: "idle" }
  | { kind: "saving" }
  | { kind: "saved"; requiresReload: boolean; requiresRestart: boolean }
  | { kind: "error"; message: string };

export default function SettingsPage() {
  const [envelope, setEnvelope] = useState<ConfigEnvelope | null>(null);
  const [draft, setDraft] = useState<AppConfig | null>(null);
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [metrics, setMetrics] = useState<MetricsSnapshot | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [save, setSave] = useState<SaveStatus>({ kind: "idle" });
  /// Selected profile name (from the dropdown). Empty = no selection,
  /// no overrides previewed; clicking Apply with empty is a no-op.
  const [selectedProfile, setSelectedProfile] = useState("");
  /// Status of the most recent profile apply, mirrored on top of the
  /// normal save banner.
  const [profileApply, setProfileApply] = useState<SaveStatus>({ kind: "idle" });

  useEffect(() => {
    let cancel = false;
    (async () => {
      try {
        const [cfg, h, m] = await Promise.all([
          getConfig(),
          getHealth().catch(() => null),
          getMetrics().catch(() => null),
        ]);
        if (cancel) return;
        setEnvelope(cfg);
        setDraft(cfg.config);
        setHealth(h);
        setMetrics(m);
      } catch (e) {
        if (!cancel) setErr(e instanceof Error ? e.message : String(e));
      } finally {
        if (!cancel) setLoading(false);
      }
    })();
    return () => {
      cancel = true;
    };
  }, []);

  // Live-tail the metrics + health independent of the form so the
  // banner stays current while editing. Slow tick (5 s) — anything
  // higher fights with the form's typing latency.
  useEffect(() => {
    const id = setInterval(async () => {
      const [h, m] = await Promise.all([
        getHealth().catch(() => null),
        getMetrics().catch(() => null),
      ]);
      setHealth(h);
      setMetrics(m);
    }, 5000);
    return () => clearInterval(id);
  }, []);

  if (loading) {
    return <div style={{ padding: 20, color: "var(--ll-text-muted)" }}>loading config…</div>;
  }
  if (err || !envelope || !draft) {
    return (
      <div style={{ padding: 20 }}>
        <div style={errBox}>error loading config: {err ?? "unknown"}</div>
      </div>
    );
  }

  const dirty = JSON.stringify(draft) !== JSON.stringify(envelope.config);

  async function onSave() {
    if (!draft) return;
    setSave({ kind: "saving" });
    try {
      const result = await putConfig(draft);
      // Refresh the on-disk view so the "previous" snapshot tracks
      // what's now on disk — the watcher will re-broadcast but the
      // form's notion of "current" comes from this fetch.
      const fresh = await getConfig();
      setEnvelope(fresh);
      setDraft(fresh.config);
      setSave({
        kind: "saved",
        requiresReload: result.requires_model_reload,
        requiresRestart: result.requires_server_restart,
      });
    } catch (e) {
      setSave({ kind: "error", message: e instanceof Error ? e.message : String(e) });
    }
  }

  function onReset() {
    if (envelope) setDraft(envelope.config);
    setSave({ kind: "idle" });
  }

  async function onApplyProfile() {
    if (!selectedProfile) return;
    // Refuse to apply on top of unsaved draft edits — the on-disk
    // merge would silently lose the user's typing. The form makes
    // this clear via the Discard button.
    if (dirty) {
      setProfileApply({
        kind: "error",
        message:
          "Discard or save your current edits before switching profiles — applying a profile overwrites the file you're editing.",
      });
      return;
    }
    setProfileApply({ kind: "saving" });
    try {
      const result = await applyConfigProfile(selectedProfile);
      // Re-fetch so the form reflects the merged on-disk state.
      const fresh = await getConfig();
      setEnvelope(fresh);
      setDraft(fresh.config);
      setProfileApply({
        kind: "saved",
        requiresReload: result.requires_model_reload,
        requiresRestart: result.requires_server_restart,
      });
    } catch (e) {
      setProfileApply({
        kind: "error",
        message: e instanceof Error ? e.message : String(e),
      });
    }
  }

  // Setters — keep these as small typed lambdas instead of one
  // generic setter so TS catches typos in nested paths.
  const set = {
    model: (patch: Partial<AppConfig["model"]>) =>
      setDraft({ ...draft!, model: { ...draft!.model, ...patch } }),
    inference: (patch: Partial<AppConfig["inference"]>) =>
      setDraft({ ...draft!, inference: { ...draft!.inference, ...patch } }),
    server: (patch: Partial<AppConfig["server"]>) =>
      setDraft({ ...draft!, server: { ...draft!.server, ...patch } }),
    ui: (patch: Partial<AppConfig["ui"]>) =>
      setDraft({ ...draft!, ui: { ...draft!.ui, ...patch } }),
  };

  return (
    <div style={{ padding: 20, paddingBottom: 80 }}>
      <header style={{ marginBottom: 16 }}>
        <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>Settings</h2>
        <p style={{ fontSize: 12, color: "var(--ll-text-muted)", margin: "4px 0 0 0" }}>
          Edits write straight to{" "}
          <code style={inlineCode}>{envelope.config_path ?? "config.toml"}</code>.
          Sections labelled <span style={badgeLive}>live</span> hot-apply
          via the on-disk watcher. Sections labelled{" "}
          <span style={badgeReload}>reload</span> need the model reloaded
          (Models page → click Load). Sections labelled{" "}
          <span style={badgeRestart}>restart</span> need a server restart.
        </p>
      </header>

      {save.kind === "saved" && (
        <div style={save.requiresRestart || save.requiresReload ? infoBoxWarn : infoBoxOk}>
          Saved.{" "}
          {save.requiresRestart
            ? "Server fields changed — restart rustllama for them to take effect."
            : save.requiresReload
              ? "Model / inference fields changed — reload the model from the Models page."
              : "Changes hot-applied via the watcher."}
        </div>
      )}
      {save.kind === "error" && (
        <div style={errBox}>save failed: {save.message}</div>
      )}

      <ProfilesSection
        profiles={draft.profiles}
        selected={selectedProfile}
        onSelect={setSelectedProfile}
        onApply={onApplyProfile}
        status={profileApply}
        dirty={dirty}
      />

      <section style={{ ...card, marginTop: 16 }}>
        <SectionHeader title="Server" hint="restart" />
        <Field label="Status">
          <span style={{ fontSize: 13 }}>
            {health?.draining ? "draining" : health ? "live" : "offline"}
          </span>
        </Field>
        <NumberField
          label="Port"
          hint="default 11434"
          value={draft.server.port}
          min={1}
          max={65535}
          onChange={(v) => set.server({ port: v })}
        />
        <TextField
          label="Bind address"
          hint="127.0.0.1 = localhost only; 0.0.0.0 = LAN"
          value={draft.server.bind_addr}
          onChange={(v) => set.server({ bind_addr: v })}
        />
        <NumberField
          label="Concurrency"
          hint="In-flight chats per model. Each fork uses its own KV cache."
          value={draft.server.concurrency}
          min={1}
          max={64}
          onChange={(v) => set.server({ concurrency: v })}
        />
        <NumberField
          label="Max pending / model"
          hint="Backpressure cap — beyond this requests get 503 + Retry-After."
          value={draft.server.max_pending_per_model}
          min={1}
          max={4096}
          onChange={(v) => set.server({ max_pending_per_model: v })}
        />
        <NumberField
          label="Max loaded models"
          hint="Warm pool cap (LRU evicts non-default models)."
          value={draft.server.max_loaded_models}
          min={1}
          max={64}
          onChange={(v) => set.server({ max_loaded_models: v })}
        />
        <TextField
          label="API key"
          hint="Empty = no auth (default). Sent as `Authorization: Bearer <key>`."
          value={draft.server.api_key}
          onChange={(v) => set.server({ api_key: v })}
        />
        <TextField
          label="CORS origins"
          hint="Comma-separated. Empty = none. `*` allows any origin."
          value={draft.server.cors_origins.join(", ")}
          onChange={(v) =>
            set.server({
              cors_origins: v
                .split(",")
                .map((s) => s.trim())
                .filter((s) => s.length > 0),
            })
          }
        />
        <BoolField
          label="Audit log"
          hint="Append a JSONL line per request to the audit log file. Off by default — opt in for LAN-exposed deployments. Never records request bodies, omits Authorization values, masks common credential-shaped query parameters (api_key, access_token, password, signature, and friends)."
          value={draft.server.audit_log ?? false}
          onChange={(v) => set.server({ audit_log: v })}
        />
        <TextField
          label="Audit log path"
          hint="Destination JSONL file. Empty = `<user-data>/logs/audit.log.jsonl`."
          value={draft.server.audit_log_path ?? ""}
          onChange={(v) => set.server({ audit_log_path: v })}
        />
      </section>

      <AuditLogTailPanel />


      <LanAccessPanel
        bindAddrDraft={draft.server.bind_addr}
        apiKeyDraft={draft.server.api_key}
        dirty={dirty}
      />

      <section style={{ ...card, marginTop: 16 }}>
        <SectionHeader title="Model" hint="reload" />
        <TextField
          label="GGUF path"
          hint="Absolute path. Mutually exclusive with hub ref."
          value={draft.model.path ?? ""}
          onChange={(v) => set.model({ path: v.length > 0 ? v : null })}
        />
        <TextField
          label="Hub ref"
          hint="owner/repo:filename.gguf — pulled into the local cache."
          value={draft.model.hub ?? ""}
          onChange={(v) => set.model({ hub: v.length > 0 ? v : null })}
        />
        <TextField
          label="Chat template"
          hint="auto = read from GGUF; chatml / llama3 / inline Jinja."
          value={draft.model.chat_template}
          onChange={(v) => set.model({ chat_template: v })}
        />
        <ChatTemplatePreview template={draft.model.chat_template} />
        <KV k="Loaded right now" v={metrics?.model_id ?? health?.model_id ?? "—"} />
      </section>

      <section style={{ ...card, marginTop: 16 }}>
        <SectionHeader title="Inference" hint="reload" />
        <NumberField
          label="Context size"
          hint="Max tokens (prompt + completion) per request."
          value={draft.inference.ctx_size}
          min={512}
          max={1048576}
          onChange={(v) => set.inference({ ctx_size: v })}
        />
        <NumberField
          label="Batch size"
          hint="Prefill chunk. Larger = faster but more RAM per chunk."
          value={draft.inference.batch_size}
          min={1}
          max={65536}
          onChange={(v) => set.inference({ batch_size: v })}
        />
        <NumberField
          label="Threads"
          hint="0 = auto (physical cores)."
          value={draft.inference.threads}
          min={0}
          max={256}
          onChange={(v) => set.inference({ threads: v })}
        />
        <NumberField
          label="GPU layers"
          hint="Layers whose weights live in VRAM (currently a no-op — see Hardware)."
          value={draft.inference.n_gpu_layers}
          min={0}
          max={999}
          onChange={(v) => set.inference({ n_gpu_layers: v })}
        />
        <SelectField
          label="KV dtype (both K and V)"
          hint="Default for both K and V. Override per-channel below to split."
          value={draft.inference.kv_dtype}
          options={["f32", "q8_0", "q4_0", "tq1", "tq2", "tq4", "tq8", "nvfp4"]}
          onChange={(v) => set.inference({ kv_dtype: v })}
        />
        <SelectField
          label="K dtype (override)"
          hint="(empty = use KV dtype). Engine storage today couples K/V — when V differs, V silently mirrors K until per-side KV storage lands."
          value={draft.inference.k_dtype ?? ""}
          options={["", "f32", "q8_0", "q4_0", "tq1", "tq2", "tq4", "tq8", "nvfp4"]}
          onChange={(v) =>
            set.inference({ k_dtype: v.length > 0 ? v : null })
          }
        />
        <SelectField
          label="V dtype (override)"
          hint="(empty = use KV dtype). V is more sensitive to precision than K — a common future pattern is K=tq4, V=q8_0."
          value={draft.inference.v_dtype ?? ""}
          options={["", "f32", "q8_0", "q4_0", "tq1", "tq2", "tq4", "tq8", "nvfp4"]}
          onChange={(v) =>
            set.inference({ v_dtype: v.length > 0 ? v : null })
          }
        />
        <TextField
          label="KV bias sidecar (path)"
          hint="K-cache mean-centering bias GGUF (rustllama kv-calibrate). Empty = auto-discover <model>.kvbias.gguf. Applies on q4_0 KV only; exactly softmax-invariant."
          value={draft.inference.kv_bias_path ?? ""}
          onChange={(v) =>
            set.inference({ kv_bias_path: v.length > 0 ? v : null })
          }
        />
        <BoolField
          label="Speculative decoding (n-gram)"
          hint="Prompt-lookup drafting + one batched verify per round. No second model, zero RAM. Raw-softmax sampling; grammar requests fall back to the classic sampler."
          value={draft.inference.speculative_ngram}
          onChange={(v) => set.inference({ speculative_ngram: v })}
        />
        <TextField
          label="Speculative draft model (path)"
          hint="Small same-tokenizer GGUF that drafts tokens for the big model to verify (e.g. a Qwen3.8-4B distill for Bonsai-2-27B). Tokenizer compatibility is verified at load. Takes precedence over n-gram."
          value={draft.inference.speculative_draft_path ?? ""}
          onChange={(v) =>
            set.inference({ speculative_draft_path: v.length > 0 ? v : null })
          }
        />
        <NumberField
          label="Speculative draft K"
          hint="Candidates drafted per round on the draft-model path. Higher = bigger win when accepted, more wasted verify on a miss."
          value={draft.inference.speculative_draft_k}
          min={1}
          max={32}
          onChange={(v) => set.inference({ speculative_draft_k: v })}
        />
        <BoolField
          label="Flash attention"
          hint="Fused softmax-attention. Currently no-op until SYCL dispatch wires up."
          value={draft.inference.flash_attention}
          onChange={(v) => set.inference({ flash_attention: v })}
        />
        <BoolField
          label="Prefix cache"
          hint="Reuse KV for shared prompt prefixes (huge speedup for chat / coding flows)."
          value={draft.inference.prefix_cache}
          onChange={(v) => set.inference({ prefix_cache: v })}
        />
        <NumberField
          label="Prefix snapshots kept"
          hint="0 disables the multi-snapshot pool; the engine still does single-snapshot LCP."
          value={draft.inference.prefix_cache_max_snapshots}
          min={0}
          max={64}
          onChange={(v) => set.inference({ prefix_cache_max_snapshots: v })}
        />
        <BoolField
          label="Keep quant raw"
          hint="Skip dequant-to-F16 on small tensors. Saves 1-3 GB on a 7B-24B model; modest matvec cost. Recommended on ≤16 GB hosts."
          value={draft.inference.keep_quant_raw}
          onChange={(v) => set.inference({ keep_quant_raw: v })}
        />
        <NumberField
          label="Max tool iterations"
          hint="Per-response cap on tool-call bodies. 0 disables the cap."
          value={draft.inference.max_tool_iterations}
          min={0}
          max={256}
          onChange={(v) => set.inference({ max_tool_iterations: v })}
        />
      </section>

      <section style={{ ...card, marginTop: 16 }}>
        <SectionHeader title="UI" hint="live" />
        <SelectField
          label="Theme"
          hint="`system` follows OS dark/light."
          value={draft.ui.theme}
          options={["system", "dark", "light"]}
          onChange={(v) => set.ui({ theme: v })}
        />
        <NumberField
          label="Font size"
          value={draft.ui.font_size}
          min={8}
          max={32}
          onChange={(v) => set.ui({ font_size: v })}
        />
        <Field
          label="Code theme"
          hint="Syntax-highlight palette for assistant code blocks. Live-applies — no model reload."
        >
          <select
            style={inputStyle}
            value={draft.ui.code_theme}
            onChange={(e) => set.ui({ code_theme: e.target.value })}
          >
            {CODE_THEMES.map((t) => (
              <option key={t.name} value={t.name}>
                {t.label}
              </option>
            ))}
          </select>
        </Field>
      </section>

      <section style={{ ...card, marginTop: 16 }}>
        <SectionHeader title="Hardware" />
        <KV
          k="SYCL devices visible"
          v={
            metrics
              ? metrics.sycl_device_count > 0
                ? `${metrics.sycl_device_count} (oneAPI detected)`
                : "0 (no Intel GPU / oneAPI runtime)"
              : "—"
          }
        />
        <p style={{ fontSize: 12, color: "var(--ll-text-muted)", marginTop: 8, lineHeight: 1.5 }}>
          {metrics && metrics.sycl_device_count > 0 ? (
            <>
              Your Intel GPU is detected by the SYCL runtime ✓. The
              forward pass dispatches RMSNorm, RoPE, SwiGLU,
              FlashAttention decode + prefill, and quantized matvec
              (Q8_0, Q4_K, Q5_K, Q6_K) to the GPU when the layer is
              GPU-resident per <code style={inlineCode}>n_gpu_layers</code>{" "}
              and the weight isn't pinned to CPU via{" "}
              <code style={inlineCode}>placement.overrides</code>. Kernel
              local-work-size is read per-shape from the tuner cache
              (run <code style={inlineCode}>rustllama tune</code> once
              per device + model to populate it). Non-quantized
              (F16 / BF16 / F32) matvec and the embedding gather still
              run on CPU — completion of those paths is a smaller
              cleanup pass than the headline dispatch wire-up.
            </>
          ) : (
            <>
              No SYCL devices visible. The SYCL backend is always
              compiled in; to use it, install an Intel GPU with its
              driver and the Level Zero (or OpenCL) runtime. Building
              from source additionally needs the Intel oneAPI Base
              Toolkit — see{" "}
              <code style={inlineCode}>scripts\build-env.bat</code>.
            </>
          )}
        </p>
      </section>

      <section style={{ ...card, marginTop: 16 }}>
        <SectionHeader title="KV dtype reference" />
        <ul style={{ color: "var(--ll-text-muted)", fontSize: 12, lineHeight: 1.6, margin: "4px 0 0 16px" }}>
          <li><code style={inlineCode}>f32</code> — default; no quantization, exact attention.</li>
          <li><code style={inlineCode}>q8_0</code> — per-row int8 with f32 scale (~4× memory cut).</li>
          <li><code style={inlineCode}>tq1</code> / <code style={inlineCode}>tq2</code> / <code style={inlineCode}>tq4</code> / <code style={inlineCode}>tq8</code> — TurboQuant: Walsh-Hadamard rotation + uniform N-bit quant.</li>
          <li><code style={inlineCode}>q4_0</code> — ggml Q4_0 blocks (int4 + f16 scale per 32 elements, ~7× memory cut); the Prism-compatible 4-bit KV and the hybrid-model (Bonsai/Qwen3.6-family) quantized option.</li>
          <li><code style={inlineCode}>nvfp4</code> — NVIDIA NVFP4 (E2M1 + FP8 E4M3 scale per 16-element block).</li>
        </ul>
      </section>

      <CrashLogsPanel />

      <div style={bottomBar}>
        <button
          style={dirty ? btnPrimary : btnPrimaryDisabled}
          disabled={!dirty || save.kind === "saving"}
          onClick={onSave}
        >
          {save.kind === "saving" ? "saving…" : "Save changes"}
        </button>
        <button
          style={dirty ? btnSecondary : btnSecondaryDisabled}
          disabled={!dirty || save.kind === "saving"}
          onClick={onReset}
        >
          Discard
        </button>
      </div>
    </div>
  );
}

// ----- form primitives -----

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div style={{ padding: "8px 0", borderTop: "1px solid var(--ll-border)" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <span style={{ width: 200, color: "var(--ll-text-muted)", fontSize: 13 }}>{label}</span>
        <div style={{ flex: 1 }}>{children}</div>
      </div>
      {hint && <div style={{ paddingLeft: 212, color: "var(--ll-text-faint)", fontSize: 11, marginTop: 2 }}>{hint}</div>}
    </div>
  );
}

function TextField({
  label,
  hint,
  value,
  onChange,
}: {
  label: string;
  hint?: string;
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <Field label={label} hint={hint}>
      <input
        style={inputStyle}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
    </Field>
  );
}

function NumberField({
  label,
  hint,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  hint?: string;
  value: number;
  min?: number;
  max?: number;
  onChange: (v: number) => void;
}) {
  return (
    <Field label={label} hint={hint}>
      <input
        type="number"
        style={inputStyle}
        value={value}
        min={min}
        max={max}
        onChange={(e) => {
          const n = Number(e.target.value);
          if (!Number.isNaN(n)) onChange(n);
        }}
      />
    </Field>
  );
}

function BoolField({
  label,
  hint,
  value,
  onChange,
}: {
  label: string;
  hint?: string;
  value: boolean;
  onChange: (v: boolean) => void;
}) {
  return (
    <Field label={label} hint={hint}>
      <input
        type="checkbox"
        checked={value}
        onChange={(e) => onChange(e.target.checked)}
      />
    </Field>
  );
}

function SelectField({
  label,
  hint,
  value,
  options,
  onChange,
}: {
  label: string;
  hint?: string;
  value: string;
  options: string[];
  onChange: (v: string) => void;
}) {
  return (
    <Field label={label} hint={hint}>
      <select
        style={inputStyle}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {options.map((o) => (
          <option key={o} value={o}>{o}</option>
        ))}
      </select>
    </Field>
  );
}

function KV({ k, v }: { k: string; v: string | number }) {
  return (
    <div style={{ display: "flex", padding: "6px 0", borderTop: "1px solid var(--ll-border)", fontSize: 13 }}>
      <span style={{ width: 200, color: "var(--ll-text-muted)" }}>{k}</span>
      <code style={{ ...inlineCode, background: "transparent", padding: 0 }}>{v}</code>
    </div>
  );
}

function ProfilesSection({
  profiles,
  selected,
  onSelect,
  onApply,
  status,
  dirty,
}: {
  profiles: ConfigProfile[];
  selected: string;
  onSelect: (name: string) => void;
  onApply: () => void;
  status: SaveStatus;
  dirty: boolean;
}) {
  // Preview the affected sections for the chosen profile so the user
  // can see *what* "Apply lan" would do before clicking. Empty when
  // nothing is selected.
  const previewed = profiles.find((p) => p.name === selected) ?? null;

  return (
    <section style={card}>
      <SectionHeader title="Profiles" />
      <p style={{ fontSize: 12, color: "var(--ll-text-muted)", margin: "0 0 12px 0", lineHeight: 1.5 }}>
        Quick-switch between named <code style={inlineCode}>[[profiles]]</code>{" "}
        in config.toml. Applying merges the profile's sparse overrides into the
        on-disk config; the watcher live-applies what it can and surfaces a
        reload/restart hint for the rest.
      </p>
      {status.kind === "saved" && (
        <div style={status.requiresRestart || status.requiresReload ? infoBoxWarn : infoBoxOk}>
          Profile applied.{" "}
          {status.requiresRestart
            ? "Server fields changed — restart rustllama."
            : status.requiresReload
              ? "Model / inference fields changed — reload the model from the Models page."
              : "Changes hot-applied via the watcher."}
        </div>
      )}
      {status.kind === "error" && (
        <div style={errBox}>profile apply failed: {status.message}</div>
      )}

      {profiles.length === 0 ? (
        <div style={{ color: "var(--ll-text-muted)", fontSize: 12 }}>
          No profiles defined. Add <code style={inlineCode}>[[profiles]]</code>{" "}
          blocks to <code style={inlineCode}>config.toml</code> to enable quick-
          switching — e.g. one profile per task ("coding", "chat", "lan").
        </div>
      ) : (
        <>
          <Field label="Profile" hint="Sparse overrides — only sections defined in the profile change.">
            <select
              style={inputStyle}
              value={selected}
              onChange={(e) => onSelect(e.target.value)}
            >
              <option value="">— select —</option>
              {profiles.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.name}
                </option>
              ))}
            </select>
          </Field>
          {previewed && <ProfilePreview profile={previewed} />}
          <div style={{ marginTop: 12, display: "flex", gap: 8, alignItems: "center" }}>
            <button
              style={
                selected && !dirty && status.kind !== "saving"
                  ? btnPrimary
                  : btnPrimaryDisabled
              }
              disabled={!selected || dirty || status.kind === "saving"}
              onClick={onApply}
            >
              {status.kind === "saving" ? "applying…" : "Apply profile"}
            </button>
            {dirty && (
              <span style={{ fontSize: 11, color: "var(--ll-yellow)" }}>
                Discard or save current edits first.
              </span>
            )}
          </div>
        </>
      )}
    </section>
  );
}

function ProfilePreview({ profile }: { profile: ConfigProfile }) {
  // Build a flat list of overridden field labels so the user sees a
  // concrete preview before applying. Keys that aren't set in the
  // profile don't appear — that's the whole point of "sparse".
  const rows: { section: string; key: string; value: string }[] = [];
  if (profile.model) {
    if (profile.model.path) rows.push({ section: "model", key: "path", value: profile.model.path });
    if (profile.model.hub) rows.push({ section: "model", key: "hub", value: profile.model.hub });
    if (profile.model.chat_template)
      rows.push({ section: "model", key: "chat_template", value: profile.model.chat_template });
  }
  if (profile.inference) {
    const i = profile.inference;
    if (i.n_gpu_layers !== undefined)
      rows.push({ section: "inference", key: "n_gpu_layers", value: String(i.n_gpu_layers) });
    if (i.ctx_size !== undefined)
      rows.push({ section: "inference", key: "ctx_size", value: String(i.ctx_size) });
    if (i.batch_size !== undefined)
      rows.push({ section: "inference", key: "batch_size", value: String(i.batch_size) });
    if (i.threads !== undefined)
      rows.push({ section: "inference", key: "threads", value: String(i.threads) });
    if (i.kv_dtype) rows.push({ section: "inference", key: "kv_dtype", value: i.kv_dtype });
  }
  if (profile.server) {
    const s = profile.server;
    if (s.bind_addr) rows.push({ section: "server", key: "bind_addr", value: s.bind_addr });
    if (s.port !== undefined) rows.push({ section: "server", key: "port", value: String(s.port) });
    if (s.max_pending_per_model !== undefined)
      rows.push({
        section: "server",
        key: "max_pending_per_model",
        value: String(s.max_pending_per_model),
      });
    if (s.max_loaded_models !== undefined)
      rows.push({
        section: "server",
        key: "max_loaded_models",
        value: String(s.max_loaded_models),
      });
    if (s.concurrency !== undefined)
      rows.push({ section: "server", key: "concurrency", value: String(s.concurrency) });
  }

  if (rows.length === 0) {
    return (
      <div style={{ fontSize: 12, color: "var(--ll-text-muted)", padding: "8px 0" }}>
        (this profile defines no overrides — applying it is a no-op)
      </div>
    );
  }

  return (
    <div style={{ marginTop: 8 }}>
      <div
        style={{
          fontSize: 10,
          color: "var(--ll-text-muted)",
          textTransform: "uppercase",
          letterSpacing: 0.3,
          marginBottom: 6,
        }}
      >
        Overrides
      </div>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.section}.${r.key}`} style={{ borderTop: "1px solid var(--ll-border)" }}>
              <td style={{ padding: "4px 6px", color: "var(--ll-text-muted)", width: 90 }}>
                {r.section}
              </td>
              <td style={{ padding: "4px 6px", width: 180 }}>
                <code style={inlineCode}>{r.key}</code>
              </td>
              <td style={{ padding: "4px 6px" }}>
                <code style={inlineCode}>{r.value}</code>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SectionHeader({ title, hint }: { title: string; hint?: "live" | "reload" | "restart" }) {
  const badge = hint === "live" ? badgeLive : hint === "reload" ? badgeReload : hint === "restart" ? badgeRestart : null;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
      <h3 style={{ margin: 0, fontSize: 14, fontWeight: 600 }}>{title}</h3>
      {badge && <span style={badge}>{hint}</span>}
    </div>
  );
}

function AuditLogTailPanel() {
  const [data, setData] = useState<AuditTailResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = async () => {
    setLoading(true);
    try {
      setData(await getAuditLogTail(50));
      setErr(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
    // Poll less often than the metrics panels — audit entries
    // arrive on user actions, not continuously. 15s is enough to
    // catch a request without burning fetch bandwidth.
    const id = setInterval(refresh, 15_000);
    return () => clearInterval(id);
  }, []);

  return (
    <section style={{ ...card, marginTop: 16 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          marginBottom: 8,
        }}
      >
        <SectionHeader title="Recent audit entries" />
        <button onClick={refresh} disabled={loading} style={btnSecondary}>
          {loading ? "loading…" : "Refresh"}
        </button>
      </div>
      <p
        style={{
          fontSize: 12,
          color: "var(--ll-text-muted)",
          margin: "0 0 12px 0",
          lineHeight: 1.5,
        }}
      >
        Most recent first. The audit-log toggle above gates new writes;
        this panel reads the file regardless so you can review history
        from previously-enabled sessions. The redaction guarantees from
        the writer apply here: request bodies and header values are
        never recorded, and common credential-shaped query parameters
        (<code style={inlineCode}>api_key</code>,{" "}
        <code style={inlineCode}>access_token</code>,{" "}
        <code style={inlineCode}>password</code>,{" "}
        <code style={inlineCode}>signature</code>, and friends — full list in{" "}
        <code style={inlineCode}>redact_query_string</code>) are masked as{" "}
        <code style={inlineCode}>***</code>.
      </p>
      {err && <div style={errBox}>{err}</div>}
      {data && !data.path && (
        <div style={{ color: "var(--ll-text-muted)", fontSize: 12 }}>
          No audit log file on disk yet. Enable the toggle above + make a
          request to populate.
        </div>
      )}
      {data && data.path && data.entries.length === 0 && (
        <div style={{ color: "var(--ll-text-muted)", fontSize: 12 }}>
          File exists but is empty: <code style={inlineCode}>{data.path}</code>
        </div>
      )}
      {data && data.entries.length > 0 && (
        <>
          <div style={{ fontSize: 11, color: "var(--ll-text-muted)", marginBottom: 6 }}>
            Showing {data.entries.length} of {data.total_entries} total · file:{" "}
            <code style={{ ...inlineCode, fontSize: 10 }}>{data.path}</code>
          </div>
          <div
            style={{
              maxHeight: 320,
              overflowY: "auto",
              border: "1px solid var(--ll-border)",
              borderRadius: 4,
            }}
          >
            <table
              style={{
                width: "100%",
                borderCollapse: "collapse",
                fontSize: 11,
              }}
            >
              <thead
                style={{
                  position: "sticky",
                  top: 0,
                  background: "var(--ll-bg-elev)",
                }}
              >
                <tr style={{ color: "var(--ll-text-muted)", textAlign: "left" }}>
                  <th style={{ padding: "4px 6px", fontWeight: 500 }}>When</th>
                  <th style={{ padding: "4px 6px", fontWeight: 500 }}>
                    Method
                  </th>
                  <th style={{ padding: "4px 6px", fontWeight: 500 }}>Path</th>
                  <th style={{ padding: "4px 6px", fontWeight: 500 }}>
                    Status
                  </th>
                  <th
                    style={{
                      padding: "4px 6px",
                      fontWeight: 500,
                      textAlign: "right",
                    }}
                  >
                    Latency
                  </th>
                </tr>
              </thead>
              <tbody>
                {data.entries.map((e, i) => (
                  <tr
                    key={i}
                    style={{ borderTop: "1px solid var(--ll-border)" }}
                  >
                    <td
                      style={{
                        padding: "4px 6px",
                        whiteSpace: "nowrap",
                        color: "var(--ll-text-muted)",
                      }}
                    >
                      {new Date(e.ts_ms).toLocaleTimeString()}
                    </td>
                    <td style={{ padding: "4px 6px" }}>
                      <code style={inlineCode}>{e.method}</code>
                    </td>
                    <td style={{ padding: "4px 6px", wordBreak: "break-all" }}>
                      <code style={{ ...inlineCode, fontSize: 10 }}>
                        {e.path}
                        {e.query ? `?${e.query}` : ""}
                      </code>
                    </td>
                    <td
                      style={{
                        padding: "4px 6px",
                        color:
                          e.status >= 500
                            ? "var(--ll-red)"
                            : e.status >= 400
                              ? "var(--ll-yellow)"
                              : "var(--ll-text)",
                      }}
                    >
                      {e.status}
                    </td>
                    <td
                      style={{
                        padding: "4px 6px",
                        textAlign: "right",
                        color: "var(--ll-text-muted)",
                      }}
                    >
                      {e.latency_ms} ms
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}

function CrashLogsPanel() {
  const [data, setData] = useState<CrashLogsList | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [selected, setSelected] = useState<CrashLogEntry | null>(null);
  const [selectedBody, setSelectedBody] = useState<string | null>(null);
  const [selectedErr, setSelectedErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const refresh = async () => {
    setLoading(true);
    try {
      setData(await listCrashLogs());
      setErr(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const view = async (e: CrashLogEntry) => {
    setSelected(e);
    setSelectedBody(null);
    setSelectedErr(null);
    try {
      setSelectedBody(await getCrashLog(e.name));
    } catch (err) {
      setSelectedErr(err instanceof Error ? err.message : String(err));
    }
  };

  const remove = async (e: CrashLogEntry) => {
    if (
      !confirm(
        `Delete ${e.name}? The file lives at ${e.path} and will be removed from disk.`,
      )
    )
      return;
    try {
      await deleteCrashLog(e.name);
      if (selected?.name === e.name) {
        setSelected(null);
        setSelectedBody(null);
      }
      await refresh();
    } catch (err) {
      setErr(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <section style={{ ...card, marginTop: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <SectionHeader title="Crash logs" />
        <button onClick={refresh} disabled={loading} style={btnSecondary}>
          {loading ? "loading…" : "Refresh"}
        </button>
      </div>
      <p style={{ fontSize: 12, color: "var(--ll-text-muted)", margin: "0 0 12px 0", lineHeight: 1.5 }}>
        The runtime panic hook drops a file in{" "}
        <code style={inlineCode}>{data?.dir ?? "(unknown)"}</code> on every
        panic, with version, command line, panic message + backtrace.
        Newest first; older files stay on disk until you delete them.
      </p>
      {err && <div style={errBox}>{err}</div>}
      {data && data.entries.length === 0 && !err && (
        <div style={{ color: "var(--ll-text-muted)", fontSize: 12 }}>
          No crash logs — the server has never panicked. (Or the crash log
          dir doesn't exist yet.)
        </div>
      )}
      {data && data.entries.length > 0 && (
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr style={{ color: "var(--ll-text-muted)", textAlign: "left" }}>
              <th style={{ padding: "6px 4px", fontWeight: 500 }}>When</th>
              <th style={{ padding: "6px 4px", fontWeight: 500 }}>File</th>
              <th style={{ padding: "6px 4px", fontWeight: 500 }}>Size</th>
              <th style={{ padding: "6px 4px" }} />
            </tr>
          </thead>
          <tbody>
            {data.entries.map((e) => (
              <tr key={e.name} style={{ borderTop: "1px solid var(--ll-border)" }}>
                <td style={{ padding: "6px 4px" }}>
                  {new Date(e.epoch_secs * 1000).toLocaleString()}
                </td>
                <td style={{ padding: "6px 4px" }}>
                  <code style={{ ...inlineCode, fontSize: 11 }}>{e.name}</code>
                </td>
                <td style={{ padding: "6px 4px" }}>
                  {e.size_bytes < 1024
                    ? `${e.size_bytes} B`
                    : `${(e.size_bytes / 1024).toFixed(1)} KiB`}
                </td>
                <td style={{ padding: "6px 4px", textAlign: "right", whiteSpace: "nowrap" }}>
                  <button
                    onClick={() => view(e)}
                    style={{ ...btnSecondary, marginRight: 4 }}
                  >
                    View
                  </button>
                  <button
                    onClick={() => remove(e)}
                    style={btnDanger}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {selected && (
        <div
          style={{
            marginTop: 12,
            border: "1px solid var(--ll-border-strong)",
            borderRadius: 4,
            overflow: "hidden",
          }}
        >
          <div
            style={{
              padding: "6px 10px",
              background: "var(--ll-bg-elev)",
              borderBottom: "1px solid var(--ll-border-strong)",
              fontSize: 11,
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <code style={{ ...inlineCode, fontSize: 11 }}>{selected.name}</code>
            <button
              onClick={() => {
                setSelected(null);
                setSelectedBody(null);
              }}
              style={btnGhost}
            >
              ✕
            </button>
          </div>
          {selectedErr ? (
            <div style={errBox}>{selectedErr}</div>
          ) : selectedBody === null ? (
            <div style={{ padding: 10, fontSize: 11, color: "var(--ll-text-muted)" }}>
              loading…
            </div>
          ) : (
            <pre style={crashLogPre}>{selectedBody}</pre>
          )}
        </div>
      )}
    </section>
  );
}

function LanAccessPanel({
  bindAddrDraft,
  apiKeyDraft,
  dirty,
}: {
  /// Current value in the bind_addr field — used to warn the user
  /// when the saved-on-disk bind_addr (which `/v1/lan_info` reads)
  /// doesn't match what the user is editing. The QR encodes the
  /// SAVED bind_addr, not the draft, so editing without saving
  /// shouldn't silently mislead.
  bindAddrDraft: string;
  apiKeyDraft: string;
  dirty: boolean;
}) {
  const [info, setInfo] = useState<LanInfoResult | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [copied, setCopied] = useState<"url" | "key" | null>(null);

  const refresh = async () => {
    try {
      setInfo(await getLanInfo());
      setErr(null);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const copy = async (text: string, which: "url" | "key") => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(which);
      setTimeout(() => setCopied(null), 1500);
    } catch {
      // Some webviews don't grant clipboard permission. Silently
      // ignore — the text is still visible for manual copy.
    }
  };

  if (err) {
    return (
      <section style={{ ...card, marginTop: 16 }}>
        <SectionHeader title="LAN access" />
        <div style={errBox}>{err}</div>
      </section>
    );
  }
  if (!info) {
    return (
      <section style={{ ...card, marginTop: 16 }}>
        <SectionHeader title="LAN access" />
        <div style={{ color: "var(--ll-text-muted)", fontSize: 13 }}>discovering…</div>
      </section>
    );
  }

  const savedBindIsLoopback =
    info.bind_addr.startsWith("127.") || info.bind_addr === "::1";
  const draftMismatch = dirty && info.bind_addr !== bindAddrDraft;
  // Open server on the LAN without auth is the security trap.
  // Surface it loudly when the saved config matches that combo —
  // the user's intent is "broadcast", so this is the moment to
  // recommend setting an api_key.
  const openOnLan = !savedBindIsLoopback && !info.api_key_set;

  return (
    <section style={{ ...card, marginTop: 16 }}>
      <SectionHeader title="LAN access" />
      <p style={{ fontSize: 12, color: "var(--ll-text-muted)", margin: "0 0 12px 0", lineHeight: 1.5 }}>
        Reflects the SAVED <code style={inlineCode}>bind_addr</code> /
        <code style={inlineCode}> port</code> /
        <code style={inlineCode}> api_key</code> on disk. The QR encodes that
        URL — another device on your network can scan it and reach this
        server.
      </p>

      {draftMismatch && (
        <div style={infoBoxWarn}>
          You've edited the Server section but haven't saved yet — the QR
          below still reflects what's on disk
          (<code style={inlineCode}>{info.bind_addr}</code>), not what you're
          typing
          (<code style={inlineCode}>{bindAddrDraft}</code>). Save changes
          and the QR will refresh.
        </div>
      )}

      {savedBindIsLoopback ? (
        <div style={{ fontSize: 13, color: "var(--ll-text)", lineHeight: 1.6 }}>
          <p style={{ margin: "0 0 6px 0" }}>
            Local-only — bound to{" "}
            <code style={inlineCode}>{info.bind_addr}:{info.port}</code>.
          </p>
          <p style={{ margin: 0, color: "var(--ll-text-muted)", fontSize: 12 }}>
            To enable LAN access, change <code style={inlineCode}>bind_addr</code>{" "}
            above to <code style={inlineCode}>0.0.0.0</code>, save, then
            restart rustllama.
          </p>
        </div>
      ) : info.url ? (
        <div>
          {openOnLan && (
            <div style={{ ...infoBoxWarn, marginBottom: 12 }}>
              <strong>Open server on the LAN.</strong> Anyone on this network
              can reach <code style={inlineCode}>{info.url}</code> and use this
              model. Set an <code style={inlineCode}>api_key</code> in the
              Server section above to require auth.
              {apiKeyDraft.length > 0 && dirty && " (You're typing one — save to apply.)"}
            </div>
          )}
          <div style={{ display: "flex", gap: 20, alignItems: "flex-start", flexWrap: "wrap" }}>
            {info.qr_svg && (
              <div
                // qrcode-rs SVG output is `<svg>` markup with
                // `<path>`s and `<rect>`s — no script tags, deterministic
                // structure, safe to inline. The Tauri webview's CSP
                // also blocks inline scripts so a future regression
                // would be defanged at the boundary.
                dangerouslySetInnerHTML={{ __html: info.qr_svg }}
                style={{ background: "white", padding: 8, borderRadius: 4 }}
              />
            )}
            <div style={{ flex: 1, minWidth: 200 }}>
              <div style={inspectGridRow}>
                <Label text="URL" />
                <ValueRow>
                  <code style={{ ...inlineCode, wordBreak: "break-all" }}>
                    {info.url}
                  </code>
                  <button
                    onClick={() => copy(info.url!, "url")}
                    style={btnSmall}
                  >
                    {copied === "url" ? "✓ copied" : "Copy"}
                  </button>
                </ValueRow>
              </div>
              <div style={inspectGridRow}>
                <Label text="LAN IP" />
                <ValueRow>
                  <code style={inlineCode}>
                    {info.primary_lan_ip ?? "—"}
                  </code>
                </ValueRow>
              </div>
              <div style={inspectGridRow}>
                <Label text="Auth" />
                <ValueRow>
                  {info.api_key_set ? (
                    <>
                      <span style={badgeLive}>required</span>
                      {apiKeyDraft && (
                        <button
                          onClick={() => copy(apiKeyDraft, "key")}
                          style={btnSmall}
                          title="Copies the api_key from your current edits — not necessarily what's on disk"
                        >
                          {copied === "key" ? "✓ copied" : "Copy api_key"}
                        </button>
                      )}
                    </>
                  ) : (
                    <span style={badgeRestart}>none</span>
                  )}
                </ValueRow>
              </div>
              <button onClick={refresh} style={{ ...btnSecondary, marginTop: 8 }}>
                Refresh
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div style={{ fontSize: 13, color: "var(--ll-text)", lineHeight: 1.6 }}>
          Bound to <code style={inlineCode}>{info.bind_addr}</code> but no
          routable LAN interface was discovered. If this is wrong, check that
          the network is up and try Refresh.
          <div style={{ marginTop: 8 }}>
            <button onClick={refresh} style={btnSecondary}>
              Refresh
            </button>
          </div>
        </div>
      )}
    </section>
  );
}

function Label({ text }: { text: string }) {
  return (
    <span
      style={{
        fontSize: 10,
        color: "var(--ll-text-muted)",
        textTransform: "uppercase",
        letterSpacing: 0.3,
        width: 90,
        flexShrink: 0,
      }}
    >
      {text}
    </span>
  );
}

function ValueRow({ children }: { children: React.ReactNode }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
      {children}
    </span>
  );
}

const inspectGridRow: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 8,
  padding: "6px 0",
  borderTop: "1px solid var(--ll-border)",
  fontSize: 13,
};
const btnSmall: React.CSSProperties = {
  padding: "2px 8px",
  background: "transparent",
  color: "var(--ll-text)",
  border: "1px solid var(--ll-border-strong)",
  borderRadius: 4,
  cursor: "pointer",
  fontSize: 11,
};

/// Reserved chat_template values that aren't Jinja — these are
/// short aliases the engine resolves from elsewhere (GGUF metadata
/// for `"auto"`; future built-in templates for `"chatml"` /
/// `"llama3"`). The preview panel can't render those usefully so
/// we suppress it and show a one-line hint instead.
const NON_JINJA_TEMPLATE_ALIASES = ["", "auto", "chatml", "llama3"];

/// Sample conversation used by the live preview. Three turns covers
/// the common branches in real-world templates: system message
/// (some templates inject defaults when absent), user, assistant.
/// Hard-coded rather than user-editable to keep this an "S (v1.x)"
/// add-on — the value is in seeing what the *template* does, not
/// what the messages do.
const PREVIEW_SAMPLE_MESSAGES: { role: string; content: string }[] = [
  { role: "system", content: "You are a helpful assistant." },
  { role: "user", content: "What is 2 + 2?" },
  { role: "assistant", content: "4." },
];

function ChatTemplatePreview({ template }: { template: string }) {
  const trimmed = template.trim();
  const isAlias = NON_JINJA_TEMPLATE_ALIASES.includes(trimmed);
  const [rendered, setRendered] = useState<string | null>(null);
  const [usedSpecials, setUsedSpecials] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isAlias) {
      setRendered(null);
      setErr(null);
      setLoading(false);
      return;
    }
    const ctrl = new AbortController();
    setLoading(true);
    const timer = setTimeout(async () => {
      try {
        const r = await previewChatTemplate({
          template,
          messages: PREVIEW_SAMPLE_MESSAGES,
          addGenerationPrompt: true,
          signal: ctrl.signal,
        });
        if (ctrl.signal.aborted) return;
        setRendered(r.rendered);
        setUsedSpecials(r.used_engine_specials);
        setErr(null);
      } catch (e) {
        if (ctrl.signal.aborted) return;
        setRendered(null);
        setErr(e instanceof Error ? e.message : String(e));
      } finally {
        if (!ctrl.signal.aborted) setLoading(false);
      }
    }, 300);
    return () => {
      ctrl.abort();
      clearTimeout(timer);
    };
  }, [template, isAlias]);

  return (
    <div style={{ padding: "8px 0 0 212px" }}>
      <div
        style={{
          fontSize: 10,
          color: "var(--ll-text-muted)",
          textTransform: "uppercase",
          letterSpacing: 0.3,
          marginBottom: 4,
        }}
      >
        Preview {loading && "· rendering…"}
      </div>
      {isAlias ? (
        <div style={{ fontSize: 11, color: "var(--ll-text-faint)" }}>
          {trimmed === "" || trimmed === "auto"
            ? "Preview is hidden for `auto` — the engine reads the embedded template from the GGUF at load time."
            : `Preview is hidden for built-in alias \`${trimmed}\` — paste an inline Jinja template to see live output.`}
        </div>
      ) : err ? (
        <div style={previewErrBox}>{err}</div>
      ) : rendered === null ? (
        <div style={{ fontSize: 11, color: "var(--ll-text-faint)" }}>—</div>
      ) : (
        <>
          <pre style={previewPre}>{rendered.length > 0 ? rendered : "(empty render)"}</pre>
          <div style={{ fontSize: 10, color: "var(--ll-text-faint)", marginTop: 4 }}>
            Rendered against a 3-turn sample (system / user / assistant).{" "}
            {usedSpecials
              ? "BOS/EOS substituted from the active model's tokenizer."
              : "No model loaded — BOS/EOS placeholders render as empty."}
          </div>
        </>
      )}
    </div>
  );
}

// ----- styles -----

const card: React.CSSProperties = {
  background: "var(--ll-bg-elev)",
  border: "1px solid var(--ll-border)",
  borderRadius: 6,
  padding: 16,
};
const inputStyle: React.CSSProperties = {
  width: "100%",
  background: "var(--ll-bg)",
  color: "var(--ll-text)",
  border: "1px solid var(--ll-border-strong)",
  borderRadius: 4,
  padding: "4px 8px",
  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
  fontSize: 13,
  boxSizing: "border-box",
};
const inlineCode: React.CSSProperties = {
  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
  background: "var(--ll-bg)",
  padding: "1px 6px",
  borderRadius: 3,
  fontSize: 12,
};
const errBox: React.CSSProperties = {
  padding: 10,
  background: "var(--ll-red-soft)",
  border: "1px solid var(--ll-red)",
  borderRadius: 4,
  color: "var(--ll-red)",
  fontSize: 13,
  marginBottom: 16,
};
const infoBoxOk: React.CSSProperties = {
  padding: 10,
  background: "var(--ll-green-soft)",
  border: "1px solid var(--ll-green)",
  borderRadius: 4,
  color: "var(--ll-green)",
  fontSize: 13,
  marginBottom: 16,
};
const infoBoxWarn: React.CSSProperties = {
  padding: 10,
  background: "var(--ll-yellow-soft)",
  border: "1px solid var(--ll-yellow)",
  borderRadius: 4,
  color: "var(--ll-yellow)",
  fontSize: 13,
  marginBottom: 16,
};
const badgeBase: React.CSSProperties = {
  display: "inline-block",
  padding: "1px 8px",
  borderRadius: 10,
  fontSize: 10,
  fontWeight: 600,
  textTransform: "uppercase",
  letterSpacing: 0.5,
};
const badgeLive: React.CSSProperties = { ...badgeBase, background: "var(--ll-green-soft)", color: "var(--ll-green)", border: "1px solid var(--ll-green)" };
const badgeReload: React.CSSProperties = { ...badgeBase, background: "var(--ll-info-soft)", color: "var(--ll-accent)", border: "1px solid var(--ll-accent)" };
const badgeRestart: React.CSSProperties = { ...badgeBase, background: "var(--ll-yellow-soft)", color: "var(--ll-yellow)", border: "1px solid var(--ll-yellow)" };
const bottomBar: React.CSSProperties = {
  position: "fixed",
  bottom: 0,
  left: 0,
  right: 0,
  padding: "12px 20px",
  background: "var(--ll-bg)",
  borderTop: "1px solid var(--ll-border)",
  display: "flex",
  gap: 8,
  zIndex: 10,
};
const btnPrimary: React.CSSProperties = {
  background: "var(--ll-green)",
  color: "white",
  border: "1px solid var(--ll-green)",
  padding: "6px 16px",
  borderRadius: 4,
  fontSize: 13,
  cursor: "pointer",
};
const btnPrimaryDisabled: React.CSSProperties = { ...btnPrimary, opacity: 0.4, cursor: "not-allowed" };
const btnSecondary: React.CSSProperties = {
  background: "var(--ll-border)",
  color: "var(--ll-text)",
  border: "1px solid var(--ll-border-strong)",
  padding: "6px 16px",
  borderRadius: 4,
  fontSize: 13,
  cursor: "pointer",
};
const btnSecondaryDisabled: React.CSSProperties = { ...btnSecondary, opacity: 0.4, cursor: "not-allowed" };
const btnGhost: React.CSSProperties = {
  background: "transparent",
  color: "var(--ll-text-muted)",
  border: "1px solid transparent",
  padding: "2px 8px",
  borderRadius: 4,
  fontSize: 12,
  cursor: "pointer",
};
const btnDanger: React.CSSProperties = {
  padding: "4px 10px",
  background: "transparent",
  color: "var(--ll-red)",
  border: "1px solid var(--ll-red)",
  borderRadius: 4,
  cursor: "pointer",
  fontSize: 12,
};
const crashLogPre: React.CSSProperties = {
  margin: 0,
  padding: "10px 12px",
  background: "var(--ll-bg)",
  color: "var(--ll-text)",
  fontSize: 11,
  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
  whiteSpace: "pre-wrap",
  wordBreak: "break-word",
  maxHeight: 360,
  overflowY: "auto",
};
const previewPre: React.CSSProperties = {
  margin: 0,
  padding: "10px 12px",
  background: "var(--ll-bg)",
  border: "1px solid var(--ll-border-strong)",
  borderRadius: 4,
  color: "var(--ll-text)",
  fontSize: 11,
  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
  whiteSpace: "pre-wrap",
  wordBreak: "break-word",
  maxHeight: 280,
  overflowY: "auto",
};
const previewErrBox: React.CSSProperties = {
  padding: "8px 10px",
  background: "rgba(248, 81, 73, 0.08)",
  border: "1px solid rgba(248, 81, 73, 0.4)",
  borderRadius: 4,
  color: "var(--ll-red)",
  fontSize: 11,
  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
  whiteSpace: "pre-wrap",
};
