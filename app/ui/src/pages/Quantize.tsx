// Quantize page: re-encode a cached GGUF file to a smaller target
// dtype. Wraps the `quantize_model` Tauri command (which drives
// `rustllama_gguf::quantize::quantize_gguf` end-to-end). MVP UI —
// path text inputs (no file picker yet), target dropdown, optional
// APEX tier dropdown, optional recipe-file path, "Run" button, and
// a results panel.
//
// Quantize is CPU-bound and can take minutes on a real model; the
// page disables the Run button + shows an indeterminate spinner
// while the IPC call is in flight. Per-tensor progress events
// (`quantize://progress`) are a follow-up — v1 reports completion
// stats only.

import { invoke } from "@tauri-apps/api/core";
import { useState } from "react";

// Targets the v1 pipeline supports. Order matches the bpw curve
// (lowest → highest) so the dropdown reads top-to-bottom as
// smallest-to-biggest output.
const TARGETS: { name: string; label: string; bpw: string }[] = [
  { name: "iq1_s", label: "IQ1_S", bpw: "1.56 bpw" },
  { name: "iq1_m", label: "IQ1_M", bpw: "1.75 bpw" },
  { name: "tq1_0", label: "TQ1_0", bpw: "1.69 bpw" },
  { name: "tq2_0", label: "TQ2_0", bpw: "2.0 bpw" },
  { name: "iq2_xxs", label: "IQ2_XXS", bpw: "2.06 bpw" },
  { name: "iq2_xs", label: "IQ2_XS", bpw: "2.31 bpw" },
  { name: "iq2_s", label: "IQ2_S", bpw: "2.56 bpw" },
  { name: "q2_k", label: "Q2_K", bpw: "2.625 bpw" },
  { name: "iq3_xxs", label: "IQ3_XXS", bpw: "3.06 bpw" },
  { name: "iq3_s", label: "IQ3_S", bpw: "3.44 bpw" },
  { name: "q3_k", label: "Q3_K", bpw: "3.44 bpw" },
  { name: "iq4_nl", label: "IQ4_NL", bpw: "4.5 bpw" },
  { name: "iq4_xs", label: "IQ4_XS", bpw: "4.25 bpw" },
  { name: "q4_0", label: "Q4_0", bpw: "4.5 bpw" },
  { name: "q4_1", label: "Q4_1", bpw: "5.0 bpw" },
  { name: "q4_k", label: "Q4_K (recommended)", bpw: "4.5 bpw" },
  { name: "q5_0", label: "Q5_0", bpw: "5.5 bpw" },
  { name: "q5_1", label: "Q5_1", bpw: "6.0 bpw" },
  { name: "q5_k", label: "Q5_K", bpw: "5.5 bpw" },
  { name: "q6_k", label: "Q6_K", bpw: "6.5 bpw" },
  { name: "q8_0", label: "Q8_0", bpw: "8.5 bpw" },
  { name: "q8_1", label: "Q8_1", bpw: "9.0 bpw" },
  { name: "q8_k", label: "Q8_K", bpw: "9.125 bpw" },
  { name: "bf16", label: "BF16", bpw: "16 bpw" },
  { name: "f16", label: "F16", bpw: "16 bpw" },
  { name: "f32", label: "F32 (no quantization)", bpw: "32 bpw" },
];

const APEX_TIERS: { name: string; label: string; desc: string }[] = [
  { name: "", label: "(none)", desc: "Uniform target across all tensors" },
  { name: "i-quality", label: "I-Quality", desc: "Best perplexity; routed Q4_K/Q6_K, shared Q8_0, attn Q6_K" },
  { name: "quality", label: "Quality", desc: "Routed Q3_K/Q5_K, shared Q8_0, attn Q6_K" },
  { name: "balanced", label: "Balanced", desc: "Routed Q3_K/Q4_K, shared Q6_K, attn Q5_K" },
  { name: "mini", label: "Mini", desc: "Routed Q2_K/Q4_K, shared Q5_K, attn Q5_K" },
  { name: "nano", label: "Nano", desc: "Most aggressive: routed Q2_K, shared Q4_K, attn Q4_K" },
];

interface QuantizeResult {
  tensors_total: number;
  tensors_requantized: number;
  tensors_passthrough: number;
  bytes_in: number;
  bytes_out: number;
  elapsed_ms: number;
  target: string;
  n_layers: number;
}

function formatBytes(n: number): string {
  const mb = n / (1024 * 1024);
  if (mb < 1024) return `${mb.toFixed(1)} MiB`;
  return `${(mb / 1024).toFixed(2)} GiB`;
}

export default function QuantizePage() {
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");
  const [target, setTarget] = useState("q4_k");
  const [apex, setApex] = useState("");
  const [recipe, setRecipe] = useState("");
  const [keepOutput, setKeepOutput] = useState(true);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QuantizeResult | null>(null);

  const onRun = async () => {
    if (!input.trim() || !output.trim()) {
      setError("Both input and output paths are required.");
      return;
    }
    setError(null);
    setResult(null);
    setRunning(true);
    try {
      const res = await invoke<QuantizeResult>("quantize_model", {
        input: input.trim(),
        output: output.trim(),
        target,
        apex: apex.trim() === "" ? null : apex.trim(),
        recipe: recipe.trim() === "" ? null : recipe.trim(),
        keepOutput,
      });
      setResult(res);
    } catch (e) {
      setError(String(e));
    } finally {
      setRunning(false);
    }
  };

  const inputStyle: React.CSSProperties = {
    width: "100%",
    background: "var(--ll-bg)",
    color: "var(--ll-text)",
    border: "1px solid var(--ll-border-strong)",
    borderRadius: 6,
    padding: "8px 10px",
    fontSize: 13,
    fontFamily:
      "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
  };

  return (
    <div style={{ padding: 24, maxWidth: 880 }}>
      <h2 style={{ margin: "0 0 16px 0", color: "var(--ll-text)", fontSize: 18 }}>
        Quantize
      </h2>
      <p style={{ color: "var(--ll-text-muted)", fontSize: 13, marginBottom: 20, lineHeight: 1.5 }}>
        Re-encode a GGUF model to a smaller target dtype. Source can be
        any supported quant (F32, F16, BF16, Q4_0/1, Q5_0/1, Q8_0, Q2/3/4/5/6/8_K,
        TQ1/2, IQ1–IQ4); output spans the same set. Norms and biases are
        passed through at source dtype automatically.
      </p>

      <fieldset
        style={{
          border: "1px solid var(--ll-border-strong)",
          borderRadius: 8,
          padding: 16,
          marginBottom: 20,
        }}
      >
        <legend style={{ padding: "0 8px", color: "var(--ll-text)", fontSize: 13 }}>
          Paths
        </legend>

        <label style={{ display: "block", marginBottom: 12 }}>
          <div style={{ color: "var(--ll-text-muted)", fontSize: 12, marginBottom: 4 }}>
            Source GGUF
          </div>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="C:\Users\you\.cache\rustllama\models\foo.gguf"
            style={inputStyle}
          />
        </label>

        <label style={{ display: "block" }}>
          <div style={{ color: "var(--ll-text-muted)", fontSize: 12, marginBottom: 4 }}>
            Output GGUF (created or overwritten)
          </div>
          <input
            type="text"
            value={output}
            onChange={(e) => setOutput(e.target.value)}
            placeholder="C:\Users\you\.cache\rustllama\models\foo.Q4_K.gguf"
            style={inputStyle}
          />
        </label>
      </fieldset>

      <fieldset
        style={{
          border: "1px solid var(--ll-border-strong)",
          borderRadius: 8,
          padding: 16,
          marginBottom: 20,
        }}
      >
        <legend style={{ padding: "0 8px", color: "var(--ll-text)", fontSize: 13 }}>
          Target
        </legend>

        <label style={{ display: "block", marginBottom: 12 }}>
          <div style={{ color: "var(--ll-text-muted)", fontSize: 12, marginBottom: 4 }}>
            Default dtype (used for any tensor not matched by APEX or recipe)
          </div>
          <select
            value={target}
            onChange={(e) => setTarget(e.target.value)}
            style={inputStyle}
          >
            {TARGETS.map((t) => (
              <option key={t.name} value={t.name}>
                {t.label} — {t.bpw}
              </option>
            ))}
          </select>
        </label>

        <label style={{ display: "block", marginBottom: 12 }}>
          <div style={{ color: "var(--ll-text-muted)", fontSize: 12, marginBottom: 4 }}>
            APEX profile (mixed-precision for MoE models — first match wins)
          </div>
          <select
            value={apex}
            onChange={(e) => setApex(e.target.value)}
            style={inputStyle}
          >
            {APEX_TIERS.map((a) => (
              <option key={a.name} value={a.name}>
                {a.label} — {a.desc}
              </option>
            ))}
          </select>
        </label>

        <label style={{ display: "block", marginBottom: 12 }}>
          <div style={{ color: "var(--ll-text-muted)", fontSize: 12, marginBottom: 4 }}>
            Recipe file (optional; `{'<glob> <dtype>'}` per line)
          </div>
          <input
            type="text"
            value={recipe}
            onChange={(e) => setRecipe(e.target.value)}
            placeholder="(leave blank to skip)"
            style={inputStyle}
          />
        </label>

        <label
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            color: "var(--ll-text)",
            fontSize: 13,
          }}
        >
          <input
            type="checkbox"
            checked={keepOutput}
            onChange={(e) => setKeepOutput(e.target.checked)}
          />
          Keep LM head at source precision (recommended for &lt;4 bpw targets)
        </label>
      </fieldset>

      <button
        onClick={onRun}
        disabled={running || !input.trim() || !output.trim()}
        style={{
          background: running ? "var(--ll-border)" : "var(--ll-green)",
          color: "var(--ll-text)",
          border: "1px solid var(--ll-border-strong)",
          borderRadius: 6,
          padding: "10px 20px",
          fontSize: 14,
          fontWeight: 500,
          cursor: running ? "not-allowed" : "pointer",
        }}
      >
        {running ? "Running…" : "Run quantize"}
      </button>

      {error && (
        <div
          style={{
            marginTop: 20,
            padding: 12,
            background: "var(--ll-red-soft)",
            border: "1px solid var(--ll-red)",
            borderRadius: 6,
            color: "var(--ll-red)",
            fontSize: 13,
            fontFamily:
              "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
            whiteSpace: "pre-wrap",
          }}
        >
          {error}
        </div>
      )}

      {result && (
        <div
          style={{
            marginTop: 20,
            padding: 16,
            background: "var(--ll-bg)",
            border: "1px solid var(--ll-border-strong)",
            borderRadius: 8,
            color: "var(--ll-text)",
          }}
        >
          <h3 style={{ margin: "0 0 12px 0", color: "var(--ll-green)", fontSize: 14 }}>
            Quantize complete
          </h3>
          <div style={{ display: "grid", gridTemplateColumns: "180px 1fr", rowGap: 6, fontSize: 13 }}>
            <span style={{ color: "var(--ll-text-muted)" }}>Default target:</span>
            <span>{result.target}</span>
            <span style={{ color: "var(--ll-text-muted)" }}>Block layers detected:</span>
            <span>{result.n_layers}</span>
            <span style={{ color: "var(--ll-text-muted)" }}>Tensors re-encoded:</span>
            <span>{result.tensors_requantized} of {result.tensors_total}</span>
            <span style={{ color: "var(--ll-text-muted)" }}>Passthrough:</span>
            <span>{result.tensors_passthrough}</span>
            <span style={{ color: "var(--ll-text-muted)" }}>Source size:</span>
            <span>{formatBytes(result.bytes_in)}</span>
            <span style={{ color: "var(--ll-text-muted)" }}>Output size:</span>
            <span>
              {formatBytes(result.bytes_out)}{" "}
              <span style={{ color: "var(--ll-text-muted)" }}>
                ({((result.bytes_out / Math.max(result.bytes_in, 1)) * 100).toFixed(1)}% of original)
              </span>
            </span>
            <span style={{ color: "var(--ll-text-muted)" }}>Elapsed:</span>
            <span>{(result.elapsed_ms / 1000).toFixed(2)}s</span>
          </div>
        </div>
      )}
    </div>
  );
}
