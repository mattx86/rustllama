import { type CSSProperties, useState } from "react";
import { decide, type DecideResult } from "../api";

type Mode = "choice" | "score" | "boolean";

/// Typed-decision playground: pick Choice / Score / Boolean, give a
/// context + the candidate options (or a yes/no question), and see the
/// model's typed answer with a probability bar per option. Reuses the
/// server's `/v1/decide/*` endpoints (constrained scoring on the loaded
/// model). Shows a "calibrated" badge when a per-model temperature (fitted
/// by the tune sweep's decision-calibration stage) was applied.
export default function DecidePage() {
  const [mode, setMode] = useState<Mode>("choice");
  const [context, setContext] = useState(
    'Classify the sentiment.\nReview: "I absolutely love this, best purchase ever!"\nSentiment:',
  );
  const [optionsText, setOptionsText] = useState(" positive\n negative\n neutral");
  const [question, setQuestion] = useState("Is this review positive?");
  const [result, setResult] = useState<DecideResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const labels =
    mode === "boolean"
      ? ["yes", "no"]
      : optionsText
          .split("\n")
          .map((s) => s.trim())
          .filter((s) => s.length > 0);

  const run = async () => {
    setBusy(true);
    setErr(null);
    setResult(null);
    try {
      const body: Record<string, unknown> = { context };
      if (mode === "choice") body.options = labels;
      else if (mode === "score") body.levels = labels;
      else body.question = question;
      setResult(await decide(mode, body));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  // Probabilities to render as bars: choice/score come back as an array;
  // boolean is a single P(yes) we expand to yes/no.
  const bars: { label: string; p: number }[] = result
    ? mode === "boolean"
      ? [
          { label: "yes", p: result.probability ?? 0 },
          { label: "no", p: 1 - (result.probability ?? 0) },
        ]
      : (result.probabilities ?? []).map((p, i) => ({
          label: labels[i] ?? `#${i}`,
          p,
        }))
    : [];

  return (
    <div style={{ padding: 24, maxWidth: 760, margin: "0 auto" }}>
      <h2 style={{ margin: "0 0 4px 0", fontSize: 18 }}>Decide (typed decisions)</h2>
      <p style={{ color: "var(--ll-text-muted)", fontSize: 13, marginTop: 0 }}>
        Score candidate options against the loaded model and return a typed
        value + probabilities — no prose generation. Choice picks one, Score
        rates an ordered scale, Boolean is a calibrated yes/no.
      </p>

      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        {(["choice", "score", "boolean"] as Mode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            style={mode === m ? tabActive : tab}
          >
            {m[0].toUpperCase() + m.slice(1)}
          </button>
        ))}
      </div>

      <label style={lbl}>Context</label>
      <textarea
        value={context}
        onChange={(e) => setContext(e.target.value)}
        rows={5}
        style={{ ...field, fontFamily: "ui-monospace, monospace", resize: "vertical" }}
      />

      {mode === "boolean" ? (
        <>
          <label style={lbl}>Question (yes/no)</label>
          <input value={question} onChange={(e) => setQuestion(e.target.value)} style={field} />
        </>
      ) : (
        <>
          <label style={lbl}>{mode === "score" ? "Levels" : "Options"} (one per line)</label>
          <textarea
            value={optionsText}
            onChange={(e) => setOptionsText(e.target.value)}
            rows={4}
            style={{ ...field, fontFamily: "ui-monospace, monospace", resize: "vertical" }}
          />
        </>
      )}

      <button onClick={run} disabled={busy} style={{ ...primary, marginTop: 8 }}>
        {busy ? "Deciding…" : "Decide"}
      </button>

      {err && (
        <div style={{ marginTop: 12, color: "var(--ll-red)", fontSize: 13 }}>{err}</div>
      )}

      {result && (
        <div style={{ marginTop: 20 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
            <span style={{ fontWeight: 700, fontSize: 15 }}>
              {mode === "boolean"
                ? `${result.value ? "YES" : "NO"} (${((result.probability ?? 0) * 100).toFixed(1)}%)`
                : `${result.value}`}
            </span>
            {mode === "score" && result.score != null && (
              <span style={{ color: "var(--ll-text-muted)", fontSize: 13 }}>
                expected {result.score.toFixed(2)}
              </span>
            )}
            <span style={result.calibrated ? badgeOk : badgeMuted}>
              {result.calibrated ? "calibrated" : "raw confidence"}
            </span>
          </div>
          {bars.map((b) => (
            <div key={b.label} style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6 }}>
              <span style={{ width: 120, fontSize: 13, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {b.label}
              </span>
              <div style={{ flex: 1, height: 12, background: "var(--ll-border)", borderRadius: 6, overflow: "hidden" }}>
                <div
                  style={{
                    height: "100%",
                    width: `${Math.min(100, b.p * 100)}%`,
                    background: "var(--ll-accent, #4f8cff)",
                    borderRadius: 6,
                    transition: "width 0.3s ease",
                  }}
                />
              </div>
              <span style={{ width: 56, textAlign: "right", fontSize: 12, fontVariantNumeric: "tabular-nums" }}>
                {(b.p * 100).toFixed(1)}%
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const lbl: CSSProperties = {
  display: "block",
  fontSize: 12,
  color: "var(--ll-text-muted)",
  margin: "10px 0 4px",
};
const field: CSSProperties = {
  width: "100%",
  boxSizing: "border-box",
  background: "var(--ll-bg)",
  color: "var(--ll-text)",
  border: "1px solid var(--ll-border)",
  borderRadius: 6,
  padding: "8px 10px",
  fontSize: 13,
};
const tab: CSSProperties = {
  padding: "6px 14px",
  borderRadius: 6,
  border: "1px solid var(--ll-border)",
  background: "var(--ll-bg)",
  color: "var(--ll-text)",
  cursor: "pointer",
};
const tabActive: CSSProperties = { ...tab, background: "var(--ll-accent, #4f8cff)", color: "#fff", borderColor: "transparent" };
const primary: CSSProperties = {
  padding: "8px 18px",
  borderRadius: 6,
  border: "none",
  background: "var(--ll-accent, #4f8cff)",
  color: "#fff",
  cursor: "pointer",
  fontWeight: 600,
};
const badgeOk: CSSProperties = {
  fontSize: 11,
  padding: "2px 8px",
  borderRadius: 10,
  background: "rgba(63,185,80,0.15)",
  color: "var(--ll-green)",
};
const badgeMuted: CSSProperties = {
  fontSize: 11,
  padding: "2px 8px",
  borderRadius: 10,
  background: "var(--ll-border)",
  color: "var(--ll-text-muted)",
};
