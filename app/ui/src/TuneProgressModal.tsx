import { type CSSProperties, useEffect, useRef, useState } from "react";
import { getTuneProgress, type TuneProgress } from "./api";

/// Inner window shown while a model is loading and/or being auto-tuned.
/// Polls `GET /v1/tune/progress` on a 1s tick. When a full-sweep auto-tune
/// is in flight it renders the live stage bar + step log; otherwise it
/// shows a simple indeterminate "loading" state (a cached model skips the
/// sweep and this closes as soon as the parent's load resolves).
///
/// The parent owns `open` (tie it to the load/re-tune request being
/// in-flight) and closes the modal when its promise settles.
export default function TuneProgressModal({
  open,
  title,
  subtitle,
  onCancelClose,
}: {
  open: boolean;
  title?: string;
  subtitle?: string;
  /// Optional: when set, a "Close" button shows once the sweep is done or
  /// errored (used by the manual re-tune flow, where the request has
  /// already returned). Omit for the load flow (parent auto-closes).
  onCancelClose?: () => void;
}) {
  const [p, setP] = useState<TuneProgress | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    if (!open) {
      setP(null);
      return;
    }
    let cancel = false;
    const tick = async () => {
      try {
        const snap = await getTuneProgress();
        if (!cancel) setP(snap);
      } catch {
        /* transient — keep last */
      }
    };
    tick();
    timer.current = window.setInterval(tick, 1000);
    return () => {
      cancel = true;
      if (timer.current != null) window.clearInterval(timer.current);
    };
  }, [open]);

  if (!open) return null;

  const tuning = !!p && p.active;
  const stageTotal = p && p.stage_total > 0 ? p.stage_total : 7;
  const stageIdx = p ? p.stage_idx : 0;
  const pct = p ? Math.max(0, Math.min(100, p.pct)) : 0;
  const errored = !!p?.error;

  return (
    <div style={backdrop}>
      <div style={panel} onClick={(e) => e.stopPropagation()}>
        <div style={header}>
          <span style={{ fontWeight: 700 }}>
            {title ?? (tuning ? "Auto-tuning model" : "Loading model")}
          </span>
          {subtitle && (
            <span style={{ color: "var(--ll-text-faint)", fontSize: 12 }}>
              {subtitle}
            </span>
          )}
        </div>

        <div style={body}>
          {tuning ? (
            <>
              <div style={{ fontSize: 13, marginBottom: 6 }}>
                Stage {stageIdx}/{stageTotal}
                {p?.stage_name ? ` — ${p.stage_name}` : ""}
              </div>
              <Bar pct={pct} />
              <div
                style={{
                  color: "var(--ll-text-faint)",
                  fontSize: 12,
                  marginTop: 6,
                }}
              >
                {pct.toFixed(0)}% · first-load tuning runs once per model and
                is remembered afterward.
              </div>
              {p && p.log.length > 0 && (
                <pre style={logBox}>{p.log.slice(-14).join("\n")}</pre>
              )}
            </>
          ) : errored ? (
            <div style={{ color: "var(--ll-red)", fontSize: 13 }}>
              Auto-tune failed: {p?.error}. The model still loads at safe
              defaults.
            </div>
          ) : (
            <>
              <div style={{ fontSize: 13, marginBottom: 6 }}>
                {p?.done ? "Finishing up…" : "Loading…"}
              </div>
              <BarIndeterminate />
              <div
                style={{
                  color: "var(--ll-text-faint)",
                  fontSize: 12,
                  marginTop: 6,
                }}
              >
                Large GGUFs can take 30–90 s to map. A first-ever load also
                runs a one-time tuning sweep.
              </div>
            </>
          )}
        </div>

        {onCancelClose && (p?.done || errored) && (
          <div style={footer}>
            <button style={closeBtn} onClick={onCancelClose}>
              Close
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function Bar({ pct }: { pct: number }) {
  return (
    <div style={track}>
      <div
        style={{
          ...fill,
          width: `${pct}%`,
          transition: "width 0.4s ease",
        }}
      />
    </div>
  );
}

function BarIndeterminate() {
  return (
    <div style={track}>
      <div style={{ ...fill, width: "35%", animation: "ll-indet 1.2s ease-in-out infinite" }} />
      <style>
        {`@keyframes ll-indet { 0%{margin-left:-35%} 100%{margin-left:100%} }`}
      </style>
    </div>
  );
}

const backdrop: CSSProperties = {
  position: "fixed",
  inset: 0,
  background: "rgba(0,0,0,0.5)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  zIndex: 1000,
};
const panel: CSSProperties = {
  background: "var(--ll-bg-elevated, var(--ll-bg))",
  border: "1px solid var(--ll-border)",
  borderRadius: 10,
  width: "min(560px, 92vw)",
  boxShadow: "0 12px 40px rgba(0,0,0,0.4)",
  overflow: "hidden",
};
const header: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: 2,
  padding: "14px 16px",
  borderBottom: "1px solid var(--ll-border)",
};
const body: CSSProperties = { padding: "16px" };
const footer: CSSProperties = {
  display: "flex",
  justifyContent: "flex-end",
  padding: "12px 16px",
  borderTop: "1px solid var(--ll-border)",
};
const track: CSSProperties = {
  height: 10,
  borderRadius: 6,
  background: "var(--ll-border)",
  overflow: "hidden",
};
const fill: CSSProperties = {
  height: "100%",
  background: "var(--ll-accent, #4f8cff)",
  borderRadius: 6,
};
const logBox: CSSProperties = {
  marginTop: 12,
  maxHeight: 180,
  overflow: "auto",
  background: "var(--ll-bg)",
  border: "1px solid var(--ll-border)",
  borderRadius: 6,
  padding: "8px 10px",
  fontSize: 11,
  lineHeight: 1.4,
  color: "var(--ll-text-faint)",
  whiteSpace: "pre-wrap",
  wordBreak: "break-word",
};
const closeBtn: CSSProperties = {
  padding: "6px 14px",
  borderRadius: 6,
  border: "1px solid var(--ll-border)",
  background: "var(--ll-bg)",
  color: "var(--ll-text)",
  cursor: "pointer",
};
