// Top-level GUI shell: persistent left rail with the four pages
// (Chat / Models / Settings / Status) wired via react-router. Layout
// is intentionally minimal — Tauri 2 + native chrome do the heavy
// lifting; we get out of the way.

import { Component, type ReactNode, useEffect, useState } from "react";
import {
  HashRouter,
  Link,
  NavLink,
  Route,
  Routes,
  useLocation,
} from "react-router-dom";

import ChatPage from "./pages/Chat";
import ModelsPage from "./pages/Models";
import QuantizePage from "./pages/Quantize";
import SettingsPage from "./pages/Settings";
import StatusPage from "./pages/Status";
import DecidePage from "./pages/Decide";
import { getConfig, getHealth, getMetrics, type MetricsSnapshot } from "./api";
import { findCodeTheme } from "./themes";

/* ---- Inline line-icons (no icon dependency) ----------------------- */

type IconProps = { className?: string };
const Svg = ({ children }: { children: ReactNode }) => (
  <svg
    className="ll-nav-icon"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.9}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    {children}
  </svg>
);
const IconChat = (_: IconProps) => (
  <Svg>
    <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
  </Svg>
);
const IconModels = (_: IconProps) => (
  <Svg>
    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z" />
    <polyline points="3.27 6.96 12 12.01 20.73 6.96" />
    <line x1="12" y1="22.08" x2="12" y2="12" />
  </Svg>
);
const IconDeveloper = (_: IconProps) => (
  <Svg>
    <polyline points="16 18 22 12 16 6" />
    <polyline points="8 6 2 12 8 18" />
  </Svg>
);
const IconQuantize = (_: IconProps) => (
  <Svg>
    <polygon points="12 2 2 7 12 12 22 7 12 2" />
    <polyline points="2 17 12 22 22 17" />
    <polyline points="2 12 12 17 22 12" />
  </Svg>
);
const IconSettings = (_: IconProps) => (
  <Svg>
    <line x1="4" y1="21" x2="4" y2="14" />
    <line x1="4" y1="10" x2="4" y2="3" />
    <line x1="12" y1="21" x2="12" y2="12" />
    <line x1="12" y1="8" x2="12" y2="3" />
    <line x1="20" y1="21" x2="20" y2="16" />
    <line x1="20" y1="12" x2="20" y2="3" />
    <line x1="1" y1="14" x2="7" y2="14" />
    <line x1="9" y1="8" x2="15" y2="8" />
    <line x1="17" y1="16" x2="23" y2="16" />
  </Svg>
);
const IconSun = (_: IconProps) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.9} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <circle cx="12" cy="12" r="4.2" />
    <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </svg>
);
const IconMoon = (_: IconProps) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.9} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
  </svg>
);

/// Dark/light theme (two themes only, per the design brief — no
/// pluggable system). State mirrors the `data-theme` attribute on
/// `<html>` (seeded before paint by the inline script in index.html)
/// and persists to localStorage. Defaults to dark.
function useTheme(): ["dark" | "light", () => void] {
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    const attr = document.documentElement.getAttribute("data-theme");
    return attr === "light" ? "light" : "dark";
  });
  const toggle = () => {
    setTheme((prev) => {
      const next = prev === "dark" ? "light" : "dark";
      if (next === "light") {
        document.documentElement.setAttribute("data-theme", "light");
      } else {
        document.documentElement.removeAttribute("data-theme");
      }
      try {
        localStorage.setItem("ll-theme", next);
      } catch {
        /* storage blocked — the in-memory attribute still applies */
      }
      return next;
    });
  };
  return [theme, toggle];
}

/// Catches any uncaught render error from a page and renders the
/// message + stack inline. Without this, a JS-side crash hands the
/// webview off to Tauri's generic "Oops" / blank screen and the user
/// has no way to see what broke. Resets when the URL hash changes
/// (i.e., when the user navigates away from the failing page).
class ErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state = { error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: { componentStack: string }) {
    // eslint-disable-next-line no-console
    console.error("rustllama UI error:", error, info.componentStack);
  }

  componentDidMount() {
    // Reset on hash change so navigating away clears the error.
    window.addEventListener("hashchange", this.reset);
  }

  componentWillUnmount() {
    window.removeEventListener("hashchange", this.reset);
  }

  reset = () => {
    if (this.state.error) this.setState({ error: null });
  };

  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 24, color: "var(--ll-red)", fontSize: 13 }}>
          <h2 style={{ margin: "0 0 12px 0", color: "var(--ll-red)" }}>
            UI error
          </h2>
          <pre
            style={{
              background: "var(--ll-bg)",
              border: "1px solid var(--ll-red)",
              borderRadius: 4,
              padding: 12,
              fontSize: 12,
              overflow: "auto",
              maxHeight: "60vh",
              whiteSpace: "pre-wrap",
            }}
          >
            {String(this.state.error)}
            {"\n\n"}
            {this.state.error.stack ?? "(no stack)"}
          </pre>
          <p style={{ marginTop: 12, color: "var(--ll-text-muted)", fontSize: 12 }}>
            Navigate to another page (left rail) to reset.
          </p>
        </div>
      );
    }
    return this.props.children;
  }
}

const fmtGiB = (b?: number | null) =>
  b == null ? "—" : `${(b / 1073741824).toFixed(1)} GB`;

function meterColor(frac: number) {
  return frac > 0.9
    ? "var(--ll-red)"
    : frac > 0.75
      ? "var(--ll-yellow)"
      : "var(--ll-accent)";
}

/// Windows calls the swap file the "page file"; Linux/macOS call it
/// swap. The Tauri webview renders natively, so its userAgent carries
/// the host OS — pick the label the user's platform actually uses.
const swapLabel =
  typeof navigator !== "undefined" && /Windows/i.test(navigator.userAgent)
    ? "Page"
    : "Swap";

interface StatusBarProps {
  status: "ok" | "draining" | "down" | "unknown";
  modelId?: string;
  hint?: string;
}

/// Bottom status bar (LM Studio-style): live health dot, RAM + VRAM
/// usage meters, decode throughput, and the active model id. RAM/VRAM
/// come from `/v1/metrics` (`gpu_sysman` is `null` on non-Intel / mock
/// builds — the VRAM item hides itself). Polls every 2s; independent of
/// the 5s health poll in the shell so throughput stays responsive.
function StatusBar({ status, modelId, hint }: StatusBarProps) {
  const [m, setM] = useState<MetricsSnapshot | null>(null);
  useEffect(() => {
    let cancel = false;
    const tick = async () => {
      try {
        const snap = await getMetrics();
        if (!cancel) setM(snap);
      } catch {
        if (!cancel) setM(null);
      }
    };
    tick();
    const id = setInterval(tick, 2000);
    return () => {
      cancel = true;
      clearInterval(id);
    };
  }, []);

  const dot =
    status === "ok"
      ? "var(--ll-green)"
      : status === "draining"
        ? "var(--ll-yellow)"
        : status === "down"
          ? "var(--ll-red)"
          : "var(--ll-text-faint)";
  const label =
    status === "ok"
      ? "Ready"
      : status === "draining"
        ? "Draining"
        : status === "down"
          ? "Offline"
          : "Checking…";

  const ramUsed =
    m && m.ram_total_bytes ? m.ram_total_bytes - m.ram_available_bytes : null;
  const ramFrac =
    ramUsed != null && m ? ramUsed / m.ram_total_bytes : 0;
  // CPU utilization + brand for the CPU-over-RAM paired row.
  const cpuUtil = m?.cpu_utilization_pct ?? null;
  const cpuBrand = m?.cpu_brand && m.cpu_brand.length > 0 ? m.cpu_brand : "CPU";

  const vt = m?.gpu_sysman?.vram_total_bytes ?? null;
  const vf = m?.gpu_sysman?.vram_free_bytes ?? null;

  // Pagefile / swap: the committed memory (RAM + pagefile) beyond what
  // physical RAM holds ≈ what's backed by the pagefile/swap.
  const commitTotal = m?.commit_total_bytes ?? null;
  const commitAvail = m?.commit_available_bytes ?? null;
  const swapTotal =
    commitTotal != null && m ? Math.max(0, commitTotal - m.ram_total_bytes) : null;
  const swapUsed =
    swapTotal != null && commitTotal != null && commitAvail != null && m
      ? Math.max(
          0,
          Math.min(
            swapTotal,
            commitTotal - commitAvail - (m.ram_total_bytes - m.ram_available_bytes),
          ),
        )
      : null;
  const swapFrac = swapUsed != null && swapTotal ? swapUsed / swapTotal : 0;

  // Disk I/O rates (bytes/sec), split read/write — Page/Swap (system paging)
  // and Model (this process's file I/O). Each drives a two-bar meter.
  const pageIoR = m?.page_io_read_bytes_per_sec ?? null;
  const pageIoW = m?.page_io_write_bytes_per_sec ?? null;
  const modelIoR = m?.model_io_read_bytes_per_sec ?? null;
  const modelIoW = m?.model_io_write_bytes_per_sec ?? null;
  const hasPageIo = pageIoR != null || pageIoW != null;
  const hasModelIo = modelIoR != null || modelIoW != null;
  // Fixed bar reference so the meters stay comparable across ticks
  // (~300 MB/s ≈ a busy SATA SSD; NVMe bursts clamp at full).
  const IO_MAX = 300 * 1024 * 1024;
  const ioFrac = (v: number | null) => (v != null ? Math.min(100, (v / IO_MAX) * 100) : 0);
  const fmtMB = (v: number | null) => (v != null ? (v / 1048576).toFixed(1) : "—");
  // One I/O row: label, then R [bar] N MB/s and W [bar] N MB/s inline.
  const ioRow = (label: string, r: number | null, w: number | null, title: string) => (
    <div className="ll-status-row" title={title}>
      <span className="ll-status-label">{label}</span>
      <span style={{ color: "var(--ll-text-faint)" }}>R</span>
      <div className="ll-meter" style={{ width: 40 }}>
        <div
          className="ll-meter-fill"
          style={{ width: `${ioFrac(r)}%`, background: "var(--ll-accent, #4f8cff)" }}
        />
      </div>
      <span className="ll-status-value" style={{ whiteSpace: "nowrap" }}>
        {fmtMB(r)} MB/s
      </span>
      <span style={{ color: "var(--ll-text-faint)" }}>W</span>
      <div className="ll-meter" style={{ width: 40 }}>
        <div
          className="ll-meter-fill"
          style={{ width: `${ioFrac(w)}%`, background: "var(--ll-yellow)" }}
        />
      </div>
      <span className="ll-status-value" style={{ whiteSpace: "nowrap" }}>
        {fmtMB(w)} MB/s
      </span>
    </div>
  );

  // Per-GPU VRAM bars: prefer the deduped `gpus` list; fall back to the
  // single Sysman snapshot for servers that don't send `gpus` yet.
  const gpuBars =
    m?.gpus && m.gpus.length > 0
      ? m.gpus
      : vt != null
        ? [{ index: 0, vendor: "intel", name: "GPU", vram_total_bytes: vt, vram_free_bytes: vf }]
        : [];

  const toks = m?.ema_tok_s ?? m?.last_tok_s ?? null;

  return (
    <footer className="ll-statusbar" title={status === "down" ? hint : undefined}>
      <div className="ll-status-item">
        <span className="ll-status-dot" style={{ background: dot }} />
        <span className="ll-status-value">{label}</span>
      </div>

      {ramUsed != null && m && (
        <>
        <div className="ll-status-pair">
          {/* CPU utilization (top) — tooltip: brand + the same % as the label. */}
          <div
            className="ll-status-row"
            title={cpuUtil != null ? `${cpuBrand} — ${cpuUtil.toFixed(0)}%` : cpuBrand}
          >
            <span className="ll-status-label">CPU 0</span>
            <div className="ll-meter">
              <div
                className="ll-meter-fill"
                style={{
                  width: cpuUtil != null ? `${Math.min(100, cpuUtil).toFixed(0)}%` : "0%",
                  background: meterColor((cpuUtil ?? 0) / 100),
                }}
              />
            </div>
            <span className="ll-status-value">
              {cpuUtil != null ? `${cpuUtil.toFixed(0)}%` : "n/a"}
            </span>
          </div>
          {/* RAM (middle) — tooltip mirrors the label: used / total. */}
          <div
            className="ll-status-row"
            title={`RAM: ${fmtGiB(ramUsed)} / ${fmtGiB(m.ram_total_bytes)}`}
          >
            <span className="ll-status-label">RAM</span>
            <div className="ll-meter">
              <div
                className="ll-meter-fill"
                style={{
                  width: `${Math.min(100, ramFrac * 100).toFixed(0)}%`,
                  background: meterColor(ramFrac),
                }}
              />
            </div>
            <span className="ll-status-value">
              {fmtGiB(ramUsed)} / {fmtGiB(m.ram_total_bytes)}
            </span>
          </div>
        </div>
        <div className="ll-status-pair">
          {/* Page/Swap used + its read/write I/O, in a column to the right
              of the CPU/RAM pair. */}
          {swapUsed != null && swapTotal != null && swapTotal > 0 && (
            <div
              className="ll-status-row"
              title={`${swapLabel}: ${fmtGiB(swapUsed)} / ${fmtGiB(swapTotal)}`}
            >
              <span className="ll-status-label">{swapLabel}</span>
              <div className="ll-meter">
                <div
                  className="ll-meter-fill"
                  style={{
                    width: `${Math.min(100, swapFrac * 100).toFixed(0)}%`,
                    background: meterColor(swapFrac),
                  }}
                />
              </div>
              <span className="ll-status-value">
                {fmtGiB(swapUsed)} / {fmtGiB(swapTotal)}
              </span>
            </div>
          )}
          {/* Page/Swap disk I/O — R [bar] N MB/s  W [bar] N MB/s. */}
          {hasPageIo &&
            ioRow(
              `${swapLabel} I/O`,
              pageIoR,
              pageIoW,
              `${swapLabel} disk I/O — read (page-ins from disk) / write (page-outs to ${
                swapLabel === "Page" ? "the page file" : "swap"
              })`,
            )}
          {/* Model disk I/O — read = page-in rate (mmap model), write = process. */}
          {hasModelIo &&
            ioRow(
              "Model I/O",
              modelIoR,
              modelIoW,
              "Model disk I/O — read: page-in (hard-fault) rate, dominated by the memory-mapped model being read from disk (load / cold faults); write: the server process's file writes (KV / conversation DB)",
            )}
        </div>
        </>
      )}

      {gpuBars.map((g, i) => {
        const gt = g.vram_total_bytes ?? null;
        const gf = g.vram_free_bytes ?? null;
        const used = gt != null && gf != null ? gt - gf : null;
        const frac = used != null && gt ? used / gt : 0;
        const util = g.utilization_pct ?? null;
        // Always label by the stable index ("GPU 0", "GPU 1", …) so the
        // number is explicit even with a single GPU.
        const gpuLabel = `GPU ${g.index}`;
        return (
          <div className="ll-status-pair" key={`${g.name}-${i}`}>
            {/* GPU utilization (top) — tooltip: GPU name + the same % as the label. */}
            <div
              className="ll-status-row"
              title={util != null ? `${g.name} — ${util.toFixed(0)}%` : g.name}
            >
              <span className="ll-status-label">{gpuLabel}</span>
              <div className="ll-meter">
                <div
                  className="ll-meter-fill"
                  style={{
                    width: util != null ? `${Math.min(100, util).toFixed(0)}%` : "0%",
                    background: meterColor((util ?? 0) / 100),
                  }}
                />
              </div>
              <span className="ll-status-value">
                {util != null ? `${util.toFixed(0)}%` : "n/a"}
              </span>
            </div>
            {/* VRAM (bottom) — tooltip mirrors the label: used / total. */}
            <div
              className="ll-status-row"
              title={`${g.name}: ${used != null ? fmtGiB(used) : "—"} / ${fmtGiB(gt)}`}
            >
              <span className="ll-status-label">VRAM</span>
              <div className="ll-meter">
                <div
                  className="ll-meter-fill"
                  style={{
                    width: used != null ? `${Math.min(100, frac * 100).toFixed(0)}%` : "0%",
                    background: meterColor(frac),
                  }}
                />
              </div>
              <span className="ll-status-value">
                {gt != null
                  ? used != null
                    ? `${fmtGiB(used)} / ${fmtGiB(gt)}`
                    : `— / ${fmtGiB(gt)}`
                  : "—"}
              </span>
            </div>
          </div>
        );
      })}

      {/* Always visible — shows the last/EMA decode rate, or 0.0 when idle. */}
      <div className="ll-status-item" title="Decode throughput (EMA)">
        <span className="ll-status-label">tok/s</span>
        <span className="ll-status-value">{(toks ?? 0).toFixed(1)}</span>
      </div>

      <span className="ll-status-model" title={modelId ?? "No model loaded"}>
        {modelId ?? "No model loaded"}
      </span>
    </footer>
  );
}

function Shell() {
  const loc = useLocation();
  const [theme, toggleTheme] = useTheme();
  const [status, setStatus] = useState<"ok" | "draining" | "down" | "unknown">(
    "unknown",
  );
  const [modelId, setModelId] = useState<string | undefined>();
  const [statusHint, setStatusHint] = useState<string | undefined>();
  useEffect(() => {
    let cancel = false;
    const tick = async () => {
      try {
        const h = await getHealth();
        if (cancel) return;
        if (h.draining) setStatus("draining");
        else if (h.status === "ok") setStatus("ok");
        else setStatus("down");
        setModelId(h.model_id);
        setStatusHint(undefined);
      } catch (e) {
        if (!cancel) {
          setStatus("down");
          // `fetch` failures across origins surface as bare
          // `TypeError: Failed to fetch` — same as a real network
          // outage. Distinguishing is impossible from JS, but we can
          // surface the most-likely cause so the user has a thread
          // to pull on instead of a blank "offline" indicator.
          const msg = e instanceof Error ? e.message : String(e);
          setStatusHint(
            msg.toLowerCase().includes("failed to fetch") ||
              msg.toLowerCase().includes("networkerror")
              ? "server unreachable or CORS blocked — see %LOCALAPPDATA%\\rustllama\\models\\gui.log"
              : msg,
          );
        }
      }
    };
    tick();
    // Poll every 5s. Cheap; gives the chat page an early heads-up if
    // the server falls over mid-session.
    const id = setInterval(tick, 5000);
    return () => {
      cancel = true;
      clearInterval(id);
    };
  }, [loc.pathname]);

  const navItem = (
    to: string,
    label: string,
    Icon: (props: IconProps) => ReactNode,
  ) => (
    <NavLink
      to={to}
      end
      className={({ isActive }) => `ll-nav-item${isActive ? " active" : ""}`}
    >
      <Icon />
      <span>{label}</span>
    </NavLink>
  );

  return (
    <div className="ll-shell">
      <nav className="ll-sidebar">
        <Link to="/" className="ll-brand">
          <span className="ll-brand-mark">r</span>
          <span className="ll-brand-name">rustllama</span>
        </Link>
        <div className="ll-nav">
          {navItem("/", "Chat", IconChat)}
          {navItem("/models", "Models", IconModels)}
          {navItem("/decide", "Decide", IconDeveloper)}
          {navItem("/status", "Status", IconDeveloper)}
          {navItem("/quantize", "Quantize", IconQuantize)}
          {navItem("/settings", "Settings", IconSettings)}
        </div>
        <div className="ll-sidebar-footer">
          <button
            className="ll-theme-toggle"
            onClick={toggleTheme}
            title="Toggle dark / light theme"
          >
            {theme === "dark" ? <IconSun /> : <IconMoon />}
            <span>{theme === "dark" ? "Light mode" : "Dark mode"}</span>
          </button>
        </div>
      </nav>
      <main className="ll-main">
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<ChatPage />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/decide" element={<DecidePage />} />
            <Route path="/quantize" element={<QuantizePage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/status" element={<StatusPage />} />
          </Routes>
        </ErrorBoundary>
      </main>
      <StatusBar status={status} modelId={modelId} hint={statusHint} />
    </div>
  );
}

/// Sync the active highlight.js theme with `[ui].code_theme` in
/// config.toml. Manages a single `<style id="hljs-theme-active">`
/// element under `<head>`; updates its contents whenever the polled
/// config name changes. Default `"github-dark"` (and any unknown
/// value) clears the injected block so the static github-dark
/// stylesheet in `index.html` shows through unchanged.
function useCodeThemeInjector() {
  const [themeName, setThemeName] = useState<string>("github-dark");

  // Poll the config endpoint on the same 5s cadence as the health
  // probe — cheap, picks up Settings-page saves within a tick.
  useEffect(() => {
    let cancel = false;
    const tick = async () => {
      try {
        const env = await getConfig();
        if (cancel) return;
        const next = env.config.ui?.code_theme ?? "github-dark";
        setThemeName(next);
      } catch {
        // Mock-engine / offline: keep the current value.
      }
    };
    tick();
    const id = setInterval(tick, 5000);
    return () => {
      cancel = true;
      clearInterval(id);
    };
  }, []);

  // Imperatively manage the <style> element. Putting `<style>` in
  // JSX would render it inside `<body>` which still works but
  // splits the theme rules from the rest of the head <style> block
  // — appending to <head> keeps DOM-cascade order obvious.
  useEffect(() => {
    const id = "hljs-theme-active";
    let el = document.getElementById(id) as HTMLStyleElement | null;
    if (!el) {
      el = document.createElement("style");
      el.id = id;
      document.head.appendChild(el);
    }
    el.textContent = findCodeTheme(themeName).css;
    return () => {
      // Don't remove on unmount — App.tsx mounts for the lifetime
      // of the webview, so dropping it would only matter on a hard
      // remount (HMR during dev). Leave the element in place.
    };
  }, [themeName]);
}

export default function App() {
  // HashRouter keeps deep links working in Tauri without configuring
  // the bundler to handle history-mode routes server-side.
  useCodeThemeInjector();
  return (
    <HashRouter>
      <Shell />
    </HashRouter>
  );
}
