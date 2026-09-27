import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./theme.css";

// Catch unhandled JS errors + promise rejections at the window level
// and stash the most recent one on a sentinel attribute. The error
// boundary in App.tsx handles render-time errors; this covers async
// failures from outside React's render cycle (fetch abort, stale
// closures firing after unmount, etc.). On a real crash we surface
// the message via a fixed-position banner — without it, the webview
// goes silently white.
function showCrashBanner(msg: string) {
  const id = "rustllama-crash-banner";
  let el = document.getElementById(id);
  if (!el) {
    el = document.createElement("div");
    el.id = id;
    el.style.cssText = [
      "position: fixed",
      "top: 0",
      "left: 0",
      "right: 0",
      "z-index: 99999",
      "padding: 8px 16px",
      "background: #5a1a1a",
      "color: #ffd0d0",
      "font-family: ui-monospace, SFMono-Regular, monospace",
      "font-size: 12px",
      "border-bottom: 1px solid #f85149",
      "max-height: 30vh",
      "overflow: auto",
      "white-space: pre-wrap",
    ].join(";");
    document.body.appendChild(el);
  }
  el.textContent = `rustllama UI runtime error: ${msg}`;
}

window.addEventListener("error", (ev) => {
  // eslint-disable-next-line no-console
  console.error("window error:", ev.error || ev.message);
  showCrashBanner(`${ev.message}\n${(ev.error as Error | undefined)?.stack ?? ""}`);
});

window.addEventListener("unhandledrejection", (ev) => {
  // eslint-disable-next-line no-console
  console.error("unhandled rejection:", ev.reason);
  const r = ev.reason;
  const msg =
    r instanceof Error
      ? `${r.message}\n${r.stack ?? ""}`
      : String(r);
  showCrashBanner(`unhandled rejection: ${msg}`);
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
