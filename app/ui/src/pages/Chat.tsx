// Streaming chat page with a conversation sidebar (sqlite-backed
// via the server's /api/conversations routes when `--features
// history` is on). Without that feature the sidebar gracefully
// degrades to a no-op — the chat itself still works against the
// in-memory transcript.
//
// Each user message + assistant response is persisted on completion
// (`onDone`); the sidebar entry's `updated_at` bumps so the active
// conversation hops to the top of the list.

import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
// highlight.js is a transitive dep of rehype-highlight; rollup can't
// resolve its `/styles/*.css` exports without a direct package entry.
// The github-dark theme is inlined into index.html instead.
import {
  appendMessage,
  cancelRequest,
  createConversation,
  deleteConversation,
  getConfig,
  getConversation,
  chatOnce,
  getMetrics,
  historyAvailable,
  listConversations,
  listModels,
  listOllamaTags,
  loadModel,
  setDefaultModel,
  streamChat,
  tokenizeCount,
  unloadModel,
  type AskUser,
  type ChatMessage,
  type ConversationSummary,
  type ModelInfo,
  type SystemPrompt,
  type TagsModel,
  type ToolCall,
  type UsageInfo,
} from "../api";
import TuneProgressModal from "../TuneProgressModal";

type Phase = "idle" | "streaming" | "error";

/* ---- Reasoning ("thinking") handling ------------------------------ */
// Reasoning models (DeepSeek-R1, QwQ, …) wrap their chain-of-thought in
// <think>…</think> (or <thinking>…</thinking>). We never render that raw
// stream in the transcript — only a collapsed status chip ("Thinking…" →
// "Thought") and the final answer after the closing tag.
function splitThinking(content: string): {
  thinking: string;
  answer: string;
  hasThinking: boolean;
  thinkingDone: boolean;
} {
  const open = content.match(/<think(?:ing)?>/i);
  if (!open || open.index === undefined) {
    return {
      thinking: "",
      answer: content,
      hasThinking: false,
      thinkingDone: true,
    };
  }
  const rest = content.slice(open.index + open[0].length);
  const close = rest.match(/<\/think(?:ing)?>/i);
  if (!close || close.index === undefined) {
    // Still streaming inside the think block — no answer yet.
    return { thinking: rest, answer: "", hasThinking: true, thinkingDone: false };
  }
  const before = content.slice(0, open.index);
  const after = rest.slice(close.index + close[0].length);
  return {
    thinking: rest.slice(0, close.index),
    answer: (before + after).trim(),
    hasThinking: true,
    thinkingDone: true,
  };
}

function ThinkingBlock({ thinking, done }: { thinking: string; done: boolean }) {
  const [open, setOpen] = useState(false);
  const expandable = done && thinking.trim().length > 0;
  return (
    <div className="ll-think">
      <span
        className="ll-think-chip"
        onClick={() => expandable && setOpen((o) => !o)}
        style={{ cursor: expandable ? "pointer" : "default" }}
        title={expandable ? "Show / hide reasoning" : undefined}
      >
        <span className={`ll-think-dot${done ? "" : " active"}`} />
        {done ? "Thought" : "Thinking…"}
        {expandable && (
          <span style={{ color: "var(--ll-text-faint)", fontSize: 11 }}>
            {open ? "hide" : "show"}
          </span>
        )}
      </span>
      {open && expandable && (
        <div className="ll-think-body">{thinking.trim()}</div>
      )}
    </div>
  );
}

/* ---- Context compaction ------------------------------------------- */
// Rough token estimate (≈3.6 chars/token) — precise enough to decide
// when to compact without a tokenize round-trip.
function estTokens(msgs: ChatMessage[]): number {
  return Math.ceil(
    msgs.reduce((n, m) => n + (m.content?.length ?? 0), 0) / 3.6,
  );
}

// Ask the model itself to summarize the older turns into ~targetTokens.
async function summarizeHistory(
  older: ChatMessage[],
  targetTokens: number,
): Promise<string> {
  const transcript = older
    .map((m) => `${m.role.toUpperCase()}: ${m.content}`)
    .join("\n\n");
  const summary = await chatOnce(
    [
      {
        role: "system",
        content:
          "You are compacting a conversation to fit a limited context window. " +
          "Summarize the exchange below, preserving every important fact, decision, " +
          "name, number, code snippet, file path, and open question needed to " +
          `continue seamlessly. Be concise — aim for about ${targetTokens} tokens. ` +
          "Output only the summary, with no preamble.",
      },
      { role: "user", content: transcript },
    ],
    {
      temperature: 0.2,
      maxTokens: Math.max(256, Math.min(targetTokens * 2, 2048)),
    },
  );
  return summary.trim();
}

// Keep a leading system prompt + the last few turns verbatim; replace
// everything older with a single model-written summary message.
const COMPACT_KEEP_RECENT = 4;
async function compactContext(
  hist: ChatMessage[],
  ctxBudget: number | null,
): Promise<ChatMessage[]> {
  const lead = hist[0]?.role === "system" ? [hist[0]] : [];
  const body = hist.slice(lead.length);
  if (body.length <= COMPACT_KEEP_RECENT + 1) return hist;
  const recent = body.slice(-COMPACT_KEEP_RECENT);
  const older = body.slice(0, body.length - COMPACT_KEEP_RECENT);
  if (older.length === 0) return hist;
  const target = ctxBudget ? Math.max(400, Math.round(ctxBudget * 0.25)) : 800;
  const summary = await summarizeHistory(older, target);
  if (!summary) return hist;
  return [
    ...lead,
    {
      role: "system",
      content: `[Summary of ${older.length} earlier messages]\n${summary}`,
    },
    ...recent,
  ];
}

/* ---- Model-loader bar (LM Studio top bar) ------------------------- */
// Load / switch / eject the active model from the chat view. Loaded
// models come from /v1/models; cached-on-disk models from /api/tags
// (loaded by file-stem `name`). Selecting a loaded model sets it default;
// selecting a cached one loads it, then makes it default.
function ModelLoaderBar({
  activeModelId,
  onChanged,
}: {
  activeModelId?: string;
  onChanged?: () => void;
}) {
  const [loaded, setLoaded] = useState<ModelInfo[]>([]);
  const [cached, setCached] = useState<TagsModel[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  // Model name shown in the load/auto-tune progress modal.
  const [progressModel, setProgressModel] = useState("");

  const refresh = async () => {
    const [l, c] = await Promise.all([
      listModels().catch(() => [] as ModelInfo[]),
      listOllamaTags().catch(() => [] as TagsModel[]),
    ]);
    setLoaded(l);
    setCached(c);
  };
  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, []);

  const loadedIds = new Set(loaded.map((m) => m.id));
  const available = cached.filter((c) => {
    const stem = c.name.replace(/\.gguf$/i, "");
    return !loadedIds.has(stem) && !loadedIds.has(c.name);
  });

  const onSelect = async (val: string) => {
    if (!val || busy) return;
    setErr(null);
    const sep = val.indexOf(":");
    const kind = val.slice(0, sep);
    const key = val.slice(sep + 1);
    try {
      if (kind === "loaded") {
        setBusy("Switching…");
        await setDefaultModel(key);
      } else {
        setProgressModel(key);
        setBusy("Loading…");
        const res = await loadModel({ name: key });
        if (res?.model_id && !res.is_default) {
          await setDefaultModel(res.model_id);
        }
      }
      await refresh();
      onChanged?.();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const eject = async () => {
    if (!activeModelId || busy) return;
    setBusy("Ejecting…");
    setErr(null);
    try {
      await unloadModel(activeModelId);
      await refresh();
      onChanged?.();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="ll-topbar">
      <span
        style={{
          color: "var(--ll-text-faint)",
          fontSize: 11,
          fontWeight: 700,
          letterSpacing: 0.5,
        }}
      >
        MODEL
      </span>
      <select
        className="ll-select"
        value={activeModelId ? `loaded:${activeModelId}` : ""}
        onChange={(e) => onSelect(e.target.value)}
        disabled={!!busy}
        style={{ minWidth: 240, maxWidth: 420 }}
        title="Load, switch, or eject the active model"
      >
        <option value="" disabled>
          {loaded.length ? "Select a model…" : "Select a model to load…"}
        </option>
        {loaded.length > 0 && (
          <optgroup label="Loaded">
            {loaded.map((m) => (
              <option key={`l-${m.id}`} value={`loaded:${m.id}`}>
                {m.id}
              </option>
            ))}
          </optgroup>
        )}
        {available.length > 0 && (
          <optgroup label="Available — click to load">
            {available.map((c) => (
              <option key={`c-${c.name}`} value={`cached:${c.name}`}>
                {c.name.replace(/\.gguf$/i, "")}
                {c.details?.quantization_level
                  ? ` · ${c.details.quantization_level}`
                  : ""}
              </option>
            ))}
          </optgroup>
        )}
      </select>
      {busy && (
        <span style={{ fontSize: 12, color: "var(--ll-accent)" }}>{busy}</span>
      )}
      {activeModelId && !busy && (
        <button
          className="ll-btn"
          onClick={eject}
          title="Unload the active model"
        >
          Eject
        </button>
      )}
      {err && (
        <span
          style={{
            fontSize: 12,
            color: "var(--ll-red)",
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            maxWidth: 320,
          }}
          title={err}
        >
          {err}
        </span>
      )}
      <TuneProgressModal open={busy === "Loading…"} subtitle={progressModel} />
    </div>
  );
}

export default function ChatPage() {
  const [history, setHistory] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  // Attached images (data: URIs) for the next send. Vision-enabled
  // servers accept them via OpenAI image_url content blocks; a
  // text-only server responds 400 and the error banner explains.
  const [attachedImages, setAttachedImages] = useState<string[]>([]);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const attachImageFiles = (files: FileList | null) => {
    if (!files) return;
    const MAX_BYTES = 10 * 1024 * 1024;
    Array.from(files).forEach((f) => {
      if (!/^image\/(png|jpe?g)$/.test(f.type)) return;
      if (f.size > MAX_BYTES) return;
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === "string") {
          setAttachedImages((prev) => [...prev, reader.result as string]);
        }
      };
      reader.readAsDataURL(f);
    });
  };
  const [streaming, setStreaming] = useState<Phase>("idle");
  // True while the model is summarizing older turns to fit the context
  // window (auto before a send that would overflow, or via the Compact
  // button). Blocks re-entry and drives the composer status.
  const [compacting, setCompacting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Sampling controls (right-side inference panel). These replace the
  // values that used to be hardcoded (temperature 0.7 / max_tokens 512).
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);
  const [seed, setSeed] = useState<string>("");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [lastUsage, setLastUsage] = useState<UsageInfo | null>(null);
  /// OpenAI `system_fingerprint` from the most recent streamed
  /// response. Carries the backend's config fingerprint (server
  /// version + model id + KV dtype). When this changes between
  /// consecutive turns of the same conversation, the user has
  /// effectively been silently moved to a different backend —
  /// surface it so they can decide whether to start fresh.
  /// The getter binding is intentionally discarded — we only read
  /// the previous value via React's setter-callback form.
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [, setLastFingerprint] = useState<string | null>(null);
  const [fingerprintChanged, setFingerprintChanged] =
    useState<{ prev: string; current: string } | null>(null);
  const [pending, setPending] = useState<string>("");
  /// CLARIFY: a pending `ask_user` question. While set, the transcript
  /// shows option buttons; clicking one appends it as a user turn and
  /// continues the conversation.
  const [pendingQuestion, setPendingQuestion] = useState<AskUser | null>(null);
  /// A pending tool-call proposal awaiting approve/deny (auto-tools OFF).
  const [pendingTools, setPendingTools] = useState<ToolCall[] | null>(null);
  /// Auto-run tools: accept proposed tool calls without prompting. The chat
  /// never executes tools — this only gates the confirm UI. Persisted per
  /// browser (localStorage, best-effort).
  const [autoTools, setAutoTools] = useState<boolean>(() => {
    try {
      return localStorage.getItem("rustllama.autoTools") === "1";
    } catch {
      return false;
    }
  });
  const toggleAutoTools = () => {
    setAutoTools((v) => {
      const next = !v;
      try {
        localStorage.setItem("rustllama.autoTools", next ? "1" : "0");
      } catch {
        /* private mode / blocked storage — in-memory only */
      }
      return next;
    });
  };
  /// CLARIFY: when ON (default), the server may inject the reserved
  /// `ask_user` tool so the model can pause and ask a clarifying question
  /// with selectable options instead of guessing (`onAskUser` renders the
  /// chooser). Persisted per browser (localStorage, best-effort). Defaults
  /// ON unless the user previously turned it off. When OFF the request goes
  /// through the plain (no-tool) path.
  const [clarify, setClarify] = useState<boolean>(() => {
    try {
      return localStorage.getItem("rustllama.clarify") !== "0";
    } catch {
      return true;
    }
  });
  const toggleClarify = () => {
    setClarify((v) => {
      const next = !v;
      try {
        localStorage.setItem("rustllama.clarify", next ? "1" : "0");
      } catch {
        /* private mode / blocked storage — in-memory only */
      }
      return next;
    });
  };
  const abortRef = useRef<AbortController | null>(null);
  /// Server-side request id (`chatcmpl-…`) captured from the first
  /// SSE chunk. Used by `stop()` to POST `/v1/cancel` so the
  /// engine actually stops generating, not just so the client
  /// drops SSE chunks on the floor. Cleared on stream completion.
  const requestIdRef = useRef<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  /// Whether the keyboard-shortcuts overlay is showing. Toggled by
  /// `?` (when no input is focused) and by the header "Shortcuts"
  /// button; Esc closes it.
  const [showShortcuts, setShowShortcuts] = useState(false);

  // History sidebar state.
  const [historyOn, setHistoryOn] = useState<boolean>(false);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeConvId, setActiveConvId] = useState<number | null>(null);

  // Whether the server has a default model loaded. `null` while we
  // haven't polled yet, true/false thereafter. Drives the "no model
  // loaded" banner above the chat input — without it, hitting Send
  // produces a generic 404 from the server with no path forward.
  const [hasModel, setHasModel] = useState<boolean | null>(null);
  /// Model id from `/healthz` — captured alongside `hasModel` so the
  /// conversation export header can name the model that produced the
  /// session, not just "rustllama".
  const [modelId, setModelId] = useState<string>("");

  useEffect(() => {
    let cancel = false;
    const probe = async () => {
      try {
        const r = await fetch("http://127.0.0.1:11434/healthz");
        if (!r.ok || cancel) return;
        const j = await r.json();
        if (cancel) return;
        // Server returns empty model_id when started with no default
        // (the "fresh install, never configured" case).
        const id = typeof j.model_id === "string" ? j.model_id : "";
        setHasModel(id.length > 0);
        setModelId(id);
      } catch {
        if (!cancel) setHasModel(null);
      }
    };
    probe();
    const id = setInterval(probe, 5000);
    return () => {
      cancel = true;
      clearInterval(id);
    };
  }, []);

  // Server-side token count for the current `draft`. `null` while
  // the request is in-flight or before the first tick; updated by
  // the debounced effect below.
  const [draftTokens, setDraftTokens] = useState<number | null>(null);
  /// System-prompt library from `config.toml`. The dropdown above
  /// the input chooses which entry (if any) prepends a `role:
  /// "system"` message to *new* conversations. Empty array hides the
  /// dropdown entirely.
  const [systemPrompts, setSystemPrompts] = useState<SystemPrompt[]>([]);
  /// Name of the user-picked system prompt, or `""` for "none". An
  /// auto-pick based on the active model's `default_for_model` only
  /// fires when `userPickedPrompt` is false — touching the dropdown
  /// flips this to true so a model reload doesn't clobber the choice.
  const [selectedPromptName, setSelectedPromptName] = useState<string>("");
  const [userPickedPrompt, setUserPickedPrompt] = useState(false);

  useEffect(() => {
    let cancel = false;
    (async () => {
      try {
        const env = await getConfig();
        if (cancel) return;
        setSystemPrompts(env.config.system_prompts ?? []);
      } catch {
        // /v1/config unreachable (mock engine without config path) —
        // hide the dropdown rather than show an error inline.
        if (!cancel) setSystemPrompts([]);
      }
    })();
    return () => {
      cancel = true;
    };
  }, []);

  // Auto-select the prompt whose `default_for_model` matches the
  // currently-loaded model — but only while the user hasn't picked
  // something else. This makes "load Qwen-coder → coder prompt
  // auto-selected" Just Work without overriding deliberate choices.
  useEffect(() => {
    if (userPickedPrompt) return;
    if (!modelId || systemPrompts.length === 0) return;
    const matched = systemPrompts.find(
      (p) => p.default_for_model && p.default_for_model === modelId,
    );
    setSelectedPromptName(matched ? matched.name : "");
  }, [modelId, systemPrompts, userPickedPrompt]);

  /// Context-window size of the active model (`/v1/metrics.ctx_size`).
  /// Drives the budget % shown next to the token count; `null` means
  /// the budget hint is hidden (mock engine, no model loaded, or
  /// `/v1/metrics` unreachable). Polled on a slow interval — the
  /// model only changes on Load and the value never changes between
  /// loads.
  const [ctxBudget, setCtxBudget] = useState<number | null>(null);

  useEffect(() => {
    let cancel = false;
    const tick = async () => {
      try {
        const m = await getMetrics();
        if (cancel) return;
        // ctx_size = 0 from a mock engine; treat as "no budget known"
        // rather than divide-by-zero in the percentage.
        setCtxBudget(m.ctx_size > 0 ? m.ctx_size : null);
      } catch {
        if (!cancel) setCtxBudget(null);
      }
    };
    tick();
    const id = setInterval(tick, 10_000);
    return () => {
      cancel = true;
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    // 300ms debounce + abort-on-change to avoid hammering /v1/tokenize
    // on every keystroke. Pinning the prior token count to `null`
    // while a request is in flight surfaces a "…" placeholder so the
    // user knows the value is being recomputed.
    if (draft.trim().length === 0) {
      setDraftTokens(0);
      return;
    }
    const ctrl = new AbortController();
    const timer = setTimeout(async () => {
      try {
        const r = await tokenizeCount(draft, { signal: ctrl.signal, addBos: false });
        if (!ctrl.signal.aborted) setDraftTokens(r.count);
      } catch {
        // Server may not support tokenize (mock engine) — hide the
        // meter rather than show "error" inline.
        if (!ctrl.signal.aborted) setDraftTokens(null);
      }
    }, 300);
    return () => {
      ctrl.abort();
      clearTimeout(timer);
    };
  }, [draft]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [history, pending]);

  // Global keyboard shortcuts. Document-level listener so the user
  // doesn't have to keep focus on a specific element. `?` only fires
  // when the active element isn't an input/textarea — otherwise it'd
  // clobber the user typing `?` into the chat.
  useEffect(() => {
    const isTypingTarget = (el: EventTarget | null) => {
      if (!(el instanceof HTMLElement)) return false;
      const tag = el.tagName;
      return (
        tag === "INPUT" ||
        tag === "TEXTAREA" ||
        tag === "SELECT" ||
        el.isContentEditable
      );
    };
    const onKey = (e: KeyboardEvent) => {
      // Esc always closes the overlay regardless of focus — common
      // dismiss affordance.
      if (e.key === "Escape" && showShortcuts) {
        setShowShortcuts(false);
        e.preventDefault();
        return;
      }
      const mod = e.ctrlKey || e.metaKey;
      // Ctrl/Cmd shortcuts fire even with input focus (browser
      // chrome already swallows Ctrl+L for the URL bar in a normal
      // browser tab, but Tauri's webview surfaces it to the page).
      if (mod && !e.shiftKey && !e.altKey) {
        if (e.key === "l" || e.key === "L") {
          e.preventDefault();
          textareaRef.current?.focus();
          return;
        }
        if (e.key === "k" || e.key === "K") {
          // Don't kill an in-flight stream silently — the user may
          // have pressed Ctrl+K by accident. `startNew()` aborts +
          // resets; matches what clicking the header New button does.
          e.preventDefault();
          startNew();
          return;
        }
        if (e.key === "e" || e.key === "E") {
          if (history.length === 0 || streaming === "streaming") return;
          e.preventDefault();
          downloadTextFile(
            buildExportFilename("md"),
            buildExportMarkdown(history, modelId),
            "text/markdown",
          );
          return;
        }
        if (e.key === "r" || e.key === "R") {
          if (streaming === "streaming") return;
          // Only fire when there's a trailing assistant turn to
          // replace — otherwise Ctrl+R is the browser refresh
          // gesture and we shouldn't claim it for nothing.
          const hasAssistant = history.some((m) => m.role === "assistant");
          if (!hasAssistant) return;
          e.preventDefault();
          regenerateLast();
          return;
        }
      }
      if (e.key === "?" && !isTypingTarget(document.activeElement)) {
        e.preventDefault();
        setShowShortcuts((v) => !v);
        return;
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [history, modelId, streaming, showShortcuts]);

  // Detect history availability + load conversations on mount.
  useEffect(() => {
    let cancel = false;
    (async () => {
      const on = await historyAvailable();
      if (cancel) return;
      setHistoryOn(on);
      if (on) {
        try {
          const list = await listConversations();
          if (!cancel) setConversations(list);
        } catch {
          // ignore — empty list is fine
        }
      }
    })();
    return () => {
      cancel = true;
    };
  }, []);

  const refreshList = async () => {
    if (!historyOn) return;
    try {
      setConversations(await listConversations());
    } catch {
      // tolerate transient failures
    }
  };

  const loadConv = async (id: number) => {
    if (streaming === "streaming") return;
    setError(null);
    setPending("");
    try {
      const c = await getConversation(id);
      if (!c) return;
      const msgs: ChatMessage[] = c.messages.map((m) => ({
        role: m.role === "tool" ? "system" : (m.role as ChatMessage["role"]),
        content: m.content,
      }));
      setHistory(msgs);
      setActiveConvId(id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  /// Two-part cancel of any in-flight chat stream: POST /v1/cancel
  /// so the server engine stops generating, then abort the fetch
  /// so the SSE reader tears down. Safe to call when no stream is
  /// running (the ref is null → no POST, abort is a no-op). Shared
  /// by `stop()` and `startNew()` since both must release engine
  /// resources, not just disconnect the client.
  const cancelAndAbortInFlight = () => {
    const id = requestIdRef.current;
    if (id) {
      cancelRequest(id).catch(() => {
        // Cancel-side failure shouldn't block the local abort —
        // the user just loses the server-side cleanup.
      });
    }
    abortRef.current?.abort();
  };

  const startNew = () => {
    cancelAndAbortInFlight();
    setHistory([]);
    setPending("");
    setError(null);
    setStreaming("idle");
    setLastUsage(null);
    setActiveConvId(null);
    setPendingQuestion(null);
    setPendingTools(null);
  };

  const removeConv = async (id: number) => {
    if (!historyOn) return;
    try {
      await deleteConversation(id);
    } catch {
      /* ignore */
    }
    if (activeConvId === id) startNew();
    await refreshList();
  };

  /// One-line description of proposed tool calls (auto-run note / footer).
  const describeToolCalls = (calls: ToolCall[]) =>
    "Proposed tool call: " +
    calls.map((c) => `${c.name}(${c.arguments})`).join(", ");

  /// Stream one assistant turn against `toSend` and fold the result into the
  /// transcript. Shared by `send()`, the CLARIFY option pick, and tool-deny
  /// so a continuation reuses the exact same handlers (persistence, usage,
  /// fingerprint, plus the ask_user / tool_calls policy).
  const streamAssistantTurn = async (
    toSend: ChatMessage[],
    convId: number | null,
  ) => {
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    requestIdRef.current = null;
    let acc = "";
    await streamChat(
      toSend,
      {
        temperature,
        maxTokens,
        seed: seed.trim() === "" ? undefined : Number(seed),
        // CLARIFY opt-in: routes through the tools path (ask_user only).
        allowClarify: clarify,
        abort: ctrl.signal,
      },
      {
        onStart: (id) => {
          requestIdRef.current = id;
        },
        onContent: (delta) => {
          acc += delta;
          setPending(acc);
        },
        // CLARIFY: stash the question so the transcript renders a chooser.
        onAskUser: (q) => {
          setPendingTools(null);
          setPendingQuestion(q);
        },
        // Tool-call proposal: auto-run accepts silently (a transcript note);
        // otherwise raise an approve/deny prompt.
        onToolCalls: (calls) => {
          if (calls.length === 0) return;
          if (autoTools) {
            setHistory((h) => [
              ...h,
              { role: "assistant", content: describeToolCalls(calls) },
            ]);
          } else {
            setPendingQuestion(null);
            setPendingTools(calls);
          }
        },
        onDone: async (info) => {
          if (acc.length > 0) {
            setHistory((h) => [...h, { role: "assistant", content: acc }]);
            // Persist the assistant turn + refresh the sidebar so the
            // active conversation pops to the top by updated_at.
            if (historyOn && convId !== null) {
              try {
                await appendMessage(convId, "assistant", acc);
                await refreshList();
              } catch (e) {
                console.warn("history persist failed:", e);
              }
            }
          }
          setPending("");
          setStreaming("idle");
          requestIdRef.current = null;
          if (info?.usage) setLastUsage(info.usage);
          if (info?.systemFingerprint) {
            const next = info.systemFingerprint;
            setLastFingerprint((prev) => {
              if (prev && prev !== next) {
                setFingerprintChanged({ prev, current: next });
              }
              return next;
            });
          }
        },
        onError: async (e) => {
          if (e.name === "AbortError" || /aborted/i.test(e.message)) {
            if (acc.length > 0) {
              setHistory((h) => [...h, { role: "assistant", content: acc }]);
              if (historyOn && convId !== null) {
                try {
                  await appendMessage(convId, "assistant", acc);
                  await refreshList();
                } catch {
                  /* ignore */
                }
              }
            }
            setPending("");
            setStreaming("idle");
            requestIdRef.current = null;
            return;
          }
          setError(e.message);
          setPending("");
          setStreaming("error");
          requestIdRef.current = null;
        },
      },
    );
  };

  /// CLARIFY: the user picked an option. Record the question + answer and
  /// continue the conversation.
  const chooseOption = async (opt: string) => {
    if (streaming === "streaming") return;
    const q = pendingQuestion;
    if (!q) return;
    setPendingQuestion(null);
    const newHistory: ChatMessage[] = [
      ...history,
      { role: "assistant", content: q.prompt },
      { role: "user", content: opt },
    ];
    setHistory(newHistory);
    setError(null);
    setPending("");
    setStreaming("streaming");
    const convId = activeConvId;
    if (historyOn && convId !== null) {
      try {
        await appendMessage(convId, "assistant", q.prompt);
        await appendMessage(convId, "user", opt);
      } catch (e) {
        console.warn("history persist failed:", e);
      }
    }
    await streamAssistantTurn(newHistory, convId);
  };

  /// Approve a proposed tool call — the chat has no executor, so this just
  /// dismisses the prompt (the note stays in the transcript).
  const approveTools = () => setPendingTools(null);

  /// Deny a proposed tool call: append a brief "don't run that" user turn
  /// and continue so the model answers directly.
  const denyTools = async () => {
    if (streaming === "streaming") return;
    setPendingTools(null);
    const denyMsg: ChatMessage = {
      role: "user",
      content: "Please don't run that tool — answer directly instead.",
    };
    const newHistory = [...history, denyMsg];
    setHistory(newHistory);
    setError(null);
    setPending("");
    setStreaming("streaming");
    const convId = activeConvId;
    if (historyOn && convId !== null) {
      try {
        await appendMessage(convId, "user", denyMsg.content);
      } catch (e) {
        console.warn("history persist failed:", e);
      }
    }
    await streamAssistantTurn(newHistory, convId);
  };

  const send = async () => {
    if (!draft.trim() || streaming === "streaming") return;
    const userText = draft.trim();
    const userMsg: ChatMessage =
      attachedImages.length > 0
        ? { role: "user", content: userText, images: attachedImages }
        : { role: "user", content: userText };
    setAttachedImages([]);
    // On the first turn of a conversation, if a system-prompt entry
    // is selected, materialize it as the conversation's leading
    // message. Subsequent turns inherit it because it's now in
    // `history`. Switching prompts mid-conversation is intentionally
    // a no-op — the system message belongs to the conversation, not
    // the live dropdown.
    const startingFresh = history.length === 0;
    const picked =
      startingFresh && selectedPromptName
        ? systemPrompts.find((p) => p.name === selectedPromptName)
        : null;
    const messagesToPrepend: ChatMessage[] = picked
      ? [{ role: "system", content: picked.body }]
      : [];
    const newHistory = [...messagesToPrepend, ...history, userMsg];
    setHistory(newHistory);
    setDraft("");
    setError(null);
    setPending("");
    // A fresh manual send supersedes any pending clarify / tool prompt.
    setPendingQuestion(null);
    setPendingTools(null);
    setStreaming("streaming");

    // Persist the user message immediately so a network blip doesn't
    // lose it. Create the conversation on the first turn — the
    // conversation's title is the first user message, truncated.
    let convId = activeConvId;
    if (historyOn) {
      try {
        if (convId === null) {
          const title =
            userText.length > 60 ? userText.slice(0, 57) + "…" : userText;
          convId = await createConversation(title);
          setActiveConvId(convId);
        }
        // Persist the system prompt first so the conversation
        // sidebar's full transcript reconstructs the same context
        // shape as the in-memory `history`. Skipped silently when
        // the appendMessage flow is unavailable.
        if (picked) {
          await appendMessage(convId, "system", picked.body);
        }
        await appendMessage(convId, "user", userText);
      } catch (e) {
        // History errors shouldn't break the chat. Surface to console
        // and continue with the in-memory transcript.
        console.warn("history persist failed:", e);
      }
    }

    // Smart compaction: if this turn would overflow ~75% of the model's
    // context window, ask the model to summarize the older turns first,
    // then send the compacted transcript. Best-effort — on failure we
    // send the full history and let the server truncate as before.
    let toSend = newHistory;
    if (ctxBudget && estTokens(newHistory) > ctxBudget * 0.75) {
      setCompacting(true);
      try {
        const compacted = await compactContext(newHistory, ctxBudget);
        if (compacted !== newHistory) {
          toSend = compacted;
          setHistory(compacted);
        }
      } catch (e) {
        console.warn("auto-compaction failed:", e);
      } finally {
        setCompacting(false);
      }
    }

    await streamAssistantTurn(toSend, convId);
  };

  const stop = () => {
    cancelAndAbortInFlight();
  };

  /// Re-run generation against the conversation as if the last
  /// assistant turn had never happened. Behaves like the CLI's
  /// `/regenerate` command: pop the last assistant message, stream
  /// a new one, append it. The dropped message is gone from the
  /// in-memory transcript but persists in the sqlite conversation
  /// history (the schema is append-only by design) — so the user
  /// gets a clean re-roll in the UI while the audit trail keeps
  /// every attempt. For a full edit/branch tree we'd need new
  /// endpoints; that's tracked as the M-sized roadmap follow-up.
  const regenerateLast = async () => {
    if (streaming === "streaming") return;
    // Find the trailing assistant message; bail if there isn't one
    // (e.g. the user hasn't sent anything yet, or the last send
    // errored before producing content).
    let lastAssistantIdx = -1;
    for (let i = history.length - 1; i >= 0; i--) {
      if (history[i].role === "assistant") {
        lastAssistantIdx = i;
        break;
      }
    }
    if (lastAssistantIdx === -1) return;
    const trimmed = history.slice(0, lastAssistantIdx);
    if (!trimmed.some((m) => m.role === "user")) return;

    setHistory(trimmed);
    setError(null);
    setPending("");
    setStreaming("streaming");

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    requestIdRef.current = null;
    let acc = "";
    const convId = activeConvId;
    await streamChat(
      trimmed,
      {
        temperature,
        maxTokens,
        seed: seed.trim() === "" ? undefined : Number(seed),
        // CLARIFY opt-in: routes through the tools path (ask_user only).
        allowClarify: clarify,
        abort: ctrl.signal,
      },
      {
        onStart: (id) => {
          requestIdRef.current = id;
        },
        onContent: (delta) => {
          acc += delta;
          setPending(acc);
        },
        onDone: async (info) => {
          if (acc.length > 0) {
            setHistory((h) => [...h, { role: "assistant", content: acc }]);
            if (historyOn && convId !== null) {
              try {
                await appendMessage(convId, "assistant", acc);
                await refreshList();
              } catch (e) {
                console.warn("history persist failed:", e);
              }
            }
          }
          setPending("");
          setStreaming("idle");
          requestIdRef.current = null;
          if (info?.usage) setLastUsage(info.usage);
          if (info?.systemFingerprint) {
            const next = info.systemFingerprint;
            setLastFingerprint((prev) => {
              if (prev && prev !== next) {
                setFingerprintChanged({ prev, current: next });
              }
              return next;
            });
          }
        },
        onError: async (e) => {
          if (e.name === "AbortError" || /aborted/i.test(e.message)) {
            if (acc.length > 0) {
              setHistory((h) => [...h, { role: "assistant", content: acc }]);
              if (historyOn && convId !== null) {
                try {
                  await appendMessage(convId, "assistant", acc);
                  await refreshList();
                } catch {
                  /* ignore */
                }
              }
            }
            setPending("");
            setStreaming("idle");
            requestIdRef.current = null;
            return;
          }
          setError(e.message);
          setPending("");
          setStreaming("error");
          requestIdRef.current = null;
        },
      },
    );
  };

  // Manual "Compact" — summarize the conversation on demand. Auto
  // compaction also runs inside `send` when a turn would overflow.
  const handleCompact = async () => {
    if (streaming === "streaming" || compacting) return;
    if (history.filter((m) => m.role !== "system").length < 3) return;
    setCompacting(true);
    setError(null);
    try {
      const compacted = await compactContext(history, ctxBudget);
      if (compacted !== history) setHistory(compacted);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setCompacting(false);
    }
  };

  const renderMsg = (
    m: ChatMessage,
    key: string,
    opts?: { showRegenerate?: boolean },
  ) => (
    <div
      key={key}
      style={{
        margin: "12px 0",
        padding: "10px 14px",
        background: m.role === "user" ? "var(--ll-bg-elev)" : "var(--ll-bg)",
        borderLeft: m.role === "user" ? "2px solid var(--ll-accent)" : "2px solid var(--ll-green)",
        borderRadius: 4,
        // User messages stay verbatim so literal `*` / `_` aren't
        // re-interpreted by the markdown renderer. Assistant output
        // is markdown by convention; the inner `.markdown` class gets
        // its own block/list/code styling so we don't need `pre-wrap`
        // here.
        whiteSpace: m.role === "user" ? "pre-wrap" : "normal",
        fontSize: 14,
        lineHeight: 1.6,
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          marginBottom: 4,
        }}
      >
        <div
          style={{
            fontSize: 11,
            color: "var(--ll-text-muted)",
            textTransform: "uppercase",
            letterSpacing: 0.4,
            flex: 1,
          }}
        >
          {m.role}
        </div>
        {opts?.showRegenerate && (
          <button
            onClick={regenerateLast}
            disabled={streaming === "streaming"}
            style={btnGhost}
            title="Re-run the last user turn — replaces this response with a fresh sample (⌘/Ctrl+R)"
          >
            ↻ Regenerate
          </button>
        )}
      </div>
      {m.role === "user"
        ? m.content
        : (() => {
            // Reasoning models: collapse <think>…</think> to a chip and
            // render only the final answer as markdown.
            const { thinking, answer, hasThinking, thinkingDone } =
              splitThinking(m.content);
            return (
              <>
                {hasThinking && (
                  <ThinkingBlock thinking={thinking} done={thinkingDone} />
                )}
                {answer && (
                  <div className="markdown">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      rehypePlugins={[rehypeHighlight]}
                    >
                      {answer}
                    </ReactMarkdown>
                  </div>
                )}
              </>
            );
          })()}
    </div>
  );

  const sidebarWidth = historyOn ? 240 : 0;

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: `${historyOn ? `${sidebarWidth}px ` : ""}1fr${settingsOpen ? " 300px" : ""}`,
        // Fill the `.ll-main` grid cell exactly (which is 100vh minus the
        // status bar). Using 100vh here overflowed `.ll-main` by the
        // status-bar height, so `.ll-main` scrolled the whole page and
        // pushed the model-loader bar off the top.
        height: "100%",
        minHeight: 0,
      }}
    >
      {historyOn && (
        <aside style={{ borderRight: "1px solid var(--ll-border)", overflowY: "auto", background: "var(--ll-bg)" }}>
          <div style={{ padding: "12px 14px", borderBottom: "1px solid var(--ll-border)", display: "flex", alignItems: "center", gap: 8 }}>
            <button onClick={startNew} style={btnSecondary} disabled={streaming === "streaming"}>
              + New
            </button>
            <button onClick={refreshList} style={btnGhost}>
              ↻
            </button>
          </div>
          {conversations.length === 0 && (
            <div style={{ padding: 14, fontSize: 12, color: "var(--ll-text-muted)" }}>No conversations yet.</div>
          )}
          {conversations.map((c) => (
            <div
              key={c.id}
              onClick={() => loadConv(c.id)}
              style={{
                padding: "10px 14px",
                borderBottom: "1px solid var(--ll-border)",
                cursor: streaming === "streaming" ? "not-allowed" : "pointer",
                background: c.id === activeConvId ? "var(--ll-bg-elev)" : "transparent",
                borderLeft: c.id === activeConvId ? "2px solid var(--ll-accent)" : "2px solid transparent",
                fontSize: 13,
                display: "flex",
                alignItems: "center",
                gap: 8,
              }}
            >
              <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {c.title || "(untitled)"}
              </span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  removeConv(c.id);
                }}
                style={btnTiny}
                title="Delete"
              >
                ×
              </button>
            </div>
          ))}
        </aside>
      )}

      <div style={{ display: "flex", flexDirection: "column", height: "100%", minHeight: 0, minWidth: 0 }}>
        <ModelLoaderBar
          activeModelId={modelId || undefined}
          onChanged={async () => {
            try {
              const r = await fetch("http://127.0.0.1:11434/healthz");
              const j = await r.json();
              const id = typeof j.model_id === "string" ? j.model_id : "";
              setHasModel(id.length > 0);
              setModelId(id);
            } catch {
              /* the 5s health poll will catch up */
            }
          }}
        />
        <header style={{ padding: "14px 20px", borderBottom: "1px solid var(--ll-border)", display: "flex", alignItems: "center", gap: 12 }}>
          <h2 style={{ margin: 0, fontSize: 16, fontWeight: 600 }}>Chat</h2>
          {!historyOn && (
            <button onClick={startNew} disabled={streaming === "streaming"} style={btnSecondary}>
              New chat
            </button>
          )}
          <button
            onClick={() => setShowShortcuts(true)}
            style={btnGhost}
            title="Keyboard shortcuts (press `?`)"
          >
            ⌨ Shortcuts
          </button>
          <button
            onClick={() => setSettingsOpen((v) => !v)}
            style={settingsOpen ? btnSecondary : btnGhost}
            title="Inference settings — temperature, max tokens, seed"
          >
            ⚙ Settings
          </button>
          <label
            title="When on, tool calls the model proposes are accepted without asking. This chat never executes tools; it only gates the confirm prompt."
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              fontSize: 12,
              color: "var(--ll-text-muted)",
              cursor: "pointer",
              userSelect: "none",
            }}
          >
            <input
              type="checkbox"
              checked={autoTools}
              onChange={toggleAutoTools}
              style={{ accentColor: "var(--ll-accent)", cursor: "pointer" }}
            />
            Auto-run tools
          </label>
          <label
            title="When on, the model can pause and ask you a clarifying question with selectable options instead of guessing. Turning it off sends the plain request with no tools."
            style={{
              display: "flex",
              alignItems: "center",
              gap: 6,
              fontSize: 12,
              color: "var(--ll-text-muted)",
              cursor: "pointer",
              userSelect: "none",
            }}
          >
            <input
              type="checkbox"
              checked={clarify}
              onChange={toggleClarify}
              style={{ accentColor: "var(--ll-accent)", cursor: "pointer" }}
            />
            Clarifying questions
          </label>
          {history.filter((m) => m.role !== "system").length >= 3 && (
            <button
              onClick={handleCompact}
              style={btnGhost}
              title="Ask the model to summarize older turns so the conversation fits the context window"
              disabled={streaming === "streaming" || compacting}
            >
              {compacting ? "Compacting…" : "⚗ Compact"}
            </button>
          )}
          {history.length > 0 && (
            <>
              <button
                onClick={() =>
                  downloadTextFile(
                    buildExportFilename("md"),
                    buildExportMarkdown(history, modelId),
                    "text/markdown",
                  )
                }
                style={btnGhost}
                title="Download conversation as Markdown"
                disabled={streaming === "streaming"}
              >
                Export MD
              </button>
              <button
                onClick={() =>
                  downloadTextFile(
                    buildExportFilename("json"),
                    buildExportJson(history, modelId),
                    "application/json",
                  )
                }
                style={btnGhost}
                title="Download conversation as JSON"
                disabled={streaming === "streaming"}
              >
                Export JSON
              </button>
            </>
          )}
          {lastUsage && (
            <span style={{ marginLeft: "auto", fontSize: 11, color: "var(--ll-text-muted)", fontFamily: "ui-monospace, monospace" }}>
              {lastUsage.completion_tokens} tok
              {typeof lastUsage.decode_ms === "number" && lastUsage.decode_ms > 0 &&
                ` · ${(lastUsage.completion_tokens / (lastUsage.decode_ms / 1000)).toFixed(1)} tok/s`}
              {typeof lastUsage.prefill_ms === "number" && lastUsage.prefill_ms > 0 &&
                ` · ${lastUsage.prefill_ms.toFixed(0)} ms prefill`}
              {typeof lastUsage.cache_hit_tokens === "number" && lastUsage.cache_hit_tokens > 0 &&
                ` · ${lastUsage.cache_hit_tokens} cached`}
            </span>
          )}
        </header>
        {hasModel === false && (
          <div
            style={{
              margin: "12px 20px",
              padding: "12px 14px",
              background: "rgba(210, 153, 34, 0.08)",
              border: "1px solid rgba(210, 153, 34, 0.5)",
              borderRadius: 4,
              color: "var(--ll-yellow)",
              fontSize: 13,
              lineHeight: 1.5,
            }}
          >
            <div style={{ fontWeight: 600, marginBottom: 6 }}>No model loaded</div>
            <div style={{ color: "var(--ll-text)" }}>
              The server is up but has no default model. Open the{" "}
              <a href="#/models" style={{ color: "var(--ll-accent)" }}>Models</a> page
              to load one — pull from HuggingFace if you don't have a GGUF cached, then click <em>Load</em>.
            </div>
          </div>
        )}
        <div style={{ flex: 1, overflowY: "auto", padding: "0 20px" }}>
          {history.length === 0 && pending.length === 0 && (
            <div style={{ color: "var(--ll-text-muted)", padding: "40px 0", textAlign: "center", fontSize: 13 }}>
              Start a conversation. Streamed via /v1/chat/completions.
              {historyOn && " Persisted automatically."}
            </div>
          )}
          {(() => {
            // Compute the trailing assistant index once; only that
            // message gets a Regenerate button. Also suppress the
            // button when a stream is in flight — the rendered `pending`
            // block represents an even-newer attempt that's about to
            // become the new trailing assistant.
            let lastAssistantIdx = -1;
            if (pending.length === 0 && streaming !== "streaming") {
              for (let i = history.length - 1; i >= 0; i--) {
                if (history[i].role === "assistant") {
                  lastAssistantIdx = i;
                  break;
                }
              }
            }
            return history.map((m, i) =>
              renderMsg(m, `m${i}`, { showRegenerate: i === lastAssistantIdx }),
            );
          })()}
          {pending.length > 0 && renderMsg({ role: "assistant", content: pending }, "pending")}
          {pendingQuestion && (
            <div
              style={{
                margin: "12px 0",
                padding: "12px 14px",
                background: "var(--ll-bg-elev)",
                border: "1px solid var(--ll-border-strong)",
                borderRadius: 6,
              }}
            >
              <div style={{ fontSize: 13, marginBottom: 10, color: "var(--ll-text)" }}>
                {pendingQuestion.prompt || "Choose an option:"}
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {pendingQuestion.options.map((opt, i) => (
                  <button
                    key={i}
                    onClick={() => chooseOption(opt)}
                    disabled={streaming === "streaming"}
                    style={btnSecondary}
                  >
                    {opt}
                  </button>
                ))}
              </div>
            </div>
          )}
          {pendingTools && (
            <div
              style={{
                margin: "12px 0",
                padding: "12px 14px",
                background: "var(--ll-yellow-soft)",
                border: "1px solid var(--ll-yellow)",
                borderRadius: 6,
              }}
            >
              <div style={{ fontSize: 13, marginBottom: 10, color: "var(--ll-text)" }}>
                The model proposes a tool call:{" "}
                <code>{describeToolCalls(pendingTools)}</code>. This chat can't
                run tools — approve to acknowledge, or deny to have it answer
                directly.
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <button
                  onClick={approveTools}
                  disabled={streaming === "streaming"}
                  style={btnPrimary}
                >
                  Approve
                </button>
                <button
                  onClick={denyTools}
                  disabled={streaming === "streaming"}
                  style={btnSecondary}
                >
                  Deny
                </button>
              </div>
            </div>
          )}
          {error && (
            <div style={{ margin: "12px 0", padding: "10px 14px", background: "var(--ll-red-soft)", border: "1px solid var(--ll-red)", borderRadius: 4, color: "var(--ll-red)", fontSize: 13 }}>
              error: {error}
            </div>
          )}
          {fingerprintChanged && (
            <div
              style={{
                margin: "12px 0",
                padding: "10px 14px",
                background: "var(--ll-yellow-soft)",
                border: "1px solid var(--ll-yellow)",
                borderRadius: 4,
                color: "var(--ll-yellow)",
                fontSize: 13,
                display: "flex",
                alignItems: "center",
                gap: 12,
              }}
            >
              <span style={{ flex: 1 }}>
                Backend config changed mid-conversation: <code>{fingerprintChanged.prev}</code>{" "}
                → <code>{fingerprintChanged.current}</code>. The model id, KV
                dtype, or server version was swapped — earlier turns and
                later turns may not be comparable.
              </span>
              <button
                onClick={() => setFingerprintChanged(null)}
                style={{
                  background: "transparent",
                  border: "1px solid var(--ll-yellow)",
                  color: "var(--ll-yellow)",
                  padding: "4px 10px",
                  borderRadius: 4,
                  fontSize: 12,
                  cursor: "pointer",
                }}
              >
                dismiss
              </button>
            </div>
          )}
          <div ref={bottomRef} />
        </div>
        <footer style={{ borderTop: "1px solid var(--ll-border)", padding: "12px 20px" }}>
          {systemPrompts.length > 0 && (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginBottom: 8,
                fontSize: 11,
                color: "var(--ll-text-muted)",
              }}
            >
              <span>System prompt:</span>
              <select
                value={selectedPromptName}
                onChange={(e) => {
                  setSelectedPromptName(e.target.value);
                  setUserPickedPrompt(true);
                }}
                disabled={history.length > 0}
                title={
                  history.length > 0
                    ? "Start a new conversation to change the system prompt"
                    : "Prepended to new conversations as a system-role message"
                }
                style={{
                  background: "var(--ll-bg)",
                  color: "var(--ll-text)",
                  border: "1px solid var(--ll-border-strong)",
                  borderRadius: 4,
                  padding: "2px 6px",
                  fontFamily: "inherit",
                  fontSize: 12,
                  opacity: history.length > 0 ? 0.5 : 1,
                }}
              >
                <option value="">— none —</option>
                {systemPrompts.map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.name}
                    {p.default_for_model === modelId && p.default_for_model
                      ? "  (default for this model)"
                      : ""}
                  </option>
                ))}
              </select>
              {history.length > 0 && (
                <span style={{ color: "var(--ll-text-faint)" }}>
                  (locked for this conversation)
                </span>
              )}
            </div>
          )}
          {attachedImages.length > 0 && (
            <div style={{ display: "flex", gap: 6, marginBottom: 6, flexWrap: "wrap" }}>
              {attachedImages.map((src, i) => (
                <div key={i} style={{ position: "relative" }}>
                  <img
                    src={src}
                    alt={`attachment ${i + 1}`}
                    style={{
                      width: 56,
                      height: 56,
                      objectFit: "cover",
                      borderRadius: 4,
                      border: "1px solid var(--ll-border-strong)",
                    }}
                  />
                  <button
                    onClick={() =>
                      setAttachedImages((prev) => prev.filter((_, j) => j !== i))
                    }
                    title="Remove image"
                    style={{
                      position: "absolute",
                      top: -6,
                      right: -6,
                      width: 18,
                      height: 18,
                      lineHeight: "14px",
                      fontSize: 11,
                      borderRadius: 9,
                      border: "1px solid var(--ll-border-strong)",
                      background: "var(--ll-bg-elev)",
                      color: "var(--ll-text)",
                      cursor: "pointer",
                      padding: 0,
                    }}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}
          <div style={{ display: "flex", gap: 8 }}>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg"
              multiple
              style={{ display: "none" }}
              onChange={(e) => {
                attachImageFiles(e.target.files);
                e.target.value = "";
              }}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={streaming === "streaming"}
              title="Attach image (PNG/JPEG, needs a vision-enabled model — set [model].mmproj)"
              style={{
                padding: "0 10px",
                background: "var(--ll-bg-elev)",
                color: "var(--ll-text)",
                border: "1px solid var(--ll-border-strong)",
                borderRadius: 4,
                cursor: "pointer",
                fontSize: 16,
              }}
            >
              🖼
            </button>
            <textarea
              ref={textareaRef}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send();
                }
              }}
              placeholder="Message…"
              rows={2}
              style={{
                flex: 1,
                padding: 10,
                background: "var(--ll-bg)",
                color: "var(--ll-text)",
                border: "1px solid var(--ll-border-strong)",
                borderRadius: 4,
                fontSize: 14,
                fontFamily: "inherit",
                resize: "vertical",
              }}
            />
            {streaming === "streaming" ? (
              <button onClick={stop} style={btnStop}>Stop</button>
            ) : (
              <button onClick={send} disabled={!draft.trim()} style={btnPrimary}>
                Send
              </button>
            )}
          </div>
          <div style={{ marginTop: 6, fontSize: 11, color: "var(--ll-text-muted)", display: "flex", justifyContent: "space-between" }}>
            <span>
              {draft.length > 0 ? (
                draftTokens !== null ? (
                  <DraftMeter
                    tokens={draftTokens}
                    chars={draft.length}
                    ctxBudget={ctxBudget}
                  />
                ) : (
                  <>{draft.length} chars · counting tokens…</>
                )
              ) : (
                <>Shift+Enter for a newline</>
              )}
            </span>
            <span>{streaming === "streaming" ? "streaming…" : ""}</span>
          </div>
        </footer>
      </div>

      {settingsOpen && (
        <aside
          style={{
            borderLeft: "1px solid var(--ll-border)",
            background: "var(--ll-bg)",
            overflowY: "auto",
            padding: "18px 18px 24px",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              marginBottom: 14,
            }}
          >
            <h3 style={{ margin: 0, fontSize: 14, fontWeight: 600, flex: 1 }}>
              Inference
            </h3>
            <button
              onClick={() => setSettingsOpen(false)}
              style={btnTiny}
              title="Close"
            >
              ×
            </button>
          </div>

          <label
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontSize: 13,
              fontWeight: 500,
              margin: "12px 0 6px",
            }}
          >
            <span>Temperature</span>
            <span
              style={{
                color: "var(--ll-accent)",
                fontVariantNumeric: "tabular-nums",
              }}
            >
              {temperature.toFixed(2)}
            </span>
          </label>
          <input
            type="range"
            min={0}
            max={2}
            step={0.05}
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
            style={{ width: "100%", accentColor: "var(--ll-accent)" }}
          />
          <p
            style={{
              margin: "6px 0 0",
              fontSize: 11,
              color: "var(--ll-text-faint)",
              lineHeight: 1.4,
            }}
          >
            0 = deterministic; higher = more varied.
          </p>

          <label
            style={{ display: "block", fontSize: 13, fontWeight: 500, margin: "16px 0 6px" }}
          >
            Max tokens
          </label>
          <input
            type="number"
            min={1}
            max={32768}
            value={maxTokens}
            onChange={(e) =>
              setMaxTokens(Math.max(1, Math.floor(Number(e.target.value) || 1)))
            }
            style={{
              width: "100%",
              boxSizing: "border-box",
              padding: "7px 10px",
              background: "var(--ll-bg-elev-2)",
              border: "1px solid var(--ll-border-strong)",
              borderRadius: 7,
              color: "var(--ll-text)",
              fontSize: 13,
            }}
          />
          <p
            style={{
              margin: "6px 0 0",
              fontSize: 11,
              color: "var(--ll-text-faint)",
              lineHeight: 1.4,
            }}
          >
            Upper bound on tokens generated per turn.
          </p>

          <label
            style={{ display: "block", fontSize: 13, fontWeight: 500, margin: "16px 0 6px" }}
          >
            Seed
          </label>
          <input
            type="text"
            inputMode="numeric"
            placeholder="random"
            value={seed}
            onChange={(e) => setSeed(e.target.value.replace(/[^0-9]/g, ""))}
            style={{
              width: "100%",
              boxSizing: "border-box",
              padding: "7px 10px",
              background: "var(--ll-bg-elev-2)",
              border: "1px solid var(--ll-border-strong)",
              borderRadius: 7,
              color: "var(--ll-text)",
              fontSize: 13,
            }}
          />
          <p
            style={{
              margin: "6px 0 0",
              fontSize: 11,
              color: "var(--ll-text-faint)",
              lineHeight: 1.4,
            }}
          >
            Fixed seed → reproducible sampling. Blank = random each turn.
          </p>
        </aside>
      )}

      {showShortcuts && (
        <ShortcutsOverlay onClose={() => setShowShortcuts(false)} />
      )}
    </div>
  );
}

function ShortcutsOverlay({ onClose }: { onClose: () => void }) {
  // Use mac key labels when running on macOS so the user sees the
  // right modifier. Tauri's webview reports navigator.platform like
  // a browser would; fall through to "Ctrl" on anything we can't
  // identify.
  const isMac =
    typeof navigator !== "undefined" &&
    /mac/i.test(navigator.platform || navigator.userAgent || "");
  const mod = isMac ? "⌘" : "Ctrl";
  const rows: { keys: string; what: string }[] = [
    { keys: "Enter", what: "Send message" },
    { keys: "Shift + Enter", what: "Insert a newline in the message" },
    { keys: `${mod} + L`, what: "Focus the chat input" },
    { keys: `${mod} + K`, what: "Start a new conversation (aborts any in-flight stream)" },
    { keys: `${mod} + E`, what: "Export current conversation as Markdown" },
    { keys: `${mod} + R`, what: "Regenerate the last assistant response" },
    { keys: "?", what: "Toggle this shortcuts overlay" },
    { keys: "Esc", what: "Close this overlay" },
  ];
  return (
    <div style={shortcutsBackdrop} onClick={onClose}>
      <div style={shortcutsPanel} onClick={(e) => e.stopPropagation()}>
        <header style={shortcutsHeader}>
          <h3 style={{ margin: 0, fontSize: 14, fontWeight: 600 }}>Keyboard shortcuts</h3>
          <button onClick={onClose} style={btnGhost} aria-label="Close shortcuts">
            ✕
          </button>
        </header>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
          <tbody>
            {rows.map((r) => (
              <tr key={r.keys} style={{ borderTop: "1px solid var(--ll-border)" }}>
                <td style={{ padding: "8px 10px", width: 180 }}>
                  <kbd style={kbdStyle}>{r.keys}</kbd>
                </td>
                <td style={{ padding: "8px 10px", color: "var(--ll-text)" }}>{r.what}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

const shortcutsBackdrop: React.CSSProperties = {
  position: "fixed",
  inset: 0,
  background: "rgba(0, 0, 0, 0.6)",
  zIndex: 1000,
  display: "flex",
  alignItems: "flex-start",
  justifyContent: "center",
  paddingTop: 80,
};
const shortcutsPanel: React.CSSProperties = {
  width: "min(520px, 92vw)",
  background: "var(--ll-bg)",
  border: "1px solid var(--ll-border-strong)",
  borderRadius: 8,
  overflow: "hidden",
};
const shortcutsHeader: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  padding: "12px 16px",
  borderBottom: "1px solid var(--ll-border-strong)",
  background: "var(--ll-bg-elev)",
};
const kbdStyle: React.CSSProperties = {
  display: "inline-block",
  padding: "2px 8px",
  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
  fontSize: 11,
  background: "var(--ll-border)",
  color: "var(--ll-text)",
  border: "1px solid var(--ll-border-strong)",
  borderRadius: 3,
};

/// Render the draft meter line. Shows the token count and char
/// count always; when `ctxBudget` is known, also shows the percentage
/// of the model's context window the *draft alone* would consume, and
/// colors the percentage yellow at >50%, red at >90%. Excludes prior
/// history from the budget on purpose — tokenizing the whole
/// transcript on every keystroke is expensive, and even a rough draft-
/// only signal is enough to catch the common "I pasted a giant log
/// and it would blow the ctx" mistake.
function DraftMeter({
  tokens,
  chars,
  ctxBudget,
}: {
  tokens: number;
  chars: number;
  ctxBudget: number | null;
}) {
  const tokenWord = tokens === 1 ? "" : "s";
  const charWord = chars === 1 ? "" : "s";
  if (ctxBudget === null) {
    return (
      <>
        {tokens} token{tokenWord} · {chars} char{charWord}
      </>
    );
  }
  const pct = (tokens / ctxBudget) * 100;
  const tone =
    pct >= 100
      ? "var(--ll-red)"
      : pct >= 90
        ? "var(--ll-red)"
        : pct >= 50
          ? "var(--ll-yellow)"
          : "var(--ll-text-muted)";
  const pctLabel = pct >= 100 ? `${pct.toFixed(0)}% — exceeds ctx_size` : `${pct.toFixed(pct < 10 ? 1 : 0)}% of ctx`;
  return (
    <>
      {tokens} / {ctxBudget.toLocaleString()} token{tokenWord} ·{" "}
      <span style={{ color: tone, fontWeight: pct >= 90 ? 600 : 400 }}>
        {pctLabel}
      </span>{" "}
      · {chars} char{charWord}
    </>
  );
}

// ============================================================
// Conversation export helpers
// ============================================================
//
// Two output formats: Markdown for human reading + sharing, JSON
// for re-import or scripted processing. Both serialize the live
// `history` array (the in-memory ChatMessage[] used by the chat
// loop) without touching the persisted-conversation sqlite —
// users can export ad-hoc sessions that were never saved.

/// Build a Markdown rendering of the conversation suitable for
/// pasting into a doc, file, or chat. Roles are bold-labeled;
/// turns are separated by `---` rules. ISO timestamp + model id
/// in the header so the export carries its provenance.
function buildExportMarkdown(history: ChatMessage[], modelId: string): string {
    const now = new Date().toISOString();
    const head = [
        `# Conversation`,
        ``,
        `Exported ${now} from rustllama${modelId ? ` (model: \`${modelId}\`)` : ""}.`,
        ``,
        `---`,
        ``,
    ].join("\n");
    const body = history
        .map((m) => `**${m.role}:**\n\n${m.content}`)
        .join("\n\n---\n\n");
    return head + body + "\n";
}

/// Build a JSON dump of the conversation — same shape callers
/// can later re-feed into `/v1/chat/completions` (the `messages`
/// array is OpenAI-compatible). Pretty-printed with 2-space
/// indent so a human can read it; round-trips through
/// `JSON.parse` unchanged.
function buildExportJson(history: ChatMessage[], modelId: string): string {
    return JSON.stringify(
        {
            exported_at: new Date().toISOString(),
            model_id: modelId || null,
            messages: history,
        },
        null,
        2,
    );
}

/// File-safe timestamped filename: `rustllama-chat-2026-05-17T14-30-15.md`.
/// Colons + dots in the ISO timestamp would break some Windows
/// filename conventions, so they're swapped for hyphens; only the
/// `YYYY-MM-DDTHH-MM-SS` prefix is kept (drops the sub-second + Z).
function buildExportFilename(ext: string): string {
    const iso = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    return `rustllama-chat-${iso}.${ext}`;
}

/// Trigger a browser download of `content` as `filename`. Works
/// in the Tauri 2 webview (which supports the standard anchor-
/// download flow). Revokes the blob URL after the click to free
/// the resource.
function downloadTextFile(filename: string, content: string, mime: string): void {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

const btnPrimary: React.CSSProperties = {
  padding: "0 18px",
  background: "var(--ll-accent)",
  color: "white",
  border: "1px solid var(--ll-accent)",
  borderRadius: 7,
  cursor: "pointer",
  fontSize: 14,
  fontWeight: 500,
};

const btnSecondary: React.CSSProperties = {
  padding: "6px 14px",
  background: "transparent",
  color: "var(--ll-text)",
  border: "1px solid var(--ll-border-strong)",
  borderRadius: 4,
  cursor: "pointer",
  fontSize: 12,
};

const btnStop: React.CSSProperties = {
  padding: "0 18px",
  background: "var(--ll-red)",
  color: "white",
  border: "1px solid var(--ll-red)",
  borderRadius: 4,
  cursor: "pointer",
  fontSize: 14,
};

const btnGhost: React.CSSProperties = {
  padding: "4px 10px",
  background: "transparent",
  color: "var(--ll-text-muted)",
  border: "1px solid var(--ll-border-strong)",
  borderRadius: 4,
  cursor: "pointer",
  fontSize: 12,
};

const btnTiny: React.CSSProperties = {
  padding: "2px 8px",
  background: "transparent",
  color: "var(--ll-text-muted)",
  border: "1px solid var(--ll-border-strong)",
  borderRadius: 3,
  cursor: "pointer",
  fontSize: 12,
};
