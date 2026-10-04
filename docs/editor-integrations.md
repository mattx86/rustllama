# Editor integrations

rustllama exposes three API surfaces — **OpenAI**, **Anthropic Messages**,
and **Ollama** — so most editor / coding-agent tools point at it with
zero special-casing. This document collects the exact config snippets
for the editors we've verified, plus the per-tool caveats worth knowing
about before you commit.

## What rustllama exposes

| Surface | Endpoints | Used by |
|---------|-----------|---------|
| OpenAI | `POST /v1/chat/completions`, `POST /v1/completions` (with FIM `suffix`), `GET /v1/models` | Continue, Aider, Cursor, Cline, Roo, most others |
| Anthropic | `POST /v1/messages` | Claude Code |
| Ollama | `POST /api/chat`, `POST /api/generate`, `POST /api/pull`, `POST /api/tags` | Continue (Ollama provider mode), Open WebUI, others |

**Default address:** `http://127.0.0.1:11434/v1` (for LAN access, run
`rustllama serve --ip 0.0.0.0 --port 11434`, or set
`[server].bind_addr = "0.0.0.0"` in `config.toml`).

**Auth:** the server runs **without** an API key by default, so editor configs
can pass any non-empty placeholder (e.g. `"sk-rustllama"`) and it is ignored.
To actually require a token — the usual case when exposing rustllama on a LAN —
set one and the server enforces `Authorization: Bearer <token>` on every
request (constant-time compared; `/healthz` stays open). Three ways to set it,
highest precedence first:

```bash
rustllama serve --api-key sk-secret       # 1. CLI flag (wins)
RUSTLLAMA_API_KEY=sk-secret rustllama serve   # 2. env var
#   [server]                              # 3. config.toml
#   api_key = "sk-secret"
```

When a key is set, loopback (`127.0.0.1`) clients are still exempt by default so
local tools keep working without a token; set `[server].require_auth_loopback =
true` to require it from local clients too. In every editor config below,
replace the placeholder `sk-rustllama` with your real key if you set one.

**Model id:** the server's `/v1/models` reports model ids derived from
the GGUF file stem (e.g. `qwen2.5-coder-7b-instruct-q4_k_m`). Editor
configs that name a model need this exact string — `GET /v1/models`
shows what's loaded.

---

## Continue.dev (VS Code extension)

Continue uses two separate model entries — one for chat, one for
tab-autocomplete (FIM). Both point at rustllama.

Edit `~/.continue/config.yaml`:

```yaml
models:
  - title: rustllama-qwen-coder
    provider: openai
    model: qwen2.5-coder-7b-instruct-q4_k_m
    apiBase: http://127.0.0.1:11434/v1
    apiKey: sk-rustllama
    contextLength: 8192
    completionOptions:
      temperature: 0.2

tabAutocompleteModel:
  title: rustllama-qwen-coder-fim
  provider: openai
  model: qwen2.5-coder-7b-instruct-q4_k_m
  apiBase: http://127.0.0.1:11434/v1
  apiKey: sk-rustllama
  template: qwen
```

Why `template: qwen` — Continue's tab-autocomplete uses
`/v1/completions` with `prompt` + `suffix`. rustllama's FIM endpoint
auto-detects Qwen-style FIM tokens (`<|fim_prefix|>`,
`<|fim_suffix|>`, `<|fim_middle|>`) and wraps the request
accordingly. DeepSeek-Coder + StarCoder/CodeLlama variants also auto-
detect; use `template: codellama` or omit when in doubt.

**Caveats:**
- If your loaded model has no FIM special tokens (most non-coding
  models), Continue's tab-autocomplete fires `/v1/completions` with
  `suffix` and gets back `400 Bad Request` with a clear error
  mentioning FIM. Use a coding-specific GGUF.
- Continue's "context length" hint should match `[inference].ctx_size`
  in your `config.toml` — mismatches cause silent truncation.

---

## Aider (CLI coding agent)

Aider speaks OpenAI through litellm. Two config paths:

**Inline:**

```bash
aider --openai-api-base http://127.0.0.1:11434/v1 \
      --openai-api-key sk-rustllama \
      --model openai/qwen2.5-coder-7b-instruct-q4_k_m
```

**Env vars (recommended for shell aliases):**

```bash
export OPENAI_API_BASE=http://127.0.0.1:11434/v1
export OPENAI_API_KEY=sk-rustllama
aider --model openai/qwen2.5-coder-7b-instruct-q4_k_m
```

The `openai/` prefix on the model id is litellm's provider marker —
required even though we're routing to rustllama. Without it litellm
tries the public OpenAI endpoint.

**Caveats:**
- Aider auto-detects the model's context window via `/v1/models` —
  if you see "context window too small" errors, confirm
  `[inference].ctx_size` matches the model card.
- Aider's `--no-stream` is useful for first-time setup so errors
  surface as HTTP status codes instead of streaming SSE that's
  harder to read.

---

## Cursor (closed-source IDE)

Cursor supports a "Custom OpenAI base URL" in `Settings → Models →
OpenAI API`. Point it at `http://127.0.0.1:11434/v1` with any non-
empty API key.

**Caveats — Cursor has the most rustllama-specific friction of the
listed editors:**
- Cursor calls `POST /v1/embeddings` for its semantic-code-search
  feature. rustllama serves a real `/v1/embeddings` (BERT/BGE, with
  base64 + MRL `dimensions`) once an embedding model is configured under
  `[embeddings]`; without one it returns HTTP 501. So the
  embeddings-dependent Cursor features (sidebar code search, `@codebase`
  references) work when an embedding model is loaded.
- Cursor's tab-autocomplete model is hardcoded to its proprietary
  models — you can't substitute rustllama for that path.
- Use Cursor when you want chat/edit features against a local model;
  use Continue if you also want local tab-autocomplete.

---

## Cline / Roo Code (VS Code extensions)

Both extensions support an "OpenAI Compatible" provider in their
settings UI. Configure with:

- **API Base URL**: `http://127.0.0.1:11434/v1`
- **API Key**: any non-empty string
- **Model ID**: the model name from `GET /v1/models`

**Caveats:**
- Roo Code's "auto-approve" features tied to tool-call grammar work
  with rustllama — the server returns proper `tool_calls` blocks via
  the chat-completions endpoint when `tools` is specified.
- Cline's "Use Browser Tool" and other agentic features depend on
  tool-call quality more than the API surface; pick a tool-tuned
  coding model (Qwen2.5-Coder, DeepSeek-Coder) for best results.

---

## OpenCode (terminal coding agent)

[OpenCode](https://opencode.ai) talks to any OpenAI-compatible endpoint through
a custom provider. Point it at rustllama's `/v1` base URL and give it the bearer
token you set with `--api-key` / `RUSTLLAMA_API_KEY` / `[server].api_key`; it is
sent as `Authorization: Bearer <token>`, which rustllama enforces.

In OpenCode's config (`~/.config/opencode/opencode.json`), define an
OpenAI-compatible provider:

```json
{
  "provider": {
    "rustllama": {
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:11434/v1",
        "apiKey": "sk-secret"
      },
      "models": {
        "qwen2.5-coder-7b-instruct-q4_k_m": {}
      }
    }
  }
}
```

The model id must match what `GET /v1/models` reports. `apiKey` must equal the
server's configured key (not a placeholder) whenever auth is on; over loopback
with the default `require_auth_loopback = false` any value works. Check
OpenCode's current provider docs for the exact config keys — the
OpenAI-compatible provider surface is what rustllama targets.

**Caveats:**
- For non-loopback use (another host on the LAN), start the server with
  `serve --ip 0.0.0.0 --api-key <token>` and use `http://<host>:11434/v1`.
- OpenCode's agentic flows lean on tool-call quality; pick a tool-tuned coding
  model (Qwen2.5-Coder, DeepSeek-Coder) for best results.

---

## Claude Code

rustllama implements the Anthropic Messages API at `POST /v1/messages`,
which Claude Code uses as its primary endpoint.

Set the environment variable Claude Code consults for its base URL:

```bash
export ANTHROPIC_BASE_URL=http://127.0.0.1:11434
export ANTHROPIC_API_KEY=sk-rustllama
```

Then launch Claude Code as usual. Model selection happens via the
session UI — use any model id that `/v1/models` reports.

**Caveats:**
- The Anthropic Messages API has more fields than rustllama
  surfaces (cache_control, batches, files API). The chat + tool-call
  paths are wired; advanced Anthropic-specific features may not be.
- Claude Code's default expects the Anthropic API. Some Claude-Code-
  specific behaviors (e.g. computer use, file uploads) are not
  available against a local model regardless of API compatibility.

---

## Ollama-compatible clients

rustllama implements the Ollama API at `/api/chat`, `/api/generate`,
`/api/pull`, `/api/tags`. Most Ollama clients can point at rustllama
on its default port (11434 happens to match Ollama's default — by
design, so swap-in works without reconfig).

Examples that work out of the box:
- Open WebUI (with `OLLAMA_API_BASE_URL=http://127.0.0.1:11434`)
- Ollama CLI itself (`OLLAMA_HOST=http://127.0.0.1:11434 ollama list`)
- Any tool that takes a `base_url` and speaks Ollama protocol

**Caveats:**
- `/api/embeddings` and `/api/embed` serve real embeddings (BERT/BGE)
  when an embedding model is configured under `[embeddings]`; without one
  they return HTTP 501. Embedding models are a separate model category
  from chat models, so Ollama-API clients that need embeddings (e.g. some
  RAG flows in Open WebUI) work once one is loaded.
- `/api/pull` proxies to HuggingFace via the same hub puller as
  `rustllama model pull` — pass `model: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:qwen2.5-coder-7b-instruct-q4_k_m.gguf"`
  shape strings rather than raw `qwen2.5-coder:7b` Ollama-style ids.

---

## Verifying a connection

Before committing time to an editor config, confirm rustllama is
serving:

```bash
# 1. Server up + model loaded?
curl http://127.0.0.1:11434/v1/models

# 2. Chat works?
curl http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<paste-model-id-from-step-1>","messages":[{"role":"user","content":"hello"}],"max_tokens":16}'

# 3. FIM works? (only on coding models)
curl http://127.0.0.1:11434/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<id>","prompt":"def hello():\n    ","suffix":"\nhello()","max_tokens":16}'
```

If step 3 returns `{"error":"... fill-in-middle special tokens ..."}`,
the loaded model isn't a coding model — Continue/Tabby tab-autocomplete
won't work against it. Step 1 + 2 working is enough for chat-only
flows (Aider, Cursor chat, Claude Code).

## Reporting integration issues

If an editor not listed here works against rustllama, send a PR
adding it to this file with the working config snippet. If one
listed here breaks, file an issue including the editor version,
the exact request the editor sent (proxy logs via `mitmproxy
--mode reverse:http://127.0.0.1:11434`), and rustllama's
`gui.log` from that session.
