# Coming from llama.cpp

A quick map from the `llama.cpp` (`llama-cli` / `llama-server`) command-line
flags you know to where the same knob lives in rustllama.

## The one structural difference

`llama.cpp` puts everything on a single command line. rustllama is
**server-first** (closest to `llama-server`), so the flags split into two
groups:

- **Load-time settings** (context size, GPU offload, flash attention, KV-cache
  dtype, threads, batch) live in **`config.toml`** under `[model]` /
  `[inference]` / `[server]`. They apply when a model loads.
- **Per-request sampling** (temperature, top-k, top-p, penalties, seed,
  max tokens) is sent in **each API request's JSON body** (OpenAI / Anthropic /
  Ollama), or passed to the one-shot **`rustllama generate`** command. These are
  *not* flags on `rustllama serve`.

Second difference worth knowing: rustllama **autotunes** many of the load-time
knobs on a model's first load (GPU/CPU placement, flash attention + its
break-even threshold, KV-cache dtype, batch size, threads) and caches the
winners per machine. So the `-ngl` / `-fa` / `-ctk` / `-t` / `-b` dials you'd
set by hand in llama.cpp are usually chosen for you — set them explicitly only
to override the planner.

## Flag → rustllama map

| llama.cpp | What it does | rustllama equivalent |
|---|---|---|
| `-m, --model FILE` | pick the model | `[model].path` (or a hub ref), `rustllama serve --model FILE` (repeatable; first = default), or `rustllama model pull` + `model use` |
| `-c, --ctx-size N` | context window (`n_ctx`) | `[inference].ctx_size` (default **8192**) |
| `-ngl, --n-gpu-layers N` | layers to offload to GPU | `[inference].n_gpu_layers` (**999 = auto** placement; a number overrides the planner) |
| `-fa, --flash-attn` | fused flash-attention kernel | `[inference].flash_attention` (default **on**, autotuned; `[inference].flash_attention_kv_min` sets the KV-length break-even) |
| `-b, --batch-size N` | prompt batch size | `[inference].batch_size` (autotuned) |
| `-t, --threads N` | CPU threads | `[inference].threads` (autotuned via `tune --threads`) |
| `-ctk/-ctv, --cache-type-k/v` | KV-cache dtype (e.g. `q8_0`) | `[inference].kv_dtype` (both) or `k_dtype` / `v_dtype` (per-side); autotuned, with a coherence guardrail that can force `f32` |
| `--temp N` | sampling temperature | request `temperature` / `generate --temperature` (default **0.7**; `0` = greedy) |
| `--top-k N` | keep top-K tokens | request `top_k` / `generate --top-k` (default **40**; `0` = off). *(rustllama accepts `top_k` on its OpenAI endpoint as an extension.)* |
| `--top-p N` | nucleus sampling | request `top_p` / `generate --top-p` (default **0.95**; `1.0` = off) |
| `--min-p N` | min-p truncation | **not supported** (no equivalent today) |
| `--repeat-penalty N` | repetition penalty | request `repeat_penalty` / `generate --repeat-penalty` (default **1.1**; `1.0` = off) |
| `--mirostat N` (`--mirostat-ent`/`--mirostat-lr`) | Mirostat sampler | request `mirostat` (0/1/2), `mirostat_tau`, `mirostat_eta` |
| `-s, --seed N` | RNG seed | request `seed` / `generate --seed` |
| `-n, --predict N` | max new tokens (`n_predict`) | request `max_tokens` / `generate --max-tokens` |
| `--grammar` / `--json-schema` | constrained decoding | OpenAI `response_format: {"type":"json_object"}` (JSON-constrained). Tool/function calling also uses grammar under the hood; raw GBNF files aren't accepted directly. |
| `--host` / `--port` | bind address | `rustllama serve --ip A --port P`, or `[server].bind_addr` / `[server].port` (default `127.0.0.1:11434`) |
| `--api-key KEY` | require a bearer token | `rustllama serve --api-key KEY`, `RUSTLLAMA_API_KEY`, or `[server].api_key` (see [editor-integrations.md](editor-integrations.md)) |

## Worked examples

### Load-time settings — `config.toml`

```toml
[model]
path = "/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"

[inference]
ctx_size = 8192          # -c 8192
n_gpu_layers = 999       # -ngl: 999 = auto placement (or a specific count)
flash_attention = true   # -fa (on by default; autotuned)
batch_size = 512         # -b 512
# threads = 16           # -t 16  (omit to let the autotuner pick)
kv_dtype = "q8_0"        # -ctk q8_0 -ctv q8_0 (quantized KV cache)

[server]
bind_addr = "0.0.0.0"    # --host 0.0.0.0
port = 11434             # --port 11434
# api_key = "sk-secret"  # --api-key sk-secret
```

Then: `rustllama serve` (first load autotunes and caches the winners).

### Per-request sampling — HTTP (OpenAI-compatible)

The `--temp` / `--top-k` / `--top-p` / `-n` equivalents go in the request body,
exactly as you'd send them to `llama-server`:

```bash
curl http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-secret' \
  -d '{
    "model": "qwen2.5-coder-7b-instruct-q4_k_m",
    "messages": [{"role": "user", "content": "hello"}],
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "max_tokens": 256,
    "seed": 42
  }'
```

### Per-request sampling — one-shot CLI

`rustllama generate` mirrors a quick `llama-cli` run:

```bash
rustllama generate "write a haiku about llamas" \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.9 \
  --repeat-penalty 1.1 \
  --max-tokens 256 \
  --seed 42
```

## Defaults at a glance

| Knob | llama.cpp default | rustllama default |
|---|---|---|
| flash attention | auto (recent) / off (older) | **on**, autotuned, KV-min gated |
| temperature | 0.8 | **0.7** |
| top-k | 40 | **40** |
| top-p | 0.9 | **0.95** |
| ctx size | 4096 (0 = from model) | **8192** |
| repeat penalty | 1.1 | **1.1** |

## Beyond the basics

rustllama's sampler is a superset of the temp/top-k/top-p stack: it also has
`typical_p`, OpenAI-style `frequency_penalty` / `presence_penalty`, and
**Mirostat v1/v2**. The one notable gap vs llama.cpp is **min-p**. See
[editor-integrations.md](editor-integrations.md) for pointing coding tools
(Continue, Aider, OpenCode, Cursor, Cline/Roo, Claude Code) at a running server.
