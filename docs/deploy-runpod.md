# Running rustllama on RunPod (and other NVIDIA GPU clouds)

rustllama's `linux-x86_64` release is a self-contained OpenAI / Anthropic /
Ollama server, so it drops onto a RunPod **GPU Pod** cleanly. No rustllama code
changes are needed — just match two host requirements and expose the port.

## Two host requirements (the gotchas)

1. **glibc ≥ 2.39 → use an Ubuntu 24.04 base.** The release is built on
   Rocky 10 / glibc 2.39. Many default RunPod templates are **Ubuntu 22.04
   (glibc 2.35)** and the binary won't start there (`GLIBC_2.39 not found`).
2. **NVIDIA driver ≥ R580 on the host.** The build uses **CUDA 13.0 with static
   cudart**, so only the driver matters at runtime — but CUDA 13 needs R580+.
   On an older host (driver 535/550) CUDA init fails and rustllama **falls back
   to CPU automatically** (still works, just no GPU accel). Check the instance's
   driver before expecting GPU speed.

GPU-arch coverage is fine: the build ships SASS for sm_80/86/89/90 (A100, A40/
A5000/A6000/3090, L4/L40/4090-Ada, H100) plus forward-compatible PTX that JITs
onto newer cards (Blackwell). Validate once with `rustllama doctor --cuda-parity`
on the pod before trusting output on a GPU you haven't run before.

## Path A — bare GPU Pod (no image to build)

1. Launch a **GPU Pod** on an **Ubuntu 24.04 + CUDA 13** template (driver ≥ R580).
2. In the pod terminal:
   ```bash
   curl -L -o rl.tar.gz \
     https://github.com/mattx86/rustllama/releases/download/v0.1.0/rustllama-0.1.0-linux-x86_64.tar.gz
   tar xzf rl.tar.gz && cd rustllama-0.1.0-linux-x86_64
   ./rustllama doctor --cuda-parity          # validate GPU kernels on this card
   ./rustllama model pull Qwen/Qwen2.5-Coder-7B-Instruct-GGUF:qwen2.5-coder-7b-instruct-q4_k_m.gguf
   ./rustllama serve --ip 0.0.0.0 --port 8000 --api-key sk-secret --model <pulled.gguf>
   ```
3. Expose **port 8000** in the pod config. RunPod gives an HTTP proxy URL like
   `https://<pod-id>-8000.proxy.runpod.net`; call `…/v1/chat/completions` with
   `Authorization: Bearer sk-secret`. (If SSE streaming misbehaves through the
   HTTP proxy, expose the port as a **TCP** port instead.)

## Path B — container image (repeatable)

`scripts/Dockerfile.runpod` drops the release onto a CUDA 13 / Ubuntu 24.04 base:

```bash
docker build --build-arg RUSTLLAMA_VERSION=0.1.0 -t ghcr.io/<you>/rustllama -f scripts/Dockerfile.runpod .
docker push ghcr.io/<you>/rustllama
```

Then create a RunPod Pod/template from that image. Set `RUSTLLAMA_API_KEY` as an
env var and pass the model as container args (appended to `serve`):

```
--model /workspace/models/model.gguf
```

(Adjust the base tag in the Dockerfile to a CUDA-13.x `*-runtime-ubuntu24.04`
image that exists at build time.)

## Persistence — use a network volume

A pod's container storage is wiped on stop. Mount a **RunPod network volume**
(e.g. at `/workspace`) and keep the GGUFs there so you don't re-download them,
then point `--model` at the volume path. rustllama is binary-relative by
default, so the first-load **autotune cache** (`tuning/`) lives next to the
binary and is lost on a cold start unless the binary/state also lives on the
volume (Path A: install into `/workspace/rustllama/…`). Re-tuning on first load
is a one-time cost per boot otherwise.

## Notes

- **Bind address:** always `--ip 0.0.0.0` on a pod (the default `127.0.0.1`
  isn't reachable through the proxy). Set `--api-key` since the proxy URL is
  guessable.
- **Large models:** a 30B-class MoE at Q8_0 is ~35 GB — pick a GPU/instance with
  enough VRAM (or rely on rustllama's RAM+VRAM hybrid placement) and a volume
  with room for the download.
- **Serverless:** RunPod Serverless expects a handler, so a persistent **Pod**
  (above) fits rustllama's HTTP server best; a serverless wrapper would need a
  small adapter and isn't provided yet.
- See [editor-integrations.md](editor-integrations.md) to point coding tools at
  the exposed endpoint, and the README's "Binding & authentication" section.
