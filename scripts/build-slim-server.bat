@echo off
REM Slim server build (opt Tranche 6): the headless deployment binary.
REM Strips everything a server-only host doesn't need, on top of the
REM `dist` profile (fat LTO, debuginfo stripped):
REM
REM   --no-default-features drops (all default-ON in normal builds):
REM     encoder      quantize/re-encode pipeline + IQ grid-search
REM                  tables (~3-6 MB .text). `rustllama quantize`
REM                  reports the feature as unavailable.
REM     history      bundled sqlite conversation store (~10.9 MB
REM                  static). /api/conversations 501s; `rustllama
REM                  conv` reports unavailable.
REM     tree-sitter  code-aware RAG chunking grammars (rust+python).
REM                  RAG still works with the plain-text chunker.
REM   `gui` is never built here — the native egui GUI deps are skipped.
REM
REM   All three compute backends (SYCL + CUDA + CPU) are always compiled
REM   in — there is no CPU-only / no-oneAPI variant any more — so this
REM   still requires Intel oneAPI (icx) and the CUDA Toolkit (nvcc).
REM
REM Usage:  scripts\build-slim-server.bat  [extra cargo args]
REM Output: target\dist\rustllama.exe
call "%~dp0build-env.bat" build --profile dist -p rustllama --no-default-features --manifest-path app\desktop\Cargo.toml %*
