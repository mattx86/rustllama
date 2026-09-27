@echo off
setlocal ENABLEEXTENSIONS
REM Fast Linux CPU-core test inside the plain Rocky Linux 10 image (no GPU
REM toolchains), from Windows via Docker Desktop. Mirrors the CI job: it
REM `cargo test`s only the crates that have NO Intel-oneAPI / CUDA-toolkit
REM dependency, so it runs without icx/nvcc.
REM
REM   scripts\build-linux-docker.bat
REM
REM This is NOT a full build: because the compute kernels are real-only,
REM the GPU-dependent crates (models, engine, server, cli, app) require
REM Intel oneAPI (icx/icpx) + the CUDA Toolkit (nvcc). For a full Linux
REM build or test — including the GUI artifact and all three backends —
REM use scripts\build-linux-gpu-docker.bat (the GPU image has both).
REM
REM Source is bind-mounted; the Linux target dir + crate registry live in
REM named volumes so they never mix with the Windows target/.

set "IMG=rustllama-build-linux"
for %%I in ("%~dp0..") do set "REPO=%%~fI"

echo ^>^> ensuring plain Rocky Linux 10 image (%IMG%)...
docker image inspect %IMG% >nul 2>&1 || docker build -t %IMG% -f "%~dp0Dockerfile.linux" "%~dp0."
if errorlevel 1 ( echo ERROR: docker image build failed & exit /b 1 )

echo ^>^> testing the toolchain-free CPU-core crates in the plain image...
docker run --rm ^
  -v "%REPO%:/work:ro" ^
  -v rustllama_rocky10_target:/target ^
  -v rustllama_cargo_reg:/usr/local/cargo/registry ^
  -e CARGO_TARGET_DIR=/target ^
  -w /work ^
  %IMG% cargo test ^
    -p rustllama-config -p rustllama-tensor -p rustllama-tokenizer ^
    -p rustllama-gguf -p rustllama-kernels-cpu -p rustllama-l0-sys ^
    -p rustllama-runtime -p rustllama-hub %*
exit /b %ERRORLEVEL%
