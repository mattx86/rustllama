@echo off
setlocal ENABLEEXTENSIONS
REM Build rustllama for ARM64 / ARMv9 Linux with the NVIDIA CUDA backend
REM (targeting an NVIDIA Grace / DGX Spark, GB10). Ubuntu 24.04 aarch64 +
REM CUDA-for-ARM; NO Intel oneAPI (SYCL is x86_64-only → kernels-sycl
REM builds a no-op stub, so the binary runs on CPU + native CUDA).
REM
REM   scripts\build-linux-arm64-gpu-docker.bat            headless (server)
REM   scripts\build-linux-arm64-gpu-docker.bat --gui      + Tauri GUI (arm64 webkit)
REM
REM On an x86 host this uses buildx + QEMU to emulate aarch64 (SLOW — the
REM whole workspace compiles under emulation). On a native ARM host it runs
REM natively and fast. COMPILING here validates the ARM build; EXECUTION
REM needs the physical NVIDIA GPU + driver (the DGX Spark itself).

set "IMG=rustllama-build-arm64-gpu"
for %%I in ("%~dp0..") do set "REPO=%%~fI"

REM Default to headless unless the caller passes --gui (the DGX Spark is a
REM server/dev target; headless is the common case and needs no arm64 GTK).
set "ARGS=%*"
if "%ARGS%"=="" set "ARGS=--headless"

echo ^>^> ensuring QEMU binfmt for cross-arch emulation (no-op on native ARM)...
docker run --rm --privileged tonistiigi/binfmt --install arm64 >nul 2>&1

echo ^>^> building ARM64 GPU image (%IMG%: Ubuntu 24.04 aarch64 + CUDA-for-ARM)...
docker buildx build --platform linux/arm64 -t %IMG% -f "%~dp0Dockerfile.linux-arm64-gpu" --load "%~dp0."
if errorlevel 1 ( echo ERROR: ARM64 GPU image build failed & exit /b 1 )

echo ^>^> running scripts/build.sh (aarch64) in the ARM64 image...
docker run --rm --platform linux/arm64 ^
  -v "%REPO%:/work" ^
  -v rustllama_arm64_target:/target ^
  -v rustllama_arm64_cargo_reg:/usr/local/cargo/registry ^
  -e CARGO_TARGET_DIR=/target ^
  -w /work ^
  %IMG% bash scripts/build.sh %ARGS%
exit /b %ERRORLEVEL%
