@echo off
setlocal ENABLEEXTENSIONS
REM ===========================================================================
REM Build the Linux aarch64 (ARMv9) headless release artifact for NVIDIA Grace
REM / DGX Spark, inside the Ubuntu 24.04 aarch64 + CUDA-for-ARM image, then
REM package it self-contained:
REM   release\rustllama-<ver>-linux-aarch64.tar.gz
REM (CLI + server; CUDA + CPU/NEON. SYCL is x86-only → no-op stub, no bundled
REM  oneAPI libs.)
REM
REM On an x86 host this cross-builds via buildx + QEMU (SLOW — the whole
REM workspace compiles under emulation, can take a long time). On a native ARM
REM host it runs natively.
REM ===========================================================================
set "IMG=rustllama-build-arm64-gpu"
for %%I in ("%~dp0..") do set "REPO=%%~fI"

echo ^>^> ensuring QEMU binfmt for cross-arch emulation (no-op on native ARM)...
docker run --rm --privileged tonistiigi/binfmt --install arm64 >nul 2>&1

echo ^>^> building ARM64 GPU image (%IMG%: Ubuntu 24.04 aarch64 + CUDA-for-ARM)...
docker buildx build --platform linux/arm64 -t %IMG% -f "%~dp0Dockerfile.linux-arm64-gpu" --load "%~dp0."
if errorlevel 1 ( echo ERROR: ARM64 GPU image build failed & exit /b 1 )

echo ^>^> building headless + packaging release archive (in-container, aarch64)...
docker run --rm --platform linux/arm64 ^
  -v "%REPO%:/work" ^
  -v rustllama_arm64_target:/target ^
  -v rustllama_arm64_cargo_reg:/usr/local/cargo/registry ^
  -e CARGO_TARGET_DIR=/target ^
  -w /work ^
  %IMG% bash scripts/_release_build_linux.sh aarch64
exit /b %ERRORLEVEL%
