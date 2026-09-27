@echo off
setlocal ENABLEEXTENSIONS
REM ===========================================================================
REM Build the Linux x86_64 headless release artifact inside the Rocky Linux 10
REM GPU image (Intel oneAPI icx + CUDA nvcc), then package it self-contained:
REM   release\rustllama-<ver>-linux-x86_64.tar.gz
REM (CLI + server, all backends compiled in, bundled oneAPI SYCL runtime .so's)
REM ===========================================================================
set "BASE=rustllama-build-linux"
set "IMG=rustllama-build-gpu"
for %%I in ("%~dp0..") do set "REPO=%%~fI"

echo ^>^> ensuring base Rocky 10 image (%BASE%)...
docker image inspect %BASE% >nul 2>&1 || docker build -t %BASE% -f "%~dp0Dockerfile.linux" "%~dp0."
if errorlevel 1 ( echo ERROR: base image build failed & exit /b 1 )

echo ^>^> building GPU image (%IMG%: + CUDA nvcc + Intel oneAPI icx)...
docker build -t %IMG% -f "%~dp0Dockerfile.linux-gpu" "%~dp0."
if errorlevel 1 ( echo ERROR: GPU image build failed & exit /b 1 )

echo ^>^> building headless + packaging release archive (in-container)...
docker run --rm ^
  -v "%REPO%:/work" ^
  -v rustllama_gpu_target:/target ^
  -v rustllama_cargo_reg:/usr/local/cargo/registry ^
  -e CARGO_TARGET_DIR=/target ^
  -w /work ^
  %IMG% bash scripts/_release_build_linux.sh x86_64
exit /b %ERRORLEVEL%
