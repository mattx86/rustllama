@echo off
setlocal ENABLEEXTENSIONS
REM ===========================================================================
REM Build the Linux x86_64 GUI (native egui) release artifact inside the Rocky
REM Linux 10 GPU image (Intel oneAPI icx + CUDA nvcc), then package it self-
REM contained: release\rustllama-<ver>-linux-x86_64.tar.gz. The single binary is
REM a superset (GUI + CLI + server, all backends compiled in) + the bundled
REM oneAPI SYCL runtime .so's; the egui GL/X11 libs are host-provided.
REM ===========================================================================
set "BASE=rustllama-build-linux"
set "IMG=rustllama-build-gpu"
for %%I in ("%~dp0..") do set "REPO=%%~fI"

echo ^>^> building base Rocky 10 image (%BASE%; egui/GL deps + patchelf)...
REM Always build so a changed Dockerfile (e.g. the egui dep swap) actually takes
REM effect; Docker layer caching keeps this fast when unchanged. (Skip-if-exists
REM previously left a stale base that predated the egui dep swap.)
docker build -t %BASE% -f "%~dp0Dockerfile.linux" "%~dp0."
if errorlevel 1 ( echo ERROR: base image build failed & exit /b 1 )

echo ^>^> building GPU image (%IMG%: + CUDA nvcc + Intel oneAPI icx)...
docker build -t %IMG% -f "%~dp0Dockerfile.linux-gpu" "%~dp0."
if errorlevel 1 ( echo ERROR: GPU image build failed & exit /b 1 )

echo ^>^> building GUI + packaging release archive (in-container)...
docker run --rm ^
  -v "%REPO%:/work" ^
  -v rustllama_gpu_target:/target ^
  -v rustllama_cargo_reg:/usr/local/cargo/registry ^
  -e CARGO_TARGET_DIR=/target ^
  -w /work ^
  %IMG% bash scripts/_release_build_linux.sh x86_64
exit /b %ERRORLEVEL%
