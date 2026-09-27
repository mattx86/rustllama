@echo off
setlocal ENABLEEXTENSIONS
REM Build / test rustllama for Linux with BOTH GPU backends (Intel SYCL +
REM NVIDIA CUDA, always compiled in) inside the dedicated Rocky Linux 10
REM GPU image (CUDA nvcc + Intel oneAPI icx), from Windows. Passes its
REM arguments straight to scripts/build.sh, e.g.:
REM
REM   scripts\build-linux-gpu-docker.bat            headless, SYCL + CUDA
REM   scripts\build-linux-gpu-docker.bat --gui      + Tauri desktop GUI
REM
REM NOTE: this COMPILES both GPU backends; running them needs the physical
REM GPU (an NVIDIA driver for CUDA, an Intel GPU + Level Zero for SYCL),
REM which a Docker container on a non-GPU host does not have. Source is
REM mounted read-write (tauri_build) and the GPU build artifacts use their
REM own /target volume, separate from the CPU build's.

set "BASE=rustllama-build-linux"
set "IMG=rustllama-build-gpu"
for %%I in ("%~dp0..") do set "REPO=%%~fI"

echo ^>^> ensuring base Rocky 10 image (%BASE%)...
docker image inspect %BASE% >nul 2>&1 || docker build -t %BASE% -f "%~dp0Dockerfile.linux" "%~dp0."
if errorlevel 1 ( echo ERROR: base image build failed & exit /b 1 )

echo ^>^> building GPU image (%IMG%: + CUDA nvcc + Intel oneAPI icx)...
docker build -t %IMG% -f "%~dp0Dockerfile.linux-gpu" "%~dp0."
if errorlevel 1 ( echo ERROR: GPU image build failed & exit /b 1 )

echo ^>^> running scripts/build.sh in the GPU image...
docker run --rm ^
  -v "%REPO%:/work" ^
  -v rustllama_gpu_target:/target ^
  -v rustllama_cargo_reg:/usr/local/cargo/registry ^
  -e CARGO_TARGET_DIR=/target ^
  -w /work ^
  %IMG% bash scripts/build.sh %*
exit /b %ERRORLEVEL%
