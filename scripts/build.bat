@echo off
setlocal ENABLEEXTENSIONS ENABLEDELAYEDEXPANSION
REM Standardized rustllama build/test for Windows — the counterpart to
REM scripts/build.sh on Linux. All THREE compute backends (Intel SYCL +
REM NVIDIA CUDA + CPU) are always compiled into the binary; there are no
REM per-backend features. Startup detection picks which to run (or all):
REM Intel via SYCL/Level-Zero, NVIDIA via CUDA, CPU always.
REM
REM   scripts\build.bat            GUI artifact: desktop GUI + CLI + server
REM   scripts\build.bat --headless server-only build (no GUI)
REM   scripts\build.bat test       run the test suites
REM   scripts\build.bat --debug    debug profile
REM   scripts\build.bat -- <args>  pass extra args to cargo
REM
REM GUI is the default artifact so a single binary serves the desktop GUI
REM OR the CLI/server. The GUI is native egui/eframe (crates/rustllama-gui)
REM — NOT Tauri/webkit — so there is NO JS frontend to prebuild (no pnpm, no
REM app/ui). On Windows it links only winit + the OS OpenGL stack.
REM
REM Because all backends are built in, this REQUIRES both GPU toolchains:
REM Intel oneAPI (icx) and the CUDA Toolkit (nvcc, `winget install
REM Nvidia.CUDA`). The oneAPI/MSVC environment is set up by
REM build-env.bat, which this delegates to (single source of truth
REM for the toolchain env); nvcc is found via CUDA_PATH (set by the CUDA
REM installer) and uses that same MSVC host compiler.

set "CMD=build"
set "PROFILE=--release"
set "FEATS=gui"
set "EXTRA="

:parse
if "%~1"=="" goto parsed
if /I "%~1"=="build"     ( set "CMD=build"   & shift & goto parse )
if /I "%~1"=="test"      ( set "CMD=test"    & shift & goto parse )
if /I "%~1"=="check"     ( set "CMD=check"   & shift & goto parse )
if /I "%~1"=="clean"     ( set "CMD=clean"   & shift & goto parse )
if /I "%~1"=="--debug"    ( set "PROFILE="          & shift & goto parse )
if /I "%~1"=="--release"  ( set "PROFILE=--release" & shift & goto parse )
if /I "%~1"=="--gui"      ( set "FEATS=gui"         & shift & goto parse )
if /I "%~1"=="--headless" ( set "FEATS="            & shift & goto parse )
if "%~1"=="--"           ( shift & goto collect )
set "EXTRA=%EXTRA% %~1"
shift
goto parse
:collect
if "%~1"=="" goto parsed
set "EXTRA=%EXTRA% %~1"
shift
goto collect
:parsed

set "ENV=%~dp0build-env.bat"

REM The GUI is native egui — no JS frontend to prebuild, so there is no pnpm
REM step here. The `gui` cargo feature pulls in crates/rustllama-gui directly.

REM Only pass --features when non-empty (--headless leaves FEATS empty,
REM giving a server-only binary from the app crate's default features).
set "FEATFLAG="
if not "%FEATS%"=="" set "FEATFLAG=--features %FEATS%"

if /I "%CMD%"=="build" (
    echo ^>^> rustllama build ^(Windows, features: %FEATS%^)
    call "%ENV%" build %PROFILE% %FEATFLAG% --manifest-path app\desktop\Cargo.toml -j 2 %EXTRA%
) else if /I "%CMD%"=="check" (
    echo ^>^> rustllama check ^(Windows, features: %FEATS%^)
    call "%ENV%" check %PROFILE% %FEATFLAG% --manifest-path app\desktop\Cargo.toml %EXTRA%
) else if /I "%CMD%"=="test" (
    echo ^>^> rustllama test ^(Windows^)
    call "%ENV%" test %PROFILE% -p rustllama-kernels-cpu -p rustllama-config -p rustllama-tensor -p rustllama-tokenizer -p rustllama-safetensors -p rustllama-models -p rustllama-engine -p rustllama-server %EXTRA%
) else if /I "%CMD%"=="clean" (
    call "%ENV%" clean %EXTRA%
)
set "RC=%ERRORLEVEL%"
endlocal & exit /b %RC%
