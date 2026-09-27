@echo off
REM Build-environment wrapper: sets up the MSVC + Intel oneAPI (icx) +
REM CUDA (nvcc) toolchains, then runs `cargo` with whatever args follow.
REM The single source of truth for the Windows build environment; every
REM other build script delegates its env setup here. Pass cargo args
REM after the script name, e.g.:
REM
REM   scripts\build-env.bat build --release --features gui ^
REM       --manifest-path app\src-tauri\Cargo.toml
REM   (SYCL + CUDA + CPU are always compiled in — no backend features;
REM    `--features gui` is the default artifact, added by the caller.)
REM
REM We bypass oneAPI's setvars.bat (which has parsing issues on
REM 2026.0 under nested cmd /c invocations) and set the env vars
REM icx + cc-rs need directly. Also sources MSVC's vcvarsall.bat so
REM the C++ STL headers + libs are reachable.
setlocal ENABLEEXTENSIONS
set "ONEAPI_ROOT=C:\Program Files (x86)\Intel\oneAPI"

REM Prefer the `latest` junction; oneAPI's installer maintains it.
set "COMPILER_DIR=%ONEAPI_ROOT%\compiler\latest"
if not exist "%COMPILER_DIR%\bin\icx.exe" (
    echo ERROR: icx.exe not found at "%COMPILER_DIR%\bin\". Reinstall oneAPI.
    exit /b 1
)

set "VSWHERE=C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo ERROR: vswhere.exe not found at "%VSWHERE%". Install Visual Studio with C++ workload.
    exit /b 1
)
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -property installationPath`) do set "VSINSTALL=%%i"
if "%VSINSTALL%"=="" (
    echo ERROR: vswhere could not locate a Visual Studio installation.
    exit /b 1
)
call "%VSINSTALL%\VC\Auxiliary\Build\vcvarsall.bat" x64 >nul 2>&1
if errorlevel 1 (
    echo ERROR: vcvarsall x64 failed
    exit /b 1
)

REM oneAPI dirs prepended so its tools win over any same-named tools
REM brought in by vcvarsall.
set "PATH=%COMPILER_DIR%\bin;%PATH%"
set "LIB=%COMPILER_DIR%\lib;%LIB%"
set "INCLUDE=%COMPILER_DIR%\include;%INCLUDE%"
set "CMPLR_ROOT=%COMPILER_DIR%"

REM CUDA toolchain: nvcc is required because rustllama-kernels-cuda is
REM always compiled in (real-only, no cuda feature). Prefer an ambient
REM CUDA_PATH; otherwise auto-detect the installed toolkit so the build
REM works even from a shell that predates the CUDA install. The nvcc bin
REM dir goes on PATH so build.rs (find_nvcc) resolves it.
if not defined CUDA_PATH (
    for /d %%v in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*") do set "CUDA_PATH=%%v"
)
if defined CUDA_PATH if exist "%CUDA_PATH%\bin\nvcc.exe" set "PATH=%CUDA_PATH%\bin;%PATH%"

REM SYCL C++ TU is built -MD (dynamic CRT) by cc-rs static_crt(false),
REM so force Rust to link the dynamic CRT too. Default workspace
REM rustflags in .cargo/config.toml use +crt-static which would clash
REM and produce LNK4098 + unresolved _W_Getdays/_W_Getmonths.
set "RUSTFLAGS=-C target-feature=-crt-static"

cargo %*
endlocal
