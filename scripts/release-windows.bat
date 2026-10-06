@echo off
setlocal ENABLEEXTENSIONS
REM ===========================================================================
REM Build the Windows release artifact:
REM   release\rustllama-<ver>-windows-x86_64.zip
REM Self-contained: the desktop GUI + CLI + server (all backends compiled in)
REM plus the bundled Intel oneAPI SYCL runtime DLLs, so it runs without an
REM oneAPI install. CUDA is statically linked. Output lands in release\.
REM ===========================================================================
pushd "%~dp0.."

REM Compile the CUDA tensor-core kernels for the x86 NVIDIA targets, so the
REM on-device self-check (tune --validate-kernels) can validate + auto-enable
REM them on a Windows Hopper (H100) / Blackwell (RTX 50xx) host. Accelerated
REM arches (sm_90a / sm_120a) are REQUIRED for the FP8-wgmma / FP4 TC paths to
REM compile (build.rs gates RSL_HOPPER_TC / RSL_BLACKWELL_TC on them); sm_90a is
REM a superset of sm_90 so the base Hopper path still runs. Release build only —
REM dev `_build.bat` keeps the default arches for faster iteration.
set "RUSTLLAMA_CUDA_ARCHS=75;80;86;89;90a;100a;120a"

echo ^>^> [1/3] building the GUI artifact (frontend + all-backends binary)...
call "%~dp0_build.bat"
if errorlevel 1 ( echo ERROR: GUI build failed & popd & exit /b 1 )

echo ^>^> [2/3] staging Intel oneAPI redistributable SYCL DLLs...
call "%~dp0build-env.bat" xtask stage-redist --out target\redist-staging
if errorlevel 1 ( echo ERROR: stage-redist failed & popd & exit /b 1 )

echo ^>^> [3/3] assembling + zipping the release archive...
REM The packaging script reads the version from Cargo.toml itself (avoids
REM fragile quote-escaping when passing it through cmd).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0_release_package.ps1" ^
  -Os windows -Arch x86_64 ^
  -Binary target\release\rustllama.exe -LibDir target\redist-staging -Ext zip
set "RC=%ERRORLEVEL%"
popd
exit /b %RC%
