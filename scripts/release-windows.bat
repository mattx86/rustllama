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
