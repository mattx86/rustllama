@echo off
REM Wrapper: build the frontend (pnpm) then the release GUI artifact
REM (the default build). Invoked as a single short command to avoid
REM harness cmd-string mangling.
pushd "%~dp0..\app\ui"
call pnpm install --frozen-lockfile
if errorlevel 1 ( echo ERROR: pnpm install failed & popd & exit /b 1 )
call pnpm build
if errorlevel 1 ( echo ERROR: pnpm build failed & popd & exit /b 1 )
popd
call "%~dp0build-env.bat" build --release --features gui --manifest-path app\src-tauri\Cargo.toml -j 2
exit /b %ERRORLEVEL%
