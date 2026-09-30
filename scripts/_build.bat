@echo off
REM Wrapper: build the release GUI artifact (the default build). The GUI is
REM now a native egui/eframe desktop app (crates/rustllama-gui) — there is no
REM pnpm/JS frontend to build. Invoked as a single short command to avoid
REM harness cmd-string mangling.
call "%~dp0build-env.bat" build --release --features gui --manifest-path app\desktop\Cargo.toml -j 2
exit /b %ERRORLEVEL%
