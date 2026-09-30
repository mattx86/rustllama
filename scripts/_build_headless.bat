@echo off
call "%~dp0build-env.bat" build --release --manifest-path app\desktop\Cargo.toml -j 2
exit /b %ERRORLEVEL%
