@echo off
setlocal

echo ============================================================
echo  GPULOAD V3 — Windows Build
echo  VAE decode backends: default, cuda, directml scaffold
echo ============================================================


set CL=/Zc:preprocessor
set "NVCC_CCBIN=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64"

:: Build release — CPU-only (default feature, works everywhere)
echo.
echo Building generate (CPU release)...
cargo build --bin generate --release
if errorlevel 1 (
    echo ERROR: Build failed.
    exit /b 1
)

echo.
echo Build OK!  Binary: target\release\generate.exe
echo.
echo Example usage:
echo   target\release\generate.exe --help
echo.
echo To build with CUDA support, add the --features cuda flag:
echo   cargo build --bin generate --release --features cuda
echo.