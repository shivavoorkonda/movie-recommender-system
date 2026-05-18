@echo off
REM ─────────────────────────────────────────────────────────────────
REM  Movie Recommender System - Launcher
REM  Forces UTF-8 output encoding for Windows compatibility
REM ─────────────────────────────────────────────────────────────────
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

cd /d "%~dp0\src"

IF "%1"=="" (
    echo Running full pipeline...
    python main.py %*
) ELSE (
    python main.py %*
)
