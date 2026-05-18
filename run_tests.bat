@echo off
REM ─────────────────────────────────────────────────────────────────
REM  Movie Recommender System - Run Tests
REM ─────────────────────────────────────────────────────────────────
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

cd /d "%~dp0"
python -m pytest tests/ -v %*
