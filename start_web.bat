@echo off
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"
echo [CineAI] Starting Flask server on http://localhost:5000
python web\app.py
