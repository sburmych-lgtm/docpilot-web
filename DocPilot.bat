@echo off
title DocPilot
cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" desktop.py
) else (
    echo [!] Спочатку запусти install.bat
    pause
)
