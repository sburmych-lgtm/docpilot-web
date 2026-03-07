@echo off
title DocPilot - Встановлення
cd /d "%~dp0"
echo ========================================
echo   DocPilot - Встановлення залежностей
echo ========================================
echo.

:: Check Python
where py >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [!] Python не знайдено!
    echo     Завантаж Python 3.12+ з https://www.python.org/downloads/
    echo     При встановленні постав галочку "Add Python to PATH"
    pause
    exit /b 1
)

echo [1/3] Створюємо віртуальне середовище...
py -3 -m venv .venv
if %ERRORLEVEL% neq 0 (
    echo [!] Помилка створення venv
    pause
    exit /b 1
)

echo [2/3] Встановлюємо PyTorch (CPU)...
".venv\Scripts\pip.exe" install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo [3/3] Встановлюємо решту залежностей...
".venv\Scripts\pip.exe" install -r requirements.txt

echo.
echo ========================================
echo   Готово! Запускай DocPilot.bat
echo ========================================
pause
