@echo off
REM ============================================================================
REM One-Click Label Studio Launcher for Local Historical IE Annotation & Review
REM ============================================================================

setlocal enabledelayedexpansion
title Label Studio Local Review Server

echo ============================================================================
echo   Bangla Textbook OCR & Historical IE - Label Studio Local Server
echo ============================================================================
echo.

REM Set project root directory
cd /d "%~dp0.."
set "PROJECT_ROOT=%CD%"

REM Set Label Studio local file serving variables so scanned images load properly
set "LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true"
set "LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=%PROJECT_ROOT%"
set "LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK=true"

echo [*] Project Root: %PROJECT_ROOT%
echo [*] Local Scans Serving Enabled: %LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED%
echo [*] Configured XML Interface: configs\label_studio_interface.xml
echo [*] Pre-packaged Tasks: data\gold_standard\label_studio_tasks.json
echo.

REM Check if Python environment has label-studio
where label-studio >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo [+] Found label-studio in system PATH.
    set "LS_CMD=label-studio"
) else (
    if exist "%PROJECT_ROOT%\.venv\Scripts\label-studio.exe" (
        echo [+] Found label-studio in .venv virtual environment.
        set "LS_CMD=%PROJECT_ROOT%\.venv\Scripts\label-studio.exe"
    ) else (
        echo [!] label-studio is not yet installed in .venv.
        echo [*] Installing label-studio into virtual environment...
        "%PROJECT_ROOT%\.venv\Scripts\python.exe" -m pip install label-studio
        set "LS_CMD=%PROJECT_ROOT%\.venv\Scripts\label-studio.exe"
    )
)

echo.
echo ============================================================================
echo   Starting Label Studio on http://localhost:8080 ...
echo   Instructions:
echo     1. Open your browser at http://localhost:8080
echo     2. Create Project: "Bangla Historical Textbook IE (Dev Set)"
echo     3. In 'Labeling Setup', paste: configs\label_studio_interface.xml
echo     4. In 'Import', drag and drop: data\gold_standard\label_studio_tasks.json
echo     5. Press Ctrl+C in this terminal window to stop the server.
echo ============================================================================
echo.

"%LS_CMD%" start --port 8080

pause
