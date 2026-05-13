@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

REM ============================================================
REM Manga Translator launcher
REM Optional OCR: PaddleOCR-VL via llama.cpp
REM ============================================================

REM ---- PaddleOCR-VL paths ----
REM Put GGUF files here, or edit these paths.
set "PADDLEOCR_VL_MODEL_PATH=%~dp0model\paddleocr_vl\model.gguf"
set "PADDLEOCR_VL_MMPROJ_PATH=%~dp0model\paddleocr_vl\mmproj.gguf"

REM llama.cpp folder. Edit this to Koharu's bundled llama.cpp folder if needed.
set "LLAMA_CPP_DIR=%~dp0tools\llama.cpp"

REM Server settings.
set "PADDLEOCR_VL_HOST=127.0.0.1"
set "PADDLEOCR_VL_PORT=8088"
set "PADDLEOCR_VL_SERVER_URL=http://%PADDLEOCR_VL_HOST%:%PADDLEOCR_VL_PORT%"
set "PADDLEOCR_VL_MAX_WORKERS=1"

REM GPU layers for llama.cpp. Use 99 for GPU offload, 0 for CPU-only.
set "PADDLEOCR_VL_NGL=99"

REM App verbose logs.
set "VERBOSE_LOG=0"

REM ============================================================
REM Virtual environment
REM ============================================================

if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo Please create it first:
    echo.
    echo py -3.11 -m venv .venv
    echo .venv\Scripts\activate
    echo pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo Using Python:
where python
python --version

REM ============================================================
REM Optional: PaddleOCR-VL llama.cpp server
REM ============================================================

echo.
echo Checking PaddleOCR-VL configuration...

set "PADDLEOCR_VL_AVAILABLE=1"

if not exist "%PADDLEOCR_VL_MODEL_PATH%" (
    echo [WARN] PaddleOCR-VL model not found:
    echo        %PADDLEOCR_VL_MODEL_PATH%
    set "PADDLEOCR_VL_AVAILABLE=0"
)

if not exist "%PADDLEOCR_VL_MMPROJ_PATH%" (
    echo [WARN] PaddleOCR-VL mmproj not found:
    echo        %PADDLEOCR_VL_MMPROJ_PATH%
    set "PADDLEOCR_VL_AVAILABLE=0"
)

set "LLAMA_SERVER_EXE="

if exist "%LLAMA_CPP_DIR%\llama-server.exe" (
    set "LLAMA_SERVER_EXE=%LLAMA_CPP_DIR%\llama-server.exe"
) else if exist "%LLAMA_CPP_DIR%\llama-server" (
    set "LLAMA_SERVER_EXE=%LLAMA_CPP_DIR%\llama-server"
) else (
    for /f "delims=" %%I in ('where llama-server.exe 2^>nul') do (
        if not defined LLAMA_SERVER_EXE set "LLAMA_SERVER_EXE=%%I"
    )
    for /f "delims=" %%I in ('where llama-server 2^>nul') do (
        if not defined LLAMA_SERVER_EXE set "LLAMA_SERVER_EXE=%%I"
    )
)

if not defined LLAMA_SERVER_EXE (
    echo [WARN] llama-server not found.
    echo        Set LLAMA_CPP_DIR or add llama-server to PATH.
    set "PADDLEOCR_VL_AVAILABLE=0"
)

if "%PADDLEOCR_VL_AVAILABLE%"=="1" (
    echo PaddleOCR-VL model:
    echo   %PADDLEOCR_VL_MODEL_PATH%
    echo PaddleOCR-VL mmproj:
    echo   %PADDLEOCR_VL_MMPROJ_PATH%
    echo llama-server:
    echo   %LLAMA_SERVER_EXE%
    echo Server URL:
    echo   %PADDLEOCR_VL_SERVER_URL%
    echo.

    call :CHECK_PADDLEOCR_SERVER
    if errorlevel 1 (
        echo Starting PaddleOCR-VL llama.cpp server...

        start "PaddleOCR-VL llama-server" "%LLAMA_SERVER_EXE%" ^
            -m "%PADDLEOCR_VL_MODEL_PATH%" ^
            --mmproj "%PADDLEOCR_VL_MMPROJ_PATH%" ^
            --host "%PADDLEOCR_VL_HOST%" ^
            --port "%PADDLEOCR_VL_PORT%" ^
            -c 4096 ^
            -ngl %PADDLEOCR_VL_NGL%

        echo Waiting for PaddleOCR-VL server to become ready...

        set /a WAIT_COUNT=0

        :WAIT_PADDLEOCR_VL
        timeout /t 2 /nobreak >nul
        set /a WAIT_COUNT+=2

        call :CHECK_PADDLEOCR_SERVER
        if errorlevel 1 (
            if !WAIT_COUNT! GEQ 60 (
                echo [WARN] PaddleOCR-VL server was not ready after 60 seconds.
                echo        The app will still start. Check the llama-server window for errors.
            ) else (
                goto WAIT_PADDLEOCR_VL
            )
        ) else (
            echo PaddleOCR-VL server is ready.
        )
    ) else (
        echo PaddleOCR-VL server is already running.
    )
) else (
    echo.
    echo [INFO] PaddleOCR-VL will not be pre-started.
    echo        You can still use Chrome Lens as the optional fallback OCR provider.
    echo        To enable PaddleOCR-VL:
    echo        - Put model.gguf and mmproj.gguf in model\paddleocr_vl\
    echo        - Put llama-server.exe in tools\llama.cpp\ or edit LLAMA_CPP_DIR
)

REM ============================================================
REM Start app
REM ============================================================

echo.
echo Starting Manga Translator...
echo Open this in your browser if it does not open automatically:
echo http://127.0.0.1:5000
echo.

start http://127.0.0.1:5000
python app.py

echo.
pause
exit /b 0

REM ============================================================
REM Helpers
REM ============================================================

:CHECK_PADDLEOCR_SERVER
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "try { $r = Invoke-WebRequest -UseBasicParsing -Uri '%PADDLEOCR_VL_SERVER_URL%/health' -TimeoutSec 2; if ($r.StatusCode -lt 500) { exit 0 } else { exit 1 } } catch { try { $r = Invoke-WebRequest -UseBasicParsing -Uri '%PADDLEOCR_VL_SERVER_URL%/v1/models' -TimeoutSec 2; if ($r.StatusCode -lt 500) { exit 0 } else { exit 1 } } catch { exit 1 } }"
exit /b %ERRORLEVEL%
