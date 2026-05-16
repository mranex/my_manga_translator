@echo off
title Manga Translator Gemini

cd /d "C:\New folder\Manga-translator_Gemini\"

if not exist ".venv\Scripts\activate.bat" (
    echo Khong tim thay file .venv\Scripts\activate.bat
    echo Hay kiem tra lai duong dan ung dung.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m mmt_gui.main

if errorlevel 1 (
    echo.
    echo Ung dung bi loi khi khoi chay.
    pause
)