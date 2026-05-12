@echo off
cd /d "%~dp0"

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

echo.
echo Starting Manga Translator...
echo Open this in your browser if it does not open automatically:
echo http://127.0.0.1:5000
echo.

start http://127.0.0.1:5000
python app.py

echo.
pause