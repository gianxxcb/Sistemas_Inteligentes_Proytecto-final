@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    where py >nul 2>&1
    if errorlevel 1 (
        python -m venv .venv
    ) else (
        py -3.12 -m venv .venv
    )
    if errorlevel 1 goto failed
)
if not exist ".venv\quantlab-ready" (
    echo Instalando las dependencias por primera vez...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto failed
    echo ready>".venv\quantlab-ready"
)
".venv\Scripts\python.exe" app.py
exit /b %errorlevel%
:failed
echo No se pudo preparar QuantLab. Comprueba que tienes Python 3.12 y conexion a internet.
pause
exit /b 1
