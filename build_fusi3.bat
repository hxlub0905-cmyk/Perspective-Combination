@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

echo =============================================
echo   Fusi³ Auto Build Installer
 echo =============================================

cd /d "%~dp0"

if not exist "main.py" (
  echo [ERROR] Cannot find main.py in current folder.
  pause
  exit /b 1
)

set "PYTHON=python"
where %PYTHON% >nul 2>nul
if errorlevel 1 (
  set "PYTHON=py"
  where %PYTHON% >nul 2>nul
  if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
  )
)

echo [1/5] Creating virtual environment...
if not exist ".venv\Scripts\python.exe" (
  %PYTHON% -m venv .venv
  if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] Failed to activate virtual environment.
  pause
  exit /b 1
)

echo [2/5] Upgrading pip/setuptools/wheel...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
  echo [ERROR] Failed to upgrade pip tools.
  pause
  exit /b 1
)

echo [3/5] Installing requirements (includes python-pptx for PPTX export)...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo [ERROR] Failed to install requirements.
  pause
  exit /b 1
)

echo [4/5] Installing PyInstaller...
python -m pip install pyinstaller
if errorlevel 1 (
  echo [ERROR] Failed to install PyInstaller.
  pause
  exit /b 1
)

set "ICON_ARGS="
if exist "assets\fusi3_icon.ico" (
  set "ICON_ARGS=--icon assets\fusi3_icon.ico"
) else (
  if exist "assets\fusi3_icon.png" (
    set "ICON_ARGS=--add-data assets\fusi3_icon.png;assets"
  )
)

echo [5/5] Building EXE...
pyinstaller --noconfirm --clean --name "Fusi3" --windowed --onefile main.py --collect-all pptx --collect-all matplotlib --add-data "perscomb;perscomb" --add-data "assets;assets" %ICON_ARGS%
if errorlevel 1 (
  echo [ERROR] Build failed.
  pause
  exit /b 1
)

echo.
echo Build completed successfully.
echo EXE path: dist\Fusi3.exe
pause
exit /b 0
