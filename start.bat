@echo off
echo ========================================
echo   ReaDox PDF AI - Local Dev Launcher
echo ========================================
echo.

REM Check if backend directory exists
if not exist "backend" (
    echo ERROR: Backend directory not found!
    echo Please make sure you're in the correct directory.
    pause
    exit /b 1
)

echo [1/3] Starting Python Backend...
echo.
start "Backend Server" cmd /k "cd backend && python server.py"

echo [2/3] Waiting for backend to initialize...
timeout /t 5 /nobreak > nul

echo [3/3] Starting Frontend...
echo.
start "Frontend Server" cmd /k "npm run dev"

echo.
echo ========================================
echo   Application Started!
echo ========================================
echo.
echo Backend API: http://localhost:8000
echo API Docs:    http://localhost:8000/docs
echo Frontend:    http://localhost:5173
echo.
echo Press any key to stop all servers...
pause > nul

echo.
echo Stopping servers...
taskkill /FI "WINDOWTITLE eq Backend Server*" /T /F 2>nul
taskkill /FI "WINDOWTITLE eq Frontend Server*" /T /F 2>nul

echo.
echo All servers stopped.
pause
