@echo off
title Predictive Maintenance - Backend + Frontend
echo ════════════════════════════════════════════════════
echo   Starting Backend + Frontend Servers
echo ════════════════════════════════════════════════════
echo.
echo Starting Backend (FastAPI)...
start "Backend API" cmd /k "cd /d D:\PredictiveMaintenance && conda activate predmaint && python -m uvicorn start_api:app --reload --port 8000"
timeout /t 3 /nobreak > nul
echo.
echo Starting Frontend (React)...
start "Frontend" cmd /k "cd /d D:\PredictiveMaintenance\frontend && npm run dev"
timeout /t 5 /nobreak > nul
echo.
echo ✅ Both servers started!
echo.
echo Backend:  http://localhost:8000/docs
echo Frontend: http://localhost:8080
echo.
start http://localhost:8080