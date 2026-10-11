@echo off
title SmartPark AI - CCTV Occupancy Grid
color 0B
echo ============================================================
echo   SmartPark AI - College S5 AIML Capstone Prototype
echo   Automated Parking Slot Occupancy Detection System
echo ============================================================
echo.
echo Launching Streamlit application on local server...
echo.
python -m streamlit run app.py --server.port=8501 --server.headless=false
pause
