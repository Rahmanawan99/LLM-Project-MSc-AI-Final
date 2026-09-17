@echo off
echo ===================================================
echo   Starting Advanced Thesis Evaluation Dashboard
echo ===================================================
echo.
echo Activating virtual environment...
call ..\venv\Scripts\activate.bat

echo Starting Streamlit app...
streamlit run demo_advanced.py
