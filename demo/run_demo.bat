@echo off
echo ===================================================
echo   Starting Burger Haven AI Demo (RAG + SCE Thesis)
echo ===================================================
echo.
echo Activating virtual environment...
call ..\venv\Scripts\activate.bat

echo Starting Streamlit app...
streamlit run app.py
