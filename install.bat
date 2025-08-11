@echo off
REM ADGM Corporate Agent Installation Script for Windows
REM This script sets up the ADGM Corporate Agent project

echo  ADGM Corporate Agent Installation Script
echo ==========================================

REM Check if Python is installed
echo  Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo  Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo  Python found

REM Create virtual environment
echo 🔧 Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo  Virtual environment created
) else (
    echo  Virtual environment already exists
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Check if .env file exists
if not exist ".env" (
    echo   Warning: .env file not found
    echo  Creating .env template...
    (
        echo # ADGM Corporate Agent Environment Variables
        echo # Get your API key from: https://makersuite.google.com/app/apikey
        echo GEMINI_API_KEY=your_gemini_api_key_here
        echo.
        echo # Optional: Customize these settings
        echo # RAG_INDEX_DIR=faiss_index
        echo # EMBEDDING_DIM=3072
        echo # RAG_TOP_K=6
    ) > .env
    echo  .env template created
    echo   Please edit .env file and add your Gemini API key
) else (
    echo  .env file found
)

REM Check if FAISS index exists
if not exist "faiss_index\index.faiss" (
    echo 📚 Setting up knowledge base...
    echo   Note: This requires a valid Gemini API key in .env file
    echo 📚 Ingesting ADGM reference documents...
    python ingest_adgm_sources.py
    echo  Knowledge base setup complete
) else (
    echo  Knowledge base already exists
)

echo.
echo 🎉 Installation completed successfully!
echo.
echo  Next steps:
echo 1. Edit .env file and add your Gemini API key
echo 2. Run: python app.py
echo 3. Open browser to: http://localhost:7860
echo.
echo 📚 For more information, see README.md
echo.
echo  Happy document analysis!
pause
