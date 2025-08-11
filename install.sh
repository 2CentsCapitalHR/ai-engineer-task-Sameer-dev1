#!/bin/bash

# ADGM Corporate Agent Installation Script
# This script sets up the ADGM Corporate Agent project

set -e  # Exit on any error

echo " ADGM Corporate Agent Installation Script"
echo "=========================================="

# Check if Python 3.8+ is installed
echo " Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo " Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi

echo " Python version check passed: $python_version"

# Create virtual environment
echo "🔧 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo " Virtual environment created"
else
    echo " Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "  Warning: .env file not found"
    echo " Creating .env template..."
    cat > .env << EOF
# ADGM Corporate Agent Environment Variables
# Get your API key from: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Customize these settings
# RAG_INDEX_DIR=faiss_index
# EMBEDDING_DIM=3072
# RAG_TOP_K=6
EOF
    echo " .env template created"
    echo "  Please edit .env file and add your Gemini API key"
else
    echo " .env file found"
fi

# Check if FAISS index exists
if [ ! -d "faiss_index" ] || [ ! -f "faiss_index/index.faiss" ]; then
    echo "📚 Setting up knowledge base..."
    echo "  Note: This requires a valid Gemini API key in .env file"
    echo "📚 Ingesting ADGM reference documents..."
    python ingest_adgm_sources.py
    echo " Knowledge base setup complete"
else
    echo " Knowledge base already exists"
fi

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo " Next steps:"
echo "1. Edit .env file and add your Gemini API key"
echo "2. Run: python app.py"
echo "3. Open browser to: http://localhost:7860"
echo ""
echo "📚 For more information, see README.md"
echo ""
echo " Happy document analysis!"
