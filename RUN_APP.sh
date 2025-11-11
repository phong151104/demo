#!/bin/bash

echo "========================================"
echo "  Credit Scoring System - Startup"
echo "========================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

echo "[INFO] Python found: $(python3 --version)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
    echo "[SUCCESS] Virtual environment created"
    echo ""
fi

# Activate virtual environment
echo "[INFO] Activating virtual environment..."
source venv/bin/activate

# Install/Update requirements
echo "[INFO] Installing/Updating dependencies..."
pip install -r requirements.txt --quiet
echo "[SUCCESS] Dependencies installed"
echo ""

# Run Streamlit app
echo "[INFO] Starting Credit Scoring System..."
echo "[INFO] The app will open in your default browser"
echo "[INFO] Press Ctrl+C to stop the server"
echo ""
streamlit run app.py

