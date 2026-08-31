#!/bin/bash
# Unix/macOS script to run the AIGC Detector UI
# This script assumes you have already run: uv sync

echo "Starting AIGC Detector UI..."
echo ""
echo "Opening browser to http://localhost:8501"
echo "Press Ctrl+C to stop the server"
echo ""

uv run streamlit run app.py
