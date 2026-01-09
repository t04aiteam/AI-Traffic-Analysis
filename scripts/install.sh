#!/bin/bash
# Installation Script for Traffic AI Service

set -e

echo "=================================================="
echo "Traffic AI Service - Installation"
echo "=================================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Python is not installed or not in PATH"
    exit 1
fi

PYTHON_CMD=$(command -v python3 || command -v python)
echo "✓ Using Python: $PYTHON_CMD"

echo ""
echo "=================================================="
echo "Installing Dependencies"
echo "=================================================="
echo ""

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing from requirements.txt..."
    $PYTHON_CMD -m pip install -r requirements.txt
    echo ""
    echo "✓ Dependencies installed successfully"
else
    echo "❌ requirements.txt not found"
    exit 1
fi

echo ""
echo "=================================================="
echo "Installation Complete"
echo "=================================================="
echo ""
echo "To start the service, run:"
echo "  ./quickstart.sh"
echo ""
echo "Or manually:"
echo "  python main.py"
echo ""