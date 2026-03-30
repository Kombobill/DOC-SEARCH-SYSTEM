#!/bin/bash

# AI Document Search System - Quick Start Script

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  AI Document Search System - Quick Start                ║"
echo "║  https://github.com/yourusername/ai-document-search     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo -e "${BLUE}Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi
echo -e "${GREEN}✓ Python $(python3 --version | awk '{print $2}') found${NC}"
echo ""

# Check Node
echo -e "${BLUE}Checking Node.js installation...${NC}"
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 14+"
    exit 1
fi
echo -e "${GREEN}✓ Node.js $(node --version) found${NC}"
echo ""

# Setup Python backend
echo -e "${BLUE}Setting up Python backend...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

source venv/bin/activate

echo "Installing Python dependencies..."
pip install -q -r requirements.txt
echo -e "${GREEN}✓ Python dependencies installed${NC}"
echo ""

# Setup React frontend
echo -e "${BLUE}Setting up React frontend...${NC}"
if [ ! -d "frontend" ]; then
    echo "Creating React app..."
    npx create-react-app frontend --template minimal --silent
fi

if [ ! -f "frontend/src/App.jsx" ]; then
    cp App.jsx frontend/src/
    cp App.css frontend/src/
fi

cd frontend
npm install -q
echo -e "${GREEN}✓ React dependencies installed${NC}"
cd ..
echo ""

# Start services
echo -e "${YELLOW}Starting services...${NC}"
echo ""
echo -e "${GREEN}Backend starting on http://localhost:5000${NC}"
echo -e "${GREEN}Frontend will start on http://localhost:3000${NC}"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start backend in background
python ai_search_backend.py &
BACKEND_PID=$!

sleep 3

# Start frontend
cd frontend
npm start
