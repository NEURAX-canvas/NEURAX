#!/usr/bin/env bash
# ─── NEURAX Development Startup Script ────────────────────────────────────────
# Launches all three services (Rust backend, Python agent, React frontend)
# in the background for local development.
#
# Usage: ./start-dev.sh
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🚀 Starting NEURAX development services...${NC}"

# ─── Check prerequisites ──────────────────────────────────────────────────────

command -v cargo >/dev/null 2>&1 || { echo -e "${RED}❌ Rust/Cargo not found. Install from https://rustup.rs${NC}"; exit 1; }
command -v npm >/dev/null 2>&1 || { echo -e "${RED}❌ npm not found. Install Node.js ≥ 20${NC}"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo -e "${RED}❌ Python 3 not found.${NC}"; exit 1; }

# ─── 1. Rust Backend (neurax-service) ─────────────────────────────────────────

echo -e "${YELLOW}🔧 Starting neurax-service (Rust backend) on port 9098...${NC}"
cargo run -p neurax-service &
SERVICE_PID=$!
echo -e "${GREEN}✓ neurax-service started (PID: $SERVICE_PID)${NC}"

# ─── 2. Python Agent (neurax-agent) ───────────────────────────────────────────

echo -e "${YELLOW}🤖 Starting neurax-agent (FastAPI) on port 8099...${NC}"
cd "$SCRIPT_DIR/neurax-agent"

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt

python3 -m uvicorn app:app --host 127.0.0.1 --port 8099 --reload &
AGENT_PID=$!
deactivate
echo -e "${GREEN}✓ neurax-agent started (PID: $AGENT_PID)${NC}"

# ─── 3. React Frontend (neurax-ui) ────────────────────────────────────────────

echo -e "${YELLOW}🌐 Starting neurax-ui (React) on port 8081...${NC}"
cd "$SCRIPT_DIR/neurax-ui"

# Install deps if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}📦 Installing frontend dependencies...${NC}"
    npm install
fi

npm run dev &
UI_PID=$!
echo -e "${GREEN}✓ neurax-ui started (PID: $UI_PID)${NC}"

# ─── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  NEURAX Development Services Started${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "  Backend:  http://localhost:9098  (PID: $SERVICE_PID)"
echo -e "  Agent:    http://localhost:8099  (PID: $AGENT_PID)"
echo -e "  Frontend: http://localhost:8081  (PID: $UI_PID)"
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo "Press Ctrl+C to stop all services."

# Wait for all background processes
wait
