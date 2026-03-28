#!/bin/bash

# PersonaPlex 2-Bit Quantized Model Deployment Script
# Deploys the smallest 2-bit quantized model as a persistent background service

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==========================================="
echo "  PersonaPlex 2-Bit Quantized Deployment"
echo "==========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
SERVER_PORT=8998
MODEL_DIR="$SCRIPT_DIR/models/personaplex-7b-turbo2bit"
SERVER_LOG="$SCRIPT_DIR/server_2bit.log"
TUNNEL_LOG="$SCRIPT_DIR/tunnel_2bit.log"

# Check if model exists
if [ ! -f "$MODEL_DIR/model-turbo2bit.safetensors" ]; then
    echo -e "${RED}✗ 2-bit model not found at $MODEL_DIR${NC}"
    echo "Please download the model first."
    exit 1
fi

echo -e "${GREEN}✓ Model found: $MODEL_DIR${NC}"
echo ""

# Kill any existing processes
echo -e "${YELLOW}Stopping any existing services...${NC}"
pkill -f "moshi.server" 2>/dev/null || true
pkill -f "cloudflared" 2>/dev/null || true
sleep 2

# Start the server in background
echo -e "${CYAN}Starting PersonaPlex server with 2-bit model...${NC}"
cd "$SCRIPT_DIR/personaplex-setup"

export PYTHONPATH="$SCRIPT_DIR/personaplex-setup/moshi:$PYTHONPATH"

nohup python3 -m moshi.server \
    --hf-repo "$MODEL_DIR" \
    --moshi-weight "$MODEL_DIR/model-turbo2bit.safetensors" \
    --port $SERVER_PORT \
    > "$SERVER_LOG" 2>&1 &

SERVER_PID=$!
echo -e "${GREEN}✓ Server started (PID: $SERVER_PID)${NC}"
echo "  Logs: $SERVER_LOG"
echo ""

# Wait for server to start
echo -e "${YELLOW}Waiting for server to initialize...${NC}"
sleep 10

# Check if server is running
if ps -p $SERVER_PID > /dev/null; then
    echo -e "${GREEN}✓ Server is running${NC}"
else
    echo -e "${RED}✗ Server failed to start${NC}"
    echo "Check logs: $SERVER_LOG"
    tail -50 "$SERVER_LOG"
    exit 1
fi

# Start Cloudflare tunnel in background
echo -e "${CYAN}Starting Cloudflare tunnel...${NC}"

if ! command -v cloudflared &>/dev/null; then
    echo -e "${YELLOW}⚠ cloudflared not found - installing...${NC}"
    ARCH=$(uname -m)
    case $ARCH in
        x86_64) ARCH="amd64" ;;
        aarch64) ARCH="arm64" ;;
        *) echo -e "${RED}✗ Unsupported architecture: $ARCH${NC}"; exit 1 ;;
    esac
    
    curl -L -o cloudflared "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$ARCH"
    chmod +x cloudflared
    sudo mv cloudflared /usr/local/bin/
fi

nohup cloudflared tunnel --url "http://localhost:$SERVER_PORT" --no-autoupdate > "$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!
echo -e "${GREEN}✓ Cloudflare tunnel started (PID: $TUNNEL_PID)${NC}"
echo "  Logs: $TUNNEL_LOG"
echo ""

# Wait for tunnel URL
echo -e "${YELLOW}Waiting for tunnel URL...${NC}"
for i in {1..30}; do
    if grep -q "trycloudflare.com" "$TUNNEL_LOG" 2>/dev/null; then
        CLOUDFLARE_URL=$(grep -oP 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" | tail -1)
        if [ -n "$CLOUDFLARE_URL" ]; then
            echo -e "${GREEN}✓ Cloudflare Tunnel URL:${NC}"
            echo "  $CLOUDFLARE_URL"
            echo ""
            break
        fi
    fi
    sleep 2
done

# If URL not found, show log location
if [ -z "$CLOUDFLARE_URL" ]; then
    echo -e "${YELLOW}⚠ Tunnel started, check $TUNNEL_LOG for URL${NC}"
    CLOUDFLARE_URL="Check $TUNNEL_LOG for URL"
fi

# Save PIDs to file for later use
echo "$SERVER_PID" > "$SCRIPT_DIR/server_2bit.pid"
echo "$TUNNEL_PID" > "$SCRIPT_DIR/tunnel_2bit.pid"

# Print summary
echo "==========================================="
echo -e "${GREEN}  DEPLOYMENT COMPLETE${NC}"
echo "==========================================="
echo ""
echo -e "${CYAN}Service Information:${NC}"
echo "  Model: personaplex-7b-turbo2bit (2-bit quantized, ~2GB)"
echo "  Server PID: $SERVER_PID"
echo "  Tunnel PID: $TUNNEL_PID"
echo "  Server Port: $SERVER_PORT"
echo ""
echo -e "${CYAN}Access URLs:${NC}"
echo "  Local: http://localhost:$SERVER_PORT"
echo "  Public: $CLOUDFLARE_URL"
echo ""
echo -e "${CYAN}Log Files:${NC}"
echo "  Server: $SERVER_LOG"
echo "  Tunnel: $TUNNEL_LOG"
echo ""
echo -e "${CYAN}PID Files:${NC}"
echo "  Server: $SCRIPT_DIR/server_2bit.pid"
echo "  Tunnel: $SCRIPT_DIR/tunnel_2bit.pid"
echo ""
echo -e "${YELLOW}To stop services:${NC}"
echo "  kill \$(cat $SCRIPT_DIR/server_2bit.pid)"
echo "  kill \$(cat $SCRIPT_DIR/tunnel_2bit.pid)"
echo ""
echo "==========================================="
