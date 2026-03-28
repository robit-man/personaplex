#!/bin/bash

# PersonaPlex Single Point of Entry Automation Script
# This script handles: server startup, cloudflare tunnel, and auto-deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "  PersonaPlex Voice Cloning System"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
export HF_TOKEN="${HF_TOKEN:-}"
SERVER_PORT="${PERSONAPLEX_PORT:-8998}"
TUNNEL_NAME="${PERSONAPLEX_TUNNEL_NAME:-personaplex-voice}"

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo -e "${RED}Error: HF_TOKEN environment variable is not set.${NC}"
    echo "Please run: export HF_TOKEN=your_huggingface_token"
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo "Also make sure you've accepted the license at: https://huggingface.co/nvidia/personaplex-7b-v1"
    exit 1
fi

echo -e "${GREEN}✓ HuggingFace token configured${NC}"

# Function to start the server
start_server() {
    echo -e "${BLUE}Starting PersonaPlex server on port $SERVER_PORT...${NC}"
    
    # Add moshi to Python path and run server (no SSL for cloudflared tunnel)
    export PYTHONPATH="$SCRIPT_DIR/personaplex-setup/moshi:$PYTHONPATH"
    
    # Start server in background
    cd "$SCRIPT_DIR/personaplex-setup"
    python3 -m moshi.server --port "$SERVER_PORT" 2>&1 | tee server.log &
    SERVER_PID=$!
    
    echo -e "${GREEN}✓ Server started (PID: $SERVER_PID)${NC}"
    echo -e "${YELLOW}Server logs: server.log${NC}"
    
    # Wait for server to be ready
    echo "Waiting for server to be ready..."
    for i in {1..30}; do
        if curl -s "http://localhost:$SERVER_PORT" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Server is ready!${NC}"
            break
        fi
        sleep 1
    done
    
    # Return to script directory
    cd "$SCRIPT_DIR"
}

# Function to start cloudflare tunnel
start_tunnel() {
    echo -e "${BLUE}Starting Cloudflare tunnel...${NC}"
    
    # Kill any existing tunnel with the same name
    pkill -f "cloudflared.*$TUNNEL_NAME" || true
    
    # Start new tunnel
    cloudflared tunnel --url "http://localhost:$SERVER_PORT" 2>&1 | tee tunnel.log &
    TUNNEL_PID=$!
    
    echo -e "${GREEN}✓ Tunnel started (PID: $TUNNEL_PID)${NC}"
    echo -e "${YELLOW}Tunnel logs: tunnel.log${NC}"
}

# Function to get tunnel URL
get_tunnel_url() {
    echo ""
    echo -e "${BLUE}Fetching tunnel URL...${NC}"
    
    # Wait for tunnel to establish
    sleep 3
    
    # Extract URL from tunnel logs
    if [ -f tunnel.log ]; then
        TUNNEL_URL=$(grep -oP 'https://[\w-]+\.trycloudflare\.com' tunnel.log | tail -1)
        if [ -n "$TUNNEL_URL" ]; then
            echo ""
            echo -e "${GREEN}=========================================${NC}"
            echo -e "${GREEN}  🌐 PersonaPlex is LIVE!${NC}"
            echo -e "${GREEN}=========================================${NC}"
            echo ""
            echo -e "${GREEN}  URL: $TUNNEL_URL${NC}"
            echo ""
            echo -e "${YELLOW}  API Endpoints:${NC}"
            echo -e "  - Clone Voice:  $TUNNEL_URL/api/clone${NC}"
            echo -e "  - List Voices:  $TUNNEL_URL/api/voices${NC}"
            echo -e "  - Generate:     $TUNNEL_URL/api/generate${NC}"
            echo -e "  - Status:       $TUNNEL_URL/api/status${NC}"
            echo ""
        fi
    fi
}

# Function to show API documentation
show_api_docs() {
    echo ""
    echo -e "${BLUE}=========================================${NC}"
    echo -e "${BLUE}  📚 API Documentation${NC}"
    echo -e "${BLUE}=========================================${NC}"
    echo ""
    echo -e "${YELLOW}Voice Cloning:${NC}"
    echo "POST /api/clone"
    echo "  Body: {"
    echo "    \"voice_name\": \"my_voice\","
    echo "    \"audio_file\": \"<base64_encoded_audio>\""
    echo "  }"
    echo ""
    echo -e "${YELLOW}List Voices:${NC}"
    echo "GET /api/voices"
    echo ""
    echo -e "${YELLOW}Generate Speech:${NC}"
    echo "POST /api/generate"
    echo "  Body: {"
    echo "    \"text\": \"Hello, world!\","
    echo "    \"voice_name\": \"my_voice\""
    echo "  }"
    echo ""
    echo -e "${YELLOW}Check Status:${NC}"
    echo "GET /api/status"
    echo ""
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    
    # Kill server
    if [ -n "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
        echo -e "${GREEN}✓ Server stopped${NC}"
    fi
    
    # Kill tunnel
    if [ -n "$TUNNEL_PID" ]; then
        kill $TUNNEL_PID 2>/dev/null || true
        echo -e "${GREEN}✓ Tunnel stopped${NC}"
    fi
    
    # Kill any remaining cloudflared processes
    pkill -f "cloudflared.*$TUNNEL_NAME" || true
    
    echo -e "${GREEN}✓ Cleanup complete${NC}"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Main execution
case "${1:-start}" in
    start)
        echo -e "${GREEN}Starting PersonaPlex system...${NC}"
        start_server
        start_tunnel
        get_tunnel_url
        show_api_docs
        
        echo ""
        echo -e "${YELLOW}Press Ctrl+C to stop the system${NC}"
        echo ""
        
        # Keep script running
        wait
        ;;
    
    server-only)
        echo -e "${GREEN}Starting server only (no tunnel)...${NC}"
        start_server
        echo ""
        echo -e "${GREEN}Server running at: http://localhost:$SERVER_PORT${NC}"
        echo ""
        
        # Keep script running
        wait
        ;;
    
    tunnel-only)
        echo -e "${GREEN}Starting tunnel only (assuming server is running)...${NC}"
        start_tunnel
        get_tunnel_url
        
        # Keep script running
        wait
        ;;
    
    api-docs)
        show_api_docs
        ;;
    
    status)
        echo -e "${BLUE}Checking system status...${NC}"
        
        # Check server
        if curl -s "http://localhost:$SERVER_PORT" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ Server is running on port $SERVER_PORT${NC}"
        else
            echo -e "${RED}✗ Server is not running${NC}"
        fi
        
        # Check tunnel
        if pgrep -f "cloudflared.*$TUNNEL_NAME" > /dev/null; then
            echo -e "${GREEN}✓ Cloudflare tunnel is running${NC}"
            if [ -f tunnel.log ]; then
                TUNNEL_URL=$(grep -oP 'https://[\w-]+\.trycloudflare\.com' tunnel.log | tail -1)
                if [ -n "$TUNNEL_URL" ]; then
                    echo -e "${GREEN}  URL: $TUNNEL_URL${NC}"
                fi
            fi
        else
            echo -e "${RED}✗ Cloudflare tunnel is not running${NC}"
        fi
        ;;
    
    stop)
        echo -e "${YELLOW}Stopping PersonaPlex system...${NC}"
        cleanup
        ;;
    
    *)
        echo "Usage: $0 {start|server-only|tunnel-only|api-docs|status|stop}"
        echo ""
        echo "Commands:"
        echo "  start       Start server and tunnel (default)"
        echo "  server-only Start server only (no tunnel)"
        echo "  tunnel-only Start tunnel only (server must be running)"
        echo "  api-docs    Show API documentation"
        echo "  status      Check system status"
        echo "  stop        Stop all services"
        exit 1
        ;;
esac
