#!/bin/bash

# PersonaPlex Single Point of Entry Automation Script
# This script handles: server startup, cloudflare tunnel, localhost.run tunnel, and auto-deployment
# Supports both full bf16 weights and NF4 quantized weights for edge deployment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==========================================="
echo "  PersonaPlex Voice Cloning System"
echo "==========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
export HF_TOKEN="${HF_TOKEN:-}"
SERVER_PORT="${PERSONAPLEX_PORT:-8998}"
TUNNEL_NAME="${PERSONAPLEX_TUNNEL_NAME:-personaplex-voice}"

# Model configuration
# Options: "full" (bf16, 15.59GB), "nf4" (INT4, 4.14GB), or "turbo2bit" (2-bit, 1.86GB)
MODEL_TYPE="${PERSONAPLEX_MODEL_TYPE:-full}"

# Model repositories
FULL_MODEL="nvidia/personaplex-7b-v1"
NF4_MODEL="cudabenchmarktest/personaplex-7b-nf4"
TURBO2BIT_MODEL="personaplex-7b-turbo2bit"

# Model paths
MODELS_DIR="$SCRIPT_DIR/models"
FULL_MODEL_DIR="$MODELS_DIR/$FULL_MODEL"
NF4_MODEL_DIR="$MODELS_DIR/$NF4_MODEL"
TURBO2BIT_MODEL_DIR="$MODELS_DIR/$TURBO2BIT_MODEL"

# Current model being used
CURRENT_MODEL_DIR=""

# Process IDs for cleanup
SERVER_PID=""
TUNNEL_PID=""
SSH_TUNNEL_PID=""

# Cleanup function
cleanup() {
    echo ""
    echo -e "${YELLOW}Stopping services...${NC}"
    
    [ -n "$SERVER_PID" ] && kill $SERVER_PID 2>/dev/null || true
    [ -n "$TUNNEL_PID" ] && kill $TUNNEL_PID 2>/dev/null || true
    [ -n "$SSH_TUNNEL_PID" ] && kill $SSH_TUNNEL_PID 2>/dev/null || true
    
    # Kill any remaining cloudflared processes
    pkill -f "cloudflared.*$TUNNEL_NAME" 2>/dev/null || true
    
    echo -e "${GREEN}All services stopped.${NC}"
}

trap cleanup EXIT INT TERM

# Display model info
show_model_info() {
    echo -e "${CYAN}Model Configuration:${NC}"
    echo "  Type: $MODEL_TYPE"
    if [ "$MODEL_TYPE" = "full" ]; then
        echo "  Repository: $FULL_MODEL"
        echo "  Size: ~15.59 GB (bf16)"
        echo "  Best for: Desktop GPUs with 16GB+ VRAM"
    elif [ "$MODEL_TYPE" = "nf4" ]; then
        echo "  Repository: $NF4_MODEL"
        echo "  Size: ~4.14 GB (INT4 NF4 quantized)"
        echo "  Best for: Edge devices, Jetson AGX Orin, 8GB VRAM GPUs"
    elif [ "$MODEL_TYPE" = "turbo2bit" ]; then
        echo "  Repository: $TURBO2BIT_MODEL"
        echo "  Size: ~1.86 GB (2-bit TurboQuant NF2+WHT)"
        echo "  Dequantized Size: ~16.6 GB (matches full bf16)"
        echo "  Best for: Fast downloads, edge deployment with 16GB+ VRAM"
    fi
    echo ""
}

# Check and download models
download_models() {
    echo -e "${CYAN}Checking model availability...${NC}"
    
    # Create models directory
    mkdir -p "$MODELS_DIR"
    
    if [ "$MODEL_TYPE" = "full" ]; then
        CURRENT_MODEL_DIR="$FULL_MODEL_DIR"
        if [ -d "$CURRENT_MODEL_DIR" ] && [ -n "$(ls -A $CURRENT_MODEL_DIR 2>/dev/null)" ]; then
            echo -e "${GREEN}✓ Full model already downloaded${NC}"
            return 0
        fi
    elif [ "$MODEL_TYPE" = "nf4" ]; then
        CURRENT_MODEL_DIR="$NF4_MODEL_DIR"
        if [ -d "$CURRENT_MODEL_DIR" ] && [ -n "$(ls -A $CURRENT_MODEL_DIR 2>/dev/null)" ]; then
            echo -e "${GREEN}✓ NF4 quantized model already downloaded${NC}"
            return 0
        fi
    elif [ "$MODEL_TYPE" = "turbo2bit" ]; then
        CURRENT_MODEL_DIR="$TURBO2BIT_MODEL_DIR"
        if [ -d "$CURRENT_MODEL_DIR" ] && [ -n "$(ls -A $CURRENT_MODEL_DIR 2>/dev/null)" ]; then
            echo -e "${GREEN}✓ 2-bit TurboQuant model already downloaded${NC}"
            return 0
        fi
    fi
    
    # Check for HF_TOKEN
    if [ -z "$HF_TOKEN" ]; then
        echo -e "${RED}✗ HF_TOKEN not set${NC}"
        echo ""
        echo "To download models, you need a HuggingFace token:"
        echo "  1. Get token: https://huggingface.co/settings/tokens"
        echo "  2. Accept license: https://huggingface.co/nvidia/personaplex-7b-v1"
        echo "  3. Set token: export HF_TOKEN=your_token_here"
        echo ""
        echo "Or use the NF4 quantized model (smaller, faster to download):"
        echo "  export PERSONAPLEX_MODEL_TYPE=nf4"
        echo "  export HF_TOKEN=your_token_here"
        echo "  ./run.sh start"
        echo ""
        echo "Or use the 2-bit TurboQuant model (smallest, fastest download):"
        echo "  export PERSONAPLEX_MODEL_TYPE=turbo2bit"
        echo "  export HF_TOKEN=your_token_here"
        echo "  ./run.sh start"
        echo ""
        exit 1
    fi
    
    echo -e "${YELLOW}Downloading model (this may take a while)...${NC}"
    
    # Use huggingface-cli if available, otherwise use git
    if command -v huggingface-cli &>/dev/null; then
        if [ "$MODEL_TYPE" = "full" ]; then
            huggingface-cli download $FULL_MODEL --local-dir "$CURRENT_MODEL_DIR" --token "$HF_TOKEN"
        elif [ "$MODEL_TYPE" = "nf4" ]; then
            huggingface-cli download $NF4_MODEL --local-dir "$CURRENT_MODEL_DIR" --token "$HF_TOKEN"
        elif [ "$MODEL_TYPE" = "turbo2bit" ]; then
            huggingface-cli download $TURBO2BIT_MODEL --local-dir "$CURRENT_MODEL_DIR" --token "$HF_TOKEN"
        fi
    else
        # Fallback to git lfs
        echo "Using git lfs to download models..."
        git lfs install
        if [ "$MODEL_TYPE" = "full" ]; then
            git clone "https://$HF_TOKEN@huggingface.co/$FULL_MODEL" "$CURRENT_MODEL_DIR" 2>/dev/null || \
            git clone "https://huggingface.co/$FULL_MODEL" "$CURRENT_MODEL_DIR"
        elif [ "$MODEL_TYPE" = "nf4" ]; then
            git clone "https://$HF_TOKEN@huggingface.co/$NF4_MODEL" "$CURRENT_MODEL_DIR" 2>/dev/null || \
            git clone "https://huggingface.co/$NF4_MODEL" "$CURRENT_MODEL_DIR"
        elif [ "$MODEL_TYPE" = "turbo2bit" ]; then
            git clone "https://$HF_TOKEN@huggingface.co/$TURBO2BIT_MODEL" "$CURRENT_MODEL_DIR" 2>/dev/null || \
            git clone "https://huggingface.co/$TURBO2BIT_MODEL" "$CURRENT_MODEL_DIR"
        fi
    fi
    
    if [ -d "$CURRENT_MODEL_DIR" ] && [ -n "$(ls -A $CURRENT_MODEL_DIR 2>/dev/null)" ]; then
        echo -e "${GREEN}✓ Model downloaded successfully${NC}"
    else
        echo -e "${RED}✗ Failed to download model${NC}"
        exit 1
    fi
}

# Start the PersonaPlex server
start_server() {
    echo -e "${CYAN}Starting PersonaPlex server...${NC}"
    
    cd "$SCRIPT_DIR/personaplex-setup"
    
    # Set model path environment variable
    export PERSONAPLEX_MODEL_PATH="$CURRENT_MODEL_DIR"
    
    # Start server in background
    nohup python3 -m moshi.server > server.log 2>&1 &
    SERVER_PID=$!
    
    echo -e "${GREEN}✓ Server started (PID: $SERVER_PID)${NC}"
    echo "  Logs: personaplex-setup/server.log"
    echo ""
    
    # Wait for server to start
    sleep 5
    
    # Check if server is running
    if ! curl -s "http://localhost:$SERVER_PORT" > /dev/null; then
        echo -e "${YELLOW}⚠ Server may still be initializing...${NC}"
        sleep 10
    fi
}

# Start Cloudflare tunnel
start_cloudflare_tunnel() {
    echo -e "${CYAN}Starting Cloudflare tunnel...${NC}"
    
    if ! command -v cloudflared &>/dev/null; then
        echo -e "${YELLOW}⚠ cloudflared not found - installing...${NC}"
        
        # Download cloudflared
        ARCH=$(uname -m)
        case $ARCH in
            x86_64) ARCH="amd64" ;;
            aarch64) ARCH="arm64" ;;
            *) echo -e "${RED}✗ Unsupported architecture: $ARCH${NC}"; return 1 ;;
        esac
        
        curl -L -o cloudflared "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$ARCH"
        chmod +x cloudflared
        sudo mv cloudflared /usr/local/bin/
        
        if ! command -v cloudflared &>/dev/null; then
            echo -e "${RED}✗ Failed to install cloudflared${NC}"
            return 1
        fi
    fi
    
    # Start tunnel in background
    nohup cloudflared tunnel --url "http://localhost:$SERVER_PORT" --no-autoupdate > tunnel.log 2>&1 &
    TUNNEL_PID=$!
    
    echo -e "${GREEN}✓ Cloudflare tunnel started (PID: $TUNNEL_PID)${NC}"
    echo "  Logs: tunnel.log"
    echo ""
    
    # Extract tunnel URL from log
    sleep 3
    
    # Wait for tunnel URL to appear
    for i in {1..20}; do
        if grep -q "trycloudflare.com" tunnel.log 2>/dev/null; then
            CLOUDFLARE_URL=$(grep -oP 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' tunnel.log | tail -1)
            if [ -n "$CLOUDFLARE_URL" ]; then
                echo -e "${GREEN}✓ Cloudflare Tunnel URL:${NC}"
                echo "  $CLOUDFLARE_URL"
                echo ""
                return 0
            fi
        fi
        sleep 2
    done
    
    echo -e "${YELLOW}⚠ Cloudflare tunnel started, check tunnel.log for URL${NC}"
    return 0
}

# Start localhost.run SSH tunnel
start_ssh_tunnel() {
    echo -e "${CYAN}Starting localhost.run SSH tunnel...${NC}"
    
    if ! command -v ssh &>/dev/null; then
        echo -e "${YELLOW}⚠ ssh not found - please install openssh-client${NC}"
        return 1
    fi
    
    # Start SSH tunnel in background
    nohup ssh -o StrictHostKeyChecking=no -R 80:localhost:$SERVER_PORT nokey@localhost.run > ssh_tunnel.log 2>&1 &
    SSH_TUNNEL_PID=$!
    
    echo -e "${GREEN}✓ SSH tunnel started (PID: $SSH_TUNNEL_PID)${NC}"
    echo "  Logs: ssh_tunnel.log"
    echo ""
    
    # Extract tunnel URL from log
    sleep 3
    
    # Wait for tunnel URL to appear
    for i in {1..20}; do
        if grep -qE "(lhr\.life|lhr\.rocks|lhr\.me)" ssh_tunnel.log 2>/dev/null; then
            SSH_URL=$(grep -oP 'https://[a-zA-Z0-9]+\.(lhr\.life|lhr\.rocks|lhr\.me)' ssh_tunnel.log | tail -1)
            if [ -n "$SSH_URL" ]; then
                echo -e "${GREEN}✓ localhost.run Tunnel URL:${NC}"
                echo "  $SSH_URL"
                echo ""
                return 0
            fi
        fi
        sleep 2
    done
    
    echo -e "${YELLOW}⚠ SSH tunnel started, check ssh_tunnel.log for URL${NC}"
    return 0
}

# Show API documentation
show_api_docs() {
    echo ""
    echo -e "${CYAN}===========================================${NC}"
    echo -e "${CYAN}  PersonaPlex API Documentation${NC}"
    echo -e "${CYAN}===========================================${NC}"
    echo ""
    echo "Base URL: http://localhost:$SERVER_PORT"
    echo ""
    echo "--- Clone Voice ---"
    echo "POST /api/clone"
    echo "Content-Type: application/json"
    echo ""
    echo '{'
    echo '  "voice_name": "my_voice",'
    echo '  "audio_file": "<base64_encoded_audio>"'
    echo '}'
    echo ""
    echo "--- List Voices ---"
    echo "GET /api/voices"
    echo ""
    echo "--- Generate Speech ---"
    echo "POST /api/generate"
    echo "Content-Type: application/json"
    echo ""
    echo '{'
    echo '  "text": "Hello, world!",'
    echo '  "voice_name": "my_voice"'
    echo '}'
    echo ""
    echo "--- Check Status ---"
    echo "GET /api/status"
    echo ""
}

# Show system status
show_status() {
    echo ""
    echo -e "${CYAN}===========================================${NC}"
    echo -e "${CYAN}  System Status${NC}"
    echo -e "${CYAN}===========================================${NC}"
    echo ""
    
    # Server status
    if curl -s "http://localhost:$SERVER_PORT" > /dev/null 2>&1; then
        echo -e "Server:     ${GREEN}Running${NC} on port $SERVER_PORT"
    else
        echo -e "Server:     ${RED}Not running${NC}"
    fi
    
    # Model status
    if [ -d "$FULL_MODEL_DIR" ] && [ -n "$(ls -A $FULL_MODEL_DIR 2>/dev/null)" ]; then
        echo -e "Full Model: ${GREEN}Downloaded${NC}"
    else
        echo -e "Full Model: ${RED}Not downloaded${NC}"
    fi
    
    if [ -d "$NF4_MODEL_DIR" ] && [ -n "$(ls -A $NF4_MODEL_DIR 2>/dev/null)" ]; then
        echo -e "NF4 Model:  ${GREEN}Downloaded${NC}"
    else
        echo -e "NF4 Model:  ${RED}Not downloaded${NC}"
    fi
    
    if [ -d "$TURBO2BIT_MODEL_DIR" ] && [ -n "$(ls -A $TURBO2BIT_MODEL_DIR 2>/dev/null)" ]; then
        echo -e "2-bit Model: ${GREEN}Downloaded${NC}"
    else
        echo -e "2-bit Model: ${RED}Not downloaded${NC}"
    fi
    
    # Tunnel status
    if [ -f "tunnel.log" ] && grep -q "trycloudflare.com" tunnel.log 2>/dev/null; then
        CLOUDFLARE_URL=$(grep -oP 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' tunnel.log | tail -1)
        echo -e "Cloudflare: ${GREEN}Active${NC} - $CLOUDFLARE_URL"
    else
        echo -e "Cloudflare: ${RED}Not active${NC}"
    fi
    
    if [ -f "ssh_tunnel.log" ] && grep -qE "(lhr\.life|lhr\.rocks|lhr\.me)" ssh_tunnel.log 2>/dev/null; then
        SSH_URL=$(grep -oP 'https://[a-zA-Z0-9]+\.(lhr\.life|lhr\.rocks|lhr\.me)' ssh_tunnel.log | tail -1)
        echo -e "localhost.run: ${GREEN}Active${NC} - $SSH_URL"
    else
        echo -e "localhost.run: ${RED}Not active${NC}"
    fi
    
    echo ""
}

# Show usage
show_usage() {
    echo ""
    echo "Usage: $0 {start|start-nf4|start-turbo2bit|server-only|tunnel-only|ssh-tunnel|api-docs|status|stop|switch-model}"
    echo ""
    echo "Commands:"
    echo "  start             Start server + Cloudflare tunnel (full bf16 model)"
    echo "  start-nf4         Start server + Cloudflare tunnel (NF4 quantized model)"
    echo "  start-turbo2bit   Start server + Cloudflare tunnel (2-bit TurboQuant model)"
    echo "  server-only       Start server only (localhost)"
    echo "  tunnel-only       Start Cloudflare tunnel only"
    echo "  ssh-tunnel        Start localhost.run SSH tunnel only"
    echo "  api-docs          Show API documentation"
    echo "  status            Check system status"
    echo "  stop              Stop all services"
    echo "  switch-model      Switch between full and nf4 models"
    echo ""
    echo "Environment Variables:"
    echo "  HF_TOKEN              HuggingFace API token"
    echo "  PERSONAPLEX_PORT      Server port (default: 8998)"
    echo "  PERSONAPLEX_MODEL_TYPE Model type: 'full', 'nf4', or 'turbo2bit' (default: full)"
    echo ""
    echo "Examples:"
    echo "  ./run.sh start                                    # Start with full model"
    echo "  ./run.sh start-nf4                                # Start with NF4 model"
    echo "  ./run.sh start-turbo2bit                          # Start with 2-bit TurboQuant model"
    echo "  PERSONAPLEX_MODEL_TYPE=nf4 ./run.sh start         # Start with NF4 model"
    echo "  ./run.sh switch-model                             # Toggle between models"
    echo ""
}

# Switch between models
switch_model() {
    if [ "$MODEL_TYPE" = "full" ]; then
        export PERSONAPLEX_MODEL_TYPE="nf4"
        echo -e "${GREEN}Switched to NF4 quantized model${NC}"
        echo "Run: PERSONAPLEX_MODEL_TYPE=nf4 ./run.sh start"
    else
        export PERSONAPLEX_MODEL_TYPE="full"
        echo -e "${GREEN}Switched to full bf16 model${NC}"
        echo "Run: ./run.sh start"
    fi
}

# Main command handling
case "${1:-start}" in
    start)
        show_model_info
        download_models
        start_server
        start_cloudflare_tunnel
        echo -e "${CYAN}===========================================${NC}"
        echo -e "${CYAN}  Access URLs:${NC}"
        echo -e "${CYAN}===========================================${NC}"
        echo ""
        echo -e "Local:      http://localhost:$SERVER_PORT"
        if [ -f "tunnel.log" ]; then
            CLOUDFLARE_URL=$(grep -oP 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' tunnel.log | tail -1)
            [ -n "$CLOUDFLARE_URL" ] && echo -e "Cloudflare: $CLOUDFLARE_URL"
        fi
        echo ""
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        echo ""
        
        # Keep script running and show logs
        tail -f tunnel.log
        ;;
    
    start-nf4)
        export PERSONAPLEX_MODEL_TYPE="nf4"
        show_model_info
        download_models
        start_server
        start_cloudflare_tunnel
        echo -e "${CYAN}===========================================${NC}"
        echo -e "${CYAN}  Access URLs:${NC}"
        echo -e "${CYAN}===========================================${NC}"
        echo ""
        echo -e "Local:      http://localhost:$SERVER_PORT"
        if [ -f "tunnel.log" ]; then
            CLOUDFLARE_URL=$(grep -oP 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' tunnel.log | tail -1)
            [ -n "$CLOUDFLARE_URL" ] && echo -e "Cloudflare: $CLOUDFLARE_URL"
        fi
        echo ""
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        echo ""
        
        # Keep script running and show logs
        tail -f tunnel.log
        ;;
    
    start-turbo2bit)
        export PERSONAPLEX_MODEL_TYPE="turbo2bit"
        show_model_info
        download_models
        start_server
        start_cloudflare_tunnel
        echo -e "${CYAN}===========================================${NC}"
        echo -e "${CYAN}  Access URLs:${NC}"
        echo -e "${CYAN}===========================================${NC}"
        echo ""
        echo -e "Local:      http://localhost:$SERVER_PORT"
        if [ -f "tunnel.log" ]; then
            CLOUDFLARE_URL=$(grep -oP 'https://[a-zA-Z0-9-]+\.trycloudflare\.com' tunnel.log | tail -1)
            [ -n "$CLOUDFLARE_URL" ] && echo -e "Cloudflare: $CLOUDFLARE_URL"
        fi
        echo ""
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        echo ""
        
        # Keep script running and show logs
        tail -f tunnel.log
        ;;
    
    server-only)
        show_model_info
        download_models
        start_server
        echo -e "${GREEN}Server is running at: http://localhost:$SERVER_PORT${NC}"
        echo ""
        tail -f personaplex-setup/server.log
        ;;
    
    tunnel-only)
        start_cloudflare_tunnel
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        tail -f tunnel.log
        ;;
    
    ssh-tunnel)
        start_ssh_tunnel
        echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
        tail -f ssh_tunnel.log
        ;;
    
    api-docs)
        show_api_docs
        ;;
    
    status)
        show_status
        ;;
    
    stop)
        cleanup
        exit 0
        ;;
    
    switch-model)
        switch_model
        ;;
    
    *)
        show_usage
        exit 1
        ;;
esac
