#!/bin/bash

# PersonaPlex 2-Bit Quantized Server Launcher
# Uses the smallest 2-bit quantized model (2.07GB)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  PersonaPlex 2-Bit Quantized Server${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Model: cudabenchmarktest/personaplex-7b-turbo2bit${NC}"
echo -e "${YELLOW}Size: 2.07 GB (7.5x compression)${NC}"
echo ""

# Check if model exists
MODEL_DIR="models/personaplex-7b-turbo2bit"
if [ ! -f "$MODEL_DIR/model-turbo2bit.safetensors" ]; then
    echo -e "${RED}✗ Model not found!${NC}"
    echo "Please download the model first."
    exit 1
fi

# Activate venv
source personaplex-setup/venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR/personaplex-setup/moshi:$PYTHONPATH"

echo -e "${GREEN}Starting server in background...${NC}"
echo ""

# Start server in background with nohup
nohup python3 -m moshi.server \
    --moshi-weight "$MODEL_DIR/model-turbo2bit.safetensors" \
    --mimi-weight "$MODEL_DIR/tokenizer-e351c8d8-checkpoint125.safetensors" \
    --tokenizer "$MODEL_DIR/tokenizer_spm_32k_3.model" \
    --port 8998 \
    --device cuda \
    > server_2bit.log 2>&1 &

SERVER_PID=$!
echo $SERVER_PID > server_2bit.pid

echo -e "${GREEN}✓ Server started (PID: $SERVER_PID)${NC}"
echo -e "${GREEN}✓ Logs: server_2bit.log${NC}"
echo ""
echo -e "${YELLOW}Waiting for server to initialize...${NC}"
sleep 10

# Check if server is running
if ps -p $SERVER_PID > /dev/null; then
    echo -e "${GREEN}✓ Server is running!${NC}"
    echo ""
    echo -e "${BLUE}Access the Web UI at:${NC}"
    echo -e "  http://localhost:8998"
    echo ""
    echo -e "${YELLOW}To stop the server:${NC}"
    echo -e "  kill $(cat server_2bit.pid)"
else
    echo -e "${RED}✗ Server failed to start. Check server_2bit.log${NC}"
    tail -20 server_2bit.log
fi
