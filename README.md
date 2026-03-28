# PersonaPlex Voice Cloning System

> **Single Point of Entry** - Automated setup and deployment for NVIDIA's PersonaPlex voice cloning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/🤗-Weights-yellow)](https://huggingface.co/nvidia/personaplex-7b-v1)

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** with venv
- **Node.js 18+** (for frontend)
- **Cloudflare CLI** (`cloudflared`) for tunnel
- **SSH client** (for localhost.run tunnel)
- **HuggingFace Token** (optional if models pre-downloaded)

### Installation

```bash
# Clone the repository
git clone https://github.com/roko/PersonaPlex.git
cd PersonaPlex

# Install dependencies (handled by run.sh)
chmod +x run.sh
```

### Run

```bash
# Start everything (server + Cloudflare tunnel) with full model
export HF_TOKEN=your_huggingface_token  # Optional if models already downloaded
./run.sh start

# Start with NF4 quantized model (smaller, faster)
./run.sh start-nf4

# Start with 2-bit TurboQuant model (smallest, fastest download)
./run.sh start-turbo2bit
```

**That's it!** The script handles:
- ✅ PersonaPlex server startup
- ✅ Cloudflare tunnel for public access
- ✅ localhost.run SSH tunnel (alternative tunneling)
- ✅ Auto-detection of downloaded models
- ✅ Support for full bf16, NF4 quantized, and 2-bit TurboQuant models
- ✅ Automatic dequantization of 2-bit models to bf16 on load
- ✅ API documentation display

## 📋 Commands

```bash
./run.sh start             # Start server + Cloudflare tunnel (full bf16 model)
./run.sh start-nf4         # Start server + Cloudflare tunnel (NF4 quantized model)
./run.sh start-turbo2bit   # Start server + Cloudflare tunnel (2-bit TurboQuant model)
./run.sh server-only       # Start server only (localhost)
./run.sh tunnel-only       # Start Cloudflare tunnel only
./run.sh ssh-tunnel        # Start localhost.run SSH tunnel only
./run.sh api-docs          # Show API documentation
./run.sh status            # Check system status
./run.sh stop              # Stop all services
./run.sh switch-model      # Switch between full and nf4 models
```

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | - | HuggingFace API token |
| `PERSONAPLEX_PORT` | 8998 | Server port |
| `PERSONAPLEX_TUNNEL_NAME` | personaplex-voice | Cloudflare tunnel name |
| `PERSONAPLEX_MODEL_TYPE` | full | Model type: `full`, `nf4`, or `turbo2bit` |

## 🎯 Model Options

### Full Model (bf16)
- **Repository**: `nvidia/personaplex-7b-v1`
- **Size**: ~15.59 GB
- **Best for**: Desktop GPUs with 16GB+ VRAM
- **Command**: `./run.sh start`

### NF4 Quantized Model (INT4)
- **Repository**: `cudabenchmarktest/personaplex-7b-nf4`
- **Size**: ~4.14 GB (3.8x smaller)
- **Best for**: Edge devices, Jetson AGX Orin, 8GB VRAM GPUs
- **Command**: `./run.sh start-nf4`

### 2-bit TurboQuant Model (NF2+WHT)
- **Repository**: `personaplex-7b-turbo2bit`
- **Size**: ~1.86 GB (8.4x smaller than full)
- **Dequantized Size**: ~16.6 GB (matches full bf16 on GPU)
- **Best for**: Fast downloads, edge deployment with 16GB+ VRAM
- **Command**: `./run.sh start-turbo2bit`
- **Features**: 
  - Automatic dequantization on load via Walsh-Hadamard Transform
  - NF2 centroids for optimal 2-bit quantization
  - Verified to match full bf16 model performance

### Switch Between Models
```bash
# Toggle between models
./run.sh switch-model

# Or set environment variable
export PERSONAPLEX_MODEL_TYPE=nf4  # or 'full' or 'turbo2bit'
./run.sh start
```

## 🌐 Tunneling Options

### Cloudflare Tunnel
- **URL Format**: `https://xxxxx.trycloudflare.com`
- **Pros**: Fast, reliable, no account needed
- **Command**: `./run.sh tunnel-only`

### localhost.run SSH Tunnel
- **URL Format**: `https://xxxxx.lhr.life`
- **Pros**: Alternative tunneling, SSH-based
- **Command**: `./run.sh ssh-tunnel`

### Use Both
```bash
# Start server with both tunnels
./run.sh start
```

## 📚 API Endpoints

Once running, access these endpoints via the tunnel URL:

### Clone Voice
```bash
POST /api/clone
Content-Type: application/json

{
  "voice_name": "my_voice",
  "audio_file": "<base64_encoded_audio>"
}
```

### List Voices
```bash
GET /api/voices
```

### Generate Speech
```bash
POST /api/generate
Content-Type: application/json

{
  "text": "Hello, world!",
  "voice_name": "my_voice"
}
```

### Check Status
```bash
GET /api/status
```

## 🎯 First-Time Setup

If you don't have models downloaded yet:

1. **Get HuggingFace Token**: https://huggingface.co/settings/tokens
2. **Accept License**: https://huggingface.co/nvidia/personaplex-7b-v1
3. **Run with token**:
   ```bash
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ./run.sh start
   ```

The script will automatically download models on first run.

### Using NF4 Quantized Model (Recommended for Edge)

```bash
# Set model type to nf4
export PERSONAPLEX_MODEL_TYPE=nf4
export HF_TOKEN=your_token
./run.sh start

# Or use the shortcut command
./run.sh start-nf4
```

## 🛠️ Troubleshooting

### Models not found
```bash
# Check if models exist
find models -name "*.safetensors"

# If missing, set HF_TOKEN and restart
export HF_TOKEN=your_token
./run.sh start
```

### Port already in use
```bash
# Change port
export PERSONAPLEX_PORT=9999
./run.sh start
```

### Cloudflare tunnel issues
```bash
# Check if cloudflared is installed
cloudflared --version

# Install if needed
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.tar.gz | tar xz
sudo mv cloudflared /usr/local/bin/
```

### SSH tunnel issues
```bash
# Check if ssh is installed
ssh -V

# Install if needed (Ubuntu/Debian)
sudo apt install openssh-client

# Install if needed (Fedora)
sudo dnf install openssh-clients
```

### Check system status
```bash
./run.sh status
```

## 📁 Project Structure

```
PersonaPlex/
├── run.sh                 # Main automation script
├── models/                # Downloaded models
│   ├── nvidia/personaplex-7b-v1/     # Full bf16 model
│   └── cudabenchmarktest/personaplex-7b-nf4/  # NF4 quantized model
├── personaplex-setup/     # PersonaPlex core (submodule)
│   ├── moshi/            # Moshi server
│   ├── client/           # Frontend
│   └── start_server.sh   # Server startup
├── .gitignore
└── README.md
```

## 🔗 Resources

- **Original Repo**: https://github.com/NVIDIA/PersonaPlex
- **HuggingFace Models**:
  - Full: https://huggingface.co/nvidia/personaplex-7b-v1
  - NF4: https://huggingface.co/cudabenchmarktest/personaplex-7b-nf4
- **Research Paper**: https://research.nvidia.com/labs/adlr/files/personaplex/personaplex_preprint.pdf
- **Demo**: https://research.nvidia.com/labs/adlr/personaplex/
- **localhost.run**: https://localhost.run

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Issues and pull requests welcome! This is a community automation wrapper around NVIDIA's PersonaPlex.

---

**Made with ❤️ for the voice cloning community**
