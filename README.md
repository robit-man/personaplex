# PersonaPlex Voice Cloning System

> **Single Point of Entry** - Automated setup and deployment for NVIDIA's PersonaPlex voice cloning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/🤗-Weights-yellow)](https://huggingface.co/nvidia/personaplex-7b-v1)

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** with venv
- **Node.js 18+** (for frontend)
- **Cloudflare CLI** (`cloudflared`) for tunnel
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
# Start everything (server + tunnel)
export HF_TOKEN=your_huggingface_token  # Optional if models already downloaded
./run.sh start
```

**That's it!** The script handles:
- ✅ PersonaPlex server startup
- ✅ Cloudflare tunnel for public access
- ✅ Auto-detection of downloaded models
- ✅ API documentation display

## 📋 Commands

```bash
./run.sh start          # Start server + tunnel (default)
./run.sh server-only    # Start server only (localhost)
./run.sh tunnel-only    # Start tunnel only
./run.sh status         # Check system status
./run.sh stop           # Stop all services
./run.sh api-docs       # Show API documentation
```

## 🔧 Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | - | HuggingFace API token |
| `PERSONAPLEX_PORT` | 8998 | Server port |
| `PERSONAPLEX_TUNNEL_NAME` | personaplex-voice | Cloudflare tunnel name |

## 📚 API Endpoints

Once running, access these endpoints via the Cloudflare tunnel URL:

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

## 🛠️ Troubleshooting

### Models not found
```bash
# Check if models exist
find personaplex-setup/models -name "*.safetensors"

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

## 📁 Project Structure

```
PersonaPlex/
├── run.sh                 # Main automation script
├── personaplex-setup/     # PersonaPlex core (submodule)
│   ├── moshi/            # Moshi server
│   ├── client/           # Frontend
│   └── start_server.sh   # Server startup
├── .gitignore
└── README.md
```

## 🔗 Resources

- **Original Repo**: https://github.com/NVIDIA/PersonaPlex
- **HuggingFace Models**: https://huggingface.co/nvidia/personaplex-7b-v1
- **Research Paper**: https://research.nvidia.com/labs/adlr/files/personaplex/personaplex_preprint.pdf
- **Demo**: https://research.nvidia.com/labs/adlr/personaplex/

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Issues and pull requests welcome! This is a community automation wrapper around NVIDIA's PersonaPlex.

---

**Made with ❤️ for the voice cloning community**

