# PersonaPlex

Full-duplex voice AI with native 2-bit quantization, voice cloning, and a minimal dark UI.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![HuggingFace](https://img.shields.io/badge/🤗-TurboQuant_2bit-yellow)](https://huggingface.co/cudabenchmarktest/personaplex-7b-turbo2bit)
[![HuggingFace](https://img.shields.io/badge/🤗-NF4-blue)](https://huggingface.co/cudabenchmarktest/personaplex-7b-nf4)

## What's Different

This is a heavily modified fork of [NVIDIA PersonaPlex](https://github.com/NVIDIA/personaplex) with:

- **Native 2-bit inference** — `Linear2bit` module keeps NF2+WHT packed weights on GPU (~10GB peak vs 19GB bf16)
- **Custom frontend** — dark grey (#1a1a1a) + amber (#ffae00) accent, mobile-friendly, no call-center presets
- **Ollama prompt expander** — type a snippet, select any local Ollama model, expand into a full persona prompt
- **Voice cloning** — press-and-hold recording + file upload → clone pipeline with optional LuxTTS synthetic generation
- **Hot-restart weight tiers** — switch between bf16/nf4/turbo2bit from the UI, server restarts without killing the tunnel
- **CPU Mimi codec** — offload audio encoder/decoder to CPU, saves ~840MB VRAM
- **Dynamic voice list** — `/api/voices` endpoint, custom voices appear first
- **Server-side Ollama proxy** — no CORS issues on tunneled connections

## VRAM Comparison

| Configuration | Peak VRAM | Download |
|--------------|-----------|----------|
| BF16 (original) | ~19 GB | 15.6 GB |
| NF4 (INT4) | ~19 GB | 4.1 GB |
| **TurboQuant 2-bit (native)** | **~10 GB** | **2.1 GB** |
| + CPU Mimi | saves 840 MB | — |

## Quick Start

```bash
git clone https://github.com/robit-man/personaplex.git
cd personaplex

# Full bf16 model
export HF_TOKEN=your_token
./run.sh start

# Native 2-bit (~10GB VRAM, no HF token needed)
./run.sh start-turbo2bit

# Or use start_server.sh directly
cd personaplex-setup
./start_server.sh native-2bit
```

## Server Modes

```bash
./start_server.sh bf16          # Full quality (~19GB VRAM)
./start_server.sh 2bit          # 2-bit download, dequant at load (~19GB)
./start_server.sh native-2bit   # Native 2-bit on GPU (~10GB peak)
./start_server.sh cpu-offload   # Split model across GPU+CPU
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | WebSocket | Full-duplex voice conversation |
| `/api/voices` | GET | List available voices (custom first) |
| `/api/clone` | POST | Upload WAV → clone voice embedding |
| `/api/clone-pipeline` | POST | Full pipeline: record → LuxTTS synth → PersonaPlex clone |
| `/api/clone-pipeline/{id}` | GET | Poll clone pipeline progress |
| `/api/status` | GET | Current tier, device, port |
| `/api/restart` | POST | Hot-restart with different weight tier |
| `/api/ollama/tags` | GET | Proxy: list Ollama models |
| `/api/ollama/generate` | POST | Proxy: generate with Ollama |

## Frontend

Minimal dark UI with amber accents:
- Custom voices at top of dropdown
- Ollama-powered prompt expander (select any local model)
- Settings panel: weight tier selector, Ollama model picker, voice cloning
- Press-and-hold voice recording for cloning
- LuxTTS synthetic data generation toggle
- Pipeline progress bar
- Transparent canvas audio visualizers
- Mobile-friendly (no pinch zoom, 16px inputs)

## Voice Cloning

Two paths:
1. **Direct clone** — upload or record audio → PersonaPlex extracts voice embedding
2. **LuxTTS pipeline** — upload short sample → LuxTTS generates synthetic training data → PersonaPlex clones from the richer dataset

## Quantized Model Repos

- **TurboQuant 2-bit**: [cudabenchmarktest/personaplex-7b-turbo2bit](https://huggingface.co/cudabenchmarktest/personaplex-7b-turbo2bit) — 2.1 GB, native inference via `linear2bit.py`
- **NF4 INT4**: [cudabenchmarktest/personaplex-7b-nf4](https://huggingface.co/cudabenchmarktest/personaplex-7b-nf4) — 4.1 GB
- **Original**: [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1) — 15.6 GB (requires HF token)

## Project Structure

```
personaplex/
├── run.sh                      # Main launcher
├── personaplex-setup/          # Core server + frontend
│   ├── moshi/moshi/
│   │   ├── server.py           # WebSocket server + REST API
│   │   ├── models/loaders.py   # 2-bit dequant in model loading
│   │   └── modules/
│   │       └── linear2bit.py   # Native 2-bit Linear module
│   ├── client/                 # React frontend (Tailwind + Vite)
│   ├── voices/personaplex/     # Voice cloning tools
│   │   ├── clone-voice.py
│   │   ├── dequant-loader.py
│   │   └── quantize-weights.py
│   └── start_server.sh         # 4-mode server launcher
└── models/                     # Local model checkpoints
```

## License

MIT — same as base PersonaPlex model.

Built on [NVIDIA PersonaPlex](https://research.nvidia.com/labs/adlr/personaplex/) by the [open-agents](https://github.com/open-agents-ai) team.
