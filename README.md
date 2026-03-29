# PersonaPlex

Full-duplex voice AI with hybrid LLM reasoning, knowledge distillation, and a custom dark UI.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![npm](https://img.shields.io/badge/npm-open--agents--ai-blue)](https://www.npmjs.com/package/open-agents-ai)
[![HuggingFace](https://img.shields.io/badge/🤗-NF4_Distilled-green)](https://huggingface.co/cudabenchmarktest/personaplex-7b-nf4-distilled)
[![HuggingFace](https://img.shields.io/badge/🤗-Hybrid-yellow)](https://huggingface.co/cudabenchmarktest/personaplex-7b-hybrid)

## What Is This

A heavily modified fork of [NVIDIA PersonaPlex](https://github.com/NVIDIA/personaplex) with:

**Hybrid Architecture** — PersonaPlex handles real-time voice I/O. A local LLM (Qwen via Ollama) handles reasoning. PersonaPlex's built-in call-center behavior is suppressed; users see the LLM's natural responses in the text stream while hearing PersonaPlex's voice.

**Knowledge Distillation** — NF4 weights distilled from the bf16 teacher (5 epochs, 3000 samples). Token match improved from 75% to 90%.

**Custom Frontend** — dark grey (#1a1a1a) background with amber (#ffae00) accents. Ultra-minimal, mobile-friendly. No call-center preset buttons.

## Interface

The frontend provides:

- **Voice selector** — custom cloned voices at top, NVIDIA defaults below
- **Ollama prompt expander** — enter a snippet (e.g., "grumpy pirate"), select any local Ollama model, click Expand to generate a full persona prompt
- **Settings panel** (collapsible):
  - Weight tier selector (bf16 / NF4 / NF4 distilled) with hot-restart
  - Ollama model picker for prompt expansion
  - Voice cloning: press-and-hold recording or file upload
  - LuxTTS synthetic dataset generation toggle
  - Pipeline progress bar during cloning
- **Conversation view** — full-height, non-scrollable page with internally scrollable text area
- **Transparent audio visualizers** — amber accent, no black backgrounds
- **GPU stats** — live VRAM, utilization, temperature in header
- **Live tier badge** — shows currently loaded model with green dot when connected

Mobile: viewport locked (no pinch zoom), 16px font on inputs to prevent iOS focus zoom.

## Quick Start

```bash
git clone https://github.com/robit-man/personaplex.git
cd personaplex

# Recommended: NF4 with hybrid reasoning
export HYBRID_LLM_MODEL=qwen3.5:27b  # or whatever Ollama model you prefer
./run.sh start-nf4

# Or use start_server.sh directly
cd personaplex-setup
./start_server.sh bf16              # Full quality (~19GB VRAM)
./start_server.sh 2bit             # 2-bit download, dequant at load
./start_server.sh native-2bit      # Native 2-bit on GPU (~10GB peak)
./start_server.sh cpu-offload      # Split model across GPU+CPU
```

Add `--hybrid` to any mode to enable LLM reasoning:
```bash
python -m moshi.server --moshi-weight model.safetensors --device cuda --hybrid
```

## Hybrid Architecture

```
User speaks → PersonaPlex (audio in/out, full-duplex)
                         ↓
            PersonaPlex generates text tokens (suppressed)
                         ↓
            Hybrid agent intercepts → Qwen/Ollama generates response
                         ↓
            LLM response displayed as text (user reads Qwen's words)
```

The user **hears** PersonaPlex's voice. The user **reads** Qwen's intelligent response. Dynamic model escalation: 4B for quick exchanges, 27B for conversation, 122B for complex analysis.

## Distillation Results

| Model | Token Match vs bf16 | Status |
|-------|---------------------|--------|
| bf16 (teacher) | 100% | Reference |
| NF4 raw | 75% | Coherent |
| **NF4 distilled** | **90%** | Recommended |
| TurboQuant 2-bit | 4% | Broken (incoherent) |

Training: 5 epochs, 3000 samples, 73 min on A100. Loss: 0.58 → 0.07.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | WebSocket | Full-duplex voice conversation |
| `/api/voices` | GET | Available voices (custom first) |
| `/api/clone` | POST | Upload WAV → clone voice embedding |
| `/api/clone-pipeline` | POST | Record → LuxTTS synth → PersonaPlex clone |
| `/api/clone-pipeline/{id}` | GET | Poll pipeline progress |
| `/api/status` | GET | Tier, device, GPU stats |
| `/api/restart` | POST | Hot-restart with different weight tier |
| `/api/hybrid` | GET | Hybrid mode status |
| `/api/ollama/tags` | GET | Proxy: Ollama models (no CORS) |
| `/api/ollama/generate` | POST | Proxy: Ollama generation |

## Voice Cloning

1. **Direct clone** — upload or press-and-hold to record audio → PersonaPlex extracts voice embedding (.pt)
2. **LuxTTS pipeline** — upload short sample → LuxTTS generates 5 synthetic sentences → concatenate → PersonaPlex clones from the richer dataset

## HuggingFace Models

| Repo | Size | Quality | Use Case |
|------|------|---------|----------|
| [personaplex-7b-nf4-distilled](https://huggingface.co/cudabenchmarktest/personaplex-7b-nf4-distilled) | 16.7 GB | 90% match | **Recommended** |
| [personaplex-7b-hybrid](https://huggingface.co/cudabenchmarktest/personaplex-7b-hybrid) | Scripts only | — | Hybrid agent code |
| [personaplex-7b-nf4](https://huggingface.co/cudabenchmarktest/personaplex-7b-nf4) | 4.1 GB | 75% match | Smaller download |
| [personaplex-7b-turbo2bit](https://huggingface.co/cudabenchmarktest/personaplex-7b-turbo2bit) | 2.1 GB | 4% match | Research only |
| [nvidia/personaplex-7b-v1](https://huggingface.co/nvidia/personaplex-7b-v1) | 15.6 GB | 100% | Requires HF token |

## Project Structure

```
personaplex/
├── run.sh                          # Main launcher
├── personaplex-setup/
│   ├── moshi/moshi/
│   │   ├── server.py               # WebSocket + REST + hybrid agent
│   │   ├── hybrid_agent.py         # LLM reasoning layer
│   │   ├── models/loaders.py       # NF4/2-bit dequant support
│   │   └── modules/linear2bit.py   # Native 2-bit Linear module
│   ├── client/                     # React frontend (Tailwind + Vite)
│   ├── voices/personaplex/         # Voice cloning tools
│   └── start_server.sh             # Multi-mode launcher
├── hybrid/                         # Training + eval
│   ├── sft_v2.py                   # SFT with forward_train
│   ├── calibrated_quant.py         # GPTQ-style quantization
│   ├── prompts.json                # Anti-call-center prompts
│   └── combined_training_data.json # 32K training pairs
├── distillation/
│   ├── distill_v2.py               # Knowledge distillation script
│   └── dataset_v2/                 # Teacher logit dataset
└── eval_quant.py                   # Quality evaluation harness
```

## License

MIT — same as base PersonaPlex.

Built by [open-agents-ai](https://www.npmjs.com/package/open-agents-ai) on [NVIDIA PersonaPlex](https://research.nvidia.com/labs/adlr/personaplex/).
