# Jetson NF4 deployment

This deployment path is for the local fork in this repository and the public
`cudabenchmarktest/personaplex-7b-nf4` Hugging Face model artifacts. It does
not fetch NVIDIA/personaplex source. The `personaplex-setup` path is a gitlink
in this fork; when that runtime content is absent, the scripts use the packaged
`moshi` runtime and a cached bf16 conversion of the NF4 weights.

## Host detected on 2026-07-16

- Jetson L4T: `R36.3.0`
- CUDA toolkit: `12.2`
- Python: `3.10.12`
- Architecture: `aarch64`
- Before setup, system torch was `2.10.0+cpu`, with
  `torch.cuda.is_available() == False`
- After setup, `.venv-jetson` torch is
  `2.4.0a0+07cecf4168.nv24.05`, with CUDA `12.2` and
  `torch.cuda.is_available() == True`
- NF4 artifacts downloaded to `models/cudabenchmarktest/personaplex-7b-nf4`
- Packaged fallback runtime installed as `moshi==0.2.13`

NVIDIA's Jetson PyTorch docs require JetPack plus PyTorch system packages, and
their install flow uses Jetson-specific `linux_aarch64` wheels. NVIDIA forum
guidance for JetPack 6.0 names the `v60` PyTorch 2.4 wheel for Python 3.10 on
Jetson. The setup script pins that wheel because this repo's Moshi runtime
requires `torch>=2.2,<2.5`.

Sources:

- NVIDIA install docs: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
- NVIDIA Jetson wheel guidance: https://forums.developer.nvidia.com/t/install-pytorch-with-cuda-on-jetson-orin-nano-devloper-kit/297427
- Public NF4 model: https://huggingface.co/cudabenchmarktest/personaplex-7b-nf4

## Setup

Run:

```bash
./scripts/setup_jetson_nf4.sh
```

The script:

- validates Jetson `aarch64`, Python 3.10, and L4T;
- installs required Ubuntu packages when missing;
- creates `.venv-jetson`;
- installs NVIDIA's JetPack 6.0 PyTorch wheel from `developer.download.nvidia.com`;
- installs runtime Python dependencies without installing any alternate
  PersonaPlex source;
- installs the packaged `moshi` fallback runtime;
- downloads the NF4 model, Mimi tokenizer, text tokenizer, helper files, and
  `voices/OverBarn.pt`;
- verifies `torch.cuda.is_available()`.

On this host, sudo is not available non-interactively. The script reported
`libomp-dev` as the only missing apt package and continued. `sphn` built
successfully from source anyway, and the final CUDA verification passed.

If the host is a different JetPack/CUDA release, set `PERSONAPLEX_TORCH_URL`
and `PERSONAPLEX_TORCH_WHEEL_NAME` before running the script.

## Start

For the managed local server, Cloudflare quick tunnel, TUI monitor, and cleanup
path, run:

```bash
./scripts/deploy_nf4_cloudflared.sh
```

The deploy script keeps the monitor in the foreground. On an interactive
terminal it shows local and tunnel health, server/tunnel PIDs, recent logs, and
Jetson telemetry through `jtop` when available. `Ctrl+C` stops the Cloudflare
tunnel and the PersonaPlex server started by the deploy script, releasing the
CUDA memory held by the server process. Set `PERSONAPLEX_TUI=0` for log-only
monitoring, or `PERSONAPLEX_CLEANUP_ON_EXIT=0` to leave the server running.

Start in the foreground:

```bash
./scripts/start_nf4_server.sh
```

Runtime selection:

- If `personaplex-setup/moshi` contains the fork runtime, it is used directly
  with `model-nf4.safetensors`.
- If that path is absent, the script uses packaged `moshi`, dequantizes
  `model-nf4.safetensors`, and writes
  `.cache/personaplex/model-bf16.safetensors`.

Or start in the background:

```bash
PERSONAPLEX_BACKGROUND=1 ./scripts/start_nf4_server.sh
```

Defaults:

- URL: `http://0.0.0.0:8998`
- model dir: `models/cudabenchmarktest/personaplex-7b-nf4`
- venv: `.venv-jetson`
- log: `server_nf4.log`
- pid: `server_nf4.pid`

Useful overrides:

```bash
PERSONAPLEX_PORT=8999 ./scripts/start_nf4_server.sh
PERSONAPLEX_CPU_MIMI=1 ./scripts/start_nf4_server.sh
PERSONAPLEX_STATIC=none ./scripts/start_nf4_server.sh
PERSONAPLEX_BF16_MODEL_PATH=/data/personaplex/model-bf16.safetensors ./scripts/start_nf4_server.sh
PERSONAPLEX_EXTRA_ARGS="--ssl /tmp/personaplex-ssl" ./scripts/start_nf4_server.sh
```

## Runtime boundary

The current checkout tracks `personaplex-setup` as a gitlink at commit
`ca864aa923e96d4dd08c5d9895638c94c7df2802`, but this repository does not
include a `.gitmodules` URL. Because the deployment must use this fork, the
scripts refuse to clone a replacement runtime from another repository. The
fallback path uses the PyPI `moshi` server plus the model repository's
`dequant-loader.py` helper instead.

The downloaded `personaplex-7b-nf4-distilled` repository is not the server
weight source. It contains `student_best.pt` and training metadata. The runnable
NF4 server artifacts are in `cudabenchmarktest/personaplex-7b-nf4`.
