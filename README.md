# PersonaPlex Jetson NF4 Deployment

This fork is set up for running a PersonaPlex NF4 server on Jetson hardware.
The local deployment path uses:

- Jetson AGX Orin / L4T R36.3 / CUDA 12.2 / Python 3.10
- NVIDIA Jetson CUDA PyTorch in `.venv-jetson`
- Public Hugging Face model artifacts from `cudabenchmarktest/personaplex-7b-nf4`
- The fork runtime expected under `personaplex-setup/moshi`

Detailed deployment notes are in
[`docs/JETSON_NF4_DEPLOYMENT.md`](docs/JETSON_NF4_DEPLOYMENT.md).

## One-Command Deployment

Run the complete local deployment and public demo tunnel:

```bash
./scripts/deploy_nf4_cloudflared.sh
```

Prerequisite: `personaplex-setup/moshi` must contain this fork's runtime source.

That command:

1. installs or verifies the Jetson Python environment;
2. downloads/verifies the NF4 model artifacts;
3. starts PersonaPlex locally at `http://localhost:8998`;
4. installs `cloudflared` into `.cache/bin` if it is not already available;
5. runs `cloudflared tunnel --url http://localhost:8998`;
6. prints a temporary `https://*.trycloudflare.com` URL for the demo.

Leave the command running while the colleague tests the endpoint. Stop the
tunnel with `Ctrl+C`; stop a background PersonaPlex server with:

```bash
kill "$(cat server_nf4.pid)"
```

Useful overrides:

```bash
PERSONAPLEX_PORT=8999 ./scripts/deploy_nf4_cloudflared.sh
PERSONAPLEX_SKIP_SETUP=1 ./scripts/deploy_nf4_cloudflared.sh
CLOUDFLARED_BIN=/usr/bin/cloudflared ./scripts/deploy_nf4_cloudflared.sh
```

## Repository Layout

- `scripts/setup_jetson_nf4.sh`: creates the Jetson venv, installs CUDA PyTorch,
  installs Python runtime dependencies, and downloads NF4 model artifacts.
- `scripts/start_nf4_server.sh`: starts the local PersonaPlex NF4 server.
- `scripts/deploy_nf4_cloudflared.sh`: runs setup, starts the server, waits for
  the local endpoint, and opens a Cloudflare quick tunnel.
- `personaplex_control/`: control protocol and server adapter code.
- `models/cudabenchmarktest/personaplex-7b-nf4/`: local NF4 model artifact
  directory.
- `docs/`: setup notes, architecture notes, and training/evaluation references.

## Current Host State

Verified on this machine:

- L4T: `R36.3.0`
- CUDA toolkit: `12.2`
- Python: `3.10.12`
- Architecture: `aarch64`
- `.venv-jetson` torch: `2.4.0a0+07cecf4168.nv24.05`
- `torch.cuda.is_available()`: `True`
- CUDA device: `Orin`
- NF4 model directory size: about `4.6G`

The setup script reported `libomp-dev` as missing, but `sphn` still built
successfully and CUDA verification passed. If you want the host packages fully
aligned, run:

```bash
sudo apt-get update
sudo apt-get install -y libomp-dev
```

## Required Runtime Source

This checkout tracks `personaplex-setup` as a gitlink at:

```text
ca864aa923e96d4dd08c5d9895638c94c7df2802
```

The repository currently does not include a `.gitmodules` URL for that gitlink.
Restore this fork's runtime source so that this path exists before starting the
server:

```text
personaplex-setup/moshi
```

The start script intentionally does not fetch replacement runtime source. It
uses the fork runtime present in this working tree.

## Setup

Run the Jetson setup script:

```bash
./scripts/setup_jetson_nf4.sh
```

The script performs these steps:

1. validates Jetson `aarch64`, Python 3.10, and L4T;
2. creates `.venv-jetson`;
3. installs NVIDIA's JetPack 6.0 PyTorch wheel;
4. installs PersonaPlex runtime Python dependencies;
5. downloads the public NF4 model files;
6. verifies CUDA from inside `.venv-jetson`.

The model files are downloaded from:

```text
cudabenchmarktest/personaplex-7b-nf4
```

Expected files:

```text
model-nf4.safetensors
tokenizer-e351c8d8-checkpoint125.safetensors
tokenizer_spm_32k_3.model
voices/OverBarn.pt
config.json
dequant-loader.py
linear2bit.py
clone-voice.py
README.md
```

For a different JetPack/CUDA release, override the PyTorch wheel:

```bash
PERSONAPLEX_TORCH_URL=<wheel-url> \
PERSONAPLEX_TORCH_WHEEL_NAME=<wheel-file-name> \
PERSONAPLEX_TORCH_WHEEL_BYTES=<expected-byte-count> \
./scripts/setup_jetson_nf4.sh
```

## Start The Server

Foreground:

```bash
./scripts/start_nf4_server.sh
```

Background:

```bash
PERSONAPLEX_BACKGROUND=1 ./scripts/start_nf4_server.sh
```

Defaults:

- host: `0.0.0.0`
- port: `8998`
- URL: `http://localhost:8998`
- log: `server_nf4.log`
- pid file: `server_nf4.pid`

Useful overrides:

```bash
PERSONAPLEX_PORT=8999 ./scripts/start_nf4_server.sh
PERSONAPLEX_CPU_MIMI=1 ./scripts/start_nf4_server.sh
PERSONAPLEX_EXTRA_ARGS="--ssl /tmp/personaplex-ssl" ./scripts/start_nf4_server.sh
```

Stop a background server:

```bash
kill "$(cat server_nf4.pid)"
```

## Verify

Check the Jetson PyTorch install:

```bash
.venv-jetson/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0))
PY
```

Check the endpoint after the runtime source is restored and the server is
running:

```bash
curl -I http://localhost:8998
```

## Cloudflare Tunnel

Cloudflare Tunnel is only needed when the PersonaPlex endpoint must be reachable
outside the Jetson's local network. For local testing on the same machine or
LAN, use `http://localhost:8998` or the Jetson's LAN IP directly.

### Native Install On Ubuntu 22.04

Install `cloudflared` from Cloudflare's apt repository:

```bash
sudo mkdir -p --mode=0755 /usr/share/keyrings
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg \
  | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared jammy main' \
  | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt-get update
sudo apt-get install -y cloudflared
cloudflared --version
```

Cloudflare package docs:

- https://pkg.cloudflare.com/
- https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/

### Quick Demo Tunnel

Use a temporary public URL for a demo:

```bash
cloudflared tunnel --url http://localhost:8998
```

Copy the generated `https://*.trycloudflare.com` URL and share it with the
colleague testing the server.

### Named Tunnel

Use a named tunnel when you need a stable hostname:

```bash
cloudflared tunnel login
cloudflared tunnel create personaplex
cloudflared tunnel route dns personaplex personaplex.example.com
```

Create `~/.cloudflared/personaplex.yml`:

```yaml
tunnel: <tunnel-uuid>
credentials-file: /home/egg/.cloudflared/<tunnel-uuid>.json

ingress:
  - hostname: personaplex.example.com
    service: http://localhost:8998
  - service: http_status:404
```

Run it manually:

```bash
cloudflared tunnel --config ~/.cloudflared/personaplex.yml run personaplex
```

Install it as a Linux service after the manual run works:

```bash
sudo cloudflared service install
sudo systemctl enable --now cloudflared
sudo systemctl status cloudflared
```

Cloudflare service docs:

- https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/local-management/as-a-service/linux/

## Useful Commands

Show current repo state:

```bash
git status --short --branch
```

Show downloaded model files:

```bash
find models/cudabenchmarktest/personaplex-7b-nf4 -maxdepth 2 -type f -printf '%P %s\n' | sort
```

Show disk use:

```bash
du -sh .venv-jetson .cache/jetson-wheels models/cudabenchmarktest/personaplex-7b-nf4
```

## License

Code in this repository is MIT unless a file states otherwise. PersonaPlex
runtime code, model weights, and voice assets remain under their upstream
licenses.
