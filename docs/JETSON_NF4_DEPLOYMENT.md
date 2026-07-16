# Jetson NF4 deployment

This deployment path is for the local fork in this repository and the public
`cudabenchmarktest/personaplex-7b-nf4` Hugging Face model artifacts. It does
not fetch NVIDIA/personaplex source. The `personaplex-setup` path is a gitlink
in this fork; restore that fork runtime content before starting the server.

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
- downloads the NF4 model, Mimi tokenizer, text tokenizer, helper files, and
  `voices/OverBarn.pt`;
- verifies `torch.cuda.is_available()`.

On this host, sudo is not available non-interactively. The script reported
`libomp-dev` as the only missing apt package and continued. `sphn` built
successfully from source anyway, and the final CUDA verification passed.

If the host is a different JetPack/CUDA release, set `PERSONAPLEX_TORCH_URL`
and `PERSONAPLEX_TORCH_WHEEL_NAME` before running the script.

## Start

After the fork runtime gitlink is restored under `personaplex-setup/moshi`,
start in the foreground:

```bash
./scripts/start_nf4_server.sh
```

Current start result before restoring the gitlink:

```text
[personaplex-server] ERROR: fork runtime missing at /home/egg/Documents/personaplex/personaplex-setup/moshi. Restore this fork's personaplex-setup gitlink/source; this script will not fetch upstream source.
```

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
PERSONAPLEX_EXTRA_ARGS="--ssl /tmp/personaplex-ssl" ./scripts/start_nf4_server.sh
```

## Runtime boundary

The current checkout tracks `personaplex-setup` as a gitlink at commit
`ca864aa923e96d4dd08c5d9895638c94c7df2802`, but this repository does not
include a `.gitmodules` URL. Because the deployment must use this fork, the
scripts refuse to clone a replacement runtime from another repository. Restore
the matching fork runtime content at `personaplex-setup/moshi` before starting.

The downloaded `personaplex-7b-nf4-distilled` repository is not the server
weight source. It contains `student_best.pt` and training metadata. The runnable
NF4 server artifacts are in `cudabenchmarktest/personaplex-7b-nf4`.
