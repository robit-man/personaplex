#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PERSONAPLEX_VENV:-$ROOT_DIR/.venv-jetson}"
WHEEL_DIR="${PERSONAPLEX_WHEEL_DIR:-$ROOT_DIR/.cache/jetson-wheels}"
MODEL_REPO="${PERSONAPLEX_MODEL_REPO:-cudabenchmarktest/personaplex-7b-nf4}"
MODEL_DIR="${PERSONAPLEX_MODEL_DIR:-$ROOT_DIR/models/cudabenchmarktest/personaplex-7b-nf4}"
PYTHON_BIN="${PYTHON:-python3}"

TORCH_WHEEL_NAME="${PERSONAPLEX_TORCH_WHEEL_NAME:-torch-2.4.0a0+07cecf4168.nv24.05.14710581-cp310-cp310-linux_aarch64.whl}"
TORCH_URL="${PERSONAPLEX_TORCH_URL:-https://developer.download.nvidia.com/compute/redist/jp/v60/pytorch/$TORCH_WHEEL_NAME}"
TORCH_WHEEL_BYTES="${PERSONAPLEX_TORCH_WHEEL_BYTES:-1047045276}"

required_files=(
  "model-nf4.safetensors"
  "tokenizer-e351c8d8-checkpoint125.safetensors"
  "tokenizer_spm_32k_3.model"
  "voices/OverBarn.pt"
  "config.json"
  "dequant-loader.py"
  "linear2bit.py"
  "clone-voice.py"
  "README.md"
)

log() {
  printf '[personaplex-jetson] %s\n' "$*"
}

fail() {
  printf '[personaplex-jetson] ERROR: %s\n' "$*" >&2
  exit 1
}

need_command() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

detect_platform() {
  local arch
  arch="$(uname -m)"
  [[ "$arch" == "aarch64" ]] || fail "expected Jetson aarch64, got $arch"

  local py_tag
  py_tag="$("$PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  [[ "$py_tag" == "3.10" ]] || fail "expected Python 3.10 for NVIDIA cp310 Jetson wheel, got $py_tag"

  if [[ -r /etc/nv_tegra_release ]]; then
    local l4t
    l4t="$(head -1 /etc/nv_tegra_release)"
    log "detected $l4t"
    if [[ "$l4t" != *"R36 (release), REVISION: 3."* ]]; then
      log "warning: this script is pinned for JetPack 6.0 / L4T R36.3; override PERSONAPLEX_TORCH_URL if this host differs"
    fi
  else
    fail "/etc/nv_tegra_release not found; this setup is only for NVIDIA Jetson"
  fi

  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | tail -1
  else
    log "warning: nvcc not found in PATH; CUDA runtime may still be installed"
  fi
}

install_system_packages() {
  if [[ "${PERSONAPLEX_SKIP_APT:-0}" == "1" ]]; then
    log "skipping apt package check because PERSONAPLEX_SKIP_APT=1"
    return
  fi

  local packages=(
    python3-venv
    python3-pip
    libopenblas-dev
    libopenmpi-dev
    libomp-dev
    libopus-dev
    pkg-config
    build-essential
  )
  local missing=()
  local pkg
  for pkg in "${packages[@]}"; do
    if ! dpkg-query -W "$pkg" >/dev/null 2>&1; then
      missing+=("$pkg")
    fi
  done

  if (( ${#missing[@]} == 0 )); then
    log "system packages already present"
    return
  fi

  need_command sudo
  if ! sudo -n true >/dev/null 2>&1; then
    log "missing system packages but sudo is not available non-interactively: ${missing[*]}"
    log "manual command: sudo apt-get update && sudo apt-get install -y ${missing[*]}"
    log "continuing because the required CUDA/cuDNN/OpenBLAS/Opus packages may already be present"
    return
  fi
  log "installing system packages: ${missing[*]}"
  sudo apt-get update
  sudo apt-get install -y "${missing[@]}"
}

create_venv() {
  log "creating venv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
}

install_torch() {
  if "$VENV_DIR/bin/python" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() and str(torch.__version__).startswith("2.4.0a0+07cecf4168") else 1)
PY
  then
    log "existing Jetson CUDA PyTorch verified"
    return
  fi

  mkdir -p "$WHEEL_DIR"
  local wheel_path="$WHEEL_DIR/$TORCH_WHEEL_NAME"
  local wheel_bytes=0
  if [[ -f "$wheel_path" ]]; then
    wheel_bytes="$(stat -c '%s' "$wheel_path")"
  fi
  if [[ "$wheel_bytes" != "$TORCH_WHEEL_BYTES" ]]; then
    log "downloading NVIDIA Jetson PyTorch wheel"
    need_command curl
    curl -L --fail --continue-at - --retry 8 --retry-all-errors --retry-delay 2 --speed-limit 131072 --speed-time 60 "$TORCH_URL" -o "$wheel_path"
  else
    log "using cached PyTorch wheel $wheel_path"
  fi
  wheel_bytes="$(stat -c '%s' "$wheel_path")"
  [[ "$wheel_bytes" == "$TORCH_WHEEL_BYTES" ]] || fail "incomplete PyTorch wheel: got $wheel_bytes bytes, expected $TORCH_WHEEL_BYTES"

  log "installing Jetson CUDA PyTorch"
  "$VENV_DIR/bin/python" -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
  "$VENV_DIR/bin/python" -m pip install --no-cache-dir numpy==1.26.1 "Cython<3" packaging
  "$VENV_DIR/bin/python" -m pip install --no-cache-dir "$wheel_path"
}

install_python_deps() {
  log "installing PersonaPlex runtime dependencies"
  "$VENV_DIR/bin/python" -m pip install --no-cache-dir \
    "safetensors>=0.4.0,<0.5" \
    "huggingface-hub>=0.24,<0.25" \
    "einops==0.7" \
    "sentencepiece==0.2" \
    "sounddevice==0.5" \
    "sphn>=0.1.4,<0.2" \
    "aiohttp>=3.10.5,<3.11" \
    requests

  if [[ -f "$ROOT_DIR/personaplex-setup/moshi/pyproject.toml" ]]; then
    log "installing fork runtime from personaplex-setup/moshi"
    "$VENV_DIR/bin/python" -m pip install --no-deps -e "$ROOT_DIR/personaplex-setup/moshi"
  else
    log "fork runtime is not materialized at personaplex-setup/moshi; setup will continue but server start will fail until that gitlink is restored"
  fi
}

download_model() {
  log "ensuring NF4 model artifacts from $MODEL_REPO"
  mkdir -p "$MODEL_DIR"
  local joined
  joined="$(printf '%s\n' "${required_files[@]}")"
  PERSONAPLEX_MODEL_REPO="$MODEL_REPO" PERSONAPLEX_MODEL_DIR="$MODEL_DIR" PERSONAPLEX_ALLOW_PATTERNS="$joined" "$VENV_DIR/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download

patterns = [line for line in os.environ["PERSONAPLEX_ALLOW_PATTERNS"].splitlines() if line]
snapshot_download(
    repo_id=os.environ["PERSONAPLEX_MODEL_REPO"],
    local_dir=os.environ["PERSONAPLEX_MODEL_DIR"],
    allow_patterns=patterns,
    token=False,
)
PY
}

verify() {
  log "verifying Python packages and CUDA"
  "$VENV_DIR/bin/python" - <<'PY'
import importlib
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")
for name in ("aiohttp", "sentencepiece", "sphn", "safetensors", "huggingface_hub"):
    importlib.import_module(name)
print("dependency_imports ok")
PY

  local missing=()
  local file
  for file in "${required_files[@]}"; do
    [[ -s "$MODEL_DIR/$file" ]] || missing+=("$file")
  done
  if (( ${#missing[@]} > 0 )); then
    fail "missing model files under $MODEL_DIR: ${missing[*]}"
  fi
  log "model artifacts verified in $MODEL_DIR"
}

main() {
  need_command "$PYTHON_BIN"
  detect_platform
  install_system_packages
  create_venv
  install_torch
  install_python_deps
  download_model
  verify
  log "setup complete"
  log "start with: $ROOT_DIR/scripts/start_nf4_server.sh"
}

main "$@"
