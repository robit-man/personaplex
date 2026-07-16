#!/usr/bin/env bash
set -euo pipefail

readonly VORYN_ROOT="/home/roko/Documents/Projects/github_repos/vox_docker_eval/voryn"
readonly RUNTIME_ENV="/srv/voxrn_cache/personaplex-systemd/personaplex-runtime.env"
readonly PYTHON="$VORYN_ROOT/.voicebox-venv/bin/python"
[[ -r "$RUNTIME_ENV" ]] || { printf 'PersonaPlex runtime contract missing: %s\n' "$RUNTIME_ENV" >&2; exit 78; }
source "$RUNTIME_ENV"
readonly RESOURCE_ROOT="/srv/voxrn_cache/personaplex-lanes/gpu${PERSONAPLEX_VOICEBOX_GPU}"
readonly PORT="$PERSONAPLEX_VOICEBOX_PORT"

cd "$VORYN_ROOT"
set -a
source ./.env
set +a

if [[ ! -x "$PYTHON" || ! -d "$VORYN_ROOT/.voicebox-src" ]]; then
  printf 'Voicebox runtime is not installed under %s\n' "$VORYN_ROOT" >&2
  exit 78
fi

env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$PERSONAPLEX_VOICEBOX_GPU" VOICEBOX_CUDA_VISIBLE_DEVICES="$PERSONAPLEX_VOICEBOX_GPU" "$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit('Voicebox requires CUDA; CPU fallback is forbidden')
name = torch.cuda.get_device_name(0)
if 'A100' not in name.upper():
    raise SystemExit(f'Voicebox requires an A100, found {name}')
PY

cd "$VORYN_ROOT/.voicebox-src"
exec env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="$PERSONAPLEX_VOICEBOX_GPU" \
  VOICEBOX_CUDA_VISIBLE_DEVICES="$PERSONAPLEX_VOICEBOX_GPU" \
  VOXRN_RESOURCE_ROOT="$RESOURCE_ROOT" \
  VOICEBOX_MODELS_DIR=/srv/voxrn_cache/models \
  HF_HOME=/srv/voxrn_cache/huggingface \
  HF_HUB_CACHE=/srv/voxrn_cache/models \
  HF_HUB_DISABLE_XET=1 \
  HUGGINGFACE_HUB_CACHE=/srv/voxrn_cache/huggingface/hub \
  TORCH_HOME=/srv/voxrn_cache/torch \
  "$PYTHON" -m backend.main --host "$PERSONAPLEX_BIND_HOST" --port "$PORT" --data-dir "$RESOURCE_ROOT/voicebox"
