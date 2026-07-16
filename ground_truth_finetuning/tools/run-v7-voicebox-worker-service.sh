#!/usr/bin/env bash
set -euo pipefail

readonly VORYN_ROOT="/home/roko/Documents/Projects/github_repos/vox_docker_eval/voryn"
readonly RUNTIME_ENV="/srv/voxrn_cache/personaplex-systemd/personaplex-runtime.env"
readonly PYTHON="$VORYN_ROOT/.voicebox-venv/bin/python"
lane="${1:?voicebox lane index is required}"
[[ -r "$RUNTIME_ENV" ]] || { printf 'PersonaPlex runtime contract missing: %s\n' "$RUNTIME_ENV" >&2; exit 78; }
source "$RUNTIME_ENV"

case "$lane" in
  0) physical_gpu="${PERSONAPLEX_VOICEBOX_LANE0_GPU:?}"; port="${PERSONAPLEX_VOICEBOX_LANE0_PORT:?}" ;;
  1) physical_gpu="${PERSONAPLEX_VOICEBOX_LANE1_GPU:?}"; port="${PERSONAPLEX_VOICEBOX_LANE1_PORT:?}" ;;
  2) physical_gpu="${PERSONAPLEX_VOICEBOX_LANE2_GPU:?}"; port="${PERSONAPLEX_VOICEBOX_LANE2_PORT:?}" ;;
  *) printf 'unsupported voicebox lane index: %s\n' "$lane" >&2; exit 64 ;;
esac
readonly RESOURCE_ROOT="/srv/voxrn_cache/personaplex-lanes/gpu${physical_gpu}"

cd "$VORYN_ROOT"
set -a
source ./.env
set +a
[[ -x "$PYTHON" && -d "$VORYN_ROOT/.voicebox-src" ]] || { printf 'Voicebox runtime is not installed under %s\n' "$VORYN_ROOT" >&2; exit 78; }

env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$physical_gpu" VOICEBOX_CUDA_VISIBLE_DEVICES="$physical_gpu" "$PYTHON" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit('Voicebox requires CUDA; CPU fallback is forbidden')
if 'A100' not in torch.cuda.get_device_name(0).upper():
    raise SystemExit(f'Voicebox requires an A100, found {torch.cuda.get_device_name(0)}')
PY

cd "$VORYN_ROOT/.voicebox-src"
exec env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="$physical_gpu" \
  VOICEBOX_CUDA_VISIBLE_DEVICES="$physical_gpu" \
  VOXRN_RESOURCE_ROOT="$RESOURCE_ROOT" \
  VOICEBOX_MODELS_DIR=/srv/voxrn_cache/models \
  HF_HOME=/srv/voxrn_cache/huggingface \
  HF_HUB_CACHE=/srv/voxrn_cache/models \
  HF_HUB_DISABLE_XET=1 \
  HUGGINGFACE_HUB_CACHE=/srv/voxrn_cache/huggingface/hub \
  TORCH_HOME=/srv/voxrn_cache/torch \
  "$PYTHON" -m backend.main --host "$PERSONAPLEX_BIND_HOST" --port "$port" --data-dir "$RESOURCE_ROOT/voicebox"
