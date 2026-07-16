#!/usr/bin/env bash
set -euo pipefail

readonly RUNTIME_ENV="/srv/voxrn_cache/personaplex-systemd/personaplex-runtime.env"
lane="${1:?semantic lane index is required}"
[[ -r "$RUNTIME_ENV" ]] || { printf 'PersonaPlex runtime contract missing: %s\n' "$RUNTIME_ENV" >&2; exit 78; }
source "$RUNTIME_ENV"

case "$lane" in
  0) control_port="${PERSONAPLEX_CONTROL_LANE0_PORT:?}"; physical_gpu="${PERSONAPLEX_CONTROL_LANE0_GPU:?}" ;;
  1) control_port="${PERSONAPLEX_CONTROL_LANE1_PORT:?}"; physical_gpu="${PERSONAPLEX_CONTROL_LANE1_GPU:?}" ;;
  2) control_port="${PERSONAPLEX_CONTROL_LANE2_PORT:?}"; physical_gpu="${PERSONAPLEX_CONTROL_LANE2_GPU:?}" ;;
  *) printf 'unsupported semantic lane index: %s\n' "$lane" >&2; exit 64 ;;
esac

exec env \
  OLLAMA_HOST="${PERSONAPLEX_BIND_HOST}:${control_port}" \
  OLLAMA_MODELS=/srv/ollama/models \
  OLLAMA_CONTEXT_LENGTH="$PERSONAPLEX_CONTROL_NUM_CTX" \
  OLLAMA_NUM_PARALLEL=1 \
  OLLAMA_MAX_LOADED_MODELS=1 \
  OLLAMA_KEEP_ALIVE=10m \
  CUDA_VISIBLE_DEVICES="$physical_gpu" \
  GGML_VK_VISIBLE_DEVICES= \
  HIP_VISIBLE_DEVICES= \
  /usr/local/bin/ollama serve
