#!/usr/bin/env bash
set -euo pipefail

readonly RUNTIME_ENV="/srv/voxrn_cache/personaplex-systemd/personaplex-runtime.env"
[[ -r "$RUNTIME_ENV" ]] || { printf 'PersonaPlex runtime contract missing: %s\n' "$RUNTIME_ENV" >&2; exit 78; }
source "$RUNTIME_ENV"

exec env \
  OLLAMA_HOST="${PERSONAPLEX_BIND_HOST}:${PERSONAPLEX_CONTROL_PORT}" \
  OLLAMA_MODELS=/srv/ollama/models \
  OLLAMA_CONTEXT_LENGTH="$PERSONAPLEX_CONTROL_NUM_CTX" \
  OLLAMA_NUM_PARALLEL=1 \
  OLLAMA_MAX_LOADED_MODELS=1 \
  OLLAMA_KEEP_ALIVE=10m \
  CUDA_VISIBLE_DEVICES="$PERSONAPLEX_CONTROL_GPU" \
  GGML_VK_VISIBLE_DEVICES= \
  HIP_VISIBLE_DEVICES= \
  /usr/local/bin/ollama serve
