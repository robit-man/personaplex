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

resolve_manifest() {
  local reference="${PERSONAPLEX_CONTROL_MODEL:?}"
  local repository tag namespace name
  repository="${reference%:*}"
  tag="${reference##*:}"
  if [[ "$repository" == "$reference" ]]; then
    repository="$reference"
    tag="latest"
  fi
  if [[ "$repository" == */* ]]; then
    namespace="${repository%%/*}"
    name="${repository#*/}"
  else
    namespace="library"
    name="$repository"
  fi
  printf '/srv/ollama/models/manifests/registry.ollama.ai/%s/%s/%s\n' \
    "$namespace" "$name" "$tag"
}

manifest="$(resolve_manifest)"
[[ -r "$manifest" ]] || {
  printf 'PersonaPlex control model manifest missing: %s\n' "$manifest" >&2
  exit 78
}
model_bytes="$(jq -er '[.layers[] | select(.mediaType == "application/vnd.ollama.image.model") | .size] | add | floor' "$manifest")"
gpu_total_mib="$(nvidia-smi --id="$physical_gpu" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d '[:space:]')"
[[ "$model_bytes" =~ ^[1-9][0-9]*$ && "$gpu_total_mib" =~ ^[1-9][0-9]*$ ]] || {
  printf 'PersonaPlex dynamic GPU/model discovery failed for physical GPU %s\n' "$physical_gpu" >&2
  exit 78
}

# Admit one concurrent decode lane per model-footprint multiple within the
# programme's discovered 80%% VRAM budget. Ollama shares resident weights, so
# this deliberately overestimates per-request memory while scaling naturally
# across model sizes and GPU classes without a machine-specific lane count.
gpu_budget_bytes=$((gpu_total_mib * 1024 * 1024 * 80 / 100))
parallelism=$((gpu_budget_bytes / model_bytes))
(( parallelism >= 1 )) || parallelism=1

printf '{"event":"personaplex_ornith_worker_contract","physicalGpu":%s,"gpuTotalMiB":%s,"model":"%s","modelBytes":%s,"numCtx":%s,"parallelism":%s}\n' \
  "$physical_gpu" "$gpu_total_mib" "$PERSONAPLEX_CONTROL_MODEL" "$model_bytes" \
  "$PERSONAPLEX_CONTROL_NUM_CTX" "$parallelism" >&2

if [[ "${PERSONAPLEX_WORKER_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

exec env \
  OLLAMA_HOST="${PERSONAPLEX_BIND_HOST}:${control_port}" \
  OLLAMA_MODELS=/srv/ollama/models \
  OLLAMA_CONTEXT_LENGTH="$PERSONAPLEX_CONTROL_NUM_CTX" \
  OLLAMA_NUM_PARALLEL="$parallelism" \
  OLLAMA_MAX_LOADED_MODELS=1 \
  OLLAMA_KEEP_ALIVE=10m \
  CUDA_VISIBLE_DEVICES="$physical_gpu" \
  GGML_VK_VISIBLE_DEVICES= \
  HIP_VISIBLE_DEVICES= \
  /usr/local/bin/ollama serve
