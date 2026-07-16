#!/usr/bin/env bash
set -euo pipefail

readonly VORYN_ROOT="/home/roko/Documents/Projects/github_repos/vox_docker_eval/voryn"
readonly RUNTIME_ENV="/srv/voxrn_cache/personaplex-systemd/personaplex-runtime.env"
lane="${1:?lane index is required}"
[[ -r "$RUNTIME_ENV" ]] || { printf 'PersonaPlex runtime contract missing: %s\n' "$RUNTIME_ENV" >&2; exit 78; }
source "$RUNTIME_ENV"
readonly PLAN_PATH="${PERSONAPLEX_SYNTHESIS_PLAN_PATH:?PERSONAPLEX_SYNTHESIS_PLAN_PATH is required}"
readonly voicebox_base_url="http://${PERSONAPLEX_BIND_HOST}:${PERSONAPLEX_VOICEBOX_PORT}"
readonly semantic_base_url="http://${PERSONAPLEX_BIND_HOST}:${PERSONAPLEX_CHATML_PORT}"
readonly semantic_endpoint="${semantic_base_url}/v1/chat/completions"

case "$lane" in
  0) physical_gpu=0; resource_root="/srv/voxrn_cache/personaplex-lanes/gpu0" ;;
  1) physical_gpu=1; resource_root="/srv/voxrn_cache/personaplex-lanes/gpu1" ;;
  2) physical_gpu=2; resource_root="/srv/voxrn_cache/personaplex-lanes/gpu2" ;;
  *) printf 'unsupported lane index: %s\n' "$lane" >&2; exit 64 ;;
esac

wait_for_existing_lane_worker() {
  while :; do
    local found=0 pid
    while IFS= read -r pid; do
      [[ -r "/proc/$pid/environ" ]] || continue
      if tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | rg -Fx "SYNTHESIS_LANE_INDEX=$lane" >/dev/null; then
        found=1
        printf 'lane=%s waiting for existing worker pid=%s to preserve progress\n' "$lane" "$pid"
      fi
    done < <(pgrep -f 'node scripts/run-personaplex-v7-paired-lane.js' || true)
    (( found == 0 )) && return
    sleep 20
  done
}

wait_for_runtime() {
  local attempt
  for attempt in $(seq 1 60); do
    if curl -fsS --max-time 3 "${voicebox_base_url}/health" >/dev/null && curl -fsS --max-time 3 "${semantic_base_url}/health" >/dev/null; then
      return 0
    fi
    sleep 2
  done
  printf 'PersonaPlex runtime unavailable: voicebox=%s semantic=%s\n' "$voicebox_base_url" "$semantic_base_url" >&2
  return 1
}

mkdir -p "$resource_root/logs"
wait_for_existing_lane_worker
cd "$VORYN_ROOT"
set -a
source ./.env
set +a
wait_for_runtime
printf '{"event":"personaplex_runtime_bound","lane":%s,"runtimeVersion":"%s","voicebox":"%s","semantic":"%s","model":"%s"}\n' "$lane" "$PERSONAPLEX_RUNTIME_VERSION" "$voicebox_base_url" "$semantic_base_url" "$PERSONAPLEX_CONTROL_MODEL"

exec env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="$physical_gpu" \
  VOXRN_RESOURCE_ROOT="$resource_root" \
  VOICEBOX_BASE_URL="$voicebox_base_url" \
  VOICEBOX_URL= \
  VOICEBOX_PORT="$PERSONAPLEX_VOICEBOX_PORT" \
  VOICEBOX_RETAIN_LOADED_MODELS=1 \
  VOICEBOX_REQUEST_TIMEOUT_MS=240000 \
  VOICEBOX_START_TIMEOUT_MS=240000 \
  SYNTHESIS_PLAN_PATH="$PLAN_PATH" \
  SYNTHESIS_LANE_INDEX="$lane" \
  SYNTHESIS_LANE_COUNT=3 \
  SYNTHESIS_MAX_COUNTERFACTUAL_GROUPS="${PERSONAPLEX_SYNTHESIS_BATCH_GROUPS:-8}" \
  SYNTHESIS_MAX_ATTEMPTS=2 \
  SYNTHESIS_MAX_SUFFIX_REPAIRS=3 \
  SYNTHESIS_MAX_REGENERATIONS_PER_GROUP=3 \
  SYNTHESIS_WAIT_FOR_CERTIFICATION=0 \
  SYNTHESIS_PROGRESS_NAMESPACE=v11-repairable-v8 \
  SYNTHESIS_CERTIFICATE_SCAN_ROOT=/srv/voxrn_cache/personaplex-lanes \
  SYNTHESIS_MIN_ASR_CONFIDENCE=0.45 \
  SYNTHESIS_MAX_ASR_WER="${PERSONAPLEX_SYNTHESIS_MAX_ASR_WER:-0.12}" \
  SYNTHESIS_PIPELINE_NAMESPACE=v7-final \
  SYNTHESIZE_INFERENCE_PROVIDER=ollama \
  SYNTHESIZE_INFERENCE_TIMEOUT_MS=150000 \
  SYNTHESIZE_INFERENCE_MODEL="$PERSONAPLEX_CONTROL_MODEL" \
  SYNTHESIZE_INFERENCE_ENDPOINT="$semantic_endpoint" \
  SYNTHESIZE_CONTROL_VERIFIER_FALLBACK_PROVIDER=ollama \
  SYNTHESIZE_CONTROL_VERIFIER_FALLBACK_MODEL="$PERSONAPLEX_CONTROL_MODEL" \
  SYNTHESIZE_CONTROL_VERIFIER_FALLBACK_ENDPOINT="$semantic_endpoint" \
  SYNTHESIZE_CONTROL_ENVELOPE_REPAIR_PROVIDER=ollama \
  SYNTHESIZE_CONTROL_ENVELOPE_REPAIR_MODEL="$PERSONAPLEX_CONTROL_MODEL" \
  SYNTHESIZE_CONTROL_ENVELOPE_REPAIR_ENDPOINT="$semantic_endpoint" \
  SYNTHESIZE_EVIDENCE_ENVELOPE_REPAIR_PROVIDER=ollama \
  SYNTHESIZE_EVIDENCE_ENVELOPE_REPAIR_MODEL="$PERSONAPLEX_CONTROL_MODEL" \
  SYNTHESIZE_EVIDENCE_ENVELOPE_REPAIR_ENDPOINT="$semantic_endpoint" \
  SYNTHESIZE_DIALOGUE_INFERENCE_PROVIDER=ollama \
  SYNTHESIZE_DIALOGUE_INFERENCE_MODEL="$PERSONAPLEX_CONTROL_MODEL" \
  SYNTHESIZE_DIALOGUE_INFERENCE_ENDPOINT="$semantic_endpoint" \
  SYNTHESIZE_DIALOGUE_ENVELOPE_REPAIR_PROVIDER=ollama \
  SYNTHESIZE_DIALOGUE_ENVELOPE_REPAIR_MODEL="$PERSONAPLEX_CONTROL_MODEL" \
  SYNTHESIZE_DIALOGUE_ENVELOPE_REPAIR_ENDPOINT="$semantic_endpoint" \
  SYNTHESIZE_DIALOGUE_ENVELOPE_REPAIR_FALLBACK_PROVIDER=ollama \
  SYNTHESIZE_DIALOGUE_ENVELOPE_REPAIR_FALLBACK_MODEL="$PERSONAPLEX_CONTROL_MODEL" \
  SYNTHESIZE_DIALOGUE_ENVELOPE_REPAIR_FALLBACK_ENDPOINT="$semantic_endpoint" \
  SYNTHESIZE_TARGET_TEMPERATURE=0.12 \
  SYNTHESIZE_CALLER_TEMPERATURE=0.40 \
  SYNTHESIZE_PLANNER_MAX_TOKENS=500 \
  node scripts/run-personaplex-v7-paired-lane.js
