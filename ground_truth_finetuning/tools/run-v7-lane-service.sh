#!/usr/bin/env bash
set -euo pipefail

readonly VORYN_ROOT="/home/roko/Documents/Projects/github_repos/vox_docker_eval/voryn"
readonly PLAN_PATH="/srv/personaplex_workspace/ground_truth_runs/personaplex-1000-plan.v8-counterfactual-diverse-v6.jsonl"

lane="${1:?lane index is required}"

case "$lane" in
  0)
    physical_gpu=0
    resource_root="/srv/voxrn_cache/personaplex-lanes/gpu0"
    voicebox_port=17500
    # The independent semantic plane is the resident 35B model on GPU 2;
    # dialogue remains on the local GPU-0 PersonaPlex-control model.
    inference_model="robit/ornith:35b"
    inference_endpoint="http://127.0.0.1:12084/v1/chat/completions"
    verifier_fallback_model="robit/ornith:35b"
    verifier_fallback_endpoint="http://127.0.0.1:11434/v1/chat/completions"
    dialogue_model="personaplex-control-ornith:35b"
    dialogue_endpoint="http://127.0.0.1:12080/v1/chat/completions"
    ;;
  1)
    physical_gpu=1
    resource_root="/srv/voxrn_cache/personaplex-lanes/gpu1"
    voicebox_port=17501
    # Keep the semantic judge independent of the GPU-1 dialogue model.
    inference_model="personaplex-control-ornith:35b"
    inference_endpoint="http://127.0.0.1:12080/v1/chat/completions"
    verifier_fallback_model="robit/ornith:35b"
    verifier_fallback_endpoint="http://127.0.0.1:12084/v1/chat/completions"
    dialogue_model="robit/ornith:35b"
    dialogue_endpoint="http://127.0.0.1:11434/v1/chat/completions"
    ;;
  2)
    physical_gpu=2
    resource_root="/srv/voxrn_cache/personaplex-lanes/gpu2"
    voicebox_port=17502
    # The independent semantic plane is the resident GPU-0 control model.
    inference_model="personaplex-control-ornith:35b"
    inference_endpoint="http://127.0.0.1:12080/v1/chat/completions"
    verifier_fallback_model="robit/ornith:35b"
    verifier_fallback_endpoint="http://127.0.0.1:11434/v1/chat/completions"
    dialogue_model="robit/ornith:35b"
    dialogue_endpoint="http://127.0.0.1:12084/v1/chat/completions"
    ;;
  *)
    printf 'unsupported lane index: %s\n' "$lane" >&2
    exit 64
    ;;
esac

wait_for_existing_lane_worker() {
  while :; do
    local found=0
    local pid
    while IFS= read -r pid; do
      [[ -r "/proc/$pid/environ" ]] || continue
      if tr '\0' '\n' <"/proc/$pid/environ" 2>/dev/null | rg -Fx "SYNTHESIS_LANE_INDEX=$lane" >/dev/null; then
        found=1
        printf 'lane=%s waiting for existing worker pid=%s to preserve progress\n' "$lane" "$pid"
      fi
    done < <(pgrep -f 'node scripts/run-personaplex-v7-paired-lane.js' || true)
    (( found == 0 )) && return
    sleep 20
  done
}

mkdir -p "$resource_root/logs"
wait_for_existing_lane_worker

cd "$VORYN_ROOT"
set -a
source ./.env
set +a

# Do not set VOICEBOX_BASE_URL here. It marks Voicebox as an externally
# managed service and makes the renderer fail closed when no listener owns
# the lane port. The Voryn renderer owns a local child instead, waits for
# /health, and verifies the CUDA A100 before rendering any training audio.
# Synthetic targets, typed control frames, and semantic judges must come from
# the deeper language model. The PersonaPlex-control checkpoint is the audio-
# plane adaptation target, not an upstream corpus author. The renderer remains
# independently pinned to this lane's physical CUDA device.
exec env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="$physical_gpu" \
  VOXRN_RESOURCE_ROOT="$resource_root" \
  VOICEBOX_BASE_URL= \
  VOICEBOX_URL= \
  VOICEBOX_PORT="$voicebox_port" \
  VOICEBOX_RETAIN_LOADED_MODELS=1 \
  VOICEBOX_REQUEST_TIMEOUT_MS=240000 \
  VOICEBOX_START_TIMEOUT_MS=240000 \
  SYNTHESIS_PLAN_PATH="$PLAN_PATH" \
  SYNTHESIS_LANE_INDEX="$lane" \
  SYNTHESIS_LANE_COUNT=3 \
  SYNTHESIS_MAX_COUNTERFACTUAL_GROUPS=1 \
  SYNTHESIS_MAX_ATTEMPTS=2 \
  SYNTHESIS_MAX_SUFFIX_REPAIRS=3 \
  SYNTHESIS_MAX_REGENERATIONS_PER_GROUP=3 \
  SYNTHESIS_WAIT_FOR_CERTIFICATION=0 \
  SYNTHESIS_PROGRESS_NAMESPACE=v11-repairable-v8 \
  SYNTHESIS_CERTIFICATE_SCAN_ROOT=/srv/voxrn_cache/personaplex-lanes \
  SYNTHESIS_MIN_ASR_CONFIDENCE=0.45 \
  SYNTHESIS_MAX_ASR_WER=0.25 \
  SYNTHESIS_PIPELINE_NAMESPACE=v7-final \
  SYNTHESIZE_INFERENCE_PROVIDER=ollama \
  SYNTHESIZE_INFERENCE_TIMEOUT_MS=150000 \
  SYNTHESIZE_INFERENCE_MODEL=robit/ornith:35b \
  SYNTHESIZE_INFERENCE_ENDPOINT=http://127.0.0.1:12084/v1/chat/completions \
  SYNTHESIZE_CONTROL_VERIFIER_FALLBACK_PROVIDER=ollama \
  SYNTHESIZE_CONTROL_VERIFIER_FALLBACK_MODEL=robit/ornith:35b \
  SYNTHESIZE_CONTROL_VERIFIER_FALLBACK_ENDPOINT=http://127.0.0.1:12084/v1/chat/completions \
  SYNTHESIZE_CONTROL_ENVELOPE_REPAIR_PROVIDER=ollama \
  SYNTHESIZE_CONTROL_ENVELOPE_REPAIR_MODEL=robit/ornith:35b \
  SYNTHESIZE_CONTROL_ENVELOPE_REPAIR_ENDPOINT=http://127.0.0.1:12084/v1/chat/completions \
  SYNTHESIZE_EVIDENCE_ENVELOPE_REPAIR_PROVIDER=ollama \
  SYNTHESIZE_EVIDENCE_ENVELOPE_REPAIR_MODEL=robit/ornith:35b \
  SYNTHESIZE_EVIDENCE_ENVELOPE_REPAIR_ENDPOINT=http://127.0.0.1:12084/v1/chat/completions \
  SYNTHESIZE_DIALOGUE_INFERENCE_PROVIDER=ollama \
  SYNTHESIZE_DIALOGUE_INFERENCE_MODEL=robit/ornith:35b \
  SYNTHESIZE_DIALOGUE_INFERENCE_ENDPOINT=http://127.0.0.1:12084/v1/chat/completions \
  SYNTHESIZE_DIALOGUE_ENVELOPE_REPAIR_PROVIDER=ollama \
  SYNTHESIZE_DIALOGUE_ENVELOPE_REPAIR_MODEL=robit/ornith:35b \
  SYNTHESIZE_DIALOGUE_ENVELOPE_REPAIR_ENDPOINT=http://127.0.0.1:12084/v1/chat/completions \
  SYNTHESIZE_DIALOGUE_ENVELOPE_REPAIR_FALLBACK_PROVIDER=ollama \
  SYNTHESIZE_DIALOGUE_ENVELOPE_REPAIR_FALLBACK_MODEL=robit/ornith:35b \
  SYNTHESIZE_DIALOGUE_ENVELOPE_REPAIR_FALLBACK_ENDPOINT=http://127.0.0.1:12084/v1/chat/completions \
  SYNTHESIZE_TARGET_TEMPERATURE=0.12 \
  SYNTHESIZE_CALLER_TEMPERATURE=0.40 \
  SYNTHESIZE_PLANNER_MAX_TOKENS=500 \
  node scripts/run-personaplex-v7-paired-lane.js
