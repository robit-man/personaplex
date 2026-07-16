#!/usr/bin/env bash
set -euo pipefail

readonly VORYN_ROOT="/home/roko/Documents/Projects/github_repos/vox_docker_eval/voryn"
readonly PLAN_PATH="/srv/personaplex_workspace/ground_truth_runs/personaplex-1000-plan.v8-counterfactual-diverse-v6.jsonl"

lane="${1:?lane index is required}"

case "$lane" in
  0|3)
    physical_gpu=0
    resource_root="/srv/voxrn_cache/personaplex-lanes/gpu0"
    [[ "$lane" == 3 ]] && resource_root="/srv/voxrn_cache/personaplex-lanes/workers/worker3"
    voicebox_url="http://127.0.0.1:17500"
    # The independent semantic plane is the resident 35B model on GPU 2;
    # dialogue remains on the local GPU-0 PersonaPlex-control model.
    inference_model="robit/ornith:35b"
    inference_endpoint="http://127.0.0.1:12084/v1/chat/completions"
    verifier_fallback_model="robit/ornith:35b"
    verifier_fallback_endpoint="http://127.0.0.1:11434/v1/chat/completions"
    dialogue_model="personaplex-control-ornith:35b"
    dialogue_endpoint="http://127.0.0.1:12080/v1/chat/completions"
    ;;
  1|4)
    physical_gpu=1
    resource_root="/srv/voxrn_cache/personaplex-lanes/gpu1"
    [[ "$lane" == 4 ]] && resource_root="/srv/voxrn_cache/personaplex-lanes/workers/worker4"
    voicebox_url="http://127.0.0.1:17501"
    # Keep the semantic judge independent of the GPU-1 dialogue model.
    inference_model="personaplex-control-ornith:35b"
    inference_endpoint="http://127.0.0.1:12080/v1/chat/completions"
    verifier_fallback_model="robit/ornith:35b"
    verifier_fallback_endpoint="http://127.0.0.1:12084/v1/chat/completions"
    dialogue_model="robit/ornith:35b"
    dialogue_endpoint="http://127.0.0.1:11434/v1/chat/completions"
    ;;
  2|5)
    physical_gpu=2
    resource_root="/srv/voxrn_cache/personaplex-lanes/gpu2"
    [[ "$lane" == 5 ]] && resource_root="/srv/voxrn_cache/personaplex-lanes/workers/worker5"
    voicebox_url="http://127.0.0.1:17502"
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

exec env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES="$physical_gpu" \
  VOXRN_RESOURCE_ROOT="$resource_root" \
  VOICEBOX_BASE_URL="$voicebox_url" \
  SYNTHESIS_PLAN_PATH="$PLAN_PATH" \
  SYNTHESIS_LANE_INDEX="$lane" \
  SYNTHESIS_LANE_COUNT=6 \
  SYNTHESIS_MAX_COUNTERFACTUAL_GROUPS=1 \
  SYNTHESIS_MAX_ATTEMPTS=3 \
  SYNTHESIS_WAIT_FOR_CERTIFICATION=0 \
  SYNTHESIS_PROGRESS_NAMESPACE=v10-diverse-v6-par6 \
  SYNTHESIS_CERTIFICATE_SCAN_ROOT=/srv/voxrn_cache/personaplex-lanes \
  SYNTHESIS_MIN_ASR_CONFIDENCE=0.45 \
  SYNTHESIS_MAX_ASR_WER=0.25 \
  SYNTHESIS_PIPELINE_NAMESPACE=v7-final \
  SYNTHESIZE_INFERENCE_PROVIDER=ollama \
  SYNTHESIZE_INFERENCE_MODEL="$inference_model" \
  SYNTHESIZE_INFERENCE_ENDPOINT="$inference_endpoint" \
  SYNTHESIZE_CONTROL_VERIFIER_FALLBACK_PROVIDER=ollama \
  SYNTHESIZE_CONTROL_VERIFIER_FALLBACK_MODEL="$verifier_fallback_model" \
  SYNTHESIZE_CONTROL_VERIFIER_FALLBACK_ENDPOINT="$verifier_fallback_endpoint" \
  SYNTHESIZE_CONTROL_ENVELOPE_REPAIR_PROVIDER=ollama \
  SYNTHESIZE_CONTROL_ENVELOPE_REPAIR_MODEL="$verifier_fallback_model" \
  SYNTHESIZE_CONTROL_ENVELOPE_REPAIR_ENDPOINT="$verifier_fallback_endpoint" \
  SYNTHESIZE_EVIDENCE_ENVELOPE_REPAIR_PROVIDER=ollama \
  SYNTHESIZE_EVIDENCE_ENVELOPE_REPAIR_MODEL="$verifier_fallback_model" \
  SYNTHESIZE_EVIDENCE_ENVELOPE_REPAIR_ENDPOINT="$verifier_fallback_endpoint" \
  SYNTHESIZE_DIALOGUE_INFERENCE_PROVIDER=ollama \
  SYNTHESIZE_DIALOGUE_INFERENCE_MODEL="$dialogue_model" \
  SYNTHESIZE_DIALOGUE_INFERENCE_ENDPOINT="$dialogue_endpoint" \
  SYNTHESIZE_DIALOGUE_ENVELOPE_REPAIR_PROVIDER=ollama \
  SYNTHESIZE_DIALOGUE_ENVELOPE_REPAIR_MODEL="$inference_model" \
  SYNTHESIZE_DIALOGUE_ENVELOPE_REPAIR_ENDPOINT="$inference_endpoint" \
  SYNTHESIZE_DIALOGUE_ENVELOPE_REPAIR_FALLBACK_PROVIDER=ollama \
  SYNTHESIZE_DIALOGUE_ENVELOPE_REPAIR_FALLBACK_MODEL="$verifier_fallback_model" \
  SYNTHESIZE_DIALOGUE_ENVELOPE_REPAIR_FALLBACK_ENDPOINT="$verifier_fallback_endpoint" \
  SYNTHESIZE_TARGET_TEMPERATURE=0.12 \
  SYNTHESIZE_CALLER_TEMPERATURE=0.40 \
  SYNTHESIZE_PLANNER_MAX_TOKENS=500 \
  node scripts/run-personaplex-v7-paired-lane.js
