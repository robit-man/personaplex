#!/usr/bin/env bash
set -euo pipefail

readonly VORYN_ROOT="/home/roko/Documents/Projects/github_repos/vox_docker_eval/voryn"
lane="${1:?lane index is required}"

case "$lane" in
  0|3)
    resource_root="/srv/voxrn_cache/personaplex-lanes/gpu0"
    [[ "$lane" == 3 ]] && resource_root="/srv/voxrn_cache/personaplex-lanes/workers/worker3"
    inference_model="robit/ornith:35b"
    inference_endpoint="http://127.0.0.1:12084/v1/chat/completions"
    repair_model="robit/ornith:35b"
    repair_endpoint="http://127.0.0.1:11434/v1/chat/completions"
    ;;
  1|4)
    resource_root="/srv/voxrn_cache/personaplex-lanes/gpu1"
    [[ "$lane" == 4 ]] && resource_root="/srv/voxrn_cache/personaplex-lanes/workers/worker4"
    inference_model="personaplex-control-ornith:35b"
    inference_endpoint="http://127.0.0.1:12080/v1/chat/completions"
    repair_model="robit/ornith:35b"
    repair_endpoint="http://127.0.0.1:12084/v1/chat/completions"
    ;;
  2|5)
    resource_root="/srv/voxrn_cache/personaplex-lanes/gpu2"
    [[ "$lane" == 5 ]] && resource_root="/srv/voxrn_cache/personaplex-lanes/workers/worker5"
    inference_model="personaplex-control-ornith:35b"
    inference_endpoint="http://127.0.0.1:12080/v1/chat/completions"
    repair_model="robit/ornith:35b"
    repair_endpoint="http://127.0.0.1:11434/v1/chat/completions"
    ;;
  *)
    printf 'unsupported lane index: %s\n' "$lane" >&2
    exit 64
    ;;
esac

readonly interval_seconds="${CERTIFY_INTERVAL_SECONDS:-120}"

cd "$VORYN_ROOT"
set -a
source ./.env
set +a

while :; do
  if ! env \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="$((lane % 3))" \
    VOXRN_RESOURCE_ROOT="$resource_root" \
    SYNTHESIS_LANE_INDEX="$lane" \
    SYNTHESIS_LANE_COUNT=6 \
    SYNTHESIS_PROGRESS_NAMESPACE=v10-diverse-v6-par6 \
    SYNTHESIZE_CERTIFIER_ENDPOINT="$inference_endpoint" \
    SYNTHESIZE_CERTIFIER_MODEL="$inference_model" \
    SYNTHESIZE_CERTIFIER_REPAIR_ENDPOINT="$repair_endpoint" \
    SYNTHESIZE_CERTIFIER_REPAIR_MODEL="$repair_model" \
    node scripts/certify-personaplex-v7-paired-queue.js --lane="$lane" --namespace=v10-diverse-v6-par6 --max=1; then
    printf 'lane=%s certificate pass failed; retaining raw artifacts for retry\n' "$lane" >&2
  fi
  sleep "$interval_seconds"
done
