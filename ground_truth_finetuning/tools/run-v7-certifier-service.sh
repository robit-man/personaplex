#!/usr/bin/env bash
set -euo pipefail

readonly VORYN_ROOT="/home/roko/Documents/Projects/github_repos/vox_docker_eval/voryn"
readonly RUNTIME_ENV="/srv/voxrn_cache/personaplex-systemd/personaplex-runtime.env"
lane="${1:?lane index is required}"
[[ -r "$RUNTIME_ENV" ]] || { printf 'PersonaPlex runtime contract missing: %s\n' "$RUNTIME_ENV" >&2; exit 78; }
source "$RUNTIME_ENV"
readonly semantic_base_url="http://${PERSONAPLEX_BIND_HOST}:${PERSONAPLEX_CHATML_PORT}"
readonly semantic_endpoint="${semantic_base_url}/v1/chat/completions"

case "$lane" in
  0) resource_root="/srv/voxrn_cache/personaplex-lanes/gpu0" ;;
  1) resource_root="/srv/voxrn_cache/personaplex-lanes/gpu1" ;;
  2) resource_root="/srv/voxrn_cache/personaplex-lanes/gpu2" ;;
  *) printf 'unsupported lane index: %s\n' "$lane" >&2; exit 64 ;;
esac

readonly interval_seconds="${CERTIFY_INTERVAL_SECONDS:-15}"
cd "$VORYN_ROOT"
set -a
source ./.env
set +a
printf '{"event":"personaplex_runtime_bound","lane":%s,"runtimeVersion":"%s","semantic":"%s","model":"%s"}\n' "$lane" "$PERSONAPLEX_RUNTIME_VERSION" "$semantic_base_url" "$PERSONAPLEX_CONTROL_MODEL"

while :; do
  if ! curl -fsS --max-time 3 "${semantic_base_url}/health" >/dev/null; then
    printf 'lane=%s semantic control proxy unavailable at %s; retrying\n' "$lane" "$semantic_base_url" >&2
  elif ! env \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES="$((lane % 3))" \
    VOXRN_RESOURCE_ROOT="$resource_root" \
    SYNTHESIS_LANE_INDEX="$lane" \
    SYNTHESIS_LANE_COUNT=3 \
    SYNTHESIS_PROGRESS_NAMESPACE=v11-repairable-v8 \
    SYNTHESIZE_CERTIFIER_ENDPOINT="$semantic_endpoint" \
    SYNTHESIZE_CERTIFIER_MODEL="$PERSONAPLEX_CONTROL_MODEL" \
    SYNTHESIZE_CERTIFIER_REPAIR_ENDPOINT="$semantic_endpoint" \
    SYNTHESIZE_CERTIFIER_REPAIR_MODEL="$PERSONAPLEX_CONTROL_MODEL" \
    node scripts/certify-personaplex-v7-paired-queue.js --lane="$lane" --namespace=v11-repairable-v8 --max=1; then
    printf 'lane=%s certificate pass failed; retaining raw artifacts for retry\n' "$lane" >&2
  fi
  sleep "$interval_seconds"
done
