#!/usr/bin/env bash
set -uo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
runtime_env="${PERSONAPLEX_GENERATIVE_ENV:-/srv/voxrn_cache/personaplex-systemd/personaplex-openrouter.env}"

if [[ ! -r "$runtime_env" ]]; then
  printf 'runtime environment is not readable: %s\n' "$runtime_env" >&2
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$runtime_env"
set +a

retry_seconds="${PERSONAPLEX_TRANSPORT_OUTAGE_RETRY_SECONDS:-${PERSONAPLEX_GENERATIVE_RETRY_MAX_SECONDS:-30}}"
if [[ ! "$retry_seconds" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  printf 'transport outage retry interval must be numeric\n' >&2
  exit 2
fi

cd "$repository_root"
while true; do
  PYTHONPATH="$repository_root${PYTHONPATH:+:$PYTHONPATH}" \
    python3 ground_truth_finetuning/tools/build_scenarios_from_blueprints_v5.py "$@"
  status=$?
  if (( status == 0 )); then
    exit 0
  fi
  if (( status != 75 )); then
    exit "$status"
  fi
  printf 'physical model route unavailable; exact resume in %ss\n' "$retry_seconds" >&2
  sleep "$retry_seconds"
done
