#!/usr/bin/env bash
set -euo pipefail

readonly SUITE_ROOT="/srv/personaplex_workspace/robit-man-personaplex/ground_truth_finetuning"
readonly RUNTIME_ENV="/srv/voxrn_cache/personaplex-systemd/personaplex-runtime.env"
[[ -r "$RUNTIME_ENV" ]] || { printf 'PersonaPlex runtime contract missing: %s\n' "$RUNTIME_ENV" >&2; exit 78; }
source "$RUNTIME_ENV"

exec python3 "$SUITE_ROOT/tools/materialize_diverse_synthesis_cascade.py" \
  --planner-endpoint "http://${PERSONAPLEX_BIND_HOST}:${PERSONAPLEX_CHATML_PORT}/v1/chat/completions" \
  --planner-model "$PERSONAPLEX_CONTROL_MODEL" \
  "$@"
