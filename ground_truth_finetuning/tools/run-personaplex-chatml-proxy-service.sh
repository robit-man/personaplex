#!/usr/bin/env bash
set -euo pipefail

readonly VORYN_ROOT="/home/roko/Documents/Projects/github_repos/vox_docker_eval/voryn"
readonly RUNTIME_ENV="/srv/voxrn_cache/personaplex-systemd/personaplex-runtime.env"
readonly NODE="/home/roko/.local/share/fnm/node-versions/v24.14.0/installation/bin/node"
[[ -r "$RUNTIME_ENV" ]] || { printf 'PersonaPlex runtime contract missing: %s\n' "$RUNTIME_ENV" >&2; exit 78; }
[[ -x "$NODE" ]] || { printf 'Node runtime missing: %s\n' "$NODE" >&2; exit 78; }
source "$RUNTIME_ENV"

cd "$VORYN_ROOT"
exec env \
  PERSONAPLEX_BIND_HOST="$PERSONAPLEX_BIND_HOST" \
  PERSONAPLEX_CONTROL_MODEL="$PERSONAPLEX_CONTROL_MODEL" \
  PERSONAPLEX_CONTROL_NUM_CTX="$PERSONAPLEX_CONTROL_NUM_CTX" \
  OLLAMA_UPSTREAM="http://${PERSONAPLEX_BIND_HOST}:${PERSONAPLEX_CONTROL_PORT}" \
  PORT="$PERSONAPLEX_CHATML_PORT" \
  OLLAMA_KEEP_ALIVE=10m \
  "$NODE" scripts/ollama-chatml-proxy.js
