#!/usr/bin/env bash
set -euo pipefail

readonly VORYN_ROOT="/home/roko/Documents/Projects/github_repos/vox_docker_eval/voryn"
readonly RUNTIME_ENV="/srv/voxrn_cache/personaplex-systemd/personaplex-runtime.env"
readonly NODE="/home/roko/.local/share/fnm/node-versions/v24.14.0/installation/bin/node"
lane="${1:?semantic lane index is required}"
[[ -r "$RUNTIME_ENV" ]] || { printf 'PersonaPlex runtime contract missing: %s\n' "$RUNTIME_ENV" >&2; exit 78; }
[[ -x "$NODE" ]] || { printf 'Node runtime missing: %s\n' "$NODE" >&2; exit 78; }
source "$RUNTIME_ENV"

case "$lane" in
  0) control_port="${PERSONAPLEX_CONTROL_LANE0_PORT:?}"; proxy_port="${PERSONAPLEX_CHATML_LANE0_PORT:?}" ;;
  1) control_port="${PERSONAPLEX_CONTROL_LANE1_PORT:?}"; proxy_port="${PERSONAPLEX_CHATML_LANE1_PORT:?}" ;;
  2) control_port="${PERSONAPLEX_CONTROL_LANE2_PORT:?}"; proxy_port="${PERSONAPLEX_CHATML_LANE2_PORT:?}" ;;
  *) printf 'unsupported semantic lane index: %s\n' "$lane" >&2; exit 64 ;;
esac

cd "$VORYN_ROOT"
exec env \
  PERSONAPLEX_BIND_HOST="$PERSONAPLEX_BIND_HOST" \
  PERSONAPLEX_CONTROL_MODEL="$PERSONAPLEX_CONTROL_MODEL" \
  PERSONAPLEX_CONTROL_NUM_CTX="$PERSONAPLEX_CONTROL_NUM_CTX" \
  OLLAMA_UPSTREAM="http://${PERSONAPLEX_BIND_HOST}:${control_port}" \
  PORT="$proxy_port" \
  OLLAMA_KEEP_ALIVE=10m \
  "$NODE" scripts/ollama-chatml-proxy.js
