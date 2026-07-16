#!/usr/bin/env bash
set -euo pipefail

readonly RUNTIME_ENV="/srv/voxrn_cache/personaplex-systemd/personaplex-runtime.env"
[[ -r "$RUNTIME_ENV" ]] || { printf 'PersonaPlex runtime contract missing: %s\n' "$RUNTIME_ENV" >&2; exit 78; }
source "$RUNTIME_ENV"

readonly CONTROL_BASE_URL="http://${PERSONAPLEX_BIND_HOST}:${PERSONAPLEX_CONTROL_PORT}"
readonly CHATML_BASE_URL="http://${PERSONAPLEX_BIND_HOST}:${PERSONAPLEX_CHATML_PORT}"
readonly VOICEBOX_BASE_URL="http://${PERSONAPLEX_BIND_HOST}:${PERSONAPLEX_VOICEBOX_PORT}"
proxy_health="$(curl -fsS --max-time 5 "${CHATML_BASE_URL}/health")"
voicebox_health="$(curl -fsS --max-time 5 "${VOICEBOX_BASE_URL}/health")"
control_residency="$(curl -fsS --max-time 5 "${CONTROL_BASE_URL}/api/ps")"

jq -n \
  --arg runtimeVersion "$PERSONAPLEX_RUNTIME_VERSION" \
  --arg runtimeContract "$RUNTIME_ENV" \
  --arg controlBaseUrl "$CONTROL_BASE_URL" \
  --arg chatmlBaseUrl "$CHATML_BASE_URL" \
  --arg voiceboxBaseUrl "$VOICEBOX_BASE_URL" \
  --argjson proxy "$proxy_health" \
  --argjson voicebox "$voicebox_health" \
  --argjson control "$control_residency" \
  '{runtimeVersion: $runtimeVersion, runtimeContract: $runtimeContract, controlBaseUrl: $controlBaseUrl, chatmlBaseUrl: $chatmlBaseUrl, voiceboxBaseUrl: $voiceboxBaseUrl, proxy: $proxy, voicebox: $voicebox, control: $control}'
