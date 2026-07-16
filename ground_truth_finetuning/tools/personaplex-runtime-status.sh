#!/usr/bin/env bash
set -euo pipefail

readonly RUNTIME_ENV="/srv/voxrn_cache/personaplex-systemd/personaplex-runtime.env"
[[ -r "$RUNTIME_ENV" ]] || { printf 'PersonaPlex runtime contract missing: %s\n' "$RUNTIME_ENV" >&2; exit 78; }
source "$RUNTIME_ENV"

semantic_lanes='[]'
voicebox_lanes='[]'
for lane in 0 1 2; do
  control_var="PERSONAPLEX_CONTROL_LANE${lane}_PORT"
  chatml_var="PERSONAPLEX_CHATML_LANE${lane}_PORT"
  semantic_gpu_var="PERSONAPLEX_CONTROL_LANE${lane}_GPU"
  voicebox_port_var="PERSONAPLEX_VOICEBOX_LANE${lane}_PORT"
  voicebox_gpu_var="PERSONAPLEX_VOICEBOX_LANE${lane}_GPU"
  control_port="${!control_var}"
  chatml_port="${!chatml_var}"
  semantic_gpu="${!semantic_gpu_var}"
  voicebox_port="${!voicebox_port_var}"
  voicebox_gpu="${!voicebox_gpu_var}"
  control_url="http://${PERSONAPLEX_BIND_HOST}:${control_port}"
  chatml_url="http://${PERSONAPLEX_BIND_HOST}:${chatml_port}"
  voicebox_url="http://${PERSONAPLEX_BIND_HOST}:${voicebox_port}"
  proxy_health="$(curl -fsS --max-time 5 "${chatml_url}/health" || printf 'null')"
  control_residency="$(curl -fsS --max-time 5 "${control_url}/api/ps" || printf 'null')"
  voicebox_health="$(curl -fsS --max-time 5 "${voicebox_url}/health" || printf 'null')"
  semantic_lanes="$(jq -cn --argjson lanes "$semantic_lanes" --arg lane "$lane" --arg gpu "$semantic_gpu" --arg controlUrl "$control_url" --arg chatmlUrl "$chatml_url" --argjson proxy "$proxy_health" --argjson control "$control_residency" '$lanes + [{lane: ($lane | tonumber), gpu: ($gpu | tonumber), controlBaseUrl: $controlUrl, chatmlBaseUrl: $chatmlUrl, proxy: $proxy, control: $control}]')"
  voicebox_lanes="$(jq -cn --argjson lanes "$voicebox_lanes" --arg lane "$lane" --arg gpu "$voicebox_gpu" --arg voiceboxUrl "$voicebox_url" --argjson voicebox "$voicebox_health" '$lanes + [{lane: ($lane | tonumber), gpu: ($gpu | tonumber), voiceboxBaseUrl: $voiceboxUrl, voicebox: $voicebox}]')"
done

jq -n \
  --arg runtimeVersion "$PERSONAPLEX_RUNTIME_VERSION" \
  --arg runtimeContract "$RUNTIME_ENV" \
  --arg synthesisPlan "$PERSONAPLEX_SYNTHESIS_PLAN_PATH" \
  --argjson semanticLanes "$semantic_lanes" \
  --argjson voiceboxLanes "$voicebox_lanes" \
  '{runtimeVersion: $runtimeVersion, runtimeContract: $runtimeContract, synthesisPlan: $synthesisPlan, semanticLanes: $semanticLanes, voiceboxLanes: $voiceboxLanes}'
