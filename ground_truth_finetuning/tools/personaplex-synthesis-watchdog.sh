#!/usr/bin/env bash
set -euo pipefail

# Conservative local watchdog for the CUDA-only synthesis lanes. It repairs
# dead services, but does not terminate a running lane merely because a long
# controlled conversation has not yet emitted a bundle.

readonly RUNTIME_ENV="/srv/voxrn_cache/personaplex-systemd/personaplex-runtime.env"
readonly LOG_DIR="/srv/voxrn_cache/personaplex-systemd"
readonly LOG_FILE="${LOG_DIR}/personaplex-synthesis-watchdog.log"
readonly STALE_SECONDS="${PERSONAPLEX_SYNTHESIS_STALE_SECONDS:-1800}"
readonly HOST_MEMORY_MAX_PERCENT="${PERSONAPLEX_SYNTHESIS_HOST_MEMORY_MAX_PERCENT:-80}"
readonly HOST_MEMORY_RESUME_PERCENT="${PERSONAPLEX_SYNTHESIS_HOST_MEMORY_RESUME_PERCENT:-75}"
readonly MEMORY_PRESSURE_FILE="${LOG_DIR}/personaplex-synthesis-host-memory-pressure"

mkdir -p "$LOG_DIR"
[[ -r "$RUNTIME_ENV" ]] || { printf '%s event=runtime_contract_missing path=%s\n' "$(date -Is)" "$RUNTIME_ENV" >>"$LOG_FILE"; exit 78; }
source "$RUNTIME_ENV"

log() {
  printf '%s %s\n' "$(date -Is)" "$*" >>"$LOG_FILE"
}

host_memory_percent() {
  awk '/MemTotal:/ { total=$2 } /MemAvailable:/ { available=$2 } END { if (total > 0) printf "%.2f", (total - available) * 100 / total; else exit 1 }' /proc/meminfo
}

host_memory_percent_now="$(host_memory_percent)"
if awk -v used="$host_memory_percent_now" -v max="$HOST_MEMORY_MAX_PERCENT" 'BEGIN { exit !(used >= max) }'; then
  printf '%s\n' "$host_memory_percent_now" >"$MEMORY_PRESSURE_FILE"
  log "event=host_memory_backpressure used_percent=$host_memory_percent_now max_percent=$HOST_MEMORY_MAX_PERCENT action=skip_restarts"
  exit 0
fi
if [[ -e "$MEMORY_PRESSURE_FILE" ]] && awk -v used="$host_memory_percent_now" -v resume="$HOST_MEMORY_RESUME_PERCENT" 'BEGIN { exit !(used <= resume) }'; then
  rm -f "$MEMORY_PRESSURE_FILE"
  log "event=host_memory_backpressure_released used_percent=$host_memory_percent_now resume_percent=$HOST_MEMORY_RESUME_PERCENT"
fi

unit_running() {
  [[ "$(systemctl --user is-active "$1" 2>/dev/null || true)" == "active" ]]
}

unit_is_installed() {
  [[ "$(systemctl --user show "$1" -p LoadState --value 2>/dev/null || true)" == "loaded" ]]
}

restart_unit() {
  local unit="$1"
  if unit_is_installed "$unit"; then
    systemctl --user restart "$unit"
  else
    log "event=unit_missing unit=$unit"
  fi
}

restart_if_dead() {
  local unit="$1"
  unit_is_installed "$unit" || { log "event=unit_missing unit=$unit"; return; }
  if unit_running "$unit"; then
    return
  fi
  restart_unit "$unit"
  log "event=unit_restarted unit=$unit reason=inactive"
}

port_for_lane() {
  case "$1:$2" in
    0:voicebox) printf '%s' "$PERSONAPLEX_VOICEBOX_LANE0_PORT" ;;
    1:voicebox) printf '%s' "$PERSONAPLEX_VOICEBOX_LANE1_PORT" ;;
    2:voicebox) printf '%s' "$PERSONAPLEX_VOICEBOX_LANE2_PORT" ;;
    0:semantic) printf '%s' "$PERSONAPLEX_CHATML_LANE0_PORT" ;;
    1:semantic) printf '%s' "$PERSONAPLEX_CHATML_LANE1_PORT" ;;
    2:semantic) printf '%s' "$PERSONAPLEX_CHATML_LANE2_PORT" ;;
  esac
}

for lane in 0 1 2; do
  voicebox_port="$(port_for_lane "$lane" voicebox)"
  semantic_port="$(port_for_lane "$lane" semantic)"
  restart_if_dead "personaplex-v7-lane@${lane}.service"
  restart_if_dead "personaplex-ornith-worker@${lane}.service"
  restart_if_dead "personaplex-ornith-chatml-proxy-worker@${lane}.service"
  restart_if_dead "personaplex-v7-voicebox-worker@${lane}.service"

  if ! curl -fsS --max-time 5 "http://${PERSONAPLEX_BIND_HOST}:${semantic_port}/health" >/dev/null; then
    restart_unit "personaplex-ornith-worker@${lane}.service"
    restart_unit "personaplex-ornith-chatml-proxy-worker@${lane}.service"
    log "event=semantic_restarted lane=$lane reason=healthcheck_failed"
  fi
  if ! curl -fsS --max-time 5 "http://${PERSONAPLEX_BIND_HOST}:${voicebox_port}/health" >/dev/null; then
    restart_unit "personaplex-v7-voicebox-worker@${lane}.service"
    log "event=voicebox_restarted lane=$lane reason=healthcheck_failed"
  fi

  progress="/srv/voxrn_cache/personaplex-lanes/gpu${lane}/personaplex-v7-paired-lane-${lane}.v11-repairable-v8.progress.v1.json"
  activity="/srv/voxrn_cache/personaplex-lanes/gpu${lane}/logs/personaplex-v7-paired-lane-${lane}.jsonl"
  admitted=0
  unresolved=0
  replacement=0
  if [[ -r "$progress" ]]; then
    read -r admitted unresolved replacement < <(jq -r '[(.admitted|length), (.unresolved|length), (.replacementRequired|length)] | @tsv' "$progress")
  fi
  if [[ -e "$activity" ]]; then
    last_activity="$(stat -c %Y "$activity")"
    age=$(( $(date +%s) - last_activity ))
    if (( age > STALE_SECONDS )); then
      log "event=stale_progress lane=$lane age_seconds=$age admitted=$admitted unresolved=$unresolved replacement_required=$replacement action=preserved_for_diagnosis"
    else
      log "event=lane_healthy lane=$lane activity_age_seconds=$age admitted=$admitted unresolved=$unresolved replacement_required=$replacement"
    fi
  else
    log "event=activity_log_missing lane=$lane admitted=$admitted unresolved=$unresolved replacement_required=$replacement"
  fi
done
