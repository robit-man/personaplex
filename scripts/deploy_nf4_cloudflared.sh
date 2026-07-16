#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PERSONAPLEX_VENV:-$ROOT_DIR/.venv-jetson}"
PORT="${PERSONAPLEX_PORT:-8998}"
HOST="${PERSONAPLEX_HOST:-0.0.0.0}"
LOCAL_URL="${PERSONAPLEX_LOCAL_URL:-http://localhost:$PORT}"
CLOUDFLARED_DIR="${PERSONAPLEX_CLOUDFLARED_DIR:-$ROOT_DIR/.cache/bin}"
CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-}"
SERVER_PID_FILE="${PERSONAPLEX_PID_FILE:-$ROOT_DIR/server_nf4.pid}"
SERVER_LOG_FILE="${PERSONAPLEX_LOG_FILE:-$ROOT_DIR/server_nf4.log}"
SERVER_WAIT_SECONDS="${PERSONAPLEX_SERVER_WAIT_SECONDS:-180}"
TUNNEL_PID_FILE="${PERSONAPLEX_TUNNEL_PID_FILE:-$ROOT_DIR/tunnel_nf4.pid}"
TUNNEL_LOG_FILE="${PERSONAPLEX_TUNNEL_LOG_FILE:-$ROOT_DIR/tunnel_nf4.log}"
TUNNEL_WAIT_SECONDS="${PERSONAPLEX_TUNNEL_WAIT_SECONDS:-90}"
MONITOR_REFRESH_SECONDS="${PERSONAPLEX_MONITOR_REFRESH_SECONDS:-2}"
CLEANUP_ON_EXIT="${PERSONAPLEX_CLEANUP_ON_EXIT:-1}"

SERVER_PID=""
SERVER_STARTED=0
TUNNEL_PID=""
TUNNEL_URL=""
TUI_ACTIVE=0

log() {
  printf '[personaplex-deploy] %s\n' "$*" >&2
}

fail() {
  printf '[personaplex-deploy] ERROR: %s\n' "$*" >&2
  exit 1
}

pid_is_running() {
  local pid="${1:-}"
  [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1
}

read_pid_file() {
  local path="$1"
  [[ -f "$path" ]] || return 1
  tr -cd '0-9' <"$path"
}

terminate_pid() {
  local pid="$1"
  local label="$2"
  local deadline

  pid_is_running "$pid" || return 0
  log "stopping $label pid $pid"
  kill "$pid" >/dev/null 2>&1 || true
  deadline=$((SECONDS + 20))
  while (( SECONDS < deadline )); do
    pid_is_running "$pid" || return 0
    sleep 1
  done
  log "$label pid $pid did not exit after SIGTERM; sending SIGKILL"
  kill -9 "$pid" >/dev/null 2>&1 || true
}

cleanup() {
  local exit_code="${1:-$?}"
  trap - EXIT INT TERM
  set +e

  if [[ "$TUI_ACTIVE" == "1" ]]; then
    tput cnorm 2>/dev/null || true
    tput rmcup 2>/dev/null || true
    TUI_ACTIVE=0
  fi

  if [[ -n "$TUNNEL_PID" ]]; then
    terminate_pid "$TUNNEL_PID" "Cloudflare tunnel"
    rm -f "$TUNNEL_PID_FILE"
  fi

  if [[ "$CLEANUP_ON_EXIT" == "1" && -n "$SERVER_PID" ]]; then
    if [[ "$SERVER_STARTED" == "1" || "${PERSONAPLEX_CLEANUP_EXISTING:-0}" == "1" ]]; then
      terminate_pid "$SERVER_PID" "PersonaPlex server"
      rm -f "$SERVER_PID_FILE"
      log "PersonaPlex server exited; GPU VRAM held by that process is released"
    else
      log "leaving pre-existing PersonaPlex server pid $SERVER_PID running"
    fi
  fi

  exit "$exit_code"
}

trap 'cleanup 130' INT
trap 'cleanup 143' TERM
trap 'cleanup "$?"' EXIT

ensure_cloudflared() {
  if [[ -n "$CLOUDFLARED_BIN" ]]; then
    [[ -x "$CLOUDFLARED_BIN" ]] || fail "CLOUDFLARED_BIN is not executable: $CLOUDFLARED_BIN"
    printf '%s\n' "$CLOUDFLARED_BIN"
    return
  fi

  if command -v cloudflared >/dev/null 2>&1; then
    command -v cloudflared
    return
  fi

  local arch suffix url target
  arch="$(uname -m)"
  case "$arch" in
    aarch64|arm64) suffix="arm64" ;;
    x86_64|amd64) suffix="amd64" ;;
    *) fail "unsupported architecture for automatic cloudflared download: $arch" ;;
  esac

  mkdir -p "$CLOUDFLARED_DIR"
  target="$CLOUDFLARED_DIR/cloudflared"
  url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-$suffix"

  if [[ ! -x "$target" ]]; then
    log "cloudflared not found; downloading $url"
    curl -L --fail --retry 5 --retry-all-errors --retry-delay 2 "$url" -o "$target"
    chmod +x "$target"
  fi

  printf '%s\n' "$target"
}

wait_for_server() {
  local deadline
  deadline=$((SECONDS + SERVER_WAIT_SECONDS))
  while (( SECONDS < deadline )); do
    if curl -fsS -I "$LOCAL_URL" >/dev/null 2>&1 || curl -fsS "$LOCAL_URL" >/dev/null 2>&1; then
      log "local server is responding at $LOCAL_URL"
      return
    fi
    if "$VENV_DIR/bin/python" - "$LOCAL_URL" <<'PY' >/dev/null 2>&1
import socket
import sys
from urllib.parse import urlparse

url = urlparse(sys.argv[1])
port = url.port or (443 if url.scheme == "https" else 80)
with socket.create_connection((url.hostname, port), timeout=1):
    pass
PY
    then
      log "local server port is open at $LOCAL_URL"
      return
    fi
    if [[ -f "$SERVER_PID_FILE" ]] && ! kill -0 "$(cat "$SERVER_PID_FILE")" >/dev/null 2>&1; then
      if [[ -f "$SERVER_LOG_FILE" ]]; then
        log "server exited early; last log lines follow"
        tail -80 "$SERVER_LOG_FILE" >&2
      fi
      fail "local server process exited before $LOCAL_URL responded"
    fi
    sleep 2
  done
  fail "local server did not respond at $LOCAL_URL within ${SERVER_WAIT_SECONDS}s"
}

start_server() {
  local existing_pid
  existing_pid="$(read_pid_file "$SERVER_PID_FILE" || true)"
  if pid_is_running "$existing_pid"; then
    SERVER_PID="$existing_pid"
    SERVER_STARTED=0
    log "server already running with pid $SERVER_PID"
    return
  fi
  rm -f "$SERVER_PID_FILE"

  log "starting PersonaPlex server on $LOCAL_URL"
  PERSONAPLEX_BACKGROUND=1 \
  PERSONAPLEX_PORT="$PORT" \
  PERSONAPLEX_HOST="$HOST" \
  PERSONAPLEX_PID_FILE="$SERVER_PID_FILE" \
  PERSONAPLEX_LOG_FILE="$SERVER_LOG_FILE" \
    "$ROOT_DIR/scripts/start_nf4_server.sh"

  SERVER_PID="$(read_pid_file "$SERVER_PID_FILE" || true)"
  [[ -n "$SERVER_PID" ]] || fail "server did not write pid file: $SERVER_PID_FILE"
  SERVER_STARTED=1
}

start_cloudflare_tunnel() {
  local cloudflared="$1"

  : >"$TUNNEL_LOG_FILE"
  log "starting Cloudflare quick tunnel for $LOCAL_URL"
  "$cloudflared" tunnel --url "$LOCAL_URL" --no-autoupdate >"$TUNNEL_LOG_FILE" 2>&1 &
  TUNNEL_PID="$!"
  printf '%s\n' "$TUNNEL_PID" >"$TUNNEL_PID_FILE"
}

wait_for_tunnel_url() {
  local deadline
  deadline=$((SECONDS + TUNNEL_WAIT_SECONDS))
  while (( SECONDS < deadline )); do
    TUNNEL_URL="$(grep -Eo 'https://[[:alnum:]-]+\.trycloudflare\.com' "$TUNNEL_LOG_FILE" 2>/dev/null | tail -1 || true)"
    if [[ -n "$TUNNEL_URL" ]]; then
      log "Cloudflare tunnel URL: $TUNNEL_URL"
      return
    fi
    if [[ -n "$TUNNEL_PID" ]] && ! pid_is_running "$TUNNEL_PID"; then
      if [[ -f "$TUNNEL_LOG_FILE" ]]; then
        log "cloudflared exited early; last log lines follow"
        tail -80 "$TUNNEL_LOG_FILE" >&2
      fi
      fail "Cloudflare tunnel exited before a trycloudflare.com URL was generated"
    fi
    sleep 1
  done
  fail "Cloudflare tunnel did not generate a URL within ${TUNNEL_WAIT_SECONDS}s"
}

local_status() {
  if ss -ltn 2>/dev/null | grep -qE "[[:space:]](127\\.0\\.0\\.1|0\\.0\\.0\\.0|\\[::\\]):$PORT[[:space:]]"; then
    printf 'listening'
  else
    printf 'pending'
  fi
}

tunnel_status() {
  if pid_is_running "$TUNNEL_PID"; then
    printf 'connected'
  else
    printf 'pending'
  fi
}

gpu_status() {
  local jtop_sample smi sample
  if command -v python3 >/dev/null 2>&1; then
    jtop_sample="$(python3 - <<'PY' 2>/dev/null || true
from jtop import jtop

with jtop(interval=0.2) as jetson:
    if not jetson.ok():
        raise SystemExit(1)
    stats = jetson.stats
    memory = jetson.memory
    ram = memory.get("RAM", {})
    ram_used = float(ram.get("used", 0)) / 1024 / 1024
    ram_total = float(ram.get("tot", 0)) / 1024 / 1024
    gpu = stats.get("GPU", "n/a")
    temp = stats.get("Temp gpu", "n/a")
    power = stats.get("Power VDD_GPU_SOC", stats.get("Power TOT", "n/a"))
    if isinstance(power, (int, float)):
        power = f"{power / 1000:.1f}W"
    if isinstance(temp, (int, float)):
        temp = f"{temp:.1f}C"
    if isinstance(gpu, (int, float)):
        gpu = f"{gpu:.1f}%"
    print(f"jtop GPU {gpu} temp {temp} RAM {ram_used:.1f}/{ram_total:.1f}GiB power {power}")
PY
)"
    if [[ -n "$jtop_sample" ]]; then
      printf '%s' "$jtop_sample"
      return
    fi
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    smi="$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
    if [[ -n "$smi" && "$smi" != *"[N/A]"* ]]; then
      printf 'nvidia-smi memory/utilization: %s' "$smi"
      return
    fi
  fi

  if command -v tegrastats >/dev/null 2>&1; then
    sample="$(timeout 2 tegrastats --interval 1000 2>/dev/null | head -1 || true)"
    if [[ -n "$sample" ]]; then
      printf '%s' "$sample"
      return
    fi
  fi

  printf 'GPU telemetry unavailable'
}

print_tail() {
  local file="$1"
  local lines="$2"
  local width="$3"

  if [[ -s "$file" ]]; then
    tail -n "$lines" "$file" | sed -E 's/\x1B\[[0-9;]*[A-Za-z]//g' | cut -c "1-$width"
  else
    printf '(no log output yet)\n'
  fi
}

print_tunnel_tail() {
  local width="$1"

  if [[ -s "$TUNNEL_LOG_FILE" ]]; then
    tail -n 80 "$TUNNEL_LOG_FILE" \
      | grep -Ev 'Incoming request ended abruptly: context canceled|Request failed error="Incoming request ended abruptly' \
      | tail -n 6 \
      | sed -E 's/\x1B\[[0-9;]*[A-Za-z]//g' \
      | cut -c "1-$width"
  else
    printf '(no log output yet)\n'
  fi
}

render_monitor() {
  local cols local_state tunnel_state server_state cloudflared_state
  cols="$(tput cols 2>/dev/null || printf '100')"
  (( cols < 60 )) && cols=60

  local_state="$(local_status)"
  tunnel_state="$(tunnel_status)"
  server_state="stopped"
  cloudflared_state="stopped"
  pid_is_running "$SERVER_PID" && server_state="running"
  pid_is_running "$TUNNEL_PID" && cloudflared_state="running"

  tput clear 2>/dev/null || printf '\033[H\033[2J'
  printf 'PersonaPlex NF4 Monitor\n'
  printf '%*s\n' "$cols" '' | tr ' ' '-'
  printf 'Local     %s  [%s]\n' "$LOCAL_URL" "$local_state"
  printf 'Tunnel    %s  [%s]\n' "$TUNNEL_URL" "$tunnel_state"
  printf 'Server    pid %s  [%s]\n' "${SERVER_PID:-unknown}" "$server_state"
  printf 'Tunnel    pid %s  [%s]\n' "${TUNNEL_PID:-unknown}" "$cloudflared_state"
  printf 'GPU       %s\n' "$(gpu_status)" | cut -c "1-$cols"
  printf '\nServer log: %s\n' "$SERVER_LOG_FILE"
  print_tail "$SERVER_LOG_FILE" 8 "$cols"
  printf '\nTunnel log: %s (client-cancel noise hidden)\n' "$TUNNEL_LOG_FILE"
  print_tunnel_tail "$cols"
  printf '\nCtrl+C stops the tunnel and the PersonaPlex server started by this script.\n'
}

monitor_loop() {
  if [[ "${PERSONAPLEX_TUI:-1}" == "1" && -t 1 && "${TERM:-}" != "dumb" ]]; then
    tput smcup 2>/dev/null || true
    tput civis 2>/dev/null || true
    TUI_ACTIVE=1
    while true; do
      render_monitor
      pid_is_running "$TUNNEL_PID" || fail "Cloudflare tunnel process stopped"
      pid_is_running "$SERVER_PID" || fail "PersonaPlex server process stopped"
      sleep "$MONITOR_REFRESH_SECONDS"
    done
  fi

  log "Cloudflare tunnel URL: $TUNNEL_URL"
  log "monitor disabled because stdout is not an interactive terminal"
  log "leave this process running; stop with Ctrl+C"
  while true; do
    pid_is_running "$TUNNEL_PID" || fail "Cloudflare tunnel process stopped"
    pid_is_running "$SERVER_PID" || fail "PersonaPlex server process stopped"
    sleep "$MONITOR_REFRESH_SECONDS"
  done
}

main() {
  cd "$ROOT_DIR"

  if [[ "${PERSONAPLEX_SKIP_SETUP:-0}" != "1" ]]; then
    log "running Jetson NF4 setup"
    "$ROOT_DIR/scripts/setup_jetson_nf4.sh"
  else
    log "skipping setup because PERSONAPLEX_SKIP_SETUP=1"
  fi

  local cloudflared
  cloudflared="$(ensure_cloudflared)"

  start_server
  wait_for_server
  start_cloudflare_tunnel "$cloudflared"
  wait_for_tunnel_url
  monitor_loop
}

main "$@"
