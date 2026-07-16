#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PERSONAPLEX_PORT:-8998}"
HOST="${PERSONAPLEX_HOST:-0.0.0.0}"
LOCAL_URL="${PERSONAPLEX_LOCAL_URL:-http://localhost:$PORT}"
CLOUDFLARED_DIR="${PERSONAPLEX_CLOUDFLARED_DIR:-$ROOT_DIR/.cache/bin}"
CLOUDFLARED_BIN="${CLOUDFLARED_BIN:-}"
SERVER_PID_FILE="${PERSONAPLEX_PID_FILE:-$ROOT_DIR/server_nf4.pid}"
SERVER_WAIT_SECONDS="${PERSONAPLEX_SERVER_WAIT_SECONDS:-180}"

log() {
  printf '[personaplex-deploy] %s\n' "$*" >&2
}

fail() {
  printf '[personaplex-deploy] ERROR: %s\n' "$*" >&2
  exit 1
}

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
  local deadline now
  deadline=$((SECONDS + SERVER_WAIT_SECONDS))
  while (( SECONDS < deadline )); do
    if curl -fsS -I "$LOCAL_URL" >/dev/null 2>&1 || curl -fsS "$LOCAL_URL" >/dev/null 2>&1; then
      log "local server is responding at $LOCAL_URL"
      return
    fi
    sleep 2
  done
  fail "local server did not respond at $LOCAL_URL within ${SERVER_WAIT_SECONDS}s"
}

start_server() {
  if [[ -f "$SERVER_PID_FILE" ]] && kill -0 "$(cat "$SERVER_PID_FILE")" >/dev/null 2>&1; then
    log "server already running with pid $(cat "$SERVER_PID_FILE")"
    return
  fi

  log "starting PersonaPlex server on $LOCAL_URL"
  PERSONAPLEX_BACKGROUND=1 \
  PERSONAPLEX_PORT="$PORT" \
  PERSONAPLEX_HOST="$HOST" \
    "$ROOT_DIR/scripts/start_nf4_server.sh"
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

  log "starting Cloudflare quick tunnel for $LOCAL_URL"
  log "leave this process running; stop with Ctrl+C"
  exec "$cloudflared" tunnel --url "$LOCAL_URL"
}

main "$@"
