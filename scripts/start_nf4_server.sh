#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PERSONAPLEX_VENV:-$ROOT_DIR/.venv-jetson}"
MODEL_DIR="${PERSONAPLEX_MODEL_DIR:-$ROOT_DIR/models/cudabenchmarktest/personaplex-7b-nf4}"
RUNTIME_DIR="${PERSONAPLEX_RUNTIME_DIR:-$ROOT_DIR/personaplex-setup/moshi}"
PORT="${PERSONAPLEX_PORT:-8998}"
HOST="${PERSONAPLEX_HOST:-0.0.0.0}"
DEVICE="${PERSONAPLEX_DEVICE:-cuda}"
LOG_FILE="${PERSONAPLEX_LOG_FILE:-$ROOT_DIR/server_nf4.log}"
PID_FILE="${PERSONAPLEX_PID_FILE:-$ROOT_DIR/server_nf4.pid}"

fail() {
  printf '[personaplex-server] ERROR: %s\n' "$*" >&2
  exit 1
}

[[ -x "$VENV_DIR/bin/python" ]] || fail "venv missing at $VENV_DIR; run scripts/setup_jetson_nf4.sh"
[[ -f "$RUNTIME_DIR/moshi/server.py" ]] || fail "fork runtime missing at $RUNTIME_DIR. Restore this fork's personaplex-setup gitlink/source; this script will not fetch upstream source."
[[ -s "$MODEL_DIR/model-nf4.safetensors" ]] || fail "missing $MODEL_DIR/model-nf4.safetensors; run scripts/setup_jetson_nf4.sh"
[[ -s "$MODEL_DIR/tokenizer-e351c8d8-checkpoint125.safetensors" ]] || fail "missing Mimi tokenizer in $MODEL_DIR"
[[ -s "$MODEL_DIR/tokenizer_spm_32k_3.model" ]] || fail "missing text tokenizer in $MODEL_DIR"

export PYTHONPATH="$ROOT_DIR:$RUNTIME_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

args=(
  -m moshi.server
  --host "$HOST"
  --port "$PORT"
  --moshi-weight "$MODEL_DIR/model-nf4.safetensors"
  --mimi-weight "$MODEL_DIR/tokenizer-e351c8d8-checkpoint125.safetensors"
  --tokenizer "$MODEL_DIR/tokenizer_spm_32k_3.model"
  --device "$DEVICE"
)

if [[ "${PERSONAPLEX_CPU_MIMI:-0}" == "1" ]]; then
  args+=(--cpu-mimi)
fi

if [[ -n "${PERSONAPLEX_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args=($PERSONAPLEX_EXTRA_ARGS)
  args+=("${extra_args[@]}")
fi

if [[ "${PERSONAPLEX_BACKGROUND:-0}" == "1" ]]; then
  printf '[personaplex-server] starting in background on http://%s:%s\n' "$HOST" "$PORT"
  nohup "$VENV_DIR/bin/python" "${args[@]}" >"$LOG_FILE" 2>&1 &
  echo "$!" >"$PID_FILE"
  printf '[personaplex-server] pid: %s\n' "$(cat "$PID_FILE")"
  printf '[personaplex-server] log: %s\n' "$LOG_FILE"
else
  printf '[personaplex-server] starting on http://%s:%s\n' "$HOST" "$PORT"
  exec "$VENV_DIR/bin/python" "${args[@]}"
fi
