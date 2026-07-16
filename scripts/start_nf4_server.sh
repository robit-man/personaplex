#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PERSONAPLEX_VENV:-$ROOT_DIR/.venv-jetson}"
MODEL_DIR="${PERSONAPLEX_MODEL_DIR:-$ROOT_DIR/models/cudabenchmarktest/personaplex-7b-nf4}"
RUNTIME_DIR="${PERSONAPLEX_RUNTIME_DIR:-$ROOT_DIR/personaplex-setup/moshi}"
BF16_MODEL_PATH="${PERSONAPLEX_BF16_MODEL_PATH:-$ROOT_DIR/.cache/personaplex/model-bf16.safetensors}"
BF16_LOCK_FILE="${PERSONAPLEX_BF16_LOCK_FILE:-$ROOT_DIR/.cache/personaplex/model-bf16.lock}"
DEQUANT_SCRIPT="$ROOT_DIR/scripts/dequant_nf4_to_bf16.py"
SERVER_CONFIG_PATH="${PERSONAPLEX_SERVER_CONFIG_PATH:-$ROOT_DIR/.cache/personaplex/moshi-legacy-config.json}"
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
[[ -s "$MODEL_DIR/model-nf4.safetensors" ]] || fail "missing $MODEL_DIR/model-nf4.safetensors; run scripts/setup_jetson_nf4.sh"
[[ -s "$MODEL_DIR/tokenizer-e351c8d8-checkpoint125.safetensors" ]] || fail "missing Mimi tokenizer in $MODEL_DIR"
[[ -s "$MODEL_DIR/tokenizer_spm_32k_3.model" ]] || fail "missing text tokenizer in $MODEL_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

ensure_packaged_runtime() {
  "$VENV_DIR/bin/python" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("moshi.server") is not None else 1)
PY
}

write_packaged_config() {
  mkdir -p "$(dirname "$SERVER_CONFIG_PATH")"
  "$VENV_DIR/bin/python" - "$SERVER_CONFIG_PATH" <<'PY'
import json
import sys
from pathlib import Path
from moshi.models import loaders

config = dict(loaders._lm_kwargs)
config["model_type"] = "moshi"
config["n_q"] = 16
config["dep_q"] = 16
Path(sys.argv[1]).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
PY
}

default_dequant_device() {
  "$VENV_DIR/bin/python" - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
}

prepare_bf16_model() {
  [[ -s "$DEQUANT_SCRIPT" ]] || fail "missing $DEQUANT_SCRIPT"
  mkdir -p "$(dirname "$BF16_MODEL_PATH")"
  local dequant_device
  dequant_device="${PERSONAPLEX_DEQUANT_DEVICE:-$(default_dequant_device)}"

  printf '[personaplex-server] fork runtime missing at %s; using packaged moshi with cached bf16 weights\n' "$RUNTIME_DIR" >&2
  printf '[personaplex-server] bf16 cache: %s\n' "$BF16_MODEL_PATH" >&2
  printf '[personaplex-server] dequant device: %s\n' "$dequant_device" >&2

  exec 9>"$BF16_LOCK_FILE"
  printf '[personaplex-server] waiting for bf16 cache lock: %s\n' "$BF16_LOCK_FILE" >&2
  flock 9

  if "$VENV_DIR/bin/python" "$DEQUANT_SCRIPT" --check --output "$BF16_MODEL_PATH" \
      && [[ "$BF16_MODEL_PATH" -nt "$MODEL_DIR/model-nf4.safetensors" ]]; then
    printf '[personaplex-server] using existing bf16 cache\n' >&2
    return
  fi

  "$VENV_DIR/bin/python" "$DEQUANT_SCRIPT" \
    --input "$MODEL_DIR/model-nf4.safetensors" \
    --output "$BF16_MODEL_PATH" \
    --device "$dequant_device"
  "$VENV_DIR/bin/python" "$DEQUANT_SCRIPT" --check --output "$BF16_MODEL_PATH" || fail "bf16 cache was not created at $BF16_MODEL_PATH"
}

server_supports_arg() {
  "$VENV_DIR/bin/python" -m moshi.server --help 2>&1 | grep -q -- "$1"
}

if [[ -f "$RUNTIME_DIR/moshi/server.py" ]]; then
  SERVER_RUNTIME="fork"
  MOSHI_WEIGHT="$MODEL_DIR/model-nf4.safetensors"
  export PYTHONPATH="$ROOT_DIR:$RUNTIME_DIR:${PYTHONPATH:-}"
  printf '[personaplex-server] using fork runtime at %s\n' "$RUNTIME_DIR" >&2
else
  ensure_packaged_runtime || fail "fork runtime missing at $RUNTIME_DIR and packaged moshi.server is not installed; run scripts/setup_jetson_nf4.sh"
  SERVER_RUNTIME="packaged"
  export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
  export NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}"
  export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
  prepare_bf16_model
  write_packaged_config
  MOSHI_WEIGHT="$BF16_MODEL_PATH"
fi

args=(
  -m moshi.server
  --host "$HOST"
  --port "$PORT"
  --moshi-weight "$MOSHI_WEIGHT"
  --mimi-weight "$MODEL_DIR/tokenizer-e351c8d8-checkpoint125.safetensors"
  --tokenizer "$MODEL_DIR/tokenizer_spm_32k_3.model"
  --device "$DEVICE"
)

if [[ "$SERVER_RUNTIME" == "packaged" ]]; then
  args+=(--config-path "$SERVER_CONFIG_PATH")
fi

if [[ -n "${PERSONAPLEX_STATIC:-}" ]]; then
  args+=(--static "$PERSONAPLEX_STATIC")
fi

if [[ "${PERSONAPLEX_CPU_MIMI:-0}" == "1" ]]; then
  if server_supports_arg "--cpu-mimi"; then
    args+=(--cpu-mimi)
  else
    printf '[personaplex-server] PERSONAPLEX_CPU_MIMI=1 ignored; selected runtime does not support --cpu-mimi\n' >&2
  fi
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
