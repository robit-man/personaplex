#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PERSONAPLEX_VENV:-$ROOT_DIR/.venv-jetson}"
MODEL_DIR="${PERSONAPLEX_MODEL_DIR:-$ROOT_DIR/models/cudabenchmarktest/personaplex-7b-nf4}"
MODEL_REPO="${PERSONAPLEX_MODEL_REPO:-cudabenchmarktest/personaplex-7b-nf4}"
RUNTIME_DIR="${PERSONAPLEX_RUNTIME_DIR:-$ROOT_DIR/personaplex-setup/moshi}"
PORT="${PERSONAPLEX_PORT:-8998}"
HOST="${PERSONAPLEX_HOST:-0.0.0.0}"
DEVICE="${PERSONAPLEX_DEVICE:-cuda}"
DTYPE="${PERSONAPLEX_NF4_DTYPE:-fp16}"
VOICE_PROMPT_DIR="${PERSONAPLEX_VOICE_PROMPT_DIR:-$MODEL_DIR/voices}"
DEFAULT_STATIC_DIR="${PERSONAPLEX_STATIC_DIR:-$ROOT_DIR/.cache/personaplex/static/dist}"
STATIC="${PERSONAPLEX_STATIC:-}"
LOG_FILE="${PERSONAPLEX_LOG_FILE:-$ROOT_DIR/server_nf4.log}"
PID_FILE="${PERSONAPLEX_PID_FILE:-$ROOT_DIR/server_nf4.pid}"
QUALITY_GATE="${PERSONAPLEX_NF4_QUALITY_GATE:-required}"
QUALITY_REPORT="${PERSONAPLEX_NF4_QUALITY_REPORT:-$ROOT_DIR/.cache/personaplex/nf4-quality-report.json}"

fail() {
  printf '[personaplex-server] ERROR: %s\n' "$*" >&2
  exit 1
}

[[ -x "$VENV_DIR/bin/python" ]] || fail "venv missing at $VENV_DIR; run scripts/setup_jetson_nf4.sh"
[[ -f "$RUNTIME_DIR/moshi/server.py" ]] || fail "vendored runtime missing at $RUNTIME_DIR; BF16 fallback is intentionally disabled"
[[ -f "$ROOT_DIR/personaplex_nf4/direct_nf4.py" ]] || fail "direct NF4 runtime package is missing"
[[ -s "$MODEL_DIR/model-nf4.safetensors" ]] || fail "missing $MODEL_DIR/model-nf4.safetensors; run scripts/setup_jetson_nf4.sh"
[[ -s "$MODEL_DIR/tokenizer-e351c8d8-checkpoint125.safetensors" ]] || fail "missing Mimi tokenizer in $MODEL_DIR"
[[ -s "$MODEL_DIR/tokenizer_spm_32k_3.model" ]] || fail "missing text tokenizer in $MODEL_DIR"
[[ -d "$VOICE_PROMPT_DIR" ]] || fail "missing voice prompt directory: $VOICE_PROMPT_DIR"
[[ "$DTYPE" == "fp16" || "$DTYPE" == "bf16" ]] || fail "PERSONAPLEX_NF4_DTYPE must be fp16 or bf16"
[[ "$QUALITY_GATE" == "required" || "$QUALITY_GATE" == "off" ]] || fail "PERSONAPLEX_NF4_QUALITY_GATE must be required or off"

if [[ -z "$STATIC" ]]; then
  if [[ -s "$DEFAULT_STATIC_DIR/index.html" ]]; then
    STATIC="$DEFAULT_STATIC_DIR"
  else
    cached_static="$(find "${HF_HOME:-$HOME/.cache/huggingface}/hub" -path '*/models--kyutai--moshi-artifacts/*/dist/index.html' -print -quit 2>/dev/null || true)"
    if [[ -n "$cached_static" ]]; then
      STATIC="$(dirname "$cached_static")"
    else
      STATIC="none"
    fi
  fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PERSONAPLEX_NF4_DTYPE="$DTYPE"
export NO_TORCH_COMPILE="${NO_TORCH_COMPILE:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export NO_CUDA_GRAPH="${NO_CUDA_GRAPH:-1}"
export PYTHONPATH="$ROOT_DIR:$RUNTIME_DIR:${PYTHONPATH:-}"

"$VENV_DIR/bin/python" - "$MODEL_DIR/model-nf4.safetensors" <<'PY'
import sys
import torch
from personaplex_nf4.direct_nf4 import verify_nf4_checkpoint

if not torch.cuda.is_available():
    raise SystemExit("direct NF4 requires CUDA; CPU execution is disabled")
verify_nf4_checkpoint(sys.argv[1])
PY

if [[ "$QUALITY_GATE" == "required" ]]; then
  "$VENV_DIR/bin/python" "$ROOT_DIR/scripts/verify_nf4_quality_gate.py" --report "$QUALITY_REPORT"
fi

server_supports_arg() {
  "$VENV_DIR/bin/python" -m moshi.server --help 2>&1 | grep -q -- "$1"
}

args=(
  -m personaplex_nf4.server
  --host "$HOST"
  --port "$PORT"
  --moshi-weight "$MODEL_DIR/model-nf4.safetensors"
  --mimi-weight "$MODEL_DIR/tokenizer-e351c8d8-checkpoint125.safetensors"
  --tokenizer "$MODEL_DIR/tokenizer_spm_32k_3.model"
  --hf-repo "$MODEL_REPO"
  --voice-prompt-dir "$VOICE_PROMPT_DIR"
  --device "$DEVICE"
)

if [[ -n "$STATIC" ]]; then
  args+=(--static "$STATIC")
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
  printf '[personaplex-server] starting packed NF4 runtime in background on http://%s:%s\n' "$HOST" "$PORT"
  nohup "$VENV_DIR/bin/python" "${args[@]}" >"$LOG_FILE" 2>&1 &
  echo "$!" >"$PID_FILE"
  printf '[personaplex-server] pid: %s\n' "$(cat "$PID_FILE")"
  printf '[personaplex-server] log: %s\n' "$LOG_FILE"
else
  printf '[personaplex-server] starting packed NF4 runtime on http://%s:%s\n' "$HOST" "$PORT"
  exec "$VENV_DIR/bin/python" "${args[@]}"
fi
