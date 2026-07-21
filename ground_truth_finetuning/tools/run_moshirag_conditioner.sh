#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${PERSONAPLEX_RUNTIME_ENV:-/srv/voxrn_cache/personaplex-systemd/personaplex-runtime.env}"
test -r "${ENV_FILE}" || { printf 'runtime source of truth missing: %s\n' "${ENV_FILE}" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

required=(
  PERSONAPLEX_SHARED_CACHE_ROOT
  PERSONAPLEX_MOSHIRAG_SOURCE_ROOT
  PERSONAPLEX_MOSHIRAG_RAG_DIR
  PERSONAPLEX_MOSHIRAG_ARC_DIR
  PERSONAPLEX_MOSHIRAG_CONDITIONER_HOST
  PERSONAPLEX_MOSHIRAG_CONDITIONER_PORT
  PERSONAPLEX_MOSHIRAG_CONDITIONER_GPU
  PERSONAPLEX_MOSHIRAG_RELEASE_REVISION
)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { printf 'required runtime variable missing: %s\n' "${name}" >&2; exit 1; }
done

VENV_ROOT="${PERSONAPLEX_MOSHIRAG_CONDITIONER_VENV:-${PERSONAPLEX_SHARED_CACHE_ROOT}/personaplex/venvs/moshirag-conditioner}"
PYTHON="${VENV_ROOT}/bin/python"
test -x "${PYTHON}" || { printf 'conditioner environment missing; run bootstrap_moshirag_conditioner.sh\n' >&2; exit 1; }
if [[ -z "${PERSONAPLEX_MOSHIRAG_TOKENIZER_DIR:-}" ]]; then
  TOKENIZER_RECORD="${PERSONAPLEX_MOSHIRAG_TOKENIZER_RECORD:-${PERSONAPLEX_SHARED_CACHE_ROOT}/personaplex/imports/moshirag-tokenizer.json}"
  test -r "${TOKENIZER_RECORD}" || { printf 'tokenizer record missing: %s\n' "${TOKENIZER_RECORD}" >&2; exit 1; }
  PERSONAPLEX_MOSHIRAG_TOKENIZER_DIR="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["path"])' "${TOKENIZER_RECORD}")"
fi
test -d "${PERSONAPLEX_MOSHIRAG_TOKENIZER_DIR}" || { printf 'tokenizer directory missing: %s\n' "${PERSONAPLEX_MOSHIRAG_TOKENIZER_DIR}" >&2; exit 1; }

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${PERSONAPLEX_MOSHIRAG_CONDITIONER_GPU}"
export PYTHONUNBUFFERED=1
PACKING_REVISION="${PERSONAPLEX_MOSHIRAG_PACKING_REVISION:-arc4-field-slots-v2}"
exec "${PYTHON}" "${REPO_ROOT}/personaplex_control/moshirag_conditioner_server.py" \
  --source-root "${PERSONAPLEX_MOSHIRAG_SOURCE_ROOT}" \
  --rag-config "${PERSONAPLEX_MOSHIRAG_RAG_DIR}/config.json" \
  --rag-model "${PERSONAPLEX_MOSHIRAG_RAG_DIR}/model.safetensors" \
  --arc-model "${PERSONAPLEX_MOSHIRAG_ARC_DIR}/model.safetensors" \
  --tokenizer "${PERSONAPLEX_MOSHIRAG_TOKENIZER_DIR}" \
  --device cuda:0 \
  --physical-cuda-device "${PERSONAPLEX_MOSHIRAG_CONDITIONER_GPU}" \
  --host "${PERSONAPLEX_MOSHIRAG_CONDITIONER_HOST}" \
  --port "${PERSONAPLEX_MOSHIRAG_CONDITIONER_PORT}" \
  --packing-revision "${PACKING_REVISION}" \
  --release-revision "${PERSONAPLEX_MOSHIRAG_RELEASE_REVISION}"
