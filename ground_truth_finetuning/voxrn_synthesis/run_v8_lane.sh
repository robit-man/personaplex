#!/usr/bin/env bash
set -euo pipefail

: "${VORYN_CHECKOUT:?Set VORYN_CHECKOUT to the Voryn checkout.}"
: "${CUDA_VISIBLE_DEVICES:?Set exactly one allowed physical CUDA device.}"

case "${CUDA_VISIBLE_DEVICES}" in
  0|1|2) ;;
  *) echo "PersonaPlex synthesis permits only one physical CUDA device: 0, 1, or 2." >&2; exit 2 ;;
esac

script="${VORYN_CHECKOUT}/scripts/run-personaplex-v7-paired-lane.js"
[[ -f "${script}" ]] || { echo "Missing Voryn lane runner: ${script}" >&2; exit 2; }
exec node "${script}"
