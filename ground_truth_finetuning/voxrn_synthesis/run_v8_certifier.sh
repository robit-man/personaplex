#!/usr/bin/env bash
set -euo pipefail

: "${VORYN_CHECKOUT:?Set VORYN_CHECKOUT to the Voryn checkout.}"
: "${SYNTHESIS_LANE_INDEX:?Set the logical synthesis lane.}"

script="${VORYN_CHECKOUT}/scripts/certify-personaplex-v7-paired-queue.js"
[[ -f "${script}" ]] || { echo "Missing Voryn certification runner: ${script}" >&2; exit 2; }
exec node "${script}" "--lane=${SYNTHESIS_LANE_INDEX}"
