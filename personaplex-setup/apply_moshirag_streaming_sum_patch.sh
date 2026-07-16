#!/usr/bin/env bash
set -euo pipefail

# Applies the reviewed compatibility patch to an isolated PersonaPlex source
# checkout. It never touches a running deployment and fails if the inspected
# source differs from the expected upstream shape.
SOURCE_ROOT="${1:?usage: apply_moshirag_streaming_sum_patch.sh /absolute/path/to/personaplex/moshi}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="${SCRIPT_DIR}/moshirag_streaming_sum.patch"

if [[ ! -f "${SOURCE_ROOT}/moshi/models/lm.py" ]]; then
  printf 'expected PersonaPlex source at %s/moshi/models/lm.py\n' "${SOURCE_ROOT}" >&2
  exit 2
fi

if rg -q 'def update_streaming_sum_tensors' "${SOURCE_ROOT}/moshi/models/lm.py"; then
  printf 'Moshirag streaming-sum support is already present; no patch applied.\n'
  exit 0
fi

patch --directory="${SOURCE_ROOT}" --strip=1 --forward --batch --input="${PATCH_FILE}"
rg -q 'def update_streaming_sum_tensors' "${SOURCE_ROOT}/moshi/models/lm.py"
rg -q 'streaming_sum: torch.Tensor' "${SOURCE_ROOT}/moshi/models/lm.py"
printf 'Applied Moshirag-compatible streaming-sum support to %s\n' "${SOURCE_ROOT}"
