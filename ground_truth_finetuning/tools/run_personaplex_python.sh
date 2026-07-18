#!/usr/bin/env bash
# Run a PersonaPlex handoff stage with the explicitly provisioned interpreter.
set -euo pipefail

python_bin="${PERSONAPLEX_PYTHON:-/usr/bin/python3}"
if [[ ! -x "$python_bin" ]]; then
  printf 'PERSONAPLEX_PYTHON is not executable: %s\n' "$python_bin" >&2
  exit 64
fi

exec "$python_bin" "$@"
