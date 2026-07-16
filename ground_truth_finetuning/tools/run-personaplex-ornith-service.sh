#!/usr/bin/env bash
set -euo pipefail

exec "$(dirname "$0")/run-personaplex-ornith-worker-service.sh" 1
