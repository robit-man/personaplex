#!/usr/bin/env bash
set -euo pipefail

exec "$(dirname "$0")/run-personaplex-chatml-proxy-worker-service.sh" 1
