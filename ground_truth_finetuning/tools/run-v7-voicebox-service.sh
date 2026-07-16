#!/usr/bin/env bash
set -euo pipefail

exec "$(dirname "$0")/run-v7-voicebox-worker-service.sh" 2
