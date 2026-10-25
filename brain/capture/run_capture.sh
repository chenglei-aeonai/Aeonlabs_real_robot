#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/../../calibration/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo 'Run ./calibration/setup.sh first.' >&2
    exit 1
fi
exec "$PY" "$HERE/capture_hand.py" "$@"
