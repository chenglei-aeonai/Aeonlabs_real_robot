#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/hand_retarget/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo 'Run ./brain/pipeline/setup.sh first.' >&2
    exit 1
fi
exec "$PY" -m hand_retarget.image_edit "$@"
