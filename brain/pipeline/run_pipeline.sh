#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/hand_retarget/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo 'Run ./brain/pipeline/setup.sh first.' >&2
    exit 1
fi
export OPENBLAS_NUM_THREADS="${RETARGET_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${RETARGET_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${RETARGET_NUM_THREADS:-1}"
exec "$PY" -m hand_retarget.inference "$@"
