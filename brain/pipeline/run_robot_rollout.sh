#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OPENBLAS_NUM_THREADS="${RETARGET_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${RETARGET_NUM_THREADS:-1}"
exec "$HERE/hand_retarget/.venv/bin/python" -m hand_retarget.robot_rollout "$@"
