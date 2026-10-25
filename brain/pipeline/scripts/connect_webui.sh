#!/usr/bin/env bash
set -euo pipefail
# Run on the laptop. The destination must be the exact host running the UI.
CSCS_USER="${1:?usage: connect_webui.sh <CSCS user> <host> [local port]}"
CSCS_NODE="${2:?specify the exact login or allocated compute node}"
LOCAL_PORT="${3:-8040}"
exec ssh -J "${CSCS_USER}@ela.cscs.ch" \
  -o ExitOnForwardFailure=yes -N \
  -L "${LOCAL_PORT}:127.0.0.1:8040" "${CSCS_USER}@${CSCS_NODE}"
