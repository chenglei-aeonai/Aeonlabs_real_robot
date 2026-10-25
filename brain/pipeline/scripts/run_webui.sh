#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
if [[ -n "${RETARGET_PYTHON:-}" ]]; then
  exec "$RETARGET_PYTHON" -m hand_retarget.webapp "$@"
fi
exec uv run --project "$REPO_DIR/hand_retarget" h3-retarget-web "$@"
