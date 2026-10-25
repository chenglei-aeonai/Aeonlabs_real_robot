#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec uv sync --project "$HERE/hand_retarget" --group dev --locked
