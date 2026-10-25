#!/usr/bin/env bash
#SBATCH --job-name=h3-retarget-web
#SBATCH --account=go091-0
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=h3-retarget-web-%j.log
set -euo pipefail
: "${RETARGET_REPO:?Set RETARGET_REPO to the experimental checkout}"
: "${RETARGET_PYTHON:?Set RETARGET_PYTHON to its Daint/aarch64 Python environment}"
: "${GEMINI_API_KEY:?Configure GEMINI_API_KEY in the submission environment}"
EXPERIMENT="${H3_RETARGET_RUNTIME:-/capstor/store/cscs/2go/go091/aeon-webui/h3-retarget-pipeline-20261020}"
source /capstor/store/cscs/2go/go091/aeon-webui/contact-anchor-13k/code/webui/runtime/contact_store_env.sh
export H3_WEBUI_ROOT=/capstor/store/cscs/2go/go091/aeon-webui/hf15-mirrored-12983/code/webui
export H3_SERVE_PROFILE=fast-align-hf15-mirrored-12983
export H3_SOL_ENABLED=1 H3_CONTACT_STARTUP_FAST=0
export H3_BOOT_STAGE_WORKERS=8 H3_BOOT_STAGE_DIR=/dev/shm
export H3_WEBUI_BASE="${EXPERIMENT}/web-state-${SLURM_JOB_ID}"
export H3_URL=http://127.0.0.1:30121
export UPLIFT_PYTHON="${UPLIFT_PYTHON:-/capstor/store/cscs/2go/go091/aeon-webui/envs/wilor/bin/python}"
export UPLIFT_SCRIPT="${UPLIFT_SCRIPT:-/capstor/store/cscs/2go/go091/aeon-webui/hf15-mirrored-12983/code/webui/runtime/uplift/uplift_photo.py}"
mkdir -p "$H3_WEBUI_BASE"
"$RETARGET_PYTHON" -c 'import hand_retarget.webapp, pinocchio, dex_retargeting'
"$UPLIFT_PYTHON" --version
bash "$EXPERIMENT/launch_server.sh" > "$H3_WEBUI_BASE/h3.log" 2>&1 &
H3_PID=$!
bash "$RETARGET_REPO/scripts/run_webui.sh" --port 8040 \
  --runs "$RETARGET_REPO/out/web-runs" > "$H3_WEBUI_BASE/web.log" 2>&1 &
WEB_PID=$!
trap 'kill "$H3_PID" "$WEB_PID" 2>/dev/null || true; wait "$H3_PID" "$WEB_PID" 2>/dev/null || true' EXIT
echo "Web UI on $(hostname):8040 (loopback); forward this exact host through SSH."
echo "H3 and Web logs: $H3_WEBUI_BASE"
wait -n "$H3_PID" "$WEB_PID"
