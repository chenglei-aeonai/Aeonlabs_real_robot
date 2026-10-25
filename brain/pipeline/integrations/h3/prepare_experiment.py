"""Prepare a patched shared runtime and bounded GPU smoke job, without submitting."""
import argparse
from pathlib import Path
import shlex
import shutil
import subprocess

from install import patched_source

DEPLOY = Path("/capstor/store/cscs/2go/go091/aeon-webui/hf15-mirrored-12983")
SAMPLE = Path("/capstor/store/cscs/2go/go091/aeon-webui/human-env-demo/custom-frames/mcd16.npz")
STORE_ENV = Path("/capstor/store/cscs/2go/go091/aeon-webui/contact-anchor-13k/code/webui/runtime/contact_store_env.sh")


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"launcher version mismatch: {before!r}")
    return text.replace(before, after, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    source = DEPLOY / "code/minimax-h3"
    # Validate both the runtime and launcher before creating a staging tree.
    patched = patched_source((source / "training/h3_joint/serve_hands.py").read_text())
    launcher = (DEPLOY / "code/webui/runtime/serve_hands.shared.sbatch").read_text()
    runtime = out / "minimax-h3"
    launcher = replace_once(launcher, f"export H3_RUNTIME_ROOT={source}",
                            f"export H3_RUNTIME_ROOT={shlex.quote(str(runtime))}")
    launcher = replace_once(launcher, "export PORT=30011", "export PORT=30121")
    launcher = replace_once(launcher, "OUT_DIR=/capstor/store/cscs/2go/go091/aeon-webui/h3-serve-output",
                            f"OUT_DIR={shlex.quote(str(out / 'server_outputs'))}")
    # This smoke needs only the H3 model; stop before the optional refiner block.
    marker = "# ---- prompt refiner / rollout planner on this node's spare GPU"
    if launcher.count(marker) != 1:
        raise ValueError("launcher refiner marker changed")
    launcher = launcher.split(marker)[0] + 'exec "$VENV/bin/python" h3_joint/serve_hands.py --host 127.0.0.1 --port "$PORT"\n'
    out.mkdir(parents=True, exist_ok=False)
    runtime.mkdir()
    # Preserve the source layout and shared assets; copy only code to be patched.
    for child in source.iterdir():
        if child.name != "training":
            (runtime / child.name).symlink_to(child, target_is_directory=child.is_dir())
    (runtime / "training").mkdir()
    for child in (source / "training").iterdir():
        target = runtime / "training" / child.name
        if child.name == "h3_joint":
            shutil.copytree(child, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            target.symlink_to(child, target_is_directory=child.is_dir())
    package = runtime / "training/h3_joint"
    (package / "serve_hands.py").write_text(patched)
    shutil.copyfile(Path(__file__).with_name("robot_custom_frames.py"), package / "robot_custom_frames.py")
    shutil.copyfile(Path(__file__).with_name("smoke_client.py"), out / "smoke_client.py")
    (out / "launch_server.sh").write_text(launcher)
    q = lambda value: shlex.quote(str(value))
    job = f"""#!/bin/bash
#SBATCH --job-name=h3-retarget-smoke
#SBATCH --account=go091-0
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --time=00:45:00
#SBATCH --output={out}/slurm-%j.log
set -euo pipefail
source {q(STORE_ENV)}
export H3_WEBUI_ROOT={q(DEPLOY / 'code/webui')}
export H3_SERVE_PROFILE=fast-align-hf15-mirrored-12983
export H3_SOL_ENABLED=1 H3_CONTACT_STARTUP_FAST=0
export H3_BOOT_STAGE_WORKERS=8 H3_BOOT_STAGE_DIR=/dev/shm
export H3_WEBUI_BASE={q(out / 'state')}
mkdir -p "$H3_WEBUI_BASE"
bash {q(out / 'launch_server.sh')} > {q(out / 'server.log')} 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT
"$H3_TRAIN_VENV/bin/python" {q(out / 'smoke_client.py')} \\
  --bundle {q(SAMPLE)} --out {q(out)}/generation-"$SLURM_JOB_ID"
"""
    (out / "smoke.sbatch").write_text(job)
    subprocess.run(["bash", "-n", str(out / "launch_server.sh")], check=True)
    subprocess.run(["bash", "-n", str(out / "smoke.sbatch")], check=True)
    print(f"Prepared patched runtime and smoke job: {out / 'smoke.sbatch'}")


if __name__ == "__main__":
    main()
