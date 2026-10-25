"""Install the narrow H3 extension into an explicitly selected experimental checkout.

Checks every edit before writing anything. Does not change a deployed service,
checkpoint, launch configuration or training source unless that path is supplied.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def patched_source(source: str) -> str:
    changes = [
        ('    app = FastAPI(title="h3-hands-v1 TI2V server")',
         '    app = FastAPI(title="h3-hands-v1 TI2V server")\n'
         '    from h3_joint.robot_custom_frames import install_routes\n'
         '    install_routes(app, engine)'),
        ('        out = {"ready": bool(engine.ready), "model": engine.model_name,',
         '        out = {"robot_custom_frames_v1": True,\n'
         '               "ready": bool(engine.ready), "model": engine.model_name,'),
        ('        t0 = time.perf_counter()\n        if native_frames is not None:',
         '        t0 = time.perf_counter()\n'
         '        if cid.startswith("robot::"):\n'
         '            if native_frames is not None or n_windows != 1:\n'
         '                raise ValueError("robot frames require the native 107-frame recipe")\n'
         '            from h3_joint.robot_custom_frames import conditioning_items\n'
         '            item_cond, item_uncond = conditioning_items(self, cid, conditions[0].prompt)\n'
         '            t_encode = time.perf_counter() - t0\n'
         '        elif native_frames is not None:'),
    ]
    for before, after in changes:
        if source.count(before) != 1:
            raise ValueError(f"H3 runtime version mismatch/already patched: {before!r}")
        source = source.replace(before, after, 1)
    compile(source, "serve_hands.py", "exec")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h3_worktree", type=Path)
    args = parser.parse_args()
    package = args.h3_worktree / "training/h3_joint"
    target = package / "serve_hands.py"
    source = patched_source(target.read_text())
    helper = package / "robot_custom_frames.py"
    if helper.exists():
        raise ValueError(f"refusing to overwrite {helper}")
    shutil.copyfile(Path(__file__).with_name("robot_custom_frames.py"), helper)
    target.write_text(source)
    print(f"Installed into {package}; review git diff before starting the experimental service.")


if __name__ == "__main__":
    main()
