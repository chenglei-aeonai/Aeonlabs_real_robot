#!/usr/bin/env python3
"""Submit to the same backend as the browser. Python standard library only."""
import argparse
import base64
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8040")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--hfov", type=float, required=True)
    camera = parser.add_mutually_exclusive_group(required=True)
    camera.add_argument("--fixed-camera-pitch", type=float)
    camera.add_argument("--camera-transforms", type=Path, help="JSON camera-to-demo matrix or matrices")
    parser.add_argument("--robot-from-demo", type=Path, help="JSON 4x4 placement")
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--edited-image", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    body = {"instruction": args.instruction, "hfov": args.hfov,
            "fixed_camera_pitch": args.fixed_camera_pitch, "seed": args.seed}
    for path, key in ((args.image, "image_b64"), (args.mask, "mask_b64"),
                      (args.edited_image, "edited_image_b64")):
        if path:
            body[key] = base64.b64encode(path.read_bytes()).decode()
    for path, key in ((args.camera_transforms, "camera_to_demo"), (args.robot_from_demo, "robot_from_demo")):
        if path:
            body[key] = json.loads(path.read_text())
    request = Request(args.url.rstrip("/") + "/api/jobs", data=json.dumps(body).encode(),
                      headers={"Content-Type": "application/json", "X-Retarget-Client": "1"})
    with urlopen(request, timeout=60) as response:
        job = json.load(response)
    (args.out / "submission.json").write_text(json.dumps(job, indent=2))
    print("Submitted:", job["id"], flush=True)
    deadline = time.monotonic() + 3700
    previous = None
    while time.monotonic() < deadline:
        with urlopen(args.url.rstrip("/") + "/api/jobs/" + job["id"], timeout=30) as response:
            job = json.load(response)
        stage = (job["status"], job.get("progress", {}).get("stage"))
        if stage != previous:
            print(*stage, flush=True)
            previous = stage
        if job["status"] == "completed":
            for name, route in job["artifacts"].items():
                if Path(name).name != name:
                    raise ValueError("Invalid artifact filename")
                with urlopen(args.url.rstrip("/") + route, timeout=180) as response:
                    part = args.out / (name + ".part")
                    with part.open("wb") as output:
                        while chunk := response.read(1024 * 1024):
                            output.write(chunk)
                    part.replace(args.out / name)
            print("Saved:", args.out.resolve())
            return
        if job["status"] in ("failed", "interrupted"):
            raise RuntimeError(job.get("error", job["status"]))
        time.sleep(2)
    raise TimeoutError("Job still pending; use submission.json to retrieve it later.")


if __name__ == "__main__":
    main()
