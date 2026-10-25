# Brain

`capture/` contains the camera/robot capture tools, MANO conversion, tests,
reference diagrams, and saved results. See [capture/README.md](capture/README.md).

From the repository root, capture with:

```bash
./brain/capture/run_capture.sh
```

Outputs are saved under `brain/capture/results/<UTC timestamp>/`.

`pipeline/` contains the extracted RobotRetargeting PR #1 inference pipeline,
web UI, and retargeting dependencies. See [pipeline/README.md](pipeline/README.md).
Start it with `./brain/pipeline/run_webui.sh --port 8040`.

The retargeting and supervised real-robot execution path is documented in
[pipeline/docs/physical_replay.md](pipeline/docs/physical_replay.md). Its source
snapshot was imported from `chenglei-aeonai/aeon-brain` commit
`ee62c0532e6110761970594de8928375de60e5ed`; this repository also supplies the
controller, hand bridge and collision-model dependencies used for execution.
