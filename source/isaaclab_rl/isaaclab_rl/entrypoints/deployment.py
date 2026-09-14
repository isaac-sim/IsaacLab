# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in fixed-scene export before training, isolated from the training process."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from time import perf_counter


def export_training_scene(task: str, env_cfg, args_cli) -> Path | None:
    """Write ``deployment.usda`` in this run's log directory from global rank zero.

    A fresh process builds the actual task with one environment and stops before
    event creation. The parent does not instantiate a second scene or touch its RNGs.
    Failures are fatal when explicitly requested; training never silently omits the USD.
    """
    rank_key = "JAX_RANK" if getattr(args_cli, "ml_framework", "torch").startswith("jax") else "RANK"
    rank = int(os.environ.get(rank_key, "0"))
    if rank != 0:
        return None
    if not getattr(env_cfg, "log_dir", None):
        raise ValueError("Deployment export requires the training run log_dir.")
    import gymnasium as gym

    from isaaclab.app import scan

    cfg = deepcopy(env_cfg)
    cfg.scene.num_envs = 1
    # Cameras/sensors remain part of the scene; training video recorders are not.
    cfg.video_recorders = []
    output = Path(env_cfg.log_dir).resolve() / "deployment.usda"
    output.parent.mkdir(parents=True, exist_ok=True)
    launch = {"headless": True, "device": cfg.sim.device, "visualizer": "none"}
    config_scan = scan(cfg, launch)
    bootstrap = {"needs_kit": config_scan.needs_kit, "enable_cameras": config_scan.has_kit_camera, **launch}
    with tempfile.TemporaryDirectory(prefix="isaaclab-deployment-") as folder:
        payload = Path(folder) / "scene.pickle"
        payload.write_bytes(pickle.dumps((gym.spec(task).entry_point, cfg)))
        (Path(folder) / "launch.json").write_text(json.dumps(bootstrap))
        start = perf_counter()
        subprocess.run(
            [sys.executable, "-m", "isaaclab_rl.entrypoints.deployment", str(payload), str(output)], check=True
        )
        elapsed = perf_counter() - start
    report = output.with_suffix(".metrics.json")
    metrics = json.loads(report.read_text())
    metrics["process_wall_seconds"] = elapsed
    report.write_text(json.dumps(metrics, indent=2) + "\n")
    return output


def _worker(payload: Path, output: Path) -> None:
    """Bootstrap native schemas before importing the serialized task configuration."""
    import resource

    launch = json.loads((payload.parent / "launch.json").read_text())
    needs_kit = launch.pop("needs_kit")
    app = None
    if needs_kit:
        from isaaclab.app import AppLauncher

        app = AppLauncher(launch).app
    try:
        import importlib

        from isaaclab.sim import SceneExporter

        entry, cfg = pickle.loads(payload.read_bytes())
        if isinstance(entry, str):
            module, symbol = entry.split(":")
            entry = getattr(importlib.import_module(module), symbol)
        timings = SceneExporter.export_task(lambda: entry(cfg=cfg), str(output))
        metrics = {
            "seconds": timings,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "output_bytes": output.stat().st_size,
        }
        output.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    finally:
        if app is not None:
            app.close()


if __name__ == "__main__":
    _worker(Path(sys.argv[1]), Path(sys.argv[2]))
