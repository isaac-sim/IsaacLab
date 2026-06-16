#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import json
import random
import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).resolve().parent.parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from backend_identity import split_backend_key  # noqa: E402
from launch_config import hydra_args_for_task  # noqa: E402
from task_config import TaskConfig  # noqa: E402


def _presets_for_backend(task_id: str, identity) -> str:
    task = TaskConfig(
        task_id=task_id,
        physics_backend=identity.physics_backend,
        render_backend=identity.render_backend,
        preset="default",
        num_envs=1,
        num_frames=1,
        excluded_frames_raw=[],
        camera_resolution=None,
        timeout_minutes=1,
        fps_mean_floor={},
        caches=[],
    )
    args = hydra_args_for_task(task)
    if not args:
        return "default"
    return args[0].split("=", 1)[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id")
    parser.add_argument("--backend")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--fps_mean", type=float, default=200.0)
    parser.add_argument("--failure_phase", default="none")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Simulate failures
    if args.failure_phase == "import":
        # Emit a fake traceback and exit non-zero without writing perf file
        print("Traceback (most recent call last):")
        print('  File "<string>", line 1, in <module>')
        print("ImportError: simulated import failure")
        sys.exit(1)

    if args.failure_phase == "init":
        # Emit AppLauncher init message then exit non-zero (no perf file)
        print("AppLauncher initialization complete")
        sys.exit(2)

    # Prepare FPS series
    n = int(args.num_frames)
    rng = random.Random(0)
    noise = 5.0
    fps_series = [max(0.0, rng.gauss(args.fps_mean, noise)) for _ in range(n)]

    # Write perf_regression_gate_info.json
    identity = split_backend_key(args.backend)
    if identity is None:
        raise RuntimeError(f"Cannot parse backend identity from {args.backend!r}")
    benchmark_info_phase = {
        "phase_name": "benchmark_info",
        "metadata": [
            {"name": "stub benchmark_info task", "data": args.task_id},
            {"name": "stub benchmark_info num_envs", "data": args.num_envs},
            {"name": "stub benchmark_info num_frames", "data": args.num_frames},
            {"name": "stub benchmark_info seed", "data": args.seed},
            {"name": "stub benchmark_info physics_backend", "data": identity.physics_backend},
            {"name": "stub benchmark_info render_backend", "data": identity.render_backend},
            {"name": "stub benchmark_info backend_key", "data": identity.backend_key},
            {"name": "stub benchmark_info presets", "data": _presets_for_backend(args.task_id, identity)},
        ],
    }

    runtime_phase = {
        "phase_name": "runtime",
        "measurements": [
            {
                "name": "Step Frametimes",
                "value": {
                    "Environment step effective FPS": fps_series,
                    "Environment step times": [1000.0 / max(fps, 1.0) for fps in fps_series],
                },
            }
        ],
    }
    info_path = out_dir / "perf_regression_gate_info.json"
    with info_path.open("w") as fh:
        json.dump([benchmark_info_phase, runtime_phase], fh)

    # Print Step Frametimes marker so classify_failure_phase can see it
    print("Step Frametimes")

    # For runtime failure emulate crash after printing frame times
    if args.failure_phase == "runtime":
        print("RuntimeError: simulated crash during runtime")
        sys.exit(3)

    # Successful exit
    sys.exit(0)


if __name__ == "__main__":
    main()
