# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert tasks.json into the GitHub Actions bench matrix JSON.

Prints a JSON array to stdout, one object per (task_id, backend) combination,
containing the fields consumed by the ``bench`` job matrix in perf-regression-gate.yaml.

Usage::

    python3 tools/perf_regression_gate/tasks_to_ci_matrix.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from task_config import load_tasks  # noqa: E402

# Physics Hydra group value per backend (keep in sync with local_runner / build_bench_result).
_PHYSICS_TOKEN = {"physx": "physx", "newton": "newton_mjwarp"}

# Cold image build budget [min]: the bench job's GitHub timeout must cover the full
# "Pull CI image" step too, which on a runner without an ECR cache builds the Isaac
# Sim CI image from source (~10 GB base pull + apt/pip layers, observed ~25-35 min
# cold). Warm runs reuse the runner's Docker layer cache and finish in seconds, so
# this only pads the first-ever build per runner.
_IMAGE_BUILD_BUDGET_MIN = 45

tasks = load_tasks()
rows = []
for task in tasks:
    cam_w, cam_h = task.camera_resolution if task.camera_resolution else (0, 0)
    rows.append(
        {
            "task_id": task.task_id,
            "physics_backend": task.physics_backend,
            "render_backend": task.render_backend or "",
            "num_envs": task.num_envs,
            "num_frames": task.num_frames,
            "bench_timeout_s": task.timeout_minutes * 60,
            "job_timeout_minutes": task.timeout_minutes + 15 + _IMAGE_BUILD_BUDGET_MIN,
            # Launch-command fields (consumed verbatim by the workflow so the run's
            # config matches what build_bench_result's drift guard expects).
            "physics": _PHYSICS_TOKEN.get(task.physics_backend, task.physics_backend),
            "render_preset": task.render_backend or "",
            "enable_cameras": bool(task.enable_cameras),
            "camera_width": cam_w,
            "camera_height": cam_h,
            "seed": task.seed if task.seed is not None else "",
        }
    )

print(json.dumps(rows))
