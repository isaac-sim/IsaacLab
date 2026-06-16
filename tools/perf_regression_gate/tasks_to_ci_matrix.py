# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert tasks.json into the GitHub Actions bench matrix JSON

Prints a JSON array to stdout, one object per (task_id, backend) combination,
containing the fields consumed by the ``bench`` job matrix in perf-regression-gate.yaml.

Usage::

    python3 tools/perf_regression_gate/tasks_to_ci_matrix.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from launch_config import hydra_args_for_task  # noqa: E402
from task_config import load_tasks  # noqa: E402

tasks = load_tasks()
rows = []
for task in tasks:
    rows.append({
        "task_id": task.task_id,
        "physics_backend": task.physics_backend,
        "render_backend": task.render_backend or "",
        "num_envs": task.num_envs,
        "num_frames": task.num_frames,
        "seed": task.seed if task.seed is not None else "",
        "hydra_args": " ".join(hydra_args_for_task(task)),
        "bench_timeout_s": task.timeout_minutes * 60,
        "job_timeout_minutes": max(30, task.timeout_minutes + 15),
    })

print(json.dumps(rows))
