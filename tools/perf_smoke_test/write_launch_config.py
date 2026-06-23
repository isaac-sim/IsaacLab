# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Write launch_config.json for a perf-smoke task/backend job"""

import argparse
import sys
from pathlib import Path

_MODULE_DIR = Path(__file__).parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from backend_identity import make_backend_key, normalize_render_backend
from gpu_identity import gpu_model_config_keys
from launch_config import hydra_args_for_task, task_to_launch_config, write_launch_config
from task_config import get_task


def _parse_args():
    parser = argparse.ArgumentParser(description="Write launch_config.json for a perf-smoke benchmark job")
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--physics_backend", required=True)
    parser.add_argument("--render_backend", default="")
    parser.add_argument("--gpu_model", default="L40S")
    parser.add_argument("--artifact_dir", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    render_backend = normalize_render_backend(args.render_backend)
    backend_key = make_backend_key(args.physics_backend, render_backend)
    task = get_task(args.task_id, backend_key)
    fps_mean_floor = 0.0
    for gpu_key in gpu_model_config_keys(args.gpu_model):
        configured_floor = task.fps_mean_floor.get(gpu_key, {}).get(task.backend_key)
        if configured_floor is not None:
            fps_mean_floor = float(configured_floor)
            break
    config = task_to_launch_config(
        task,
        fps_mean_floor=fps_mean_floor,
        gpu_model=args.gpu_model,
        hydra_args=hydra_args_for_task(task),
    )
    path = write_launch_config(args.artifact_dir, config)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
