# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Benchmark Newton visualizer loading: PhysX backend + Newton visualizer.

Single script: flip the flag below to compare original vs cloner path.
Use --task to load a different env (e.g. Isaac-Cartpole-v0, Isaac-Lift-Cube-Franka-v0).

.. code-block:: bash

    ./isaaclab.sh -p scripts/mtrepte/run_newton_vis_loading_benchmark.py \
      --headless --visualizer newton --num_envs 4096
    ./isaaclab.sh -p scripts/mtrepte/run_newton_vis_loading_benchmark.py \
      --headless --visualizer newton --task Isaac-Lift-Cube-Franka-v0 --num_envs 64
"""

# -----------------------------------------------------------------------------
# Flag: set to False for original path (SDP builds Newton from USD), True for
# cloner path (SDP uses prebuilt artifact from clone). Override via --use_prebuilt.
# -----------------------------------------------------------------------------
USE_CLONER_PREBUILT = False

"""Launch Isaac Sim Simulator first."""

import os

# Apply flag before any scene/SDP code runs (env var is read in PhysxSceneDataProvider)
os.environ.setdefault("ISAACLAB_NEWTON_VIS_USE_PREBUILT", "1" if USE_CLONER_PREBUILT else "0")

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Benchmark Newton viz loading (PhysX + Newton visualizer).")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Rough-Anymal-D-v0",
    help="Gym task name (e.g. Isaac-Velocity-Rough-Anymal-D-v0, Isaac-Lift-Cube-Franka-v0).",
)
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments.")
parser.add_argument("--benchmark_steps", type=int, default=2, help="Steps before exit (reset + step to trigger SDP).")
parser.add_argument(
    "--use_prebuilt",
    type=lambda x: x.lower() in ("1", "true", "yes"),
    default=None,
    metavar="BOOL",
    help="Override USE_CLONER_PREBUILT: use prebuilt Newton artifact (default: use script flag).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# CLI override for the flag
if args_cli.use_prebuilt is not None:
    os.environ["ISAACLAB_NEWTON_VIS_USE_PREBUILT"] = "1" if args_cli.use_prebuilt else "0"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401 -- register tasks
from isaaclab_tasks.utils.hydra import resolve_preset_defaults
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device or "cuda:0",
        num_envs=args_cli.num_envs,
    )
    # Resolve all PresetCfgs (e.g. VelocityEnvContactSensorCfg, physics) to concrete configs
    # so InteractiveScene sees known asset types (ContactSensorCfg, PhysxCfg, etc.).
    env_cfg = resolve_preset_defaults(env_cfg)
    # Ensure physics is a single PhysicsCfg (SimulationContext expects class_type)
    if hasattr(env_cfg.sim.physics, "physx"):
        env_cfg.sim.physics = env_cfg.sim.physics.physx
    elif hasattr(env_cfg.sim.physics, "default"):
        env_cfg.sim.physics = env_cfg.sim.physics.default

    env = gym.make(args_cli.task, cfg=env_cfg)

    step_count = 0
    while simulation_app.is_running() and step_count < args_cli.benchmark_steps:
        if step_count == 0:
            env.reset()
        else:
            action = env.action_space.sample()
            action_tensor = action if torch.is_tensor(action) else torch.as_tensor(np.asarray(action))
            env.step(action_tensor)
        step_count += 1

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
