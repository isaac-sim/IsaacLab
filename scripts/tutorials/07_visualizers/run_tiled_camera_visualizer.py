# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the visualizer tiled camera panel.

.. code-block:: bash

    # Kit visualizer tiled camera panel
    uv run python scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py \
 --task Isaac-Velocity-Rough-AnymalD --num_envs 256 --viz kit

    # Newton visualizer tiled camera panel
    uv run python scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py \
        --task IsaacContrib-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor --num_envs 25 --viz newton

"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys

import gymnasium as gym
import torch
from isaaclab_newton.renderers import NewtonWarpRendererCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from isaaclab_visualizers.kit import KitVisualizerCfg
from isaaclab_visualizers.newton import NewtonGLVisualizerCfg

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.sensors import CameraCfg
from isaaclab.sim import PinholeCameraCfg
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
from isaaclab.visualizers import SceneCameraCfg

from isaaclab_tasks.utils import resolve_task_config, setup_preset_cli

logger = logging.getLogger(__name__)

KIT_DEFAULT_TASK = "Isaac-Velocity-Rough-AnymalD"
NEWTON_DEFAULT_TASK = "IsaacContrib-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor"


def _configure_visualizers(env_cfg, args_cli: argparse.Namespace) -> None:
    """Declare a scene camera before launch; every viewer borrows the same sensor."""
    visualizers = args_cli.visualizer
    if any(kind not in ("kit", "newton_gl") for kind in visualizers):
        raise ValueError("This tutorial supports --viz kit or --viz newton_gl.")
    camera_cfg = next((cfg for cfg in vars(env_cfg.scene).values() if isinstance(cfg, CameraCfg)), None)
    if camera_cfg is None:
        eye, target = torch.tensor([[3.0, 3.0, 3.0]]), torch.zeros((1, 3))
        rotation = quat_from_matrix(create_rotation_matrix_from_view(eye, target, device="cpu"))[0]
        # Attaching the sensor to the base gives it the robot's pose through the normal camera view.
        camera_cfg = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/StreamingCamera",
            width=320,
            height=240,
            data_types=["rgb"],
            spawn=PinholeCameraCfg(focal_length=24.0, clipping_range=(0.1, 1.0e5)),
            offset=CameraCfg.OffsetCfg(pos=(3.0, 3.0, 3.0), rot=tuple(rotation.tolist()), convention="opengl"),
            renderer_cfg=IsaacRtxRendererCfg() if "kit" in visualizers else NewtonWarpRendererCfg(),
        )
        env_cfg.scene.streaming_camera = camera_cfg
    env_cfg.sim.visualizer_cfgs = [
        (KitVisualizerCfg if kind == "kit" else NewtonGLVisualizerCfg)(
            streaming_view=True,
            streaming_envs=36 if kind == "kit" else 12,
            cameras=[SceneCameraCfg(prim_path=camera_cfg.prim_path)],
        )
        for kind in visualizers
    ]


def _resolve_task(args_cli: argparse.Namespace) -> str:
    """Resolve the task for the selected visualizer."""
    if args_cli.task is not None:
        return args_cli.task
    if "newton_gl" in args_cli.visualizer:
        return NEWTON_DEFAULT_TASK
    return KIT_DEFAULT_TASK


# add argparse arguments
parser = argparse.ArgumentParser(description="Showcase the Kit/Newton visualizer tiled camera panel.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append simulation launcher cli args
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli, hydra_args = setup_preset_cli(parser)
args_cli.task = _resolve_task(args_cli)
sys.argv = [sys.argv[0]] + hydra_args


def main():
    """Run a random-action environment with a tiled camera visualizer."""
    # parse configuration via Hydra (supports preset selection, e.g. presets=newton_mjwarp)
    env_cfg, _ = resolve_task_config(args_cli.task, "")
    _configure_visualizers(env_cfg, args_cli)

    with launch_simulation(env_cfg, args_cli):
        # override with CLI arguments
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # create environment
        env = gym.make(args_cli.task, cfg=env_cfg)

        # print info (this is vectorized environment)
        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")
        env.reset()

        # keep stepping until all visualizer windows have been closed
        sim = env.unwrapped.sim
        if not sim.visualizers:
            logger.warning("No visualizers found. Exiting.")
            env.close()
            return

        while True:
            if sim.visualizers and not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                break
            with torch.inference_mode():
                actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                env.step(actions)

        env.close()


if __name__ == "__main__":
    main()
