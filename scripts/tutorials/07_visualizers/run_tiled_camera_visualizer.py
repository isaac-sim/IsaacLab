# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the visualizer tiled camera panel.

.. code-block:: bash

    # Kit visualizer tiled camera panel
    ./isaaclab.sh -p scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py \
        --enable_cameras --task Isaac-Velocity-Rough-Anymal-D-v0 --num_envs 256 --viz kit

    # Newton visualizer tiled camera panel
    ./isaaclab.sh -p scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py \
        --task Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-v0 --num_envs 25 --viz newton

"""

from __future__ import annotations

import argparse
import contextlib
import sys

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

KIT_DEFAULT_TASK = "Isaac-Velocity-Rough-Anymal-D-v0"
NEWTON_DEFAULT_TASK = "Isaac-Stack-Cube-Galbot-Left-Arm-Gripper-Visuomotor-v0"
SUPPORTED_TILED_VISUALIZERS = {"kit", "newton"}
UNSUPPORTED_TILED_VISUALIZERS = {"rerun", "viser"}


def _requested_visualizers(args_cli: argparse.Namespace) -> list[str]:
    """Return requested visualizers, defaulting to Kit for this tutorial."""
    visualizers = args_cli.visualizer or ["kit"]
    visualizers = [str(visualizer).lower() for visualizer in visualizers]

    if "none" in visualizers:
        raise ValueError("This demo requires a tiled-camera visualizer. Use '--viz kit' or '--viz newton'.")
    unsupported = sorted(set(visualizers) & UNSUPPORTED_TILED_VISUALIZERS)
    if unsupported:
        raise ValueError(
            "The visualizer tiled camera panel is only implemented for Kit and Newton. "
            f"Unsupported selection: {unsupported}."
        )
    unknown = sorted(set(visualizers) - SUPPORTED_TILED_VISUALIZERS)
    if unknown:
        raise ValueError(f"Unknown visualizer selection for this demo: {unknown}.")
    return visualizers


def _make_kit_visualizer_cfg(args_cli: argparse.Namespace):
    """Create generated tiled cameras that follow each Anymal-D base."""
    from isaaclab_visualizers.kit import KitVisualizerCfg

    visualizer_cfg = KitVisualizerCfg()
    visualizer_cfg.tiled_cam_view = True
    visualizer_cfg.tiled_cam_num = 36
    visualizer_cfg.tiled_cam_prim_path = None
    # Here is an alternative eye position for a top down view
    # visualizer_cfg.tiled_cam_eye = (0.0, 0.0, 5.0)
    visualizer_cfg.tiled_cam_eye = (3.0, 3.0, 3.0)
    visualizer_cfg.tiled_cam_target_prim_path = "/World/envs/*/Robot/base"
    return visualizer_cfg


def _make_newton_visualizer_cfg():
    """Stream existing Galbot wrist cameras into the Newton tiled camera panel."""
    from isaaclab_visualizers.newton import NewtonVisualizerCfg

    visualizer_cfg = NewtonVisualizerCfg()
    visualizer_cfg.tiled_cam_view = True
    visualizer_cfg.tiled_cam_num = 12
    # Here are other robot mounted camera options for this environment
    # visualizer_cfg.tiled_cam_prim_path = "/World/envs/env_.*/Robot/left_arm_camera_sim_view_frame/left_camera"
    # visualizer_cfg.tiled_cam_prim_path = "/World/envs/env_.*/Robot/right_arm_camera_sim_view_frame/right_camera"
    visualizer_cfg.tiled_cam_prim_path = "/World/envs/env_.*/Robot/head_camera_sim_view_frame/head_camera"
    visualizer_cfg.tiled_cam_eye = None
    visualizer_cfg.tiled_cam_target_prim_path = None
    return visualizer_cfg


def _configure_visualizers(env_cfg, args_cli: argparse.Namespace) -> None:
    """Attach tiled camera visualizer configs to the environment simulation config."""
    visualizers = _requested_visualizers(args_cli)
    args_cli.visualizer = visualizers
    env_cfg.sim.visualizer_cfgs = [
        _make_kit_visualizer_cfg(args_cli) if visualizer == "kit" else _make_newton_visualizer_cfg()
        for visualizer in visualizers
    ]


def _resolve_task(args_cli: argparse.Namespace) -> str:
    """Resolve the task for the selected visualizer."""
    if args_cli.task is not None:
        return args_cli.task
    if "newton" in _requested_visualizers(args_cli):
        return NEWTON_DEFAULT_TASK
    return KIT_DEFAULT_TASK


# add argparse arguments
parser = argparse.ArgumentParser(description="Showcase the Kit/Newton visualizer tiled camera panel.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
add_launcher_args(parser)
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
            print("[WARN]: No visualizers found. Exiting.")
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
