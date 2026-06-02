# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates the visualizer tiled camera panel.

.. code-block:: bash

    # Kit visualizer tiled camera panel
    ./isaaclab.sh -p scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py \
        --task Isaac-Velocity-Rough-Anymal-D-Play-v0 --num_envs 16 --viz kit

    # Newton visualizer tiled camera panel
    ./isaaclab.sh -p scripts/tutorials/07_visualizers/run_tiled_camera_visualizer.py \
        --task Isaac-Dexsuite-Kuka-Allegro-Lift-Play-v0 --num_envs 16 --viz newton \
        presets=duo_camera,rgb128,newton_renderer

"""

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
    fold_preset_tokens,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

KIT_DEFAULT_TASK = "Isaac-Velocity-Rough-Anymal-D-Play-v0"
NEWTON_DEFAULT_TASK = "Isaac-Dexsuite-Kuka-Allegro-Lift-Play-v0"
SUPPORTED_TILED_VISUALIZERS = {"kit", "newton"}
UNSUPPORTED_TILED_VISUALIZERS = {"rerun", "viser"}


def _requested_visualizers(args_cli: argparse.Namespace) -> list[str]:
    """Return requested visualizers, defaulting to Kit for this tutorial."""
    visualizers = args_cli.visualizer or ["kit"]
    return [str(visualizer).lower() for visualizer in visualizers]


def _configure_visualizers(env_cfg, args_cli: argparse.Namespace) -> None:
    """Attach tiled camera visualizer configs to the environment simulation config."""
    from isaaclab_visualizers.kit import KitVisualizerCfg
    from isaaclab_visualizers.newton import NewtonVisualizerCfg

    visualizers = _requested_visualizers(args_cli)
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

    args_cli.visualizer = visualizers
    visualizer_cfgs = []
    for visualizer in visualizers:
        if visualizer == "kit":
            # Generated Kit cameras are not found by the task config scan.
            args_cli.enable_cameras = True
            visualizer_cfg = KitVisualizerCfg()

            # Edit these VisualizerCfg fields to customize the generated tiled cameras.
            visualizer_cfg.tiled_cam_view = True
            visualizer_cfg.tiled_cam_num = 64
            visualizer_cfg.tiled_cam_prim_path = None
            visualizer_cfg.tiled_cam_eye = (2.5, -3.0, 1.6)
            visualizer_cfg.tiled_cam_target_prim_path = "/World/envs/*/Robot/base"
        else:
            visualizer_cfg = NewtonVisualizerCfg()

            # Edit these VisualizerCfg fields to display a different existing camera.
            visualizer_cfg.tiled_cam_view = True
            visualizer_cfg.tiled_cam_num = 64
            visualizer_cfg.tiled_cam_prim_path = None
            visualizer_cfg.tiled_cam_eye = (2.5, -3.0, 1.6)
            visualizer_cfg.tiled_cam_target_prim_path = "/World/envs/*/Robot/ee_link/palm_link"

        visualizer_cfgs.append(visualizer_cfg)

    env_cfg.sim.visualizer_cfgs = visualizer_cfgs

    # TODO: Temporary workaround for Dexsuite duo_camera. The nested wrist camera
    # path currently matches both the cloned Robot subtree and the cloned Camera
    # template, so Camera construction fails before the visualizer initializes.
    if "newton" in visualizers and getattr(getattr(env_cfg, "scene", None), "wrist_camera", None) is not None:
        env_cfg.scene.wrist_camera = None
        if hasattr(getattr(env_cfg, "observations", None), "wrist_image"):
            env_cfg.observations.wrist_image = None


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
sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)


def main():
    """Run a random-action environment with a tiled camera visualizer."""
    torch.manual_seed(42)
    # TODO: Temporary workaround for Anymal-D ActuatorNetLSTM reset failures
    # with CUDNN_STATUS_NOT_INITIALIZED in this demo path.
    torch.backends.cudnn.enabled = False

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
        while True:
            if sim.visualizers and not any(v.is_running() and not v.is_closed for v in sim.visualizers):
                break
            with torch.inference_mode():
                actions = 2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1
                env.step(actions)

        env.close()


if __name__ == "__main__":
    main()
