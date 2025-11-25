# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run a random MP agent using the shared MP registry (default: box-pushing)."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Random MP agent for Isaac Lab environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--base_id",
    type=str,
    default="Isaac-Box-Pushing-Dense-step-Franka-v0",
    help="Base step-based Gym env id to upgrade with MP.",
)
parser.add_argument("--mp_type", type=str, default="ProDMP", choices=["ProDMP", "ProMP", "DMP"], help="MP backend.")
parser.add_argument(
    "--mp_id",
    type=str,
    default=None,
    help="Optional MP env id to register; defaults to derived name for box pushing.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app early to ensure omni modules are available
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.box_pushing.mp_wrapper import BoxPushingMPWrapper
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_tasks.utils.mp import upgrade


def _disable_command_debug_vis(env_cfg):
    """Turn off command debug visuals to silence USD instancer warnings."""
    try:
        commands = getattr(env_cfg, "commands", None)
        if commands and hasattr(commands, "object_pose"):
            commands.object_pose.debug_vis = False
    except Exception:
        pass


def main():
    """Random MP actions on a registered MP environment."""
    # Derive default MP id for box pushing; otherwise require explicit mp_id.
    mp_id = args_cli.mp_id
    if mp_id is None and "Box-Pushing" in args_cli.base_id:
        parts = args_cli.base_id.split("/")
        base_name = parts[-1] if len(parts) > 1 else args_cli.base_id
        mp_id = f"Isaac_MP/{base_name.replace('Isaac-', '')}-{args_cli.mp_type}"
    elif mp_id is None:
        raise ValueError("Please provide --mp_id for non-box-pushing tasks.")

    env_id = upgrade(
        mp_id=mp_id,
        base_id=args_cli.base_id,
        mp_wrapper_cls=BoxPushingMPWrapper,
        mp_type=args_cli.mp_type,
        device=args_cli.device,
    )

    env_cfg = parse_env_cfg(
        args_cli.base_id, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    _disable_command_debug_vis(env_cfg)
    env = gym.make(env_id, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    obs, _ = env.reset()
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.randn(env.action_space.shape, device=env.unwrapped.device)
            obs, rew, term, trunc, info = env.step(actions)
            if term.any() or trunc.any():
                env.reset()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
