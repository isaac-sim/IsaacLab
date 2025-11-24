# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Run Isaac Lab and FancyGym box pushing side by side with zero actions."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero-action comparison between Isaac Lab and FancyGym.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations in Isaac Lab.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of Isaac Lab environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Isaac Lab task name.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch

import fancy_gym  # noqa: F401

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Run both simulators with zero actions."""
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_fg = gym.make("fancy/BoxPushingDense-v0", render_mode="rgb_array" if args_cli.headless else "human")

    env_fg.reset(seed=42)
    env.reset()

    joint_positions = []
    joint_positions_fg = []

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions_fg = np.zeros_like(env_fg.action_space.sample())

            obs, _, _, _, _ = env.step(actions)
            joint_positions.append(obs["policy"][0, :7].tolist())

            obs_fg, _, terminated, truncated, _ = env_fg.step(actions_fg)
            joint_positions_fg.append(obs_fg[:7].tolist())

            if not args_cli.headless:
                env_fg.render()

            if terminated or truncated:
                joint_positions = []
                joint_positions_fg = []
                env_fg.reset()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
