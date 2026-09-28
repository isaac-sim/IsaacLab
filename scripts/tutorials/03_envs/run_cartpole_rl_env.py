# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to run the RL environment for the cartpole balancing task.

.. code-block:: bash

    uv run python scripts/tutorials/03_envs/run_cartpole_rl_env.py --num_envs 32

Trailing ``key=value`` arguments (e.g. ``physics=isaacsim_physx``) are forwarded as Hydra-style
overrides to the task configuration; see :func:`~isaaclab_tasks.utils.parse_env_cfg`.

"""

"""Parse the command-line arguments first."""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.utils import instantiate

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on running the cartpole RL environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append simulation launcher cli args
add_launcher_args(parser)
# tutorials should open Kit visualizer by default
parser.set_defaults(visualizer=["kit"])
# parse the arguments, forwarding unrecognized ones as Hydra-style task config overrides
args_cli, hydra_overrides = parser.parse_known_args()

"""Rest everything follows."""

import torch

from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Main function."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        "Isaac-Cartpole", device=args_cli.device, num_envs=args_cli.num_envs, overrides=hydra_overrides
    )
    # Launch the simulator runtime that the configuration needs
    with launch_simulation(env_cfg, args_cli):
        # setup RL environment
        env = instantiate(env_cfg)

        # simulate physics
        count = 0
        while env.sim.is_running():
            with torch.inference_mode():
                # reset
                if count % 300 == 0:
                    count = 0
                    env.reset()
                    print("-" * 80)
                    print("[INFO]: Resetting environment...")
                # sample random actions
                joint_efforts = torch.randn_like(env.action_manager.action)
                # step the environment
                obs, rew, terminated, truncated, info = env.step(joint_efforts)
                # print current orientation of pole
                print("[Env 0]: Pole joint: ", obs["policy"][0][1].item())
                # update counter
                count += 1

        # close the environment
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
