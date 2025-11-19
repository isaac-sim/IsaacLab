# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--output_dir", type=str, default=None, help="Path to the output directory.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=True)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    outs = env.unwrapped.get_IO_descriptors
    out_observations = outs["observations"]
    out_actions = outs["actions"]
    out_articulations = outs["articulations"]
    out_scene = outs["scene"]
    # Make a yaml file with the output
    import yaml

    name = args_cli.task.lower().replace("-", "_")
    name = name.replace(" ", "_")

    if not os.path.exists(args_cli.output_dir):
        os.makedirs(args_cli.output_dir)

    with open(os.path.join(args_cli.output_dir, f"{name}_IO_descriptors.yaml"), "w") as f:
        print(f"[INFO]: Exporting IO descriptors to {os.path.join(args_cli.output_dir, f'{name}_IO_descriptors.yaml')}")
        yaml.safe_dump(outs, f)

    for k in out_actions:
        print(f"--- Action term: {k['name']} ---")
        k.pop("name")
        for k1, v1 in k.items():
            print(f"{k1}: {v1}")

    for obs_group_name, obs_group in out_observations.items():
        print(f"--- Obs group: {obs_group_name} ---")
        for k in obs_group:
            print(f"--- Obs term: {k['name']} ---")
            k.pop("name")
            for k1, v1 in k.items():
                print(f"{k1}: {v1}")

    for articulation_name, articulation_data in out_articulations.items():
        print(f"--- Articulation: {articulation_name} ---")
        for k1, v1 in articulation_data.items():
            print(f"{k1}: {v1}")

    for k1, v1 in out_scene.items():
        print(f"{k1}: {v1}")

    env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
