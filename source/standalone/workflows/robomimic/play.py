# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a trained policy from robomimic."""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Pytorch model checkpoint to load.")
args_cli = parser.parse_args()

# launch the simulator
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)

"""Rest everything follows."""


import gym
import torch

import robomimic  # noqa: F401
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.utils import parse_env_cfg


def main():
    """Run a trained policy from robomimic with Isaac Orbit environment."""
    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, use_gpu=not args_cli.cpu, num_envs=1)
    # modify configuration
    env_cfg.control.control_type = "inverse_kinematics"
    env_cfg.control.inverse_kinematics.command_type = "pose_rel"
    env_cfg.terminations.episode_timeout = False
    env_cfg.terminations.is_success = True
    env_cfg.observations.return_dict_obs_in_group = True

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless)

    # acquire device
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    # restore policy
    policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=args_cli.checkpoint, device=device, verbose=True)

    # reset environment
    obs_dict = env.reset()
    # robomimic only cares about policy observations
    obs = obs_dict["policy"]
    # simulate environment
    while simulation_app.is_running():
        # compute actions
        actions = policy(obs)
        actions = torch.from_numpy(actions).to(device=device).view(1, env.action_space.shape[0])
        # apply actions
        obs_dict, _, _, _ = env.step(actions)
        # check if simulator is stopped
        if env.unwrapped.sim.is_stopped():
            break
        # robomimic only cares about policy observations
        obs = obs_dict["policy"]

    # close the simulator
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
