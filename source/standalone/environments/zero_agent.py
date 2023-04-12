# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run an environment with zero action agent."""

"""Launch Isaac Sim Simulator first."""


import argparse
import sys

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
args_cli, remaining = parser.parse_known_args()
# clear out sys.argv
sys.argv = [sys.argv[0]] + remaining
sys.argc = len(sys.argv)

# launch the simulator
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)

"""Rest everything follows."""


import gym
import hydra
import torch
from omegaconf import OmegaConf

import omni.isaac.contrib_envs  # noqa: F401
import omni.isaac.orbit_envs  # noqa: F401
from omni.isaac.orbit_envs.isaac_env_cfg import IsaacEnvCfg
from omni.isaac.orbit_envs.utils.parse_cfg import register_cfg_to_hydra


@hydra.main(config_path=None, config_name=args_cli.task, version_base="1.3")
def main(env_cfg: IsaacEnvCfg):
    """Zero actions agent with Isaac Orbit environment."""
    # get underlying object
    env_cfg = OmegaConf.to_object(env_cfg)
    # modify the environment configuration
    if env_cfg.sim.device == "cpu":
        env_cfg.sim.use_gpu_pipeline = False
        env_cfg.sim.physx.use_gpu = False
    elif "cuda" in env_cfg.sim.device:
        env_cfg.sim.use_gpu_pipeline = True
        env_cfg.sim.physx.use_gpu = True
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg, headless=args_cli.headless)

    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # compute zero actions
        actions = torch.zeros((env.num_envs, env.action_space.shape[0]), device=env.device)
        # apply actions
        _, _, _, _ = env.step(actions)
        # check if simulator is stopped
        if env.unwrapped.sim.is_stopped():
            break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # configure hydra configuration
    register_cfg_to_hydra(args_cli.task)
    # run main function
    main()
