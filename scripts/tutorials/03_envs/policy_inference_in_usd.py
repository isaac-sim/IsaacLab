# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates policy inference in a prebuilt USD environment.

In this example, we use a locomotion policy to control the H1 robot. The robot was trained
using Isaac-Velocity-Rough-H1. The robot is commanded to move forward at a constant velocity.

.. code-block:: bash

    # Run the script
    uv run python scripts/tutorials/03_envs/policy_inference_in_usd.py --checkpoint /path/to/jit/checkpoint.pt

"""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on inferencing a policy on an H1 robot in a warehouse.")
parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint exported as jit.", required=True)

add_launcher_args(parser)
# parse the arguments, forwarding unrecognized ones as Hydra-style task config overrides
args_cli, hydra_overrides = parser.parse_known_args()

import os

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, read_file

from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Main function."""
    # load the trained jit policy
    policy_path = os.path.abspath(args_cli.checkpoint)
    file = read_file(policy_path)
    policy = torch.jit.load(file, map_location=args_cli.device)

    # setup environment
    env_cfg = parse_env_cfg("Isaac-Velocity-Rough-H1", device=args_cli.device, num_envs=1, overrides=hydra_overrides)
    env_cfg.play_mode()
    env_cfg.curriculum = None
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
    )
    # The warehouse is enclosed: start the height-scan rays below its roof so they hit the floor.
    env_cfg.scene.height_scanner.offset.pos = (0.0, 0.0, 2.0)
    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False

    with launch_simulation(env_cfg, args_cli):
        # create environment
        env = ManagerBasedRLEnv(cfg=env_cfg)

        # run inference with the policy
        obs, _ = env.reset()
        with torch.inference_mode():
            while env.sim.is_headless_or_exist_active_visualizer():
                action = policy(obs["policy"])
                obs, _, _, _, _ = env.step(action)
        env.close()


if __name__ == "__main__":
    main()
