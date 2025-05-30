# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script creates a simple environment with a floating cube. The cube is controlled by a PD
controller to track an arbitrary target position.

While going through this tutorial, we recommend you to pay attention to how a custom action term
is defined. The action term is responsible for processing the raw actions and applying them to the
scene entities.

We also define an event term called 'randomize_scale' that randomizes the scale of
the cube. This event term has the mode 'prestartup', which means that it is applied on the USD stage
before the simulation starts. Additionally, the flag 'replicate_physics' is set to False,
which means that the cube is not replicated across multiple environments but rather each
environment gets its own cube instance.

The rest of the environment is similar to the previous tutorials.

.. code-block:: bash

    # Run the script
    ./isaaclab.sh -p scripts/tutorials/03_envs/create_fumehood_env.py --num_envs 1

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a floating cube environment.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import time
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.fumehood.fumehood_env_cfg import FumehoodEnvCfg
##
# Custom action term
##

def main():
    """Main function."""

    # setup base environment
    env = ManagerBasedRLEnv(cfg=FumehoodEnvCfg())

    # setup target position commands
    target_position = torch.rand(env.num_envs, 3, device=env.device) * 2
    target_position[:, 2] += 2.0
    # offset all targets so that they move to the world origin
    target_position -= env.scene.env_origins

    # simulate physics
    count = 0
    obs, _ = env.reset()

    time.sleep(10)
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                obs, _ = env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # step env
            obs, _ = env.step(target_position)
            # print mean squared position error between target and current position
            error = torch.norm(obs["policy"] - target_position).mean().item()
            print(f"[Step: {count:04d}]: Mean position error: {error:.4f}")
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
