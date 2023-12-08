# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the environment concept that combines a scene with an action, observation and randomization manager.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the concept of an Environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import math
import torch
import traceback

import carb

import omni.isaac.orbit.envs.mdp as mdp
from omni.isaac.orbit.assets import RigidObject
from omni.isaac.orbit.envs import BaseEnv, BaseEnvCfg
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers.action_manager import ActionTerm, ActionTermCfg
from omni.isaac.orbit.utils import configclass

from omni.isaac.orbit_tasks.classic.cartpole import CartpoleSceneCfg

# Cartpole Action Configuration


class CartpoleActionTerm(ActionTerm):
    _asset: RigidObject
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg, env: BaseEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros(env.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(env.num_envs, 1, device=self.device)

        # gains of controller
        self.p_gain = 1500.0
        self.d_gain = 10.0

        # extract the joint id of the slider_to_cart joint
        joint_ids, _ = self._asset.find_joints(["slider_to_cart", "cart_to_pole"])
        self.slider_to_cart_joint_id = joint_ids[0]
        self.cart_to_pole_joint_id = joint_ids[1]

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions

        joint_pos = (
            self._asset.data.joint_pos[:, self.cart_to_pole_joint_id]
            - self._asset.data.default_joint_pos[:, self.cart_to_pole_joint_id]
        )
        joint_vel = (
            self._asset.data.joint_vel[:, self.cart_to_pole_joint_id]
            - self._asset.data.default_joint_vel[:, self.cart_to_pole_joint_id]
        )

        self._processed_actions[:] = self.p_gain * (actions - joint_pos) - self.d_gain * joint_vel

    def apply_actions(self):
        # set slider joint target
        self._asset.set_joint_effort_target(self.processed_actions, joint_ids=[self.slider_to_cart_joint_id])


@configclass
class CartpoleActionTermCfg(ActionTermCfg):
    class_type: CartpoleActionTerm = CartpoleActionTerm


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = CartpoleActionTermCfg(asset_name="robot")


# Cartpole Observation Configuration
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


# Cartpole Randomization Configuration


@configclass
class RandomizationCfg:
    """Configuration for randomization."""

    # reset
    reset_cart_position = RandTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )

    reset_pole_position = RandTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )


# Cartpole Environment Configuration


@configclass
class CartpoleEnvCfg(BaseEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    randomization: RandomizationCfg = RandomizationCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True


# Main


def main():
    """Main function."""

    # setup base environment
    env = BaseEnv(cfg=CartpoleEnvCfg())
    obs = env.reset()

    target_position = torch.zeros(env.num_envs, 1, device=env.device)

    # simulate physics
    count = 0
    while simulation_app.is_running():
        # reset
        if count % 300 == 0:
            env.reset()
            count = 0

        # step env
        obs, _ = env.step(target_position)

        # print current orientation of pole
        print(obs["policy"][0][1].item())
        # update counter
        count += 1


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
