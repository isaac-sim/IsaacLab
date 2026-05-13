# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from A10_Single.robot.a10_single_cfg import A10_SINGLE_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class A10ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch robot to franka
        self.scene.robot = A10_SINGLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["link6"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["link6"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["link6"]

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", 
            joint_names=["joint1",
                         "joint2",
                         "joint3",
                         "joint4",
                         "joint5",
                         "joint6",], 
            scale=0.5, 
            use_default_offset=True
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "link6"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


@configclass
class A10ReachEnvCfg_PLAY(A10ReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class A10ReachFruitEnvCfg(A10ReachEnvCfg):
    """A10 reach scene with three colored fruit targets on the table."""

    def __post_init__(self):
        super().__post_init__()

        # Yellow lemon
        self.scene.lemon = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Lemon",
            spawn=sim_utils.SphereCfg(
                radius=0.028,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.96, 0.86, 0.18), roughness=0.35),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.50, -0.11, 0.03)),
        )
        # Green apple
        self.scene.apple = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Apple",
            spawn=sim_utils.SphereCfg(
                radius=0.032,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.23, 0.78, 0.25), roughness=0.35),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.52, 0.00, 0.035)),
        )
        # Pink strawberry (cone placeholder)
        self.scene.strawberry = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Strawberry",
            spawn=sim_utils.ConeCfg(
                radius=0.022,
                height=0.05,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.46, 0.76), roughness=0.35),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.49, 0.11, 0.035)),
        )
