# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_newton.envs.mdp.actions.newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg
from isaaclab_newton.ik.newton_ik_objectives_cfg import NewtonIKJointLimitObjectiveCfg, NewtonIKPoseObjectiveCfg
from isaaclab_newton.ik.newton_ik_solver_cfg import NewtonIKSolverCfg

from isaaclab.utils.configclass import configclass

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class FrankaReachEnvCfg(joint_pos_env_cfg.FrankaReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Newton IK consumes the replicated robot prototype.
        self.sim.physics = self.sim.physics.newton_mjwarp
        self.scene.table = None
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.arm_action = NewtonInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            controller=NewtonIKSolverCfg(optimizer="lm", jacobian_mode="analytic", iterations=24),
            objectives=[
                NewtonIKPoseObjectiveCfg(
                    body_name="panda_hand",
                    body_offset_pos=(0.0, 0.0, 0.107),
                    command_type="pose",
                    use_relative_mode=True,
                    scale=0.2,
                ),
                NewtonIKJointLimitObjectiveCfg(weight=0.1),
            ],
        )


@configclass
class FrankaReachEnvCfg_PLAY(FrankaReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
