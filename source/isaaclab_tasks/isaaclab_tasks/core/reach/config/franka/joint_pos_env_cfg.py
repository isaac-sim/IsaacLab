# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab_newton.physics import FeatherPGSSolverCfg, NewtonCfg

import isaaclab.envs.mdp as mdp
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.reach.reach_env_cfg import ReachEnvCfg, ReachPhysicsCfg

##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class FrankaReachEnvCfg(ReachEnvCfg):
    @configclass
    class PhysicsCfg(ReachPhysicsCfg):
        """Franka reach solver presets owned by this task composition."""

        feather_pgs = NewtonCfg(
            solver_cfg=FeatherPGSSolverCfg(
                pgs_mode="matrix_free",
                update_mass_matrix_interval=1,
                enable_joint_limits=True,
                joint_limit_activation_gap=0.1,
                pgs_iterations=8,
                pgs_velocity_iterations=0,
                dense_max_constraints=64,
                mf_max_constraints=32,
                hinv_jt_kernel="auto",
                pgs_warmstart=False,
                pgs_omega=1.0,
                pgs_beta=0.05,
                pgs_cfm=1.0e-6,
                serial_kernel_block_dim=64,
                row_watermark=False,
            ),
            num_substeps=1,
            debug_mode=False,
            use_cuda_graph=False,
        )

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.sim.physics = self.PhysicsCfg()

        # switch robot to franka
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # override rewards
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["panda_hand"]

        # override actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        # override command generator body
        # end-effector is along z-direction
        self.commands.ee_pose.body_name = "panda_hand"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)


##
# Play configuration
##


@configclass
class FrankaReachEnvCfg_PLAY(FrankaReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
