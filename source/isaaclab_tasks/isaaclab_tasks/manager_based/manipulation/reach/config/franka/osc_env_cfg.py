# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaReachEnvCfg(joint_pos_env_cfg.FrankaReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We remove stiffness and damping for the shoulder and forearm joints for effort control
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["panda_shoulder"].stiffness = 0.0
        self.scene.robot.actuators["panda_shoulder"].damping = 0.0
        self.scene.robot.actuators["panda_forearm"].stiffness = 0.0
        self.scene.robot.actuators["panda_forearm"].damping = 0.0
        self.scene.robot.spawn.rigid_props.disable_gravity = True

        # If closed-loop contact force control is desired, contact sensors should be enabled for the robot
        # self.scene.robot.spawn.activate_contact_sensors = True

        self.actions.arm_action = OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            # If a task frame different from articulation root/base is desired, a RigidObject, e.g., "task_frame",
            # can be added to the scene and its relative path could provided as task_frame_rel_path
            # task_frame_rel_path="task_frame",
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_abs"],
                impedance_mode="variable_kp",
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=False,
                motion_stiffness_task=100.0,
                motion_damping_ratio_task=1.0,
                motion_stiffness_limits_task=(50.0, 200.0),
                nullspace_control="position",
            ),
            nullspace_joint_pos_target="center",
            position_scale=1.0,
            orientation_scale=1.0,
            stiffness_scale=100.0,
        )
        # Removing these observations as they are not needed for OSC and we want keep the observation space small
        self.observations.policy.joint_pos = None
        self.observations.policy.joint_vel = None


@configclass
class FrankaReachEnvCfg_PLAY(FrankaReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
