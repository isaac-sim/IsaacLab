# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings

from isaaclab_newton.sim.schemas import MujocoRigidBodyCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.reach.config.franka import franka_reach_env_cfg
from isaaclab_tasks.utils import preset


class _DeprecatedDiffIKAbsWeight(float):
    """Marker for the deprecated ``diffik_abs`` no-op alias; replaced by a plain float during validation."""


@configclass
class FrankaReachEnvCfg(franka_reach_env_cfg.FrankaReachEnvCfg):
    def validate_config(self) -> None:
        """Validate the physics backend and warn about the deprecated ``diffik_abs`` alias."""
        super().validate_config()
        weight = self.rewards.action_magnitude.weight
        if isinstance(weight, _DeprecatedDiffIKAbsWeight):
            warnings.warn(
                "Preset 'diffik_abs' is deprecated for 'Isaac-Reach-Franka-OSC' and has no effect: the task always"
                " uses the operational space controller. Drop 'presets=diffik_abs' from the command line; the alias"
                " will be removed in a future release.",
                FutureWarning,
                stacklevel=2,
            )
            self.rewards.action_magnitude.weight = float(weight)

    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # Use an explicit actuator to enforce the USD-authored effort limits for effort control. Keep the
        # asset's solver velocity limit: the Menagerie USD authors none, so dropping it leaves the arm unbounded.
        arm_actuator = self.scene.robot.actuators["panda_arm"]
        self.scene.robot.actuators["panda_arm"] = IdealPDActuatorCfg(
            joint_names_expr=arm_actuator.joint_names_expr,
            joint_velocity_limit=arm_actuator.joint_velocity_limit,
            stiffness=0.0,
            damping=0.0,
        )
        for rigid_props in self.scene.robot.spawn.rigid_props:
            if isinstance(rigid_props, PhysxRigidBodyCfg):
                rigid_props.disable_gravity = True
            elif isinstance(rigid_props, MujocoRigidBodyCfg):
                rigid_props.gravcomp = 1.0

        # The OSC action term replaces the parent's arm-controller presets, so the parent's controller-keyed
        # variants would only zero the action-magnitude reward weight or attach 6D teleop devices here. Resolve
        # their defaults and keep ``diffik_abs`` as a deprecated no-op alias so existing command lines keep working.
        self.teleop_devices = self.teleop_devices.default
        default_weight = self.rewards.action_magnitude.weight.default
        self.rewards.action_magnitude.weight = preset(
            default=default_weight, diffik_abs=_DeprecatedDiffIKAbsWeight(default_weight)
        )

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

    def play_mode(self):
        # play-mode overrides of parent
        super().play_mode()

        # make a smaller scene for play
        self.scene.num_envs = 16
