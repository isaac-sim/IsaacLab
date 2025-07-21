# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp

from ...reach_env_cfg import EventCfg


# Observation configurations
@configclass
class StateNoNoiseObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class StateNoisyObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.1, n_max=0.1))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.1, n_max=0.1))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class JointRandPositionFrictionEventCfg(EventCfg):

    reset_robot_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "gaussian",
        },
    )


@configclass
class JointRandPositionFrictionAmartureEventCfg(JointRandPositionFrictionEventCfg):
    """Configuration for events."""

    reset_robot_joint_amature = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "gaussian",
        },
    )


@configclass
class EnvConfigurables:
    env: dict[str, any] = {
        "observations": {
            "state_obs_no_noise": StateNoNoiseObservationsCfg(),
            "state_obs_noisy": StateNoisyObservationsCfg(),
        },
        "actions.arm_action": {
            "ik_abs_arm_action": mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller=mdp.DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=False, ik_method="dls"
                ),
                body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
            ),
            "ik_rel_arm_action": mdp.DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller=mdp.DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=True, ik_method="dls"
                ),
                scale=0.5,
                body_offset=mdp.DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.107]),
            ),
            "joint_pos_arm_action": mdp.JointPositionActionCfg(
                asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
            ),
            "osc_arm_action": mdp.OperationalSpaceControllerActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                body_name="panda_hand",
                controller_cfg=mdp.OperationalSpaceControllerCfg(
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
            ),
        },
        "events": {
            "rand_joint_pos_friction": JointRandPositionFrictionEventCfg(),
            "rand_joint_pos_friction_amarture": JointRandPositionFrictionAmartureEventCfg(),
        },
        "events.reset_robot_joints": {
            "aggressive": EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.0, 2.0),
                    "velocity_range": (0.0, 1.0),
                },
            ),
            "easy": EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.0, 0.5),
                    "velocity_range": (0.0, 0.0),
                },
            ),
        },
    }
