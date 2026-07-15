# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for the Franka pour task (grasp a dynamic cup of MPM media and pour)."""

from isaaclab.envs.mdp import (  # noqa: F401
    AbsBinaryJointPositionActionCfg,
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
    RelativeJointPositionActionCfg,
    action_l2,
    action_rate_l2,
    joint_pos_rel,
    joint_vel_l2,
    joint_vel_rel,
    last_action,
    time_out,
)

from .actions import (  # noqa: F401
    CurriculumGripperPositionAction,
    CurriculumGripperPositionActionCfg,
    CurriculumJointPositionAction,
    CurriculumJointPositionActionCfg,
    TrajectoryJointPositionAction,
    TrajectoryJointPositionActionCfg,
)
from .curriculums import PourCurriculum  # noqa: F401
from .events import reset_pour_scene  # noqa: F401
from .observations import (  # noqa: F401
    arm_reference_error_obs,
    arm_reference_phase_obs,
    cup_pose_obs,
    cup_to_target_obs,
    cup_velocity_obs,
    ee_pose_obs,
    finger_position_obs,
    finger_velocity_obs,
    grasp_to_tcp_quat_obs,
    gripper_contact_obs,
    gripper_target_obs,
    gripper_width_obs,
    held_delivery_history_obs,
    lost_grasp_dwell_obs,
    particle_fractions_obs,
    particle_transfer_obs,
    pour_target_fraction_obs,
    success_dwell_obs,
    target_position_c_obs,
    target_pose_obs,
    tcp_pose_obs,
    tcp_to_grasp_obs,
    tcp_to_grasp_position_c_obs,
    time_remaining_obs,
    trajectory_status_obs,
)
from .rewards import (  # noqa: F401
    AlignProgress,
    ApproachProgress,
    GraspLiftProgress,
    HeldDeliveryProgress,
    LiftProgress,
    NewlyDeliveredParticles,
    NewlySpilledParticles,
    PourReferenceProgress,
    PourTaskProgress,
    PourTiltProgress,
    align_command_progress,
    align_cup_over_target,
    finite_joint_velocity_l2,
    grasp_cup,
    lift_command_progress,
    lift_cup,
    media_target_distance_tanh,
    particles_in_source,
    particles_in_target,
    pour_success_bonus,
    sustained_pour_success,
    reach_cup,
    spilled_particles,
    terminal_failure,
    tcp_cup_distance_tanh,
    tilt_command_progress,
    tilt_over_target,
)
from .reset_dataset import (  # noqa: F401
    PourResetDatasetCurriculum,
    reset_dataset_difficulty,
)
from .reset_mixture import (  # noqa: F401
    RESET_MIXTURE_REGION_NAMES,
    RESET_MIXTURE_STAGE_NAMES,
    PourResetMixture,
)
from .terminations import (  # noqa: F401
    excessive_spill,
    extreme_rigid_state,
    immediate_pour_success,
    lost_lifted_grasp,
    nonterminating_stable_pour_success,
    nonfinite_failure,
    particle_out_of_bounds,
    stable_pour_success,
    unsuccessful_time_out,
)
