# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Built-in terms, the MDP2 counterparts of :mod:`isaaclab.envs.mdp`.

Terms keep the names of their MDP1 functions, and bind scene data through inputs instead of ``asset_cfg``::

    from isaaclab_experimental import mdp2
    from isaaclab_experimental.mdp2 import mdp

    mdp2.Observation(
        mdp.joint_pos_rel,
        inputs={
            "joint_pos": robot.joint_pos,
            "default_joint_pos": robot.default_joint_pos,
            "joint_ids": robot.joints(".*"),
        },
    )
"""

from .commands import heading_velocity_command, uniform_pose_command, uniform_velocity_command
from .curriculums import difficulty_scheduler
from .events import (
    push_by_setting_velocity,
    randomize_actuator_gains,
    randomize_joint_parameters,
    randomize_rigid_body_com,
    randomize_rigid_body_inertia,
    randomize_rigid_body_mass,
    randomize_rigid_body_material,
    reset_joints_by_offset,
    reset_joints_by_scale,
    reset_root_state_uniform,
)
from .observations import (
    base_ang_vel,
    base_lin_vel,
    generated_commands,
    joint_pos,
    joint_pos_limit_normalized,
    joint_pos_rel,
    joint_vel,
    joint_vel_rel,
    last_action,
    projected_gravity,
    root_pos_w,
    root_quat_w,
)
from .rewards import (
    action_l2,
    action_rate_l2,
    ang_vel_xy_l2,
    base_height_l2,
    feet_air_time,
    flat_orientation_l2,
    is_alive,
    is_terminated,
    joint_acc_l2,
    joint_deviation_l1,
    joint_pos_limits,
    joint_pos_target_l2,
    joint_torques_l2,
    joint_vel_l1,
    joint_vel_l2,
    lin_vel_z_l2,
    track_ang_vel_z_exp,
    track_lin_vel_xy_exp,
    undesired_contacts,
    value,
)
from .terminations import (
    bad_orientation,
    illegal_contact,
    is_positive,
    joint_pos_out_of_manual_limit,
    joint_vel_out_of_limit,
    root_height_below_minimum,
)
