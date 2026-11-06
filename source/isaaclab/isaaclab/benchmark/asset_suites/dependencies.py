# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared and backend-variant data-property dependency declarations."""

ARTICULATION_DEPENDENCIES = {
    "root_link_lin_vel_w": ("root_link_vel_w",),
    "root_link_ang_vel_w": ("root_link_vel_w",),
    "root_link_lin_vel_b": ("root_link_lin_vel_w", "root_link_quat_w"),
    "root_link_ang_vel_b": ("root_link_ang_vel_w", "root_link_quat_w"),
    "root_com_pos_w": ("root_com_pose_w",),
    "root_com_quat_w": ("root_com_pose_w",),
    "root_com_lin_vel_b": ("root_com_lin_vel_w", "root_link_quat_w"),
    "root_com_ang_vel_b": ("root_com_ang_vel_w", "root_link_quat_w"),
    "root_com_lin_vel_w": ("root_com_vel_w",),
    "root_com_ang_vel_w": ("root_com_vel_w",),
    "root_link_pos_w": ("root_link_pose_w",),
    "root_link_quat_w": ("root_link_pose_w",),
    "body_link_lin_vel_w": ("body_link_vel_w",),
    "body_link_ang_vel_w": ("body_link_vel_w",),
    "body_link_pos_w": ("body_link_pose_w",),
    "body_link_quat_w": ("body_link_pose_w",),
    "body_com_pos_w": ("body_com_pose_w",),
    "body_com_quat_w": ("body_com_pose_w",),
    "body_com_lin_vel_w": ("body_com_vel_w",),
    "body_com_ang_vel_w": ("body_com_vel_w",),
    "body_com_lin_acc_w": ("body_com_acc_w",),
    "body_com_ang_acc_w": ("body_com_acc_w",),
    "body_com_quat_b": ("body_com_pose_b",),
}

RIGID_OBJECT_DEPENDENCIES = ARTICULATION_DEPENDENCIES
OVPHYSX_RIGID_OBJECT_DEPENDENCIES = ARTICULATION_DEPENDENCIES

OBJECT_COLLECTION_DEPENDENCIES = {
    "object_link_lin_vel_w": ("object_link_vel_w",),
    "object_link_ang_vel_w": ("object_link_vel_w",),
    "object_com_pos_w": ("object_com_pose_w",),
    "object_com_quat_w": ("object_com_pose_w",),
    "object_com_lin_vel_w": ("object_com_vel_w",),
    "object_com_ang_vel_w": ("object_com_vel_w",),
    "object_link_pos_w": ("object_link_pose_w",),
    "object_link_quat_w": ("object_link_pose_w",),
    "object_com_lin_acc_w": ("object_com_acc_w",),
    "object_com_ang_acc_w": ("object_com_acc_w",),
    "object_com_quat_b": ("object_com_pose_b",),
}

BODY_COLLECTION_DEPENDENCIES = {
    "body_link_pos_w": ("body_link_pose_w",),
    "body_link_quat_w": ("body_link_pose_w",),
    "body_link_lin_vel_w": ("body_link_vel_w",),
    "body_link_ang_vel_w": ("body_link_vel_w",),
    "body_com_pos_w": ("body_com_pose_w",),
    "body_com_quat_w": ("body_com_pose_w",),
    "body_com_lin_vel_w": ("body_com_vel_w",),
    "body_com_ang_vel_w": ("body_com_vel_w",),
    "body_com_lin_acc_w": ("body_com_acc_w",),
    "body_com_ang_acc_w": ("body_com_acc_w",),
    "body_com_quat_b": ("body_com_pose_b",),
}
