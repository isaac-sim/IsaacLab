# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backward-compatible re-exports of the (now robot-neutral) stack event functions.

These functions used to live here but are robot-agnostic and shared by every stack robot config,
so they were moved to :mod:`.stack_events`. This module re-exports them so existing imports
(``from isaaclab_tasks.contrib.stack.mdp import franka_stack_events``) keep working. New configs
should import from :mod:`.stack_events` directly.
"""

from .stack_events import (
    randomize_joint_by_gaussian_offset,
    randomize_object_pose,
    randomize_rigid_objects_in_focus,
    randomize_scene_lighting_domelight,
    randomize_visual_texture_material,
    sample_object_poses,
    sample_random_color,
    set_default_joint_pose,
)

__all__ = [
    "randomize_joint_by_gaussian_offset",
    "randomize_object_pose",
    "randomize_rigid_objects_in_focus",
    "randomize_scene_lighting_domelight",
    "randomize_visual_texture_material",
    "sample_object_poses",
    "sample_random_color",
    "set_default_joint_pose",
]
