# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.utils.math import quat_apply, subtract_frame_transforms


def get_random_prop_gains(default_values, noise_levels, num_envs, device):
    """Helper function to randomize controller gains."""
    c_param_noise = torch.rand((num_envs, default_values.shape[1]), dtype=torch.float32, device=device)
    c_param_noise = c_param_noise @ torch.diag(torch.tensor(noise_levels, dtype=torch.float32, device=device))
    c_param_multiplier = 1.0 + c_param_noise
    decrease_param_flag = torch.rand((num_envs, default_values.shape[1]), dtype=torch.float32, device=device) > 0.5
    c_param_multiplier = torch.where(decrease_param_flag, 1.0 / c_param_multiplier, c_param_multiplier)

    prop_gains = default_values * c_param_multiplier

    return prop_gains


def change_FT_frame(source_F, source_T, source_frame, target_frame):
    """Convert force/torque reading from source to target frame.

    The wrench is re-expressed in the target frame's axes and the torque is taken about the
    target frame's origin.

    Args:
        source_F: Force in source frame [N]. Shape is (N, 3).
        source_T: Torque in source frame, about the source frame's origin [N·m]. Shape is (N, 3).
        source_frame: Tuple of (quat_xyzw, pos) for source frame, both expressed in a common frame.
        target_frame: Tuple of (quat_xyzw, pos) for target frame, both expressed in the same common frame.

    Returns:
        Tuple of (target_F, target_T) - force [N] and torque [N·m] in target frame.
    """
    # Modern Robotics eq. 3.95: F_t = Ad_{T_st}^T F_s, written in terms of T_ts (source pose in the target frame).
    source_pos_in_target, source_quat_in_target = subtract_frame_transforms(
        target_frame[1], target_frame[0], source_frame[1], source_frame[0]
    )

    target_F = quat_apply(source_quat_in_target, source_F)
    target_T = quat_apply(source_quat_in_target, source_T) + torch.cross(source_pos_in_target, target_F, dim=-1)
    return target_F, target_T
