# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_inv


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

    Args:
        source_F: Force in source frame.
        source_T: Torque in source frame.
        source_frame: Tuple of (quat_xyzw, pos) for source frame.
        target_frame: Tuple of (quat_xyzw, pos) for target frame.

    Returns:
        Tuple of (target_F, target_T) - force and torque in target frame.
    """
    # Modern Robotics eq. 3.95
    # Compute inverse of source frame
    source_quat_inv = quat_inv(source_frame[0])
    source_pos_inv = -quat_apply(source_quat_inv, source_frame[1])

    # Combine: source_inv * target = target_T_source
    target_T_source_pos, target_T_source_quat = combine_frame_transforms(
        source_pos_inv, source_quat_inv, target_frame[1], target_frame[0]
    )

    target_F = quat_apply(target_T_source_quat, source_F)
    target_T = quat_apply(target_T_source_quat, (source_T + torch.cross(target_T_source_pos, source_F, dim=-1)))
    return target_F, target_T
