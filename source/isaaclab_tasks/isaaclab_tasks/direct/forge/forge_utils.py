# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

import isaacsim.core.utils.torch as torch_utils


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
    """Convert force/torque reading from source to target frame."""
    # Modern Robotics eq. 3.95
    source_frame_inv = torch_utils.tf_inverse(source_frame[0], source_frame[1])
    target_T_source_quat, target_T_source_pos = torch_utils.tf_combine(
        source_frame_inv[0], source_frame_inv[1], target_frame[0], target_frame[1]
    )
    target_F = torch_utils.quat_apply(target_T_source_quat, source_F)
    target_T = torch_utils.quat_apply(
        target_T_source_quat, (source_T + torch.cross(target_T_source_pos, source_F, dim=-1))
    )
    return target_F, target_T
