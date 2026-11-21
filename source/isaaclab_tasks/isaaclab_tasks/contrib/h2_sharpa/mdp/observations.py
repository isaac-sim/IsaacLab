# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared H2 camera observation processing."""

import torch


def apply_camera_response(
    image: torch.Tensor,
    response_gamma: tuple[float, float, float] = (1.0, 1.0, 1.0),
    response_gain: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Apply the fitted color response to a raw uint8 RGB observation."""
    output_dtype = image.dtype
    image = image.to(torch.float32)
    image = (image / 255.0).clamp(0.0, 1.0)
    image = image - 0.18 * image.pow(3)
    warm_balance = image.new_tensor((1.01, 0.99, 0.96))
    image = (image * warm_balance).clamp_(0.0, 1.0)

    # Match the real camera's sRGB response, white balance, and exposure.
    gamma = image.new_tensor(response_gamma)
    gain = image.new_tensor(response_gain)
    image = image.pow(gamma) * gain

    return (image * 255.0).clamp_(0.0, 255.0).to(output_dtype)
