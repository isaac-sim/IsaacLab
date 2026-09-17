# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observations for the H2 pick-and-place apple task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp as base_mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_robot_joint_states(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return robot position, velocity, and torque in articulation order."""
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos.torch
    joint_vel = robot.data.joint_vel.torch
    joint_torque = robot.data.applied_torque.torch
    return torch.cat([joint_pos, joint_vel, joint_torque], dim=-1)


def warm_rgb_image(
    env: ManagerBasedRLEnv,
    sensor_cfg,
    data_type: str = "rgb",
    normalize: bool = False,
    response_gamma: tuple[float, float, float] = (1.0, 1.0, 1.0),
    response_gain: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """Apply the fitted real-camera color response to an RTX RGB image."""
    image = base_mdp.image(
        env,
        sensor_cfg=sensor_cfg,
        data_type=data_type,
        normalize=normalize,
    )
    if data_type != "rgb":
        return image

    output_dtype = image.dtype
    image = image.to(torch.float32)
    image = (image / 255.0).clamp(0.0, 1.0)
    image = image - 0.18 * image.pow(3)
    warm_balance = image.new_tensor((1.01, 0.99, 0.96))
    image = (image * warm_balance).clamp_(0.0, 1.0)

    # The renderer hands back near-linear radiance, so the gamma is the sRGB
    # encode the real sensor applies and the per-camera gain is its white balance
    # and exposure.  Keeping them at those physical values means scene radiance
    # has to come from lighting and materials rather than from a fitted curve.
    gamma = image.new_tensor(response_gamma)
    gain = image.new_tensor(response_gain)
    image = image.pow(gamma) * gain

    return (image * 255.0).clamp_(0.0, 255.0).to(output_dtype)
