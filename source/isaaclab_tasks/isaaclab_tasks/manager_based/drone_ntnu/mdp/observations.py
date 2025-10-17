# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create drone observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch, torch.jit
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_shape,
)

_vae_model = None #TODO @mihirk @welfr this is bad, need fix

def image_latents(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    
    # TODO @mihirk change to use image_features API  
    
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type]

    # depth image conversion
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    # rgb/depth/normals image normalization
    if normalize:
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 10.0
            images[images == -float("inf")] = 10.0
            images[images > 10.0] = 10.0
            images = images / 10.0  # normalize to 0-1
            images[images < 0.02] = -1.0  # set very close values to -1
        elif "normals" in data_type:
            images = (images + 1.0) * 0.5
    global _vae_model
    if _vae_model is None:
        # load the model from the pt file in the same directory as this file
        import os
        model_path = os.path.join(os.path.dirname(__file__), "vae_model.pt")
        _vae_model = torch.jit.load(model_path, map_location=env.device)
        _vae_model.eval()
    with torch.no_grad():
        latents = _vae_model(images.squeeze(-1).half())
    return latents


"""
Actions.
"""


@generic_io_descriptor(dtype=torch.float32, observation_type="Action", on_inspect=[record_shape])
def last_action_navigation(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    clamped_action = torch.clamp(env.action_manager.action, min=-1.0, max=1.0)
    processed_actions = torch.zeros(env.num_envs, 4, device=env.device)
    max_speed = 2.0  # [m/s]
    max_yawrate = torch.pi / 3.0  # [rad/s]
    max_inclination_angle = torch.pi / 4.0  # [rad]

    clamped_action[:, 0] += 1.0  # only allow positive thrust commands [0, 2]
    processed_actions[:, 0] = (
        clamped_action[:, 0]
        * torch.cos(max_inclination_angle * clamped_action[:, 1])
        * max_speed
        / 2.0
    )
    processed_actions[:, 1] = 0.0  # set lateral thrust command to 0
    processed_actions[:, 2] = (
        clamped_action[:, 0]
        * torch.sin(max_inclination_angle * clamped_action[:, 1])
        * max_speed
        / 2.0
    )
    processed_actions[:, 3] = clamped_action[:, 2] * max_yawrate
    return processed_actions
"""
Commands.
"""


@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def generated_commands(env: ManagerBasedRLEnv, command_name: str | None = None, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    asset: RigidObject = env.scene[asset_cfg.name]    
    current_position_w = asset.data.root_pos_w - env.scene.env_origins
    command = env.command_manager.get_command(command_name)
    current_position_b = math_utils.quat_apply_inverse(asset.data.root_link_quat_w, command[:, :3] - current_position_w)
    current_position_b_dir = current_position_b / (torch.norm(current_position_b, dim=-1, keepdim=True) + 1e-8)
    current_position_b_mag = torch.norm(current_position_b, dim=-1, keepdim=True)
    return torch.cat((current_position_b_dir, current_position_b_mag), dim=-1)

def base_roll_pitch(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Roll and pitch of the base in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # extract euler angles (in world frame)
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    # normalize angle to [-pi, pi]
    roll = torch.atan2(torch.sin(roll), torch.cos(roll))
    pitch = torch.atan2(torch.sin(pitch), torch.cos(pitch))

    return torch.cat((roll.unsqueeze(-1), pitch.unsqueeze(-1)), dim=-1)
