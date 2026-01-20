# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create drone observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.jit
from isaaclab_contrib.assets import Multirotor

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera, MultiMeshRayCasterCamera, RayCasterCamera, TiledCamera

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from isaaclab.envs.utils.io_descriptors import generic_io_descriptor, record_shape

from isaaclab_tasks import ISAACLAB_TASKS_EXT_DIR

"""
State.
"""


def base_roll_pitch(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return the base roll and pitch in the simulation world frame.

    Parameters:
        env: Manager-based environment providing the scene and tensors.
        asset_cfg: Scene entity config pointing to the target robot (default: "robot").

    Returns:
        torch.Tensor: Shape (num_envs, 2). Column 0 is roll, column 1 is pitch.
        Values are radians normalized to [-pi, pi], expressed in the world frame.

    Notes:
        - Euler angles are computed from asset.data.root_quat_w using XYZ convention.
        - Only roll and pitch are returned; yaw is omitted.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # extract euler angles (in world frame)
    roll, pitch, _ = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    # normalize angle to [-pi, pi]
    roll = math_utils.wrap_to_pi(roll)
    pitch = math_utils.wrap_to_pi(pitch)

    return torch.cat((roll.unsqueeze(-1), pitch.unsqueeze(-1)), dim=-1)


"""
Sensors
"""


class VAEModelManager:
    """Manager for the VAE model."""

    _model = None

    @classmethod
    def get_model(cls, device):
        """Get or load the VAE model."""
        if cls._model is None:
            import os

            model_path = os.path.join(ISAACLAB_TASKS_EXT_DIR, "data", "drone_arl", "vae_model.pt")
            cls._model = torch.jit.load(model_path, map_location=device)
            cls._model.eval()
        return cls._model


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
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera | MultiMeshRayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type]

    # depth image conversion
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)

    # rgb/depth/normals image normalization
    if normalize:
        if data_type == "distance_to_image_plane":
            images[images == float("inf")] = 10.0
            images[images == -float("inf")] = 10.0
            images[images > 10.0] = 10.0
            images = images / 10.0  # normalize to 0-1
            images[images < 0.02] = -1.0  # set very close values to -1
        else:
            raise ValueError(f"Image data type: {data_type} not supported")

    _vae_model = VAEModelManager.get_model(env.device)

    with torch.no_grad():
        latents = _vae_model(images.squeeze(-1).half())

    return latents


"""
Actions.
"""


@generic_io_descriptor(dtype=torch.float32, observation_type="Action", on_inspect=[record_shape])
def last_action_navigation(env: ManagerBasedEnv, action_name: str = "velocity_commands") -> torch.Tensor:
    """The last processed velocity commands from the navigation action term.

    This function accesses the velocity commands (vx, vy, vz, yaw_rate) that
    were computed by the NavigationAction term. This avoids duplicating the
    action processing logic.

    Args:
        env: Manager-based environment providing the action manager.
        action_name: Name of the navigation action term. Defaults to "velocity_commands".

    Returns:
        torch.Tensor: Shape (num_envs, 4) containing [vx, vy, vz, yaw_rate] commands.
    """
    action_term = env.action_manager.get_term(action_name)
    # Access the velocity_commands property from NavigationAction
    return action_term.prev_velocity_commands


"""
Commands.
"""


@generic_io_descriptor(dtype=torch.float32, observation_type="Command", on_inspect=[record_shape])
def generated_drone_commands(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Generate a body-frame direction and distance to the commanded position.

    This observation reads a command from env.command_manager identified by command_name,
    interprets its first three components as a target position in the world frame, and
    returns:
        [dir_x, dir_y, dir_z, distance]
    where dir_* is the unit vector from the current body origin to the target, expressed
    in the multirotor body (root link) frame, and distance is the Euclidean separation.

    Parameters:
        env: Manager-based RL environment providing scene and command manager.
        command_name: Name of the command term to query from the command manager.
        asset_cfg: Scene entity config for the multirotor asset (default: "robot").

    Returns:
        torch.Tensor: Shape (num_envs, 4) with body-frame unit direction (3) and distance (1).

    Frame conventions:
        - Current position is asset.data.root_pos_w relative to env.scene.env_origins (world frame).
        - Body orientation uses asset.data.root_link_quat_w to rotate world vectors into the body frame.

    Assumptions:
        - env.command_manager.get_command(command_name) returns at least three values
          representing a world-frame target position per environment.
        - A small epsilon (1e-8) is used to guard against zero-length direction vectors.
    """
    asset: Multirotor = env.scene[asset_cfg.name]
    current_position_w = asset.data.root_pos_w - env.scene.env_origins
    command = env.command_manager.get_command(command_name)
    current_position_b = math_utils.quat_apply_inverse(asset.data.root_link_quat_w, command[:, :3] - current_position_w)
    current_position_b_dir = current_position_b / (torch.norm(current_position_b, dim=-1, keepdim=True) + 1e-8)
    current_position_b_mag = torch.norm(current_position_b, dim=-1, keepdim=True)
    return torch.cat((current_position_b_dir, current_position_b_mag), dim=-1)
