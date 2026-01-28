# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create drone observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.jit

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import Camera, MultiMeshRayCasterCamera, RayCasterCamera, TiledCamera

from isaaclab_contrib.assets import Multirotor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import ObservationTermCfg

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


class ImageLatentObservation(ManagerTermBase):
    """Callable observation term that returns VAE latents from camera images.

    This observation term extracts images from a configured camera sensor, normalizes them
    based on the data type, and passes them through a pre-trained VAE model to obtain
    latent representations. The VAE model is loaded once and cached on the class to avoid
    repeated disk loads across all instances.

    The term is designed to work with the Isaac Lab observation manager and integrates
    seamlessly with other observation terms in multi-modal observation spaces.

    Attributes:
        camera_sensor: The camera sensor to extract images from (TiledCamera, Camera,
            RayCasterCamera, or MultiMeshRayCasterCamera).
        data_type: Type of data to extract from the sensor (e.g., "distance_to_image_plane").
        convert_perspective_to_orthogonal: Whether to convert perspective depth to orthogonal.
        normalize: Whether to normalize images before passing to VAE.

    Example:
        To use this in an environment configuration:

        .. code-block:: python

            depth_latent = ObsTerm(
                func=mdp.ImageLatentObservation,
                params={
                    "sensor_cfg": SceneEntityCfg("depth_camera"),
                    "data_type": "distance_to_image_plane",
                    "normalize": True,
                },
            )
    """

    _model: torch.jit.ScriptModule | None = None

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        """Initialize the image latent observation term.

        Extracts configuration from cfg.params and caches a reference to the camera sensor
        for efficient repeated access during observation collection.

        Args:
            cfg: Configuration object containing the observation term configuration,
                including params dict with:
                - sensor_cfg (SceneEntityCfg): Scene entity config for the camera sensor.
                - data_type (str): Data type to extract from the sensor.
                - convert_perspective_to_orthogonal (bool, optional): Whether to convert
                  perspective to orthogonal depth. Defaults to False.
                - normalize (bool, optional): Whether to normalize images. Defaults to True.
            env: The manager-based RL environment instance.

        Raises:
            KeyError: If required params ("sensor_cfg", "data_type") are missing.
            RuntimeError: If the specified camera sensor is not found in the scene.
        """
        super().__init__(cfg, env)
        self.camera_sensor: TiledCamera | Camera | RayCasterCamera | MultiMeshRayCasterCamera = env.scene.sensors[
            cfg.params["sensor_cfg"].name
        ]
        self.data_type: str = cfg.params["data_type"]
        self.convert_perspective_to_orthogonal: bool = (False,)
        self.normalize: bool = True

    @classmethod
    def _get_model(cls, device):
        """Load or retrieve the cached VAE model.

        The model is loaded from disk only once per process and cached on the class.
        Subsequent calls return the cached instance, avoiding repeated I/O and model
        initialization overhead.

        Args:
            device: PyTorch device to load the model onto (e.g., "cpu", "cuda:0").

        Returns:
            Loaded VAE model as a TorchScript ScriptModule, set to evaluation mode.

        Raises:
            FileNotFoundError: If the VAE model file cannot be found at the expected path.
            RuntimeError: If the model cannot be loaded (e.g., corrupted file).
        """
        if cls._model is None:
            model_path = os.path.join(ISAACLAB_TASKS_EXT_DIR, "data", "drone_arl", "vae_model.pt")
            cls._model = torch.jit.load(model_path, map_location=device)
            cls._model.eval()
        return cls._model

    def __call__(self, env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, data_type: str) -> torch.Tensor:
        """Compute VAE latents for the current camera frame.

        Extracts images from the camera sensor, applies normalization if configured,
        and passes them through the VAE model to obtain latent representations.

        Args:
            env: The manager-based environment providing scene and device information.
            sensor_cfg: Scene entity config for the camera sensor (unused, already set in __init__).
            data_type: Data type to extract from the sensor (unused, already set in __init__).
            convert_perspective_to_orthogonal: Whether to convert perspective to orthogonal depth
                (unused, already set in __init__).
            normalize: Whether to normalize images (unused, already set in __init__).

        Returns:
            torch.Tensor: Latent representations from the VAE model.
                Shape is determined by the VAE architecture (typically (num_envs, latent_dim)).

        Raises:
            ValueError: If data_type is "distance_to_image_plane" but normalize is False,
                or if an unsupported data_type is encountered with normalize=True.
            RuntimeError: If the VAE model inference fails or tensors have incompatible shapes.

        Notes:
            - Images are converted to float16 before passing to the VAE for efficiency.
            - Infinity values in depth images are clamped to 10.0 during normalization.
            - Very small depth values (< 0.02) are set to -1.0 to indicate invalid regions.
            - The parameters (sensor_cfg, data_type, etc.) are ignored here as they are
            already stored during initialization. They are included in the signature only
            to satisfy the observation manager's parameter validation.
        """
        images = self.camera_sensor.data.output[self.data_type]

        if (self.data_type == "distance_to_camera") and self.convert_perspective_to_orthogonal:
            images = math_utils.orthogonalize_perspective_depth(images, self.camera_sensor.data.intrinsic_matrices)

        if self.normalize:
            if self.data_type == "distance_to_image_plane":
                images[images == float("inf")] = 10.0
                images[images == -float("inf")] = 10.0
                images[images > 10.0] = 10.0
                images = images / 10.0
                images[images < 0.02] = -1.0
            else:
                raise ValueError(f"Image data type: {self.data_type} not supported")

        vae_model = self._get_model(env.device)
        with torch.no_grad():
            latents = vae_model(images.squeeze(-1).half())

        return latents


"""
Actions.
"""


@generic_io_descriptor(dtype=torch.float32, observation_type="Action", on_inspect=[record_shape])
def last_action_navigation(env: ManagerBasedEnv, action_name: str = "velocity_commands") -> torch.Tensor:
    """The last processed position/velocity/acceleration commands from the navigation action term.

    This function accesses the position/velocity/acceleration commands (vx, vy, vz, yaw_rate) that
    were computed by the NavigationAction term. This avoids duplicating the
    action processing logic.

    Args:
        env: Manager-based environment providing the action manager.
        action_name: Name of the navigation action term. Defaults to "velocity_commands".

    Returns:
        torch.Tensor: Shape (num_envs, 4) containing position/velocity/acceleration commands.
    """
    action_term = env.action_manager.get_term(action_name)
    # Access the velocity_commands property from NavigationAction
    return action_term.prev_commands


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
