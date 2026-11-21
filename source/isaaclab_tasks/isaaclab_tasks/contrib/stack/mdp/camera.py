# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera domain randomization for stack policies."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from pxr import UsdGeom

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.noise import NoiseModel

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import Camera

    from .camera_noise_cfg import EpisodeCameraNoiseCfg

logger = logging.getLogger(__name__)


class EpisodeCameraNoise(NoiseModel):
    """Apply episode-consistent photometric variation and per-frame sensor noise."""

    def __init__(self, noise_model_cfg: EpisodeCameraNoiseCfg, num_envs: int, device: str):
        super().__init__(noise_model_cfg, num_envs, device)
        self._exposure_range = self._validate_range("exposure_range", noise_model_cfg.exposure_range, positive=True)
        self._contrast_range = self._validate_range("contrast_range", noise_model_cfg.contrast_range, positive=True)
        self._white_balance_range = self._validate_range(
            "white_balance_range", noise_model_cfg.white_balance_range, positive=True
        )
        self._brightness_range = self._validate_range("brightness_range", noise_model_cfg.brightness_range)
        scalar_shape = (num_envs, 1, 1, 1)
        self._exposure = torch.ones(scalar_shape, device=device)
        self._contrast = torch.ones(scalar_shape, device=device)
        self._white_balance = torch.ones((num_envs, 3, 1, 1), device=device)
        self._brightness = torch.zeros(scalar_shape, device=device)
        self.reset()

    @staticmethod
    def _validate_range(name: str, values: tuple[float, float], *, positive: bool = False) -> tuple[float, float]:
        lower, upper = values
        if lower > upper or (positive and lower <= 0.0):
            qualifier = "positive and ordered" if positive else "ordered"
            raise ValueError(f"{name} must be {qualifier}, got {values}.")
        return float(lower), float(upper)

    @staticmethod
    def _resample(
        tensor: torch.Tensor, env_ids: Sequence[int] | torch.Tensor | None, value_range: tuple[float, float]
    ) -> None:
        if env_ids is None:
            tensor.uniform_(*value_range)
        else:
            tensor[env_ids] = torch.empty_like(tensor[env_ids]).uniform_(*value_range)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        """Resample photometric parameters for the selected environments."""
        self._resample(self._exposure, env_ids, self._exposure_range)
        self._resample(self._contrast, env_ids, self._contrast_range)
        self._resample(self._white_balance, env_ids, self._white_balance_range)
        self._resample(self._brightness, env_ids, self._brightness_range)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply photometric and sensor variation to NCHW RGB in the ``[0, 1]`` range."""
        if data.ndim != 4 or data.shape[1] != 3:
            raise ValueError(f"EpisodeCameraNoise expects NCHW RGB input, got shape {tuple(data.shape)}.")
        data = (data - 0.5) * self._contrast + 0.5
        data = data * self._exposure * self._white_balance + self._brightness
        return super().__call__(data)


def randomize_camera_calibration(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | torch.Tensor | slice | None,
    lookat_distance: float,
    eye_position_noise: tuple[float, float, float],
    lookat_position_noise: tuple[float, float, float],
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_camera"),
) -> None:
    """Apply one small, independent extrinsic perturbation per camera.

    Startup-only randomization models variation in camera mounting.

    Args:
        env: The stack environment containing the camera sensor.
        env_ids: Environments whose camera poses are randomized, or ``None``
            for every environment.
        lookat_distance: Distance from the configured camera position to the nominal look-at point [m].
        eye_position_noise: Independent uniform position-noise half-widths for
            the camera eye [m].
        lookat_position_noise: Independent uniform position-noise half-widths
            for the look-at point [m].
        sensor_cfg: Scene entity selecting the camera.
    """
    camera: Camera = env.scene.sensors[sensor_cfg.name]
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids if env_ids is not None else slice(None)]
    resolved_env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)
    if resolved_env_ids.numel() == 0:
        return

    origins = env.scene.env_origins[resolved_env_ids]
    eye_offset = origins.new_tensor(camera.cfg.offset.pos).unsqueeze(0)
    orientation = math_utils.convert_camera_frame_orientation_convention(
        origins.new_tensor(camera.cfg.offset.rot), origin=camera.cfg.offset.convention, target="opengl"
    )
    camera_rotation = math_utils.matrix_from_quat(orientation)
    lookat_offset = eye_offset - lookat_distance * camera_rotation[:, 2]

    eye_noise = (2.0 * torch.rand_like(origins) - 1.0) * origins.new_tensor(eye_position_noise)
    lookat_noise = (2.0 * torch.rand_like(origins) - 1.0) * origins.new_tensor(lookat_position_noise)
    eyes = origins + eye_offset + eye_noise
    targets = origins + lookat_offset + lookat_noise

    # Preserve the configured roll relative to a stage-up look-at frame.
    up_axis = UsdGeom.GetStageUpAxis(camera.stage)
    nominal_view = math_utils.create_rotation_matrix_from_view(eye_offset, lookat_offset, up_axis, env.device)
    roll = nominal_view.mT @ camera_rotation
    views = math_utils.create_rotation_matrix_from_view(eyes, targets, up_axis, env.device)
    valid = torch.isfinite(views).all(dim=(-2, -1)).nonzero(as_tuple=True)[0]
    if valid.numel() == 0:
        raise ValueError("Camera calibration requires distinct finite eye and look-at positions.")
    skipped = resolved_env_ids.numel() - valid.numel()
    if skipped:
        logger.warning("Skipping %d camera poses with undefined look-at directions.", skipped)
    camera.set_world_poses(
        eyes[valid],
        math_utils.quat_from_matrix(views[valid] @ roll),
        env_ids=resolved_env_ids[valid],
        convention="opengl",
    )
