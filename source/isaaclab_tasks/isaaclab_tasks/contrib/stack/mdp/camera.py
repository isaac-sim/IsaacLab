# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Camera domain randomization for stack policies."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseModel, NoiseModelCfg, UniformNoiseCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.sensors import Camera


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
    def _resample(tensor: torch.Tensor, env_ids: Sequence[int] | None, value_range: tuple[float, float]) -> None:
        if env_ids is None:
            tensor.uniform_(*value_range)
        else:
            tensor[env_ids] = torch.empty_like(tensor[env_ids]).uniform_(*value_range)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
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
        return self._noise_model_cfg.noise_cfg.func(data, self._noise_model_cfg.noise_cfg)


@configclass
class EpisodeCameraNoiseCfg(NoiseModelCfg):
    """Episode-consistent camera randomization with small per-frame pixel noise."""

    class_type: type[EpisodeCameraNoise] | str = EpisodeCameraNoise
    """Noise-model implementation used by the observation manager."""

    noise_cfg: UniformNoiseCfg = UniformNoiseCfg(n_min=-0.025, n_max=0.025)
    """Per-frame additive pixel noise in normalized image units."""

    exposure_range: tuple[float, float] = (0.75, 1.25)
    """Uniform episode-level multiplicative exposure range."""

    contrast_range: tuple[float, float] = (0.85, 1.15)
    """Uniform episode-level contrast range around mid-gray."""

    white_balance_range: tuple[float, float] = (0.90, 1.10)
    """Uniform episode-level per-channel white-balance range."""

    brightness_range: tuple[float, float] = (-0.05, 0.05)
    """Uniform episode-level additive brightness range in normalized image units."""


def randomize_camera_calibration(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    eye: tuple[float, float, float],
    lookat: tuple[float, float, float],
    eye_position_noise: tuple[float, float, float],
    lookat_position_noise: tuple[float, float, float],
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("base_camera"),
) -> None:
    """Apply one small, independent extrinsic perturbation per camera.

    This startup-only randomization models mounting and hand-eye calibration
    error without changing the camera during an episode. The real deployment
    therefore needs only an approximately matching calibrated view rather than
    the exact simulated transform.

    Args:
        env: The stack environment containing the camera sensor.
        env_ids: Environments whose camera poses are randomized, or ``None``
            for every environment.
        eye: Nominal camera position relative to each environment origin.
        lookat: Nominal look-at point relative to each environment origin.
        eye_position_noise: Independent uniform position-noise half-widths for
            the camera eye, in meters.
        lookat_position_noise: Independent uniform position-noise half-widths
            for the look-at point, in meters.
        sensor_cfg: Scene entity selecting the camera.
    """
    camera: Camera = env.scene.sensors[sensor_cfg.name]
    if env_ids is None:
        resolved_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        resolved_env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)
    if resolved_env_ids.numel() == 0:
        return

    origins = env.scene.env_origins[resolved_env_ids]
    eye_offset = origins.new_tensor(eye).expand(resolved_env_ids.numel(), -1)
    lookat_offset = origins.new_tensor(lookat).expand(resolved_env_ids.numel(), -1)
    eye_amplitude = origins.new_tensor(eye_position_noise)
    lookat_amplitude = origins.new_tensor(lookat_position_noise)
    eye_noise = (2.0 * torch.rand_like(eye_offset) - 1.0) * eye_amplitude
    lookat_noise = (2.0 * torch.rand_like(lookat_offset) - 1.0) * lookat_amplitude

    camera.set_world_poses_from_view(
        origins + eye_offset + eye_noise,
        origins + lookat_offset + lookat_noise,
        env_ids=resolved_env_ids,
    )
