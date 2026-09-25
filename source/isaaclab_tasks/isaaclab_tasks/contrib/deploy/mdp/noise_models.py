# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Noise models specific to deployment tasks."""

from __future__ import annotations

__all__ = [
    "ResetSampledConstantNoiseModel",
    "ResetSampledConstantNoiseModelCfg",
    "ResetSampledQuaternionNoiseModel",
    "ResetSampledQuaternionNoiseModelCfg",
]

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
from isaaclab.utils.noise import ConstantNoiseCfg, NoiseModel, NoiseModelCfg

if TYPE_CHECKING:
    from isaaclab.utils.noise import NoiseCfg


class ResetSampledConstantNoiseModel(NoiseModel):
    """Noise model that samples noise ONLY during reset and applies it consistently.

    The noise is sampled from the configured distribution ONLY during reset and applied consistently
    until the next reset. Unlike regular noise that generates new random values every step,
    this model maintains the same noise values throughout an episode.

    Note:
        This noise model was used since the noise randimization should only be done at reset time.
        Other noise models(Eg: GaussianNoise) were not used since this randomizes the noise at every time-step.
    """

    def __init__(self, noise_model_cfg: NoiseModelCfg, num_envs: int, device: str):
        # initialize parent class
        super().__init__(noise_model_cfg, num_envs, device)
        # store the noise configuration
        self._noise_cfg = noise_model_cfg.noise_cfg
        self._sampled_noise = torch.zeros((num_envs, 1), device=self._device)
        self._num_components: int | None = None

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the noise model by sampling NEW noise values.

        This method samples new noise for the specified environments using the configured noise function.
        The sampled noise will remain constant until the next reset.

        Args:
            env_ids: The environment ids to reset the noise model for. Defaults to None,
                in which case all environments are considered.
        """
        # resolve the environment ids
        if env_ids is None:
            env_ids = slice(None)

        # Use the existing noise function to sample new noise
        # Create dummy data to sample from the noise function
        dummy_data = torch.zeros(
            (env_ids.stop - env_ids.start if isinstance(env_ids, slice) else len(env_ids), 1), device=self._device
        )

        # Sample noise using the configured noise function
        sampled_noise = self._noise_model_cfg.noise_cfg.func(dummy_data, self._noise_model_cfg.noise_cfg)

        self._sampled_noise[env_ids] = sampled_noise

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the pre-sampled noise to the data.

        This method applies the noise that was sampled during the last reset.
        No new noise is generated - the same values are used consistently.

        Args:
            data: The data to apply the noise to. Shape is (num_envs, ...).

        Returns:
            The data with the noise applied. Shape is the same as the input data.
        """
        # on first apply, expand noise to match last dim of data
        if self._num_components is None:
            *_, self._num_components = data.shape
            # expand noise from (num_envs,1) to (num_envs, num_components)
            self._sampled_noise = self._sampled_noise.repeat(1, self._num_components)

        # apply the noise based on operation
        if self._noise_cfg.operation == "add":
            return data + self._sampled_noise
        elif self._noise_cfg.operation == "scale":
            return data * self._sampled_noise
        elif self._noise_cfg.operation == "abs":
            return self._sampled_noise
        else:
            raise ValueError(f"Unknown operation in noise: {self._noise_cfg.operation}")


@configclass
class ResetSampledConstantNoiseModelCfg(NoiseModelCfg):
    """Configuration for a noise model that samples noise ONLY during reset."""

    class_type: type = ResetSampledConstantNoiseModel

    noise_cfg: NoiseCfg = MISSING
    """The noise configuration for the noise.

    Based on this configuration, the noise is sampled at every reset of the noise model.
    """


class ResetSampledQuaternionNoiseModel(NoiseModel):
    """Noise model that applies a rotation perturbation to quaternion observations.

    At each episode reset, small Euler angle perturbations (roll, pitch, yaw) are sampled
    uniformly from configurable ranges and converted to a perturbation quaternion. This
    perturbation is then applied via quaternion multiplication at every step, producing a
    geometrically valid rotated quaternion (unlike additive noise on raw components).

    The perturbation is held constant for the entire episode until the next reset.
    """

    def __init__(self, noise_model_cfg: NoiseModelCfg, num_envs: int, device: str):
        super().__init__(noise_model_cfg, num_envs, device)
        self._roll_range = noise_model_cfg.roll_range
        self._pitch_range = noise_model_cfg.pitch_range
        self._yaw_range = noise_model_cfg.yaw_range
        # Identity quaternion in (x, y, z, w) format
        self._perturbation_quat = torch.zeros((num_envs, 4), device=device)
        self._perturbation_quat[:, 3] = 1.0

    def reset(self, env_ids: Sequence[int] | None = None):
        """Sample new rotation perturbations for the specified environments.

        Args:
            env_ids: The environment ids to reset. Defaults to None (all environments).
        """
        if env_ids is None:
            env_ids = slice(None)

        num_resets = env_ids.stop - env_ids.start if isinstance(env_ids, slice) else len(env_ids)

        roll = torch.empty(num_resets, device=self._device).uniform_(*self._roll_range)
        pitch = torch.empty(num_resets, device=self._device).uniform_(*self._pitch_range)
        yaw = torch.empty(num_resets, device=self._device).uniform_(*self._yaw_range)

        self._perturbation_quat[env_ids] = quat_from_euler_xyz(roll, pitch, yaw)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the pre-sampled rotation perturbation to the quaternion data.

        Args:
            data: Quaternion observations in (x, y, z, w) format. Shape is (num_envs, 4).

        Returns:
            Perturbed quaternions in (x, y, z, w) format. Shape is (num_envs, 4).
        """
        return quat_mul(self._perturbation_quat, data)


@configclass
class ResetSampledQuaternionNoiseModelCfg(NoiseModelCfg):
    """Configuration for a quaternion noise model that samples rotation perturbations at reset.

    The perturbation is specified as independent uniform ranges for roll, pitch, and yaw
    (in radians). At each episode reset, Euler angles are sampled and converted to a
    perturbation quaternion that is multiplied with the observed quaternion.
    """

    class_type: type = ResetSampledQuaternionNoiseModel

    noise_cfg: ConstantNoiseCfg = ConstantNoiseCfg(bias=0.0)
    """Unused placeholder inherited from NoiseModelCfg. Quaternion perturbation is
    controlled by roll_range, pitch_range, and yaw_range instead."""

    roll_range: tuple[float, float] = (-0.01745, 0.01745)
    """Uniform range for roll perturbation in radians. Default is ±1 degree."""

    pitch_range: tuple[float, float] = (-0.01745, 0.01745)
    """Uniform range for pitch perturbation in radians. Default is ±1 degree."""

    yaw_range: tuple[float, float] = (-0.01745, 0.01745)
    """Uniform range for yaw perturbation in radians. Default is ±1 degree."""
