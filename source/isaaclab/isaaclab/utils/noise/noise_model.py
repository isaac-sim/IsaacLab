# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from . import noise_cfg

##
# Noise as functions.
##


def _move_params_to_device(cfg: noise_cfg.NoiseCfg, data: torch.Tensor, *names: str) -> None:
    """Move tensor-valued noise parameters onto the data device (once, on first call)."""
    for name in names:
        value = getattr(cfg, name)
        if isinstance(value, torch.Tensor) and value.device != data.device:
            setattr(cfg, name, value.to(data.device))


def _apply_noise(data: torch.Tensor, noise: torch.Tensor | float, operation: str) -> torch.Tensor:
    """Combine ``noise`` with ``data`` according to the configured operation."""
    if operation == "add":
        return data + noise
    if operation == "scale":
        return data * noise
    if operation == "abs":
        # always a new tensor: the noise may be a config parameter (e.g. the constant noise bias)
        return torch.zeros_like(data) + noise
    raise ValueError(f"Unknown operation in noise: {operation}")


def constant_noise(data: torch.Tensor, cfg: noise_cfg.ConstantNoiseCfg) -> torch.Tensor:
    """Applies a constant noise bias to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        cfg: The configuration parameters for constant noise.

    Returns:
        The data modified by the noise parameters provided.
    """
    _move_params_to_device(cfg, data, "bias")
    return _apply_noise(data, cfg.bias, cfg.operation)


def uniform_noise(data: torch.Tensor, cfg: noise_cfg.UniformNoiseCfg) -> torch.Tensor:
    """Applies a uniform noise to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        cfg: The configuration parameters for uniform noise.

    Returns:
        The data modified by the noise parameters provided.
    """

    _move_params_to_device(cfg, data, "n_min", "n_max")
    noise = torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min
    return _apply_noise(data, noise, cfg.operation)


def gaussian_noise(data: torch.Tensor, cfg: noise_cfg.GaussianNoiseCfg) -> torch.Tensor:
    """Applies a gaussian noise to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        cfg: The configuration parameters for gaussian noise.

    Returns:
        The data modified by the noise parameters provided.
    """

    _move_params_to_device(cfg, data, "mean", "std")
    noise = cfg.mean + cfg.std * torch.randn_like(data)
    return _apply_noise(data, noise, cfg.operation)


##
# Noise models as classes
##


class NoiseModel:
    """Base class for noise models."""

    def __init__(self, noise_model_cfg: noise_cfg.NoiseModelCfg, num_envs: int, device: str):
        """Initialize the noise model.

        Args:
            noise_model_cfg: The noise configuration to use.
            num_envs: The number of environments.
            device: The device to use for the noise model.
        """
        self._noise_model_cfg = noise_model_cfg
        self._num_envs = num_envs
        self._device = device

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the noise model.

        This method can be implemented by derived classes to reset the noise model.
        This is useful when implementing temporal noise models such as random walk.

        Args:
            env_ids: The environment ids to reset the noise model for. Defaults to None,
                in which case all environments are considered.
        """
        pass

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply the noise to the data.

        Args:
            data: The data to apply the noise to. Shape is (num_envs, ...).

        Returns:
            The data with the noise applied. Shape is the same as the input data.
        """
        return self._noise_model_cfg.noise_cfg.func(data, self._noise_model_cfg.noise_cfg)


class NoiseModelWithAdditiveBias(NoiseModel):
    """Noise model with an additive bias.

    The bias term is sampled from a the specified distribution on reset.
    """

    def __init__(self, noise_model_cfg: noise_cfg.NoiseModelWithAdditiveBiasCfg, num_envs: int, device: str):
        # initialize parent class
        super().__init__(noise_model_cfg, num_envs, device)
        # store the bias noise configuration
        self._bias_noise_cfg = noise_model_cfg.bias_noise_cfg
        self._bias = torch.zeros((num_envs, 1), device=self._device)
        self._num_components: int | None = None
        self._sample_bias_per_component = noise_model_cfg.sample_bias_per_component

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the noise model.

        This method resets the bias term for the specified environments.

        Args:
            env_ids: The environment ids to reset the noise model for. Defaults to None,
                in which case all environments are considered.
        """
        # resolve the environment ids
        if env_ids is None:
            env_ids = slice(None)
        # reset the bias term
        self._bias[env_ids] = self._bias_noise_cfg.func(self._bias[env_ids], self._bias_noise_cfg)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """Apply bias noise to the data.

        Args:
            data: The data to apply the noise to. Shape is (num_envs, ...).

        Returns:
            The data with the noise applied. Shape is the same as the input data.
        """
        # if sample_bias_per_component, on first apply, expand bias to match last dim of data
        if self._sample_bias_per_component and self._num_components is None:
            *_, self._num_components = data.shape
            # expand bias from (num_envs,1) to (num_envs, num_components)
            self._bias = self._bias.repeat(1, self._num_components)
            # now re-sample that expanded bias in-place
            self.reset()
        return super().__call__(data) + self._bias
