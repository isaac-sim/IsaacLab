# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import noise_cfg


def constant_noise(data: torch.Tensor, cfg: noise_cfg.ConstantNoiseCfg) -> torch.Tensor:
    """Constant noise."""
    if cfg.operation == "add":
        return data + cfg.bias
    elif cfg.operation == "scale":
        return data * cfg.bias
    elif cfg.operation == "abs":
        return torch.zeros_like(data) + cfg.bias
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")


def uniform_noise(data: torch.Tensor, cfg: noise_cfg.UniformNoiseCfg) -> torch.Tensor:
    """Uniform noise."""
    if cfg.operation == "add":
        return data + torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min
    elif cfg.operation == "scale":
        return data * (torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min)
    elif cfg.operation == "abs":
        return torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")


def gaussian_noise(data: torch.Tensor, cfg: noise_cfg.GaussianNoiseCfg) -> torch.Tensor:
    """Gaussian noise."""
    if cfg.operation == "add":
        return data + cfg.mean + cfg.std * torch.randn_like(data)
    elif cfg.operation == "scale":
        return data * (cfg.mean + cfg.std * torch.randn_like(data))
    elif cfg.operation == "abs":
        return cfg.mean + cfg.std * torch.randn_like(data)
    else:
        raise ValueError(f"Unknown operation in noise: {cfg.operation}")


class NoiseModel:
    """Base class for noise models."""

    def __init__(self, num_envs: int, noise_model_cfg: noise_cfg.NoiseModelCfg):
        """Initialize the noise model.

        Args:
            num_envs: The number of environments.
            noise_model_cfg: The noise configuration to use.
        """
        self._num_envs = num_envs
        self._noise_model_cfg = noise_model_cfg

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        r"""Apply the noise to the data.

        Args:
            data: The data to apply the noise to, which is a tensor of shape (num_envs, \*data_shape).
        """
        return self._noise_model_cfg.noise_cfg.func(data, self._noise_model_cfg.noise_cfg)

    def reset(self, env_ids: Sequence[int]):
        """Reset the noise model.

        This method can be implemented by derived classes to reset the noise model.
        This is useful when implementing temporal noise models such as random walk.

        Args:
            env_ids: The environment ids to reset the noise model for.
        """
        pass


class NoiseModelWithAdditiveBias(NoiseModel):
    """Noise model with an additive bias.

    The bias term is sampled from a the specified distribution on reset.

    """

    def __init__(self, num_envs: int, noise_model_cfg: noise_cfg.NoiseModelWithAdditiveBiasCfg, device: str):
        super().__init__(num_envs, noise_model_cfg)
        self._device = device
        self._bias_noise_cfg = noise_model_cfg.bias_noise_cfg
        self._bias = torch.zeros((num_envs, 1), device=self._device)

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        r"""Apply the noise + bias.

        Args:
            data: The data to apply the noise to, which is a tensor of shape (num_envs, \*data_shape).
        """
        return super().apply(data) + self._bias

    def reset(self, env_ids: Sequence[int]):
        """Reset the noise model.

        This method resets the bias term for the specified environments.

        Args:
            env_ids: The environment ids to reset the noise model for.
        """
        self._bias[env_ids] = self._bias_noise_cfg.func(self._bias[env_ids], self._bias_noise_cfg)
