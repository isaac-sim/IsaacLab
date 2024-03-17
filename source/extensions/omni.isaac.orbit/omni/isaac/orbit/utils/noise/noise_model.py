# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import noise_cfg


def constant_bias_noise(data: torch.Tensor, cfg: noise_cfg.ConstantBiasNoiseCfg) -> torch.Tensor:
    """Add a constant noise."""
    return data + cfg.bias


def additive_uniform_noise(data: torch.Tensor, cfg: noise_cfg.UniformNoiseCfg) -> torch.Tensor:
    """Adds a noise sampled from a uniform distribution."""
    return data + torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min


def additive_gaussian_noise(data: torch.Tensor, cfg: noise_cfg.GaussianNoiseCfg) -> torch.Tensor:
    """Adds a noise sampled from a gaussian distribution."""
    return data + cfg.mean + cfg.std * torch.randn_like(data)
