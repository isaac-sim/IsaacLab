# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import Callable

from omni.isaac.orbit.utils import configclass

from . import noise_model


@configclass
class NoiseCfg:
    """Base configuration for a noise term."""

    func: Callable[[torch.Tensor, NoiseCfg], torch.Tensor] = MISSING
    """The function to be called for applying the noise.

    Note:
        The shape of the input and output tensors must be the same.
    """


@configclass
class AdditiveUniformNoiseCfg(NoiseCfg):
    """Configuration for a additive uniform noise term."""

    func = noise_model.additive_uniform_noise

    n_min: float = -1.0
    """The minimum value of the noise. Defaults to -1.0."""
    n_max: float = 1.0
    """The maximum value of the noise. Defaults to 1.0."""


@configclass
class AdditiveGaussianNoiseCfg(NoiseCfg):
    """Configuration for a additive gaussian noise term."""

    func = noise_model.additive_gaussian_noise

    mean: float = 0.0
    """The mean of the noise. Defaults to 0.0."""
    std: float = 1.0
    """The standard deviation of the noise. Defaults to 1.0."""


@configclass
class ConstantBiasNoiseCfg(NoiseCfg):
    """Configuration for a constant bias noise term."""

    func = noise_model.constant_bias_noise

    bias: float = 0.0
    """The bias to add. Defaults to 0.0."""
