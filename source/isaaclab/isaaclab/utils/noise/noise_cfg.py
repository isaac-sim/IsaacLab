# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING, dataclass
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.utils import config_field

if TYPE_CHECKING:
    from .noise_model import NoiseModel, NoiseModelWithAdditiveBias


@dataclass
class NoiseCfg:
    """Base configuration for a noise term."""

    func: Callable[[torch.Tensor, NoiseCfg], torch.Tensor] = config_field(MISSING)
    """The function to be called for applying the noise.

    Note:
        The shape of the input and output tensors must be the same.
    """
    operation: Literal["add", "scale", "abs"] = config_field("add")
    """The operation to apply the noise on the data. Defaults to "add"."""


@dataclass
class ConstantNoiseCfg(NoiseCfg):
    """Configuration for an additive constant noise term."""

    func: str = config_field("{DIR}.noise_model:constant_noise")

    bias: torch.Tensor | float = config_field(0.0)
    """The bias to add. Defaults to 0.0."""


@dataclass
class UniformNoiseCfg(NoiseCfg):
    """Configuration for a additive uniform noise term."""

    func: str = config_field("{DIR}.noise_model:uniform_noise")

    n_min: torch.Tensor | float = config_field(-1.0)
    """The minimum value of the noise. Defaults to -1.0."""
    n_max: torch.Tensor | float = config_field(1.0)
    """The maximum value of the noise. Defaults to 1.0."""


@dataclass
class GaussianNoiseCfg(NoiseCfg):
    """Configuration for an additive gaussian noise term."""

    func: str = config_field("{DIR}.noise_model:gaussian_noise")

    mean: torch.Tensor | float = config_field(0.0)
    """The mean of the noise. Defaults to 0.0."""
    std: torch.Tensor | float = config_field(1.0)
    """The standard deviation of the noise. Defaults to 1.0."""


##
# Noise models
##


@dataclass
class NoiseModelCfg:
    """Configuration for a noise model."""

    class_type: type[NoiseModel] | str = config_field("{DIR}.noise_model:NoiseModel")
    """The class type of the noise model."""

    noise_cfg: NoiseCfg = config_field(MISSING)
    """The noise configuration to use."""

    func: Callable[[torch.Tensor], torch.Tensor] | None = config_field(None)
    """Function or callable class used by this noise model.

    The function must take a single `torch.Tensor` (the batch of observations) as input
    and return a `torch.Tensor` of the same shape with noise applied.

    It also supports `callable classes <https://docs.python.org/3/reference/datamodel.html#object.__call__>`_,
    i.e. classes that implement the ``__call__()`` method. In this case, the class should inherit from the
    :class:`NoiseModel` class and implement the required methods.

    This field is used internally by :class:ObservationManager and is not meant to be set directly.
    """


@dataclass
class NoiseModelWithAdditiveBiasCfg(NoiseModelCfg):
    """Configuration for an additive gaussian noise with bias model."""

    class_type: type[NoiseModelWithAdditiveBias] | str = config_field("{DIR}.noise_model:NoiseModelWithAdditiveBias")

    bias_noise_cfg: NoiseCfg = config_field(MISSING)
    """The noise configuration for the bias.

    Based on this configuration, the bias is sampled at every reset of the noise model.
    """

    sample_bias_per_component: bool = config_field(True)
    """Whether to sample a separate bias for each data component.

    Defaults to True.
    """
