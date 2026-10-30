# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-native noise configuration (experimental)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

import warp as wp

from isaaclab.utils import configclass

from . import noise_model


@configclass
class NoiseCfg:
    """Configuration for a Warp-native noise term.

    Experimental fork of :class:`isaaclab.utils.noise.NoiseCfg` adapted for the
    Warp-first calling convention where noise functions operate **in-place** on a
    ``wp.array`` buffer and return ``None``.
    """

    func: Callable[[wp.array, NoiseCfg], None] = MISSING
    """The function to be called for applying the noise.

    The function must take a ``wp.array`` as the first argument and the noise
    configuration as the second argument.  It operates **in-place** (no return value).
    """

    operation: Literal["add", "scale", "abs"] = "add"
    """The operation to apply the noise on the data. Defaults to ``"add"``."""


@configclass
class ConstantNoiseCfg(NoiseCfg):
    """Configuration for a constant noise term (Warp-native)."""

    func = noise_model.constant_noise

    bias: float = 0.0
    """The bias to add. Defaults to 0.0."""


@configclass
class UniformNoiseCfg(NoiseCfg):
    """Configuration for a uniform noise term (Warp-native)."""

    func = noise_model.uniform_noise

    n_min: float = -1.0
    """The minimum value of the noise. Defaults to -1.0."""
    n_max: float = 1.0
    """The maximum value of the noise. Defaults to 1.0."""


@configclass
class GaussianNoiseCfg(NoiseCfg):
    """Configuration for a gaussian noise term (Warp-native)."""

    func = noise_model.gaussian_noise

    mean: float = 0.0
    """The mean of the noise. Defaults to 0.0."""
    std: float = 1.0
    """The standard deviation of the noise. Defaults to 1.0."""
