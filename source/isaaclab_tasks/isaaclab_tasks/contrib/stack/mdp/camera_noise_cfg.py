# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Photometric noise configuration for Franka stack policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseModelCfg, UniformNoiseCfg

if TYPE_CHECKING:
    from .camera import EpisodeCameraNoise


@configclass
class EpisodeCameraNoiseCfg(NoiseModelCfg):
    """Episode-consistent camera randomization with small per-frame pixel noise."""

    class_type: type[EpisodeCameraNoise] | str = "{DIR}.camera:EpisodeCameraNoise"
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
