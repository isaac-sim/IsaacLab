# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-native noise implementations (experimental).

Re-exports stable configs and classes, then overrides the function-based
noise models (``constant_noise``, ``uniform_noise``, ``gaussian_noise``)
and their configs with Warp-native versions that operate **in-place** on
``wp.array``.

Calling convention (matches Warp MDP terms)::

    noise_cfg.func(data_wp, noise_cfg) -> None   # in-place on wp.array
"""

from isaaclab.utils.noise import *  # noqa: F401,F403

# Override with Warp-native implementations
from .noise_cfg import ConstantNoiseCfg, GaussianNoiseCfg, NoiseCfg, UniformNoiseCfg  # noqa: F401
from .noise_model import NoiseModel, constant_noise, gaussian_noise, uniform_noise  # noqa: F401
