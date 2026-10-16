# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated import path for Warp-native tensor processors."""

import warnings

from .modifier import bias, clip, scale
from .modifier_base import ModifierBase
from .modifier_cfg import ModifierCfg

__all__ = ["ModifierBase", "ModifierCfg", "bias", "clip", "scale"]

warnings.warn(
    "isaaclab_experimental.utils.modifiers is deprecated."
    " Use isaaclab_experimental.utils.tensor_processors instead.",
    DeprecationWarning,
    stacklevel=2,
)
