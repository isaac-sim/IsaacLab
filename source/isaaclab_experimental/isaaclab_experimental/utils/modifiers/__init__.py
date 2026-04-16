# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-native modifier implementations (experimental).

Re-exports stable configs and base classes, then overrides the function-based
modifiers (``scale``, ``bias``, ``clip``) with Warp-native versions that
operate in-place on ``wp.array``.

Calling convention (matches Warp MDP terms)::

    modifier.func(data_wp, **params) -> None   # in-place on wp.array
"""

from .modifier import bias, clip, scale  # noqa: F401
from .modifier_base import ModifierBase  # noqa: F401

# Override with Warp-native implementations
from .modifier_cfg import ModifierCfg  # noqa: F401
