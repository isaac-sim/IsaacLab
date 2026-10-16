# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Warp-native processor implementations (experimental).

Re-exports stable configs and base classes, then overrides the function-based
tensor processors (``scale``, ``bias``, ``clip``) with Warp-native versions that
operate in-place on ``wp.array``.

Calling convention (matches Warp MDP terms)::

    processor.func(data_wp, **params) -> None   # in-place on wp.array
"""

from .processors import bias, clip, scale  # noqa: F401
from .tensor_processor import TensorProcessor  # noqa: F401

# Override with Warp-native implementations
from .tensor_processor_cfg import TensorProcessorCfg  # noqa: F401
