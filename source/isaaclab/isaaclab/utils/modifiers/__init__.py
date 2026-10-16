# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated import path for tensor processors."""

import warnings

from ..module import lazy_export

warnings.warn(
    "isaaclab.utils.modifiers is deprecated. Use isaaclab.utils.tensor_processors instead.",
    DeprecationWarning,
    stacklevel=2,
)

lazy_export()
