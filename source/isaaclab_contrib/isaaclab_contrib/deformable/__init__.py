# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Deprecated imports for Newton deformables; use :mod:`isaaclab_newton.assets`."""

import warnings

from isaaclab.utils.module import lazy_export

warnings.warn(
    "isaaclab_contrib.deformable is deprecated; import DeformableObject and DeformableObjectData"
    " from isaaclab_newton.assets instead.",
    DeprecationWarning,
    stacklevel=2,
)

lazy_export()
