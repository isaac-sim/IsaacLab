# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PhysX-specific benchmark utilities.

This package provides helper functions for creating benchmark inputs
specific to PhysX-based assets (Articulation, RigidObject, etc.).
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .benchmark_utils import make_tensor_body_ids, make_tensor_env_ids, make_tensor_joint_ids, make_warp_body_mask, make_warp_env_mask, make_warp_joint_mask

from isaaclab.utils.module import lazy_export

lazy_export(
    ("benchmark_utils", [
        "make_tensor_body_ids",
        "make_tensor_env_ids",
        "make_tensor_joint_ids",
        "make_warp_body_mask",
        "make_warp_env_mask",
        "make_warp_joint_mask",
    ]),
)
