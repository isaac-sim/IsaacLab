# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Validation shared by Newton's batched rigid-asset views."""

from newton import Model
from newton.selection import ArticulationView


def _validate_asset_view(view: ArticulationView, asset_kind: str) -> None:
    """Require matching asset layouts while allowing unrelated scene shapes to vary."""
    required = ("JOINT", "JOINT_DOF", "JOINT_COORD", "BODY", "SHAPE")
    unavailable = [name for name in required if getattr(Model.AttributeFrequency, name) not in view.frequency_layouts]
    root_metadata = (view.root_joint_type, view.is_fixed_base, view.is_floating_base)
    if view.is_sparse or unavailable or any(value is None for value in root_metadata):
        raise ValueError(f"{asset_kind} instances require matching layouts in every world: {unavailable}.")
