# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Layout checks for the Newton views that back Isaac Lab assets."""

from __future__ import annotations

from newton import Model
from newton.selection import ArticulationView

_BOUND_FREQUENCIES = (
    Model.AttributeFrequency.JOINT,
    Model.AttributeFrequency.JOINT_DOF,
    Model.AttributeFrequency.JOINT_COORD,
    Model.AttributeFrequency.BODY,
)


def require_strided_joint_and_body_rows(view: ArticulationView, prim_path: str) -> None:
    """Raise if the view gathers joint or body rows into copies instead of striding over the model arrays.

    Asset data binds joint and body arrays once and reads and writes them in place, so their rows must be
    regularly spaced between the view's worlds. Shape rows may be gathered; shape writers scatter them back.

    Args:
        view: The view backing the asset.
        prim_path: Prim path of the asset, for the error message.

    Raises:
        ValueError: If a joint or body layout of the view uses explicit model indices.
    """
    for frequency in _BOUND_FREQUENCIES:
        layout = view.frequency_layouts.get(frequency)
        if layout is not None and layout.uses_explicit_model_indices:
            raise ValueError(f"Newton {frequency.name} rows of '{prim_path}' are not regularly spaced between worlds.")
