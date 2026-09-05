# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility aliases for supported Newton releases."""

from typing import Any

import newton

try:
    from newton import ModelFlags
except ImportError:
    from newton.solvers import SolverNotifyFlags as ModelFlags


def refit_shape_bvh(model: Any, state: Any) -> None:
    """Refit a model's shape BVH using the API available in the installed Newton release."""
    if hasattr(model, "bvh_refit_shapes"):
        model.bvh_refit_shapes(state)
    else:
        newton.geometry.refit_bvh_shape(model, state)


__all__ = ["ModelFlags", "refit_shape_bvh"]
