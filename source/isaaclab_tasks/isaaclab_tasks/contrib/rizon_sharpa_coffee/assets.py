# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Hosted coffee props and dimensions used by particle seeding and tool motion."""

import os

import numpy as np

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

COFFEE_ASSET_DIR = os.environ.get(
    "ISAACLAB_COFFEE_ASSET_ROOT", f"{ISAACLAB_NUCLEUS_DIR}/Environments/RizonSharpaCoffee"
).rstrip("/")
"""USD bundle root; override to use an internal development release or local bundle."""

TABLE_TOP_Z = 0.0
BOWL_POSITION = (0.52, -0.12, TABLE_TOP_Z)
CUP_POSITION = (0.60, 0.02, TABLE_TOP_Z)
BOWL_INNER_RADIUS = 0.065
BOWL_HEIGHT = 0.040
BOWL_BOTTOM_THICKNESS = 0.004
CUP_INNER_RADIUS = 0.039
CUP_HEIGHT = 0.056
CUP_BOTTOM_THICKNESS = 0.004
SPOON_BOWL_LENGTH = 0.050
SPOON_BOWL_WIDTH = 0.032
COFFEE_HEAP_APEX_HEIGHT = 0.080

# Ordered from the flat cavity bottom to its upper rim. This profile also owns
# particle seeding clearance, so changes to the bowl do not invalidate its fill.
_BOWL_INNER_PROFILE = (
    (0.038, BOWL_BOTTOM_THICKNESS),
    (0.043, 0.006),
    (0.049, 0.012),
    (0.056, 0.022),
    (0.062, 0.032),
    (BOWL_INNER_RADIUS, BOWL_HEIGHT - 0.001),
    (0.066, BOWL_HEIGHT),
)


def coffee_cavity_radius(height: float | np.ndarray) -> float | np.ndarray:
    """Return the circular seed radius through the cavity and an above-rim heap.

    The circular profile fits the centered oval cavity below its rim.
    Above the rim it tapers to zero at ``COFFEE_HEAP_APEX_HEIGHT`` without adding
    collision walls. Seeders should subtract their particle radius and collider
    margin from this radius. Heights outside the profile return zero.
    """
    profile = np.asarray((*_BOWL_INNER_PROFILE, (0.0, COFFEE_HEAP_APEX_HEIGHT)))
    radius = np.interp(height, profile[:, 1], profile[:, 0], left=0.0, right=0.0)
    return float(radius) if np.isscalar(height) else radius
