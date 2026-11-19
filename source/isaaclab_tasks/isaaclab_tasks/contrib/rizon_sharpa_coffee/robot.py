# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rizon4s/Sharpa asset and joint contract shared with the workstation RJ45 demo.

The prepared USD and textures are hosted separately with their original
Fabrics-Sim licensing. See the demo README for provisioning and provenance.
"""

from __future__ import annotations

import math

from isaaclab_newton.sim.schemas import NewtonArticulationCfg

from isaaclab_assets.robots.rizon_sharpa import ARM_JOINT_NAMES, HAND_JOINT_NAMES, RIZON4S_SHARPA_CFG

from .assets import COFFEE_ASSET_DIR

ASSET_USD_NAME = "rizon4s_sharpa.usda"
ROBOT_ASSET_DIR = f"{COFFEE_ASSET_DIR}/Robot"

DEFAULT_SPOON_OFFSET_POS = (0.19638407055658647, -0.0036995580810314636, 0.07643507665596544)
DEFAULT_SPOON_OFFSET_ROT = (1.0, 0.0, 0.0, 0.0)

COFFEE_READY_POSITIONS = (
    -0.704524636269,
    -0.584716141224,
    0.548297226429,
    1.619004607201,
    2.447559595108,
    2.105417251587,
    -0.419155776501,
)
"""Measured IK branch with the spoon at ``(0.4995, -0.1265, 0.09)`` m and 60° pitch.

The spoon uses ``DEFAULT_SPOON_OFFSET_POS`` and 180 degrees about palm X.
This places the held utensil close above the source bowl, ready to descend.
"""
# Measured pinch angles [deg].
_PINCH_ANGLES_DEG = {
    "right_thumb_CMC_FE": 57.326863284,
    "right_thumb_CMC_AA": -6.056199517,
    "right_thumb_MCP_FE": 39.987930283,
    "right_thumb_MCP_AA": -12.580265177,
    "right_thumb_IP": 64.864070603,
    "right_index_MCP_FE": 31.665550149,
    "right_index_MCP_AA": 8.050335071,
    "right_index_PIP": 84.817118264,
    "right_index_DIP": 52.919252691,
    "right_middle_MCP_FE": 43.862671108,
    "right_middle_MCP_AA": 6.597351456,
    "right_middle_PIP": 75.055080927,
    "right_middle_DIP": 63.272558488,
    "right_ring_MCP_FE": 86.912901978,
    "right_ring_MCP_AA": 1.054959321,
    "right_ring_PIP": 81.046617336,
    "right_ring_DIP": 52.093328444,
    "right_pinky_CMC": 6.662629932,
    "right_pinky_MCP_FE": 69.243995079,
    "right_pinky_MCP_AA": 3.46157645,
    "right_pinky_PIP": 79.104594737,
    "right_pinky_DIP": 51.017046556,
}
HAND_HOLD_POSITIONS = tuple(math.radians(_PINCH_ANGLES_DEG[name]) for name in HAND_JOINT_NAMES)
"""Collision-screened pinch at one handle cross-section [rad].

The index distal surface and thumb elastomer oppose each other near the handle
rear, with 0.8 mm cross-finger clearance and no tool penetration. Frictional
retention is not certified; the demo continues to use its mounted utensil.
"""


COFFEE_ROBOT_CFG = RIZON4S_SHARPA_CFG.replace(
    spawn=RIZON4S_SHARPA_CFG.spawn.replace(
        usd_path=f"{ROBOT_ASSET_DIR}/{ASSET_USD_NAME}",
        articulation_props=NewtonArticulationCfg(self_collision_enabled=False),
    ),
    init_state=RIZON4S_SHARPA_CFG.init_state.replace(
        joint_pos={
            **dict(zip(ARM_JOINT_NAMES, COFFEE_READY_POSITIONS, strict=True)),
            **dict(zip(HAND_JOINT_NAMES, HAND_HOLD_POSITIONS, strict=True)),
        },
    ),
)
