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
from typing import TYPE_CHECKING

from .assets import COFFEE_ASSET_DIR

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg

ASSET_USD_NAME = "rizon4s_sharpa.usda"
ROBOT_ASSET_DIR = f"{COFFEE_ASSET_DIR}/Robot"
PALM_BODY_NAME = "r_palm_ctrl"
"""Canonical palm: +X toward the knuckles, +Z out of the palm."""

DEFAULT_SPOON_OFFSET_POS = (0.19638407055658647, -0.0036995580810314636, 0.07643507665596544)
DEFAULT_SPOON_OFFSET_ROT = (1.0, 0.0, 0.0, 0.0)

ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
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
# Named joints, hard limits [rad] and measured pinch angles [deg], in action order.
_HAND_JOINTS = (
    ("right_thumb_CMC_FE", (-0.1745, 1.9199), 57.326863284),
    ("right_thumb_CMC_AA", (-0.3491, 0.3491), -6.056199517),
    ("right_thumb_MCP_FE", (-0.5236, 1.3963), 39.987930283),
    ("right_thumb_MCP_AA", (-0.3491, 0.3491), -12.580265177),
    ("right_thumb_IP", (0.0, 1.7453), 64.864070603),
    ("right_index_MCP_FE", (-0.17453293, 1.5708), 31.665550149),
    ("right_index_MCP_AA", (-0.3491, 0.3491), 8.050335071),
    ("right_index_PIP", (0.0, 1.7453), 84.817118264),
    ("right_index_DIP", (0.0, 1.3963), 52.919252691),
    ("right_middle_MCP_FE", (-0.17453293, 1.5708), 43.862671108),
    ("right_middle_MCP_AA", (-0.3491, 0.3491), 6.597351456),
    ("right_middle_PIP", (0.0, 1.7453), 75.055080927),
    ("right_middle_DIP", (0.0, 1.3963), 63.272558488),
    ("right_ring_MCP_FE", (-0.17453293, 1.5708), 86.912901978),
    ("right_ring_MCP_AA", (-0.3491, 0.3491), 1.054959321),
    ("right_ring_PIP", (0.0, 1.7453), 81.046617336),
    ("right_ring_DIP", (0.0, 1.3963), 52.093328444),
    ("right_pinky_CMC", (0.0, 0.2618), 6.662629932),
    ("right_pinky_MCP_FE", (-0.17453293, 1.5708), 69.243995079),
    ("right_pinky_MCP_AA", (-0.3491, 0.3491), 3.461576450),
    ("right_pinky_PIP", (0.0, 1.7453), 79.104594737),
    ("right_pinky_DIP", (0.0, 1.3963), 51.017046556),
)
HAND_JOINT_NAMES = tuple(name for name, _, _ in _HAND_JOINTS)
HAND_JOINT_LIMITS = tuple(limits for _, limits, _ in _HAND_JOINTS)
HAND_HOLD_POSITIONS = tuple(math.radians(angle) for _, _, angle in _HAND_JOINTS)
"""Collision-screened pinch at one handle cross-section [rad].

The index distal surface and thumb elastomer oppose each other near the handle
rear, with 0.8 mm cross-finger clearance and no tool penetration. Frictional
retention is not certified; the demo continues to use its mounted utensil.
"""


def get_robot_cfg() -> ArticulationCfg:
    """Return the 29-joint robot at a tabletop-height base with a curled hand."""
    from isaaclab_newton.sim.schemas import MujocoJointCfg, NewtonArticulationCfg

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    from isaaclab_assets.robots.flexiv import FLEXIV_RIZON4S_CFG

    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ROBOT_ASSET_DIR}/{ASSET_USD_NAME}",
            articulation_props=NewtonArticulationCfg(self_collision_enabled=False),
            joint_drive_props={"/Physics/.*": [MujocoJointCfg(actuatorgravcomp=True)]},
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                **dict(zip(ARM_JOINT_NAMES, COFFEE_READY_POSITIONS, strict=True)),
                **dict(zip(HAND_JOINT_NAMES, HAND_HOLD_POSITIONS, strict=True)),
            },
        ),
        actuators={
            **{
                name: actuator.replace(friction=None, armature=None)
                for name, actuator in FLEXIV_RIZON4S_CFG.actuators.items()
            },
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["right_.*"],
                joint_effort_limit=3.3,
                joint_velocity_limit=6.0,
                stiffness=24.0,
                damping=1.2,
            ),
        },
        soft_joint_pos_limit_factor=0.95,
    )
