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
import os
from typing import TYPE_CHECKING

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

if TYPE_CHECKING:
    from isaaclab.assets import ArticulationCfg

ASSET_REVISION = "d0dbd1ddaefc4996db546949a7dfb37e39afcbeb"
ASSET_BUNDLE_SHA256 = "ae5d22792b44fb6d29a7691d4276bc061a5529132f01e7a0eb5795a482595d63"
ASSET_USD_NAME = "rizon4s_sharpa.usda"
ASSET_ROOT_ENV = "ISAACLAB_FABRICS_SIM_RIZON_SHARPA_ROOT"
ROBOT_ASSET_DIR = os.environ.get(ASSET_ROOT_ENV, f"{ISAACLAB_NUCLEUS_DIR}/Contrib/RizonSharpaCoffee/Robot/v1").rstrip(
    "/"
)
BASE_BODY_NAME = "base_link"
PALM_BODY_NAME = "r_palm_ctrl"
"""Canonical palm: +X toward the knuckles, +Z out of the palm."""

DEFAULT_SPOON_OFFSET_POS = (0.19638407055658647, -0.0036995580810314636, 0.07643507665596544)
DEFAULT_SPOON_OFFSET_ROT = (1.0, 0.0, 0.0, 0.0)

ARM_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8))
ARM_HOME_POSITIONS = (0.0, -0.698, 0.0, 1.571, 0.0, 0.698, 0.0)
"""Reference arm home used by the workstation RJ45 environment [rad]."""
COFFEE_READY_POSITIONS = (
    -1.081903255791,
    0.013804070751,
    0.268143828426,
    2.645703260349,
    2.465847724197,
    0.413775564614,
    -0.366468701795,
)
"""Measured IK branch with the level spoon at ``(0.508, -0.12, 0.22)`` m.

The spoon uses ``DEFAULT_SPOON_OFFSET_POS`` and 180 degrees about palm X.
This places the held utensil above the source bowl, ready to descend.
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


def get_robot_cfg(*, contact_grasp: bool = False) -> ArticulationCfg:
    """Return the 29-joint robot at a tabletop-height base with a curled hand."""
    from isaaclab_newton.sim.schemas import MujocoJointCfg, NewtonArticulationCfg

    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

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
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-2]"],
                joint_effort_limit=123.0,
                joint_velocity_limit=2.094,
                stiffness=6000.0,
                damping=108.5,
            ),
            "elbow": ImplicitActuatorCfg(
                joint_names_expr=["joint[3-4]"],
                joint_effort_limit=64.0,
                joint_velocity_limit=2.443,
                stiffness=4200.0,
                damping=90.7,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["joint[5-7]"],
                joint_effort_limit=39.0,
                joint_velocity_limit=4.887,
                stiffness=1500.0,
                damping=54.2,
            ),
            "hand": ImplicitActuatorCfg(
                joint_names_expr=["right_.*"],
                joint_effort_limit=3.3,
                joint_velocity_limit=6.0,
                stiffness=24.0,
                damping=0.2 if contact_grasp else 1.2,
            ),
        },
        soft_joint_pos_limit_factor=0.95,
    )
