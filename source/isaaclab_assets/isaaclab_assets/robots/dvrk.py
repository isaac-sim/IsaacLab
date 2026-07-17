# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the dVRK Patient Side Manipulator (PSM).

The following configuration is available:

* :obj:`DVRK_PSM_CFG`: The fixed-base dVRK PSM surgical robot.

Asset provenance:

* Catalogue: `Isaac for Healthcare asset catalogue <https://github.com/isaac-for-healthcare/i4h-asset-catalog>`__
* Catalogue release: ``v0.6.0`` (tag commit ``bee7e9314bb8f1c78f7e178a7840d708eda9ffb1``)
* Content revision: ``c189487``
* Source: ``https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/Healthcare/0.6.0/c189487/Robots/dVRK/PSM/psm.usd``
* Catalogue licence: `Apache License 2.0 <https://github.com/isaac-for-healthcare/i4h-asset-catalog/blob/v0.6.0/LICENSE>`__

The USD has SHA-256 ``5730339c3b806f17a5228c69b97464d0b3469888002f62fb23d9621f746347c8``
and default prim ``/psm``.
"""

import math

from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from .dvrk_asset import DVRK_PSM_USD_PATH
from .dvrk_asset import DVRK_PSM_USD_SHA256 as DVRK_PSM_USD_SHA256

##
# Asset metadata and articulation names
##

DVRK_PSM_DEFAULT_PRIM_PATH = "/psm"
"""Default prim authored by the pinned dVRK PSM USD."""

DVRK_PSM_ARM_JOINT_NAMES = [
    "psm_yaw_joint",
    "psm_pitch_end_joint",
    "psm_main_insertion_joint",
    "psm_tool_roll_joint",
    "psm_tool_pitch_joint",
    "psm_tool_yaw_joint",
]
"""Ordered names of the six PSM arm joints."""

DVRK_PSM_JAW_JOINT_NAMES = ["psm_tool_gripper1_joint", "psm_tool_gripper2_joint"]
"""Ordered names of the two PSM jaw joints."""

DVRK_PSM_TOOL_TIP_BODY_NAME = "psm_tool_tip_link"
"""Name of the PSM end-effector body."""

DVRK_PSM_JAW_BODY_NAMES = ["psm_tool_gripper1_link", "psm_tool_gripper2_link"]
"""Ordered names of the two PSM jaw collision bodies."""

DVRK_PSM_JAW_JOINT_LIMITS = ((-math.pi / 6.0, 0.0), (0.0, math.pi / 6.0))
"""Joint limits [rad] of the pinned PSM jaws, in :data:`DVRK_PSM_JAW_JOINT_NAMES` order."""

DVRK_PSM_JAW_OPEN_POS = (-math.pi / 6.0, math.pi / 6.0)
"""Fully-open jaw endpoint [rad], in :data:`DVRK_PSM_JAW_JOINT_NAMES` order."""

DVRK_PSM_JAW_CLOSED_POS = (0.0, 0.0)
"""Fully-closed jaw endpoint [rad], in :data:`DVRK_PSM_JAW_JOINT_NAMES` order."""


def _validate_jaw_endpoint(endpoint: tuple[float, float], name: str) -> None:
    """Check a jaw command endpoint against the limits authored in the pinned USD."""
    for joint_name, value, (lower, upper) in zip(
        DVRK_PSM_JAW_JOINT_NAMES, endpoint, DVRK_PSM_JAW_JOINT_LIMITS, strict=True
    ):
        if not lower <= value <= upper:
            raise ValueError(f"{name} value for {joint_name} ({value}) is outside [{lower}, {upper}].")


_validate_jaw_endpoint(DVRK_PSM_JAW_OPEN_POS, "DVRK_PSM_JAW_OPEN_POS")
_validate_jaw_endpoint(DVRK_PSM_JAW_CLOSED_POS, "DVRK_PSM_JAW_CLOSED_POS")

##
# Configuration
##

DVRK_PSM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=DVRK_PSM_USD_PATH,
        # Contact sensors in a task can filter this reporting to DVRK_PSM_JAW_BODY_NAMES.
        activate_contact_sensors=True,
        articulation_props=PhysxArticulationRootPropertiesCfg(
            fix_root_link=True,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "psm_yaw_joint": 0.0,
            "psm_pitch_end_joint": 0.0,
            "psm_main_insertion_joint": 0.12,
            "psm_tool_roll_joint": 0.0,
            "psm_tool_pitch_joint": 0.0,
            "psm_tool_yaw_joint": 0.0,
            "psm_tool_gripper1_joint": DVRK_PSM_JAW_OPEN_POS[0],
            "psm_tool_gripper2_joint": DVRK_PSM_JAW_OPEN_POS[1],
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=DVRK_PSM_ARM_JOINT_NAMES,
            stiffness=None,
            damping=None,
            effort_limit_sim=None,
            velocity_limit_sim=None,
        ),
        "jaws": ImplicitActuatorCfg(
            joint_names_expr=DVRK_PSM_JAW_JOINT_NAMES,
            stiffness=None,
            damping=None,
            effort_limit_sim=None,
            velocity_limit_sim=None,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of the fixed-base dVRK PSM with USD-authored drives and joint limits."""
