# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Flexiv Rizon4s arm with the 22-DoF Sharpa right hand."""

from isaaclab_newton.sim.schemas import MujocoJointCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from .flexiv import FLEXIV_RIZON4S_CFG

ROBOT_USD_PATH = (
    f"{ISAACLAB_NUCLEUS_DIR}/Environments/RizonSharpaRJ45/Robots/Rizon4sSharpa/rizon4s_sharpa_no_spheres_generated.usd"
)
PALM_BODY_NAME = "r_palm_ctrl"
"""Palm frame with X toward the knuckles and Z out of the palm."""

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
HAND_JOINT_NAMES = (
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
)

HAND_JOINT_LIMITS_RAD = (
    (-0.1745, 1.9199),
    (-0.3491, 0.3491),
    (-0.5236, 1.3963),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (-0.17453293, 1.5708),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (0.0, 1.3963),
    (-0.17453293, 1.5708),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (0.0, 1.3963),
    (-0.17453293, 1.5708),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (0.0, 1.3963),
    (0.0, 0.2618),
    (-0.17453293, 1.5708),
    (-0.3491, 0.3491),
    (0.0, 1.7453),
    (0.0, 1.3963),
)
"""Right-hand limits in :data:`HAND_JOINT_NAMES` order [rad]."""

RIZON4S_SHARPA_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_USD_PATH,
        joint_drive_props={"/Physics/.*": [MujocoJointCfg(actuatorgravcomp=True)]},
        activate_contact_sensors=False,
    ),
    init_state=FLEXIV_RIZON4S_CFG.init_state.copy(),
    actuators={
        # Keep the armature and friction authored on the Sharpa assembly.
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
"""Newton manipulation configuration for the Rizon4s and Sharpa hand."""
