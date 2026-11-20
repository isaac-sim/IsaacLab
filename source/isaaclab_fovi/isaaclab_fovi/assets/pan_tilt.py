# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""A two-joint pan-tilt mount that turns a camera, for physical and hybrid gaze.

The mount stands on a fixed base. ``pan_joint`` turns about the base's +Z axis (positive turns left) and
``tilt_joint`` about the panned +X axis (positive turns up). At zero the mount faces +Y; a camera prim under
``tilt_link`` at :data:`PAN_TILT_CAMERA_OFFSET` looks along it with +Z up. The asset has no visual geometry.
"""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import CameraCfg

PAN_TILT_USD_PATH = str(Path(__file__).with_name("pan_tilt.usda"))
"""The mount's USD, converted from its URDF with fixed joints merged; it loads with or without Kit."""

PAN_TILT_CAMERA_OFFSET = CameraCfg.OffsetCfg(
    pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.70710678, 0.70710678), convention="world"
)
"""Pose of a camera under ``tilt_link`` that looks along the mount's forward axis (world convention)."""

PAN_TILT_STEREO_BASELINE = 0.065
"""Distance between the stereo eyes [m], about a human's."""

PAN_TILT_STEREO_OFFSETS = tuple(
    PAN_TILT_CAMERA_OFFSET.replace(pos=(side * PAN_TILT_STEREO_BASELINE / 2, 0.0, 0.0)) for side in (-1.0, 1.0)
)
"""Poses (left, right) of two cameras under ``tilt_link`` that look along the mount's forward axis, for binocular
gaze."""

PAN_TILT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/PanTilt",
    # Gravity stays on: Newton reads a body's disable-gravity flag as the whole scene's.
    spawn=sim_utils.UsdFileCfg(usd_path=PAN_TILT_USD_PATH, fix_root_link=True),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={"pan_joint": 0.0, "tilt_joint": 0.0}),
    actuators={
        "pan_tilt": ImplicitActuatorCfg(
            joint_names_expr=["pan_joint", "tilt_joint"],
            stiffness=10.0,
            damping=1.4,
            joint_effort_limit=20.0,
            armature=0.05,
        )
    },
    soft_joint_pos_limit_factor=1.0,
)
"""The pan-tilt mount: about critically damped position drives that settle in about 0.3 s.

The armature carries the drives' inertia: the links themselves are too light for a stable stiff drive at Isaac Lab's
usual physics step."""
