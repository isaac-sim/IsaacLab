# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Panda robot bindings.

The task ships a single robot, so these bind the ``default`` of each robot preset
rather than a named variant -- ``IsaacContrib-Factory-Franka`` needs no
``presets=franka`` to select it.
"""

from __future__ import annotations

__all__: list[str] = []

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import preset

from .. import mdp
from ..assembly_keypoints import PANDA_HAND
from ..factory_assets_cfg import FRANKA_PANDA_NEWTON_CFG, FRANKA_PANDA_PHYSX_CFG
from ..factory_presets import (
    EndEffectorBodyCfg,
    GripperGraspOffsetCfg,
    GripperJointNamesCfg,
    IKJointNamesCfg,
    JointEffortNamesCfg,
    RobotActionsCfg,
    RobotArticulationCfg,
)

EndEffectorBodyCfg.default = "panda_fingertip_centered"
GripperJointNamesCfg.default = ["panda_finger.*"]
IKJointNamesCfg.default = ["panda_joint.*"]
GripperGraspOffsetCfg.default = PANDA_HAND.gripper_center_grasp_point
JointEffortNamesCfg.default = "(?!panda_joint7$|panda_finger_.*$).*"


RobotArticulationCfg.default = preset(
    default=FRANKA_PANDA_PHYSX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    physx=FRANKA_PANDA_PHYSX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
    newton_mjwarp=FRANKA_PANDA_NEWTON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot"),
)


@configclass
class FrankaActionsCfg:
    arm_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale={"(?!panda_joint7).*": 0.02, "panda_joint7": 0.2},
        use_zero_offset=True,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


RobotActionsCfg.default = FrankaActionsCfg()
