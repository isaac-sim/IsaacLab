# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pathlib
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

FILE_DIR = pathlib.Path(__file__).parent


def get_digit_articulation_cfg(
    fixed_base: bool = False,
    actuator_type: Literal["explicit", "implicit"] = "implicit",
) -> ArticulationCfg:
    """
    Get an articulation config for a Agility Digit robot.
    The robot can be made hanging in the air with the parameter `fixed_base`.
    """
    if actuator_type == "implicit":
        ActuatorCfg = ImplicitActuatorCfg
    elif actuator_type == "explicit":
        ActuatorCfg = IdealPDActuatorCfg

    articulation_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Agility/Digit/digit_v4.usd",
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.05),
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "all": ActuatorCfg(
                joint_names_expr=".*",
                stiffness=None,
                damping=None,
            ),
        },
    )

    if fixed_base:
        articulation_cfg.init_state.pos = (0.0, 0.0, 2.0)
        articulation_cfg.spawn.articulation_props.fix_root_link = True

    return articulation_cfg
