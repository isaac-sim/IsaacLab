# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for RealHand P7 bimanual robots.

The following configurations are available:

* :obj:`P7_L6_CFG`: P7 dual-arm robot with two RealHand L6 dexterous hands.
* :obj:`P7_O6_CFG`: P7 dual-arm robot with two RealHand O6 dexterous hands.
* :obj:`P7_L20_CFG`: P7 dual-arm robot with two RealHand L20 dexterous hands.

The versioned robot assets and their license information are hosted at:
https://huggingface.co/realhandinc/realhand-teleop/tree/main/assets/isaaclab
"""

from isaaclab_physx.sim.schemas import PhysxArticulationCfg, PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_REALHAND_ASSET_REVISION = "4979eb3fbecc7cbb06728a58edfe0c9c6a0a7400"
_REALHAND_ASSET_ROOT = (
    f"https://huggingface.co/realhandinc/realhand-teleop/resolve/{_REALHAND_ASSET_REVISION}/assets/isaaclab/robots"
)

_P7_HOME_JOINT_POS = {
    "left_joint1": 0.4014901088150363,
    "left_joint2": 0.5517304757432392,
    "left_joint3": 0.24490410906952312,
    "left_joint4": -0.888493123948427,
    "left_joint5": -0.4891875163310463,
    "left_joint6": 0.4559396579034241,
    "left_joint7": -0.19427215553367727,
    "right_joint1": 0.4015177394625187,
    "right_joint2": -0.5521344368361245,
    "right_joint3": 0.24572695517009388,
    "right_joint4": 0.8886354864268062,
    "right_joint5": -0.4901601026743809,
    "right_joint6": -0.4556307512290781,
    "right_joint7": 0.19395175851152108,
}

_MIMIC_JOINT_EXPR = [
    ".*_hand_thumb_ip",
    ".*_hand_(index|middle|ring|pinky)_dip",
]


def _make_p7_cfg(
    usd_path: str,
    hand_joint_expr: list[str],
    *,
    mimic_joint_expr: list[str] | None = None,
    hand_armature: float = 0.0,
    hand_velocity_limit: float | None = None,
) -> ArticulationCfg:
    """Build one P7 configuration while keeping model-specific hand joints explicit."""
    hand_actuator_kwargs = {}
    if hand_velocity_limit is not None:
        hand_actuator_kwargs["joint_velocity_limit"] = hand_velocity_limit

    actuators = {
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["left_joint[1-7]", "right_joint[1-7]"],
            joint_effort_limit=300.0,
            stiffness=1100.0,
            damping=55.0,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=hand_joint_expr,
            joint_effort_limit=4.0,
            stiffness=80.0,
            damping=5.0,
            armature=hand_armature,
            **hand_actuator_kwargs,
        ),
    }
    if mimic_joint_expr:
        actuators["hand_mimics"] = ImplicitActuatorCfg(
            joint_names_expr=mimic_joint_expr,
            joint_effort_limit=0.0,
            stiffness=0.0,
            damping=0.0,
            armature=hand_armature,
            **hand_actuator_kwargs,
        )

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=False,
            variants={"Physics": "physx"},
            rigid_props=PhysxRigidBodyCfg(
                disable_gravity=False,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=2.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=[
                PhysxArticulationCfg(
                    enabled_self_collisions=False,
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=4,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.001,
                ),
            ],
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.25),
            joint_pos={**_P7_HOME_JOINT_POS, ".*_hand_.*": 0.0},
            joint_vel={".*": 0.0},
        ),
        actuators=actuators,
        soft_joint_pos_limit_factor=1.0,
    )


P7_L6_CFG = _make_p7_cfg(
    f"{_REALHAND_ASSET_ROOT}/P7_l6_bimanual/P7_l6_bimanual.usda",
    hand_joint_expr=[".*_hand_.*"],
)
"""RealHand P7 bimanual robot with two L6 dexterous hands."""


P7_O6_CFG = _make_p7_cfg(
    f"{_REALHAND_ASSET_ROOT}/P7_o6_bimanual/P7_o6_bimanual.usda",
    hand_joint_expr=[
        ".*_hand_thumb_cmc_(yaw|pitch)",
        ".*_hand_(index|middle|ring|pinky)_mcp_pitch",
    ],
    mimic_joint_expr=_MIMIC_JOINT_EXPR,
    hand_armature=0.001,
)
"""RealHand P7 bimanual robot with two O6 dexterous hands."""


P7_L20_CFG = _make_p7_cfg(
    f"{_REALHAND_ASSET_ROOT}/P7_L20_bimanual/P7_L20_bimanual.usda",
    hand_joint_expr=[
        ".*_hand_thumb_(cmc_roll|cmc_yaw|cmc_pitch|mcp)",
        ".*_hand_(index|middle|ring|pinky)_(mcp_roll|mcp_pitch|pip)",
    ],
    mimic_joint_expr=_MIMIC_JOINT_EXPR,
    hand_armature=0.001,
    hand_velocity_limit=1.0,
)
"""RealHand P7 bimanual robot with two L20 dexterous hands."""
