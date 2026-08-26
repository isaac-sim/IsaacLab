# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations for OneRobotics A1 robots.

The following configurations are available:

* :obj:`ONEROBOTICS_A1_UNIMANUAL_CFG`: fixed-base 7-DoF right arm.
* :obj:`ONEROBOTICS_A1_CFG`: compatibility alias for the unimanual configuration.

The review-stage configurations retrieve the source URDF and meshes from the
public OneRobotics A1 source repository. Set ``ONEROBOTICS_A1_ASSET_DIR`` to
use a local checkout.

The robot model assets are copyright 2026 OneRobotics and licensed under
CC BY 4.0. The source URDF currently identifies the Link7 mass and inertia as
a v1 flange placeholder; those values are retained without modification.
The review-stage source content was audited at commit
``9e8beb4f1acdea73b1d8edce8919454d8c90d464`` and is protected by asset hashes
in the focused tests. The loader follows the public repository's default branch
until maintainers approve a final hosted asset URI.

References:

* OneRobotics: http://www.onerobot.com/
* Model source: https://github.com/katazen/onerobot_h1
* Asset license: https://github.com/katazen/onerobot_h1/blob/main/ASSET_LICENSE_STATUS.md
"""

import math
import os

from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg, PhysxRigidBodyPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import retrieve_git_asset_path

_ONEROBOTICS_A1_ASSET_REPO_URL = "https://github.com/katazen/onerobot_h1.git"
_ONEROBOTICS_A1_ASSET_SOURCE = os.environ.get("ONEROBOTICS_A1_ASSET_DIR", _ONEROBOTICS_A1_ASSET_REPO_URL)


def _retrieve_a1_asset(relative_path: str) -> str:
    """Retrieve an A1 asset, refreshing an older cache only when needed."""
    try:
        return retrieve_git_asset_path(_ONEROBOTICS_A1_ASSET_SOURCE, relative_path)
    except FileNotFoundError:
        return retrieve_git_asset_path(_ONEROBOTICS_A1_ASSET_SOURCE, relative_path, force_update=True)


_ONEROBOTICS_A1_RIGHT_URDF_PATH = _retrieve_a1_asset("source/h1_reach/h1_reach/assets/urdf/A1_2026/a1_r.urdf")

# OneRobotics hardware values reflected at the arm joints.
_A1_MOTOR_ROTOR_INERTIA = 2.193e-5
_A1_4340_GEAR_RATIO = 48.19
_A1_4310_GEAR_RATIO = 10.0
_A1_4340_ARMATURE = _A1_MOTOR_ROTOR_INERTIA * _A1_4340_GEAR_RATIO**2
_A1_4310_ARMATURE = _A1_MOTOR_ROTOR_INERTIA * _A1_4310_GEAR_RATIO**2
_A1_4340_RATED_SPEED = 25.0 * 2.0 * math.pi / 60.0
_A1_4310_RATED_SPEED = 120.0 * 2.0 * math.pi / 60.0

_A1_JOINT_STIFFNESS = {".*joint[1-4].*": 150.0, ".*joint[5-7].*": 40.0}
_A1_JOINT_DAMPING = {".*joint[1-4].*": 4.0, ".*joint[5-7].*": 1.0}


def _a1_actuators() -> dict[str, ImplicitActuatorCfg]:
    """Create the A1 implicit actuator groups from confirmed hardware values."""
    return {
        "arm_4340": ImplicitActuatorCfg(
            joint_names_expr=[".*joint[1-3].*"],
            actuator_effort_limit=15.0,
            joint_effort_limit=15.0,
            actuator_velocity_limit=_A1_4340_RATED_SPEED,
            joint_velocity_limit=_A1_4340_RATED_SPEED,
            stiffness=150.0,
            damping=4.0,
            armature=_A1_4340_ARMATURE,
        ),
        "arm_4310": ImplicitActuatorCfg(
            joint_names_expr=[".*joint[4-7].*"],
            actuator_effort_limit=3.0,
            joint_effort_limit=3.0,
            actuator_velocity_limit=_A1_4310_RATED_SPEED,
            joint_velocity_limit=_A1_4310_RATED_SPEED,
            stiffness={".*joint4.*": 150.0, ".*joint[5-7].*": 40.0},
            damping={".*joint4.*": 4.0, ".*joint[5-7].*": 1.0},
            armature=_A1_4310_ARMATURE,
        ),
    }


def _a1_urdf_spawn(asset_path: str) -> sim_utils.UrdfFileCfg:
    """Create the common A1 URDF spawn configuration."""
    return sim_utils.UrdfFileCfg(
        asset_path=asset_path,
        fix_base=True,
        self_collision=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=_A1_JOINT_STIFFNESS,
                damping=_A1_JOINT_DAMPING,
            )
        ),
        activate_contact_sensors=False,
        rigid_props=PhysxRigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=PhysxArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    )


ONEROBOTICS_A1_UNIMANUAL_CFG = ArticulationCfg(
    spawn=_a1_urdf_spawn(_ONEROBOTICS_A1_RIGHT_URDF_PATH),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            ".*joint1.*": 0.0,
            ".*joint2.*": -0.6,
            ".*joint3.*": 0.0,
            ".*joint4.*": 1.0,
            ".*joint5.*": 0.0,
            ".*joint6.*": 0.5,
            ".*joint7.*": 0.0,
        },
    ),
    actuators=_a1_actuators(),
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of the OneRobotics A1 fixed-base right arm."""

ONEROBOTICS_A1_CFG = ONEROBOTICS_A1_UNIMANUAL_CFG
"""Compatibility alias for :obj:`ONEROBOTICS_A1_UNIMANUAL_CFG`."""
