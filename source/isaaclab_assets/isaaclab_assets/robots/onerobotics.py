# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the OneRobotics A1 robotic arm.

The :obj:`ONEROBOTICS_A1_CFG` configuration represents the canonical fixed-base,
7-DoF right arm controlled by implicit joint-position actuators. The review-stage
integration retrieves the source URDF and meshes from the public OneRobotics A1
source repository. Set ``ONEROBOTICS_A1_ASSET_DIR`` to use a local checkout.

The robot model assets are copyright 2026 OneRobotics and licensed under CC BY 4.0.
The expected review-stage content was audited at commit
``ca6d705f37b0dc296bfe7f33f7c83d780c3d3a70`` and is protected by asset hashes in
the focused tests. The temporary loader initially retrieves the public repository's
default branch and then reuses its local cache until maintainers approve a final
hosted asset URI.

References:

* OneRobotics: http://www.onerobot.com/
* Model source: https://github.com/katazen/onerobot_h1
* Asset license: https://github.com/katazen/onerobot_h1/blob/ca6d705f37b0dc296bfe7f33f7c83d780c3d3a70/ASSET_LICENSE_STATUS.md
"""

import os

from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg, PhysxRigidBodyPropertiesCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import retrieve_git_asset_path

_ONEROBOTICS_A1_ASSET_REPO_URL = "https://github.com/katazen/onerobot_h1.git"
_ONEROBOTICS_A1_ASSET_SOURCE = os.environ.get("ONEROBOTICS_A1_ASSET_DIR", _ONEROBOTICS_A1_ASSET_REPO_URL)
_ONEROBOTICS_A1_URDF_PATH = retrieve_git_asset_path(
    _ONEROBOTICS_A1_ASSET_SOURCE,
    "source/h1_reach/h1_reach/assets/urdf/A1/a1_right.urdf",
)

ONEROBOTICS_A1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_ONEROBOTICS_A1_URDF_PATH,
        fix_base=True,
        self_collision=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness={".*joint[1-4].*": 60.0, ".*joint[5-7].*": 30.0},
                damping={".*joint[1-4].*": 6.0, ".*joint[5-7].*": 3.0},
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
    ),
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
    actuators={
        "arm_proximal": ImplicitActuatorCfg(
            joint_names_expr=[".*joint[1-4].*"],
            joint_effort_limit=30.0,
            joint_velocity_limit=3.0,
            stiffness=60.0,
            damping=6.0,
            # Preserved from the validated Isaac Lab integration for numerical stability.
            # Replace only when measured reflected motor inertia becomes available.
            armature=0.05,
        ),
        "arm_distal": ImplicitActuatorCfg(
            joint_names_expr=[".*joint[5-7].*"],
            joint_effort_limit=12.0,
            joint_velocity_limit=3.0,
            stiffness=30.0,
            damping=3.0,
            armature=0.05,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of the OneRobotics A1 fixed-base right arm."""
