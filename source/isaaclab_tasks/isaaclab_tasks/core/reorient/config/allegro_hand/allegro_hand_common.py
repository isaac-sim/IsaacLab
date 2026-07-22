# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Allegro Hand identity shared by the Direct and manager-based reorientation tasks.

Asset and marker configurations, joint/body name lists, backend physics
presets, and the sim mixin. No task tunables.
"""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

ALLEGRO_FINGERTIP_BODY_NAMES: list[str] = [
    "index_link_3",
    "middle_link_3",
    "ring_link_3",
    "thumb_link_3",
]
"""Allegro Hand fingertip body names."""

ALLEGRO_ACTUATED_JOINT_NAMES: list[str] = [
    "index_joint_0",
    "middle_joint_0",
    "ring_joint_0",
    "thumb_joint_0",
    "index_joint_1",
    "index_joint_2",
    "index_joint_3",
    "middle_joint_1",
    "middle_joint_2",
    "middle_joint_3",
    "ring_joint_1",
    "ring_joint_2",
    "ring_joint_3",
    "thumb_joint_1",
    "thumb_joint_2",
    "thumb_joint_3",
]
"""Allegro Hand actuated joint names, in the Direct task's actuation order."""


@configclass
class ObjectCfg(PresetCfg):
    physx = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1.2, 1.2, 1.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.17, 0.56), rot=(0.0, 0.0, 0.0, 1.0)),
    )
    newton_mjwarp = ArticulationCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1.2, 1.2, 1.2),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -0.17, 0.565), rot=(0.0, 0.0, 0.0, 1.0), joint_pos={}, joint_vel={}
        ),
        actuators={},
        articulation_root_prim_path="",
    )
    ovphysx = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=400.0),
            scale=(1.2, 1.2, 1.2),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.17, 0.56), rot=(0.0, 0.0, 0.0, 1.0)),
    )
    default = physx


@configclass
class PhysicsCfg(PresetCfg):
    physx = PhysxCfg(
        bounce_threshold_velocity=0.2,
    )
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=80,
            nconmax=70,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=2,
            iterations=100,
            # save_to_mjcf="AllegroHand.xml",
        ),
        num_substeps=2,
        debug_mode=False,
    )
    ovphysx = OvPhysxCfg()
    default = physx


# Scene pieces shared verbatim by the manager-based variant.
ROBOT_CFG = ALLEGRO_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot")
OBJECT_CFG = ObjectCfg()
GOAL_OBJECT_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/goal_marker",
    markers={
        "goal": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(1.2, 1.2, 1.2),
        )
    },
)
