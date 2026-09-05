# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Allegro Hand identity shared by the Direct and manager-based reorientation tasks.

Asset and marker configurations, joint/body name lists, backend physics
presets, and the sim mixin. No task tunables.
"""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG

ALLEGRO_HAND_ROBOT_CFG = ALLEGRO_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

CUBE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/object",
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
        collision_props=sim_utils.CollisionPropertiesCfg(
            mesh_collision_property=sim_utils.MeshCollisionPropertiesCfg(mesh_approximation_name="convexHull")
        ),
        scale=(1.2, 1.2, 1.2),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.17, 0.56), rot=(0.0, 0.0, 0.0, 1.0)),
)
"""In-hand cube for the Allegro reorientation task."""


@configclass
class PhysicsCfg(PresetCfg):
    isaacsim_physx = PhysxCfg(
        bounce_threshold_velocity=0.2,
    )
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            integrator="implicitfast",
            njmax=80,
            nconmax=70,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=2,
        ),
        num_substeps=2,
    )
    ovphysx = OvPhysxCfg()
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    default = newton_mjwarp


GOAL_OBJECT_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/goal_marker",
    markers={
        "goal": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(1.2, 1.2, 1.2),
        )
    },
)
