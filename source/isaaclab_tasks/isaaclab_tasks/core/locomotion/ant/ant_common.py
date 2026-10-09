# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configurations shared by the direct and manager-based Ant environments."""

from isaaclab_newton.physics import KaminoPADMMSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils
from isaaclab.physics import PhysxAutoCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.visualizers import TrackingCameraCfg, VisualizerCfg

from isaaclab_tasks.utils import PresetCfg

VISUALIZER_CFG = VisualizerCfg(
    eye=(3.5, -1.0, 2.8),
    lookat=(0.0, 0.0, 0.0),
    focal_length=26.0,
    streaming_envs=[0],
    cameras=[
        TrackingCameraCfg(
            track_path="robot",
            follow_heading=True,
            heading_smoothing_time_constant=0.5,
        ),
    ],
)
"""Recording view following the Ant with smoothed heading."""

TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="plane",
    collision_group=-1,
    physics_material=sim_utils.PhysxRigidBodyMaterialCfg(
        friction_combine_mode="average",
        restitution_combine_mode="average",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    ),
    debug_vis=False,
)
"""Flat ground plane the Ant walks on."""

JOINT_GEARS: dict[str, float] = {".*": 15.0}
"""Effort scale per joint [N·m], keyed by joint name expression."""

FEET_BODY_NAMES: list[str] = ["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
"""Bodies whose incoming wrench is observed."""

WALK_TARGET_POS: tuple[float, float, float] = (1000.0, 0.0, 0.0)
"""Walk target [m] relative to the environment origin, far enough away that it is never reached."""


@configclass
class AntPhysicsCfg(PresetCfg):
    """Physics backend presets for the Ant environments."""

    isaacsim_physx: PhysxCfg = PhysxCfg(bounce_threshold_velocity=0.2)
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=45,
            nconmax=25,
            cone="pyramidal",
            integrator="implicitfast",
            impratio=1,
        ),
        num_substeps=1,
        debug_mode=False,
    )
    newton_kamino: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoPADMMSolverCfg(sparse_jacobian=True),
        debug_mode=False,
        use_cuda_graph=True,
    )
    default: NewtonCfg = newton_mjwarp
