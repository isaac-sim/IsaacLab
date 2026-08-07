# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Franka surface deformable lifting environment."""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg
from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.sim.schemas import PhysxDeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import PhysxSurfaceDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass

from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
from isaaclab_contrib.deformable.newton_manager_cfg import NewtonModelCfg, VBDSolverCfg

from isaaclab_tasks.core.lift import mdp
from isaaclab_tasks.utils import PresetCfg

from .franka_soft_env_cfg import (
    FRANKA_CAMERA_CFG,
    FrankaCameraObservationsCfg,
    FrankaSoftEnvCfg,
    _FrankaSoftSceneCfg,
)
from .franka_soft_env_cfg import (
    EventCfg as FrankaSoftEventCfg,
)

##
# Scene definition
##

ROBOT_SHAPE_MATERIAL_MU = 100.0
"""Franka collision-shape friction coefficient [dimensionless] used for Newton cloth contact."""

ROBOT_SHAPE_MATERIAL_BODY_NAMES = ".*"
"""Franka body-name regex receiving :data:`ROBOT_SHAPE_MATERIAL_MU`."""


@configclass
class PhysicsCfg(PresetCfg):
    # Newton physics: MJWarp rigid + VBD soft, coupled through lagged proxies
    newton_mjwarp_vbd_proxy: NewtonCfg = NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name="rigid",
                    solver_cfg=MJWarpSolverCfg(
                        njmax=40,
                        nconmax=20,
                        cone="elliptic",
                        ls_iterations=20,
                        integrator="implicitfast",
                        ccd_iterations=100,
                    ),
                    bodies=[r"/World/envs/env_.*/Robot"],
                ),
                CouplerEntryCfg(
                    name="soft",
                    solver_cfg=VBDSolverCfg(iterations=10),
                    all_particles=True,
                    include_static_shapes=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source="rigid",
                    destination="soft",
                    bodies=[
                        r"/World/envs/env_.*/Robot/panda_hand",
                        r"/World/envs/env_.*/Robot/panda_(left|right)finger",
                    ],
                    collide_interval=1,
                )
            ],
            iterations=1,
            model_cfg=NewtonModelCfg(
                soft_contact_ke=1e3,
                soft_contact_kd=1e-5,
                soft_contact_mu=0.5,
            ),
        ),
        default_shape_cfg=NewtonShapeCfg(ke=1e3, kd=1e-5, mu=1e-4),
        num_substeps=10,
        use_cuda_graph=True,
    )

    ovphysx: OvPhysxCfg = OvPhysxCfg()

    default = newton_mjwarp_vbd_proxy


@configclass
class DeformableCfg(PresetCfg):
    """Preset config for the deformable object, matching the Newton example."""

    newton_mjwarp_vbd_proxy: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Deformable",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.2)),
        spawn=sim_utils.MeshRectangleCfg(
            size=(0.2, 0.2),
            resolution=(30, 30),
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=50.0,
                particle_radius=0.005,
                tri_ke=5e2,
                tri_ka=5e2,
                tri_kd=1e-3,
                edge_ke=2.0,
                edge_kd=1e-3,
            ),
        ),
    )

    ovphysx: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Deformable",
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.4, 0.0, 0.2)),
        spawn=sim_utils.MeshRectangleCfg(
            size=(0.2, 0.2),
            resolution=(30, 30),
            deformable_props=PhysxDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.1)),
            physics_material=PhysxSurfaceDeformableBodyMaterialCfg(
                density=50.0,
                youngs_modulus=2000.0,
                poissons_ratio=0.25,
                surface_thickness=0.005,
                surface_stretch_stiffness=0.8,
                surface_shear_stiffness=0.7,
                surface_bend_stiffness=0.6,
                elasticity_damping=0.03,
                bend_damping=0.04,
            ),
        ),
    )

    default = newton_mjwarp_vbd_proxy


@configclass
class FrankaClothSceneCfg(_FrankaSoftSceneCfg):
    """Scene for the Franka surface deformable environment."""

    deformable: DeformableCfg = DeformableCfg()

    # Static collidable cube the cloth drops onto (sits on the table top at z = 0).
    cube: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 0.0, 0.04)),
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.01, 0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.25)),
        ),
    )

    def __post_init__(self) -> None:
        super().__post_init__()

        # increase franka gripper stiffness
        self.robot.actuators["panda_hand"].effort_limit_sim = 500.0
        self.robot.actuators["panda_hand"].stiffness = 2000.0
        self.robot.actuators["panda_hand"].damping = 100.0


@configclass
class FrankaClothScenePresetCfg(PresetCfg):
    """Preset config for the Franka surface deformable scene."""

    newton_mjwarp_vbd_proxy: FrankaClothSceneCfg = FrankaClothSceneCfg(
        num_envs=128, env_spacing=2.5, replicate_physics=True
    )

    ovphysx: FrankaClothSceneCfg = FrankaClothSceneCfg(num_envs=128, env_spacing=2.5, replicate_physics=True)

    default = newton_mjwarp_vbd_proxy


@configclass
class FrankaClothCameraSceneCfg(FrankaClothSceneCfg):
    """Franka cloth scene with a base camera."""

    base_camera: CameraCfg = FRANKA_CAMERA_CFG


@configclass
class ActionsCfg:
    """7-dim arm joint position + 1-dim binary gripper."""

    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], scale=0.1, use_default_offset=True
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.05},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class EventCfg(FrankaSoftEventCfg):
    """Reset and startup events for the Franka cloth environment."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=ROBOT_SHAPE_MATERIAL_BODY_NAMES),
            "static_friction_range": (ROBOT_SHAPE_MATERIAL_MU, ROBOT_SHAPE_MATERIAL_MU),
            "dynamic_friction_range": (ROBOT_SHAPE_MATERIAL_MU, ROBOT_SHAPE_MATERIAL_MU),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )


def _make_ovphysx_event_cfg() -> EventCfg:
    """Create cloth events that select all robot shapes on OvPhysX."""
    cfg = EventCfg()
    cfg.robot_physics_material.params["asset_cfg"] = SceneEntityCfg("robot")
    return cfg


@configclass
class EventPresetCfg(PresetCfg):
    """Preset config for Franka cloth startup and reset events."""

    newton_mjwarp_vbd_proxy: EventCfg = EventCfg()
    ovphysx: EventCfg = _make_ovphysx_event_cfg()

    default = newton_mjwarp_vbd_proxy


##
# Environment configuration
##


@configclass
class FrankaClothEnvCfg(FrankaSoftEnvCfg):
    """Manager-based RL environment: Franka Panda lifting a surface deformable."""

    # Scene settings
    scene: FrankaClothScenePresetCfg = FrankaClothScenePresetCfg()
    # Basic settings
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    events: EventPresetCfg = EventPresetCfg()

    def __post_init__(self) -> None:
        # general settings
        self.decimation = 1
        self.episode_length_s = 5.0

        # simulation settings
        self.sim.dt = 1 / 60.0
        self.sim.render_interval = self.decimation

        # Hint for the viewport camera when running interactively with --viz kit.
        # Using default_visualizer_cfg rather than visualizer_cfgs avoids forcing
        # Kit viewport creation in kitless/headless contexts.
        from isaaclab_visualizers.kit import KitVisualizerCfg

        self.sim.default_visualizer_cfg = KitVisualizerCfg(
            origin_type="asset", origin_track_path="robot", origin_env_index=0, eye=(1.25, -1.5, 0.6)
        )
        self.sim.physics = PhysicsCfg()


@configclass
class FrankaClothCameraEnvCfg(FrankaClothEnvCfg):
    """Visual Franka surface-deformable lifting environment."""

    scene: FrankaClothCameraSceneCfg = FrankaClothCameraSceneCfg(num_envs=128, env_spacing=2.5, replicate_physics=True)
    observations: FrankaCameraObservationsCfg = FrankaCameraObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # Warm up the RTX render product/annotator (Newton skips the PhysX assets_loading render loop).
        self.num_rerenders_on_reset = 2
