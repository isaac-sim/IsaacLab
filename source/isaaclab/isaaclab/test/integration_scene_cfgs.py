# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared core-only scene configurations for Isaac Lab integration tests."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

_CARTPOLE_TEST_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Classic/Cartpole/cartpole.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 2.0),
        joint_pos={"slider_to_cart": 0.0, "cart_to_pole": 0.0},
    ),
    actuators={
        "cart_actuator": ImplicitActuatorCfg(
            joint_names_expr=["slider_to_cart"],
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "pole_actuator": ImplicitActuatorCfg(
            joint_names_expr=["cart_to_pole"],
            effort_limit_sim=400.0,
            stiffness=0.0,
            damping=0.0,
        ),
    },
)


@configclass
class CartpoleTestSceneCfg(InteractiveSceneCfg):
    """Configuration for a minimal cart-pole articulation scene.

    The scene intentionally contains only the robot because its integration-test hosts do
    not assert ground or lighting behavior.
    """

    robot: ArticulationCfg = _CARTPOLE_TEST_CFG.copy()


@configclass
class ArticulationRigidObjectSceneCfg(CartpoleTestSceneCfg):
    """Configuration for a minimal scene with articulation and rigid-object state."""

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyBaseCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionBaseCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.1)),
    )


_RENDERING_CARTPOLE_CFG = _CARTPOLE_TEST_CFG.replace(
    spawn=_CARTPOLE_TEST_CFG.spawn.replace(semantic_tags=[("class", "robot")]),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.9, 0.35, 2.0),
        joint_pos={"slider_to_cart": -0.25, "cart_to_pole": 0.45},
    ),
)


@configclass
class RenderingSceneCfg(InteractiveSceneCfg):
    """Core-only ground, camera slot, and lighting shared by rendering scenes."""

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.CuboidCfg(
            size=(6.0, 6.0, 0.1),
            collision_props=sim_utils.CollisionBaseCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.22, 0.28), roughness=0.8),
            semantic_tags=[("class", "ground")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.05)),
    )
    camera: CameraCfg | None = None
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=2400.0, color=(1.0, 0.91, 0.78), angle=0.7),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.3826834, -0.2209424, 0.0, 0.8968727)),
    )
    fill_light = AssetBaseCfg(
        prim_path="/World/FillLight",
        spawn=sim_utils.DomeLightCfg(intensity=650.0, color=(0.58, 0.68, 1.0)),
    )


@configclass
class RenderingTestSceneCfg(RenderingSceneCfg):
    """Small deterministic scene shared by renderer and visualizer integration tests.

    Assets use deliberate composition poses and distinct materials so a single scene exercises
    geometry, articulation, lighting, depth, segmentation, and motion.
    """

    robot: ArticulationCfg = _RENDERING_CARTPOLE_CFG.copy()
    moving_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/MovingCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.35, 0.35, 0.35),
            rigid_props=sim_utils.RigidBodyBaseCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionBaseCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.12, 0.55, 0.95),
                metallic=0.15,
                roughness=0.25,
            ),
            semantic_tags=[("class", "moving_cube")],
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.15, -0.85, 0.55),
            rot=(0.0, 0.0, 0.1305262, 0.9914449),
            lin_vel=(0.6, 0.25, 0.0),
            ang_vel=(0.0, 0.0, 1.0),
        ),
    )
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(1.15, 0.8, 0.12),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.25, 0.08),
                roughness=0.55,
            ),
            semantic_tags=[("class", "table")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.95, 0.55, 0.46)),
    )
    cylinder = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder",
        spawn=sim_utils.CylinderCfg(
            radius=0.16,
            height=0.42,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.22, 0.18),
                metallic=0.4,
                roughness=0.2,
            ),
            semantic_tags=[("class", "cylinder")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.7, 0.47, 0.73)),
    )
    sphere = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Sphere",
        spawn=sim_utils.SphereCfg(
            radius=0.2,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.28, 0.85, 0.32),
                roughness=0.35,
            ),
            semantic_tags=[("class", "sphere")],
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.18, 0.66, 0.72)),
    )
