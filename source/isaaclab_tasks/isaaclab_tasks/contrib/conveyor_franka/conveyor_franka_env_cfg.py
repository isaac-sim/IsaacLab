# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the force-driven conveyor and Franka demonstration scene."""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.franka import FRANKA_PANDA_MENAGERIE_CFG

from .conveyor_geometry import (
    BELT_CENTER_X,
    BELT_CENTER_Y,
    BELT_COLOR,
    BELT_TURN_RADIUS,
    GUARD_COLOR,
    PARCEL_COLOR,
    MeshSpec,
    belt_mesh_spec,
    guard_mesh_specs,
)

_DYNAMIC_PROPERTIES = sim_utils.RigidBodyBaseCfg()


def _srgb_to_linear_channel(value: float) -> float:
    """Convert an sRGB channel to the linear value expected by USD displayColor."""
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


@configclass
class ActionsCfg:
    """Empty action configuration for zero-action scene playback."""

    pass


@configclass
class ObservationsCfg:
    """Empty observation configuration while the task objective is being designed."""

    pass


@configclass
class ConveyorForceCfg:
    """Configuration for force-based conveyor traction."""

    speed: float = 0.35
    """Tangential conveyor surface speed [m/s]."""

    friction: float = 0.5
    """Coulomb friction coefficient used to limit traction."""

    normal_threshold: float = 0.95
    """Minimum upward contact-normal alignment in the range [0, 1]."""

    def __post_init__(self) -> None:
        """Validate conveyor force parameters."""
        if self.speed < 0.0:
            raise ValueError(f"Conveyor speed must be non-negative, got {self.speed}.")
        if self.friction < 0.0:
            raise ValueError(f"Conveyor friction must be non-negative, got {self.friction}.")
        if not 0.0 <= self.normal_threshold <= 1.0:
            raise ValueError(f"Conveyor normal threshold must be in [0, 1], got {self.normal_threshold}.")


@sim_utils.clone
def _spawn_shape_with_display_color(
    prim_path: str,
    cfg: sim_utils.ShapeCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn a primitive and author a renderer-independent USD display color."""
    if isinstance(cfg, sim_utils.CuboidCfg):
        prim = sim_utils.spawn_cuboid(prim_path, cfg, translation, orientation, **kwargs)
    elif isinstance(cfg, sim_utils.CylinderCfg):
        prim = sim_utils.spawn_cylinder(prim_path, cfg, translation, orientation, **kwargs)
    elif isinstance(cfg, sim_utils.CapsuleCfg):
        prim = sim_utils.spawn_capsule(prim_path, cfg, translation, orientation, **kwargs)
    elif isinstance(cfg, sim_utils.SphereCfg):
        prim = sim_utils.spawn_sphere(prim_path, cfg, translation, orientation, **kwargs)
    else:
        raise TypeError(f"Unsupported colored primitive configuration: {type(cfg).__name__}")

    if cfg.visual_material is not None:
        from pxr import Usd, UsdGeom

        display_color = tuple(_srgb_to_linear_channel(value) for value in cfg.visual_material.diffuse_color)
        for child in Usd.PrimRange(prim):
            if child.IsA(UsdGeom.Gprim):
                UsdGeom.Gprim(child).CreateDisplayColorAttr([display_color])
    return prim


def _static_cuboid(
    prim_path: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
    rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    friction: float = 0.7,
    roughness: float = 0.75,
    metallic: float = 0.0,
) -> AssetBaseCfg:
    """Build a static colliding cuboid configuration."""
    spawn = sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionBaseCfg(),
        physics_material=RigidBodyMaterialBaseCfg(
            static_friction=friction,
            dynamic_friction=friction,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=color,
            roughness=roughness,
            metallic=metallic,
        ),
    )
    spawn.func = _spawn_shape_with_display_color
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
        spawn=spawn,
    )


def _static_mesh(
    prim_path: str,
    spec: MeshSpec,
    color: tuple[float, float, float],
    friction: float,
    roughness: float,
    metallic: float,
) -> AssetBaseCfg:
    """Build a static colliding triangle-mesh configuration."""
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.MeshCustomCfg(
            vertices=spec.vertices,
            faces=spec.faces,
            collision_props=sim_utils.CollisionBaseCfg(),
            physics_material=RigidBodyMaterialBaseCfg(
                static_friction=friction,
                dynamic_friction=friction,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=roughness,
                metallic=metallic,
            ),
        ),
    )


def _parcel(
    name: str,
    spawn: sim_utils.ShapeCfg,
    pos: tuple[float, float, float],
) -> RigidObjectCfg:
    """Build a dynamic parcel configuration."""
    spawn.rigid_props = _DYNAMIC_PROPERTIES
    spawn.mass_props = sim_utils.MassPropertiesCfg(mass=0.25)
    spawn.collision_props = sim_utils.CollisionBaseCfg()
    spawn.physics_material = RigidBodyMaterialBaseCfg(
        # The force driver supplies traction explicitly. This is just above MuJoCo's
        # minimum valid coefficient and mirrors Newton's force-conveyor example.
        static_friction=1.1e-5,
        dynamic_friction=1.1e-5,
        restitution=0.05,
    )
    spawn.func = _spawn_shape_with_display_color
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
        spawn=spawn,
    )


@configclass
class ConveyorFrankaSceneCfg(InteractiveSceneCfg):
    """Scene with two counter-rotating racetrack conveyors around a table-mounted Franka."""

    # Use the MuJoCo Menagerie-derived model with Newton's MuJoCo MJWarp solver.
    robot = FRANKA_PANDA_MENAGERIE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.35,
                "panda_joint3": 0.0,
                "panda_joint4": -2.35,
                "panda_joint5": 0.0,
                "panda_joint6": 2.0,
                "panda_joint7": 0.78,
                "panda_finger_joint.*": 0.04,
            }
        ),
    )

    tabletop = _static_cuboid(
        prim_path="{ENV_REGEX_NS}/Tabletop",
        size=(2.0, 1.9, 0.08),
        pos=(0.50, 0.0, -0.04),
        color=(0.32, 0.34, 0.37),
    )
    table_pedestal = _static_cuboid(
        prim_path="{ENV_REGEX_NS}/TablePedestal",
        size=(0.75, 0.55, 0.76),
        pos=(0.25, 0.0, -0.46),
        color=(0.18, 0.20, 0.23),
    )

    parcel_left_box = _parcel(
        "ParcelLeftBox",
        sim_utils.CuboidCfg(
            size=(0.075, 0.055, 0.06),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=PARCEL_COLOR, roughness=0.8),
        ),
        (BELT_CENTER_X - 0.12, BELT_CENTER_Y + BELT_TURN_RADIUS, 0.085),
    )
    parcel_left_cylinder = _parcel(
        "ParcelLeftCylinder",
        sim_utils.CylinderCfg(
            radius=0.032,
            height=0.065,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.48, 0.82), roughness=0.8),
        ),
        (BELT_CENTER_X + 0.18, BELT_CENTER_Y - BELT_TURN_RADIUS, 0.085),
    )
    parcel_right_box = _parcel(
        "ParcelRightBox",
        sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.075),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.86, 0.34, 0.12), roughness=0.8),
        ),
        (BELT_CENTER_X + 0.14, -BELT_CENTER_Y + BELT_TURN_RADIUS, 0.0925),
    )
    parcel_right_capsule = _parcel(
        "ParcelRightCapsule",
        sim_utils.CapsuleCfg(
            radius=0.026,
            height=0.075,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.34, 0.68, 0.28), roughness=0.8),
        ),
        (BELT_CENTER_X - 0.18, -BELT_CENTER_Y - BELT_TURN_RADIUS, 0.095),
    )

    parcel_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Parcel.*",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.85)),
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=2500.0),
    )

    def __post_init__(self) -> None:
        """Generate both racetrack belts and their inner/outer guardrails."""
        for side in ("Left", "Right"):
            belt_spec = belt_mesh_spec(side)
            setattr(
                self,
                f"conveyor_{side.lower()}_belt",
                _static_mesh(
                    prim_path=f"{{ENV_REGEX_NS}}/{belt_spec.name}",
                    spec=belt_spec,
                    color=BELT_COLOR,
                    # MuJoCo requires a tiny positive value even though the force driver,
                    # rather than solver friction, supplies the belt motion.
                    friction=1.1e-5,
                    roughness=0.9,
                    metallic=0.0,
                ),
            )

            for spec in guard_mesh_specs(side):
                boundary = "inner" if spec.name.endswith("Inner") else "outer"
                setattr(
                    self,
                    f"guard_{side.lower()}_{boundary}",
                    _static_mesh(
                        prim_path=f"{{ENV_REGEX_NS}}/{spec.name}",
                        spec=spec,
                        color=GUARD_COLOR,
                        friction=0.2,
                        roughness=0.3,
                        metallic=0.8,
                    ),
                )


@configclass
class ConveyorFrankaEnvCfg(ManagerBasedRLEnvCfg):
    """Manager-based environment configuration for the conveyor Franka scene."""

    scene: ConveyorFrankaSceneCfg = ConveyorFrankaSceneCfg(num_envs=1, env_spacing=3.0, replicate_physics=True)
    conveyor_force: ConveyorForceCfg = ConveyorForceCfg()
    # MDP managers will be populated once the manipulation objective is defined.
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards = None
    terminations = None
    decimation: int = 1
    episode_length_s: float = 1.0e6

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(
                solver="newton",
                integrator="implicitfast",
                njmax=300,
                nconmax=256,
                impratio=10.0,
                cone="elliptic",
                update_data_interval=2,
                iterations=100,
                ls_iterations=15,
                ls_parallel=False,
                use_mujoco_contacts=False,
                ccd_iterations=35,
            ),
            collision_cfg=NewtonCollisionPipelineCfg(),
            default_shape_cfg=NewtonShapeCfg(),
            num_substeps=2,
            use_cuda_graph=False,
            load_visual_shapes=True,
        ),
    )

    def __post_init__(self) -> None:
        self.seed = 42
        # Frame the complete robot and both conveyor lanes in the Newton viewer.
        from isaaclab_visualizers.newton import NewtonVisualizerCfg

        self.sim.default_visualizer_cfg = NewtonVisualizerCfg(
            eye=(2.3, -2.7, 1.8),
            lookat=(0.45, 0.0, 0.35),
        )
