# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""CPU-only native-PhysX configuration for conveyor-Franka policy playback."""

from __future__ import annotations

import functools
from dataclasses import replace

from isaaclab_physx.physics import PhysxCfg, apply_surface_velocity_api
from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxSDFMeshCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.physics import SurfaceVelocitySpec
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas import CollisionFragment, UsdPhysicsCollisionCfg
from isaaclab.utils.configclass import configclass

from .conveyor_franka_env_cfg import (
    _CONTACT_GAP,
    _CUBE_CONTACT_MARGIN,
    _ROBOT_INITIAL_STATE,
    ConveyorFrankaEnvCfg,
    ConveyorFrankaSceneCfg,
    _spawn_hidden_collision_mesh,
    _validate_common_config,
)
from .conveyor_geometry import (
    ConveyorSectionSpec,
    CuboidSpec,
    MeshSpec,
    belt_collision_section_specs,
    guard_mesh_specs,
)
from .franka_robot_cfg import FRANKA_PANDA_CONVEYOR_PHYSX_CFG

_PHYSX_KINEMATIC_PROPERTIES = sim_utils.RigidBodyBaseCfg(
    rigid_body_enabled=True,
    kinematic_enabled=True,
    disable_gravity=True,
)


def _physx_collision_properties(contact_offset: float = 0.005) -> list[CollisionFragment]:
    """Build the standard collision and PhysX offset fragments for one collider."""
    return [
        UsdPhysicsCollisionCfg(collision_enabled=True),
        PhysxCollisionCfg(contact_offset=contact_offset, rest_offset=0.0),
    ]


def _physx_material(friction: float) -> PhysxRigidBodyMaterialCfg:
    """Build a deterministic PhysX material for a conveyor-task collision surface."""
    return PhysxRigidBodyMaterialCfg(
        static_friction=friction,
        dynamic_friction=friction,
        restitution=0.0,
        friction_combine_mode="min",
        restitution_combine_mode="min",
    )


def _pivot_centered_section(section: ConveyorSectionSpec) -> tuple[ConveyorSectionSpec, tuple[float, float, float]]:
    """Express a curved section around its rigid-body origin for native angular surface velocity."""
    if not section.belt.curved:
        geometry = section.geometry
        if not isinstance(geometry, CuboidSpec):
            raise TypeError("Straight PhysX conveyor sections must use native cuboids.")
        return section, geometry.position

    geometry = section.geometry
    if not isinstance(geometry, MeshSpec):
        raise TypeError("Curved PhysX conveyor sections must use closed meshes.")
    pivot = section.belt.pivot_point
    local_geometry = MeshSpec(
        name=geometry.name,
        vertices=tuple(
            (vertex[0] - pivot[0], vertex[1] - pivot[1], vertex[2] - pivot[2]) for vertex in geometry.vertices
        ),
        faces=geometry.faces,
    )
    local_belt = replace(section.belt, pivot_point=(0.0, 0.0, 0.0))
    return ConveyorSectionSpec(geometry=local_geometry, belt=local_belt), pivot


def physx_belt_section_specs(
    side: str,
    *,
    velocity: float = 0.0,
    friction_coefficient: float = 0.5,
    contact_threshold: float = 0.997,
) -> tuple[tuple[ConveyorSectionSpec, tuple[float, float, float]], ...]:
    """Return pivot-centered sections sharing the exact runtime PhysX semantics."""
    return tuple(
        _pivot_centered_section(section)
        for section in belt_collision_section_specs(
            side,
            velocity=velocity,
            friction_coefficient=friction_coefficient,
            contact_threshold=contact_threshold,
        )
    )


@sim_utils.clone
def _spawn_physx_conveyor(
    prim_path: str,
    cfg: sim_utils.MeshCustomCfg | sim_utils.CuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    *,
    belt_spec: SurfaceVelocitySpec,
    **kwargs,
):
    """Spawn a hidden belt section and author native surface velocity before PhysX parsing."""
    spawn = sim_utils.spawn_mesh_custom if isinstance(cfg, sim_utils.MeshCustomCfg) else sim_utils.spawn_cuboid
    prim = spawn(prim_path, cfg, translation, orientation, **kwargs)
    sim_utils.set_prim_visibility(prim, False)
    apply_surface_velocity_api(prim, belt_spec, velocity_scale=0.0)
    return prim


def _physx_conveyor_collision(
    prim_path: str,
    section: ConveyorSectionSpec,
    root_position: tuple[float, float, float],
    friction: float,
) -> AssetBaseCfg:
    """Build one native-velocity belt section with analytic or watertight-SDF collision."""
    geometry = section.geometry
    if isinstance(geometry, CuboidSpec):
        spawn = sim_utils.CuboidCfg(
            size=geometry.size,
            visible=False,
            rigid_props=_PHYSX_KINEMATIC_PROPERTIES,
            collision_props=_physx_collision_properties(contact_offset=_CONTACT_GAP),
            physics_material=_physx_material(friction),
        )
    else:
        spawn = sim_utils.MeshCustomCfg(
            vertices=geometry.vertices,
            faces=geometry.faces,
            visible=False,
            rigid_props=_PHYSX_KINEMATIC_PROPERTIES,
            collision_props=[
                *_physx_collision_properties(contact_offset=_CONTACT_GAP),
                PhysxSDFMeshCfg(sdf_resolution=128, sdf_subgrid_resolution=6),
            ],
            collision_approximation="sdf",
            physics_material=_physx_material(friction),
        )
    spawn.func = functools.partial(_spawn_physx_conveyor, belt_spec=section.belt)
    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=root_position),
        spawn=spawn,
    )


def _physx_guard_collision(prim_path: str, spec: MeshSpec) -> AssetBaseCfg:
    """Build one freely sliding static guide collider."""
    spawn = sim_utils.MeshCustomCfg(
        vertices=spec.vertices,
        faces=spec.faces,
        visible=False,
        collision_props=_physx_collision_properties(),
        collision_approximation="none",
        physics_material=_physx_material(1.1e-5),
    )
    spawn.func = _spawn_hidden_collision_mesh
    return AssetBaseCfg(prim_path=prim_path, spawn=spawn)


@configclass
class ConveyorFrankaPhysxSceneCfg(ConveyorFrankaSceneCfg):
    """PhysX scene preserving the Newton task's names, layout, and tensor contracts."""

    robot = FRANKA_PANDA_CONVEYOR_PHYSX_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=_ROBOT_INITIAL_STATE,
    )

    def __post_init__(self) -> None:
        """Reuse the racetrack layout and visuals, replacing only backend physics properties."""
        super().__post_init__()
        for support in (self.tabletop, self.table_pedestal):
            support.spawn.collision_props = _physx_collision_properties()
            support.spawn.physics_material = _physx_material(0.7)
        for cube_id in range(4):
            cube = getattr(self, f"cube_{cube_id}")
            cube.spawn.collision_props = _physx_collision_properties(contact_offset=_CUBE_CONTACT_MARGIN)
            cube.spawn.physics_material = PhysxRigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.6,
                restitution=0.0,
                restitution_combine_mode="min",
            )
        for side in ("Left", "Right"):
            section_keys = ("top_straight", "bottom_straight", "right_turn", "left_turn")
            for section_key, (section, root_position) in zip(section_keys, physx_belt_section_specs(side), strict=True):
                setattr(
                    self,
                    f"conveyor_{side.lower()}_{section_key}_collision",
                    _physx_conveyor_collision(
                        prim_path=f"{{ENV_REGEX_NS}}/{section.geometry.name}",
                        section=section,
                        root_position=root_position,
                        friction=0.5,
                    ),
                )

            for guard in guard_mesh_specs(side):
                boundary = "inner" if guard.name.endswith("Inner") else "outer"
                setattr(
                    self,
                    f"guard_{side.lower()}_{boundary}_collision",
                    _physx_guard_collision(f"{{ENV_REGEX_NS}}/{guard.name}Collision", guard),
                )

    def build_conveyor_belt_specs(
        self,
        *,
        velocity: float,
        friction_coefficient: float,
        contact_threshold: float,
    ) -> tuple[SurfaceVelocitySpec, ...]:
        """Return the same pivot-local belt descriptions used by the PhysX spawners."""
        return tuple(
            section.belt
            for side in ("Left", "Right")
            for section, _ in physx_belt_section_specs(
                side,
                velocity=velocity,
                friction_coefficient=friction_coefficient,
                contact_threshold=contact_threshold,
            )
        )

    def configure_conveyor(self, *, friction_coefficient: float) -> None:
        """Propagate a final command-line friction override into every belt material before spawning."""
        for side in ("left", "right"):
            for section_key in ("top_straight", "bottom_straight", "right_turn", "left_turn"):
                asset = getattr(self, f"conveyor_{side}_{section_key}_collision")
                material = asset.spawn.physics_material
                material.static_friction = friction_coefficient
                material.dynamic_friction = friction_coefficient


@configclass
class ConveyorFrankaPhysxEnvCfg(ConveyorFrankaEnvCfg):
    """Checkpoint-compatible CPU reference using native PhysX surface velocity.

    The supported and tested native ``PhysxSurfaceVelocityAPI`` path is CPU-only. Use
    :class:`ConveyorFrankaEnvCfg` for scalable GPU simulation with Newton.
    """

    scene: ConveyorFrankaPhysxSceneCfg = ConveyorFrankaPhysxSceneCfg(
        # USD surface-velocity commands are authored on the host. One environment
        # is the useful default for this CPU reference; callers may opt into a
        # small replicated batch explicitly.
        num_envs=1,
        env_spacing=3.0,
        replicate_physics=True,
    )
    sim: SimulationCfg = SimulationCfg(
        # The supported GPU contact-modification path drops contacts for shapes
        # with PhysxSurfaceVelocityAPI enabled. CPU PhysX preserves the authored
        # API and its belt contacts; fail validation on an accidental CUDA
        # override instead of letting cubes tunnel silently.
        device="cpu",
        dt=1.0 / 120.0,
        render_interval=2,
        physics=PhysxCfg(
            solver_type=1,
            solve_articulation_contact_last=True,
            max_position_iteration_count=64,
            max_velocity_iteration_count=16,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            enable_ccd=True,
        ),
        use_fabric=True,
        use_newton_actuators=True,
    )

    def validate_config(self) -> None:
        """Validate PhysX selection and the shared checkpoint-facing policy contract."""
        _validate_common_config(self)
        if not isinstance(self.sim.physics, PhysxCfg):
            raise ValueError("The PhysX conveyor task requires the native Isaac Sim PhysX backend.")
        if self.sim.device != "cpu":
            raise ValueError(
                "The native PhysX conveyor is CPU-only because enabled PhysxSurfaceVelocityAPI shapes can lose "
                "contacts under GPU dynamics. Run this task with '--device cpu'; use the Newton task for GPU "
                "simulation."
            )
        if self.scene.robot.spawn.joint_drive_props is not None:
            raise ValueError("The PhysX Franka must not author MuJoCo-only joint-drive properties.")
        if self.scene.robot.spawn.rigid_props.disable_gravity is not True:
            raise ValueError("The PhysX Franka must preserve the trained gravity-compensated policy contract.")
