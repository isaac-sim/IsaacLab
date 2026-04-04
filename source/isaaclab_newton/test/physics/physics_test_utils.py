# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared utilities for physics and sensor testing.

This module provides common functionality used across multiple test files:
- Shape type definitions and helpers
- Simulation configuration
- Shape spawning utilities
"""

import time
from contextlib import contextmanager
from enum import Enum, auto

import pytest
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import SimulationCfg

##
# Newton Configuration
##


def make_sim_cfg(
    use_mujoco_contacts: bool = False, device: str = "cuda:0", gravity: tuple = (0.0, 0.0, -9.81)
) -> SimulationCfg:
    """Create simulation configuration with specified collision pipeline.

    Args:
        use_mujoco_contacts: If True, use MuJoCo contact pipeline. If False, use Newton contact pipeline.
        device: Device to run simulation on ("cuda:0" or "cpu").
        gravity: Gravity vector (x, y, z).

    Returns:
        SimulationCfg configured for the specified collision pipeline.
    """
    solver_cfg = MJWarpSolverCfg(
        njmax=200,
        ls_iterations=20,
        cone="elliptic",
        ls_parallel=False,
        integrator="implicitfast",
        use_mujoco_contacts=use_mujoco_contacts,
    )

    newton_cfg = NewtonCfg(
        solver_cfg=solver_cfg,
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=False,
    )

    return SimulationCfg(
        dt=1.0 / 120.0,
        device=device,
        gravity=gravity,
        create_stage_in_memory=False,
        physics=newton_cfg,
    )


COLLISION_PIPELINES = [
    pytest.param(False, id="newton_contacts"),
    pytest.param(True, id="mujoco_contacts"),
]


##
# Shape Types
##


class ShapeType(Enum):
    SPHERE = auto()
    BOX = auto()
    CAPSULE = auto()
    CYLINDER = auto()
    CONE = auto()
    MESH_SPHERE = auto()
    MESH_BOX = auto()
    MESH_CAPSULE = auto()
    MESH_CYLINDER = auto()
    MESH_CONE = auto()


PRIMITIVE_SHAPES = [ShapeType.SPHERE, ShapeType.BOX, ShapeType.CAPSULE, ShapeType.CYLINDER, ShapeType.CONE]
MESH_SHAPES = [
    ShapeType.MESH_SPHERE,
    ShapeType.MESH_BOX,
    ShapeType.MESH_CAPSULE,
    ShapeType.MESH_CYLINDER,
    ShapeType.MESH_CONE,
]
ALL_SHAPES = PRIMITIVE_SHAPES + MESH_SHAPES

STABLE_SHAPES = [
    ShapeType.SPHERE,
    ShapeType.BOX,
    ShapeType.CAPSULE,
    ShapeType.CYLINDER,
    ShapeType.MESH_SPHERE,
    ShapeType.MESH_BOX,
    ShapeType.MESH_CAPSULE,
    ShapeType.MESH_CYLINDER,
]

BOX_SHAPES = [ShapeType.BOX, ShapeType.MESH_BOX]
SPHERE_SHAPES = [ShapeType.SPHERE, ShapeType.MESH_SPHERE]


def shape_type_to_str(shape_type: ShapeType) -> str:
    return shape_type.name.lower()


def is_mesh_shape(shape_type: ShapeType) -> bool:
    return shape_type in MESH_SHAPES


##
# Shape Configuration Factory
##


def create_shape_cfg(
    shape_type: ShapeType,
    prim_path: str,
    pos: tuple = (0, 0, 1.0),
    disable_gravity: bool = True,
    activate_contact_sensors: bool = True,
) -> RigidObjectCfg:
    """Create RigidObjectCfg for a shape type."""
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=disable_gravity,
        linear_damping=0.0,
        angular_damping=0.0,
    )
    collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
    mass_props = sim_utils.MassPropertiesCfg(mass=1.0)

    spawner_map = {
        ShapeType.SPHERE: lambda: sim_utils.SphereCfg(
            radius=0.25,
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            activate_contact_sensors=activate_contact_sensors,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.8)),
        ),
        ShapeType.BOX: lambda: sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            activate_contact_sensors=activate_contact_sensors,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.8, 0.4)),
        ),
        ShapeType.CAPSULE: lambda: sim_utils.CapsuleCfg(
            radius=0.15,
            height=0.3,
            axis="Z",
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            activate_contact_sensors=activate_contact_sensors,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.4)),
        ),
        ShapeType.CYLINDER: lambda: sim_utils.CylinderCfg(
            radius=0.2,
            height=0.4,
            axis="Z",
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            activate_contact_sensors=activate_contact_sensors,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.4)),
        ),
        ShapeType.CONE: lambda: sim_utils.ConeCfg(
            radius=0.25,
            height=0.5,
            axis="Z",
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            activate_contact_sensors=activate_contact_sensors,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.8)),
        ),
        ShapeType.MESH_SPHERE: lambda: sim_utils.MeshSphereCfg(
            radius=0.25,
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            activate_contact_sensors=activate_contact_sensors,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.6)),
        ),
        ShapeType.MESH_BOX: lambda: sim_utils.MeshCuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            activate_contact_sensors=activate_contact_sensors,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.2)),
        ),
        ShapeType.MESH_CAPSULE: lambda: sim_utils.MeshCapsuleCfg(
            radius=0.15,
            height=0.3,
            axis="Z",
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            activate_contact_sensors=activate_contact_sensors,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.2, 0.2)),
        ),
        ShapeType.MESH_CYLINDER: lambda: sim_utils.MeshCylinderCfg(
            radius=0.2,
            height=0.4,
            axis="Z",
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            activate_contact_sensors=activate_contact_sensors,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.6, 0.2)),
        ),
        ShapeType.MESH_CONE: lambda: sim_utils.MeshConeCfg(
            radius=0.25,
            height=0.5,
            axis="Z",
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            activate_contact_sensors=activate_contact_sensors,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.2, 0.6)),
        ),
    }

    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=spawner_map[shape_type](),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos),
    )


def get_shape_extent(shape_type: ShapeType) -> float:
    """Get the XY extent (radius/half-size) of a shape for positioning."""
    extents = {
        ShapeType.SPHERE: 0.25,
        ShapeType.BOX: 0.25,
        ShapeType.CAPSULE: 0.15,
        ShapeType.CYLINDER: 0.20,
        ShapeType.CONE: 0.25,
        ShapeType.MESH_SPHERE: 0.25,
        ShapeType.MESH_BOX: 0.25,
        ShapeType.MESH_CAPSULE: 0.15,
        ShapeType.MESH_CYLINDER: 0.20,
        ShapeType.MESH_CONE: 0.25,
    }
    return extents[shape_type]


def get_shape_height(shape_type: ShapeType) -> float:
    """Get the height of a shape for stacking calculations."""
    heights = {
        ShapeType.SPHERE: 0.5,
        ShapeType.BOX: 0.5,
        ShapeType.CAPSULE: 0.6,
        ShapeType.CYLINDER: 0.4,
        ShapeType.CONE: 0.5,
        ShapeType.MESH_SPHERE: 0.5,
        ShapeType.MESH_BOX: 0.5,
        ShapeType.MESH_CAPSULE: 0.6,
        ShapeType.MESH_CYLINDER: 0.4,
        ShapeType.MESH_CONE: 0.5,
    }
    return heights[shape_type]


@contextmanager
def phase_timer(label: str):
    """Context manager that prints elapsed wall-clock time for a test phase."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    print(f"  [{label}] {elapsed:.2f}s")


def perform_sim_step(sim, scene, dt: float):
    """Perform a single simulation step and update scene."""
    scene.write_data_to_sim()
    sim.step(render=True)
    scene.update(dt=dt)
