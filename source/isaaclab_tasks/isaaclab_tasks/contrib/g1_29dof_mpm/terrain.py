# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Particle layers and support meshes sampled from an Isaac Lab terrain generator."""

from __future__ import annotations

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh
import warp as wp
from isaaclab_newton.sim.schemas import NewtonCollisionCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMPointsCfg
from isaaclab_newton.sim.spawners.mpm.mpm import _generate_grid_particles, spawn_mpm_particles

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.sim.utils import clone
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.warp import raycast_mesh

if TYPE_CHECKING:
    from pxr import Usd


def x_sloped_terrain(difficulty: float, cfg: MeshXSlopeTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Create a planar ramp rising along +X, with its negative-X entrance at Z = 0.

    Author the plane directly so height-field quantization and steep-slope vertex
    corrections cannot introduce steps or vertical faces. The slope is dz/dx.
    """
    length, width = cfg.size
    slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    if not math.isfinite(slope):
        raise ValueError("The terrain slope must be finite.")
    vertices = np.array(
        [(0.0, 0.0, 0.0), (length, 0.0, slope * length), (length, width, slope * length), (0.0, width, 0.0)],
        dtype=np.float64,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]], process=False)
    origin = np.array((0.5 * length, 0.5 * width, 0.5 * slope * length))
    return [mesh], origin


@configclass
class MeshXSlopeTerrainCfg(SubTerrainBaseCfg):
    """A single planar ramp along X, with no slope across Y."""

    function = x_sloped_terrain
    slope_range: tuple[float, float] = (0.0, 0.4)
    """Rise/run at difficulty 0 and 1. Negative values descend along +X."""


def generate_terrain_particles(
    cfg: TerrainMPMGridCfg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, trimesh.Trimesh]:
    """Return terrain-conforming particles and their shared initial surface mesh.

    Grid Z coordinates are depths relative to the generated surface: upper[2]
    must be zero and lower[2] is the negative sand depth. Query the highest
    surface at each jittered particle XY, as in the moving-patch environment.
    Holes and overhang interiors are not filled; a missing height query is an error.
    """
    if cfg.upper[2] != 0.0:
        raise ValueError("Terrain particle upper[2] must be zero (the generated sand surface).")
    if cfg.terrain_generator.seed is None:
        raise ValueError("Set terrain_generator.seed so cloned terrain is reproducible.")
    positions, velocities, masses, radii = _generate_grid_particles(cfg)
    generator = cfg.terrain_generator.class_type(cfg.terrain_generator.copy(), device="cpu")
    surface_mesh = generator.terrain_mesh
    terrain_mesh = wp.Mesh(
        points=wp.array(surface_mesh.vertices, dtype=wp.vec3, device="cpu"),
        indices=wp.array(surface_mesh.faces.flatten(), dtype=wp.int32, device="cpu"),
    )
    ray_starts = torch.from_numpy(positions.copy())
    ray_starts[:, 2] = float(surface_mesh.bounds[1, 2] + 1.0)
    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[:, 2] = -1.0
    hits, _, _, _ = raycast_mesh(
        ray_starts,
        ray_directions,
        terrain_mesh,
        max_dist=float(surface_mesh.bounds[1, 2] - surface_mesh.bounds[0, 2] + 2.0),
    )
    if not torch.isfinite(hits).all():
        raise ValueError("Terrain height query missed the surface; check terrain coverage and spawn spacing.")
    positions[:, 2] += hits.numpy()[:, 2]
    return positions, velocities, masses, radii, surface_mesh


@clone
def spawn_terrain_mpm(
    prim_path: str,
    cfg: TerrainMPMGridCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs: object,
) -> Usd.Prim:
    """Spawn one fixed terrain bed; clone its particles and support geometry together.

    Both solvers use the same generated mesh lowered by the particle-layer depth.
    MPM's support is a kinematic rigid body; MJWarp's copy is static. Geometry is
    generated only during spawning, with no per-step height queries or recycling.
    """
    if not math.isfinite(cfg.wall_thickness) or cfg.wall_thickness <= 0.0:
        raise ValueError("wall_thickness must be positive and finite [m].")
    if not math.isfinite(cfg.wall_height) or cfg.wall_height < 0.0 or not math.isfinite(cfg.entrance_height):
        raise ValueError("wall_height must be non-negative and finite; entrance_height must be finite [m].")
    positions, velocities, masses, radii, surface_mesh = generate_terrain_particles(cfg)
    points_cfg = MPMPointsCfg(
        positions=positions,
        velocities=velocities,
        mass=masses,
        radius=radii,
        material=cfg.material,
        visible=cfg.visible,
        visual_color=cfg.visual_color,
        visual_material=cfg.visual_material,
        visual_update_frequency=cfg.visual_update_frequency,
    )
    root = spawn_mpm_particles(prim_path, points_cfg, translation=translation, orientation=orientation)
    cfg.spawn_path = prim_path
    support_mesh = surface_mesh.copy()
    support_mesh.apply_translation((0.0, 0.0, cfg.lower[2]))
    # Retain particles at the finite bed edges, down to the lowest support height.
    wall_bottom = float(support_mesh.bounds[0, 2]) - cfg.wall_thickness
    wall_top = float(surface_mesh.bounds[1, 2]) + cfg.wall_height
    walls = []
    for axis in range(2):
        for side in range(2):
            top = cfg.entrance_height if axis == 0 and side == 0 else wall_top
            size = [cfg.upper[i] - cfg.lower[i] + 2 * cfg.wall_thickness for i in range(2)]
            center = [0.5 * (cfg.upper[i] + cfg.lower[i]) for i in range(2)]
            size[axis] = cfg.wall_thickness
            center[axis] = (
                cfg.lower[axis] - 0.5 * cfg.wall_thickness if side == 0 else cfg.upper[axis] + 0.5 * cfg.wall_thickness
            )
            if top <= wall_bottom:
                raise ValueError("The retaining wall top must be above the lowest support height.")
            wall = trimesh.creation.box(extents=(*size, top - wall_bottom))
            wall.apply_translation((*center, 0.5 * (top + wall_bottom)))
            walls.append(wall)
    support_mesh = trimesh.util.concatenate([support_mesh, *walls])
    for name, rigid_props in (
        ("MPMBedFloor", sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True)),
        ("RigidBedFloor", None),
    ):
        support_cfg = sim_utils.MeshFileCfg(
            mesh=sim_utils.MeshFileCfg.TrimeshObjectCfg(mesh=support_mesh),
            rigid_props=rigid_props,
            collision_props=[
                sim_utils.UsdPhysicsCollisionCfg(collision_enabled=True),
                NewtonCollisionCfg(contact_margin=cfg.support_contact_margin, contact_gap=0.0),
            ],
            mesh_collision_props=sim_utils.UsdPhysicsMeshCollisionCfg(mesh_approximation_name="none"),
            physics_material=cfg.support_material,
            visible=name == "RigidBedFloor",
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.35, 0.37)),
        )
        support_cfg.func(f"{prim_path}/{name}", support_cfg)
    return root


@configclass
class TerrainMPMGridCfg(MPMGridCfg):
    """Fixed particle layer beneath a generated terrain surface, in the asset frame.

    Every environment clone receives the same mesh and initial particle positions.
    The generator's centered XY domain must cover the particle grid, including
    jitter. Sand depth is measured vertically, rather than along the surface normal.
    """

    func = spawn_terrain_mpm
    terrain_generator: TerrainGeneratorCfg = MISSING
    """Generator defining the initial sand surface and the support mesh shape."""
    support_contact_margin: float = 0.0
    """Contact margin of the generated support meshes [m]."""
    wall_thickness: float = 0.1
    """Thickness of retaining walls outside the particle XY bounds [m]."""
    wall_height: float = 0.1
    """Wall height above the highest generated sand surface [m]."""
    entrance_height: float = -0.05
    """Top of the negative-X retaining wall in the asset frame [m]."""
    support_material: RigidBodyMaterialBaseCfg = RigidBodyMaterialBaseCfg(static_friction=0.9, dynamic_friction=0.8)
