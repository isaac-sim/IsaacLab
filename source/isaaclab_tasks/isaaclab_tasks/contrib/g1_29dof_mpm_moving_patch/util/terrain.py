# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Textured scene terrain and generated support geometry shared by moving-patch environments."""

from __future__ import annotations

import math
from dataclasses import MISSING

import numpy as np
import trimesh
import warp as wp
from isaaclab_newton.sim.spawners.mpm import MPMParticleMaterialCfg

from pxr import Sdf, UsdGeom, UsdShade, Vt

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass

from .kernel import sample_heights


class TexturedTerrainImporter(TerrainImporter):
    """Add UV coordinates and a USD preview texture alongside the configured MDL material."""

    def import_mesh(self, name: str, mesh: trimesh.Trimesh) -> None:
        """Import scene geometry with planar UVs and an MDL-compatible preview material."""
        if any(self.cfg.mesh_origin_offset):
            mesh = mesh.copy()
            mesh.apply_translation(self.cfg.mesh_origin_offset)
        super().import_mesh(name, mesh)
        stage = sim_utils.get_current_stage()
        path = f"{self.cfg.prim_path}/{name}"
        usd_mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"{path}/mesh"))
        cfg = self.cfg
        uvs = np.asarray((mesh.vertices[:, :2] - mesh.bounds[0, :2]) * cfg.texture_repeat_per_meter, dtype=np.float32)
        UsdGeom.PrimvarsAPI(usd_mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        ).Set(Vt.Vec2fArray.FromNumpy(uvs))

        # Preserve MDL for RTX and provide the standard USD texture network for GL.
        material_path = f"{path}/visualMaterial"
        material = UsdShade.Material.Define(stage, material_path)
        preview = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        preview.CreateIdAttr("UsdPreviewSurface")
        diffuse = preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
        if cfg.color_texture:
            reader = UsdShade.Shader.Define(stage, f"{material_path}/TextureCoordinates")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            texture = UsdShade.Shader.Define(stage, f"{material_path}/ColorTexture")
            texture.CreateIdAttr("UsdUVTexture")
            texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(cfg.color_texture))
            texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
            texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            diffuse.ConnectToSource(texture.ConnectableAPI(), "rgb")
        else:
            diffuse.Set(cfg.color)
        preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        preview.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(usd_mesh.GetPrim()).Bind(material)


@configclass
class TexturedTerrainImporterCfg(TerrainImporterCfg):
    """Standard terrain import with a portable color texture for GL and RTX."""

    class_type: type = TexturedTerrainImporter

    mesh_origin_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Translation of imported mesh vertices along the terrain's local axes [m].

    Does not shift environment origins or the source mesh. Only applies to mesh imports.
    """
    visual_material: sim_utils.MdlFileCfg | None = None
    color_texture: str | None = None
    """Color texture for the USD preview material; None uses color."""
    texture_repeat_per_meter: float = 0.125
    """Number of texture repeats per world-space distance [1/m]."""
    color: tuple[float, float, float] = (0.72, 0.55, 0.34)
    """Untextured preview material RGB color."""


@configclass
class MovingPatchTerrainCfg:
    """Geometry and material settings for the simulated patch and its kinematic boundary."""

    simulated_terrain_size: tuple[float, float] = MISSING
    """Width of continuously simulated terrain [m], along X and Y."""
    boundary_terrain_size: float = MISSING
    """Width of the zero-mass boundary on each side [m]."""
    tracked_body: str = "base"
    """Exact robot body name used to center the patch in each environment."""
    particle_depth: float = MISSING
    """Vertical particle-layer depth below the undisturbed terrain surface [m]."""
    robot_spawn_height: float = MISSING
    """Initial robot root height above the sand surface [m]."""
    patch_discretization_step: float = MISSING
    """Patch-center quantization distance [m]."""
    voxel_size: float = MISSING
    """MPM background-grid voxel size [m]."""
    particles_per_cell: float = MISSING
    """Particle samples along each voxel dimension (IsaacLab MPMGridCfg convention)."""
    jitter: float = MISSING
    """Total width of the initial particle-position jitter interval [m].

    Each X, Y, and Z coordinate receives an independent uniform offset in
    ``[-jitter / 2, jitter / 2]``, matching :class:`MPMGridCfg`.
    """
    material: MPMParticleMaterialCfg = MISSING
    """Particle density and constitutive parameters, shared by initial and recycled particles."""
    floor_thickness: float = MISSING
    """Thickness of the static supporting floor below the sand [m]."""
    show_boundary_particles: bool = False
    """Render the kinematic boundary along with simulated particles."""
    visual_color: tuple[float, float, float] = (0.72, 0.55, 0.34)
    """Particle-emitter RGB color; viewer particle_color controls rendered particles."""

    @property
    def total_patch_size(self) -> tuple[float, float]:
        """Return the simulated-plus-boundary extent [m]."""
        simulated_size_x, simulated_size_y = self.simulated_terrain_size
        return (
            simulated_size_x + 2.0 * self.boundary_terrain_size,
            simulated_size_y + 2.0 * self.boundary_terrain_size,
        )

    def validate_geometry(self) -> None:
        """Reject invalid geometry before creating the scene."""
        for dimension_name in ("simulated_terrain_size",):
            dimensions = getattr(self, dimension_name)
            if len(dimensions) != 2 or not all(math.isfinite(dimension) for dimension in dimensions):
                raise ValueError(f"{dimension_name} must contain two finite values")
        positive_dimensions = (
            *self.simulated_terrain_size,
            self.boundary_terrain_size,
            self.particle_depth,
            self.patch_discretization_step,
            self.voxel_size,
            self.particles_per_cell,
            self.floor_thickness,
        )
        if not all(math.isfinite(dimension) and dimension > 0 for dimension in positive_dimensions):
            raise ValueError("Terrain dimensions, sampling and patch_discretization_step must be positive and finite")
        if not math.isfinite(self.jitter) or self.jitter < 0.0:
            raise ValueError("Particle jitter width must be finite and non-negative [m]")
        if not self.tracked_body or self.patch_discretization_step > self.boundary_terrain_size:
            raise ValueError("Set tracked_body and keep patch_discretization_step <= boundary_terrain_size")
        if self.boundary_terrain_size < 2 * self.voxel_size:
            raise ValueError("Use at least two MPM voxels of boundary terrain")


class WarpTerrainMesh:
    """Wrap terrain vertices and faces in a Warp mesh for GPU height queries.

    The mesh provides accelerated ray queries; bounds and environment origins are
    used for particle placement. Particle rendering reads simulation positions.
    """

    def __init__(self, vertices: np.ndarray, faces: np.ndarray, device: str):
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.int32).reshape(-1, 3)
        self.bounds = np.stack((self.vertices.min(axis=0), self.vertices.max(axis=0)))
        self.mesh = wp.Mesh(
            points=wp.array(self.vertices, dtype=wp.vec3, device=device),
            indices=wp.array(self.faces.flatten(), dtype=int, device=device),
        )
        self.height_query_start_z = float(self.bounds[1, 2] + 1.0)
        self.height_query_max_distance = float(self.bounds[1, 2] - self.bounds[0, 2] + 2.0)
        self.env_origins = None
        """Initial environment reset positions on the background terrain, in world coordinates [m]."""
        self.env_clone_origins = None
        """Environment clone-grid positions used to author the initial particles, in world coordinates [m]."""

    def sample_surface_heights(self, query_positions: wp.array) -> None:
        """Replace world-space query points' Z coordinates by surface heights [m]."""
        terrain_query_miss_count = wp.zeros(1, dtype=int, device=self.mesh.device)
        wp.launch(
            sample_heights,
            dim=len(query_positions),
            inputs=[
                self.mesh.id,  # mesh
                self.height_query_start_z,  # ray_z
                self.height_query_max_distance,  # ray_length
            ],
            outputs=[query_positions, terrain_query_miss_count],
            device=self.mesh.device,
        )
        if terrain_query_miss_count.numpy()[0]:
            raise ValueError("Shared terrain height query missed the surface; check terrain coverage and spawn spacing")


class BackgroundTerrainImporter(TexturedTerrainImporter):
    """Use one IsaacLab terrain with a global support collider for each subsolver."""

    def __init__(self, cfg: BackgroundTerrainImporterCfg):
        cfg.validate_geometry()
        super().__init__(cfg)

    def import_mesh(self, name: str, mesh: trimesh.Trimesh):
        """Keep the generated surface for height queries and lower the supporting collider."""
        self.background_mesh = WarpTerrainMesh(mesh.vertices, mesh.faces, self.device)
        # Preserve the generated surface; only the supporting collider is lowered.
        support_mesh = mesh.copy()
        support_mesh.apply_translation((0, 0, -self.cfg.moving_patch_terrain.particle_depth))

        # physics_cfg.py assigns "terrain" to MJWarp and "mpm_ground_mesh" to MPM.
        # The coupler does not allow one collider to belong to both solvers.
        super().import_mesh(name, support_mesh)
        super().import_mesh("mpm_ground", support_mesh)
        # Both solvers need a collider, but only one copy of the surface should be drawn.
        UsdGeom.Imageable(
            sim_utils.get_current_stage().GetPrimAtPath(f"{self.cfg.prim_path}/mpm_ground")
        ).MakeInvisible()

    def _is_heightfield_collider_requested(self, cfg):
        # The installed implicit MPM collision path consumes triangle meshes.
        return False

    def configure_env_origins(self, origins=None):
        """Place robots on grid or terrain-tile origins and sample spawn heights [m]."""
        super().configure_env_origins(origins)
        # Particles are authored at clone-grid origins, independently of reset tile selection.
        env_clone_origins = self._compute_env_origins_grid(self.cfg.num_envs, self.cfg.env_spacing)
        self.background_mesh.env_clone_origins = env_clone_origins.cpu().numpy().copy()
        env_origin_query_positions = wp.from_torch(self.env_origins.contiguous(), dtype=wp.vec3)
        self.background_mesh.sample_surface_heights(env_origin_query_positions)
        self.background_mesh.env_origins = self.env_origins.detach().cpu().numpy().copy()
        total_patch_half_size = np.asarray(self.cfg.moving_patch_terrain.total_patch_size) / 2
        env_origin_xy = self.background_mesh.env_origins[:, :2]
        if np.any(env_origin_xy - total_patch_half_size < self.background_mesh.bounds[0, :2]) or np.any(
            env_origin_xy + total_patch_half_size > self.background_mesh.bounds[1, :2]
        ):
            raise ValueError(
                "Shared terrain must contain every initial patch; increase terrain size or reduce env_spacing"
            )


@configclass
class BackgroundTerrainImporterCfg(TexturedTerrainImporterCfg):
    """Standard generated-terrain settings plus the depth of its shared support surface."""

    class_type: type = BackgroundTerrainImporter
    use_terrain_origins: bool = False

    moving_patch_terrain: MovingPatchTerrainCfg = MISSING
    """Particle sampling, material, simulated terrain and boundary settings."""

    def validate_geometry(self) -> None:
        """Validate the patch against the generator's configured terrain extent."""
        self.moving_patch_terrain.validate_geometry()
        if self.terrain_type != "generator" or self.terrain_generator is None:
            raise ValueError("Moving-patch background terrain requires terrain_type='generator'")
        terrain_generator_cfg = self.terrain_generator
        if len(terrain_generator_cfg.size) != 2 or not all(
            math.isfinite(dimension) and dimension > 0 for dimension in terrain_generator_cfg.size
        ):
            raise ValueError("terrain_generator.size must contain two positive finite values")
        if terrain_generator_cfg.num_rows < 1 or terrain_generator_cfg.num_cols < 1:
            raise ValueError("Terrain generator row and column counts must be positive")
        background_terrain_size = (
            terrain_generator_cfg.size[0] * terrain_generator_cfg.num_rows,
            terrain_generator_cfg.size[1] * terrain_generator_cfg.num_cols,
        )
        if any(
            background_extent < patch_extent
            for background_extent, patch_extent in zip(
                background_terrain_size, self.moving_patch_terrain.total_patch_size, strict=True
            )
        ):
            raise ValueError("Generated background must contain simulated plus boundary terrain")
