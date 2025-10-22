# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import re
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import omni.physics.tensors.impl.api as physx
import warp as wp
from isaacsim.core.prims import XFormPrim
from isaacsim.core.simulation_manager import SimulationManager
from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.math import convert_quat, quat_apply, quat_apply_yaw
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_multi_mesh_kernel

from ..sensor_base import SensorBase
from .ray_caster_data import RayCasterData

if TYPE_CHECKING:
    from .ray_caster_cfg import RayCasterCfg


class RayCaster(SensorBase):
    """A ray-casting sensor optimized for 2D lidar.

    This implementation assumes all environments have identical meshes and uses:
    - Height-based mesh slicing to reduce memory and computation
    - Custom Warp kernel for batched multi-mesh raycasting
    - Full vectorization with zero Python loops
    """

    cfg: RayCasterCfg
    """The configuration parameters."""

    def __init__(self, cfg: RayCasterCfg):
        """Initializes the ray-caster object.

        Args:
            cfg: The configuration parameters.
        """
        sensor_path = cfg.prim_path.split("/")[-1]
        sensor_path_is_regex = re.match(r"^[a-zA-Z0-9/_]+$", sensor_path) is None
        if sensor_path_is_regex:
            raise RuntimeError(
                f"Invalid prim path for the ray-caster sensor: {self.cfg.prim_path}."
                "\n\tHint: Please ensure that the prim path does not contain any regex patterns in the leaf."
            )
        super().__init__(cfg)
        self._data = RayCasterData()
        # Will store sliced meshes shared across all environments
        self.meshes: list[tuple[str, wp.Mesh]] = []
        self.wp_mesh_ids = None
        self.num_meshes = 0
        # Track dynamic meshes
        self.dynamic_mesh_info: list[dict] = []  # Stores {mesh_id, prim_path, env_id}
        self.dynamic_mesh_views: dict = {}  # PhysX views for fast transform queries
        self._dynamic_mesh_update_counter = 0  # Counter for decimation
        # Performance profiling
        self.enable_profiling = False
        self.profile_stats = {"dynamic_mesh_update_times": [], "raycast_times": [], "total_update_times": []}

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"2D Lidar Ray-caster @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tslice height range   : ±{self.slice_height_range}m\n"
            f"\tnumber of meshes     : {len(self.meshes)} (shared across all envs)\n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}"
        )

    @property
    def num_instances(self) -> int:
        return self._view.count

    @property
    def data(self) -> RayCasterData:
        self._update_outdated_buffers()
        return self._data

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
            num_envs_ids = self._view.count
        else:
            num_envs_ids = len(env_ids)

        r = torch.empty(num_envs_ids, 3, device=self.device)
        self.drift[env_ids] = r.uniform_(*self.cfg.drift_range)

        range_list = [self.cfg.ray_cast_drift_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device=self.device)
        self.ray_cast_drift[env_ids] = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (num_envs_ids, 3), device=self.device
        )

    def _initialize_impl(self):
        super()._initialize_impl()
        self._physics_sim_view = SimulationManager.get_physics_sim_view()

        # Ensure/Spawn prim(s)
        import isaacsim.core.utils.prims as prim_utils

        matching_prims = sim_utils.find_matching_prims(self.cfg.prim_path)
        if len(matching_prims) == 0:
            # Create prim(s) for patterns or direct path
            if ".*" in self.cfg.prim_path or "*" in self.cfg.prim_path:
                parent_path = "/".join(self.cfg.prim_path.split("/")[:-1])
                prim_name = self.cfg.prim_path.split("/")[-1]
                parent_prims = sim_utils.find_matching_prims(parent_path)
                for parent_prim in parent_prims:
                    parent_path_str = (
                        str(parent_prim.GetPath()) if hasattr(parent_prim, "GetPath") else str(parent_prim)
                    )
                    full_path = f"{parent_path_str}/{prim_name}"
                    if not prim_utils.is_prim_path_valid(full_path):
                        prim_utils.create_prim(full_path, "Xform", translation=self.cfg.offset.pos)
            else:
                if not prim_utils.is_prim_path_valid(self.cfg.prim_path):
                    prim_utils.create_prim(self.cfg.prim_path, "Xform", translation=self.cfg.offset.pos)

            # Verify
            matching_prims = sim_utils.find_matching_prims(self.cfg.prim_path)
            if len(matching_prims) == 0:
                raise RuntimeError(
                    f"Could not find or create prim with path {self.cfg.prim_path}.\n"
                    "Make sure the parent prim exists (e.g., /World/envs/env_*/Robot/chassis)"
                )

        prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if prim is None:
            raise RuntimeError(f"Failed to find a prim at path expression: {self.cfg.prim_path}")

        # Create appropriate view
        # First check if the sensor prim itself is a physics prim
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            self._view = self._physics_sim_view.create_articulation_view(self.cfg.prim_path.replace(".*", "*"))
            self._parent_body_view = None
        elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
            self._view = self._physics_sim_view.create_rigid_body_view(self.cfg.prim_path.replace(".*", "*"))
            self._parent_body_view = None
        else:
            # Sensor is not a physics prim, so find the parent rigid body
            # Navigate up the hierarchy to find a RigidBody
            parent_prim = prim.GetParent()
            parent_body_path = None

            while parent_prim and parent_prim.GetPath() != prim.GetStage().GetPseudoRoot().GetPath():
                if parent_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    parent_body_path = str(parent_prim.GetPath())
                    break
                parent_prim = parent_prim.GetParent()

            if parent_body_path:
                # Found a parent rigid body - create view for it
                # Replace env_N with env_* pattern
                parent_body_pattern = re.sub(r"env_\d+", "env_*", parent_body_path)
                parent_body_pattern = parent_body_pattern.replace("env_.*", "env_*")

                omni.log.info(f"[RayCaster] Sensor attached to rigid body: {parent_body_pattern}")
                self._parent_body_view = self._physics_sim_view.create_rigid_body_view(parent_body_pattern)
                self._view = XFormPrim(self.cfg.prim_path, reset_xform_properties=False)
            else:
                # No physics parent found - use XFormPrim
                omni.log.warn(
                    f"[RayCaster] Sensor at {prim.GetPath().pathString} is not attached to a physics body! "
                    "Using XFormPrim (position updates may not work correctly)."
                )
                self._view = XFormPrim(self.cfg.prim_path, reset_xform_properties=False)
                self._parent_body_view = None

        # Load and slice meshes
        self._initialize_warp_meshes()
        # Initialize rays
        self._initialize_rays_impl()

    def _initialize_warp_meshes(self):
        """Load meshes for raycasting - optionally slice at lidar height for 2D mode"""
        import isaacsim.core.utils.prims as prim_utils

        # Check if 3D scanning is enabled
        enable_3d = getattr(self.cfg, "enable_3d_scan", False)
        self.slice_height_range = getattr(self.cfg, "slice_height_range", 0.1)

        if enable_3d or self.slice_height_range is None:
            omni.log.info("[RayCaster] 3D scanning mode - loading full meshes (no height slicing)")
            height_min = -float("inf")
            height_max = float("inf")
            self._enable_slicing = False
        else:
            sensor_height = self.cfg.offset.pos[2]
            height_min = sensor_height - self.slice_height_range
            height_max = sensor_height + self.slice_height_range
            self._enable_slicing = True
            omni.log.info(
                f"[RayCaster] 2D scanning mode - slicing meshes at height {sensor_height}m"
                f" (±{self.slice_height_range}m)"
            )
            omni.log.info(f"[RayCaster] Height range: [{height_min}, {height_max}]")

        omni.log.info(f"[RayCaster] Mesh patterns to load: {self.cfg.mesh_prim_paths}")
        omni.log.info(f"[RayCaster] Dynamic mesh patterns: {self.cfg.dynamic_mesh_prim_paths}")
        omni.log.info("[RayCaster] Assuming all environments have identical meshes (relative to env origin)")

        # Track which meshes are dynamic
        dynamic_patterns = set(self.cfg.dynamic_mesh_prim_paths)

        for mesh_prim_path in self.cfg.mesh_prim_paths:
            is_dynamic = mesh_prim_path in dynamic_patterns
            template_path = re.sub(r"env_\.\*", "env_0", mesh_prim_path)
            template_path = re.sub(r"env_\d+", "env_0", template_path)

            matching_prims = prim_utils.find_matching_prim_paths(template_path)

            if len(matching_prims) == 0:
                omni.log.warn(f"No template meshes found for pattern: {template_path}")
                continue

            for prim_path in matching_prims:
                mesh_prim = sim_utils.get_first_matching_child_prim(
                    prim_path, lambda prim: prim.GetTypeName() == "Plane"
                )

                if mesh_prim is None:
                    mesh_prim = sim_utils.get_first_matching_child_prim(
                        prim_path, lambda prim: prim.GetTypeName() == "Mesh"
                    )

                    if mesh_prim is None or not mesh_prim.IsValid():
                        omni.log.warn(f"Invalid mesh prim path: {prim_path}")
                        continue

                    mesh_prim = UsdGeom.Mesh(mesh_prim)

                    points = np.asarray(mesh_prim.GetPointsAttr().Get())

                    # Get mesh world transform
                    xformable = UsdGeom.Xformable(mesh_prim.GetPrim())
                    from pxr import Usd

                    world_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

                    # Extract rotation and translation
                    world_translation = np.array(world_matrix.ExtractTranslation(), dtype=np.float64)
                    world_rotation_matrix = np.array(world_matrix.ExtractRotationMatrix(), dtype=np.float64).reshape(
                        3, 3
                    )

                    # Transform points to world coordinates
                    points_world = points @ world_rotation_matrix.T + world_translation

                    # Get env_0's world origin to convert to env-local coordinates
                    # Use SimulationContext if available for consistent env origin detection
                    if not hasattr(self, "_mesh_load_env0_origin"):
                        import isaacsim.core.utils.stage as stage_utils

                        from isaaclab.sim import SimulationContext

                        sim = SimulationContext.instance()
                        if hasattr(sim, "env_positions") and sim.env_positions is not None:
                            self._mesh_load_env0_origin = sim.env_positions[0].cpu().numpy().astype(np.float64)
                            omni.log.info(
                                f"[RayCaster] Using env_0 origin from SimulationContext: {self._mesh_load_env0_origin}"
                            )
                        else:
                            # Fallback to USD query
                            stage = stage_utils.get_current_stage()
                            env_0_prim = stage.GetPrimAtPath("/World/envs/env_0")
                            if env_0_prim and env_0_prim.IsValid():
                                env_0_xf = UsdGeom.Xformable(env_0_prim)
                                env_0_matrix = env_0_xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                                self._mesh_load_env0_origin = np.array(
                                    env_0_matrix.ExtractTranslation(), dtype=np.float64
                                )
                                omni.log.info(f"[RayCaster] Using env_0 origin from USD: {self._mesh_load_env0_origin}")
                            else:
                                self._mesh_load_env0_origin = np.zeros(3, dtype=np.float64)
                                omni.log.info(f"[RayCaster] Using zero env_0 origin (fallback)")

                    env_0_origin = self._mesh_load_env0_origin

                    # Convert to env_0's local coordinates
                    points = (points_world - env_0_origin).astype(np.float32)

                    # Debug: Log first mesh coordinates
                    if not hasattr(self, "_first_mesh_logged"):
                        self._first_mesh_logged = True
                        omni.log.info(f"[RayCaster] First mesh: {mesh_prim.GetPath()}")
                        omni.log.info(f"  World bounds: {points_world.min(axis=0)} to {points_world.max(axis=0)}")
                        omni.log.info(f"  Env_0 local bounds: {points.min(axis=0)} to {points.max(axis=0)}")

                    # Get face vertex indices and counts
                    indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
                    face_vertex_counts = mesh_prim.GetFaceVertexCountsAttr().Get()

                    # Triangulate if mesh has non-triangular faces
                    if face_vertex_counts is not None:
                        face_vertex_counts = np.asarray(face_vertex_counts)
                        if not np.all(face_vertex_counts == 3):
                            indices = self._triangulate_mesh(indices, face_vertex_counts)

                    sliced_points, sliced_indices = self._slice_mesh_at_height(points, indices, height_min, height_max)

                    if len(sliced_indices) == 0:
                        omni.log.warn(f"No triangles in height range for {prim_path}")
                        continue

                    wp_mesh = convert_to_warp_mesh(sliced_points, sliced_indices, device=self.device)

                    reduction_pct = 100 * (1 - len(sliced_indices) / len(indices))
                    omni.log.info(
                        f"Template mesh {mesh_prim.GetPath()}: "
                        f"{len(points)} vertices, {len(indices)} faces -> "
                        f"{len(sliced_points)} vertices, {len(sliced_indices)} faces "
                        f"({reduction_pct:.1f}% reduction)"
                    )
                else:
                    mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                    wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
                    omni.log.info(f"Created plane: {mesh_prim.GetPath()}")

                # Store mesh with dynamic flag
                self.meshes.append((prim_path, wp_mesh, is_dynamic))

        if len(self.meshes) == 0:
            raise RuntimeError(f"No meshes found for ray-casting! Patterns: {self.cfg.mesh_prim_paths}")

        self._prepare_mesh_array_for_kernel()

        omni.log.info(
            f"Initialized {len(self.meshes)} sliced Warp meshes (shared across all {self._view.count} environments)"
        )

    def _triangulate_mesh(self, indices: np.ndarray, face_vertex_counts: np.ndarray) -> np.ndarray:
        """Convert polygon mesh to triangle mesh using fan triangulation.

        Args:
            indices: Flat array of vertex indices
            face_vertex_counts: Number of vertices per face

        Returns:
            Flat array of triangle indices (each triangle uses 3 consecutive indices)
        """
        triangulated = []
        idx = 0

        for count in face_vertex_counts:
            if count < 3:
                # Skip degenerate faces
                idx += count
                continue
            elif count == 3:
                # Already a triangle
                triangulated.extend(indices[idx : idx + 3])
            else:
                # Triangulate polygon using fan from first vertex
                # For a quad [0,1,2,3], create triangles: [0,1,2], [0,2,3]
                face_indices = indices[idx : idx + count]
                for i in range(1, count - 1):
                    triangulated.extend([face_indices[0], face_indices[i], face_indices[i + 1]])

            idx += count

        return np.array(triangulated, dtype=np.int32)

    def _slice_mesh_at_height(
        self, vertices: np.ndarray, faces: np.ndarray, height_min: float, height_max: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slice mesh to keep only triangles that intersect the height range."""
        num_faces = len(faces) // 3
        faces_reshaped = faces.reshape(num_faces, 3)

        kept_faces = []
        for i in range(num_faces):
            idx0, idx1, idx2 = faces_reshaped[i]
            v0, v1, v2 = vertices[idx0], vertices[idx1], vertices[idx2]
            z_coords = [v0[2], v1[2], v2[2]]
            z_min = min(z_coords)
            z_max = max(z_coords)

            if z_max >= height_min and z_min <= height_max:
                kept_faces.append(faces_reshaped[i])

        if len(kept_faces) == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.int32)

        kept_faces = np.array(kept_faces)
        unique_vertices_indices = np.unique(kept_faces.flatten())

        old_to_new = np.full(len(vertices), -1, dtype=np.int32)
        old_to_new[unique_vertices_indices] = np.arange(len(unique_vertices_indices))

        sliced_vertices = vertices[unique_vertices_indices]
        sliced_faces = old_to_new[kept_faces].flatten()

        return sliced_vertices, sliced_faces

    def _prepare_mesh_array_for_kernel(self):
        """Prepare mesh data structure for custom Warp kernel"""
        mesh_ids = [mesh.id for _, mesh, _ in self.meshes]
        self.wp_mesh_ids = wp.array(mesh_ids, dtype=wp.uint64, device=self.device)
        self.num_meshes = len(self.meshes)

        # Now initialize dynamic mesh tracking with view count available
        self._initialize_dynamic_mesh_tracking()

    def _initialize_dynamic_mesh_tracking(self):
        """Initialize tracking for dynamic meshes after view is created"""
        if not hasattr(self, "_view") or self._view is None:
            omni.log.warn("[RayCaster] Cannot initialize dynamic mesh tracking - view not ready")
            return

        for mesh_idx, (prim_path, wp_mesh, is_dynamic) in enumerate(self.meshes):
            if not is_dynamic:
                continue

            mesh_id = wp_mesh.id

            # For dynamic meshes, track all environment instances
            if "env_0" in prim_path:
                # Generate paths for all environments
                for env_idx in range(self._view.count):
                    env_prim_path = prim_path.replace("env_0", f"env_{env_idx}")
                    self.dynamic_mesh_info.append({
                        "mesh_id": mesh_id,
                        "prim_path": env_prim_path,
                        "env_id": env_idx,
                        "mesh_index": mesh_idx,
                        "wp_mesh": wp_mesh,
                    })
            else:
                # Single static path (no environment pattern)
                self.dynamic_mesh_info.append({
                    "mesh_id": mesh_id,
                    "prim_path": prim_path,
                    "env_id": 0,
                    "mesh_index": mesh_idx,
                    "wp_mesh": wp_mesh,
                })

        if len(self.dynamic_mesh_info) > 0:
            omni.log.info(f"[RayCaster] Initialized tracking for {len(self.dynamic_mesh_info)} dynamic mesh instances")

            # Create PhysX views for fast batch transform queries
            self._create_dynamic_mesh_views()

    def _create_dynamic_mesh_views(self):
        """Create PhysX RigidBodyViews for all dynamic meshes to enable fast batch transform queries."""
        import isaacsim.core.utils.prims as prim_utils

        # Group dynamic meshes by their base pattern (without env index)
        # This allows us to create a single view per mesh type across all environments
        unique_patterns = {}

        for mesh_info in self.dynamic_mesh_info:
            prim_path = mesh_info["prim_path"]
            # Convert env_N to env_* for the pattern
            pattern = re.sub(r"env_\d+", "env_*", prim_path)

            if pattern not in unique_patterns:
                unique_patterns[pattern] = []
            unique_patterns[pattern].append(mesh_info)

        # Create a RigidBodyView for each unique pattern
        for pattern, mesh_infos in unique_patterns.items():
            try:
                # Check if the prim has RigidBodyAPI
                template_path = pattern.replace("env_*", "env_0")
                prim = prim_utils.get_prim_at_path(template_path)

                if prim and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    # Create RigidBodyView for batched queries
                    view = self._physics_sim_view.create_rigid_body_view(pattern.replace(".*", "*"))
                    self.dynamic_mesh_views[pattern] = {"view": view, "mesh_infos": mesh_infos}
                    omni.log.info(f"[RayCaster] Created PhysX view for dynamic mesh pattern: {pattern}")
                else:
                    omni.log.warn(
                        f"[RayCaster] Dynamic mesh {pattern} does not have RigidBodyAPI - will use slow USD queries"
                    )
                    self.dynamic_mesh_views[pattern] = {"view": None, "mesh_infos": mesh_infos}
            except Exception as e:
                omni.log.warn(f"[RayCaster] Failed to create view for {pattern}: {e}")
                self.dynamic_mesh_views[pattern] = {"view": None, "mesh_infos": mesh_infos}

    def _initialize_rays_impl(self):
        """Initialize ray starts and directions"""
        self.ray_starts, self.ray_directions = self.cfg.pattern_cfg.func(self.cfg.pattern_cfg, self._device)
        self.num_rays = len(self.ray_directions)

        offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device)
        offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)
        self.ray_directions = quat_apply(offset_quat.repeat(len(self.ray_directions), 1), self.ray_directions)
        self.ray_starts += offset_pos

        self.ray_starts = self.ray_starts.repeat(self._view.count, 1, 1)
        self.ray_directions = self.ray_directions.repeat(self._view.count, 1, 1)

        self.drift = torch.zeros(self._view.count, 3, device=self.device)
        self.ray_cast_drift = torch.zeros(self._view.count, 3, device=self.device)

        self._data.pos_w = torch.zeros(self._view.count, 3, device=self._device)
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self._device)
        self._data.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self._device)
        self._data.ranges = torch.zeros(self._view.count, self.num_rays, device=self._device)

    def _raycast_multi_mesh_batched(
        self, ray_starts: torch.Tensor, ray_directions: torch.Tensor, max_dist: float
    ) -> torch.Tensor:
        """Raycast against multiple meshes simultaneously using custom Warp kernel."""
        batch_size = ray_starts.shape[0]
        num_rays = ray_starts.shape[1]

        wp_ray_starts = wp.from_torch(ray_starts.contiguous(), dtype=wp.vec3)
        wp_ray_directions = wp.from_torch(ray_directions.contiguous(), dtype=wp.vec3)

        wp_hit_points = wp.zeros((batch_size, num_rays), dtype=wp.vec3, device=self.device)
        wp_hit_distances = wp.full((batch_size, num_rays), 1e10, dtype=wp.float32, device=self.device)

        wp.launch(
            kernel=raycast_multi_mesh_kernel,
            dim=(batch_size, num_rays),
            inputs=[wp_ray_starts, wp_ray_directions, self.wp_mesh_ids, self.num_meshes, max_dist],
            outputs=[wp_hit_points, wp_hit_distances],
            device=self.device,
        )

        hit_points = wp.to_torch(wp_hit_points)
        hit_distances = wp.to_torch(wp_hit_distances)

        # Set no-hit rays to inf (rays that still have distance 1e10)
        no_hit_mask = hit_distances >= 1e10
        hit_points[no_hit_mask] = 10e10

        return hit_points

    def set_env_origins(self, env_origins):
        """Set environment origins for mesh slicing."""
        self.env_origins = env_origins.to(self._device)

    def get_profile_stats(self, reset: bool = False) -> dict:
        """Get profiling statistics for performance analysis.

        Args:
            reset: If True, reset statistics after returning them.

        Returns:
            Dictionary with timing statistics (mean, std, min, max) in milliseconds
        """
        if not self.enable_profiling:
            omni.log.warn("[RayCaster] Profiling is not enabled. Set enable_profiling=True first.")
            return {}

        stats = {}
        for key, times in self.profile_stats.items():
            if len(times) > 0:
                times_ms = [t * 1000 for t in times]  # Convert to milliseconds
                stats[key] = {
                    "mean_ms": np.mean(times_ms),
                    "std_ms": np.std(times_ms),
                    "min_ms": np.min(times_ms),
                    "max_ms": np.max(times_ms),
                    "count": len(times_ms),
                }

        if reset:
            self.reset_profile_stats()

        return stats

    def reset_profile_stats(self):
        """Reset profiling statistics."""
        self.profile_stats = {"dynamic_mesh_update_times": [], "raycast_times": [], "total_update_times": []}

    def print_profile_stats(self, reset: bool = True):
        """Print profiling statistics in a readable format.

        Args:
            reset: If True, reset statistics after printing.
        """
        stats = self.get_profile_stats(reset=reset)
        if not stats:
            return

        print("\n" + "=" * 60)
        print("RayCaster Performance Statistics")
        print("=" * 60)
        print(f"Number of dynamic meshes: {len(self.dynamic_mesh_info)}")
        print(f"Total meshes: {len(self.meshes)}")
        print("-" * 60)

        for key, values in stats.items():
            name = key.replace("_", " ").title().replace("Times", "")
            print(f"\n{name}:")
            print(f"  Mean:  {values['mean_ms']:.4f} ms")
            print(f"  Std:   {values['std_ms']:.4f} ms")
            print(f"  Min:   {values['min_ms']:.4f} ms")
            print(f"  Max:   {values['max_ms']:.4f} ms")
            print(f"  Count: {values['count']}")

        # Calculate percentages
        if "dynamic_mesh_update_times" in stats and "total_update_times" in stats:
            dynamic_pct = stats["dynamic_mesh_update_times"]["mean_ms"] / stats["total_update_times"]["mean_ms"] * 100
            raycast_pct = stats["raycast_times"]["mean_ms"] / stats["total_update_times"]["mean_ms"] * 100
            print("\n" + "-" * 60)
            print("Time Breakdown:")
            print(f"  Dynamic Mesh Updates: {dynamic_pct:.1f}%")
            print(f"  Raycasting:          {raycast_pct:.1f}%")
            print(f"  Other:               {100-dynamic_pct-raycast_pct:.1f}%")

        print("=" * 60 + "\n")

    def _update_dynamic_meshes(self, env_ids: Sequence[int]):
        """Update transforms of dynamic meshes before raycasting (OPTIMIZED with PhysX views).

        For each dynamic mesh instance in the specified environments, get the current
        world transform and update the Warp mesh accordingly.

        Args:
            env_ids: Environment IDs to update dynamic meshes for
        """
        if len(self.dynamic_mesh_info) == 0:
            return

        # Convert env_ids to set for fast lookup
        env_ids_set = set(env_ids) if not isinstance(env_ids, slice) else None

        # Process each unique mesh pattern
        for pattern, view_data in self.dynamic_mesh_views.items():
            view = view_data["view"]
            mesh_infos = view_data["mesh_infos"]

            if view is not None:
                # FAST PATH: Use PhysX RigidBodyView for batched transform queries
                # Get all transforms at once (shape: [N, 7] where 7 = [pos_xyz, quat_xyzw])
                transforms = view.get_transforms()
                positions = transforms[:, :3]  # [N, 3]
                quats_xyzw = transforms[:, 3:]  # [N, 4] in xyzw format

                # Convert quaternions from xyzw to wxyz and then to rotation matrices
                quats_wxyz = convert_quat(quats_xyzw, to="wxyz")

                # Process each mesh that uses this view
                for i, mesh_info in enumerate(mesh_infos):
                    env_id = mesh_info["env_id"]

                    # Skip if not in requested env_ids
                    if env_ids_set is not None and env_id not in env_ids_set:
                        continue

                    wp_mesh = mesh_info["wp_mesh"]

                    # Cache original points on first access
                    if "original_points" not in mesh_info:
                        mesh_info["original_points"] = wp.to_torch(wp_mesh.points).cpu().numpy()

                    # Get transform for this environment
                    pos_world = positions[i]  # [3]
                    quat = quats_wxyz[i]  # [4] wxyz

                    # Convert to env-local coordinates
                    env_origin = self.env_origins[env_id]
                    pos_local = pos_world - env_origin

                    # Transform original points
                    original_points_torch = torch.from_numpy(mesh_info["original_points"]).to(self.device)

                    # Apply rotation: use quat_apply for vectorized rotation
                    rotated_points = quat_apply(quat.unsqueeze(0), original_points_torch.unsqueeze(0)).squeeze(0)

                    # Apply translation
                    transformed_points = rotated_points + pos_local

                    # Update Warp mesh
                    wp_mesh.points.assign(wp.from_torch(transformed_points))
                    wp_mesh.refit()

            else:
                # SLOW PATH: Fallback to USD queries (when PhysX view not available)
                import isaacsim.core.utils.stage as stage_utils
                from pxr import Usd

                stage = stage_utils.get_current_stage()

                for mesh_info in mesh_infos:
                    env_id = mesh_info["env_id"]

                    # Skip if not in requested env_ids
                    if env_ids_set is not None and env_id not in env_ids_set:
                        continue

                    # Get the USD prim
                    prim_path = mesh_info["prim_path"]
                    prim = stage.GetPrimAtPath(prim_path)

                    if not prim or not prim.IsValid():
                        continue

                    # Get world transform
                    xformable = UsdGeom.Xformable(prim)
                    world_matrix = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())

                    # Extract translation and rotation
                    world_translation = np.array(world_matrix.ExtractTranslation(), dtype=np.float64)
                    world_rotation_matrix = np.array(world_matrix.ExtractRotationMatrix(), dtype=np.float64).reshape(
                        3, 3
                    )

                    # Convert to env-local coordinates
                    env_origin = self.env_origins[env_id].cpu().numpy().astype(np.float64)
                    local_translation = (world_translation - env_origin).astype(np.float32)

                    wp_mesh = mesh_info["wp_mesh"]

                    # Cache original points
                    if "original_points" not in mesh_info:
                        mesh_info["original_points"] = wp.to_torch(wp_mesh.points).cpu().numpy()

                    original_points = mesh_info["original_points"]

                    # Transform points: rotate then translate
                    transformed_points = original_points @ world_rotation_matrix.T + local_translation

                    # Update the Warp mesh points
                    wp_mesh.points.assign(
                        wp.from_torch(torch.from_numpy(transformed_points.astype(np.float32)).to(self.device))
                    )
                    wp_mesh.refit()

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fully vectorized raycasting across all environments"""
        import time

        if self.enable_profiling:
            total_start = time.perf_counter()

        # Update dynamic meshes before raycasting (with optional decimation)
        if self.enable_profiling:
            dynamic_start = time.perf_counter()

        # Check if we should update dynamic meshes this frame
        should_update = (self._dynamic_mesh_update_counter % self.cfg.dynamic_mesh_update_decimation) == 0
        if should_update and len(self.dynamic_mesh_info) > 0:
            self._update_dynamic_meshes(env_ids)
        self._dynamic_mesh_update_counter += 1

        if self.enable_profiling:
            dynamic_end = time.perf_counter()
            self.profile_stats["dynamic_mesh_update_times"].append(dynamic_end - dynamic_start)

        # Get sensor poses based on view type
        # If sensor has a parent rigid body, get pose from parent + offset
        if hasattr(self, "_parent_body_view") and self._parent_body_view is not None:
            # Get parent body pose
            parent_pos, parent_quat = self._parent_body_view.get_transforms()[env_ids].split([3, 4], dim=-1)
            parent_quat = convert_quat(parent_quat, to="wxyz")

            # Apply sensor offset relative to parent body
            from isaaclab.utils.math import combine_frame_transforms

            offset_pos = (
                torch.tensor(list(self.cfg.offset.pos), device=self._device).unsqueeze(0).expand(len(env_ids), -1)
            )
            offset_quat = (
                torch.tensor(list(self.cfg.offset.rot), device=self._device).unsqueeze(0).expand(len(env_ids), -1)
            )

            pos_w, quat_w = combine_frame_transforms(parent_pos, parent_quat, offset_pos, offset_quat)

        elif isinstance(self._view, XFormPrim):
            # XFormPrim - get world pose directly
            if not self._view.is_initialized():
                self._view.initialize()
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")

        # Debug: Log the sensor position to verify it's being updated
        if not hasattr(self, "_pos_debug_logged"):
            self._pos_debug_logged = True
            omni.log.info(f"[RayCaster] Sensor position (world): {pos_w[0].cpu().numpy()}")
            if hasattr(self, "_parent_body_view") and self._parent_body_view is not None:
                omni.log.info(f"[RayCaster] Using parent body view for pose tracking")
            else:
                omni.log.info(f"[RayCaster] View type: {type(self._view)}")

        pos_w = pos_w.clone()
        quat_w = quat_w.clone()
        pos_w -= self.env_origins[env_ids]
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        if self.cfg.attach_yaw_only is not None:
            self.cfg.ray_alignment = "yaw" if self.cfg.attach_yaw_only else "base"

        if self.cfg.ray_alignment == "world":
            pos_w[:, 0:2] += self.ray_cast_drift[env_ids, 0:2]
            ray_starts_w = self.ray_starts[env_ids] + pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        elif self.cfg.ray_alignment == "yaw":
            pos_w[:, 0:2] += quat_apply_yaw(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]
            ray_starts_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        elif self.cfg.ray_alignment == "base":
            pos_w[:, 0:2] += quat_apply(quat_w, self.ray_cast_drift[env_ids])[:, 0:2]
            ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
        else:
            raise RuntimeError(f"Unsupported ray_alignment type: {self.cfg.ray_alignment}.")

        if len(self.meshes) == 0:
            self._data.ray_hits_w[env_ids] = float("inf")
            self._data.ranges[env_ids] = float("inf")
            return

        if self.enable_profiling:
            raycast_start = time.perf_counter()

        closest_hits = self._raycast_multi_mesh_batched(ray_starts_w, ray_directions_w, self.cfg.max_distance)

        if self.enable_profiling:
            raycast_end = time.perf_counter()
            self.profile_stats["raycast_times"].append(raycast_end - raycast_start)

        self._data.ray_hits_w[env_ids] = closest_hits
        self._data.ray_hits_w[env_ids, :, 2] += self.ray_cast_drift[env_ids, 2].unsqueeze(-1)

        # Add the env origins back to the hit points
        self._data.ray_hits_w[env_ids] += self.env_origins[env_ids].unsqueeze(1)

        distances = torch.norm(closest_hits - ray_starts_w, dim=-1)
        self._data.ranges[env_ids] = distances

        if self.enable_profiling:
            total_end = time.perf_counter()
            self.profile_stats["total_update_times"].append(total_end - total_start)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "ray_visualizer"):
                self.ray_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            self.ray_visualizer.set_visibility(True)
        else:
            if hasattr(self, "ray_visualizer"):
                self.ray_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        viz_points = self._data.ray_hits_w.reshape(-1, 3)
        viz_points = viz_points[~torch.any(torch.isinf(viz_points), dim=1)]
        self.ray_visualizer.visualize(viz_points)

    def _invalidate_initialize_callback(self, event):
        super()._invalidate_initialize_callback(event)
        self._view = None
