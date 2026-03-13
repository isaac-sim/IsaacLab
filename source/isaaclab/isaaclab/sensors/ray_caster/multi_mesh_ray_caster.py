# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
import trimesh
import warp as wp

import omni.physics.tensors.impl.api as physx

import isaaclab.sim as sim_utils
from isaaclab.sim.views import XformPrimView
from isaaclab.utils.math import matrix_from_quat, quat_mul
from isaaclab.utils.mesh import PRIMITIVE_MESH_TYPES, create_trimesh_from_geom_mesh, create_trimesh_from_geom_shape
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_dynamic_meshes

from .multi_mesh_ray_caster_data import MultiMeshRayCasterData
from .ray_cast_utils import obtain_world_pose_from_view
from .ray_caster import RayCaster

if TYPE_CHECKING:
    from .multi_mesh_ray_caster_cfg import MultiMeshRayCasterCfg

# import logger
logger = logging.getLogger(__name__)


class MultiMeshRayCaster(RayCaster):
    """A multi-mesh ray-casting sensor.

    The ray-caster uses a set of rays to detect collisions with meshes in the scene. The rays are
    defined in the sensor's local coordinate frame. The sensor can be configured to ray-cast against
    a set of meshes with a given ray pattern.

    The meshes are parsed from the list of primitive paths provided in the configuration. These are then
    converted to warp meshes and stored in the :attr:`meshes` list. The ray-caster then ray-casts against
    these warp meshes using the ray pattern provided in the configuration.

    Compared to the default RayCaster, the MultiMeshRayCaster provides additional functionality and flexibility as
    an extension of the default RayCaster with the following enhancements:

    - Raycasting against multiple target types : Supports primitive shapes (spheres, cubes, etc.) as well as arbitrary
      meshes.
    - Dynamic mesh tracking : Keeps track of specified meshes, enabling raycasting against moving parts
      (e.g., robot links, articulated bodies, or dynamic obstacles).
    - Memory-efficient caching : Avoids redundant memory usage by reusing mesh data across environments.

    Example usage to raycast against the visual meshes of a robot (e.g. ANYmal):

    .. code-block:: python

        ray_caster_cfg = MultiMeshRayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot",
            mesh_prim_paths=[
                "/World/Ground",
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/LF_.*/visuals"),
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/RF_.*/visuals"),
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/LH_.*/visuals"),
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/RH_.*/visuals"),
                MultiMeshRayCasterCfg.RaycastTargetCfg(prim_expr="{ENV_REGEX_NS}/Robot/base/visuals"),
            ],
            ray_alignment="world",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.02, size=(2.5, 2.5), direction=(0, 0, -1)),
        )

    """

    cfg: MultiMeshRayCasterCfg
    """The configuration parameters."""

    mesh_offsets: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    mesh_views: ClassVar[dict[str, XformPrimView | physx.ArticulationView | physx.RigidBodyView]] = {}
    """A dictionary to store mesh views for raycasting, shared across all instances.

    The keys correspond to the prim path for the mesh views, and values are the corresponding view objects.
    """

    def __init__(self, cfg: MultiMeshRayCasterCfg):
        """Initializes the ray-caster object.

        Args:
            cfg: The configuration parameters.
        """
        # Initialize base class
        super().__init__(cfg)

        # Create empty variables for storing output data
        self._num_meshes_per_env: dict[str, int] = {}
        """Keeps track of the number of meshes per env for each ray_cast target.
           Since we allow regex indexing (e.g. env_*/object_*) they can differ
        """

        self._raycast_targets_cfg: list[MultiMeshRayCasterCfg.RaycastTargetCfg] = []
        for target in self.cfg.mesh_prim_paths:
            # Legacy support for string targets. Treat them as global targets.
            if isinstance(target, str):
                self._raycast_targets_cfg.append(cfg.RaycastTargetCfg(prim_expr=target, track_mesh_transforms=False))
            else:
                self._raycast_targets_cfg.append(target)

        # Resolve regex namespace if set
        for cfg in self._raycast_targets_cfg:
            cfg.prim_expr = cfg.prim_expr.format(ENV_REGEX_NS="/World/envs/env_.*")

        # overwrite the data class
        self._data = MultiMeshRayCasterData()

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""

        return (
            f"Ray-caster @ '{self.cfg.prim_path}': \n"
            f"\tview type            : {self._view.__class__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {self._num_envs} x {sum(self._num_meshes_per_env.values())} \n"
            f"\tnumber of sensors    : {self._view.count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._view.count}"
        )

    """
    Properties
    """

    @property
    def data(self) -> MultiMeshRayCasterData:
        # update sensors if needed
        self._update_outdated_buffers()
        # return the data
        return self._data

    """
    Implementation.
    """

    def _initialize_warp_meshes(self):
        """Parse mesh prim expressions, build (or reuse) Warp meshes, and cache per-env mesh IDs.

        High-level steps (per target expression):

        1. Resolve matching prims by regex/path expression.
        2. Collect supported mesh child prims; merge into a single mesh if configured.
        3. Deduplicate identical vertex buffers (exact match) to avoid uploading duplicates to Warp.
        4. Partition mesh IDs per environment or mark as globally shared.
        5. Optionally create physics views (articulation / rigid body / fallback XForm) and cache local offsets.

        Exceptions:
            Raises a RuntimeError if:

            - No prims match the provided expression.
            - No supported mesh prims are found under a matched prim.
            - Multiple mesh prims are found but merging is disabled.

        """
        multi_mesh_ids: dict[str, list[list[int]]] = {}
        for target_cfg in self._raycast_targets_cfg:
            # target prim path to ray cast against
            target_prim_path = target_cfg.prim_expr
            # # check if mesh already casted into warp mesh and skip if so.
            if target_prim_path in multi_mesh_ids:
                logger.warning(
                    f"Mesh at target prim path '{target_prim_path}' already exists in the mesh cache. Duplicate entries"
                    " in `mesh_prim_paths`? This mesh will be skipped."
                )
                continue

            # find all matching prim paths to provided expression of the target
            target_prims = sim_utils.find_matching_prims(target_prim_path)
            if len(target_prims) == 0:
                raise RuntimeError(f"Failed to find a prim at path expression: {target_prim_path}")

            # If only one prim is found, treat it as a global prim.
            # Either it's a single global object (e.g. ground) or we are only using one env.
            is_global_prim = len(target_prims) == 1

            loaded_vertices: list[np.ndarray | None] = []
            wp_mesh_ids = []

            for target_prim in target_prims:
                # Reuse previously parsed shared mesh instance if possible.
                if target_cfg.is_shared and len(wp_mesh_ids) > 0:
                    # Verify if this mesh has already been registered in an earlier environment.
                    # Note, this check may fail, if the prim path is not following the env_.* pattern
                    # Which (worst case) leads to parsing the mesh and skipping registering it at a later stage
                    curr_prim_base_path = re.sub(r"env_\d+", "env_0", str(target_prim.GetPath()))  #
                    if curr_prim_base_path in MultiMeshRayCaster.meshes:
                        MultiMeshRayCaster.meshes[str(target_prim.GetPath())] = MultiMeshRayCaster.meshes[
                            curr_prim_base_path
                        ]
                # Reuse mesh imported by another ray-cast sensor (global cache).
                if str(target_prim.GetPath()) in MultiMeshRayCaster.meshes:
                    wp_mesh_ids.append(MultiMeshRayCaster.meshes[str(target_prim.GetPath())].id)
                    loaded_vertices.append(None)
                    continue

                mesh_prims = sim_utils.get_all_matching_child_prims(
                    target_prim.GetPath(), lambda prim: prim.GetTypeName() in PRIMITIVE_MESH_TYPES + ["Mesh"]
                )
                if len(mesh_prims) == 0:
                    warn_msg = (
                        f"No mesh prims found at path: {target_prim.GetPath()} with supported types:"
                        f" {PRIMITIVE_MESH_TYPES + ['Mesh']}"
                        " Skipping this target."
                    )
                    for prim in sim_utils.get_all_matching_child_prims(target_prim.GetPath(), lambda prim: True):
                        warn_msg += f"\n - Available prim '{prim.GetPath()}' of type '{prim.GetTypeName()}'"
                    logger.warning(warn_msg)
                    continue

                trimesh_meshes = []

                for mesh_prim in mesh_prims:
                    # check if valid
                    if mesh_prim is None or not mesh_prim.IsValid():
                        raise RuntimeError(f"Invalid mesh prim path: {target_prim}")

                    if mesh_prim.GetTypeName() == "Mesh":
                        mesh = create_trimesh_from_geom_mesh(mesh_prim)
                    else:
                        mesh = create_trimesh_from_geom_shape(mesh_prim)

                    relative_pos, relative_quat = sim_utils.resolve_prim_pose(mesh_prim, target_prim)
                    relative_pos = torch.tensor(relative_pos, dtype=torch.float32)
                    relative_quat = torch.tensor(relative_quat, dtype=torch.float32)

                    world_scale = sim_utils.resolve_prim_scale(mesh_prim)

                    rotation = matrix_from_quat(relative_quat)
                    transform = np.eye(4)
                    transform[:3, :3] = rotation.numpy()
                    transform[:3, 3] = relative_pos.numpy()
                    transform[:3, :3] = transform[:3, :3] @ np.diag(world_scale)
                    mesh.apply_transform(transform)

                    # add to list of parsed meshes
                    trimesh_meshes.append(mesh)

                if len(trimesh_meshes) == 1:
                    trimesh_mesh = trimesh_meshes[0]
                elif target_cfg.merge_prim_meshes:
                    # combine all trimesh meshes into a single mesh
                    trimesh_mesh = trimesh.util.concatenate(trimesh_meshes)
                else:
                    raise RuntimeError(
                        f"Multiple mesh prims found at path: {target_prim.GetPath()} but merging is disabled. Please"
                        " enable `merge_prim_meshes` in the configuration or specify each mesh separately."
                    )

                # check if the mesh is already registered, if so only reference the mesh
                registered_idx = _registered_points_idx(trimesh_mesh.vertices, loaded_vertices)
                if registered_idx != -1 and self.cfg.reference_meshes:
                    logger.info("Found a duplicate mesh, only reference the mesh.")
                    # Found a duplicate mesh, only reference the mesh.
                    loaded_vertices.append(None)
                    wp_mesh_ids.append(wp_mesh_ids[registered_idx])
                else:
                    loaded_vertices.append(trimesh_mesh.vertices)
                    wp_mesh = convert_to_warp_mesh(trimesh_mesh.vertices, trimesh_mesh.faces, device=self.device)
                    MultiMeshRayCaster.meshes[str(target_prim.GetPath())] = wp_mesh
                    wp_mesh_ids.append(wp_mesh.id)

                # print info
                if registered_idx != -1:
                    logger.info(f"Found duplicate mesh for mesh prims under path '{target_prim.GetPath()}'.")
                else:
                    logger.info(
                        f"Read '{len(mesh_prims)}' mesh prims under path '{target_prim.GetPath()}' with"
                        f" {len(trimesh_mesh.vertices)} vertices and {len(trimesh_mesh.faces)} faces."
                    )

            if is_global_prim:
                # reference the mesh for each environment to ray cast against
                multi_mesh_ids[target_prim_path] = [wp_mesh_ids] * self._num_envs
                self._num_meshes_per_env[target_prim_path] = len(wp_mesh_ids)
            else:
                # split up the meshes for each environment. Little bit ugly, since
                # the current order is interleaved (env1_obj1, env1_obj2, env2_obj1, env2_obj2, ...)
                multi_mesh_ids[target_prim_path] = []
                mesh_idx = 0
                n_meshes_per_env = len(wp_mesh_ids) // self._num_envs
                self._num_meshes_per_env[target_prim_path] = n_meshes_per_env
                for _ in range(self._num_envs):
                    multi_mesh_ids[target_prim_path].append(wp_mesh_ids[mesh_idx : mesh_idx + n_meshes_per_env])
                    mesh_idx += n_meshes_per_env

            if target_cfg.track_mesh_transforms:
                MultiMeshRayCaster.mesh_views[target_prim_path], MultiMeshRayCaster.mesh_offsets[target_prim_path] = (
                    self._obtain_trackable_prim_view(target_prim_path)
                )

        # throw an error if no meshes are found
        if all([target_cfg.prim_expr not in multi_mesh_ids for target_cfg in self._raycast_targets_cfg]):
            raise RuntimeError(
                f"No meshes found for ray-casting! Please check the mesh prim paths: {self.cfg.mesh_prim_paths}"
            )

        total_n_meshes_per_env = sum(self._num_meshes_per_env.values())
        self._mesh_positions_w = torch.zeros(self._num_envs, total_n_meshes_per_env, 3, device=self.device)
        self._mesh_orientations_w = torch.zeros(self._num_envs, total_n_meshes_per_env, 4, device=self.device)

        # Update the mesh positions and rotations
        mesh_idx = 0
        for target_cfg in self._raycast_targets_cfg:
            n_meshes = self._num_meshes_per_env[target_cfg.prim_expr]

            # update position of the target meshes
            pos_w, ori_w = [], []
            for prim in sim_utils.find_matching_prims(target_cfg.prim_expr):
                translation, quat = sim_utils.resolve_prim_pose(prim)
                pos_w.append(translation)
                ori_w.append(quat)
            pos_w = torch.tensor(pos_w, device=self.device, dtype=torch.float32).view(-1, n_meshes, 3)
            ori_w = torch.tensor(ori_w, device=self.device, dtype=torch.float32).view(-1, n_meshes, 4)

            self._mesh_positions_w[:, mesh_idx : mesh_idx + n_meshes] = pos_w
            self._mesh_orientations_w[:, mesh_idx : mesh_idx + n_meshes] = ori_w
            mesh_idx += n_meshes

        # flatten the list of meshes that are included in mesh_prim_paths of the specific ray caster
        multi_mesh_ids_flattened = []
        for env_idx in range(self._num_envs):
            meshes_in_env = []
            for target_cfg in self._raycast_targets_cfg:
                meshes_in_env.extend(multi_mesh_ids[target_cfg.prim_expr][env_idx])
            multi_mesh_ids_flattened.append(meshes_in_env)

        self._mesh_views = [
            self.mesh_views[target_cfg.prim_expr] if target_cfg.track_mesh_transforms else None
            for target_cfg in self._raycast_targets_cfg
        ]

        # save a warp array with mesh ids that is passed to the raycast function
        self._mesh_ids_wp = wp.array2d(multi_mesh_ids_flattened, dtype=wp.uint64, device=self.device)

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        if self.cfg.update_mesh_ids:
            self._data.ray_mesh_ids = torch.zeros(
                self._num_envs, self.num_rays, 1, device=self.device, dtype=torch.int16
            )

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data.

        Args:
            env_ids: The environment ids to update.
        """

        self._update_ray_infos(env_ids)

        # Update the mesh positions and rotations
        mesh_idx = 0
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if not target_cfg.track_mesh_transforms:
                mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]
                continue

            # update position of the target meshes
            pos_w, ori_w = obtain_world_pose_from_view(view, None)
            pos_w = pos_w.squeeze(0) if len(pos_w.shape) == 3 else pos_w
            ori_w = ori_w.squeeze(0) if len(ori_w.shape) == 3 else ori_w

            if target_cfg.prim_expr in MultiMeshRayCaster.mesh_offsets:
                pos_offset, ori_offset = MultiMeshRayCaster.mesh_offsets[target_cfg.prim_expr]
                pos_w -= pos_offset
                ori_w = quat_mul(ori_offset.expand(ori_w.shape[0], -1), ori_w)

            count = view.count
            if count != 1:  # Mesh is not global, i.e. we have different meshes for each env
                count = count // self._num_envs
                pos_w = pos_w.view(self._num_envs, count, 3)
                ori_w = ori_w.view(self._num_envs, count, 4)

            self._mesh_positions_w[:, mesh_idx : mesh_idx + count] = pos_w
            self._mesh_orientations_w[:, mesh_idx : mesh_idx + count] = ori_w
            mesh_idx += count

        self._data.ray_hits_w[env_ids], _, _, _, mesh_ids = raycast_dynamic_meshes(
            self._ray_starts_w[env_ids],
            self._ray_directions_w[env_ids],
            mesh_ids_wp=self._mesh_ids_wp,  # list with shape num_envs x num_meshes_per_env
            max_dist=self.cfg.max_distance,
            mesh_positions_w=self._mesh_positions_w[env_ids],
            mesh_orientations_w=self._mesh_orientations_w[env_ids],
            return_mesh_id=self.cfg.update_mesh_ids,
        )

        if self.cfg.update_mesh_ids:
            self._data.ray_mesh_ids[env_ids] = mesh_ids

    def __del__(self):
        super().__del__()
        if RayCaster._instance_count == 0:
            MultiMeshRayCaster.mesh_offsets.clear()
            MultiMeshRayCaster.mesh_views.clear()


"""
Helper functions
"""


def _registered_points_idx(points: np.ndarray, registered_points: list[np.ndarray | None]) -> int:
    """Check if the points are already registered in the list of registered points.

    Args:
        points: The points to check.
        registered_points: The list of registered points.

    Returns:
        The index of the registered points if found, otherwise -1.
    """
    for idx, reg_points in enumerate(registered_points):
        if reg_points is None:
            continue
        if reg_points.shape == points.shape and (reg_points == points).all():
            return idx
    return -1
