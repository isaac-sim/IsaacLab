# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import omni.physics.tensors.impl.api as physx
import warp as wp
from isaacsim.core.prims import XFormPrim
from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply, quat_apply_yaw
from isaaclab.utils.mesh import PRIMITIVE_MESH_TYPES, create_mesh_from_geom_shape, create_trimesh_from_geom_mesh
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_dynamic_meshes

from ..utils import compute_world_poses
from .ray_caster import RayCaster

if TYPE_CHECKING:
    from .multi_mesh_ray_caster_cfg import MultiMeshRayCasterCfg


class MultiMeshRayCaster(RayCaster):
    """A ray-casting sensor.

    The ray-caster uses a set of rays to detect collisions with meshes in the scene. The rays are
    defined in the sensor's local coordinate frame. The sensor can be configured to ray-cast against
    a set of meshes with a given ray pattern.

    The meshes are parsed from the list of primitive paths provided in the configuration. These are then
    converted to warp meshes and stored in the :attr:`warp_meshes` list. The ray-caster then ray-casts against
    these warp meshes using the ray pattern provided in the configuration.
    """

    cfg: MultiMeshRayCasterCfg
    """The configuration parameters."""

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
                self._raycast_targets_cfg.append(cfg.RaycastTargetCfg(target_prim_expr=target, is_global=True))
            else:
                self._raycast_targets_cfg.append(target)

        # store the views of the meshes available for raycasting to allow movement tracking
        self.mesh_views: dict[str, physx.RigidBodyView | physx.ArticulationView | XFormPrim] = {}
        # store the warp meshes available for raycasting
        self.meshes: dict[str, list[list[wp.Mesh]]] = {}

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
    Implementation.
    """

    def _initialize_warp_meshes(self):
        for target_cfg in self._raycast_targets_cfg:
            # target prim path to ray cast against
            mesh_prim_path = target_cfg.target_prim_expr
            # check if mesh already casted into warp mesh and get the number of meshes per env
            if mesh_prim_path in self.meshes:
                self._num_meshes_per_env[mesh_prim_path] = len(self.meshes[mesh_prim_path]) // self._num_envs
                continue
            paths = sim_utils.find_matching_prim_paths(mesh_prim_path)
            if len(paths) == 0:
                raise RuntimeError(f"Failed to find a prim at path expression: {mesh_prim_path}")

            loaded_vertices: list[np.ndarray | None] = []
            wp_meshes = []
            for path in paths:
                # check if the prim is a primitive object - handle these as special types
                mesh_prim = sim_utils.get_first_matching_child_prim(
                    path, lambda prim: prim.GetTypeName() in PRIMITIVE_MESH_TYPES
                )

                # if we did not find a primitive mesh, we need to read the mesh
                if mesh_prim is None:

                    # obtain the mesh prim
                    mesh_prim = sim_utils.get_first_matching_child_prim(path, lambda prim: prim.GetTypeName() == "Mesh")

                    # check if valid
                    if mesh_prim is None or not mesh_prim.IsValid():
                        raise RuntimeError(f"Invalid mesh prim path: {paths}")

                    points, faces = create_trimesh_from_geom_mesh(mesh_prim)
                    points *= np.array(sim_utils.resolve_world_scale(mesh_prim))
                    registered_idx = _registered_points_idx(points, loaded_vertices)
                    if registered_idx != -1:
                        omni.log.info("Found a duplicate mesh, only reference the mesh.")
                        # Found a duplicate mesh, only reference the mesh.
                        loaded_vertices.append(None)
                        wp_mesh = wp_meshes[registered_idx]
                    else:
                        loaded_vertices.append(points)
                        wp_mesh = convert_to_warp_mesh(points, faces, device=self.device)
                    # print info
                    omni.log.info(
                        f"Read mesh prim: {mesh_prim.GetPath()} with {len(points)} vertices and {len(faces)} faces."
                    )
                else:
                    # create mesh from primitive shape
                    mesh = create_mesh_from_geom_shape(mesh_prim)
                    mesh.vertices *= np.array(sim_utils.resolve_world_scale(mesh_prim))

                    registered_idx = _registered_points_idx(mesh.vertices, loaded_vertices)
                    if registered_idx != -1:
                        # Found a duplicate mesh, only reference the mesh.
                        loaded_vertices.append(None)
                        wp_mesh = wp_meshes[registered_idx]
                    else:
                        loaded_vertices.append(mesh.vertices)
                        wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
                    # print info
                    omni.log.info(f"Created {mesh_prim.GetTypeName()} mesh prim: {mesh_prim.GetPath()}.")
                wp_meshes.append(wp_mesh)

            if target_cfg.is_global:
                # reference the mesh for each environment to ray cast against
                self.meshes[mesh_prim_path] = [wp_meshes] * self._num_envs
                self._num_meshes_per_env[mesh_prim_path] = len(wp_meshes)
            else:
                # split up the meshes for each environment. Little bit ugly, since
                # the current order is interleaved (env1_obj1, env1_obj2, env2_obj1, env2_obj2, ...)
                self.meshes[mesh_prim_path] = []
                mesh_idx = 0
                n_meshes_per_env = len(wp_meshes) // self._num_envs
                self._num_meshes_per_env[mesh_prim_path] = n_meshes_per_env
                for _ in range(self._num_envs):
                    self.meshes[mesh_prim_path].append(wp_meshes[mesh_idx : mesh_idx + n_meshes_per_env])
                    mesh_idx += n_meshes_per_env

            if self.cfg.track_mesh_transforms:
                # create view based on the type of prim
                mesh_prim_api = sim_utils.find_first_matching_prim(mesh_prim_path)
                if mesh_prim_api.HasAPI(UsdPhysics.ArticulationRootAPI):
                    self.mesh_views[mesh_prim_path] = self._physics_sim_view.create_articulation_view(
                        mesh_prim_path.replace(".*", "*")
                    )
                    omni.log.info(f"Created articulation view for mesh prim at path: {mesh_prim_path}")
                elif mesh_prim_api.HasAPI(UsdPhysics.RigidBodyAPI):
                    self.mesh_views[mesh_prim_path] = self._physics_sim_view.create_rigid_body_view(
                        mesh_prim_path.replace(".*", "*")
                    )
                    omni.log.info(f"Created rigid body view for mesh prim at path: {mesh_prim_path}")
                else:
                    self.mesh_views[mesh_prim_path] = XFormPrim(mesh_prim_path, reset_xform_properties=False)
                    omni.log.warn(
                        f"The prim at path {mesh_prim_path} is not a physics prim, but track_mesh_transforms is"
                        " enabled! Defaulting to XFormPrim. \n The pose of the mesh will most likely not"
                        " be updated correctly when running in headless mode."
                    )

        # throw an error if no meshes are found
        if all([target_cfg.target_prim_expr not in self.meshes for target_cfg in self._raycast_targets_cfg]):
            raise RuntimeError(
                f"No meshes found for ray-casting! Please check the mesh prim paths: {self.cfg.mesh_prim_paths}"
            )
        if self.cfg.track_mesh_transforms:
            total_n_meshes_per_env = sum(self._num_meshes_per_env.values())
            self._mesh_positions_w = torch.zeros(self._num_envs, total_n_meshes_per_env, 3, device=self.device)
            self._mesh_orientations_w = torch.zeros(self._num_envs, total_n_meshes_per_env, 4, device=self.device)

        # flatten the list of meshes that are included in mesh_prim_paths of the specific ray caster
        self._meshes = []
        for env_idx in range(self._num_envs):
            meshes_in_env = []
            for target_cfg in self._raycast_targets_cfg:
                meshes_in_env.extend(self.meshes[target_cfg.target_prim_expr][env_idx])
            self._meshes.append(meshes_in_env)

        if self.cfg.track_mesh_transforms:
            self._mesh_views = [
                self.mesh_views[target_cfg.target_prim_expr] for target_cfg in self._raycast_targets_cfg
            ]

        # save a warp array with mesh ids that is passed to the raycast function
        self._mesh_ids_wp = wp.array2d([[m.id for m in b] for b in self._meshes], dtype=wp.uint64, device=self.device)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # obtain the poses of the sensors
        pos_w, quat_w = compute_world_poses(self._view, env_ids, clone=True)
        # apply drift
        pos_w += self.drift[env_ids]
        # store the poses
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        # ray cast based on the sensor poses
        if self.cfg.attach_yaw_only:
            # only yaw orientation is considered and directions are not rotated
            ray_starts_w = quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        else:
            # full orientation is considered
            ray_starts_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])

        if self.cfg.track_mesh_transforms:
            # Update the mesh positions and rotations
            mesh_idx = 0
            for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
                # update position of the target meshes
                pos_w, ori_w = compute_world_poses(view, None)
                pos_w = pos_w.squeeze(0) if len(pos_w.shape) == 3 else pos_w
                ori_w = ori_w.squeeze(0) if len(ori_w.shape) == 3 else ori_w

                count = view.count
                if not target_cfg.is_global:
                    count = count // self._num_envs
                    pos_w = pos_w.view(self._num_envs, count, 3)
                    ori_w = ori_w.view(self._num_envs, count, 4)

                self._mesh_positions_w[:, mesh_idx : mesh_idx + count] = pos_w
                self._mesh_orientations_w[:, mesh_idx : mesh_idx + count] = ori_w
                mesh_idx += count

        self._data.ray_hits_w[env_ids] = raycast_dynamic_meshes(
            ray_starts_w,
            ray_directions_w,
            mesh_ids_wp=self._mesh_ids_wp,  # list with shape num_envs x num_meshes_per_env
            max_dist=self.cfg.max_distance,
            mesh_positions_w=self._mesh_positions_w[env_ids] if self.cfg.track_mesh_transforms else None,
            mesh_orientations_w=self._mesh_orientations_w[env_ids] if self.cfg.track_mesh_transforms else None,
        )[0]


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
