# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
import trimesh
from collections.abc import Sequence
from scipy.spatial.transform import Rotation
from typing import TYPE_CHECKING

import omni.log
import omni.usd
import warp as wp
from isaacsim.core.prims import XFormPrim
from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.utils.math import quat_apply, quat_apply_yaw
from isaaclab.utils.mesh import PRIMITIVE_MESH_TYPES, create_mesh_from_geom_shape, create_trimesh_from_geom_mesh
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_dynamic_meshes

from ..utils import compute_world_poses
from .multi_mesh_ray_caster_data import MultiMeshRayCasterData
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
        multi_mesh_ids: dict[str, list[list[int]]] = {}

        for target_cfg in self._raycast_targets_cfg:
            # target prim path to ray cast against
            target_prim_path = target_cfg.target_prim_expr
            # check if mesh already casted into warp mesh and get the number of meshes per env
            if target_prim_path in multi_mesh_ids:
                self._num_meshes_per_env[target_prim_path] = len(multi_mesh_ids[target_prim_path]) // self._num_envs
                continue

            # find all matching prim paths to provided expression of the target
            target_prims = sim_utils.find_matching_prims(target_prim_path)
            if len(target_prims) == 0:
                raise RuntimeError(f"Failed to find a prim at path expression: {target_prim_path}")

            loaded_vertices: list[np.ndarray | None] = []
            wp_mesh_ids = []
            for target_prim in target_prims:

                if target_prim in MultiMeshRayCaster.meshes:
                    wp_mesh_ids.append(MultiMeshRayCaster.meshes[target_prim.GetPath()].id)
                    continue

                # check if the prim is a primitive object - handle these as special types
                mesh_prims = sim_utils.get_all_matching_child_prims(
                    target_prim.GetPath(), lambda prim: prim.GetTypeName() in PRIMITIVE_MESH_TYPES + ["Mesh"]
                )
                if len(mesh_prims) == 0:
                    raise RuntimeError(f"No mesh prims found at path: {target_prim.GetPath()}")

                trimesh_meshes = []

                for mesh_prim in mesh_prims:
                    # check if valid
                    if mesh_prim is None or not mesh_prim.IsValid():
                        raise RuntimeError(f"Invalid mesh prim path: {target_prim}")

                    if mesh_prim.GetTypeName() == "Mesh":
                        points, faces = create_trimesh_from_geom_mesh(mesh_prim)
                        mesh = trimesh.Trimesh(points, faces)
                    else:
                        mesh = create_mesh_from_geom_shape(mesh_prim)

                    # account for local offsets and world scale of the prim
                    transform = np.asarray(omni.usd.get_local_transform_matrix(mesh_prim)).T
                    world_scale = sim_utils.resolve_world_scale(mesh_prim)
                    # remove local scale from transform and apply world scale
                    rotation = transform[:3, :3]
                    for i in range(3):
                        rotation[:, i] /= np.linalg.norm(rotation[:, i])
                    # apply world scale
                    transform_scaled = transform.copy()
                    transform_scaled[:3, :3] = rotation * world_scale
                    # apply transformation to the mesh (includes affine transform)
                    mesh.apply_transform(transform_scaled)

                    # add to list of parsed meshes
                    trimesh_meshes.append(mesh)

                # resolve instancer prims
                trimesh_meshes.extend(resolve_instancer_meshes(target_prim.GetPath()))

                if len(trimesh_meshes) == 1:
                    trimesh_mesh = trimesh_meshes[0]
                elif self.cfg.merge_prim_meshes:
                    # combine all trimesh meshes into a single mesh
                    trimesh_mesh = trimesh.util.concatenate(trimesh_meshes)
                else:
                    raise RuntimeError(
                        f"Multiple mesh prims found at path: {target_prim.GetPath()} but merging is disabled. Please"
                        " enable `merge_prim_meshes` in the configuration or specify each mesh separately."
                    )

                # check if the mesh is already registered, if so only reference the mesh
                registered_idx = _registered_points_idx(trimesh_mesh.vertices, loaded_vertices)
                if registered_idx != -1:
                    omni.log.info("Found a duplicate mesh, only reference the mesh.")
                    # Found a duplicate mesh, only reference the mesh.
                    loaded_vertices.append(None)
                    wp_mesh_ids.append(wp_mesh_ids[registered_idx])
                else:
                    loaded_vertices.append(trimesh_mesh.vertices)
                    wp_mesh = convert_to_warp_mesh(trimesh_mesh.vertices, trimesh_mesh.faces, device=self.device)
                    MultiMeshRayCaster.meshes[target_prim.GetPath()] = wp_mesh
                    wp_mesh_ids.append(wp_mesh.id)

                # print info
                if registered_idx != -1:
                    omni.log.info(f"Found duplicate mesh for mesh prims under path '{target_prim.GetPath()}'.")
                else:
                    omni.log.info(
                        f"Read '{len(mesh_prims)}' mesh prims under path '{target_prim.GetPath()}' with"
                        f" {len(trimesh_mesh.vertices)} vertices and {len(trimesh_mesh.faces)} faces."
                    )

            if target_cfg.is_global:
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

            if self.cfg.track_mesh_transforms:
                # create view based on the type of prim
                mesh_prim_api = sim_utils.find_first_matching_prim(target_prim_path)
                if mesh_prim_api.HasAPI(UsdPhysics.ArticulationRootAPI):
                    self.mesh_views[target_prim_path] = self._physics_sim_view.create_articulation_view(
                        target_prim_path.replace(".*", "*")
                    )
                    omni.log.info(f"Created articulation view for mesh prim at path: {target_prim_path}")
                elif mesh_prim_api.HasAPI(UsdPhysics.RigidBodyAPI):
                    self.mesh_views[target_prim_path] = self._physics_sim_view.create_rigid_body_view(
                        target_prim_path.replace(".*", "*")
                    )
                    omni.log.info(f"Created rigid body view for mesh prim at path: {target_prim_path}")
                else:
                    self.mesh_views[target_prim_path] = XFormPrim(target_prim_path, reset_xform_properties=False)
                    omni.log.warn(
                        f"The prim at path {target_prim_path} is not a physics prim, but track_mesh_transforms is"
                        " enabled! Defaulting to XFormPrim. \n The pose of the mesh will most likely not"
                        " be updated correctly when running in headless mode."
                    )
        # throw an error if no meshes are found
        if all([target_cfg.target_prim_expr not in multi_mesh_ids for target_cfg in self._raycast_targets_cfg]):
            raise RuntimeError(
                f"No meshes found for ray-casting! Please check the mesh prim paths: {self.cfg.mesh_prim_paths}"
            )

        if self.cfg.track_mesh_transforms:
            total_n_meshes_per_env = sum(self._num_meshes_per_env.values())
            self._mesh_positions_w = torch.zeros(self._num_envs, total_n_meshes_per_env, 3, device=self.device)
            self._mesh_orientations_w = torch.zeros(self._num_envs, total_n_meshes_per_env, 4, device=self.device)

        # flatten the list of meshes that are included in mesh_prim_paths of the specific ray caster
        multi_mesh_ids_flattened = []
        for env_idx in range(self._num_envs):
            meshes_in_env = []
            for target_cfg in self._raycast_targets_cfg:
                meshes_in_env.extend(multi_mesh_ids[target_cfg.target_prim_expr][env_idx])
            multi_mesh_ids_flattened.append(meshes_in_env)

        if self.cfg.track_mesh_transforms:
            self._mesh_views = [
                self.mesh_views[target_cfg.target_prim_expr] for target_cfg in self._raycast_targets_cfg
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

        self._data.ray_hits_w[env_ids], _, _, _, mesh_ids = raycast_dynamic_meshes(
            ray_starts_w,
            ray_directions_w,
            mesh_ids_wp=self._mesh_ids_wp,  # list with shape num_envs x num_meshes_per_env
            max_dist=self.cfg.max_distance,
            mesh_positions_w=self._mesh_positions_w[env_ids] if self.cfg.track_mesh_transforms else None,
            mesh_orientations_w=self._mesh_orientations_w[env_ids] if self.cfg.track_mesh_transforms else None,
            return_mesh_id=self.cfg.update_mesh_ids,
        )

        if self.cfg.update_mesh_ids:
            self._data.ray_mesh_ids[env_ids] = mesh_ids


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


def extract_instancer_data(instancer_prim):
    instancer_data = {}

    # Iterate over all PointInstancers in the scene
    instancer = UsdGeom.PointInstancer(instancer_prim)
    proto_indices = instancer.GetProtoIndicesAttr().Get()
    positions = np.asarray(instancer.GetPositionsAttr().Get() or [])
    orientations = np.asarray(instancer.GetOrientationsAttr().Get() or [])  # order: x, y, z, w
    scales = np.asarray(instancer.GetScalesAttr().Get() or [])
    prototype_paths = [target.pathString for target in instancer.GetPrototypesRel().GetTargets()]

    # check the status of the prototype prims
    stage = instancer_prim.GetStage()
    prototype_prims_active = [stage.GetPrimAtPath(proto_path).IsActive() for proto_path in prototype_paths]

    # check number of positions, orientations, scales and prototype paths is equal to the number of instances
    assert len(positions) == len(orientations) == len(scales), (
        "Number of positions, orientations, and scales must be equal to the number of instances. Positions:"
        f" {len(positions)}, Orientations: {len(orientations)}, Scales: {len(scales)}"
    )
    assert len(positions) == np.asarray(proto_indices).shape[0], (
        f"Number of positions must be equal to the number of instances. Positions: {len(positions)}, Proto indices:"
        f" {len(proto_indices)}"
    )

    # construct the transformation matrices
    transform_matrix = np.eye(4)[None, :, :].repeat(len(positions), axis=0)
    transform_matrix[:, :3, 3] = positions
    transform_matrix[:, :3, :3] = Rotation.from_quat(orientations).as_matrix() * scales[:, np.newaxis, :]

    # get the transform of the instancer
    instancer_transform = np.asarray(omni.usd.get_world_transform_matrix(instancer_prim)).T

    # apply the instancer transform
    transform_matrix = instancer_transform @ transform_matrix

    # Group instances by prototype index
    for i, proto_index in enumerate(proto_indices):
        proto_path = prototype_paths[proto_index]

        # check if proto_prim is active
        if not prototype_prims_active[proto_index]:
            continue

        if proto_path not in instancer_data:
            instancer_data[proto_path] = []

        instancer_data[proto_path].append(transform_matrix[i])

    return instancer_data


def resolve_instancer_meshes(path):
    trimesh_meshes = []
    # Resolve instancer prims
    instancer_prims = sim_utils.get_all_matching_child_prims(path, lambda prim: prim.GetTypeName() == "PointInstancer")
    for instancer_prim in instancer_prims:
        instancer_data = extract_instancer_data(instancer_prim)
        proto_meshes = {}

        for proto_path, transforms in instancer_data.items():
            # Cache meshes for each prototype path
            if proto_path not in proto_meshes:
                proto_meshes[proto_path] = []
                meshes = sim_utils.get_all_matching_child_prims(proto_path, lambda prim: prim.GetTypeName() == "Mesh")

                for mesh_prim in meshes:
                    points, faces = create_trimesh_from_geom_mesh(mesh_prim)
                    proto_meshes[proto_path].append(trimesh.Trimesh(points, faces))

            # Apply transformations efficiently
            for mesh in proto_meshes[proto_path]:
                for transform in transforms:
                    instanced_mesh = mesh.copy()
                    instanced_mesh.apply_transform(transform)
                    trimesh_meshes.append(instanced_mesh)

    return trimesh_meshes
