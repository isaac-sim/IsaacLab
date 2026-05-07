# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import re
from abc import abstractmethod
from typing import TYPE_CHECKING

import numpy as np
import torch
import trimesh
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.utils.math import matrix_from_quat
from isaaclab.utils.mesh import PRIMITIVE_MESH_TYPES, create_trimesh_from_geom_mesh, create_trimesh_from_geom_shape
from isaaclab.utils.warp import convert_to_warp_mesh
from isaaclab.utils.warp import kernels as warp_kernels

from .base_ray_caster import BaseRayCaster
from .kernels import fill_float2d_masked_kernel, fill_vec3_inf_kernel, update_ray_caster_kernel
from .multi_mesh_ray_caster_data import MultiMeshRayCasterData

if TYPE_CHECKING:
    from .multi_mesh_ray_caster_cfg import MultiMeshRayCasterCfg

logger = logging.getLogger(__name__)


class BaseMultiMeshRayCaster(BaseRayCaster):
    """Backend-agnostic multi-mesh ray-casting sensor.

    Extends :class:`BaseRayCaster` with multiple target meshes (primitives or
    arbitrary meshes), per-target dynamic-pose tracking, and shared mesh-data
    caching across envs. Backends supply two per-step hooks —
    :meth:`~BaseRayCaster._get_sensor_transforms_wp` for the sensor body and
    :meth:`_update_target_mesh_transforms` for tracked targets — and do
    init/cleanup in standard sensor-lifecycle overrides. No
    :class:`~isaaclab.sim.views.FrameView` / Fabric path on either axis: PhysX
    reads from its simulation view, Newton from sites registered with
    :class:`~isaaclab_newton.physics.NewtonManager`.

    See :class:`~isaaclab.sensors.ray_caster.MultiMeshRayCasterCfg` for example
    usage.

    .. warning::
        Known race in :func:`raycast_dynamic_meshes_kernel`'s ``atomic_min`` +
        equality-check tie-break: hit position is always correct, but normals /
        face IDs / mesh IDs may pick up the wrong mesh on an exact-distance tie.
        Rare in practice; see `warp#1058 <https://github.com/NVIDIA/warp/issues/1058>`_.
    """

    cfg: MultiMeshRayCasterCfg
    """The configuration parameters."""

    def __init__(self, cfg: MultiMeshRayCasterCfg):
        """Initializes the ray-caster object.

        Args:
            cfg: The configuration parameters.
        """
        super().__init__(cfg)

        self._num_meshes_per_env: dict[str, int] = {}

        self._raycast_targets_cfg: list[MultiMeshRayCasterCfg.RaycastTargetCfg] = []
        for target in self.cfg.mesh_prim_paths:
            if isinstance(target, str):
                self._raycast_targets_cfg.append(cfg.RaycastTargetCfg(prim_expr=target, track_mesh_transforms=False))
            else:
                self._raycast_targets_cfg.append(target)

        for cfg in self._raycast_targets_cfg:
            cfg.prim_expr = cfg.prim_expr.format(ENV_REGEX_NS="/World/envs/env_.*")

        self._data = MultiMeshRayCasterData()

    def __str__(self) -> str:
        return (
            f"Ray-caster @ '{self.cfg.prim_path}': \n"
            f"\tbackend              : {self.__backend_name__}\n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tnumber of meshes     : {self._num_envs} x {sum(self._num_meshes_per_env.values())} \n"
            f"\tnumber of sensors    : {self._num_envs}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\ttotal number of rays : {self.num_rays * self._num_envs}"
        )

    """
    Properties
    """

    @property
    def data(self) -> MultiMeshRayCasterData:
        self._update_outdated_buffers()
        return self._data

    """
    Implementation.
    """

    def _initialize_warp_meshes(self):
        """Resolve target prims, build/reuse warp meshes, and cache per-env mesh IDs.

        Per target expression: resolve prims → collect supported child meshes (merge
        if configured) → dedup identical vertex buffers → partition IDs per env (or
        mark global) → store. Raises ``RuntimeError`` if no prims match, no supported
        mesh prims are found, or multiple mesh prims exist with merging disabled.
        """
        multi_mesh_ids: dict[str, list[list[int]]] = {}
        for target_cfg in self._raycast_targets_cfg:
            target_prim_path = target_cfg.prim_expr
            # check if mesh already casted into warp mesh and skip if so.
            if target_prim_path in multi_mesh_ids:
                logger.warning(
                    f"Mesh at target prim path '{target_prim_path}' already exists in the mesh cache. Duplicate entries"
                    " in `mesh_prim_paths`? This mesh will be skipped."
                )
                continue

            target_prims = sim_utils.find_matching_prims(target_prim_path)
            if len(target_prims) == 0:
                raise RuntimeError(f"Failed to find a prim at path expression: {target_prim_path}")

            is_global_prim = len(target_prims) == 1

            loaded_vertices: list[np.ndarray | None] = []
            wp_mesh_ids = []

            for target_prim in target_prims:
                if target_cfg.is_shared and len(wp_mesh_ids) > 0:
                    # Verify if this mesh has already been registered in an earlier environment.
                    # Note, this check may fail, if the prim path is not following the env_.* pattern
                    # Which (worst case) leads to parsing the mesh and skipping registering it at a later stage
                    curr_prim_base_path = re.sub(r"env_\d+", "env_0", str(target_prim.GetPath()))
                    base_key = (curr_prim_base_path, self._device)
                    if base_key in BaseMultiMeshRayCaster.meshes:
                        BaseMultiMeshRayCaster.meshes[(str(target_prim.GetPath()), self._device)] = (
                            BaseMultiMeshRayCaster.meshes[base_key]
                        )
                prim_key = (str(target_prim.GetPath()), self._device)
                if prim_key in BaseMultiMeshRayCaster.meshes:
                    wp_mesh_ids.append(BaseMultiMeshRayCaster.meshes[prim_key].id)
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
                    if mesh_prim is None or not mesh_prim.IsValid():
                        raise RuntimeError(f"Invalid mesh prim path: {target_prim}")

                    if mesh_prim.GetTypeName() == "Mesh":
                        mesh = create_trimesh_from_geom_mesh(mesh_prim)
                    else:
                        mesh = create_trimesh_from_geom_shape(mesh_prim)
                    scale = sim_utils.resolve_prim_scale(mesh_prim)
                    mesh.apply_scale(scale)

                    relative_pos, relative_quat = sim_utils.resolve_prim_pose(mesh_prim, target_prim)
                    relative_pos = torch.tensor(relative_pos, dtype=torch.float32)
                    relative_quat = torch.tensor(relative_quat, dtype=torch.float32)

                    rotation = matrix_from_quat(relative_quat)
                    transform = np.eye(4)
                    transform[:3, :3] = rotation.numpy()
                    transform[:3, 3] = relative_pos.numpy()
                    mesh.apply_transform(transform)

                    trimesh_meshes.append(mesh)

                if len(trimesh_meshes) == 1:
                    trimesh_mesh = trimesh_meshes[0]
                elif target_cfg.merge_prim_meshes:
                    trimesh_mesh = trimesh.util.concatenate(trimesh_meshes)
                else:
                    raise RuntimeError(
                        f"Multiple mesh prims found at path: {target_prim.GetPath()} but merging is disabled. Please"
                        " enable `merge_prim_meshes` in the configuration or specify each mesh separately."
                    )

                registered_idx = _registered_points_idx(trimesh_mesh.vertices, loaded_vertices)
                if registered_idx != -1 and self.cfg.reference_meshes:
                    logger.info("Found a duplicate mesh, only reference the mesh.")
                    loaded_vertices.append(None)
                    wp_mesh_ids.append(wp_mesh_ids[registered_idx])
                else:
                    loaded_vertices.append(trimesh_mesh.vertices)
                    wp_mesh = convert_to_warp_mesh(trimesh_mesh.vertices, trimesh_mesh.faces, device=self._device)
                    BaseMultiMeshRayCaster.meshes[(str(target_prim.GetPath()), self._device)] = wp_mesh
                    wp_mesh_ids.append(wp_mesh.id)

                if registered_idx != -1:
                    logger.info(f"Found duplicate mesh for mesh prims under path '{target_prim.GetPath()}'.")
                else:
                    logger.info(
                        f"Read '{len(mesh_prims)}' mesh prims under path '{target_prim.GetPath()}' with"
                        f" {len(trimesh_mesh.vertices)} vertices and {len(trimesh_mesh.faces)} faces."
                    )

            if is_global_prim:
                multi_mesh_ids[target_prim_path] = [wp_mesh_ids] * self._num_envs
                self._num_meshes_per_env[target_prim_path] = len(wp_mesh_ids)
            else:
                multi_mesh_ids[target_prim_path] = []
                mesh_idx = 0
                n_meshes_per_env = len(wp_mesh_ids) // self._num_envs
                self._num_meshes_per_env[target_prim_path] = n_meshes_per_env
                for _ in range(self._num_envs):
                    multi_mesh_ids[target_prim_path].append(wp_mesh_ids[mesh_idx : mesh_idx + n_meshes_per_env])
                    mesh_idx += n_meshes_per_env

        if all([target_cfg.prim_expr not in multi_mesh_ids for target_cfg in self._raycast_targets_cfg]):
            raise RuntimeError(
                f"No meshes found for ray-casting! Please check the mesh prim paths: {self.cfg.mesh_prim_paths}"
            )

        total_n_meshes_per_env = sum(self._num_meshes_per_env.values())
        self._mesh_positions_w = wp.zeros((self._num_envs, total_n_meshes_per_env), dtype=wp.vec3, device=self.device)
        self._mesh_orientations_w = wp.zeros(
            (self._num_envs, total_n_meshes_per_env), dtype=wp.quat, device=self.device
        )
        # Zero-copy torch views for writing static (init-time) poses.
        self._mesh_positions_w_torch = wp.to_torch(self._mesh_positions_w)
        self._mesh_orientations_w_torch = wp.to_torch(self._mesh_orientations_w)

        # Per-target slot ranges in the (num_envs, total_per_env) layout.
        self._target_slot_ranges: list[tuple[int, int]] = []
        running = 0
        for target_cfg in self._raycast_targets_cfg:
            n = self._num_meshes_per_env[target_cfg.prim_expr]
            self._target_slot_ranges.append((running, running + n))
            running += n

        for target_cfg, (slot_start, slot_end) in zip(self._raycast_targets_cfg, self._target_slot_ranges):
            n_meshes = slot_end - slot_start
            pos_w, ori_w = [], []
            for prim in sim_utils.find_matching_prims(target_cfg.prim_expr):
                translation, quat = sim_utils.resolve_prim_pose(prim)
                pos_w.append(translation)
                ori_w.append(quat)
            pos_w = torch.tensor(pos_w, device=self.device, dtype=torch.float32).view(-1, n_meshes, 3)
            ori_w = torch.tensor(ori_w, device=self.device, dtype=torch.float32).view(-1, n_meshes, 4)

            self._mesh_positions_w_torch[:, slot_start:slot_end] = pos_w
            self._mesh_orientations_w_torch[:, slot_start:slot_end] = ori_w

        multi_mesh_ids_flattened = []
        for env_idx in range(self._num_envs):
            meshes_in_env = []
            for target_cfg in self._raycast_targets_cfg:
                meshes_in_env.extend(multi_mesh_ids[target_cfg.prim_expr][env_idx])
            multi_mesh_ids_flattened.append(meshes_in_env)

        self._mesh_ids_wp = wp.array2d(multi_mesh_ids_flattened, dtype=wp.uint64, device=self.device)

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        # Full-size closest-hit distance buffer used as the ``atomic_min`` target across meshes —
        # contrast with :class:`BaseRayCaster`'s ``(1, 1)`` ``_dummy_ray_distance``, which is only
        # a placeholder because single-mesh raycast passes ``return_distance=0`` and never writes.
        self._ray_distance_w = wp.zeros((self._num_envs, self.num_rays), dtype=wp.float32, device=self._device)
        if self.cfg.update_mesh_ids:
            self._ray_mesh_id_w = wp.zeros((self._num_envs, self.num_rays), dtype=wp.int16, device=self._device)
            # Zero-copy torch view with the trailing dim expected by consumers of ray_mesh_ids
            self._data.ray_mesh_ids = wp.to_torch(self._ray_mesh_id_w).unsqueeze(-1)
        else:
            # Dummy 1×1 buffer so the kernel launch always has a valid array to bind
            self._ray_mesh_id_w = wp.empty((1, 1), dtype=wp.int16, device=self._device)
        # Persistent dummy buffers for unused kernel outputs; allocated once to avoid per-step allocations.
        self._dummy_normal_w = wp.empty((1, 1), dtype=wp.vec3, device=self._device)
        self._dummy_face_id_w = wp.empty((1, 1), dtype=wp.int32, device=self._device)

    def _update_buffers_impl(self, env_mask: wp.array):
        """Fills the buffers of the sensor data."""
        # sensor_xform * offset → world-frame ray starts/directions + sensor data pose
        wp.launch(
            update_ray_caster_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                self._get_sensor_transforms_wp(),
                env_mask,
                self._offset_pos_wp,
                self._offset_quat_wp,
                self._drift,
                self._ray_cast_drift,
                self._ray_starts_local,
                self._ray_directions_local,
                self._alignment_mode,
            ],
            outputs=[self._data._pos_w, self._data._quat_w, self._ray_starts_w, self._ray_directions_w],
            device=self._device,
        )
        self._update_target_mesh_transforms()

        n_meshes = self._mesh_ids_wp.shape[1]
        wp.launch(
            fill_vec3_inf_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, float("inf"), self._data._ray_hits_w],
            device=self._device,
        )
        wp.launch(
            fill_float2d_masked_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[env_mask, float("inf"), self._ray_distance_w],
            device=self._device,
        )
        # Closest-hit across meshes via atomic_min on ray_distance.
        wp.launch(
            warp_kernels.raycast_dynamic_meshes_kernel,
            dim=(n_meshes, self._num_envs, self.num_rays),
            inputs=[
                env_mask,
                self._mesh_ids_wp,
                self._ray_starts_w,
                self._ray_directions_w,
                self._data._ray_hits_w,
                self._ray_distance_w,
                self._dummy_normal_w,
                self._dummy_face_id_w,
                self._ray_mesh_id_w,
                self._mesh_positions_w,
                self._mesh_orientations_w,
                float(self.cfg.max_distance),
                int(False),
                int(False),
                int(self.cfg.update_mesh_ids),
            ],
            device=self._device,
        )

    """
    Backend-specific hooks.
    """

    @abstractmethod
    def _update_target_mesh_transforms(self) -> None:
        """Per-step write of tracked target poses into :attr:`_mesh_positions_w` /
        :attr:`_mesh_orientations_w`. Static targets keep their init-time USD pose."""
        raise NotImplementedError


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
