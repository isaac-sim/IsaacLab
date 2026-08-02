# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from typing import Any

import newton
import numpy as np
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.cloner import split_clone_template
from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster
from isaaclab.sensors.ray_caster.kernels import ALIGNMENT_BASE, update_ray_caster_kernel
from isaaclab.utils.warp import ProxyArray

from isaaclab_newton.physics import NewtonManager

from .newton_raycast_sensor_cfg import NewtonRaycastSensorCfg
from .newton_raycast_sensor_data import NewtonRaycastSensorData


@wp.kernel(enable_backward=False)
def _newton_site_world_poses_kernel(
    site_indices: wp.array(dtype=wp.int32),
    shape_body: wp.array(dtype=wp.int32),
    shape_transform: wp.array(dtype=wp.transform),
    body_q: wp.array(dtype=wp.transform),
    out_pose: wp.array(dtype=wp.transformf),
    out_pos: wp.array(dtype=wp.vec3f),
    out_quat: wp.array(dtype=wp.quatf),
):
    """Write world poses for Newton sites."""
    index = wp.tid()
    site_index = site_indices[index]
    body_index = shape_body[site_index]
    site_transform = shape_transform[site_index]
    if body_index == -1:
        world_transform = site_transform
    else:
        world_transform = wp.transform_multiply(body_q[body_index], site_transform)
    out_pose[index] = world_transform
    out_pos[index] = wp.transform_get_translation(world_transform)
    out_quat[index] = wp.transform_get_rotation(world_transform)


@wp.kernel(enable_backward=False)
def _gather_pose_by_index_kernel(
    indices: wp.array(dtype=wp.int32),
    pos_src: wp.array(dtype=wp.vec3f),
    quat_src: wp.array(dtype=wp.quatf),
    pos_dst: wp.array(dtype=wp.vec3f),
    quat_dst: wp.array(dtype=wp.quatf),
):
    """Gather poses using an index array."""
    index = wp.tid()
    source_index = indices[index]
    pos_dst[index] = pos_src[source_index]
    quat_dst[index] = quat_src[source_index]


def _newton_body_pattern(body_path: str) -> str:
    """Convert a concrete environment index to a prototype body pattern."""
    body_path = body_path.replace("{}", ".*")
    return re.sub(r"^(/World/envs/)env_\d+/", r"\1env_.*/", body_path)


def _identity_offsets(count: int, device: str) -> tuple[wp.array, wp.array]:
    """Create identity offsets for site poses that already include frame placement."""
    offset_pos = wp.zeros(count, dtype=wp.vec3f, device=device)
    identity_quat = np.zeros((count, 4), dtype=np.float32)
    identity_quat[:, 3] = 1.0
    return offset_pos, wp.array(identity_quat, dtype=wp.quatf, device=device)


class _NewtonRayCasterPoseMixin:
    """Register Newton sensor sites and update ray poses from the Newton manager state."""

    @property
    def count(self: Any) -> int:
        """Number of resolved Newton sites tracked as sensor frames."""
        return self._view_count

    def __init__(self: Any, cfg):
        """Register the sensor site before Newton model finalization."""
        super().__init__(cfg)  # pyright: ignore[reportCallIssue]
        self._sensor_site_labels = self._register_sites_for_expr(self.cfg.prim_path)

    def _register_sites_for_expr(self, prim_expr: str) -> list[str]:
        """Register Newton sites for a prim expression."""
        plan = sim_utils.SimulationContext.instance().get_clone_plan()
        if plan is not None:
            for destination_template in plan.destinations:
                destination_prefix, _ = split_clone_template(destination_template)
                if prim_expr.startswith(destination_prefix) and "/" not in prim_expr[len(destination_prefix) :]:
                    return [NewtonManager.cl_register_site(None, wp.transform(), per_world=True)]

        try:
            body_expr, fixed_pos, fixed_quat = self._resolve_rigid_body_ancestor_expr()
        except RuntimeError:
            # Preserve support for sensor paths registered before their USD prim
            # exists. Known camera/raycaster child names attach to their parent.
            body_expr = prim_expr
            if prim_expr.rsplit("/", 1)[-1].lower() in ("camera", "raycaster"):
                body_expr = prim_expr.rsplit("/", 1)[0]
            fixed_pos = None
            fixed_quat = None

        pos = fixed_pos or (0.0, 0.0, 0.0)
        quat = fixed_quat or (0.0, 0.0, 0.0, 1.0)
        site_transform = wp.transform(wp.vec3(*pos), wp.quat(*quat))

        return [NewtonManager.cl_register_site(_newton_body_pattern(body_expr), site_transform)]

    def _initialize_pose_tracking(self: Any) -> None:
        """Resolve registered site labels and allocate pose buffers."""
        site_indices = self._resolve_site_indices(self._sensor_site_labels, self.cfg.prim_path, self._num_envs)
        self._view = self
        self._view_count = len(site_indices)
        self._sensor_site_indices = wp.array(site_indices, dtype=wp.int32, device=self._device)
        self._newton_pose_w = wp.empty(self._view_count, dtype=wp.transformf, device=self._device)
        self._newton_pos_w = ProxyArray(wp.empty(self._view_count, dtype=wp.vec3f, device=self._device))
        self._newton_quat_w = ProxyArray(wp.empty(self._view_count, dtype=wp.quatf, device=self._device))
        self._offset_pos_wp, self._offset_quat_wp = _identity_offsets(self._view_count, self._device)

    def _update_ray_infos(self: Any, env_mask: wp.array) -> None:
        """Update site poses and transform local rays into world space."""
        self._update_newton_site_transforms(
            self._sensor_site_indices, self._newton_pose_w, self._newton_pos_w.warp, self._newton_quat_w.warp
        )
        pos_w = self._data.pos_w.warp
        quat_w = self._data.quat_w_world.warp if hasattr(self._data, "quat_w_world") else self._data.quat_w.warp
        ray_starts = self.ray_starts.warp if hasattr(self.ray_starts, "warp") else self._ray_starts_local
        ray_directions = (
            self.ray_directions.warp if hasattr(self.ray_directions, "warp") else self._ray_directions_local
        )
        alignment_mode = int(ALIGNMENT_BASE) if hasattr(self._data, "quat_w_world") else self._alignment_mode
        wp.launch(
            update_ray_caster_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                self._newton_pose_w,
                env_mask,
                self._offset_pos_wp,
                self._offset_quat_wp,
                self.drift.warp,
                self.ray_cast_drift.warp,
                ray_starts,
                ray_directions,
                alignment_mode,
            ],
            outputs=[pos_w, quat_w, self._ray_starts_w, self._ray_directions_w],
            device=self._device,
        )

    def get_world_poses(self: Any, indices=None) -> tuple[ProxyArray, ProxyArray]:
        """Return world poses for legacy camera helpers."""
        self._update_newton_site_transforms(
            self._sensor_site_indices, self._newton_pose_w, self._newton_pos_w.warp, self._newton_quat_w.warp
        )
        if indices is None:
            return self._newton_pos_w, self._newton_quat_w
        if not isinstance(indices, wp.array):
            indices = wp.array(indices, dtype=wp.int32, device=self._device)
        pos_w = wp.empty(indices.shape[0], dtype=wp.vec3f, device=self._device)
        quat_w = wp.empty(indices.shape[0], dtype=wp.quatf, device=self._device)
        wp.launch(
            _gather_pose_by_index_kernel,
            dim=indices.shape[0],
            inputs=[indices, self._newton_pos_w.warp, self._newton_quat_w.warp],
            outputs=[pos_w, quat_w],
            device=self._device,
        )
        return ProxyArray(pos_w), ProxyArray(quat_w)

    def _update_newton_site_transforms(
        self: Any,
        site_indices: wp.array,
        pose_buf: wp.array,
        pos_buf: wp.array,
        quat_buf: wp.array,
    ) -> None:
        """Update site transforms using the manager-bound model and state."""
        model = NewtonManager.get_model()
        state = NewtonManager.get_state_0()
        wp.launch(
            _newton_site_world_poses_kernel,
            dim=site_indices.shape[0],
            inputs=[site_indices, model.shape_body, model.shape_transform, state.body_q],
            outputs=[pose_buf, pos_buf, quat_buf],
            device=self._device,
        )

    @staticmethod
    def _resolve_site_indices(labels: list[str], prim_expr: str, num_envs: int) -> list[int]:
        """Expand registered site labels into per-environment Newton site indices."""
        site_map = NewtonManager._cl_site_index_map
        site_indices: list[int] = []
        for env_index in range(num_envs):
            for label in labels:
                error_prefix = f"RayCaster target '{prim_expr}' site label '{label}'"
                if label not in site_map:
                    raise ValueError(f"{error_prefix} was not found in NewtonManager._cl_site_index_map.")
                global_index, per_world = site_map[label]
                env_site_indices = [global_index] if per_world is None else per_world[env_index]
                site_indices.extend(env_site_indices)
        return site_indices


@wp.kernel(enable_backward=False)
def _resolve_bvh_hits_kernel(
    # input
    env_mask: wp.array(dtype=wp.bool),
    ray_starts_w: wp.array2d(dtype=wp.vec3f),
    ray_directions_w: wp.array2d(dtype=wp.vec3f),
    hit_dist: wp.array(dtype=wp.float32),
    hit_normal: wp.array(dtype=wp.vec3f),
    max_distance: float,
    ray_cast_drift: wp.array(dtype=wp.vec3f),
    # output
    ray_hits_w: wp.array2d(dtype=wp.vec3f),
    ray_distances: wp.array2d(dtype=wp.float32),
    ray_normals_w: wp.array2d(dtype=wp.vec3f),
):
    """Turn flat BVH query results into per-env hit points, distances, and normals.

    Misses (``hit_dist < 0``) and hits beyond ``max_distance`` are written as ``inf``.
    Launch with dim=(num_envs, num_rays).
    """
    env, ray = wp.tid()
    if not env_mask[env]:
        return
    idx = env * ray_starts_w.shape[1] + ray
    t = hit_dist[idx]
    if t >= 0.0 and t <= max_distance:
        hit = ray_starts_w[env, ray] + t * ray_directions_w[env, ray]
        ray_hits_w[env, ray] = wp.vec3f(hit[0], hit[1], hit[2] + ray_cast_drift[env][2])
        ray_distances[env, ray] = t
        ray_normals_w[env, ray] = hit_normal[idx]
    else:
        inf_vec = wp.vec3f(wp.inf, wp.inf, wp.inf)
        ray_hits_w[env, ray] = inf_vec
        ray_distances[env, ray] = wp.inf
        ray_normals_w[env, ray] = inf_vec


class NewtonRaycastSensor(_NewtonRayCasterPoseMixin, BaseRayCaster):
    """Ray-cast sensor that queries the whole Newton scene through the model's shape BVH.

    Rays are cast with :func:`newton.intersect_ray` against every collision
    shape in the sensor's own world plus the global world (e.g. terrain), so
    dynamic bodies are hit without configuring target meshes. The full update
    (sensor pose, ray transform, BVH query, hit resolve) is registered as a
    task with :class:`~isaaclab_newton.physics.NewtonManager`, which owns
    the shared BVH refit and sensor execution graph.
    """

    cfg: NewtonRaycastSensorCfg
    """The configuration parameters."""

    def __init__(self, cfg: NewtonRaycastSensorCfg):
        if cfg.max_distance <= 0.0:
            raise ValueError(f"max_distance must be positive, received {cfg.max_distance}.")
        super().__init__(cfg)
        self._data = NewtonRaycastSensorData()
        self._sensor_task_name: str | None = None

    @property
    def data(self) -> NewtonRaycastSensorData:
        self._update_outdated_buffers()
        return self._data

    @property
    def ray_starts_w(self) -> ProxyArray:
        """World-frame ray start positions as of the last update [m].

        Shape is (N, B), dtype ``wp.vec3f``. In torch this resolves to (N, B, 3).
        """
        return self._ray_starts_w_ta

    @property
    def ray_directions_w(self) -> ProxyArray:
        """World-frame ray directions (unit vectors) as of the last update.

        Shape is (N, B), dtype ``wp.vec3f``. In torch this resolves to (N, B, 3).
        """
        return self._ray_directions_w_ta

    def _initialize_warp_meshes(self) -> None:
        # Rays are cast against the scene BVH; no warp meshes are needed.
        return

    def _initialize_impl(self) -> None:
        super()._initialize_impl()
        if self._view_count != self._num_envs:
            raise RuntimeError(
                f"NewtonRaycastSensor '{self.cfg.prim_path}' resolved {self._view_count} Newton sites"
                f" for {self._num_envs} environments; exactly one site per environment is supported."
                " Attach the sensor to a single rigid body per environment."
            )
        ray_count = self._num_envs * self.num_rays
        self._ray_starts_w_ta = ProxyArray(self._ray_starts_w)
        self._ray_directions_w_ta = ProxyArray(self._ray_directions_w)
        # Flat views and scratch buffers for newton.intersect_ray.
        self._ray_starts_w_flat = self._ray_starts_w.reshape((ray_count,))
        self._ray_directions_w_flat = self._ray_directions_w.reshape((ray_count,))
        global_world_only = bool(getattr(self.cfg, "global_world_only", False))
        if global_world_only:
            world_ids = np.full(ray_count, -1, dtype=np.int32)
        else:
            world_ids = np.repeat(np.arange(self._num_envs, dtype=np.int32), self.num_rays)
        self._ray_worlds = wp.array(world_ids, dtype=wp.int32, device=self._device)
        self._hit_dist = wp.empty(ray_count, dtype=wp.float32, device=self._device)
        self._hit_normal = wp.empty(ray_count, dtype=wp.vec3f, device=self._device)

        self._sensor_task_name = f"newton_raycast:{self.cfg.prim_path}:{id(self)}"
        NewtonManager._register_sensor_task(self._sensor_task_name, self._launch_raycast, include_collision_shapes=True)

    def _launch_raycast(self) -> None:
        """Sensor pose + ray transform + BVH query + hit resolve (graph-capturable)."""
        self._update_ray_infos(self._is_outdated)
        newton.intersect_ray(
            NewtonManager.get_model(),
            ray_origins=self._ray_starts_w_flat,
            ray_directions=self._ray_directions_w_flat,
            ray_worlds=self._ray_worlds,
            enable_global_world=not bool(getattr(self.cfg, "global_world_only", False)),
            out_dist=self._hit_dist,
            out_normal=self._hit_normal,
        )
        wp.launch(
            _resolve_bvh_hits_kernel,
            dim=(self._num_envs, self.num_rays),
            inputs=[
                self._is_outdated,
                self._ray_starts_w,
                self._ray_directions_w,
                self._hit_dist,
                self._hit_normal,
                float(self.cfg.max_distance),
                self.ray_cast_drift.warp,
            ],
            outputs=[
                self._data._ray_hits_w,
                self._data._ray_distances,
                self._data._ray_normals_w,
            ],
            device=self._device,
        )

    def _update_buffers_impl(self, env_mask: wp.array) -> None:
        # The captured graph is bound to ``_is_outdated``; mirror any other mask into it.
        if env_mask.ptr != self._is_outdated.ptr:
            wp.copy(self._is_outdated, env_mask)
        assert self._sensor_task_name is not None
        NewtonManager._update_sensor_tasks(self._sensor_task_name)

    def _invalidate_initialize_callback(self, event) -> None:
        if self._sensor_task_name is not None:
            NewtonManager._unregister_sensor_task(self._sensor_task_name)
        self._sensor_task_name = None
        super()._invalidate_initialize_callback(event)


class RayCaster(NewtonRaycastSensor):
    """Default Newton ray caster backed by the live scene BVH."""
