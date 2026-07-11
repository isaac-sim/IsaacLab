# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import warp as wp

import newton

from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster
from isaaclab.utils.warp import ProxyArray

from isaaclab_newton.physics import NewtonManager
from isaaclab_newton.sensors.ray_caster.ray_caster import _NewtonRayCasterMixin

from .newton_raycast_sensor_cfg import NewtonRaycastSensorCfg
from .newton_raycast_sensor_data import NewtonRaycastSensorData


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


class NewtonRaycastSensor(_NewtonRayCasterMixin, BaseRayCaster):
    """Ray-cast sensor that queries the whole Newton scene through the model's shape BVH.

    Rays are cast with :func:`newton.intersect_ray` against every collision
    shape in the sensor's own world plus the global world (e.g. terrain), so
    dynamic bodies are hit without configuring target meshes. The full update
    (sensor pose, ray transform, BVH query, hit resolve) is registered as a
    task in the manager's shared :class:`~isaaclab_newton.physics.BvhTaskGraph`,
    so it executes in the same CUDA graph as the Newton tiled-camera renderer
    and reuses its BVH refit.
    """

    cfg: NewtonRaycastSensorCfg
    """The configuration parameters."""

    def __init__(self, cfg: NewtonRaycastSensorCfg):
        super().__init__(cfg)
        self._data = NewtonRaycastSensorData()
        self._bvh_task_name: str | None = None

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

    def _initialize_warp_meshes(self):
        # Rays are cast against the scene BVH; no warp meshes are needed.
        pass

    def _initialize_impl(self):
        super()._initialize_impl()
        ray_count = self._num_envs * self.num_rays
        self._ray_starts_w_ta = ProxyArray(self._ray_starts_w)
        self._ray_directions_w_ta = ProxyArray(self._ray_directions_w)
        # Flat views and scratch buffers for newton.intersect_ray.
        self._ray_starts_w_flat = self._ray_starts_w.reshape((ray_count,))
        self._ray_directions_w_flat = self._ray_directions_w.reshape((ray_count,))
        if self.cfg.global_world_only:
            world_ids = np.full(ray_count, -1, dtype=np.int32)
        else:
            world_ids = np.repeat(np.arange(self._num_envs, dtype=np.int32), self.num_rays)
        self._ray_worlds = wp.array(world_ids, dtype=wp.int32, device=self._device)
        self._hit_dist = wp.empty(ray_count, dtype=wp.float32, device=self._device)
        self._hit_normal = wp.empty(ray_count, dtype=wp.vec3f, device=self._device)

        self._bvh_task_name = f"newton_raycast:{self.cfg.prim_path}:{id(self)}"
        NewtonManager.get_bvh_graph().register(self._bvh_task_name, self._launch_raycast)

    def _launch_raycast(self) -> None:
        """Sensor pose + ray transform + BVH query + hit resolve (graph-capturable)."""
        self._update_ray_infos(self._is_outdated)
        newton.intersect_ray(
            NewtonManager.get_model(),
            ray_origins=self._ray_starts_w_flat,
            ray_directions=self._ray_directions_w_flat,
            ray_worlds=self._ray_worlds,
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

    def _update_buffers_impl(self, env_mask: wp.array):
        # The captured graph is bound to ``_is_outdated``; mirror any other mask into it.
        if env_mask.ptr != self._is_outdated.ptr:
            wp.copy(self._is_outdated, env_mask)
        NewtonManager.get_bvh_graph().run(self._bvh_task_name)

    def _invalidate_initialize_callback(self, event):
        if self._bvh_task_name is not None and NewtonManager._bvh_graph is not None:
            NewtonManager._bvh_graph.unregister(self._bvh_task_name)
        self._bvh_task_name = None
        super()._invalidate_initialize_callback(event)
