# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

from isaaclab.sensors.ray_caster import BaseMultiMeshRayCaster
from isaaclab.sensors.ray_caster._target_tracker_utils import split_env_path, walk_target_prototypes
from isaaclab.sim import SimulationContext

from isaaclab_physx.physics import PhysxManager as SimulationManager

from .ray_caster import RayCaster

if TYPE_CHECKING:
    from isaaclab.sensors.ray_caster import MultiMeshRayCasterCfg

logger = logging.getLogger(__name__)


@wp.kernel(enable_backward=False)
def _compose_target_xform_kernel(
    body_transforms: wp.array(dtype=wp.transformf),
    body_view_idx: wp.array2d(dtype=wp.int32),
    fixed_offsets: wp.array2d(dtype=wp.transformf),
    slot_dst: wp.array(dtype=wp.int32),
    mesh_positions: wp.array2d(dtype=wp.vec3),
    mesh_rotations: wp.array2d(dtype=wp.quat),
):
    """``mesh_pose[env, slot] = body_pose[body_view_idx[env, slot]] * fixed_offset[env, slot]``."""
    e, s = wp.tid()
    xf = body_transforms[body_view_idx[e, s]] * fixed_offsets[e, s]
    slot = slot_dst[s]
    mesh_positions[e, slot] = wp.transform_get_translation(xf)
    mesh_rotations[e, slot] = wp.transform_get_rotation(xf)


class MultiMeshRayCaster(BaseMultiMeshRayCaster, RayCaster):
    """PhysX backend for the multi-mesh ray-cast sensor.

    Per-prototype ``(body_path, offset)`` from
    :func:`~isaaclab.sensors.ray_caster._target_tracker_utils.walk_target_prototypes`
    + ``clone_mask`` per-env assignment → single union ``RigidObjectView`` + per-step
    compose kernel writing ``body_pose * body_to_mesh`` into ``_mesh_positions_w`` /
    ``_mesh_orientations_w``. Each env consults its own prototype's offset, so
    heterogeneous replication is handled.
    """

    cfg: MultiMeshRayCasterCfg
    __backend_name__: str = "physx"

    def _initialize_impl(self) -> None:
        """Build target-mesh ``RigidObjectView`` + per-(env, slot) view indices and offsets."""
        super()._initialize_impl()
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        sim = SimulationContext.instance()
        plans = list(sim.get_clone_plans().values()) if sim is not None else []

        body_paths: list[list[str]] = [[] for _ in range(self._num_envs)]
        offsets: list[list[list[float]]] = [[] for _ in range(self._num_envs)]
        slot_dst: list[int] = []
        unique_patterns: list[str] = []

        for target_cfg, (slot_start, slot_end) in zip(self._raycast_targets_cfg, self._target_slot_ranges):
            if not target_cfg.track_mesh_transforms:
                continue
            slot_dst.extend(range(slot_start, slot_end))
            per_proto, env_proto = walk_target_prototypes(
                target_cfg.prim_expr, plans, self._num_envs, self.cfg.prim_path
            )
            self._distribute_per_env(per_proto, env_proto, body_paths, offsets, unique_patterns)

        self._num_tracked_per_env: int = len(slot_dst)
        if self._num_tracked_per_env == 0:
            self._target_view = None
            return

        self._target_view = self._physics_sim_view.create_rigid_body_view(unique_patterns)
        path_to_idx = {p: i for i, p in enumerate(self._target_view.prim_paths)}
        view_idx_2d = np.zeros((self._num_envs, self._num_tracked_per_env), dtype=np.int32)
        for env_idx, env_body_paths in enumerate(body_paths):
            for s, body_path in enumerate(env_body_paths):
                if body_path not in path_to_idx:
                    raise RuntimeError(
                        f"MultiMeshRayCaster '{self.cfg.prim_path}': body '{body_path}' missing from"
                        f" RigidObjectView created from {unique_patterns}."
                    )
                view_idx_2d[env_idx, s] = path_to_idx[body_path]

        self._body_view_idx_wp = wp.array(view_idx_2d, dtype=wp.int32, device=self._device)
        # offsets shape (num_envs, n_tracked, 7); wp.from_torch with dtype=wp.transformf
        # collapses the trailing 7-axis so the wp.array is 2D over (env, slot).
        offsets_torch = torch.tensor(offsets, dtype=torch.float32, device=self._device).contiguous()
        self._fixed_offsets_wp = wp.from_torch(offsets_torch, dtype=wp.transformf)
        self._slot_dst_wp = wp.array(slot_dst, dtype=wp.int32, device=self._device)
        logger.info(
            f"MultiMeshRayCaster '{self.cfg.prim_path}': tracking {self._num_tracked_per_env} target"
            f" mesh(es) per env across {len(unique_patterns)} body pattern(s)."
        )

    def _distribute_per_env(
        self,
        per_proto: list[list[tuple[str, list[float]]]],
        env_proto: list[int],
        body_paths: list[list[str]],
        offsets: list[list[list[float]]],
        unique_patterns: list[str],
    ) -> None:
        """Per env: rewrite first-env body path → ``env_<env_idx>`` and ``env_*`` (union pattern)."""
        for env_idx, proto_idx in enumerate(env_proto):
            for body_path_first, offset in per_proto[proto_idx]:
                first_env, rest = split_env_path(body_path_first)
                if first_env is None:
                    body_path_env = pattern = body_path_first
                else:
                    body_path_env = f"/World/envs/env_{env_idx}/{rest}"
                    pattern = f"/World/envs/env_*/{rest}"
                body_paths[env_idx].append(body_path_env)
                offsets[env_idx].append(offset)
                if pattern not in unique_patterns:
                    unique_patterns.append(pattern)

    def _update_target_mesh_transforms(self) -> None:
        """Compose body poses with cached body-to-mesh offsets per step."""
        if self._target_view is None or self._num_tracked_per_env == 0:
            return
        body_transforms = self._target_view.get_transforms().view(wp.transformf)
        wp.launch(
            _compose_target_xform_kernel,
            dim=(self._num_envs, self._num_tracked_per_env),
            inputs=[body_transforms, self._body_view_idx_wp, self._fixed_offsets_wp, self._slot_dst_wp],
            outputs=[self._mesh_positions_w, self._mesh_orientations_w],
            device=self._device,
        )

    def _invalidate_initialize_callback(self, event) -> None:
        """Drop target view on STOP (sensor body via super)."""
        super()._invalidate_initialize_callback(event)
        self._target_view = None
