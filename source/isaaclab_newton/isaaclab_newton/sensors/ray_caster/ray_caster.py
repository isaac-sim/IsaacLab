# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.sensors.ray_caster import BaseRayCaster
from isaaclab.sensors.ray_caster._target_tracker_utils import resolve_rigid_body_anchor

from isaaclab_newton.physics import NewtonManager

if TYPE_CHECKING:
    from isaaclab.sensors.ray_caster import RayCasterCfg

logger = logging.getLogger(__name__)


class RayCaster(BaseRayCaster):
    """Newton backend for the ray-cast sensor.

    Site-based, mirroring :class:`~isaaclab_newton.sensors.pva.Pva` /
    :class:`~isaaclab_newton.sensors.frame_transformer.FrameTransformer`. ``__init__``
    walks USD to the rigid-body ancestor and registers a body-attached site (with
    the prim→body offset) plus a world-origin reference — must run before
    ``newton_replicate``. ``_initialize_impl`` resolves the per-env indices and builds
    a :class:`SensorFrameTransform` against world-origin; ``sensor.transforms`` is
    handed straight to :func:`update_ray_caster_kernel`.

    Static parents bypass sites (a global ``body=-1`` site lives in the main builder
    only, losing per-env origins) — fall back to a cached per-env ``wp.transformf``
    array from USD, like PhysX.
    """

    cfg: RayCasterCfg
    __backend_name__: str = "newton"

    def __init__(self, cfg: RayCasterCfg):
        super().__init__(cfg)
        # Rigid-body branch: site labels + ``SensorFrameTransform``. Static: per-env
        # ``wp.transformf`` array. ``_is_static_parent`` picks the branch.
        self._site_label: str | None = None
        self._world_origin_label: str | None = None
        self._sensor_index: int | None = None
        self._newton_transforms: wp.array | None = None
        self._site_args_cached: bool = False
        self._is_static_parent: bool = False
        self._cached_body_pattern: str | None = None
        self._cached_site_xform: wp.transform | None = None
        self._static_transforms: wp.array | None = None

        # Sites must register before ``newton_replicate`` (fires from ``start_simulation``).
        self._register_sites()

    def _register_sites(self) -> None:
        """Walk USD once, cache args, register sites. STOP→reinit reuses the cache —
        only the ``cl_register_site`` calls re-fire. Static branch defers per-env
        transform-array build to :meth:`_initialize_impl` (needs ``_num_envs``)."""
        if not self._site_args_cached:
            prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
            if prim is None:
                raise RuntimeError(f"Failed to find a prim at path expression: {self.cfg.prim_path}")
            anchor = resolve_rigid_body_anchor(prim)
            if anchor is None:
                self._is_static_parent = True
            else:
                ancestor, fixed_offset = anchor
                if ancestor == prim:
                    self._cached_body_pattern = self.cfg.prim_path
                    self._cached_site_xform = wp.transform()
                else:
                    relative = prim.GetPath().MakeRelativePath(ancestor.GetPath()).pathString
                    self._cached_body_pattern = self.cfg.prim_path.replace("/" + relative, "")
                    self._cached_site_xform = wp.transform(fixed_offset[:3], fixed_offset[3:])
            self._site_args_cached = True

        if self._is_static_parent:
            return  # static branch builds its per-env array in _initialize_impl
        self._world_origin_label = NewtonManager.cl_register_site(None, wp.transform())
        self._site_label = NewtonManager.cl_register_site(self._cached_body_pattern, self._cached_site_xform)

    def _initialize_impl(self) -> None:
        """Build the per-step transforms source. Static: per-env USD pose array.
        Rigid-body: resolve registered site indices, build ``SensorFrameTransform``
        against world-origin. Either way, exposes ``wp.transformf`` of shape
        ``(num_envs,)`` to :meth:`_get_sensor_transforms_wp`."""
        super()._initialize_impl()
        if self._is_static_parent:
            prims = sim_utils.find_matching_prims(self.cfg.prim_path)
            if len(prims) != self._num_envs:
                raise RuntimeError(
                    f"RayCaster '{self.cfg.prim_path}' static-parent fallback expected"
                    f" {self._num_envs} prims (one per env), got {len(prims)}."
                )
            rows = [[*p, *q] for p, q in (sim_utils.resolve_prim_pose(prim) for prim in prims)]
            poses = torch.tensor(rows, device=self._device, dtype=torch.float32).contiguous()
            self._static_transforms = wp.from_torch(poses).view(wp.transformf)
            logger.info(f"RayCaster '{self.cfg.prim_path}' initialized: {self._num_envs} envs (static parent)")
            return

        site_map = NewtonManager._cl_site_index_map
        for label in (self._world_origin_label, self._site_label):
            if label not in site_map:
                raise RuntimeError(
                    f"RayCaster '{self.cfg.prim_path}': site label '{label}'"
                    " missing from NewtonManager._cl_site_index_map."
                )

        # World-origin is global (body=-1) — same index per env.
        world_origin_global, _ = site_map[self._world_origin_label]
        references_list = [world_origin_global] * self._num_envs

        # Source site is body-attached (rigid-body branch reaches here exclusively).
        _, source_per_world = site_map[self._site_label]
        if len(source_per_world) != self._num_envs:
            raise RuntimeError(
                f"RayCaster '{self.cfg.prim_path}': source site has {len(source_per_world)}"
                f" world entries, expected {self._num_envs}."
            )
        shapes_list = []
        for env_idx, sites in enumerate(source_per_world):
            if len(sites) != 1:
                raise RuntimeError(
                    f"RayCaster '{self.cfg.prim_path}': pattern matched {len(sites)} bodies"
                    f" in env {env_idx}, expected exactly 1."
                )
            shapes_list.append(sites[0])

        self._sensor_index = NewtonManager.add_frame_transform_sensor(shapes_list, references_list)
        self._newton_transforms = NewtonManager._newton_frame_transform_sensors[self._sensor_index].transforms
        logger.info(
            f"RayCaster '{self.cfg.prim_path}' initialized: {self._num_envs} envs, sensor_index={self._sensor_index}"
        )

    def _get_sensor_transforms_wp(self) -> wp.array:
        """``SensorFrameTransform.transforms`` (rigid-body) or cached static array."""
        return self._static_transforms if self._is_static_parent else self._newton_transforms

    def _invalidate_initialize_callback(self, event) -> None:
        """Drop sensor refs on STOP and re-register sites — ``NewtonManager.close()``
        clears ``_cl_pending_sites`` first, so this revives them for the next
        ``start_simulation`` (mirrors :class:`Pva` / :class:`FrameTransformer`).
        USD-walk cache + static array are kept (immutable)."""
        super()._invalidate_initialize_callback(event)
        self._newton_transforms = None
        self._sensor_index = None
        if not self._is_static_parent:
            self._register_sites()
