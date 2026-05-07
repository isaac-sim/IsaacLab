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

from isaaclab_physx.physics import PhysxManager as SimulationManager

if TYPE_CHECKING:
    from isaaclab.sensors.ray_caster import RayCasterCfg

logger = logging.getLogger(__name__)


@wp.kernel(enable_backward=False)
def _compose_body_xform_kernel(
    body_transforms: wp.array(dtype=wp.transformf),
    fixed_xform: wp.array(dtype=wp.transformf),
    sensor_transforms: wp.array(dtype=wp.transformf),
):
    """Per-env compose: ``sensor_transforms[i] = body_transforms[i] * fixed_xform[i]``."""
    i = wp.tid()
    sensor_transforms[i] = body_transforms[i] * fixed_xform[i]


class RayCaster(BaseRayCaster):
    """PhysX backend for the ray-cast sensor.

    Per-step body pose from a ``RigidObjectView``; the constant body→Xform offset
    (resolved once at init) is composed each step in :meth:`_get_sensor_transforms_wp`
    via :func:`_compose_body_xform_kernel`. The base ``update_ray_caster_kernel`` then
    layers ``cfg.offset`` from :attr:`_offset_pos_wp` / :attr:`_offset_quat_wp`. No
    FrameView/Fabric path, so the sensor follows the body through physics integration.

    Static (non-physics) parents skip the view: per-env USD world poses are read once
    and cached as a ``wp.transformf`` array.
    """

    cfg: RayCasterCfg
    __backend_name__: str = "physx"

    def __init__(self, cfg: RayCasterCfg):
        super().__init__(cfg)
        # Rigid-body branch: ``_body_view`` + ``_fixed_transform_wp`` /
        # ``_sensor_transforms_wp`` (compose kernel I/O). Static: ``_static_transforms``.
        self._physics_sim_view = None
        self._body_view = None
        self._static_transforms: wp.array | None = None
        self._rigid_parent_expr: str | None = None
        self._fixed_transform_wp: wp.array | None = None
        self._sensor_transforms_wp: wp.array | None = None

    def _initialize_impl(self) -> None:
        """Resolve rigid-body ancestor; prep the per-step source.

        Three branches: no rigid-body ancestor → static fallback (cache per-env USD
        poses); ancestor == prim → body pose IS the sensor pose; ancestor above prim
        → cache ``T_body_to_xform``, compose with body pose each step.

        Crucially, ``T_body_to_xform`` is NOT baked into ``_offset_pos_wp`` /
        ``_offset_quat_wp``: those carry ``cfg.offset`` for the camera path
        (zero-copy torch-aliased — mutating the warp side corrupts the aliases).
        """
        super()._initialize_impl()
        self._physics_sim_view = SimulationManager.get_physics_sim_view()
        prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if prim is None:
            raise RuntimeError(f"Failed to find a prim at path expression: {self.cfg.prim_path}")

        anchor = resolve_rigid_body_anchor(prim)
        if anchor is None:
            # Static fallback: cache per-env USD world poses, no view dependency.
            prims = sim_utils.find_matching_prims(self.cfg.prim_path)
            if len(prims) != self._num_envs:
                raise RuntimeError(
                    f"RayCaster '{self.cfg.prim_path}' static-parent fallback expected"
                    f" {self._num_envs} prims (one per env), got {len(prims)}."
                )
            rows = [[*p, *q] for p, q in (sim_utils.resolve_prim_pose(prim) for prim in prims)]
            poses = torch.tensor(rows, device=self._device, dtype=torch.float32).contiguous()
            self._static_transforms = wp.from_torch(poses).view(wp.transformf)
            return
        ancestor, fixed_offset = anchor

        if ancestor == prim:
            self._rigid_parent_expr = self.cfg.prim_path
        else:
            relative = prim.GetPath().MakeRelativePath(ancestor.GetPath()).pathString
            self._rigid_parent_expr = self.cfg.prim_path.replace("/" + relative, "")
            # Pack the fixed body→Xform transform once, per env, for the compose kernel.
            fixed_xform = torch.tensor(
                [fixed_offset] * self._num_envs, device=self._device, dtype=torch.float32
            ).contiguous()
            self._fixed_transform_wp = wp.from_torch(fixed_xform).view(wp.transformf)
            self._sensor_transforms_wp = wp.zeros(self._num_envs, dtype=wp.transformf, device=self._device)
        self._body_view = self._physics_sim_view.create_rigid_body_view(self._rigid_parent_expr.replace(".*", "*"))

    def _get_sensor_transforms_wp(self) -> wp.array:
        """Per-step sensor Xform world transforms — cached static array, raw body
        pose, or composed body * fixed offset, depending on which branch initialized."""
        if self._body_view is None:
            return self._static_transforms
        body_t = self._body_view.get_transforms().view(wp.transformf)
        if self._fixed_transform_wp is None:
            return body_t
        wp.launch(
            _compose_body_xform_kernel,
            dim=self._num_envs,
            inputs=[body_t, self._fixed_transform_wp],
            outputs=[self._sensor_transforms_wp],
            device=self._device,
        )
        return self._sensor_transforms_wp

    def _invalidate_initialize_callback(self, event) -> None:
        """Drop PhysX-owned refs on STOP; keep ``_static_transforms`` (USD-derived, immutable)."""
        super()._invalidate_initialize_callback(event)
        self._body_view = None
        self._physics_sim_view = None
