# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX ray-caster sensors -- mixin + concrete RayCaster class."""

from __future__ import annotations

import contextlib
import logging
from types import SimpleNamespace
from typing import Any

import torch
import warp as wp

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sensors.ray_caster.base_ray_caster import BaseRayCaster
from isaaclab.sensors.ray_caster.kernels import copy_mesh_transforms_to_table_kernel

from isaaclab_ovphysx.physics import OvPhysxManager

logger = logging.getLogger(__name__)


def _find_physics_ancestor(prim):
    """Return the nearest ancestor of ``prim`` that carries ``UsdPhysics.RigidBodyAPI``.

    Walks upward from ``prim`` itself. Returns ``None`` if no ancestor in the
    USD hierarchy applies the API.
    """
    ancestor = prim
    while ancestor and ancestor.IsValid() and ancestor.GetPath().pathString != "/":
        if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
            return ancestor
        ancestor = ancestor.GetParent()
    return None


def _ovphysx_body_glob(body_expr: str) -> str:
    """Convert internal env regex/template expressions to ovphysx glob syntax.

    The ovphysx wheel's ``create_tensor_binding`` ``pattern=`` argument is an
    fnmatch glob, so ``{}`` template placeholders and ``.*`` regex segments
    both map to ``*``.
    """
    return body_expr.replace("{}", "*").replace(".*", "*")


class _OvPhysxRayCasterMixin:
    """OVPhysX pose tracking for ray-caster sensors.

    Lives as a multiple-inheritance mixin on top of the four
    :class:`~isaaclab.sensors.ray_caster.Base*` classes. Provides backend-
    specific pose tracking via the ovphysx ``RIGID_BODY_POSE`` tensor binding
    when the sensor prim has a rigid-body ancestor, or a one-time USD pose
    snapshot for non-physics sensor frames.

    All backend-specific surface is centralized here so the four concrete
    sensor classes can be 14-line composition modules.
    """

    @property
    def count(self: Any) -> int:
        """Number of tracked sensor frames (binding row count, or static count)."""
        return self._view_count

    def _initialize_pose_tracking(self: Any) -> None:
        """Resolve sensor prims to either a live ovphysx binding or a static snapshot."""
        from isaaclab_ovphysx import tensor_types as TT  # noqa: PLC0415

        # Generic base-class hooks read ``self._view.count``; point that
        # adapter at the sensor itself rather than constructing a separate
        # view object (matches the PhysX trick).
        self._view = self

        try:
            body_expr, fixed_pos_b, fixed_quat_b = self._resolve_rigid_body_ancestor_expr()
        except RuntimeError:
            prims = sim_utils.find_matching_prims(self.cfg.prim_path)
            if len(prims) == 0:
                raise
            body = _find_physics_ancestor(prims[0])
            if body is None:
                self._initialize_static_pose_tracking(prims)
                return
            raise

        body_glob = _ovphysx_body_glob(body_expr)

        physx = OvPhysxManager.get_physx_instance()
        if physx is None:
            raise RuntimeError(
                "OvPhysxManager has no PhysX instance yet -- sensor was constructed before "
                "PhysicsEvent.PHYSICS_READY. Ensure the simulation has been reset at least once."
            )

        self._ovphysx_body_view = physx.create_tensor_binding(
            pattern=body_glob,
            tensor_type=TT.RIGID_BODY_POSE,
        )
        if self._ovphysx_body_view.shape[0] == 0:
            raise RuntimeError(f"OVPhysX RIGID_BODY_POSE binding for pattern {body_glob!r} matched zero bodies.")

        self._view_count = int(self._ovphysx_body_view.shape[0])
        self._pose_buf = wp.zeros(self._ovphysx_body_view.shape, dtype=wp.float32, device=self._device)
        # Zero-copy reinterpret of the ``(N, 7)`` float32 staging buffer as
        # ``(N,)`` ``wp.transformf``. Cached so per-step
        # ``_get_view_transforms_wp`` reads don't churn allocations.
        self._pose_buf_transformf = wp.array(
            ptr=self._pose_buf.ptr,
            shape=(self._view_count,),
            dtype=wp.transformf,
            device=str(self._pose_buf.device),
            copy=False,
        )

        if fixed_pos_b is None or fixed_quat_b is None:
            fixed_pos_b = (0.0, 0.0, 0.0)
            fixed_quat_b = (0.0, 0.0, 0.0, 1.0)
        offset_pos = [fixed_pos_b] * self._view_count
        offset_quat = [fixed_quat_b] * self._view_count
        self._offset_pos_wp = wp.array(offset_pos[: self._view_count], dtype=wp.vec3f, device=self._device)
        self._offset_quat_contiguous = torch.tensor(
            offset_quat[: self._view_count], dtype=torch.float32, device=self._device
        )
        self._offset_quat_wp = wp.from_torch(self._offset_quat_contiguous, dtype=wp.quatf)
        self._mesh_view_bufs = {}

    def _initialize_static_pose_tracking(self: Any, prims) -> None:
        """Cache authored USD poses for non-physics sensor frames.

        Used when the sensor prim has no rigid-body ancestor (e.g. an Xform
        marker under ``/World``). The cached poses are returned every frame
        unchanged -- static prims don't move.
        """
        poses = []
        for prim in prims:
            pos, quat = sim_utils.resolve_prim_pose(prim)
            poses.append((*pos, *quat))
        self._static_view_transforms_torch = torch.tensor(poses, dtype=torch.float32, device=self._device).contiguous()
        self._static_view_transforms_wp = wp.from_torch(self._static_view_transforms_torch).view(wp.transformf)
        self._ovphysx_body_view = None
        self._view_count = len(prims)
        self._offset_pos_wp = wp.zeros(self._view_count, dtype=wp.vec3f, device=self._device)
        identity_quat = torch.zeros(self._view_count, 4, device=self._device)
        identity_quat[:, 3] = 1.0
        self._offset_quat_contiguous = identity_quat.contiguous()
        self._offset_quat_wp = wp.from_torch(self._offset_quat_contiguous, dtype=wp.quatf)
        self._mesh_view_bufs = {}

    def _get_view_transforms_wp(self: Any) -> wp.array:
        """Return tracked sensor-frame transforms as a ``wp.transformf`` array.

        Live path reads the ovphysx binding into the cached staging buffer
        every call; static path returns the cached snapshot directly.
        """
        if self._ovphysx_body_view is None:
            return self._static_view_transforms_wp
        self._ovphysx_body_view.read(self._pose_buf)
        return self._pose_buf_transformf

    def get_world_poses(self: Any, indices=None):
        """Return world poses as ``(positions, orientations)`` pose tuples.

        Camera-derived base classes inheriting this mixin call this method
        and read ``.torch`` on the returned objects. We mirror PhysX's
        :class:`SimpleNamespace` shape so the contract is identical.
        """
        transforms = self._get_view_transforms_wp()
        transforms_t = wp.to_torch(transforms).reshape(-1, 7)
        if indices is not None:
            idx = wp.to_torch(indices).to(dtype=torch.long) if isinstance(indices, wp.array) else indices
            transforms_t = transforms_t[idx]
        return SimpleNamespace(torch=transforms_t[:, 0:3]), SimpleNamespace(torch=transforms_t[:, 3:7])

    def _create_tracked_target_view(self: Any, target_prim_paths: str | list[str]):
        """Create an ovphysx RIGID_BODY_POSE binding for dynamic multi-mesh targets.

        v1 limitation: target paths must dedup to a single env-wildcard
        pattern. Multi-pattern targets raise ``NotImplementedError`` -- the
        same limit the OVPhysX ``ContactSensor`` already documents for
        ``track_pose``.
        """
        from isaaclab_ovphysx import tensor_types as TT  # noqa: PLC0415

        if isinstance(target_prim_paths, str):
            target_prim_paths = [target_prim_paths]

        body_paths: list[str] = []
        for target_prim_path in target_prim_paths:
            prims = sim_utils.find_matching_prims(target_prim_path)
            if len(prims) == 0:
                # ClonePlan-backed targets may not have USD destination prims;
                # in that case BaseMultiMeshRayCaster forwards the
                # destination owner-body expression directly.
                body_paths.append(target_prim_path)
                continue
            for prim in prims:
                body = _find_physics_ancestor(prim)
                if body is None:
                    raise RuntimeError(
                        f"Cannot track non-physics ray-cast target {target_prim_path!r} "
                        "with OVPhysX. Set track_mesh_transforms=False for static targets, "
                        "or apply RigidBodyAPI to dynamic targets."
                    )
                body_paths.append(body.GetPath().pathString)

        if len(body_paths) == 0:
            raise RuntimeError(f"No tracked target bodies resolved from: {target_prim_paths}")

        patterns = sorted({_ovphysx_body_glob(path) for path in body_paths})
        if len(patterns) > 1:
            raise NotImplementedError(
                f"OvPhysxRayCaster v1 supports a single body-type pattern for dynamic targets; "
                f"resolved {len(patterns)} patterns: {patterns}. Multi-pattern targets require "
                "per-pattern bindings and an interleaved-read kernel that does not exist yet."
            )

        physx = OvPhysxManager.get_physx_instance()
        if physx is None:
            raise RuntimeError(
                "OvPhysxManager has no PhysX instance yet -- multi-mesh target view requested "
                "before PhysicsEvent.PHYSICS_READY."
            )
        return physx.create_tensor_binding(pattern=patterns[0], tensor_type=TT.RIGID_BODY_POSE)

    def _update_mesh_transforms(self: Any) -> None:
        """Refresh dynamic multi-mesh target poses from their ovphysx bindings."""
        if not hasattr(self, "_mesh_views"):
            return
        mesh_idx = 0
        for view, target_cfg in zip(self._mesh_views, self._raycast_targets_cfg):
            if not target_cfg.track_mesh_transforms:
                mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]
                continue

            # ``view`` here is an ovphysx TensorBinding produced by
            # :meth:`_create_tracked_target_view`. Each binding owns its own
            # staging buffer cached in ``self._mesh_view_bufs`` (initialized
            # in :meth:`_initialize_pose_tracking`).
            buf = self._mesh_view_bufs.get(id(view))
            if buf is None:
                buf = wp.zeros(view.shape, dtype=wp.float32, device=self._device)
                self._mesh_view_bufs[id(view)] = buf

            view.read(buf)
            transforms_wp = wp.array(
                ptr=buf.ptr,
                shape=(int(view.shape[0]),),
                dtype=wp.transformf,
                device=str(buf.device),
                copy=False,
            )

            view_count = int(view.shape[0])
            meshes_per_env = view_count
            if view_count != 1:
                meshes_per_env = view_count // self._num_envs

            wp.launch(
                copy_mesh_transforms_to_table_kernel,
                dim=(self._num_envs, meshes_per_env),
                inputs=[
                    transforms_wp,
                    int(meshes_per_env),
                    int(mesh_idx),
                    bool(view_count == 1),
                    self._mesh_positions_w,
                    self._mesh_orientations_w,
                ],
                device=self._device,
            )
            mesh_idx += self._num_meshes_per_env[target_cfg.prim_expr]

    def _invalidate_initialize_callback(self: Any, event) -> None:
        """Release ovphysx native handles when the simulation stops."""
        super()._invalidate_initialize_callback(event)
        view = getattr(self, "_ovphysx_body_view", None)
        if view is not None:
            with contextlib.suppress(Exception):
                view.destroy()
        self._ovphysx_body_view = None

        for buf_view in getattr(self, "_mesh_views", []) or []:
            with contextlib.suppress(Exception):
                buf_view.destroy()
        self._mesh_views = []
        self._mesh_view_bufs = {}


class RayCaster(_OvPhysxRayCasterMixin, BaseRayCaster):
    """OVPhysX RayCaster implementation."""
