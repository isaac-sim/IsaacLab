# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX-backed FrameView -- Warp-native, GPU-resident pose queries."""

from __future__ import annotations

import logging
from typing import Any

import warp as wp

from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.physics import PhysicsEvent
from isaaclab.sim.views.base_frame_view import BaseFrameView
from isaaclab.sim.views.usd_frame_view import UsdFrameView
from isaaclab.utils.warp import ProxyArray

from isaaclab_ovphysx.physics import OvPhysxManager

logger = logging.getLogger(__name__)

WORLD_BODY_INDEX = -1


@wp.kernel
def _compute_site_world_transforms(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    site_local: wp.array(dtype=wp.transformf),
    out_pos: wp.array(dtype=wp.vec3f),
    out_quat: wp.array(dtype=wp.vec4f),
):
    """Compute world-space transforms for every site in the view.

    For each site *i*, computes ``world = body_q[site_body[i]] * site_local[i]``
    and splits the result into position and quaternion outputs.  When
    ``site_body[i] == -1`` the site is world-attached and ``site_local[i]`` is
    returned directly.

    Args:
        body_q: Rigid-body world transforms from the OVPhysX-backed Newton state,
            shape ``[num_bodies]``.
        site_body: Per-site body index (flat model-level), shape ``[num_sites]``.
            ``-1`` indicates a world-attached site.
        site_local: Per-site local offset relative to its parent body, shape ``[num_sites]``.
        out_pos: Output world positions [m], shape ``[num_sites]``.
        out_quat: Output world orientations as ``(qx, qy, qz, qw)``, shape ``[num_sites]``.
    """
    i = wp.tid()
    bid = site_body[i]
    if bid == -1:
        world = site_local[i]
    else:
        world = wp.transform_multiply(body_q[bid], site_local[i])
    out_pos[i] = wp.transform_get_translation(world)
    q = wp.transform_get_rotation(world)
    out_quat[i] = wp.vec4f(q[0], q[1], q[2], q[3])


@wp.kernel
def _compute_site_world_transforms_indexed(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    site_local: wp.array(dtype=wp.transformf),
    indices: wp.array(dtype=wp.int32),
    out_pos: wp.array(dtype=wp.vec3f),
    out_quat: wp.array(dtype=wp.vec4f),
):
    """Indexed variant of :func:`_compute_site_world_transforms`."""
    i = wp.tid()
    si = indices[i]
    bid = site_body[si]
    if bid == -1:
        world = site_local[si]
    else:
        world = wp.transform_multiply(body_q[bid], site_local[si])
    out_pos[i] = wp.transform_get_translation(world)
    q = wp.transform_get_rotation(world)
    out_quat[i] = wp.vec4f(q[0], q[1], q[2], q[3])


@wp.kernel
def _write_site_local_from_world_poses(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    world_pos: wp.array(dtype=wp.vec3f),
    world_quat: wp.array(dtype=wp.vec4f),
    site_local: wp.array(dtype=wp.transformf),
):
    """Update site local offsets so that sites reach desired world poses.

    For each site *i*, sets ``site_local[i] = inv(body_q[bid]) * desired_world``
    so that subsequent reads produce the requested world pose.  Does **not**
    modify ``body_q``.  World-attached sites (``site_body[i] == -1``) receive
    the desired world transform directly.

    Args:
        body_q: Rigid-body world transforms, shape ``[num_bodies]``.
        site_body: Per-site body index, shape ``[num_sites]``.
        world_pos: Desired world positions [m], shape ``[num_sites]``.
        world_quat: Desired world orientations as ``(qx, qy, qz, qw)``, shape ``[num_sites]``.
        site_local: Per-site local offset (modified in-place), shape ``[num_sites]``.
    """
    i = wp.tid()
    w_pos = world_pos[i]
    w_q = world_quat[i]
    desired_world = wp.transform(w_pos, wp.quatf(w_q[0], w_q[1], w_q[2], w_q[3]))
    bid = site_body[i]
    if bid == -1:
        site_local[i] = desired_world
    else:
        site_local[i] = wp.transform_multiply(wp.transform_inverse(body_q[bid]), desired_world)


@wp.kernel
def _write_site_local_from_world_poses_indexed(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    world_pos: wp.array(dtype=wp.vec3f),
    world_quat: wp.array(dtype=wp.vec4f),
    site_local: wp.array(dtype=wp.transformf),
):
    """Indexed variant of :func:`_write_site_local_from_world_poses`."""
    i = wp.tid()
    si = indices[i]
    w_pos = world_pos[i]
    w_q = world_quat[i]
    desired_world = wp.transform(w_pos, wp.quatf(w_q[0], w_q[1], w_q[2], w_q[3]))
    bid = site_body[si]
    if bid == -1:
        site_local[si] = desired_world
    else:
        site_local[si] = wp.transform_multiply(wp.transform_inverse(body_q[bid]), desired_world)


@wp.kernel
def _compute_site_local_transforms(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    site_local: wp.array(dtype=wp.transformf),
    parent_site_body: wp.array(dtype=wp.int32),
    parent_site_local: wp.array(dtype=wp.transformf),
    out_pos: wp.array(dtype=wp.vec3f),
    out_quat: wp.array(dtype=wp.vec4f),
):
    """Compute parent-relative transforms for every site in the view.

    For each site *i*, computes the world pose of both the site and its USD
    parent, then returns ``inv(parent_world) * prim_world``.  World-attached
    sites/parents use ``site_local`` / ``parent_site_local`` directly.
    """
    i = wp.tid()
    prim_bid = site_body[i]
    if prim_bid == -1:
        prim_world = site_local[i]
    else:
        prim_world = wp.transform_multiply(body_q[prim_bid], site_local[i])
    parent_bid = parent_site_body[i]
    if parent_bid == -1:
        parent_world = parent_site_local[i]
    else:
        parent_world = wp.transform_multiply(body_q[parent_bid], parent_site_local[i])
    local_tf = wp.transform_multiply(wp.transform_inverse(parent_world), prim_world)
    out_pos[i] = wp.transform_get_translation(local_tf)
    q = wp.transform_get_rotation(local_tf)
    out_quat[i] = wp.vec4f(q[0], q[1], q[2], q[3])


@wp.kernel
def _compute_site_local_transforms_indexed(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    site_local: wp.array(dtype=wp.transformf),
    parent_site_body: wp.array(dtype=wp.int32),
    parent_site_local: wp.array(dtype=wp.transformf),
    indices: wp.array(dtype=wp.int32),
    out_pos: wp.array(dtype=wp.vec3f),
    out_quat: wp.array(dtype=wp.vec4f),
):
    """Indexed variant of :func:`_compute_site_local_transforms`."""
    i = wp.tid()
    si = indices[i]
    prim_bid = site_body[si]
    if prim_bid == -1:
        prim_world = site_local[si]
    else:
        prim_world = wp.transform_multiply(body_q[prim_bid], site_local[si])
    parent_bid = parent_site_body[si]
    if parent_bid == -1:
        parent_world = parent_site_local[si]
    else:
        parent_world = wp.transform_multiply(body_q[parent_bid], parent_site_local[si])
    local_tf = wp.transform_multiply(wp.transform_inverse(parent_world), prim_world)
    out_pos[i] = wp.transform_get_translation(local_tf)
    q = wp.transform_get_rotation(local_tf)
    out_quat[i] = wp.vec4f(q[0], q[1], q[2], q[3])


@wp.kernel
def _write_site_local_from_local_poses(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    parent_site_body: wp.array(dtype=wp.int32),
    parent_site_local: wp.array(dtype=wp.transformf),
    local_pos: wp.array(dtype=wp.vec3f),
    local_quat: wp.array(dtype=wp.vec4f),
    site_local: wp.array(dtype=wp.transformf),
):
    """Update site local offsets so that sites reach desired parent-relative poses."""
    i = wp.tid()
    parent_bid = parent_site_body[i]
    if parent_bid == -1:
        parent_world = parent_site_local[i]
    else:
        parent_world = wp.transform_multiply(body_q[parent_bid], parent_site_local[i])
    l_pos = local_pos[i]
    l_q = local_quat[i]
    local_tf = wp.transform(l_pos, wp.quatf(l_q[0], l_q[1], l_q[2], l_q[3]))
    desired_world = wp.transform_multiply(parent_world, local_tf)
    bid = site_body[i]
    if bid == -1:
        site_local[i] = desired_world
    else:
        site_local[i] = wp.transform_multiply(wp.transform_inverse(body_q[bid]), desired_world)


@wp.kernel
def _write_site_local_from_local_poses_indexed(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    parent_site_body: wp.array(dtype=wp.int32),
    parent_site_local: wp.array(dtype=wp.transformf),
    indices: wp.array(dtype=wp.int32),
    local_pos: wp.array(dtype=wp.vec3f),
    local_quat: wp.array(dtype=wp.vec4f),
    site_local: wp.array(dtype=wp.transformf),
):
    """Indexed variant of :func:`_write_site_local_from_local_poses`."""
    i = wp.tid()
    si = indices[i]
    parent_bid = parent_site_body[si]
    if parent_bid == -1:
        parent_world = parent_site_local[si]
    else:
        parent_world = wp.transform_multiply(body_q[parent_bid], parent_site_local[si])
    l_pos = local_pos[i]
    l_q = local_quat[i]
    local_tf = wp.transform(l_pos, wp.quatf(l_q[0], l_q[1], l_q[2], l_q[3]))
    desired_world = wp.transform_multiply(parent_world, local_tf)
    bid = site_body[si]
    if bid == -1:
        site_local[si] = desired_world
    else:
        site_local[si] = wp.transform_multiply(wp.transform_inverse(body_q[bid]), desired_world)


class OvPhysxFrameView(BaseFrameView):
    """Batched prim view for non-physics prims tracked as sites on OVPhysX bodies.

    Each matched USD prim is resolved at init to a ``(body_index, site_local)``
    pair via ancestor walk: the nearest ancestor in the OVPhysX scene-data
    provider's ``_rigid_body_paths`` becomes the attachment body, and the
    relative USD transform becomes the site offset.  If no body ancestor
    exists, the prim is attached to the world frame
    (``body_index = WORLD_BODY_INDEX``).

    World poses are computed on GPU as ``body_q[body_index] * site_local`` via
    a Warp kernel, with the world-attached branch returning ``site_local``
    directly.  Both :meth:`set_world_poses` and :meth:`set_local_poses` update
    the view-owned ``site_local`` buffer -- neither writes to ``body_q``.

    Scales and visibility delegate to an internal :class:`UsdFrameView`
    (lazy-constructed on first call).

    Pose getters return :class:`~isaaclab.utils.warp.ProxyArray`.  Setters
    accept ``wp.array``.
    """

    def __init__(self, prim_path: str, device: str = "cpu", stage: Usd.Stage | None = None, **kwargs):
        """Initialize the OVPhysX site-based frame view.

        Args:
            prim_path: USD prim path pattern (may contain regex).
            device: Warp device for GPU arrays (e.g. ``"cuda:0"``).
            stage: USD stage to search. Defaults to the current stage.
            **kwargs: Forwarded to the lazy internal :class:`UsdFrameView`
                (e.g. ``validate_xform_ops``); accepted for backend-agnostic
                kwarg passing through the :class:`FrameView` factory.
        """
        self._prim_path = prim_path
        self._device = device
        self._kwargs = kwargs

        stage = sim_utils.get_current_stage() if stage is None else stage
        self._stage = stage
        self._prims: list[Usd.Prim] = sim_utils.find_matching_prims(prim_path, stage=stage)
        if not self._prims:
            raise ValueError(f"OvPhysxFrameView: pattern {prim_path!r} matched zero prims.")

        # Lazy USD view for scales / visibility.
        self._usd_view: UsdFrameView | None = None

        # Try synchronous init; defer to PHYSICS_READY if SDP not yet built.
        sdp = self._try_get_sdp()
        if sdp is not None and sdp.get_newton_state() is not None:
            self._initialize_impl(sdp)
        else:
            OvPhysxManager.register_callback(
                self._on_physics_ready,
                PhysicsEvent.PHYSICS_READY,
                name=f"ovphysx_frame_view_{prim_path}",
            )

    @staticmethod
    def _try_get_sdp() -> Any | None:
        """Return the active OVPhysX scene-data provider, or ``None`` if unavailable."""
        from isaaclab.sim.simulation_context import SimulationContext  # noqa: PLC0415

        ctx = SimulationContext.instance()
        if ctx is None:
            return None
        try:
            return ctx.initialize_scene_data_provider()
        except Exception:  # noqa: BLE001 -- SDP may not yet be built; defer.
            return None

    def _on_physics_ready(self, _event) -> None:
        """Callback invoked when the OVPhysX Newton state becomes available."""
        sdp = self._try_get_sdp()
        if sdp is None or sdp.get_newton_state() is None:
            raise RuntimeError(
                "OvPhysxFrameView: PHYSICS_READY fired but the scene data provider has no "
                "Newton state. Ensure your scene declares `requires_newton_model=True` "
                "(typically by including a sensor like ContactSensor or RayCaster)."
            )
        self._initialize_impl(sdp)

    def _initialize_impl(self, sdp: Any) -> None:
        """Resolve USD prims to OVPhysX body indices and allocate GPU buffers."""
        self._sdp = sdp
        body_labels = list(sdp._rigid_body_paths)
        body_label_to_idx = {path: idx for idx, path in enumerate(body_labels)}

        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        site_bodies: list[int] = []
        site_locals: list[list[float]] = []
        parent_bodies: list[int] = []
        parent_locals: list[list[float]] = []

        identity_xform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        resolve_cache: dict[str, tuple[int, list[float]]] = {}

        for prim in self._prims:
            body_idx, local_xform = self._resolve_ancestor_body(prim, body_label_to_idx, xform_cache)
            site_bodies.append(body_idx)
            site_locals.append(local_xform)

            parent = prim.GetParent()
            if not parent or not parent.IsValid() or parent.GetPath().pathString == "/":
                parent_bodies.append(WORLD_BODY_INDEX)
                parent_locals.append(identity_xform)
            else:
                parent_path = parent.GetPath().pathString
                if parent_path in resolve_cache:
                    pb_idx, pb_local = resolve_cache[parent_path]
                elif parent_path in body_label_to_idx:
                    pb_idx = body_label_to_idx[parent_path]
                    pb_local = identity_xform
                    resolve_cache[parent_path] = (pb_idx, pb_local)
                else:
                    pb_idx, pb_local = self._resolve_ancestor_body(parent, body_label_to_idx, xform_cache)
                    resolve_cache[parent_path] = (pb_idx, pb_local)
                parent_bodies.append(pb_idx)
                parent_locals.append(pb_local)

        device = self._device
        self._site_body = wp.array(site_bodies, dtype=wp.int32, device=device)
        self._site_local = wp.array([wp.transform(*x) for x in site_locals], dtype=wp.transformf, device=device)
        self._parent_site_body = wp.array(parent_bodies, dtype=wp.int32, device=device)
        self._parent_site_local = wp.array(
            [wp.transform(*x) for x in parent_locals], dtype=wp.transformf, device=device
        )

        self._num_bodies_snapshot = len(body_labels)

        self._pos_buf = wp.zeros(self.count, dtype=wp.vec3f, device=device)
        self._quat_buf = wp.zeros(self.count, dtype=wp.vec4f, device=device)
        self._local_pos_buf = wp.zeros(self.count, dtype=wp.vec3f, device=device)
        self._local_quat_buf = wp.zeros(self.count, dtype=wp.vec4f, device=device)
        self._pos_ta = ProxyArray(self._pos_buf)
        self._quat_ta = ProxyArray(self._quat_buf)
        self._local_pos_ta = ProxyArray(self._local_pos_buf)
        self._local_quat_ta = ProxyArray(self._local_quat_buf)

    @staticmethod
    def _resolve_ancestor_body(
        prim: Usd.Prim,
        body_label_to_idx: dict[str, int],
        xform_cache: UsdGeom.XformCache,
    ) -> tuple[int, list[float]]:
        """Walk USD ancestors to find the nearest OVPhysX body and the relative local transform.

        Returns:
            ``(body_index, [tx, ty, tz, qx, qy, qz, qw])``. ``body_index`` is
            :data:`WORLD_BODY_INDEX` when no body ancestor exists; the local
            transform in that case is the prim's world USD transform.
        """
        prim_world_tf = xform_cache.GetLocalToWorldTransform(prim)
        prim_world_tf.Orthonormalize()
        ancestor = prim.GetParent()
        while ancestor and ancestor.IsValid() and ancestor.GetPath().pathString != "/":
            ancestor_path = ancestor.GetPath().pathString
            body_idx = body_label_to_idx.get(ancestor_path)
            if body_idx is not None:
                ancestor_world_tf = xform_cache.GetLocalToWorldTransform(ancestor)
                ancestor_world_tf.Orthonormalize()
                local_tf = prim_world_tf * ancestor_world_tf.GetInverse()
                return body_idx, _gf_matrix_to_xform7(local_tf)
            ancestor = ancestor.GetParent()
        return WORLD_BODY_INDEX, _gf_matrix_to_xform7(prim_world_tf)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def prims(self) -> list[Usd.Prim]:
        """List of USD prims being managed by this view."""
        return self._prims

    @property
    def prim_paths(self) -> list[str]:
        """List of prim paths (as strings) for all prims being managed by this view."""
        if not hasattr(self, "_prim_paths_cache"):
            self._prim_paths_cache = [p.GetPath().pathString for p in self._prims]
        return self._prim_paths_cache

    @property
    def count(self) -> int:
        """Number of prims in this view."""
        return len(self._prims)

    @property
    def device(self) -> str:
        """Device where arrays are allocated (``"cpu"`` or ``"cuda:0"``)."""
        return self._device

    # ------------------------------------------------------------------
    # Initialization guard for deferred-init users
    # ------------------------------------------------------------------

    def _require_initialized(self) -> None:
        if not hasattr(self, "_site_body"):
            raise RuntimeError(
                "OvPhysxFrameView used before initialization. The view defers initialization "
                "until OvPhysxManager dispatches PhysicsEvent.PHYSICS_READY. Step the "
                "simulation once (or wait for physics to be ready) before calling pose methods."
            )

    def _current_body_q(self) -> wp.array:
        """Fetch the current OVPhysX ``body_q`` array, validating shape.

        Returns:
            ``wp.array(dtype=wp.transformf)`` of shape ``[num_bodies]``.

        Raises:
            RuntimeError: If the SDP has no Newton state or its size has changed since init.
        """
        state = self._sdp.get_newton_state()
        if state is None:
            raise RuntimeError(
                "OvPhysxFrameView: scene data provider returned no Newton state. "
                "Ensure your scene declares `requires_newton_model=True` (typically by "
                "including a sensor like ContactSensor or RayCaster)."
            )
        body_q = state.body_q
        if body_q.shape[0] != self._num_bodies_snapshot:
            raise RuntimeError(
                f"OvPhysxFrameView: body_q size changed ({body_q.shape[0]} vs "
                f"{self._num_bodies_snapshot} at init). Dynamic env counts are not supported."
            )
        return body_q

    # ------------------------------------------------------------------
    # World / local pose APIs (Tasks 5 & 6)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # World poses
    # ------------------------------------------------------------------

    def get_world_poses(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        """Get world-space positions and orientations.

        Args:
            indices: Subset of sites to query. ``None`` means all sites.

        Returns:
            A tuple ``(positions, orientations)`` of :class:`~isaaclab.utils.warp.ProxyArray`
            wrappers. Use ``.warp`` for the underlying ``wp.array`` or ``.torch`` for a
            cached zero-copy ``torch.Tensor`` view.
        """
        self._require_initialized()
        body_q = self._current_body_q()

        if indices is not None:
            n = len(indices)
            pos_buf = wp.zeros(n, dtype=wp.vec3f, device=self._device)
            quat_buf = wp.zeros(n, dtype=wp.vec4f, device=self._device)
            wp.launch(
                _compute_site_world_transforms_indexed,
                dim=n,
                inputs=[body_q, self._site_body, self._site_local, indices],
                outputs=[pos_buf, quat_buf],
                device=self._device,
            )
            return ProxyArray(pos_buf), ProxyArray(quat_buf)

        wp.launch(
            _compute_site_world_transforms,
            dim=self.count,
            inputs=[body_q, self._site_body, self._site_local],
            outputs=[self._pos_buf, self._quat_buf],
            device=self._device,
        )
        return self._pos_ta, self._quat_ta

    def set_world_poses(
        self,
        positions: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ) -> None:
        """Set world-space positions and/or orientations.

        Updates ``site_local`` so that ``body_q[body] * site_local`` yields the
        desired world pose.  Does **not** modify ``body_q``.

        Args:
            positions: Desired world positions ``(M, 3)`` [m]. ``None`` leaves
                positions unchanged.
            orientations: Desired world quaternions ``(M, 4)`` as
                ``(qx, qy, qz, qw)``. ``None`` leaves orientations unchanged.
            indices: Subset of sites to update. ``None`` means all sites.
        """
        if positions is None and orientations is None:
            return
        self._require_initialized()
        body_q = self._current_body_q()

        if positions is None or orientations is None:
            cur_pos_ta, cur_quat_ta = self.get_world_poses(indices)
            if positions is None:
                positions = cur_pos_ta.warp
            if orientations is None:
                orientations = cur_quat_ta.warp

        if indices is not None:
            wp.launch(
                _write_site_local_from_world_poses_indexed,
                dim=len(indices),
                inputs=[body_q, self._site_body, indices, positions, orientations, self._site_local],
                device=self._device,
            )
        else:
            wp.launch(
                _write_site_local_from_world_poses,
                dim=self.count,
                inputs=[body_q, self._site_body, positions, orientations, self._site_local],
                device=self._device,
            )

    # ------------------------------------------------------------------
    # Local poses (parent-relative)
    # ------------------------------------------------------------------

    def get_local_poses(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        """Get parent-relative positions and orientations.

        Computes ``inv(parent_world) * prim_world`` for each site.

        Args:
            indices: Subset of sites to query. ``None`` means all sites.

        Returns:
            A tuple ``(translations, orientations)`` of :class:`~isaaclab.utils.warp.ProxyArray`
            wrappers.
        """
        self._require_initialized()
        body_q = self._current_body_q()

        if indices is not None:
            n = len(indices)
            pos_buf = wp.zeros(n, dtype=wp.vec3f, device=self._device)
            quat_buf = wp.zeros(n, dtype=wp.vec4f, device=self._device)
            wp.launch(
                _compute_site_local_transforms_indexed,
                dim=n,
                inputs=[
                    body_q,
                    self._site_body,
                    self._site_local,
                    self._parent_site_body,
                    self._parent_site_local,
                    indices,
                ],
                outputs=[pos_buf, quat_buf],
                device=self._device,
            )
            return ProxyArray(pos_buf), ProxyArray(quat_buf)

        wp.launch(
            _compute_site_local_transforms,
            dim=self.count,
            inputs=[
                body_q,
                self._site_body,
                self._site_local,
                self._parent_site_body,
                self._parent_site_local,
            ],
            outputs=[self._local_pos_buf, self._local_quat_buf],
            device=self._device,
        )
        return self._local_pos_ta, self._local_quat_ta

    def set_local_poses(
        self,
        translations: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ) -> None:
        """Set parent-relative translations and/or orientations.

        Updates ``site_local`` only; does **not** modify ``body_q``.

        Args:
            translations: Desired parent-relative translations ``(M, 3)`` [m].
                ``None`` leaves translations unchanged.
            orientations: Desired parent-relative quaternions ``(M, 4)`` as
                ``(qx, qy, qz, qw)``. ``None`` leaves orientations unchanged.
            indices: Subset of sites to update. ``None`` means all sites.
        """
        if translations is None and orientations is None:
            return
        self._require_initialized()
        body_q = self._current_body_q()

        if translations is None or orientations is None:
            cur_pos_ta, cur_quat_ta = self.get_local_poses(indices)
            if translations is None:
                translations = cur_pos_ta.warp
            if orientations is None:
                orientations = cur_quat_ta.warp

        if indices is not None:
            wp.launch(
                _write_site_local_from_local_poses_indexed,
                dim=len(indices),
                inputs=[
                    body_q,
                    self._site_body,
                    self._parent_site_body,
                    self._parent_site_local,
                    indices,
                    translations,
                    orientations,
                    self._site_local,
                ],
                device=self._device,
            )
        else:
            wp.launch(
                _write_site_local_from_local_poses,
                dim=self.count,
                inputs=[
                    body_q,
                    self._site_body,
                    self._parent_site_body,
                    self._parent_site_local,
                    translations,
                    orientations,
                    self._site_local,
                ],
                device=self._device,
            )

    # ------------------------------------------------------------------
    # Scales & visibility -- delegate to UsdFrameView
    # ------------------------------------------------------------------

    def _ensure_usd_view(self) -> UsdFrameView:
        if self._usd_view is None:
            self._usd_view = UsdFrameView(
                self._prim_path,
                device=self._device,
                validate_xform_ops=self._kwargs.get("validate_xform_ops", True),
                stage=self._stage,
            )
        return self._usd_view

    def get_scales(self, indices: wp.array | None = None) -> wp.array:
        """Get scales for prims in the view (USD-backed)."""
        return self._ensure_usd_view().get_scales(indices)

    def set_scales(self, scales: wp.array, indices: wp.array | None = None) -> None:
        """Set scales for prims in the view (USD-backed)."""
        self._ensure_usd_view().set_scales(scales, indices)

    def get_visibility(self, indices: wp.array | None = None):
        """Get visibility for prims in the view (USD-backed).

        Note: OVPhysX runs without a Kit renderer, so visibility reads return
        the static USD stage state. Writes succeed at the USD layer but
        produce no visible change.
        """
        return self._ensure_usd_view().get_visibility(indices)

    def set_visibility(self, visibility, indices: wp.array | None = None) -> None:
        """Set visibility for prims in the view (USD-backed; no renderer effect under OVPhysX)."""
        self._ensure_usd_view().set_visibility(visibility, indices)


def _gf_matrix_to_xform7(mat: Gf.Matrix4d) -> list[float]:
    """Convert a ``Gf.Matrix4d`` to ``[tx, ty, tz, qx, qy, qz, qw]``."""
    t = mat.ExtractTranslation()
    q = mat.ExtractRotationQuat()
    imag = q.GetImaginary()
    return [float(t[0]), float(t[1]), float(t[2]), float(imag[0]), float(imag[1]), float(imag[2]), float(q.GetReal())]
