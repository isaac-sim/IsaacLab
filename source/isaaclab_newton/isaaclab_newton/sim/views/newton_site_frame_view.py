# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-backed FrameView — Warp-native, GPU-resident pose queries."""

from __future__ import annotations

import logging

import warp as wp

from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.physics import PhysicsEvent
from isaaclab.sim.views.base_frame_view import BaseFrameView

from isaaclab_newton.physics.newton_manager import NewtonManager

logger = logging.getLogger(__name__)

WORLD_BODY_INDEX = -1


# ------------------------------------------------------------------
# Warp kernels
# ------------------------------------------------------------------


@wp.kernel
def _compute_site_world_transforms(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    site_local: wp.array(dtype=wp.transformf),
    out_pos: wp.array(dtype=wp.vec3f),
    out_quat: wp.array(dtype=wp.vec4f),
):
    """Compute world transforms for all sites: ``world = body_q[body] * local``.

    When ``site_body[i] == -1`` the site is attached to the world frame and
    the world transform equals ``site_local[i]`` directly.
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
    """Compute world transforms for a subset of sites selected by ``indices``."""
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
def _gather_scales(
    shape_scale: wp.array(dtype=wp.vec3f),
    shape_body: wp.array(dtype=wp.int32),
    site_body: wp.array(dtype=wp.int32),
    num_shapes: wp.int32,
    out_scales: wp.array(dtype=wp.vec3f),
):
    """For each site, find the first shape on the same body and copy its scale."""
    i = wp.tid()
    bid = site_body[i]
    found = int(0)
    for s in range(num_shapes):
        if shape_body[s] == bid and found == 0:
            out_scales[i] = shape_scale[s]
            found = 1
    if found == 0:
        out_scales[i] = wp.vec3f(1.0, 1.0, 1.0)


@wp.kernel
def _gather_scales_indexed(
    shape_scale: wp.array(dtype=wp.vec3f),
    shape_body: wp.array(dtype=wp.int32),
    site_body: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    num_shapes: wp.int32,
    out_scales: wp.array(dtype=wp.vec3f),
):
    """Indexed variant of _gather_scales."""
    i = wp.tid()
    si = indices[i]
    bid = site_body[si]
    found = int(0)
    for s in range(num_shapes):
        if shape_body[s] == bid and found == 0:
            out_scales[i] = shape_scale[s]
            found = 1
    if found == 0:
        out_scales[i] = wp.vec3f(1.0, 1.0, 1.0)


@wp.kernel
def _scatter_scales(
    shape_scale: wp.array(dtype=wp.vec3f),
    shape_body: wp.array(dtype=wp.int32),
    site_body: wp.array(dtype=wp.int32),
    num_shapes: wp.int32,
    new_scales: wp.array(dtype=wp.vec3f),
):
    """For each site, write its scale to all shapes on the same body."""
    i = wp.tid()
    bid = site_body[i]
    for s in range(num_shapes):
        if shape_body[s] == bid:
            shape_scale[s] = new_scales[i]


@wp.kernel
def _scatter_scales_indexed(
    shape_scale: wp.array(dtype=wp.vec3f),
    shape_body: wp.array(dtype=wp.int32),
    site_body: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    num_shapes: wp.int32,
    new_scales: wp.array(dtype=wp.vec3f),
):
    """Indexed variant of _scatter_scales."""
    i = wp.tid()
    si = indices[i]
    bid = site_body[si]
    for s in range(num_shapes):
        if shape_body[s] == bid:
            shape_scale[s] = new_scales[i]


# ------------------------------------------------------------------
# World-pose site_local write kernels
# ------------------------------------------------------------------


@wp.kernel
def _write_site_local_from_world_poses(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    site_local: wp.array(dtype=wp.transformf),
    world_pos: wp.array(dtype=wp.vec3f),
    world_quat: wp.array(dtype=wp.vec4f),
):
    """Update ``site_local`` so that ``body_q[bid] * site_local == desired_world``.

    Computes ``site_local[i] = inv(body_q[bid]) * desired_world``.
    For world-attached sites (``site_body == -1``) writes the world transform
    directly into ``site_local``.
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
    site_local: wp.array(dtype=wp.transformf),
    indices: wp.array(dtype=wp.int32),
    world_pos: wp.array(dtype=wp.vec3f),
    world_quat: wp.array(dtype=wp.vec4f),
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


# ------------------------------------------------------------------
# Local-pose Warp kernels
# ------------------------------------------------------------------


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
    """Compute parent-relative transforms: ``local = inv(parent_world) * prim_world``.

    When ``site_body[i] == -1`` the prim is attached to the world frame and
    ``site_local[i]`` is its world transform.  The same convention applies to
    ``parent_site_body`` / ``parent_site_local``.
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
    """Compute parent-relative transforms for a subset of sites selected by ``indices``."""
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
    site_local: wp.array(dtype=wp.transformf),
    parent_site_body: wp.array(dtype=wp.int32),
    parent_site_local: wp.array(dtype=wp.transformf),
    local_pos: wp.array(dtype=wp.vec3f),
    local_quat: wp.array(dtype=wp.vec4f),
):
    """Update ``site_local`` so that ``inv(parent_world) * prim_world == desired_local``.

    Computes ``site_local[i] = inv(body_q[bid]) * parent_world * desired_local``.
    For world-attached sites (``site_body == -1``) the site local IS the world
    transform, so we write ``parent_world * desired_local`` directly.
    """
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
    site_local: wp.array(dtype=wp.transformf),
    parent_site_body: wp.array(dtype=wp.int32),
    parent_site_local: wp.array(dtype=wp.transformf),
    indices: wp.array(dtype=wp.int32),
    local_pos: wp.array(dtype=wp.vec3f),
    local_quat: wp.array(dtype=wp.vec4f),
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


# ------------------------------------------------------------------
# View class
# ------------------------------------------------------------------


class NewtonSiteFrameView(BaseFrameView):
    """Batched prim view for non-physics prims tracked as sites on Newton bodies.

    Each matched USD prim must be a **non-physics** prim (camera, sensor,
    Xform marker, etc.) that sits as a child of a Newton rigid body in the
    USD hierarchy.  The prim path must **not** resolve directly to a physics
    body or collision shape -- those are owned by Newton and should be
    accessed through :class:`~isaaclab_newton.assets.Articulation` or
    :class:`~isaaclab_newton.assets.RigidObject` instead.

    At init time each prim is resolved to a ``(body_index, site_local)``
    pair via ancestor walk: the nearest ancestor that appears in
    ``model.body_label`` becomes the attachment body, and the relative USD
    transform becomes the site offset.  If no body ancestor exists the prim
    is attached to the world frame (``body_index = -1``).

    World poses are computed on GPU as
    ``body_q[body_index] * site_local`` via a Warp kernel.  Both
    ``set_world_poses`` and ``set_local_poses`` update ``site_local`` --
    neither touches ``body_q``.

    All getters return ``wp.array``.  Setters accept ``wp.array``.

    Raises:
        ValueError: If any matched prim resolves to a Newton physics body
            or collision shape.
    """

    def __init__(self, prim_path: str, device: str = "cpu", stage: Usd.Stage | None = None, **kwargs):
        self._prim_path = prim_path
        self._device = device

        stage = sim_utils.get_current_stage() if stage is None else stage
        self._prims: list[Usd.Prim] = sim_utils.find_matching_prims(prim_path, stage=stage)

        model = NewtonManager.get_model()
        if model is not None:
            self._initialize_impl(model)
        else:
            self._physics_ready_handle = NewtonManager.register_callback(
                self._on_physics_ready, PhysicsEvent.PHYSICS_READY, name=f"site_view_{prim_path}"
            )

    def _on_physics_ready(self, _event) -> None:
        """Callback invoked when the Newton model becomes available."""
        self._initialize_impl(NewtonManager.get_model())

    def _initialize_impl(self, model) -> None:
        """Resolve USD prims to Newton body indices and allocate GPU buffers."""
        body_labels = list(model.body_label)
        body_label_set = set(body_labels)
        body_label_to_idx = {path: idx for idx, path in enumerate(body_labels)}
        shape_label_set = set(model.shape_label)

        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        site_bodies: list[int] = []
        site_locals: list[list[float]] = []
        parent_bodies: list[int] = []
        parent_locals: list[list[float]] = []

        identity_xform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        resolve_cache: dict[str, tuple[int, list[float]]] = {}

        for prim in self._prims:
            pp = prim.GetPath().pathString
            if pp in body_label_set:
                raise ValueError(
                    f"FrameView prim '{pp}' is a Newton physics body. "
                    "FrameView should only be used for non-physics prims (cameras, sensors, Xform markers). "
                    "Use Articulation or RigidObject APIs to control physics bodies."
                )
            if pp in shape_label_set:
                raise ValueError(
                    f"FrameView prim '{pp}' is a Newton collision shape. "
                    "FrameView should only be used for non-physics prims (cameras, sensors, Xform markers). "
                    "Use Articulation or RigidObject APIs to control collision shapes."
                )

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
        self._site_local = wp.array(
            [wp.transform(*x) for x in site_locals],
            dtype=wp.transformf,
            device=device,
        )
        self._parent_site_body = wp.array(parent_bodies, dtype=wp.int32, device=device)
        self._parent_site_local = wp.array(
            [wp.transform(*x) for x in parent_locals],
            dtype=wp.transformf,
            device=device,
        )

        self._pos_buf = wp.zeros(self.count, dtype=wp.vec3f, device=device)
        self._quat_buf = wp.zeros(self.count, dtype=wp.vec4f, device=device)
        self._local_pos_buf = wp.zeros(self.count, dtype=wp.vec3f, device=device)
        self._local_quat_buf = wp.zeros(self.count, dtype=wp.vec4f, device=device)

    @staticmethod
    def _resolve_ancestor_body(
        prim: Usd.Prim,
        body_label_to_idx: dict[str, int],
        xform_cache: UsdGeom.XformCache,
    ) -> tuple[int, list[float]]:
        """Walk USD ancestors to find the nearest Newton body and compute the relative local transform.

        Args:
            prim: The USD prim to resolve.
            body_label_to_idx: Dict mapping body prim paths to their Newton body indices.
            xform_cache: USD xform cache for efficient transform lookups.

        Returns:
            A tuple ``(body_index, local_xform_7)`` where *local_xform_7* is
            ``[tx, ty, tz, qx, qy, qz, qw]``.  If no body ancestor exists,
            ``body_index`` is :data:`WORLD_BODY_INDEX` and the local transform
            is the prim's world transform.
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

    @property
    def count(self) -> int:
        return len(self._prims)

    # ------------------------------------------------------------------
    # World poses
    # ------------------------------------------------------------------

    def get_world_poses(self, indices: wp.array | None = None) -> tuple[wp.array, wp.array]:
        state = NewtonManager.get_state_0()

        if indices is not None:
            n = len(indices)
            idx_wp = indices
            pos_buf = wp.zeros(n, dtype=wp.vec3f, device=self._device)
            quat_buf = wp.zeros(n, dtype=wp.vec4f, device=self._device)
            wp.launch(
                _compute_site_world_transforms_indexed,
                dim=n,
                inputs=[state.body_q, self._site_body, self._site_local, idx_wp],
                outputs=[pos_buf, quat_buf],
                device=self._device,
            )
            return pos_buf, quat_buf

        wp.launch(
            _compute_site_world_transforms,
            dim=self.count,
            inputs=[state.body_q, self._site_body, self._site_local],
            outputs=[self._pos_buf, self._quat_buf],
            device=self._device,
        )
        return self._pos_buf, self._quat_buf

    def set_world_poses(
        self,
        positions: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ) -> None:
        """Write world poses by updating the site's local offset.

        Computes the new ``site_local`` such that
        ``body_q[body] * new_site_local == desired_world``.
        Does not modify ``body_q``.
        """
        if positions is None and orientations is None:
            return

        state = NewtonManager.get_state_0()

        if positions is None or orientations is None:
            cur_pos, cur_quat = self.get_world_poses(indices)
            if positions is None:
                positions = cur_pos
            if orientations is None:
                orientations = cur_quat

        pos_wp = positions
        quat_wp = orientations

        if indices is not None:
            n = len(indices)
            idx_wp = indices
            wp.launch(
                _write_site_local_from_world_poses_indexed,
                dim=n,
                inputs=[state.body_q, self._site_body, self._site_local, idx_wp, pos_wp, quat_wp],
                device=self._device,
            )
        else:
            wp.launch(
                _write_site_local_from_world_poses,
                dim=self.count,
                inputs=[state.body_q, self._site_body, self._site_local, pos_wp, quat_wp],
                device=self._device,
            )

    # ------------------------------------------------------------------
    # Local poses (parent-relative)
    # ------------------------------------------------------------------

    def get_local_poses(self, indices: wp.array | None = None) -> tuple[wp.array, wp.array]:
        """Get parent-relative poses: ``local = inv(parent_world) * prim_world``."""
        state = NewtonManager.get_state_0()

        if indices is not None:
            n = len(indices)
            idx_wp = indices
            pos_buf = wp.zeros(n, dtype=wp.vec3f, device=self._device)
            quat_buf = wp.zeros(n, dtype=wp.vec4f, device=self._device)
            wp.launch(
                _compute_site_local_transforms_indexed,
                dim=n,
                inputs=[
                    state.body_q,
                    self._site_body,
                    self._site_local,
                    self._parent_site_body,
                    self._parent_site_local,
                    idx_wp,
                ],
                outputs=[pos_buf, quat_buf],
                device=self._device,
            )
            return pos_buf, quat_buf

        wp.launch(
            _compute_site_local_transforms,
            dim=self.count,
            inputs=[
                state.body_q,
                self._site_body,
                self._site_local,
                self._parent_site_body,
                self._parent_site_local,
            ],
            outputs=[self._local_pos_buf, self._local_quat_buf],
            device=self._device,
        )
        return self._local_pos_buf, self._local_quat_buf

    def set_local_poses(
        self,
        translations: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ) -> None:
        """Write parent-relative poses by updating the site's local offset.

        Computes the new ``site_local`` such that
        ``inv(parent_world) * (body_q[bid] * site_local) == desired_local``.
        """
        if translations is None and orientations is None:
            return

        state = NewtonManager.get_state_0()

        if translations is None or orientations is None:
            cur_pos, cur_quat = self.get_local_poses(indices)
            if translations is None:
                translations = cur_pos
            if orientations is None:
                orientations = cur_quat

        pos_wp = translations
        quat_wp = orientations

        if indices is not None:
            n = len(indices)
            idx_wp = indices
            wp.launch(
                _write_site_local_from_local_poses_indexed,
                dim=n,
                inputs=[
                    state.body_q,
                    self._site_body,
                    self._site_local,
                    self._parent_site_body,
                    self._parent_site_local,
                    idx_wp,
                    pos_wp,
                    quat_wp,
                ],
                device=self._device,
            )
        else:
            wp.launch(
                _write_site_local_from_local_poses,
                dim=self.count,
                inputs=[
                    state.body_q,
                    self._site_body,
                    self._site_local,
                    self._parent_site_body,
                    self._parent_site_local,
                    pos_wp,
                    quat_wp,
                ],
                device=self._device,
            )

    # ------------------------------------------------------------------
    # Scales
    # ------------------------------------------------------------------

    def get_scales(self, indices: wp.array | None = None) -> wp.array:
        model = NewtonManager.get_model()
        num_shapes = model.shape_count

        if indices is not None:
            n = len(indices)
            idx_wp = indices
            out = wp.zeros(n, dtype=wp.vec3f, device=self._device)
            wp.launch(
                _gather_scales_indexed,
                dim=n,
                inputs=[model.shape_scale, model.shape_body, self._site_body, idx_wp, num_shapes],
                outputs=[out],
                device=self._device,
            )
        else:
            out = wp.zeros(self.count, dtype=wp.vec3f, device=self._device)
            wp.launch(
                _gather_scales,
                dim=self.count,
                inputs=[model.shape_scale, model.shape_body, self._site_body, num_shapes],
                outputs=[out],
                device=self._device,
            )
        return out

    def set_scales(self, scales: wp.array, indices: wp.array | None = None) -> None:
        model = NewtonManager.get_model()
        num_shapes = model.shape_count
        scales_wp = scales

        if indices is not None:
            n = len(indices)
            idx_wp = indices
            wp.launch(
                _scatter_scales_indexed,
                dim=n,
                inputs=[model.shape_scale, model.shape_body, self._site_body, idx_wp, num_shapes, scales_wp],
                device=self._device,
            )
        else:
            wp.launch(
                _scatter_scales,
                dim=self.count,
                inputs=[model.shape_scale, model.shape_body, self._site_body, num_shapes, scales_wp],
                device=self._device,
            )


def _gf_matrix_to_xform7(mat: Gf.Matrix4d) -> list[float]:
    """Convert a ``Gf.Matrix4d`` to ``[tx, ty, tz, qx, qy, qz, qw]``."""
    t = mat.ExtractTranslation()
    q = mat.ExtractRotationQuat()
    imag = q.GetImaginary()
    return [float(t[0]), float(t[1]), float(t[2]), float(imag[0]), float(imag[1]), float(imag[2]), float(q.GetReal())]
