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
    """Compute world-space transforms for every site in the view.

    For each site *i*, computes ``world = body_q[site_body[i]] * site_local[i]``
    and splits the result into position and quaternion outputs. When
    ``site_body[i] == -1`` the site is world-attached and ``site_local[i]`` is
    returned directly.

    Args:
        body_q: Rigid-body world transforms from the Newton state, shape ``[num_bodies]``.
        site_body: Per-site body index (flat model-level), shape ``[num_sites]``.
            A value of ``-1`` indicates a world-attached site.
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
    """Indexed variant of :func:`_compute_site_world_transforms`.

    Only computes world transforms for the subset of sites selected by
    ``indices``. Thread *i* reads ``indices[i]`` to obtain the site index,
    then writes the result to ``out_pos[i]`` / ``out_quat[i]``.

    Args:
        body_q: Rigid-body world transforms from the Newton state, shape ``[num_bodies]``.
        site_body: Per-site body index (flat model-level), shape ``[num_sites]``.
        site_local: Per-site local offset relative to its parent body, shape ``[num_sites]``.
        indices: Site indices to query, shape ``[M]``.
        out_pos: Output world positions [m], shape ``[M]``.
        out_quat: Output world orientations as ``(qx, qy, qz, qw)``, shape ``[M]``.
    """
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
    """Gather per-site scales from collision shapes on the same body.

    For each site *i*, linearly scans all shapes to find the first one whose
    ``shape_body`` matches ``site_body[i]`` and copies its scale. Falls back
    to ``(1, 1, 1)`` if no shape is found on that body.

    Args:
        shape_scale: Per-shape scale vectors from the Newton model, shape ``[num_shapes]``.
        shape_body: Per-shape parent body index, shape ``[num_shapes]``.
        site_body: Per-site body index, shape ``[num_sites]``.
        num_shapes: Total number of shapes in the model.
        out_scales: Output scale per site, shape ``[num_sites]``.
    """
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
    """Indexed variant of :func:`_gather_scales`.

    Args:
        shape_scale: Per-shape scale vectors from the Newton model, shape ``[num_shapes]``.
        shape_body: Per-shape parent body index, shape ``[num_shapes]``.
        site_body: Per-site body index, shape ``[num_sites]``.
        indices: Site indices to query, shape ``[M]``.
        num_shapes: Total number of shapes in the model.
        out_scales: Output scale per queried site, shape ``[M]``.
    """
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
    site_body: wp.array(dtype=wp.int32),
    new_scales: wp.array(dtype=wp.vec3f),
    shape_body: wp.array(dtype=wp.int32),
    num_shapes: wp.int32,
    shape_scale: wp.array(dtype=wp.vec3f),
):
    """Scatter per-site scales to all collision shapes on the same body.

    For each site *i*, writes ``new_scales[i]`` to every shape whose
    ``shape_body`` matches ``site_body[i]``. Multiple shapes on the same
    body all receive the same scale.

    Args:
        site_body: Per-site body index, shape ``[num_sites]``.
        new_scales: New scale to apply per site, shape ``[num_sites]``.
        shape_body: Per-shape parent body index, shape ``[num_shapes]``.
        num_shapes: Total number of shapes in the model.
        shape_scale: Per-shape scale vectors to write into (modified in-place),
            shape ``[num_shapes]``.
    """
    i = wp.tid()
    bid = site_body[i]
    for s in range(num_shapes):
        if shape_body[s] == bid:
            shape_scale[s] = new_scales[i]


@wp.kernel
def _scatter_scales_indexed(
    site_body: wp.array(dtype=wp.int32),
    indices: wp.array(dtype=wp.int32),
    new_scales: wp.array(dtype=wp.vec3f),
    shape_body: wp.array(dtype=wp.int32),
    num_shapes: wp.int32,
    shape_scale: wp.array(dtype=wp.vec3f),
):
    """Indexed variant of :func:`_scatter_scales`.

    Args:
        site_body: Per-site body index, shape ``[num_sites]``.
        indices: Site indices to update, shape ``[M]``.
        new_scales: New scale to apply per selected site, shape ``[M]``.
        shape_body: Per-shape parent body index, shape ``[num_shapes]``.
        num_shapes: Total number of shapes in the model.
        shape_scale: Per-shape scale vectors to write into (modified in-place),
            shape ``[num_shapes]``.
    """
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
    world_pos: wp.array(dtype=wp.vec3f),
    world_quat: wp.array(dtype=wp.vec4f),
    site_local: wp.array(dtype=wp.transformf),
):
    """Update site local offsets so that the sites reach desired world poses.

    For each site *i*, computes
    ``site_local[i] = inv(body_q[site_body[i]]) * desired_world`` so that
    a subsequent ``body_q[bid] * site_local[i]`` yields the requested world
    pose. For world-attached sites (``site_body[i] == -1``) the desired world
    transform is written directly into ``site_local[i]``.

    Does **not** modify ``body_q``.

    Args:
        body_q: Rigid-body world transforms from the Newton state, shape ``[num_bodies]``.
        site_body: Per-site body index (flat model-level), shape ``[num_sites]``.
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
    """Indexed variant of :func:`_write_site_local_from_world_poses`.

    Args:
        body_q: Rigid-body world transforms from the Newton state, shape ``[num_bodies]``.
        site_body: Per-site body index (flat model-level), shape ``[num_sites]``.
        indices: Site indices to update, shape ``[M]``.
        world_pos: Desired world positions [m], shape ``[M]``.
        world_quat: Desired world orientations as ``(qx, qy, qz, qw)``, shape ``[M]``.
        site_local: Per-site local offset (modified in-place), shape ``[num_sites]``.
    """
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
    """Compute parent-relative transforms for every site in the view.

    For each site *i*, computes the world pose of both the site and its USD
    parent, then returns ``inv(parent_world) * prim_world``. When
    ``site_body[i] == -1`` the site is world-attached and ``site_local[i]``
    is used as the world transform directly. The same convention applies to
    the parent arrays.

    Args:
        body_q: Rigid-body world transforms from the Newton state, shape ``[num_bodies]``.
        site_body: Per-site body index (flat model-level), shape ``[num_sites]``.
        site_local: Per-site local offset relative to its parent body, shape ``[num_sites]``.
        parent_site_body: Per-site USD-parent body index, shape ``[num_sites]``.
        parent_site_local: Per-site USD-parent local offset, shape ``[num_sites]``.
        out_pos: Output parent-relative positions [m], shape ``[num_sites]``.
        out_quat: Output parent-relative orientations as ``(qx, qy, qz, qw)``,
            shape ``[num_sites]``.
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
    """Indexed variant of :func:`_compute_site_local_transforms`.

    Args:
        body_q: Rigid-body world transforms from the Newton state, shape ``[num_bodies]``.
        site_body: Per-site body index (flat model-level), shape ``[num_sites]``.
        site_local: Per-site local offset relative to its parent body, shape ``[num_sites]``.
        parent_site_body: Per-site USD-parent body index, shape ``[num_sites]``.
        parent_site_local: Per-site USD-parent local offset, shape ``[num_sites]``.
        indices: Site indices to query, shape ``[M]``.
        out_pos: Output parent-relative positions [m], shape ``[M]``.
        out_quat: Output parent-relative orientations as ``(qx, qy, qz, qw)``,
            shape ``[M]``.
    """
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
    """Update site local offsets so that sites reach desired parent-relative poses.

    For each site *i*, reconstructs the desired world pose as
    ``parent_world * desired_local``, then solves for the body-relative offset:
    ``site_local[i] = inv(body_q[bid]) * desired_world``. For world-attached
    sites (``site_body[i] == -1``) the world transform is written directly.

    Does **not** modify ``body_q``.

    Args:
        body_q: Rigid-body world transforms from the Newton state, shape ``[num_bodies]``.
        site_body: Per-site body index (flat model-level), shape ``[num_sites]``.
        parent_site_body: Per-site USD-parent body index, shape ``[num_sites]``.
        parent_site_local: Per-site USD-parent local offset, shape ``[num_sites]``.
        local_pos: Desired parent-relative positions [m], shape ``[num_sites]``.
        local_quat: Desired parent-relative orientations as ``(qx, qy, qz, qw)``,
            shape ``[num_sites]``.
        site_local: Per-site local offset (modified in-place), shape ``[num_sites]``.
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
    parent_site_body: wp.array(dtype=wp.int32),
    parent_site_local: wp.array(dtype=wp.transformf),
    indices: wp.array(dtype=wp.int32),
    local_pos: wp.array(dtype=wp.vec3f),
    local_quat: wp.array(dtype=wp.vec4f),
    site_local: wp.array(dtype=wp.transformf),
):
    """Indexed variant of :func:`_write_site_local_from_local_poses`.

    Args:
        body_q: Rigid-body world transforms from the Newton state, shape ``[num_bodies]``.
        site_body: Per-site body index (flat model-level), shape ``[num_sites]``.
        parent_site_body: Per-site USD-parent body index, shape ``[num_sites]``.
        parent_site_local: Per-site USD-parent local offset, shape ``[num_sites]``.
        indices: Site indices to update, shape ``[M]``.
        local_pos: Desired parent-relative positions [m], shape ``[M]``.
        local_quat: Desired parent-relative orientations as ``(qx, qy, qz, qw)``,
            shape ``[M]``.
        site_local: Per-site local offset (modified in-place), shape ``[num_sites]``.
    """
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
        """Initialize the Newton site-based frame view.

        Resolves all USD prims matching ``prim_path`` and, for each one, walks
        the USD ancestor hierarchy to find the nearest Newton rigid body. The
        relative transform between the prim and its ancestor body becomes the
        site's local offset.

        If the Newton model is already finalized the view initializes
        immediately; otherwise initialization is deferred to a
        :attr:`PhysicsEvent.PHYSICS_READY` callback.

        Args:
            prim_path: USD prim path pattern (may contain regex).
            device: Warp device for GPU arrays (e.g. ``"cuda:0"``).
            stage: USD stage to search. Defaults to the current stage.
            **kwargs: Unused; accepted for interface compatibility with other
                :class:`~isaaclab.sim.views.BaseFrameView` backends.
        """
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
    def prims(self) -> list:
        """List of USD prims being managed by this view."""
        return self._prims

    @property
    def count(self) -> int:
        """Number of prims in this view."""
        return len(self._prims)

    # ------------------------------------------------------------------
    # World poses
    # ------------------------------------------------------------------

    def get_world_poses(self, indices: wp.array | None = None) -> tuple[wp.array, wp.array]:
        """Get world-space positions and orientations.

        Args:
            indices: Subset of sites to query. ``None`` means all sites.

        Returns:
            A tuple ``(positions, orientations)`` as ``wp.array`` of shapes
            ``(M, 3)`` and ``(M, 4)`` respectively.
        """
        state = NewtonManager.get_state_0()

        if indices is not None:
            n = len(indices)
            pos_buf = wp.zeros(n, dtype=wp.vec3f, device=self._device)
            quat_buf = wp.zeros(n, dtype=wp.vec4f, device=self._device)
            wp.launch(
                _compute_site_world_transforms_indexed,
                dim=n,
                inputs=[state.body_q, self._site_body, self._site_local, indices],
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
        """Set world-space positions and/or orientations.

        Updates the internal ``site_local`` offsets so that
        ``body_q[body] * new_site_local`` yields the desired world pose.
        Does **not** modify ``body_q``.

        Args:
            positions: Desired world positions ``(M, 3)``. ``None`` leaves
                positions unchanged.
            orientations: Desired world quaternions ``(M, 4)`` as
                ``(qx, qy, qz, qw)``. ``None`` leaves orientations unchanged.
            indices: Subset of sites to update. ``None`` means all sites.
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

        if indices is not None:
            wp.launch(
                _write_site_local_from_world_poses_indexed,
                dim=len(indices),
                inputs=[state.body_q, self._site_body, indices, positions, orientations, self._site_local],
                device=self._device,
            )
        else:
            wp.launch(
                _write_site_local_from_world_poses,
                dim=self.count,
                inputs=[state.body_q, self._site_body, positions, orientations, self._site_local],
                device=self._device,
            )

    # ------------------------------------------------------------------
    # Local poses (parent-relative)
    # ------------------------------------------------------------------

    def get_local_poses(self, indices: wp.array | None = None) -> tuple[wp.array, wp.array]:
        """Get parent-relative positions and orientations.

        Computes ``inv(parent_world) * prim_world`` for each site.

        Args:
            indices: Subset of sites to query. ``None`` means all sites.

        Returns:
            A tuple ``(translations, orientations)`` as ``wp.array`` of shapes
            ``(M, 3)`` and ``(M, 4)`` respectively.
        """
        state = NewtonManager.get_state_0()

        if indices is not None:
            n = len(indices)
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
                    indices,
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
        """Set parent-relative translations and/or orientations.

        Updates the internal ``site_local`` offsets so that
        ``inv(parent_world) * (body_q[bid] * site_local)`` yields the desired
        local pose. Does **not** modify ``body_q``.

        Args:
            translations: Desired parent-relative translations ``(M, 3)``.
                ``None`` leaves translations unchanged.
            orientations: Desired parent-relative quaternions ``(M, 4)`` as
                ``(qx, qy, qz, qw)``. ``None`` leaves orientations unchanged.
            indices: Subset of sites to update. ``None`` means all sites.
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

        if indices is not None:
            wp.launch(
                _write_site_local_from_local_poses_indexed,
                dim=len(indices),
                inputs=[
                    state.body_q,
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
                    state.body_q,
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
    # Scales
    # ------------------------------------------------------------------

    def get_scales(self, indices: wp.array | None = None) -> wp.array:
        """Get per-site scales by reading from the first collision shape on the same body.

        Args:
            indices: Subset of sites to query. ``None`` means all sites.

        Returns:
            A ``wp.array`` of shape ``(M, 3)``.
        """
        model = NewtonManager.get_model()
        num_shapes = model.shape_count

        if indices is not None:
            n = len(indices)
            out = wp.zeros(n, dtype=wp.vec3f, device=self._device)
            wp.launch(
                _gather_scales_indexed,
                dim=n,
                inputs=[model.shape_scale, model.shape_body, self._site_body, indices, num_shapes],
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
        """Set per-site scales by writing to all collision shapes on the same body.

        Args:
            scales: New scales ``(M, 3)`` as ``wp.array``.
            indices: Subset of sites to update. ``None`` means all sites.
        """
        model = NewtonManager.get_model()
        num_shapes = model.shape_count

        if indices is not None:
            wp.launch(
                _scatter_scales_indexed,
                dim=len(indices),
                inputs=[self._site_body, indices, scales, model.shape_body, num_shapes, model.shape_scale],
                device=self._device,
            )
        else:
            wp.launch(
                _scatter_scales,
                dim=self.count,
                inputs=[self._site_body, scales, model.shape_body, num_shapes, model.shape_scale],
                device=self._device,
            )


def _gf_matrix_to_xform7(mat: Gf.Matrix4d) -> list[float]:
    """Convert a ``Gf.Matrix4d`` to ``[tx, ty, tz, qx, qy, qz, qw]``."""
    t = mat.ExtractTranslation()
    q = mat.ExtractRotationQuat()
    imag = q.GetImaginary()
    return [float(t[0]), float(t[1]), float(t[2]), float(imag[0]), float(imag[1]), float(imag[2]), float(q.GetReal())]
