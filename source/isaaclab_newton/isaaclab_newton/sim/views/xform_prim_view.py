# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton-backed XformPrimView — Warp-native, GPU-resident pose queries."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import warp as wp

from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sim.views.base_xform_prim_view import BaseXformPrimView

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
def _write_body_q_from_site_poses(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    site_local: wp.array(dtype=wp.transformf),
    world_pos: wp.array(dtype=wp.vec3f),
    world_quat: wp.array(dtype=wp.vec4f),
):
    """Compute ``body_q = world_site_pose * inverse(local)`` and write it.

    Launched over all sites; skips world-attached sites (``site_body == -1``)
    since they have no body to write to.
    """
    i = wp.tid()
    bid = site_body[i]
    if bid == -1:
        return

    inv_local = wp.transform_inverse(site_local[i])
    w_pos = world_pos[i]
    w_q = world_quat[i]
    world_tf = wp.transform(w_pos, wp.quatf(w_q[0], w_q[1], w_q[2], w_q[3]))
    body_q[bid] = wp.transform_multiply(world_tf, inv_local)


@wp.kernel
def _write_body_q_from_site_poses_indexed(
    body_q: wp.array(dtype=wp.transformf),
    site_body: wp.array(dtype=wp.int32),
    site_local: wp.array(dtype=wp.transformf),
    indices: wp.array(dtype=wp.int32),
    world_pos: wp.array(dtype=wp.vec3f),
    world_quat: wp.array(dtype=wp.vec4f),
):
    """Indexed variant: writes body_q for sites selected by ``indices``.

    ``world_pos`` and ``world_quat`` are dense over the index list (length M),
    while ``site_body`` and ``site_local`` are indexed via ``indices[i]``.
    """
    i = wp.tid()
    si = indices[i]
    bid = site_body[si]
    if bid == -1:
        return

    inv_local = wp.transform_inverse(site_local[si])
    w_pos = world_pos[i]
    w_q = world_quat[i]
    world_tf = wp.transform(w_pos, wp.quatf(w_q[0], w_q[1], w_q[2], w_q[3]))
    body_q[bid] = wp.transform_multiply(world_tf, inv_local)


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
# Helpers
# ------------------------------------------------------------------


def _ensure_wp_vec3f(data: wp.array, device: str) -> wp.array:
    """Pass-through for ``wp.array``; convert ``torch.Tensor`` via ``wp.from_torch``."""
    if isinstance(data, wp.array):
        return data
    import torch  # noqa: PLC0415

    if isinstance(data, torch.Tensor):
        return wp.from_torch(data.contiguous(), dtype=wp.vec3f)
    raise TypeError(f"Expected wp.array or torch.Tensor, got {type(data)}")


def _ensure_wp_vec4f(data: wp.array, device: str) -> wp.array:
    """Pass-through for ``wp.array``; convert ``torch.Tensor`` via ``wp.from_torch``."""
    if isinstance(data, wp.array):
        return data
    import torch  # noqa: PLC0415

    if isinstance(data, torch.Tensor):
        return wp.from_torch(data.contiguous(), dtype=wp.vec4f)
    raise TypeError(f"Expected wp.array or torch.Tensor, got {type(data)}")


# ------------------------------------------------------------------
# View class
# ------------------------------------------------------------------


class XformPrimView(BaseXformPrimView):
    """Batched prim view backed by Newton's native site concept on GPU.

    Each matched USD prim is resolved to a ``(body_index, local_transform)``
    pair.  World poses are computed on GPU as
    ``body_q[body_index] * local_transform`` via a Warp kernel, consistent
    with Newton's site mechanism.

    Resolution order (per prim path):

    1. **Shape label** -- look up in ``model.shape_label``.  If found, use
       ``shape_body`` and ``shape_transform`` for the body index and local
       offset.
    2. **Body label** -- look up in ``model.body_label``.  If found, the body
       itself is the site; local offset is identity.
    3. **Ancestor walk** -- walk the USD parent hierarchy until a prim whose
       path appears in ``model.body_label`` is found.  The relative transform
       from that ancestor body to the target prim is the local offset.  If no
       ancestor body exists, the site is attached to the world frame
       (``body_index = -1``) and the local offset is the prim's world
       transform.

    This supports arbitrary prims -- rigid bodies, collision shapes, cameras,
    plain Xforms, and any other Xformable prim.

    All getters return ``wp.array``.  Setters accept ``wp.array``.
    """

    def __init__(self, prim_path: str, device: str = "cpu", stage: Usd.Stage | None = None, **kwargs):
        self._prim_path = prim_path
        self._device = device

        stage = sim_utils.get_current_stage() if stage is None else stage
        self._prims: list[Usd.Prim] = sim_utils.find_matching_prims(prim_path, stage=stage)

        model = NewtonManager.get_model()
        body_labels = list(model.body_label)
        body_label_set = set(body_labels)
        shape_labels = list(model.shape_label)
        shape_body_np = model.shape_body.numpy()
        shape_xform_np = model.shape_transform.numpy()

        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

        site_bodies: list[int] = []
        site_locals: list[list[float]] = []

        identity_xform = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        for prim in self._prims:
            pp = prim.GetPath().pathString

            if pp in shape_labels:
                si = shape_labels.index(pp)
                site_bodies.append(int(shape_body_np[si]))
                site_locals.append(shape_xform_np[si].tolist())
            elif pp in body_label_set:
                site_bodies.append(body_labels.index(pp))
                site_locals.append(identity_xform)
            else:
                body_idx, local_xform = self._resolve_ancestor_body(prim, body_labels, body_label_set, xform_cache)
                site_bodies.append(body_idx)
                site_locals.append(local_xform)

        self._site_body = wp.array(site_bodies, dtype=wp.int32, device=device)
        self._site_local = wp.array(
            [wp.transform(*x) for x in site_locals],
            dtype=wp.transformf,
            device=device,
        )

        self._pos_buf = wp.zeros(self.count, dtype=wp.vec3f, device=device)
        self._quat_buf = wp.zeros(self.count, dtype=wp.vec4f, device=device)

    @staticmethod
    def _resolve_ancestor_body(
        prim: Usd.Prim,
        body_labels: list[str],
        body_label_set: set[str],
        xform_cache: UsdGeom.XformCache,
    ) -> tuple[int, list[float]]:
        """Walk USD ancestors to find the nearest Newton body and compute the relative local transform.

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
            if ancestor_path in body_label_set:
                body_idx = body_labels.index(ancestor_path)
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

    def get_world_poses(self, indices: Sequence[int] | None = None) -> tuple[wp.array, wp.array]:
        state = NewtonManager.get_state_0()

        if indices is not None:
            n = len(indices)
            idx_wp = wp.array(list(indices), dtype=wp.int32, device=self._device)
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
        indices: Sequence[int] | None = None,
    ) -> None:
        """Write world poses into Newton ``state.body_q``.

        For sites with a non-identity local transform the Warp kernel computes
        ``body_q = world_site_pose * inverse(local_transform)`` so that the
        resulting world pose of the *site* matches the requested value.
        World-attached sites (``body_index == -1``) are skipped.
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

        pos_wp = _ensure_wp_vec3f(positions, self._device)
        quat_wp = _ensure_wp_vec4f(orientations, self._device)

        if indices is not None:
            n = len(indices)
            idx_wp = wp.array(list(indices), dtype=wp.int32, device=self._device)
            wp.launch(
                _write_body_q_from_site_poses_indexed,
                dim=n,
                inputs=[state.body_q, self._site_body, self._site_local, idx_wp, pos_wp, quat_wp],
                device=self._device,
            )
        else:
            wp.launch(
                _write_body_q_from_site_poses,
                dim=self.count,
                inputs=[state.body_q, self._site_body, self._site_local, pos_wp, quat_wp],
                device=self._device,
            )

    # ------------------------------------------------------------------
    # Local poses -- delegate to world (Newton bodies live in world space)
    # ------------------------------------------------------------------

    def get_local_poses(self, indices: Sequence[int] | None = None) -> tuple[wp.array, wp.array]:
        return self.get_world_poses(indices)

    def set_local_poses(
        self,
        translations: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: Sequence[int] | None = None,
    ) -> None:
        self.set_world_poses(positions=translations, orientations=orientations, indices=indices)

    # ------------------------------------------------------------------
    # Scales
    # ------------------------------------------------------------------

    def get_scales(self, indices: Sequence[int] | None = None) -> wp.array:
        model = NewtonManager.get_model()
        num_shapes = model.shape_count

        if indices is not None:
            n = len(indices)
            idx_wp = wp.array(list(indices), dtype=wp.int32, device=self._device)
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

    def set_scales(self, scales: wp.array, indices: Sequence[int] | None = None) -> None:
        model = NewtonManager.get_model()
        num_shapes = model.shape_count
        scales_wp = _ensure_wp_vec3f(scales, self._device)

        if indices is not None:
            n = len(indices)
            idx_wp = wp.array(list(indices), dtype=wp.int32, device=self._device)
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
