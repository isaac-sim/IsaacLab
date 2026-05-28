# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX-backed FrameView -- Warp-native, GPU-resident pose queries."""

from __future__ import annotations

import logging
import re
from typing import Any

import warp as wp

from pxr import Gf, Usd, UsdGeom, UsdPhysics

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
    pair via ancestor walk: the nearest ancestor carrying ``UsdPhysics.RigidBodyAPI``
    becomes the attachment body, and the relative USD transform becomes the site
    offset. If no rigid-body ancestor exists, the prim is attached to the world
    frame (``body_index = WORLD_BODY_INDEX``) and ``site_local`` stores the prim's
    USD world transform.

    Body world poses are read each step via an OVPhysX ``RIGID_BODY_POSE`` tensor
    binding -- the same data path the contact sensor uses -- and **not** via the
    scene data provider's Newton model. This keeps the view usable in scenes
    that do not declare ``requires_newton_model=True``.

    World poses are computed on GPU as ``body_q[body_index] * site_local`` via
    a Warp kernel, with the world-attached branch returning ``site_local``
    directly. Both :meth:`set_world_poses` and :meth:`set_local_poses` update
    the view-owned ``site_local`` buffer -- neither writes to the physics state.

    Scales and visibility delegate to an internal :class:`UsdFrameView`
    (lazy-constructed on first call).

    Pose getters return :class:`~isaaclab.utils.warp.ProxyArray`.  Setters
    accept ``wp.array``.

    Limitations (v1):
        All resolved rigid-body ancestors (plus their USD parents for local-pose
        queries) must share a single env-wildcarded path pattern. Mixed
        body-types per view raise :class:`NotImplementedError`. The common
        case (one body type, wildcarded across envs) is fully supported.
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

        # Try synchronous init; defer to PHYSICS_READY if the PhysX instance is not yet alive.
        physx = self._try_get_physx()
        if physx is not None:
            self._initialize_impl(physx)
        else:
            OvPhysxManager.register_callback(
                self._on_physics_ready,
                PhysicsEvent.PHYSICS_READY,
                name=f"ovphysx_frame_view_{prim_path}",
            )

    @staticmethod
    def _try_get_physx() -> Any | None:
        """Return the active OVPhysX ``PhysX`` instance, or ``None`` if not yet created."""
        return OvPhysxManager.get_physx_instance()

    def _on_physics_ready(self, _event) -> None:
        """Callback invoked when the OVPhysX ``PhysX`` instance becomes available."""
        physx = self._try_get_physx()
        if physx is None:
            raise RuntimeError("OvPhysxFrameView: PHYSICS_READY fired but OvPhysxManager has no PhysX instance.")
        self._initialize_impl(physx)

    def _initialize_impl(self, physx: Any) -> None:
        """Resolve prims to rigid-body ancestors and create a RIGID_BODY_POSE tensor binding.

        Site discovery handles two scene-construction modes:

        * **``clone_usd=True``** (Newton-style cloning): every env has its own
          USD prims; ``find_matching_prims`` returns one prim per env, and the
          binding row count matches.
        * **``clone_usd=False``** (OVPhysX default): only ``env_0`` has authored
          USD prims; ``env_1..N`` are physics-layer clones (no USD twin). The
          RIGID_BODY_POSE binding still exposes one row per env. In that case
          the binding is the source of truth for the site count, and per-env
          site paths are synthesized from the env_0 template prim's path with
          ``env_0`` replaced by the row's env_id.
        """
        from isaaclab_ovphysx import tensor_types as TT  # noqa: PLC0415

        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        identity_xform7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        # 0. Reject prim_paths that resolve to a rigid body itself: a FrameView
        #    should track a non-physics child of a body (a sensor frame), not
        #    the body. Mirrors the Newton guard at
        #    ``newton_site_frame_view.py:572-584``.
        for prim in self._prims:
            if self._prim_or_template_has_rigid_body_api(prim):
                raise ValueError(
                    f"OvPhysxFrameView prim '{prim.GetPath().pathString}' resolves to a rigid body. "
                    "FrameView should only be used for non-physics prims (cameras, sensor mounts, "
                    "Xform markers). Use OvPhysX's RigidObject or Articulation APIs to control "
                    "physics bodies directly, or point prim_path at a non-physics child of the body."
                )

        # 1. Resolve each (template) prim's ancestor body + the prim->ancestor offset.
        per_prim_ancestor: list[str | None] = []
        per_prim_site_local: list[list[float]] = []
        for prim in self._prims:
            ap, sl = self._resolve_rigid_body_ancestor(prim, xform_cache)
            per_prim_ancestor.append(ap)
            per_prim_site_local.append(sl)

        # 2. Same resolution for each prim's USD parent (used by local-pose queries).
        parent_ancestor: list[str | None] = []
        parent_site_local: list[list[float]] = []
        for prim in self._prims:
            parent = prim.GetParent()
            if parent and parent.IsValid() and parent.GetPath().pathString != "/":
                pap, psl = self._resolve_rigid_body_ancestor(parent, xform_cache)
            else:
                pap, psl = None, identity_xform7
            parent_ancestor.append(pap)
            parent_site_local.append(psl)

        # 3. Dedup discovered ancestor paths into env-wildcarded patterns (one binding per pattern).
        all_ancestors = [p for p in (per_prim_ancestor + parent_ancestor) if p is not None]
        patterns = sorted({self._env_wildcardify(p) for p in all_ancestors})
        if len(patterns) > 1:
            raise NotImplementedError(
                f"OvPhysxFrameView v1 supports a single body-type pattern; resolved {len(patterns)}"
                f" patterns under prim_path={self._prim_path!r}: {patterns}."
            )

        # 4. Create the RIGID_BODY_POSE binding (or operate in world-only mode).
        if patterns:
            pattern = patterns[0]
            self._pose_binding = physx.create_tensor_binding(pattern=pattern, tensor_type=TT.RIGID_BODY_POSE)
            if self._pose_binding.shape[0] == 0:
                raise RuntimeError(
                    f"OvPhysxFrameView: RIGID_BODY_POSE binding for pattern {pattern!r} matched zero bodies."
                )
            self._pose_buf = wp.zeros(self._pose_binding.shape, dtype=wp.float32, device=self._device)
            binding_paths: list[str] = list(self._pose_binding.prim_paths)
        else:
            # All prims resolved as world-attached: no binding needed; kernels only hit the -1 branch.
            self._pose_binding = None
            self._pose_buf = wp.zeros((1, 7), dtype=wp.float32, device=self._device)
            binding_paths = []

        # 5. Detect clone_usd=False expansion: binding row count > number of matched USD prims.
        #    Replace per-prim arrays with one entry per binding row, all derived from the env_0 template.
        if binding_paths and len(binding_paths) > len(self._prims):
            template_ancestor = per_prim_ancestor[0]
            template_site_local = per_prim_site_local[0]
            template_parent_ancestor = parent_ancestor[0]
            template_parent_site_local = parent_site_local[0]
            template_path = self._prims[0].GetPath().pathString

            per_prim_ancestor = []
            per_prim_site_local = []
            parent_ancestor = []
            parent_site_local = []
            synthetic_prim_paths: list[str] = []
            for body_path in binding_paths:
                env_match = re.search(r"/World/envs/env_(\d+)", body_path)
                env_token = env_match.group(0) if env_match else None
                # Re-target the template path's env segment to this row's env_id.
                if env_token is not None:
                    synthetic_path = re.sub(r"/World/envs/env_\d+", env_token, template_path)
                    ap = re.sub(r"/World/envs/env_\d+", env_token, template_ancestor) if template_ancestor else None
                    pap = (
                        re.sub(r"/World/envs/env_\d+", env_token, template_parent_ancestor)
                        if template_parent_ancestor
                        else None
                    )
                else:
                    synthetic_path = template_path
                    ap = template_ancestor
                    pap = template_parent_ancestor
                per_prim_ancestor.append(ap)
                per_prim_site_local.append(template_site_local)
                parent_ancestor.append(pap)
                parent_site_local.append(template_parent_site_local)
                synthetic_prim_paths.append(synthetic_path)
            self._synthetic_prim_paths: list[str] | None = synthetic_prim_paths
        else:
            self._synthetic_prim_paths = None

        # 6. Build site_body and parent_site_body indices into the binding's row order.
        path_to_row = {p: i for i, p in enumerate(binding_paths)}
        site_bodies = [
            path_to_row.get(ap, WORLD_BODY_INDEX) if ap is not None else WORLD_BODY_INDEX for ap in per_prim_ancestor
        ]
        parent_bodies = [
            path_to_row.get(pap, WORLD_BODY_INDEX) if pap is not None else WORLD_BODY_INDEX for pap in parent_ancestor
        ]

        # 7. Allocate Warp arrays.
        device = self._device
        self._site_body = wp.array(site_bodies, dtype=wp.int32, device=device)
        self._site_local = wp.array([wp.transform(*x) for x in per_prim_site_local], dtype=wp.transformf, device=device)
        self._parent_site_body = wp.array(parent_bodies, dtype=wp.int32, device=device)
        self._parent_site_local = wp.array(
            [wp.transform(*x) for x in parent_site_local], dtype=wp.transformf, device=device
        )

        count = len(per_prim_ancestor)
        self._pos_buf = wp.zeros(count, dtype=wp.vec3f, device=device)
        self._quat_buf = wp.zeros(count, dtype=wp.vec4f, device=device)
        self._local_pos_buf = wp.zeros(count, dtype=wp.vec3f, device=device)
        self._local_quat_buf = wp.zeros(count, dtype=wp.vec4f, device=device)
        self._pos_ta = ProxyArray(self._pos_buf)
        self._quat_ta = ProxyArray(self._quat_buf)
        self._local_pos_ta = ProxyArray(self._local_pos_buf)
        self._local_quat_ta = ProxyArray(self._local_quat_buf)

    def _resolve_rigid_body_ancestor(
        self,
        prim: Usd.Prim,
        xform_cache: UsdGeom.XformCache,
    ) -> tuple[str | None, list[float]]:
        """Walk USD ancestors to find the nearest prim with ``UsdPhysics.RigidBodyAPI``.

        Under OVPhysX scenes built with ``clone_usd=False`` (the default for
        :class:`~isaaclab.scene.InteractiveScene`), only ``env_0`` carries the
        authored RigidBodyAPI -- ``env_1..N`` exist only as physics-layer clones
        and the corresponding USD prims (when present) are untyped Xforms.
        :meth:`_prim_or_template_has_rigid_body_api` handles this by checking
        the prim's env_0 equivalent when the API is not directly applied.

        Returns:
            ``(ancestor_path, [tx, ty, tz, qx, qy, qz, qw])``. ``ancestor_path`` is
            ``None`` when no rigid-body ancestor exists; the local transform in
            that case is the prim's world USD transform.
        """
        prim_world_tf = xform_cache.GetLocalToWorldTransform(prim)
        prim_world_tf.Orthonormalize()
        # If the prim itself is a rigid body (directly or via env_0 template), the site offset is identity.
        if self._prim_or_template_has_rigid_body_api(prim):
            return prim.GetPath().pathString, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ancestor = prim.GetParent()
        while ancestor and ancestor.IsValid() and ancestor.GetPath().pathString != "/":
            if self._prim_or_template_has_rigid_body_api(ancestor):
                ancestor_world_tf = xform_cache.GetLocalToWorldTransform(ancestor)
                ancestor_world_tf.Orthonormalize()
                local_tf = prim_world_tf * ancestor_world_tf.GetInverse()
                return ancestor.GetPath().pathString, _gf_matrix_to_xform7(local_tf)
            ancestor = ancestor.GetParent()
        return None, _gf_matrix_to_xform7(prim_world_tf)

    def _prim_or_template_has_rigid_body_api(self, prim: Usd.Prim) -> bool:
        """Return whether the prim (or its ``env_0`` equivalent) has ``RigidBodyAPI`` applied.

        Falls back to the env_0 template lookup so that ``clone_usd=False`` envs
        (whose USD prims lack physics schemas) still resolve to the right body.
        """
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return True
        path = prim.GetPath().pathString
        env_zero_path = self._env_zero_equivalent(path)
        if env_zero_path == path:
            return False
        template_prim = self._stage.GetPrimAtPath(env_zero_path) if self._stage is not None else None
        if template_prim is None or not template_prim.IsValid():
            return False
        return template_prim.HasAPI(UsdPhysics.RigidBodyAPI)

    @staticmethod
    def _env_zero_equivalent(path: str) -> str:
        """Replace ``/World/envs/env_<digits>`` with ``/World/envs/env_0`` for template lookup."""
        return re.sub(r"/World/envs/env_\d+", "/World/envs/env_0", path)

    @staticmethod
    def _env_wildcardify(path: str) -> str:
        """Replace ``/World/envs/env_<digits>`` with ``/World/envs/env_*`` for binding patterns."""
        return re.sub(r"/World/envs/env_\d+", "/World/envs/env_*", path)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def prims(self) -> list[Usd.Prim]:
        """List of USD prims discovered for this view.

        Under ``clone_usd=False`` scenes only ``env_0`` carries USD prims, so
        this list may be shorter than :attr:`count`. Use :attr:`prim_paths` to
        get one path per site (env-substituted for non-env_0 sites).
        """
        return self._prims

    @property
    def prim_paths(self) -> list[str]:
        """List of one prim path per site.

        For ``clone_usd=False`` scenes (where ``env_1..N`` have no USD prim)
        the paths are synthesized by replacing ``env_0`` in the template prim's
        path with each binding row's env_id.
        """
        if hasattr(self, "_synthetic_prim_paths") and self._synthetic_prim_paths is not None:
            return self._synthetic_prim_paths
        if not hasattr(self, "_prim_paths_cache"):
            self._prim_paths_cache = [p.GetPath().pathString for p in self._prims]
        return self._prim_paths_cache

    @property
    def count(self) -> int:
        """Number of sites in this view (one per binding row, or per matched prim in world-only mode)."""
        if hasattr(self, "_site_body"):
            return int(self._site_body.shape[0])
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
        """Refresh and return the body-pose array sourced from the OVPhysX tensor binding.

        Reads ``RIGID_BODY_POSE`` data into ``self._pose_buf`` and returns a
        ``wp.transformf`` view. When no rigid-body ancestors were resolved at
        init time (every prim was world-attached), the binding is ``None`` and
        the returned view is a single-element placeholder buffer -- kernels
        access it only via the world-attached (``site_body[i] == -1``) branch.

        Returns:
            ``wp.array(dtype=wp.transformf)`` -- a view over the binding-pose
            buffer ``[num_bodies]``.
        """
        if self._pose_binding is not None:
            self._pose_binding.read(self._pose_buf)
        return self._pose_buf.view(wp.transformf)

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
        """Get prim scales from the USD stage's ``xformOp:scale`` attribute.

        .. note::
            This reads the *static* USD authored value, not a live physics-state
            value. OVPhysX does not maintain a per-shape ``shape_scale`` array
            equivalent to Newton's ``model.shape_scale``, so sim-driven scale
            updates are not reflected here. For sites under ``clone_usd=False``
            envs without authored USD prims, the read returns the env_0
            template's scale via the lazy internal :class:`UsdFrameView`.

        Args:
            indices: Subset of sites to query. ``None`` means all sites.

        Returns:
            ``wp.array`` of shape ``(M, 3)``.
        """
        return self._ensure_usd_view().get_scales(indices)

    def set_scales(self, scales: wp.array, indices: wp.array | None = None) -> None:
        """Set prim scales by writing the USD ``xformOp:scale`` attribute.

        .. note::
            The write lands in the USD stage but does *not* propagate to any
            OVPhysX-side collision-shape scale. PhysX is unaffected; this is a
            stage-only annotation. Use :class:`~isaaclab_ovphysx.assets.RigidObject`
            APIs if you need to change physics-effective shape sizes.

        Args:
            scales: Scales ``(M, 3)`` as ``wp.array``.
            indices: Subset of sites to update. ``None`` means all sites.
        """
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
