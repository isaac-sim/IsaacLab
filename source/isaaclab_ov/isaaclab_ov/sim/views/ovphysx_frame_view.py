# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""OVPhysX-backed FrameView -- Warp-native, GPU-resident pose queries."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import warp as wp

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab import cloner
from isaaclab.physics import PhysicsEvent
from isaaclab.sim.views.base_frame_view import BaseFrameView
from isaaclab.sim.views.usd_frame_view import UsdFrameView
from isaaclab.sim.views.xform_space_writer import FrameViewLocalSpaceWriter, FrameViewWorldSpaceWriter
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.warp import ProxyArray

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
    """Batched frame view with read-only access to OVPhysX body frames.

    Each matched USD prim is resolved at init to a ``(body_index, site_local)``
    pair via ancestor walk: the nearest ancestor carrying ``UsdPhysics.RigidBodyAPI``
    becomes the attachment body, and the relative USD transform becomes the site
    offset. If no rigid-body ancestor exists, the prim is attached to the world
    frame (``body_index = WORLD_BODY_INDEX``) and ``site_local`` stores the prim's
    USD world transform.

    Body world poses are read through an OVPhysX ``RIGID_BODY_POSE`` tensor binding.
    A view of the body itself supports reads; writes remain owned by the asset API.

    World poses are computed on GPU as ``body_q[body_index] * site_local`` via
    a Warp kernel, with the world-attached branch returning ``site_local``
    directly. Both :meth:`set_world_poses` and :meth:`set_local_poses` update
    the view-owned ``site_local`` buffer -- neither writes to the physics state.

    Scales and visibility delegate to an internal :class:`UsdFrameView`
    (lazy-constructed on first call).

    Getters return :class:`~isaaclab.utils.warp.ProxyArray`.  Setters
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
        sim = sim_utils.SimulationContext.instance()
        plan = sim.get_clone_plan() if sim is not None else None
        self._clone_plan = plan
        matches = []
        if plan is not None:
            sources = cloner.path.get_asset_prototype_paths(plan)
            templates, starts, worlds, world_starts = cloner.path.get_world_prototype_asset_templates(
                plan, include_world_indices=True
            )
            # Resolve frame expressions through the closest declared asset, then inspect only its prototype subtree.
            for group in np.flatnonzero(np.diff(world_starts[1:])) + 1:
                for index in range(*starts[group : group + 2]):
                    if (matched := cloner.path.match(prim_path, templates[index])) is not None:
                        matches.append((group, index, matched))
        self._source_sites = []
        self._prims: list[Usd.Prim] = []
        if matches:
            suffix = min((matched.suffix for _, _, matched in matches), key=len)
            for group, index, matched in matches:
                if matched.suffix != suffix:
                    continue
                env_ids = worlds[world_starts[group] : world_starts[group + 1]]
                env_ids = env_ids[
                    resolve_matching_names(matched.instance, env_ids.astype(str), raise_when_no_match=False)[0]
                ]
                if not len(env_ids):
                    continue
                root, template = sources[plan.topology.world_prototypes[index]], templates[index]
                pattern = re.compile(root + suffix)
                prims = sim_utils.get_all_matching_child_prims(
                    root, lambda prim: pattern.fullmatch(prim.GetPath().pathString) is not None, stage=stage
                )
                self._prims.extend(prims)
                self._source_sites.extend((root, template, prim, env_ids) for prim in prims)
        else:
            self._prims = sim_utils.find_matching_prims(prim_path, stage=stage)
        if not self._prims:
            raise ValueError(f"OvPhysxFrameView: pattern {prim_path!r} matched zero prims.")

        # Lazy USD view for scales / visibility.
        self._usd_view: UsdFrameView | None = None

        # Try synchronous init; defer to PHYSICS_READY if the PhysX instance is not yet alive.
        self._backend_manager = sim.backend_manager
        physx = self._backend_manager.get_physx_instance()
        if physx is not None:
            self._initialize_impl(physx)
        else:
            self._backend_manager.register_callback(
                self._on_physics_ready,
                PhysicsEvent.PHYSICS_READY,
                name=f"ovphysx_frame_view_{prim_path}",
            )

    def _on_physics_ready(self, _event) -> None:
        """Replace any prior root view when the OVPhysX ``PhysX`` instance becomes ready."""
        physx = self._backend_manager.get_physx_instance()
        if physx is None:
            raise RuntimeError("OvPhysxFrameView: PHYSICS_READY fired but OvPhysxManager has no PhysX instance.")
        previous_root_view = getattr(self, "_root_view", None)
        if previous_root_view is not None:
            previous_root_view.close()
            self._root_view = None
            self._pose_binding = None
        self._initialize_impl(physx)

    def _initialize_impl(self, physx: Any) -> None:
        """Resolve prims to rigid-body ancestors and create a RIGID_BODY_POSE tensor binding.

        Site discovery reads authored sources and expands their declared destinations.
        Native bindings provide the row mapping for those bodies, independent of USD copies.
        """
        from isaaclab_ov import tensor_types as TT  # noqa: PLC0415
        from isaaclab_ov.sim.views.ovphysx_view import OvPhysxView  # noqa: PLC0415

        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        identity_xform7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

        self._contains_physics_frames = any(prim.HasAPI(UsdPhysics.RigidBodyAPI) for prim in self._prims)
        sources = self._source_sites or [
            (str(prim.GetPath()), str(prim.GetPath()), prim, (-1,)) for prim in self._prims
        ]
        sites = []
        for source_root, destination_template, prim, env_ids in sources:
            body, local = self._resolve_rigid_body_ancestor(prim, xform_cache)
            parent = prim.GetParent()
            parent_body, parent_local = (
                self._resolve_rigid_body_ancestor(parent, xform_cache)
                if parent and not parent.IsPseudoRoot()
                else (None, identity_xform7)
            )
            source_path = Sdf.Path(source_root)
            source_inverse = Gf.Matrix4d(1)
            if self._source_sites:
                source_match = cloner.path.match(source_root, self._clone_plan.env_template)
                if source_match is not None:
                    source_anchor = self._stage.GetPrimAtPath(
                        self._clone_plan.env_template.format(source_match.instance)
                    )
                    source_inverse = xform_cache.GetLocalToWorldTransform(source_anchor).GetInverse()
            for env_id in env_ids:
                destination = Sdf.Path(destination_template.format(env_id))
                site_path = str(destination.AppendPath(prim.GetPath().MakeRelativePath(source_path)))
                body_path = str(destination.AppendPath(Sdf.Path(body).MakeRelativePath(source_path))) if body else None
                parent_path = (
                    str(destination.AppendPath(Sdf.Path(parent_body).MakeRelativePath(source_path)))
                    if parent_body
                    else None
                )
                site_local, parent_site_local = local, parent_local
                if self._source_sites and (body is None or parent_body is None):
                    placement = Gf.Matrix4d(1)
                    if self._clone_plan.positions is not None:
                        placement.SetTranslateOnly(Gf.Vec3d(*map(float, self._clone_plan.positions[env_id])))
                    if body is None:
                        site_local = _gf_matrix_to_xform7(
                            xform_cache.GetLocalToWorldTransform(prim) * source_inverse * placement
                        )
                    if parent_body is None:
                        destination_parent = Sdf.Path(site_path).GetParentPath()
                        parent_match = cloner.path.match(str(destination_parent), self._clone_plan.env_template)
                        if parent_match is None:
                            # A shared parent above the cloned world does not inherit its placement.
                            parent_world = xform_cache.GetLocalToWorldTransform(
                                self._stage.GetPrimAtPath(destination_parent)
                            )
                        elif not parent_match.suffix:
                            parent_world = placement
                        else:
                            parent_world = xform_cache.GetLocalToWorldTransform(parent) * source_inverse * placement
                        parent_site_local = _gf_matrix_to_xform7(parent_world)
                sites.append((int(env_id), prim, site_path, body_path, site_local, parent_path, parent_site_local))
        sites.sort(key=lambda site: site[0])
        (
            _,
            self._prims,
            self._synthetic_prim_paths,
            per_prim_ancestor,
            per_prim_site_local,
            parent_ancestor,
            parent_site_local,
        ) = map(list, zip(*sites, strict=True))

        # Bind the declared destinations; native row order is independent of USD discovery order.
        body_paths = list(dict.fromkeys(path for path in (*per_prim_ancestor, *parent_ancestor) if path is not None))
        if body_paths:
            self._root_view = OvPhysxView(physx, prim_paths=body_paths, device=self._device)
            self._pose_binding = self._root_view.binding_for(TT.RIGID_BODY_POSE)
            self._pose_buf = wp.zeros(self._pose_binding.shape, dtype=wp.float32, device=self._device)
            binding_paths = list(self._pose_binding.prim_paths)
        else:
            self._root_view = self._pose_binding = None
            self._pose_buf = wp.zeros((1, 7), dtype=wp.float32, device=self._device)
            binding_paths = []

        # Map each site's body and parent to native binding rows.
        path_to_row = {p: i for i, p in enumerate(binding_paths)}
        site_bodies = [path_to_row[ap] if ap is not None else WORLD_BODY_INDEX for ap in per_prim_ancestor]
        parent_bodies = [path_to_row[pap] if pap is not None else WORLD_BODY_INDEX for pap in parent_ancestor]

        # Allocate Warp arrays.
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

        Returns:
            ``(ancestor_path, [tx, ty, tz, qx, qy, qz, qw])``. ``ancestor_path`` is
            ``None`` when no rigid-body ancestor exists; the local transform in
            that case is the prim's world USD transform.
        """
        prim_world_tf = xform_cache.GetLocalToWorldTransform(prim)
        prim_world_tf.Orthonormalize()
        # A body frame has no offset from itself.
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim.GetPath().pathString, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ancestor = prim.GetParent()
        while ancestor and ancestor.IsValid() and ancestor.GetPath().pathString != "/":
            if ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                ancestor_world_tf = xform_cache.GetLocalToWorldTransform(ancestor)
                ancestor_world_tf.Orthonormalize()
                local_tf = prim_world_tf * ancestor_world_tf.GetInverse()
                return ancestor.GetPath().pathString, _gf_matrix_to_xform7(local_tf)
            ancestor = ancestor.GetParent()
        return None, _gf_matrix_to_xform7(prim_world_tf)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def prims(self) -> list[Usd.Prim]:
        """List of one authored USD prim per site.

        Source-only clones repeat their source prim handle so the list stays aligned with
        the view count; prim_paths contains their logical destination paths.
        """
        return self._prims

    @property
    def prim_paths(self) -> list[str]:
        """List of one prim path per site.

        Native-only clones use the destination paths declared by the clone plan.
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
            self._backend_manager.update_kinematics()
            self._root_view.read_into("rigid_body_pose", self._pose_buf)
        return self._pose_buf.view(wp.transformf)

    # ------------------------------------------------------------------
    # World / local pose APIs (Tasks 5 & 6)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # World poses
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Writer factory hooks (pass-through; OvPhysX has no separate Fabric storage)
    # ------------------------------------------------------------------

    def _make_world_space_writer(self) -> FrameViewWorldSpaceWriter:
        if self._contains_physics_frames:
            raise ValueError("Use the asset API to write physics body transforms.")
        return _OvPhysxWorldSpaceWriter(self)

    def _make_local_space_writer(self) -> FrameViewLocalSpaceWriter:
        if self._contains_physics_frames:
            raise ValueError("Use the asset API to write physics body transforms.")
        return _OvPhysxLocalSpaceWriter(self)

    # ------------------------------------------------------------------
    # Backend hooks
    # ------------------------------------------------------------------

    def _get_world_poses_impl(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        """Get world-space positions and orientations."""
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

    def _apply_world_pose_write(
        self,
        positions: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ) -> None:
        """Set world-space positions and/or orientations."""
        if positions is None and orientations is None:
            return
        self._require_initialized()
        body_q = self._current_body_q()

        if positions is None or orientations is None:
            cur_pos_ta, cur_quat_ta = self._get_world_poses_impl(indices)
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

    def _get_local_poses_impl(self, indices: wp.array | None = None) -> tuple[ProxyArray, ProxyArray]:
        """Get parent-relative positions and orientations."""
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

    def _apply_local_pose_write(
        self,
        translations: wp.array | None = None,
        orientations: wp.array | None = None,
        indices: wp.array | None = None,
    ) -> None:
        """Set parent-relative translations and/or orientations."""
        if translations is None and orientations is None:
            return
        self._require_initialized()
        body_q = self._current_body_q()

        if translations is None or orientations is None:
            cur_pos_ta, cur_quat_ta = self._get_local_poses_impl(indices)
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

    def _get_local_scales_impl(self, indices: wp.array | None = None) -> ProxyArray:
        """Get local-space scales (xformOp:scale) via the USD view.

        .. note::
            This reads the *static* USD authored value, not a live physics-state
            value. OVPhysX does not maintain a per-shape ``shape_scale`` array
            equivalent to Newton's ``model.shape_scale``, so sim-driven scale
            updates are not reflected here.
        """
        return self._ensure_usd_view()._get_local_scales_impl(indices)

    def _get_world_scales_impl(self, indices: wp.array | None = None) -> ProxyArray:
        """Get world-space (composed) scales via the USD view."""
        return self._ensure_usd_view()._get_world_scales_impl(indices)

    def _apply_local_scale_write(self, scales: wp.array, indices: wp.array | None = None) -> None:
        """Set local-space scales (xformOp:scale) via the USD view.

        .. note::
            The write lands in the USD stage but does *not* propagate to any
            OVPhysX-side collision-shape scale. PhysX is unaffected; this is a
            stage-only annotation.
        """
        self._ensure_usd_view()._apply_local_scale_write(scales, indices)

    def _apply_world_scale_write(self, scales: wp.array, indices: wp.array | None = None) -> None:
        """Set world-space scales via the USD view."""
        self._ensure_usd_view()._apply_world_scale_write(scales, indices)

    def _get_scales_impl(self, indices=None):
        """OvPhysX legacy: deprecated get_scales returns local scales."""
        return self._get_local_scales_impl(indices)

    def _set_scales_impl(self, scales, indices=None):
        """OvPhysX legacy: deprecated set_scales writes local scales via a one-shot writer scope."""
        with self.xform_local_space_writer() as writer:
            writer.set_scales(scales, indices)

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
    mat.Orthonormalize()
    t = mat.ExtractTranslation()
    q = mat.ExtractRotationQuat()
    imag = q.GetImaginary()
    return [float(t[0]), float(t[1]), float(t[2]), float(imag[0]), float(imag[1]), float(imag[2]), float(q.GetReal())]


# ----------------------------------------------------------------------
# Pass-through writer classes
# ----------------------------------------------------------------------


class _OvPhysxWorldSpaceWriter(FrameViewWorldSpaceWriter):
    """OvPhysX world-space writer: pass-through to backend ``_apply_*`` hooks."""

    def set_poses(self, positions=None, orientations=None, indices=None) -> None:
        self._view._apply_world_pose_write(positions, orientations, indices)  # type: ignore[attr-defined]

    def set_scales(self, scales, indices=None) -> None:
        self._view._apply_world_scale_write(scales, indices)  # type: ignore[attr-defined]

    def get_poses(self, indices=None) -> tuple[ProxyArray, ProxyArray]:
        return self._view._get_world_poses_impl(indices)  # type: ignore[attr-defined]

    def get_scales(self, indices=None) -> ProxyArray:
        return self._view._get_world_scales_impl(indices)  # type: ignore[attr-defined]


class _OvPhysxLocalSpaceWriter(FrameViewLocalSpaceWriter):
    """OvPhysX local-space writer: pass-through to backend ``_apply_*`` hooks."""

    def set_poses(self, positions=None, orientations=None, indices=None) -> None:
        self._view._apply_local_pose_write(positions, orientations, indices)  # type: ignore[attr-defined]

    def set_scales(self, scales, indices=None) -> None:
        self._view._apply_local_scale_write(scales, indices)  # type: ignore[attr-defined]

    def get_poses(self, indices=None) -> tuple[ProxyArray, ProxyArray]:
        return self._view._get_local_poses_impl(indices)  # type: ignore[attr-defined]

    def get_scales(self, indices=None) -> ProxyArray:
        return self._view._get_local_scales_impl(indices)  # type: ignore[attr-defined]
